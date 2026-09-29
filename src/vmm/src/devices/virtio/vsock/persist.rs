// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines state and support structures for persisting Vsock devices and backends.

use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use super::*;
use crate::devices::virtio::persist::VirtioDeviceState;
use crate::snapshot::Persist;
use crate::vstate::memory::GuestMemoryMmap;

/// The Vsock serializable state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VsockState {
    /// The vsock backend state.
    pub backend: VsockBackendState,
    /// The vsock frontend state.
    pub frontend: VsockFrontendState,
}

/// The Vsock frontend serializable state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VsockFrontendState {
    /// Context Identifier.
    pub cid: u64,
    pub virtio_state: VirtioDeviceState,
    /// Whether a `TRANSPORT_RESET_EVENT` published to the guest's event queue
    /// is still awaiting the driver's acknowledgment. RX delivery stays gated
    /// until the guest acks, so a restored device must resume with the same
    /// gate state the source device had.
    pub pending_event_ack: bool,
}

/// The Vsock Unix Backend serializable state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VsockBackendState {
    /// The path for the UDS socket.
    pub uds_path: String,
    /// The last used host-side port.
    pub local_port_last: u32,
}

/// A helper structure that holds the constructor arguments for a vsock device
#[derive(Debug)]
pub struct VsockConstructorArgs<B> {
    /// Pointer to guest memory.
    pub mem: GuestMemoryMmap,
    /// Backend with its host resources already created.
    pub backend: B,
}

/// A helper structure that holds the constructor arguments for VsockUnixBackend
#[derive(Debug)]
pub struct VsockUdsConstructorArgs {
    /// cid available in VsockFrontendState.
    pub cid: u64,
}

impl Persist<'_> for VsockUnixBackend {
    type State = VsockBackendState;
    type ConstructorArgs = VsockUdsConstructorArgs;
    type Error = VsockUnixBackendError;

    fn save(&self) -> Self::State {
        VsockBackendState {
            uds_path: self.host_sock_path.clone(),
            local_port_last: self.local_port_last,
        }
    }

    fn restore(
        constructor_args: Self::ConstructorArgs,
        state: &Self::State,
    ) -> Result<Self, Self::Error> {
        let mut backend = Self::new(constructor_args.cid, state.uds_path.clone())?;
        backend.local_port_last = state.local_port_last;
        Ok(backend)
    }
}

impl<B> Persist<'_> for Vsock<B>
where
    B: VsockBackend + 'static + Debug,
{
    type State = VsockFrontendState;
    type ConstructorArgs = VsockConstructorArgs<B>;
    type Error = VsockError;

    fn save(&self) -> Self::State {
        VsockFrontendState {
            cid: self.cid(),
            virtio_state: VirtioDeviceState::from_device(self),
            pending_event_ack: self.pending_event_ack,
        }
    }

    fn restore(
        constructor_args: Self::ConstructorArgs,
        state: &Self::State,
    ) -> Result<Self, Self::Error> {
        let VsockConstructorArgs { mem, backend } = constructor_args;
        let mut vsock = Self::create(backend, state)?;
        vsock.restore_in_place(state, &mem)?;
        Ok(vsock)
    }
}

impl<B> Vsock<B>
where
    B: VsockBackend + 'static + Debug,
{
    pub fn create(backend: B, state: &VsockFrontendState) -> Result<Self, VsockError> {
        Self::new(state.cid, backend)
    }

    /// Keeps the host socket, eventfds and event loop registrations.
    pub fn restore_in_place(
        &mut self,
        state: &VsockFrontendState,
        mem: &GuestMemoryMmap,
    ) -> Result<(), VsockError> {
        state
            .virtio_state
            .apply_to(self, mem)
            .map_err(VsockError::VirtioState)?;
        self.avail_features = state.virtio_state.avail_features;
        // Drop the packets parsed from the old queues.
        self.rx_packet.clear();
        self.tx_packet.clear();
        self.pending_event_ack = state.pending_event_ack;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::os::fd::AsRawFd;

    use super::device::AVAIL_FEATURES;
    use super::*;
    use crate::devices::virtio::device::{VirtioDevice, VirtioDeviceType};
    use crate::devices::virtio::vsock::test_utils::{TestBackend, TestContext};
    use crate::utils::byte_order;

    impl Persist<'_> for TestBackend {
        type State = VsockBackendState;
        type ConstructorArgs = VsockUdsConstructorArgs;
        type Error = VsockUnixBackendError;

        fn save(&self) -> Self::State {
            VsockBackendState {
                uds_path: "test".to_owned(),
                local_port_last: 0xdeadbeef,
            }
        }

        fn restore(_: Self::ConstructorArgs, _state: &Self::State) -> Result<Self, Self::Error> {
            Ok(TestBackend::new())
        }
    }

    #[test]
    fn test_persist_pending_event_ack() {
        // The RX gate must survive a save/restore cycle: a restored device must keep
        // gating RX while a TRANSPORT_RESET ack is outstanding, and must not gate RX
        // when none is.
        let mut ctx = TestContext::new();
        for armed in [false, true] {
            ctx.device.pending_event_ack = armed;
            let state = ctx.device.save();
            assert_eq!(state.pending_event_ack, armed);

            let restored = Vsock::restore(
                VsockConstructorArgs {
                    mem: ctx.mem.clone(),
                    backend: TestBackend::new(),
                },
                &state,
            )
            .unwrap();
            assert_eq!(restored.pending_event_ack, armed);

            ctx.device.pending_event_ack = !armed;
            ctx.device.restore_in_place(&state, &ctx.mem).unwrap();
            assert_eq!(ctx.device.pending_event_ack, armed);
        }
    }

    #[test]
    fn test_persist_uds_backend() {
        let ctx = TestContext::new();
        let device_features = AVAIL_FEATURES;
        let driver_features: u64 = AVAIL_FEATURES | 1 | (1 << 32);
        let device_pages = [
            (device_features & 0xffff_ffff) as u32,
            (device_features >> 32) as u32,
        ];
        let driver_pages = [
            (driver_features & 0xffff_ffff) as u32,
            (driver_features >> 32) as u32,
        ];

        // Test serialization
        // Save backend and device state separately.
        let state = VsockState {
            backend: ctx.device.backend().save(),
            frontend: ctx.device.save(),
        };

        let serialized_data = bitcode::serialize(&state).unwrap();

        let restored_state: VsockState = bitcode::deserialize(&serialized_data).unwrap();
        let mut restored_device = Vsock::restore(
            VsockConstructorArgs {
                mem: ctx.mem.clone(),
                backend: {
                    assert_eq!(restored_state.backend.uds_path, "test".to_owned());
                    assert_eq!(restored_state.backend.local_port_last, 0xdeadbeef);
                    TestBackend::new()
                },
            },
            &restored_state.frontend,
        )
        .unwrap();

        assert_eq!(restored_device.device_type(), VirtioDeviceType::Vsock);
        assert_eq!(restored_device.avail_features_by_page(0), device_pages[0]);
        assert_eq!(restored_device.avail_features_by_page(1), device_pages[1]);
        assert_eq!(restored_device.avail_features_by_page(2), 0);

        restored_device.ack_features_by_page(0, driver_pages[0]);
        restored_device.ack_features_by_page(1, driver_pages[1]);
        restored_device.ack_features_by_page(2, 0);
        restored_device.ack_features_by_page(0, !driver_pages[0]);
        assert_eq!(
            restored_device.acked_features(),
            device_features & driver_features
        );

        // Validate config_as_bytes returns the CID in little-endian.
        let config = restored_device.config_as_bytes();
        assert_eq!(config.len(), 8);
        assert_eq!(byte_order::read_le_u64(config), ctx.cid);
    }

    #[test]
    fn test_restore_in_place() {
        let ctx = TestContext::new();
        let mut handler_ctx = ctx.create_event_handler_context();
        handler_ctx.mock_activate(ctx.mem.clone(), ctx.interrupt.clone());
        let device = &mut handler_ctx.device;
        let fds = |device: &Vsock<TestBackend>| {
            [
                device.backend.as_raw_fd(),
                device.activate_evt.as_raw_fd(),
                device.queue_events[0].as_raw_fd(),
                device.queue_events[1].as_raw_fd(),
                device.queue_events[2].as_raw_fd(),
            ]
        };
        let original_fds = fds(device);
        device.acked_features = AVAIL_FEATURES;
        let state = device.save();
        device.acked_features = 0;

        device.restore_in_place(&state, &ctx.mem).unwrap();

        assert_eq!(VirtioDeviceState::from_device(device), state.virtio_state);
        assert_eq!(fds(device), original_fds);
    }
}
