// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines the structures needed for saving/restoring net devices.

use std::io;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::TapError;
use super::device::Net;
use crate::devices::virtio::device::VirtioDevice;
use crate::devices::virtio::persist::{PersistError as VirtioStateError, VirtioDeviceState};
use crate::mmds::data_store::Mmds;
use crate::mmds::ns::MmdsNetworkStack;
use crate::mmds::persist::MmdsNetworkStackState;
use crate::rate_limiter::RateLimiter;
use crate::rate_limiter::persist::RateLimiterState;
use crate::snapshot::Persist;
use crate::utils::net::mac::MacAddr;
use crate::vstate::memory::GuestMemoryMmap;

/// Information about the network config's that are saved
/// at snapshot.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct NetConfigSpaceState {
    guest_mac: Option<MacAddr>,
    #[serde(default)]
    mtu: Option<u16>,
}

/// Information about the network device that are saved
/// at snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetState {
    pub id: String,
    pub tap_if_name: String,
    rx_rate_limiter_state: RateLimiterState,
    tx_rate_limiter_state: RateLimiterState,
    /// The associated MMDS network stack.
    pub mmds_ns: Option<MmdsNetworkStackState>,
    config_space: NetConfigSpaceState,
    pub virtio_state: VirtioDeviceState,
}

/// Auxiliary structure for creating a device when resuming from a snapshot.
#[derive(Debug)]
pub struct NetConstructorArgs {
    /// Pointer to the MMDS data store.
    pub mmds: Option<Arc<Mutex<Mmds>>>,
}

/// Errors triggered when trying to construct a network device at resume time.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum NetPersistError {
    /// Failed to create a network device: {0}
    CreateNet(#[from] super::NetError),
    /// Failed to create a rate limiter: {0}
    CreateRateLimiter(#[from] io::Error),
    /// Failed to re-create the virtio state (i.e queues etc): {0}
    VirtioState(#[from] VirtioStateError),
    /// Indicator that no MMDS is associated with this device.
    NoMmdsDataStore,
    /// Setting tap interface offload flags failed: {0}
    TapSetOffload(TapError),
}

impl<'a> Persist<'a> for Net {
    type State = NetState;
    type ConstructorArgs = NetConstructorArgs;
    type ApplyArgs = &'a GuestMemoryMmap;
    type Error = NetPersistError;

    fn save(&self) -> Self::State {
        NetState {
            id: self.id.clone(),
            tap_if_name: self.iface_name(),
            rx_rate_limiter_state: self.rx_rate_limiter.save(),
            tx_rate_limiter_state: self.tx_rate_limiter.save(),
            mmds_ns: self.mmds_ns.as_ref().map(|mmds| mmds.save()),
            config_space: NetConfigSpaceState {
                guest_mac: self.guest_mac,
                mtu: self.mtu(),
            },
            virtio_state: VirtioDeviceState::from_device(self),
        }
    }

    fn create(
        constructor_args: Self::ConstructorArgs,
        state: &Self::State,
    ) -> Result<Self, Self::Error> {
        let mut net = Net::new(
            state.id.clone(),
            &state.tap_if_name,
            state.config_space.guest_mac,
            RateLimiter::default(),
            RateLimiter::default(),
            state.config_space.mtu,
        )?;

        // The manager supplies a shared datastore for devices with an MMDS stack.
        if let Some(mmds_state) = &state.mmds_ns {
            net.mmds_ns = Some(
                MmdsNetworkStack::create(
                    constructor_args
                        .mmds
                        .ok_or(NetPersistError::NoMmdsDataStore)?,
                    mmds_state,
                )
                .unwrap(),
            );
        }

        Ok(net)
    }

    /// Keeps the TAP, eventfds and rate limiter timers, and the MMDS datastore.
    fn restore_in_place(
        &mut self,
        state: &Self::State,
        mem: &GuestMemoryMmap,
    ) -> Result<(), Self::Error> {
        state.virtio_state.apply_to(self, mem)?;
        self.avail_features = state.virtio_state.avail_features;
        self.rx_rate_limiter
            .restore_in_place(&state.rx_rate_limiter_state, ())?;
        self.tx_rate_limiter
            .restore_in_place(&state.tx_rate_limiter_state, ())?;
        if let (Some(mmds_ns), Some(mmds_ns_state)) = (&mut self.mmds_ns, &state.mmds_ns) {
            mmds_ns.restore_in_place(mmds_ns_state, ()).unwrap();
        }
        // Drop the descriptor chains parsed from the old queues, as a device reset does.
        self._reset();
        // The guest may have negotiated other features since load. Activation configures
        // the TAP offloads of a device that load creates.
        if self.is_activated() {
            self.apply_acked_features()
                .map_err(NetPersistError::TapSetOffload)?;
        }
        Ok(())
    }

    fn check_reset(&self, _state: &Self::State) -> Result<(), crate::snapshot::ResetUnsupported> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;
    use std::os::fd::AsRawFd;

    use super::*;
    use crate::devices::virtio::device::VirtioDeviceType;
    use crate::devices::virtio::net::test_utils::test::TestHelper;
    use crate::devices::virtio::net::test_utils::{default_net, default_net_no_mmds};
    use crate::devices::virtio::test_utils::default_mem;

    fn validate_save_and_restore(net: Net, mmds_ds: Option<Arc<Mutex<Mmds>>>) {
        let guest_mem = default_mem();

        let id;
        let tap_if_name;
        let has_mmds_ns;
        let allow_mmds_requests;
        let virtio_state;
        let serialized_data;

        // Create and save the net device.
        {
            let net_state = net.save();
            serialized_data = bitcode::serialize(&net_state).unwrap();

            // Save some fields that we want to check later.
            id = net.id.clone();
            tap_if_name = net.iface_name();
            has_mmds_ns = net.mmds_ns.is_some();
            allow_mmds_requests = has_mmds_ns && mmds_ds.is_some();
            virtio_state = VirtioDeviceState::from_device(&net);
        }

        // Drop the initial net device so that we don't get an error when trying to recreate the
        // TAP device.
        drop(net);
        {
            // Deserialize and restore the net device.
            let restored_state = bitcode::deserialize(&serialized_data).unwrap();
            match crate::snapshot::restore_for_test::<Net>(
                NetConstructorArgs { mmds: mmds_ds },
                &restored_state,
                &guest_mem,
            ) {
                Ok(restored_net) => {
                    // Test that virtio specific fields are the same.
                    assert_eq!(restored_net.device_type(), VirtioDeviceType::Net);
                    assert_eq!(restored_net.avail_features(), virtio_state.avail_features);
                    assert_eq!(restored_net.acked_features(), virtio_state.acked_features);
                    assert_eq!(restored_net.is_activated(), virtio_state.activated);

                    // Test that net specific fields are the same.
                    assert_eq!(&restored_net.id, &id);
                    assert_eq!(&restored_net.iface_name(), &tap_if_name);
                    assert_eq!(restored_net.mmds_ns.is_some(), allow_mmds_requests);
                    assert_eq!(restored_net.rx_rate_limiter, RateLimiter::default());
                    assert_eq!(restored_net.tx_rate_limiter, RateLimiter::default());
                }
                Err(NetPersistError::NoMmdsDataStore) => {
                    assert!(has_mmds_ns && !allow_mmds_requests)
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_persistence() {
        let mmds = Some(Arc::new(Mutex::new(Mmds::default())));
        validate_save_and_restore(default_net(), mmds.as_ref().cloned());
        validate_save_and_restore(default_net_no_mmds(), None);

        // Check what happens if MMIOVirtioDevices::restore gives us the reference to the MMDS
        // data store even if this device does not have mmds ns configured.
        // The restore should be conservative and not configure the mmds ns.
        validate_save_and_restore(default_net_no_mmds(), mmds);

        // Check what happens if MMIOVirtioDevices::restore does not give us the reference to the
        // MMDS data store. This will return an error.
        validate_save_and_restore(default_net(), None);
    }

    #[test]
    fn test_restore_in_place() {
        let mem = default_mem();
        let mut th = TestHelper::get_default(&mem);
        th.activate_net();
        let mut net = th.net();
        let fds = |net: &Net| {
            [
                net.tap.as_raw_fd(),
                net.activate_evt.as_raw_fd(),
                net.queue_evts[0].as_raw_fd(),
                net.queue_evts[1].as_raw_fd(),
                net.rx_rate_limiter.as_raw_fd(),
                net.tx_rate_limiter.as_raw_fd(),
            ]
        };
        let original_fds = fds(&net);
        let mmds = Arc::clone(&net.mmds_ns.as_ref().unwrap().mmds);
        let state = net.save();
        net.avail_features = 0;
        net.configure_mmds_network_stack(Ipv4Addr::LOCALHOST, Arc::clone(&mmds));

        net.restore_in_place(&state, &mem).unwrap();

        assert_eq!(VirtioDeviceState::from_device(&*net), state.virtio_state);
        assert_eq!(fds(&net), original_fds);
        let ns = net.mmds_ns.as_ref().unwrap();
        assert_eq!(ns.ipv4_addr(), MmdsNetworkStack::default_ipv4_addr());
        assert!(Arc::ptr_eq(&ns.mmds, &mmds));
    }
}
