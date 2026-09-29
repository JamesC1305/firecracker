// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines the structures needed for saving/restoring Virtio primitives.

use std::convert::Infallible;
use std::num::Wrapping;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::queue::{InvalidAvailIdx, QueueError};
use super::transport::mmio::IrqTrigger;
use crate::devices::virtio::device::{VirtioDevice, VirtioDeviceType};
use crate::devices::virtio::generated::virtio_ring::VIRTIO_RING_F_EVENT_IDX;
use crate::devices::virtio::queue::Queue;
use crate::devices::virtio::transport::mmio::MmioTransport;
use crate::snapshot::Persist;
use crate::vstate::memory::{GuestAddress, GuestMemoryMmap, GuestMemorySlice};

/// Errors thrown during restoring virtio state.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum PersistError {
    /// Snapshot state contains invalid queue info.
    InvalidInput,
    /// Could not restore queue: {0}
    QueueConstruction(QueueError),
    /// {0}
    InvalidAvailIdx(#[from] InvalidAvailIdx),
}

/// Queue information saved in snapshot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueState {
    /// The maximal size in elements offered by the device
    max_size: u16,

    /// The queue size in elements the driver selected
    size: u16,

    /// Indicates if the queue is finished with configuration
    ready: bool,

    /// Guest physical address of the descriptor table
    desc_table: u64,

    /// Guest physical address of the available ring
    avail_ring: u64,

    /// Guest physical address of the used ring
    used_ring: u64,

    next_avail: Wrapping<u16>,
    next_used: Wrapping<u16>,

    /// The number of added used buffers since last guest kick
    num_added: Wrapping<u16>,
}

impl<'a> Persist<'a> for Queue {
    type State = QueueState;
    type ConstructorArgs = ();
    type Error = Infallible;

    fn save(&self) -> Self::State {
        QueueState {
            max_size: self.max_size,
            size: self.size,
            ready: self.ready,
            desc_table: self.desc_table_address.0,
            avail_ring: self.avail_ring_address.0,
            used_ring: self.used_ring_address.0,
            next_avail: self.next_avail,
            next_used: self.next_used,
            num_added: self.num_added,
        }
    }

    fn restore(_: Self::ConstructorArgs, state: &Self::State) -> Result<Self, Self::Error> {
        let mut queue = Queue::new(state.max_size);
        queue.restore_in_place(state, ())?;
        Ok(queue)
    }
}

impl Queue {
    /// Applies queue state with unresolved rings; the device initializes activated queues.
    pub fn restore_in_place(&mut self, state: &QueueState, _: ()) -> Result<(), Infallible> {
        *self = Queue {
            max_size: state.max_size,
            size: state.size,
            ready: state.ready,
            desc_table_address: GuestAddress(state.desc_table),
            avail_ring_address: GuestAddress(state.avail_ring),
            used_ring_address: GuestAddress(state.used_ring),

            desc_table: GuestMemorySlice::UNRESOLVED,
            avail_ring: GuestMemorySlice::UNRESOLVED,
            used_ring: GuestMemorySlice::UNRESOLVED,

            next_avail: state.next_avail,
            next_used: state.next_used,
            uses_notif_suppression: false,
            num_added: state.num_added,
        };
        Ok(())
    }
}

/// State of a VirtioDevice.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VirtioDeviceState {
    /// Device type.
    pub device_type: VirtioDeviceType,
    /// Available virtio features.
    pub avail_features: u64,
    /// Negotiated virtio features.
    pub acked_features: u64,
    /// List of queues.
    pub queues: Vec<QueueState>,
    /// Flag for activated status.
    pub activated: bool,
}

impl VirtioDeviceState {
    /// Construct the virtio state of a device.
    pub fn from_device(device: &dyn VirtioDevice) -> Self {
        VirtioDeviceState {
            device_type: device.device_type(),
            avail_features: device.avail_features(),
            acked_features: device.acked_features(),
            queues: device.queues().iter().map(Persist::save).collect(),
            activated: device.is_activated(),
        }
    }

    /// Checks and applies saved features and queues without changing the device's activation.
    /// All queues must have the device's maximum size. Rings are resolved and marked dirty
    /// only when the saved state is activated.
    pub fn apply_to(
        &self,
        device: &mut dyn VirtioDevice,
        mem: &GuestMemoryMmap,
    ) -> Result<(), PersistError> {
        let queues = device.queues();
        let max_size = queues.first().map_or(0, |queue| queue.max_size);
        if self.device_type != device.device_type()
            || (self.acked_features & !self.avail_features) != 0
            || self.queues.len() != queues.len()
            || self.queues.iter().any(|queue| queue.max_size != max_size)
        {
            return Err(PersistError::InvalidInput);
        }

        let uses_notif_suppression = (self.acked_features & (1u64 << VIRTIO_RING_F_EVENT_IDX)) != 0;
        for (queue, state) in device.queues_mut().iter_mut().zip(&self.queues) {
            queue.restore_in_place(state, ()).unwrap();
            if self.activated {
                queue
                    .initialize(mem)
                    .map_err(PersistError::QueueConstruction)?;
            }
            if uses_notif_suppression {
                queue.enable_notif_suppression();
            }
        }
        device.set_acked_features(self.acked_features);
        Ok(())
    }
}

/// Transport information saved in snapshot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MmioTransportState {
    // The register where feature bits are stored.
    features_select: u32,
    // The register where features page is selected.
    acked_features_select: u32,
    queue_select: u32,
    device_status: u32,
    config_generation: u32,
    interrupt_status: u32,
}

/// Auxiliary structure for initializing the transport when resuming from a snapshot.
#[derive(Debug)]
pub struct MmioTransportConstructorArgs {
    /// Pointer to guest memory.
    pub mem: GuestMemoryMmap,
    /// Interrupt to use for the device
    pub interrupt: Arc<IrqTrigger>,
    /// Device associated with the current MMIO state.
    pub device: Arc<Mutex<dyn VirtioDevice>>,
    /// Is device backed by vhost-user.
    pub is_vhost_user: bool,
}

impl<'a> Persist<'a> for MmioTransport {
    type State = MmioTransportState;
    type ConstructorArgs = MmioTransportConstructorArgs;
    type Error = Infallible;

    fn save(&self) -> Self::State {
        MmioTransportState {
            features_select: self.features_select,
            acked_features_select: self.acked_features_select,
            queue_select: self.queue_select,
            device_status: self.device_status,
            config_generation: self.config_generation,
            interrupt_status: self.interrupt.irq_status.load(Ordering::SeqCst),
        }
    }

    fn restore(
        constructor_args: Self::ConstructorArgs,
        state: &Self::State,
    ) -> Result<Self, Self::Error> {
        let mut transport = MmioTransport::new(
            constructor_args.mem,
            constructor_args.interrupt,
            constructor_args.device,
            constructor_args.is_vhost_user,
        );
        transport.restore_in_place(state, ())?;
        Ok(transport)
    }
}

impl MmioTransport {
    /// Applies the register values of `state`, keeping the device and its interrupt.
    pub fn restore_in_place(
        &mut self,
        state: &MmioTransportState,
        _: (),
    ) -> Result<(), Infallible> {
        self.features_select = state.features_select;
        self.acked_features_select = state.acked_features_select;
        self.queue_select = state.queue_select;
        self.device_status = state.device_status;
        self.config_generation = state.config_generation;
        self.interrupt
            .irq_status
            .store(state.interrupt_status, Ordering::SeqCst);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use vm_memory::GuestMemoryBackend;

    use super::*;
    use crate::devices::virtio::test_utils::default_mem;
    use crate::devices::virtio::transport::mmio::IrqTrigger;
    use crate::devices::virtio::transport::mmio::tests::DummyDevice;
    use crate::vmm_config::machine_config::HugePageConfig;
    use crate::vstate::bus::BusDevice;
    use crate::vstate::memory::test_utils::into_region_ext;
    use crate::vstate::memory::{Bitmap, GuestMemoryExtension, anonymous};

    #[test]
    fn test_virtio_state_apply() {
        let mem = into_region_ext(
            anonymous(
                std::iter::once((GuestAddress(0), 0x10_000)),
                true,
                HugePageConfig::None,
            )
            .unwrap(),
        );
        let mut device = DummyDevice::new();
        // Real devices give all their queues the same maximum size.
        device.queues_mut()[1] = Queue::new(16);
        let event_idx = 1u64 << VIRTIO_RING_F_EVENT_IDX;
        device.set_avail_features(event_idx);
        for (queue, ring) in device.queues_mut().iter_mut().zip([0x1000, 0x4000]) {
            queue.size = queue.max_size;
            queue.ready = true;
            queue.desc_table_address = GuestAddress(ring);
            queue.avail_ring_address = GuestAddress(ring + 0x1000);
            queue.used_ring_address = GuestAddress(ring + 0x2000);
        }
        let mut state = VirtioDeviceState::from_device(&device);
        state.activated = true;
        state.acked_features = event_idx;
        state.queues[0].next_avail = Wrapping(7);
        state.queues[0].next_used = Wrapping(5);
        let serialized_data = bitcode::serialize(&state).unwrap();
        let mut state: VirtioDeviceState = bitcode::deserialize(&serialized_data).unwrap();

        mem.reset_dirty();
        device.queue_events()[0].write(1).unwrap();
        state.apply_to(&mut device, &mem).unwrap();

        assert!(!device.is_activated());
        assert_eq!(device.queue_events()[0].read().unwrap(), 1);
        device
            .activate(mem.clone(), Arc::new(IrqTrigger::new()))
            .unwrap();

        assert_eq!(VirtioDeviceState::from_device(&device), state);
        assert!(device.queues()[0].uses_notif_suppression);
        // The rings are dirty again, so the next reset reverts them.
        for ring in [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000] {
            let slice = mem.get_slice(GuestAddress(ring), 1).unwrap();
            assert!(slice.bitmap().dirty_at(0));
        }

        // Saved activation, not live activation, controls whether rings are resolved.
        state.activated = false;
        state.acked_features = 0;
        state.queues[0].size = 0;
        state.queues[0].avail_ring = u64::MAX;
        mem.reset_dirty();
        state.apply_to(&mut device, &mem).unwrap();
        assert!(device.is_activated());
        assert_eq!(device.acked_features(), 0);
        for (queue, saved) in device.queues().iter().zip(&state.queues) {
            assert_eq!(queue.save(), *saved);
            assert_eq!(queue.desc_table, GuestMemorySlice::UNRESOLVED);
            assert_eq!(queue.avail_ring, GuestMemorySlice::UNRESOLVED);
            assert_eq!(queue.used_ring, GuestMemorySlice::UNRESOLVED);
            assert!(!queue.uses_notif_suppression);
        }
        for ring in [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000] {
            let slice = mem.get_slice(GuestAddress(ring), 1).unwrap();
            assert!(!slice.bitmap().dirty_at(0));
        }
    }

    #[test]
    fn test_mmio_transport_apply() {
        let device: Arc<Mutex<dyn VirtioDevice>> = Arc::new(Mutex::new(DummyDevice::new()));
        let interrupt = Arc::new(IrqTrigger::new());
        let mut transport =
            MmioTransport::new(default_mem(), interrupt.clone(), device.clone(), false);
        let mut state = transport.save();
        state.features_select = 1;
        state.acked_features_select = 1;
        state.queue_select = 1;
        state.device_status = 0xf;
        state.config_generation = 17;
        state.interrupt_status = 3;
        let serialized_data = bitcode::serialize(&state).unwrap();
        let state = bitcode::deserialize(&serialized_data).unwrap();

        transport.restore_in_place(&state, ()).unwrap();

        assert_eq!(transport.save(), state);
        assert!(Arc::ptr_eq(&transport.device(), &device));
        assert!(Arc::ptr_eq(&transport.interrupt, &interrupt));
        let mut value = [0; 4];
        transport.read(0, 0x70, &mut value);
        assert_eq!(u32::from_le_bytes(value), 0xf);
    }

    #[test]
    fn test_virtiodev_sanity_checks() {
        let mem = default_mem();
        let mut device = DummyDevice::new();
        let max_size = device.queues()[0].max_size;
        device.queues_mut()[1] = Queue::new(max_size);
        let state = VirtioDeviceState::from_device(&device);

        let mut invalid = state.clone();
        invalid.device_type = VirtioDeviceType::Net;
        assert!(matches!(
            invalid.apply_to(&mut device, &mem),
            Err(PersistError::InvalidInput)
        ));

        let mut invalid = state.clone();
        invalid.queues.pop();
        assert!(matches!(
            invalid.apply_to(&mut device, &mem),
            Err(PersistError::InvalidInput)
        ));

        let mut invalid = state.clone();
        invalid.acked_features = 1;
        assert!(matches!(
            invalid.apply_to(&mut device, &mem),
            Err(PersistError::InvalidInput)
        ));

        let mut invalid = state.clone();
        invalid.queues[0].max_size = max_size + 1;
        assert!(matches!(
            invalid.apply_to(&mut device, &mem),
            Err(PersistError::InvalidInput)
        ));

        let mut invalid = state.clone();
        invalid.activated = true;
        invalid.queues[0].ready = true;
        invalid.queues[0].size = max_size + 1;
        assert!(matches!(
            invalid.apply_to(&mut device, &mem),
            Err(PersistError::QueueConstruction(QueueError::InvalidSize(_)))
        ));

        let mut invalid = state;
        invalid.activated = true;
        assert!(matches!(
            invalid.apply_to(&mut device, &mem),
            Err(PersistError::QueueConstruction(QueueError::NotReady))
        ));
    }

    #[test]
    fn test_queue_persistence() {
        let mem = default_mem();

        let mut queue = Queue::new(128);
        queue.ready = true;
        queue.size = 64;
        queue.desc_table_address = GuestAddress(0x1000);
        queue.avail_ring_address = GuestAddress(0x2000);
        queue.used_ring_address = GuestAddress(0x3000);
        queue.next_avail = Wrapping(7);
        queue.next_used = Wrapping(5);
        queue.num_added = Wrapping(2);
        queue.initialize(&mem).unwrap();

        let queue_state = queue.save();
        let serialized_data = bitcode::serialize(&queue_state).unwrap();

        let restored_state = bitcode::deserialize(&serialized_data).unwrap();
        let mut restored_queue = Queue::new(16);
        restored_queue.ready = true;
        restored_queue.initialize(&mem).unwrap();
        restored_queue.enable_notif_suppression();
        restored_queue
            .restore_in_place(&restored_state, ())
            .unwrap();

        assert_eq!(restored_queue.desc_table, GuestMemorySlice::UNRESOLVED);
        assert_eq!(restored_queue.avail_ring, GuestMemorySlice::UNRESOLVED);
        assert_eq!(restored_queue.used_ring, GuestMemorySlice::UNRESOLVED);
        assert!(!restored_queue.uses_notif_suppression);
        restored_queue.initialize(&mem).unwrap();

        assert_eq!(restored_queue, queue);
    }
}
