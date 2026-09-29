// Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines the structures needed for saving/restoring virtio-mem devices.

use std::sync::Arc;

use bitvec::vec::BitVec;
use serde::{Deserialize, Serialize};

use crate::devices::virtio::generated::virtio_mem::virtio_mem_config;
use crate::devices::virtio::mem::{MEM_NUM_QUEUES, VirtioMem, VirtioMemError};
use crate::devices::virtio::persist::{PersistError as VirtioStateError, VirtioDeviceState};
use crate::devices::virtio::queue::{FIRECRACKER_MAX_QUEUE_SIZE, Queue};
use crate::snapshot::Persist;
use crate::utils::usize_to_u64;
use crate::vstate::memory::GuestMemoryMmap;
use crate::vstate::vm::KvmVm;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtioMemState {
    pub virtio_state: VirtioDeviceState,
    addr: u64,
    region_size: u64,
    block_size: u64,
    usable_region_size: u64,
    requested_size: u64,
    slot_size: usize,
    plugged_blocks: Vec<bool>,
}

#[derive(Debug)]
pub struct VirtioMemConstructorArgs {
    vm: Arc<KvmVm>,
}

impl VirtioMemConstructorArgs {
    pub fn new(vm: Arc<KvmVm>) -> Self {
        Self { vm }
    }
}

#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum VirtioMemPersistError {
    /// Create virtio-mem: {0}
    CreateVirtioMem(#[from] VirtioMemError),
    /// Virtio state: {0}
    VirtioState(#[from] VirtioStateError),
}

impl<'a> Persist<'a> for VirtioMem {
    type State = VirtioMemState;
    type ConstructorArgs = VirtioMemConstructorArgs;
    type ApplyArgs = &'a GuestMemoryMmap;
    type Error = VirtioMemPersistError;

    fn save(&self) -> Self::State {
        VirtioMemState {
            virtio_state: VirtioDeviceState::from_device(self),
            addr: self.config.addr,
            region_size: self.config.region_size,
            block_size: self.config.block_size,
            usable_region_size: self.config.usable_region_size,
            plugged_blocks: self.plugged_blocks.iter().by_vals().collect(),
            requested_size: self.config.requested_size,
            slot_size: self.slot_size,
        }
    }

    fn create(
        constructor_args: Self::ConstructorArgs,
        state: &Self::State,
    ) -> Result<Self, Self::Error> {
        // Fixed geometry is also needed by the manager's load-time resource bookkeeping.
        Ok(VirtioMem::from_state(
            constructor_args.vm,
            vec![Queue::new(FIRECRACKER_MAX_QUEUE_SIZE); MEM_NUM_QUEUES],
            virtio_mem_config {
                addr: state.addr,
                region_size: state.region_size,
                block_size: state.block_size,
                ..Default::default()
            },
            state.slot_size,
            BitVec::new(),
        )?)
    }

    /// Keeps the VM and eventfds. The guest-memory owner restores the KVM slot map.
    fn restore_in_place(
        &mut self,
        state: &Self::State,
        mem: &GuestMemoryMmap,
    ) -> Result<(), Self::Error> {
        state.virtio_state.apply_to(self, mem)?;
        self.plugged_blocks.clear();
        self.plugged_blocks.extend(state.plugged_blocks.iter());
        self.config = virtio_mem_config {
            addr: state.addr,
            region_size: state.region_size,
            block_size: state.block_size,
            usable_region_size: state.usable_region_size,
            plugged_size: usize_to_u64(self.plugged_blocks.count_ones()) * state.block_size,
            requested_size: state.requested_size,
            ..Default::default()
        };
        self.slot_size = state.slot_size;
        self.set_avail_features(state.virtio_state.avail_features);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsRawFd;

    use super::*;
    use crate::devices::virtio::device::VirtioDevice;
    use crate::devices::virtio::mem::device::test_utils::default_virtio_mem;
    use crate::utils::u32_mib_to_bytes;
    use crate::vstate::vm::tests::setup_vm_with_memory;

    #[test]
    fn test_save_restore_state() {
        let mut original = default_virtio_mem();
        original.set_acked_features(original.avail_features());
        original.config.requested_size = u32_mib_to_bytes(128);
        original.plugged_blocks.set(1, true);
        original.config.plugged_size = u32_mib_to_bytes(2);
        let state = bitcode::deserialize(&bitcode::serialize(&original.save()).unwrap()).unwrap();
        let vm = Arc::new(setup_vm_with_memory(0x1000));
        let args = VirtioMemConstructorArgs::new(Arc::clone(&vm));
        let mut restored =
            crate::snapshot::restore_for_test::<VirtioMem>(args, &state, vm.guest_memory())
                .unwrap();
        assert_eq!(restored.config, original.config);
        assert_eq!(restored.slot_size, original.slot_size);
        assert_eq!(restored.plugged_blocks, original.plugged_blocks);

        let activate_fd = restored.activate_event().as_raw_fd();
        let queue_fd = restored.queue_events()[0].as_raw_fd();
        restored.config = virtio_mem_config::default();
        restored.plugged_blocks.fill(false);
        restored
            .restore_in_place(&state, vm.guest_memory())
            .unwrap();
        assert_eq!(restored.config, original.config);
        assert_eq!(restored.plugged_blocks, original.plugged_blocks);
        assert_eq!(restored.activate_event().as_raw_fd(), activate_fd);
        assert_eq!(restored.queue_events()[0].as_raw_fd(), queue_fd);
    }
}
