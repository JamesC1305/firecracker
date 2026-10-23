// Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines the structures needed for saving/restoring virtio-mem devices.

use std::sync::Arc;

use bitvec::vec::BitVec;
use serde::{Deserialize, Serialize};

use crate::devices::virtio::generated::virtio_mem::virtio_mem_config;
use crate::devices::virtio::mem::{VirtioMem, VirtioMemError};
use crate::devices::virtio::persist::{PersistError as VirtioStateError, VirtioDeviceState};
use crate::snapshot::{Persist, Provision};
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
pub struct VirtioMemProvisionArgs {
    vm: Arc<KvmVm>,
}

impl VirtioMemProvisionArgs {
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

    /// Keeps the VM and eventfds. The guest-memory owner restores the KVM slot map.
    fn apply_state(
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

impl Provision<'_> for VirtioMem {
    type ProvisionArgs = VirtioMemProvisionArgs;

    fn provision(args: Self::ProvisionArgs, state: &Self::State) -> Result<Self, Self::Error> {
        // Fixed geometry is also needed by the manager's load-time resource bookkeeping.
        Ok(VirtioMem::from_state(
            args.vm,
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
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsRawFd;

    use super::*;
    use crate::devices::virtio::device::VirtioDevice;
    use crate::devices::virtio::mem::device::test_utils::default_virtio_mem;
    use crate::snapshot::assert_apply_restores;
    use crate::utils::u32_mib_to_bytes;
    use crate::vstate::vm::tests::setup_vm_with_memory;

    #[test]
    fn test_save_state() {
        let dev = default_virtio_mem();
        let state = dev.save();

        assert_eq!(state.addr, dev.config.addr);
        assert_eq!(state.region_size, dev.config.region_size);
        assert_eq!(state.block_size, dev.config.block_size);
        assert_eq!(state.usable_region_size, dev.config.usable_region_size);
        assert_eq!(
            state.plugged_blocks.iter().collect::<BitVec>(),
            dev.plugged_blocks
        );
        assert_eq!(state.requested_size, dev.config.requested_size);
        assert_eq!(state.slot_size, dev.slot_size);
    }

    #[test]
    fn test_save_restore_state() {
        let mut original = default_virtio_mem();
        original.set_acked_features(original.avail_features());
        original.queues_mut()[0].size = 128;
        original.queues_mut()[0].ready = true;
        original.config.requested_size = u32_mib_to_bytes(128);
        original.config.usable_region_size = u32_mib_to_bytes(512);
        original.plugged_blocks.set(1, true);
        original.config.plugged_size = u32_mib_to_bytes(2);
        let state = bitcode::deserialize(&bitcode::serialize(&original.save()).unwrap()).unwrap();
        let vm = Arc::new(setup_vm_with_memory(0x1000));

        let mut restored =
            VirtioMem::provision(VirtioMemProvisionArgs::new(Arc::clone(&vm)), &state).unwrap();
        let activate_fd = restored.activate_event().as_raw_fd();
        let queue_fd = restored.queue_events()[0].as_raw_fd();
        assert_apply_restores(&mut restored, &state, vm.guest_memory());
        assert_eq!(restored.config.plugged_size, u32_mib_to_bytes(2));
        assert_eq!(restored.activate_event().as_raw_fd(), activate_fd);
        assert_eq!(restored.queue_events()[0].as_raw_fd(), queue_fd);
    }
}
