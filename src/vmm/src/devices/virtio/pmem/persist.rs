// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::device::{ConfigSpace, Pmem, PmemError};
use crate::devices::virtio::persist::{PersistError as VirtioStateError, VirtioDeviceState};
use crate::devices::virtio::pmem::PMEM_QUEUE_SIZE;
use crate::devices::virtio::queue::Queue;
use crate::rate_limiter::persist::RateLimiterState;
use crate::snapshot::Persist;
use crate::vmm_config::pmem::PmemConfig;
use crate::vstate::memory::GuestMemoryMmap;
use crate::vstate::vm::{KvmVm, VmError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PmemState {
    pub virtio_state: VirtioDeviceState,
    pub config_space: ConfigSpace,
    pub config: PmemConfig,
    pub rate_limiter_state: RateLimiterState,
}

#[derive(Debug)]
pub struct PmemConstructorArgs {
    pub vm: Arc<KvmVm>,
}

#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum PmemPersistError {
    /// Error resetting VirtIO state: {0}
    VirtioState(#[from] VirtioStateError),
    /// Error creating Pmem devie: {0}
    Pmem(#[from] PmemError),
    /// Error registering memory region: {0}
    KvmVm(#[from] VmError),
    /// Error restoring rate limiter: {0}
    RateLimiter(std::io::Error),
}

impl<'a> Persist<'a> for Pmem {
    type State = PmemState;
    type ConstructorArgs = PmemConstructorArgs;
    type ApplyArgs = &'a GuestMemoryMmap;
    type Error = PmemPersistError;

    fn save(&self) -> Self::State {
        PmemState {
            virtio_state: VirtioDeviceState::from_device(self),
            config_space: self.guest_region.config_space,
            config: self.config.clone(),
            rate_limiter_state: self.rate_limiter.save(),
        }
    }

    fn create(
        constructor_args: Self::ConstructorArgs,
        state: &Self::State,
    ) -> Result<Self, Self::Error> {
        Ok(Pmem::new_with_queues(
            constructor_args.vm,
            state.config.clone(),
            vec![Queue::new(PMEM_QUEUE_SIZE)],
            0,
            Some(state.config_space),
        )?)
    }

    /// Keeps the backing-file mapping, KVM memory slot, eventfds and rate limiter timer.
    /// This does not revert backing-file contents; reset rejects writable pmem before applying
    /// state, while snapshot load supports both read-only and writable devices.
    fn restore_in_place(
        &mut self,
        state: &Self::State,
        mem: &GuestMemoryMmap,
    ) -> Result<(), Self::Error> {
        state.virtio_state.apply_to(self, mem)?;
        self.rate_limiter
            .restore_in_place(&state.rate_limiter_state, ())
            .map_err(PmemPersistError::RateLimiter)
    }

    fn check_reset(&self, _state: &Self::State) -> Result<(), crate::snapshot::ResetUnsupported> {
        if self.config.read_only {
            Ok(())
        } else {
            Err(crate::snapshot::ResetUnsupported("writable pmem devices"))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches;
    use std::os::fd::AsRawFd;

    use vmm_sys_util::tempfile::TempFile;

    use super::*;
    use crate::devices::virtio::device::VirtioDevice;
    use crate::devices::virtio::test_utils::default_mem;
    use crate::vstate::memory::ByteValued;
    use crate::vstate::vm::tests::setup_vm;

    #[test]
    fn test_persistence() {
        let file = TempFile::new().unwrap();
        file.as_file().set_len(0x20_0000).unwrap();
        let config = PmemConfig {
            path_on_host: file.as_path().to_str().unwrap().to_string(),
            read_only: true,
            ..Default::default()
        };
        let mem = default_mem();
        let vm = Arc::new(setup_vm());
        let mut pmem = Pmem::new(vm.clone(), config).unwrap();
        pmem.set_acked_features(pmem.avail_features());
        let data = bitcode::serialize(&pmem.save()).unwrap();
        drop(pmem);
        let state: PmemState = bitcode::deserialize(&data).unwrap();
        let mut restored =
            crate::snapshot::restore_for_test::<Pmem>(PmemConstructorArgs { vm }, &state, &mem)
                .unwrap();
        assert_eq!(restored.config, state.config);
        assert_eq!(restored.config_as_bytes(), state.config_space.as_slice());
        assert_eq!(restored.acked_features(), state.virtio_state.acked_features);

        let resources = |dev: &Pmem| {
            (
                dev.mmap.mmap_ptr,
                dev.queue_events[0].as_raw_fd(),
                dev.activate_event.as_raw_fd(),
                dev.rate_limiter.as_raw_fd(),
            )
        };
        let before = resources(&restored);
        restored.set_acked_features(0);
        restored.restore_in_place(&state, &mem).unwrap();
        assert_eq!(restored.acked_features(), state.virtio_state.acked_features);
        assert_eq!(resources(&restored), before);
    }

    #[test]
    fn test_restore_rejects_mismatched_config_space_size() {
        let dummy_file = TempFile::new().unwrap();
        dummy_file.as_file().set_len(0x20_0000).unwrap();
        let dummy_path = dummy_file.as_path().to_str().unwrap().to_string();
        let config = PmemConfig {
            id: "1".into(),
            path_on_host: dummy_path,
            root_device: true,
            read_only: false,
            ..Default::default()
        };
        let guest_mem = default_mem();
        let vm = Arc::new(setup_vm());
        let pmem = Pmem::new(vm.clone(), config).unwrap();

        let mut pmem_state = pmem.save();
        drop(pmem);

        pmem_state.config_space.size += Pmem::ALIGNMENT;

        let err = crate::snapshot::restore_for_test::<Pmem>(
            PmemConstructorArgs { vm: vm.clone() },
            &pmem_state,
            &guest_mem,
        )
        .unwrap_err();
        assert_matches!(
            err,
            PmemPersistError::Pmem(PmemError::RestoredSizeMismatch(_, _))
        );
    }
}
