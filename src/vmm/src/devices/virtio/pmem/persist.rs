// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::device::{ConfigSpace, Pmem, PmemError};
use crate::devices::virtio::persist::{PersistError as VirtioStateError, VirtioDeviceState};
use crate::rate_limiter::persist::RateLimiterState;
use crate::snapshot::{Persist, Provision};
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
pub struct PmemProvisionArgs {
    pub vm: Arc<KvmVm>,
}

#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum PmemPersistError {
    /// Error applying VirtIO state: {0}
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

    /// Keeps the backing-file mapping, KVM memory slot, eventfds and rate limiter timer.
    fn apply_state(
        &mut self,
        state: &Self::State,
        mem: &GuestMemoryMmap,
    ) -> Result<(), Self::Error> {
        state.virtio_state.apply_to(self, mem)?;
        self.rate_limiter
            .apply_state(&state.rate_limiter_state, ())
            .map_err(PmemPersistError::RateLimiter)
    }
}

impl Provision<'_> for Pmem {
    type ProvisionArgs = PmemProvisionArgs;

    fn provision(args: Self::ProvisionArgs, state: &Self::State) -> Result<Self, Self::Error> {
        Ok(Pmem::new_with_config_space(
            args.vm,
            state.config.clone(),
            Some(state.config_space),
        )?)
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
    use crate::snapshot::assert_apply_restores;
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
        pmem.queues_mut()[0].size = 128;
        pmem.queues_mut()[0].ready = true;
        let data = bitcode::serialize(&pmem.save()).unwrap();
        drop(pmem);
        let state: PmemState = bitcode::deserialize(&data).unwrap();
        let mut restored = Pmem::provision(PmemProvisionArgs { vm }, &state).unwrap();

        let resources = |dev: &Pmem| {
            (
                dev.mmap.mmap_ptr,
                dev.queue_events[0].as_raw_fd(),
                dev.activate_event.as_raw_fd(),
                dev.rate_limiter.as_raw_fd(),
            )
        };
        let before = resources(&restored);
        assert_apply_restores(&mut restored, &state, &mem);
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
            PmemProvisionArgs { vm: vm.clone() },
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
