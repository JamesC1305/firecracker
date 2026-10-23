// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines the structures needed for saving/restoring balloon devices.

use serde::{Deserialize, Serialize};

use super::*;
use crate::devices::virtio::balloon::device::{BalloonStats, ConfigSpace, HintingState};
use crate::devices::virtio::persist::VirtioDeviceState;
use crate::snapshot::{Persist, Provision};
use crate::vstate::memory::GuestMemoryMmap;

/// Information about the balloon config's that are saved
/// at snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalloonConfigSpaceState {
    num_pages: u32,
    actual_pages: u32,
}

/// Information about the balloon stats that are saved
/// at snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalloonStatsState {
    swap_in: Option<u64>,
    swap_out: Option<u64>,
    major_faults: Option<u64>,
    minor_faults: Option<u64>,
    free_memory: Option<u64>,
    total_memory: Option<u64>,
    available_memory: Option<u64>,
    disk_caches: Option<u64>,
    hugetlb_allocations: Option<u64>,
    hugetlb_failures: Option<u64>,
    oom_kill: Option<u64>,
    alloc_stall: Option<u64>,
    async_scan: Option<u64>,
    direct_scan: Option<u64>,
    async_reclaim: Option<u64>,
    direct_reclaim: Option<u64>,
}

impl BalloonStatsState {
    fn from_stats(stats: &BalloonStats) -> Self {
        Self {
            swap_in: stats.swap_in,
            swap_out: stats.swap_out,
            major_faults: stats.major_faults,
            minor_faults: stats.minor_faults,
            free_memory: stats.free_memory,
            total_memory: stats.total_memory,
            available_memory: stats.available_memory,
            disk_caches: stats.disk_caches,
            hugetlb_allocations: stats.hugetlb_allocations,
            hugetlb_failures: stats.hugetlb_failures,
            oom_kill: stats.oom_kill,
            alloc_stall: stats.alloc_stall,
            async_scan: stats.async_scan,
            direct_scan: stats.direct_scan,
            async_reclaim: stats.async_reclaim,
            direct_reclaim: stats.direct_reclaim,
        }
    }

    fn create_stats(&self) -> BalloonStats {
        BalloonStats {
            target_pages: 0,
            actual_pages: 0,
            target_mib: 0,
            actual_mib: 0,
            swap_in: self.swap_in,
            swap_out: self.swap_out,
            major_faults: self.major_faults,
            minor_faults: self.minor_faults,
            free_memory: self.free_memory,
            total_memory: self.total_memory,
            available_memory: self.available_memory,
            disk_caches: self.disk_caches,
            hugetlb_allocations: self.hugetlb_allocations,
            hugetlb_failures: self.hugetlb_failures,
            oom_kill: self.oom_kill,
            alloc_stall: self.alloc_stall,
            async_scan: self.async_scan,
            direct_scan: self.direct_scan,
            async_reclaim: self.async_reclaim,
            direct_reclaim: self.direct_reclaim,
        }
    }
}

/// Information about the balloon that are saved
/// at snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalloonState {
    stats_polling_interval_s: u16,
    stats_desc_index: Option<u16>,
    latest_stats: BalloonStatsState,
    config_space: BalloonConfigSpaceState,
    hinting_state: HintingState,
    pub virtio_state: VirtioDeviceState,
}

impl<'a> Persist<'a> for Balloon {
    type State = BalloonState;
    type ApplyArgs = &'a GuestMemoryMmap;
    type Error = super::BalloonError;

    fn save(&self) -> Self::State {
        BalloonState {
            stats_polling_interval_s: self.stats_polling_interval_s,
            stats_desc_index: self.stats_desc_index,
            latest_stats: BalloonStatsState::from_stats(&self.latest_stats),
            hinting_state: self.hinting_state,
            config_space: BalloonConfigSpaceState {
                num_pages: self.config_space.num_pages,
                actual_pages: self.config_space.actual_pages,
            },
            virtio_state: VirtioDeviceState::from_device(self),
        }
    }

    /// Applies saved state while keeping the eventfds and stats timer.
    fn apply_state(
        &mut self,
        state: &Self::State,
        mem: &GuestMemoryMmap,
    ) -> Result<(), Self::Error> {
        state
            .virtio_state
            .apply_to(self, mem)
            .map_err(|_| Self::Error::QueueRestoreError)?;
        self.avail_features = state.virtio_state.avail_features;
        self.stats_polling_interval_s = state.stats_polling_interval_s;
        self.latest_stats = state.latest_stats.create_stats();
        self.config_space = ConfigSpace {
            num_pages: state.config_space.num_pages,
            actual_pages: state.config_space.actual_pages,
            // On restore allow the guest to reclaim pages
            free_page_hint_cmd_id: FREE_PAGE_HINT_DONE,
        };
        self.hinting_state = state.hinting_state;

        if state.virtio_state.activated && self.stats_enabled() {
            self.set_stats_desc_index(state.stats_desc_index);
            self.update_timer_state();
        }

        Ok(())
    }
}

impl Provision<'_> for Balloon {
    type ProvisionArgs = ();

    fn provision(_: Self::ProvisionArgs, state: &Self::State) -> Result<Self, Self::Error> {
        let free_page_hinting =
            state.virtio_state.avail_features & (1u64 << VIRTIO_BALLOON_F_FREE_PAGE_HINTING) != 0;
        let free_page_reporting =
            state.virtio_state.avail_features & (1u64 << VIRTIO_BALLOON_F_FREE_PAGE_REPORTING) != 0;
        Balloon::new(
            0,
            false,
            state.stats_polling_interval_s,
            free_page_hinting,
            free_page_reporting,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsRawFd;

    use super::*;
    use crate::devices::virtio::device::VirtioDevice;
    use crate::devices::virtio::test_utils::default_mem;
    use crate::devices::virtio::test_utils::test::VirtioTestHelper;

    #[test]
    fn test_persistence() {
        let guest_mem = default_mem();
        let balloon = Balloon::new(0x42, false, 2, true, true).unwrap();
        let max_size = balloon.queues[0].max_size;
        let mut th = VirtioTestHelper::new(&guest_mem, balloon);
        for queue in &mut th.device().queues {
            queue.max_size = max_size;
        }
        th.activate_device(&guest_mem);
        let mut balloon = th.device();
        balloon.stats_desc_index = Some(7);
        balloon.config_space.actual_pages = 17;
        balloon.latest_stats = BalloonStats {
            swap_in: Some(1),
            swap_out: Some(2),
            major_faults: Some(3),
            minor_faults: Some(4),
            free_memory: Some(0x1234),
            total_memory: Some(6),
            available_memory: Some(7),
            disk_caches: Some(8),
            hugetlb_allocations: Some(9),
            hugetlb_failures: Some(10),
            oom_kill: Some(11),
            alloc_stall: Some(12),
            async_scan: Some(13),
            direct_scan: Some(14),
            async_reclaim: Some(15),
            direct_reclaim: Some(16),
            ..Default::default()
        };
        balloon.hinting_state = HintingState {
            host_cmd: 37,
            last_cmd_id: 35,
            guest_cmd: Some(36),
            acknowledge_on_finish: true,
        };
        let state = bitcode::deserialize(&bitcode::serialize(&balloon.save()).unwrap()).unwrap();

        let mut balloon = Balloon::provision((), &state).unwrap();
        let timer_fd = balloon.stats_timer.as_raw_fd();
        let queue_fd = balloon.queue_evts[0].as_raw_fd();
        let activate_fd = balloon.activate_evt.as_raw_fd();
        balloon.apply_state(&state, &guest_mem).unwrap();
        // Activation is deferred to load; every other saved field must round-trip.
        assert!(!balloon.is_activated());
        let mut saved = balloon.save();
        saved.virtio_state.activated = state.virtio_state.activated;
        assert_eq!(
            bitcode::serialize(&saved).unwrap(),
            bitcode::serialize(&state).unwrap()
        );
        assert_eq!(
            balloon.config_space.free_page_hint_cmd_id,
            FREE_PAGE_HINT_DONE
        );
        assert!(balloon.stats_timer.is_armed());
        assert_eq!(balloon.stats_timer.as_raw_fd(), timer_fd);
        assert_eq!(balloon.queue_evts[0].as_raw_fd(), queue_fd);
        assert_eq!(balloon.activate_evt.as_raw_fd(), activate_fd);
    }
}
