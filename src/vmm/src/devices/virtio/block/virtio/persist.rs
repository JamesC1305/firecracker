// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines the structures needed for saving/restoring block devices.

use device::ConfigSpace;
use serde::{Deserialize, Serialize};
use vmm_sys_util::eventfd::EventFd;

use super::device::DiskProperties;
use super::*;

use crate::devices::virtio::block::virtio::device::FileEngineType;
use crate::devices::virtio::block::virtio::device::VirtioBlkTopology;
use crate::devices::virtio::block::virtio::metrics::BlockMetricsPerDevice;
use crate::devices::virtio::device::DeviceState;
use crate::devices::virtio::generated::virtio_blk::VIRTIO_BLK_F_RO;
use crate::devices::virtio::persist::VirtioDeviceState;
use crate::devices::virtio::queue::Queue;
use crate::rate_limiter::RateLimiter;
use crate::rate_limiter::persist::RateLimiterState;
use crate::snapshot::Persist;
use crate::vstate::memory::GuestMemoryMmap;

/// Holds info about block's file engine type. Gets saved in snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileEngineTypeState {
    /// Sync File Engine.
    // If the snap version does not contain the `FileEngineType`, it must have been snapshotted
    // on a VM using the Sync backend.
    #[default]
    Sync,
    /// Async File Engine.
    Async,
}

impl From<FileEngineType> for FileEngineTypeState {
    fn from(file_engine_type: FileEngineType) -> Self {
        match file_engine_type {
            FileEngineType::Sync => FileEngineTypeState::Sync,
            FileEngineType::Async => FileEngineTypeState::Async,
        }
    }
}

impl From<FileEngineTypeState> for FileEngineType {
    fn from(file_engine_type_state: FileEngineTypeState) -> Self {
        match file_engine_type_state {
            FileEngineTypeState::Sync => FileEngineType::Sync,
            FileEngineTypeState::Async => FileEngineType::Async,
        }
    }
}

/// Holds info about the block device. Gets saved in snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtioBlockState {
    id: String,
    partuuid: Option<String>,
    cache_type: CacheType,
    root_device: bool,
    disk_path: String,
    pub virtio_state: VirtioDeviceState,
    rate_limiter_state: RateLimiterState,
    file_engine_type: FileEngineTypeState,
    blk_size: u32,
    topology: VirtioBlkTopology,
    discard_sector_alignment: u32,
}

impl<'a> Persist<'a> for VirtioBlock {
    type State = VirtioBlockState;
    type ConstructorArgs = ();
    type ApplyArgs = &'a GuestMemoryMmap;
    type Error = VirtioBlockError;

    fn save(&self) -> Self::State {
        // Save device state.
        VirtioBlockState {
            id: self.id.clone(),
            partuuid: self.partuuid.clone(),
            cache_type: self.cache_type,
            root_device: self.root_device,
            disk_path: self.disk.file_path.clone(),
            virtio_state: VirtioDeviceState::from_device(self),
            rate_limiter_state: self.rate_limiter.save(),
            file_engine_type: FileEngineTypeState::from(self.file_engine_type()),
            blk_size: self.config_space.blk_size,
            topology: self.config_space.topology,
            discard_sector_alignment: self.config_space.discard_sector_alignment,
        }
    }

    fn create(_: Self::ConstructorArgs, state: &Self::State) -> Result<Self, Self::Error> {
        let is_read_only = state.virtio_state.avail_features & (1u64 << VIRTIO_BLK_F_RO) != 0;
        let rate_limiter = RateLimiter::default();

        let disk_properties = DiskProperties::new(
            state.disk_path.clone(),
            is_read_only,
            state.file_engine_type.into(),
        )?;

        let queue_evts = [EventFd::new(libc::EFD_NONBLOCK).map_err(VirtioBlockError::EventFd)?];

        let queues = BLOCK_QUEUE_SIZES.iter().map(|&s| Queue::new(s)).collect();

        let config_space = ConfigSpace {
            capacity: disk_properties.nsectors.to_le(),
            blk_size: state.blk_size,
            topology: state.topology,
            discard_sector_alignment: state.discard_sector_alignment,
            ..Default::default()
        };

        Ok(Self {
            avail_features: state.virtio_state.avail_features,
            acked_features: 0,
            config_space,
            activate_evt: EventFd::new(libc::EFD_NONBLOCK).map_err(VirtioBlockError::EventFd)?,

            queues,
            queue_evts,
            device_state: DeviceState::Inactive,

            id: state.id.clone(),
            partuuid: state.partuuid.clone(),
            cache_type: state.cache_type,
            root_device: state.root_device,
            read_only: is_read_only,

            disk: disk_properties,
            rate_limiter,
            is_io_engine_throttled: false,
            metrics: BlockMetricsPerDevice::alloc(state.id.clone()),
        })
    }

    /// Keeps the open disk, eventfds, rate limiter timer and metrics.
    fn restore_in_place(
        &mut self,
        state: &Self::State,
        mem: &GuestMemoryMmap,
    ) -> Result<(), Self::Error> {
        state
            .virtio_state
            .apply_to(self, mem)
            .map_err(VirtioBlockError::Persist)?;
        self.rate_limiter
            .restore_in_place(&state.rate_limiter_state, ())
            .map_err(VirtioBlockError::RateLimiter)
    }

    /// In-flight asynchronous I/O can write guest memory after reset reverted it.
    fn check_reset(&self, _state: &Self::State) -> Result<(), crate::snapshot::ResetUnsupported> {
        if self.file_engine_type() == FileEngineType::Async {
            return Err(crate::snapshot::ResetUnsupported(
                "block devices with the async I/O engine",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsRawFd;

    use vmm_sys_util::tempfile::TempFile;

    use super::*;
    use crate::devices::virtio::block::virtio::device::VirtioBlockConfig;
    use crate::devices::virtio::block::virtio::test_utils::default_block_with_path;
    use crate::devices::virtio::device::VirtioDevice;
    use crate::devices::virtio::test_utils::default_mem;

    #[test]
    fn test_cache_semantic_ser() {
        // We create the backing file here so that it exists for the whole lifetime of the test.
        let f = TempFile::new().unwrap();
        f.as_file().set_len(0x1000).unwrap();

        let config = VirtioBlockConfig {
            drive_id: "test".to_string(),
            path_on_host: f.as_path().to_str().unwrap().to_string(),
            is_root_device: false,
            partuuid: None,
            is_read_only: false,
            discard: false,
            cache_type: CacheType::Writeback,
            rate_limiter: None,
            file_engine_type: FileEngineType::default(),
            blk_size: None,
            topology: None,
        };

        let block = VirtioBlock::new(config).unwrap();

        // Save the block device.
        let block_state = block.save();
        let _serialized_data = bitcode::serialize(&block_state).unwrap();
    }

    #[test]
    fn test_file_engine_type() {
        // Test conversions between FileEngineType and FileEngineTypeState.
        assert_eq!(
            FileEngineTypeState::Async,
            FileEngineTypeState::from(FileEngineType::Async)
        );
        assert_eq!(
            FileEngineTypeState::Sync,
            FileEngineTypeState::from(FileEngineType::Sync)
        );
        assert_eq!(FileEngineType::Async, FileEngineTypeState::Async.into());
        assert_eq!(FileEngineType::Sync, FileEngineTypeState::Sync.into());
        // Test default impl.
        assert_eq!(FileEngineTypeState::default(), FileEngineTypeState::Sync);
    }

    #[test]
    fn test_persistence() {
        let f = TempFile::new().unwrap();
        f.as_file().set_len(0x1000).unwrap();
        let mut block = default_block_with_path(
            f.as_path().to_str().unwrap().to_string(),
            FileEngineType::Sync,
        );
        block.set_acked_features(block.avail_features());
        block.config_space.blk_size = 4096;
        let mem = default_mem();
        let data = bitcode::serialize(&block.save()).unwrap();
        let state = bitcode::deserialize(&data).unwrap();
        let mut restored =
            crate::snapshot::restore_for_test::<VirtioBlock>((), &state, &mem).unwrap();
        assert_eq!(restored.config_space, block.config_space);
        assert_eq!(restored.disk.file_path, block.disk.file_path);
        assert_eq!(restored.acked_features(), block.acked_features());

        let fds = |dev: &VirtioBlock| {
            [
                dev.disk.file_engine.file().as_raw_fd(),
                dev.queue_evts[0].as_raw_fd(),
                dev.activate_evt.as_raw_fd(),
                dev.rate_limiter.as_raw_fd(),
            ]
        };
        let before = fds(&restored);
        restored.set_acked_features(0);
        restored.restore_in_place(&state, &mem).unwrap();
        assert_eq!(restored.acked_features(), block.acked_features());
        assert_eq!(fds(&restored), before);
    }
}
