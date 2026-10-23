// Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines the structures needed for saving/restoring block devices.

use serde::{Deserialize, Serialize};

use super::VhostUserBlockError;
use super::device::VhostUserBlock;
use crate::devices::virtio::block::CacheType;
use crate::devices::virtio::persist::VirtioDeviceState;
use crate::snapshot::Persist;
use crate::vstate::memory::GuestMemoryMmap;

/// vhost-user block device state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VhostUserBlockState {
    id: String,
    partuuid: Option<String>,
    cache_type: CacheType,
    root_device: bool,
    socket_path: String,
    vu_acked_protocol_features: u64,
    config_space: Vec<u8>,
    virtio_state: VirtioDeviceState,
}

impl<'a> Persist<'a> for VhostUserBlock {
    type State = VhostUserBlockState;
    type ApplyArgs = &'a GuestMemoryMmap;
    type Error = VhostUserBlockError;

    fn save(&self) -> Self::State {
        unimplemented!("VhostUserBlock does not support snapshotting yet");
    }

    fn apply_state(
        &mut self,
        _state: &Self::State,
        _mem: &GuestMemoryMmap,
    ) -> Result<(), Self::Error> {
        Err(VhostUserBlockError::SnapshottingNotSupported)
    }
}
