// Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines the structures needed for saving/restoring entropy devices.

use serde::{Deserialize, Serialize};

use crate::devices::virtio::persist::{PersistError as VirtioStateError, VirtioDeviceState};
use crate::devices::virtio::rng::{Entropy, EntropyError};
use crate::rate_limiter::RateLimiter;
use crate::rate_limiter::persist::RateLimiterState;
use crate::snapshot::Persist;
use crate::vstate::memory::GuestMemoryMmap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyState {
    pub virtio_state: VirtioDeviceState,
    rate_limiter_state: RateLimiterState,
}

#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum EntropyPersistError {
    /// Create entropy: {0}
    CreateEntropy(#[from] EntropyError),
    /// Virtio state: {0}
    VirtioState(#[from] VirtioStateError),
    /// Restore rate limiter: {0}
    RestoreRateLimiter(#[from] std::io::Error),
}

impl<'a> Persist<'a> for Entropy {
    type State = EntropyState;
    type ConstructorArgs = ();
    type ApplyArgs = &'a GuestMemoryMmap;
    type Error = EntropyPersistError;

    fn save(&self) -> Self::State {
        EntropyState {
            virtio_state: VirtioDeviceState::from_device(self),
            rate_limiter_state: self.rate_limiter().save(),
        }
    }

    fn create(_: Self::ConstructorArgs, _state: &Self::State) -> Result<Self, Self::Error> {
        Ok(Entropy::new(RateLimiter::default())?)
    }

    /// Keeps the eventfds and the rate limiter timer.
    fn restore_in_place(
        &mut self,
        state: &Self::State,
        mem: &GuestMemoryMmap,
    ) -> Result<(), Self::Error> {
        state.virtio_state.apply_to(self, mem)?;
        self.rate_limiter
            .restore_in_place(&state.rate_limiter_state, ())?;
        self.set_avail_features(state.virtio_state.avail_features);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsRawFd;

    use super::*;
    use crate::devices::virtio::device::VirtioDevice;
    use crate::devices::virtio::test_utils::default_mem;

    #[test]
    fn test_persistence() {
        let mut entropy = Entropy::new(RateLimiter::default()).unwrap();
        entropy.set_acked_features(entropy.avail_features());
        let mem = default_mem();
        let data = bitcode::serialize(&entropy.save()).unwrap();
        let state = bitcode::deserialize(&data).unwrap();
        let mut restored = crate::snapshot::restore_for_test::<Entropy>((), &state, &mem).unwrap();
        assert_eq!(restored.acked_features(), entropy.acked_features());

        let fds = |dev: &Entropy| {
            [
                dev.queue_events()[0].as_raw_fd(),
                dev.activate_event().as_raw_fd(),
                dev.rate_limiter.as_raw_fd(),
            ]
        };
        let before = fds(&restored);
        restored.set_acked_features(0);
        restored.restore_in_place(&state, &mem).unwrap();
        assert_eq!(restored.acked_features(), entropy.acked_features());
        assert_eq!(fds(&restored), before);
    }
}
