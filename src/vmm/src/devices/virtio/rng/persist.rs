// Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines the structures needed for saving/restoring entropy devices.

use serde::{Deserialize, Serialize};

use crate::devices::virtio::persist::{PersistError as VirtioStateError, VirtioDeviceState};
use crate::devices::virtio::rng::{Entropy, EntropyError};
use crate::rate_limiter::RateLimiter;
use crate::rate_limiter::persist::RateLimiterState;
use crate::snapshot::{Persist, Provision};
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
    type ApplyArgs = &'a GuestMemoryMmap;
    type Error = EntropyPersistError;

    fn save(&self) -> Self::State {
        EntropyState {
            virtio_state: VirtioDeviceState::from_device(self),
            rate_limiter_state: self.rate_limiter().save(),
        }
    }

    /// Keeps the eventfds and the rate limiter timer.
    fn apply_state(
        &mut self,
        state: &Self::State,
        mem: &GuestMemoryMmap,
    ) -> Result<(), Self::Error> {
        state.virtio_state.apply_to(self, mem)?;
        self.rate_limiter
            .apply_state(&state.rate_limiter_state, ())?;
        self.set_avail_features(state.virtio_state.avail_features);
        Ok(())
    }
}

impl Provision<'_> for Entropy {
    type ProvisionArgs = ();

    fn provision(_: Self::ProvisionArgs, _state: &Self::State) -> Result<Self, Self::Error> {
        Ok(Entropy::new(RateLimiter::default())?)
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsRawFd;

    use super::*;
    use crate::devices::virtio::device::VirtioDevice;
    use crate::devices::virtio::test_utils::default_mem;
    use crate::snapshot::assert_apply_restores;

    #[test]
    fn test_persistence() {
        let mut entropy = Entropy::new(RateLimiter::default()).unwrap();
        entropy.set_acked_features(entropy.avail_features());
        entropy.queues_mut()[0].size = 128;
        entropy.queues_mut()[0].ready = true;
        let mem = default_mem();
        let data = bitcode::serialize(&entropy.save()).unwrap();
        let state = bitcode::deserialize(&data).unwrap();
        let fds = |dev: &Entropy| {
            [
                dev.queue_events()[0].as_raw_fd(),
                dev.activate_event().as_raw_fd(),
                dev.rate_limiter.as_raw_fd(),
            ]
        };
        let mut restored = Entropy::provision((), &state).unwrap();
        let before = fds(&restored);
        assert_apply_restores(&mut restored, &state, &mem);
        assert_eq!(fds(&restored), before);
    }
}
