// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines the structures needed for saving/restoring a RateLimiter.

use std::io;

use serde::{Deserialize, Serialize};

use super::*;
use crate::snapshot::Persist;

/// State for saving a TokenBucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBucketState {
    size: u64,
    one_time_burst: u64,
    refill_time: u64,
    budget: u64,
    elapsed_ns: u64,
}

impl<'a> Persist<'a> for TokenBucket {
    type State = TokenBucketState;
    type ConstructorArgs = ();
    type ApplyArgs = ();
    type Error = io::Error;

    fn save(&self) -> Self::State {
        TokenBucketState {
            size: self.size,
            one_time_burst: self.one_time_burst,
            refill_time: self.refill_time,
            budget: self.budget,
            // This should be safe for a duration of about 584 years.
            elapsed_ns: u64::try_from(self.last_update.elapsed().as_nanos()).unwrap(),
        }
    }

    fn create(_: Self::ConstructorArgs, state: &Self::State) -> Result<Self, Self::Error> {
        Self::new(state.size, state.one_time_burst, state.refill_time)
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))
    }

    /// Restores runtime state after construction from the saved bucket configuration.
    fn restore_in_place(
        &mut self,
        state: &Self::State,
        _: Self::ApplyArgs,
    ) -> Result<(), Self::Error> {
        let now = Instant::now();
        let last_update = now
            .checked_sub(Duration::from_nanos(state.elapsed_ns))
            .unwrap_or(now);

        self.budget = state.budget;
        self.one_time_burst = state.one_time_burst;
        self.last_update = last_update;

        Ok(())
    }
}

/// State for saving a RateLimiter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimiterState {
    ops: Option<TokenBucketState>,
    bandwidth: Option<TokenBucketState>,
}

impl<'a> Persist<'a> for RateLimiter {
    type State = RateLimiterState;
    type ConstructorArgs = ();
    type ApplyArgs = ();
    type Error = io::Error;

    fn save(&self) -> Self::State {
        RateLimiterState {
            ops: self.ops.as_ref().map(|ops| ops.save()),
            bandwidth: self.bandwidth.as_ref().map(|bw| bw.save()),
        }
    }

    fn create(_: Self::ConstructorArgs, _state: &Self::State) -> Result<Self, Self::Error> {
        Ok(Self::default())
    }

    /// Restores token buckets while retaining the EventManager-registered timer fd.
    fn restore_in_place(
        &mut self,
        state: &Self::State,
        _: Self::ApplyArgs,
    ) -> Result<(), Self::Error> {
        let apply_bucket = |state: &TokenBucketState| -> Result<TokenBucket, io::Error> {
            // Rebuild fixed bucket parameters too, since runtime PATCH can replace them.
            let mut bucket = TokenBucket::create((), state)?;
            bucket.restore_in_place(state, ())?;
            Ok(bucket)
        };
        let ops = state.ops.as_ref().map(apply_bucket).transpose()?;
        let bandwidth = state.bandwidth.as_ref().map(apply_bucket).transpose()?;
        self.ops = ops;
        self.bandwidth = bandwidth;
        // The timer is armed or has an unread expiry only while timer_active is set.
        if self.timer_active {
            self.timer_fd.disarm();
            self.timer_fd.read();
            self.timer_active = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_token_bucket_persistence() {
        let mut bucket = TokenBucket::new(1000, 2000, 3000).unwrap();
        bucket.reduce(2100);
        let serialized_data = bitcode::serialize(&bucket.save()).unwrap();
        let state: TokenBucketState = bitcode::deserialize(&serialized_data).unwrap();
        let mut restored =
            TokenBucket::new(state.size, state.one_time_burst, state.refill_time).unwrap();
        restored.restore_in_place(&state, ()).unwrap();

        assert!(bucket.partial_eq(&restored));
        assert_eq!(restored.budget, 900);
        restored.force_replenish(u64::MAX);
        assert_eq!(restored.budget, 1000);
        assert_eq!(restored.one_time_burst(), 0);
    }

    #[test]
    fn test_rate_limiter_persistence() {
        let refill_time = 100_000;
        let mut rate_limiter = RateLimiter::new(100, 0, refill_time, 10, 0, refill_time);
        assert!(rate_limiter.consume(60, TokenType::Bytes));
        assert!(rate_limiter.consume(4, TokenType::Ops));
        let serialized_data = bitcode::serialize(&rate_limiter.save()).unwrap();
        let state = bitcode::deserialize(&serialized_data).unwrap();

        let mut restored = RateLimiter::default();
        let fd = restored.as_raw_fd();
        restored.restore_in_place(&state, ()).unwrap();
        assert_eq!(restored.as_raw_fd(), fd);
        assert!(!restored.is_blocked());
        assert!(restored.consume(40, TokenType::Bytes));
        assert!(restored.consume(6, TokenType::Ops));
        assert!(!restored.consume(1, TokenType::Bytes));

        restored
            .restore_in_place(
                &RateLimiterState {
                    ops: None,
                    bandwidth: None,
                },
                (),
            )
            .unwrap();
        assert!(!restored.is_blocked());
        assert!(restored.consume(u64::MAX, TokenType::Bytes));
        assert!(restored.consume(u64::MAX, TokenType::Ops));

        restored.restore_in_place(&state, ()).unwrap();
        assert_eq!(restored.as_raw_fd(), fd);
        assert!(restored.consume(40, TokenType::Bytes));
        assert!(restored.consume(6, TokenType::Ops));
        assert!(!restored.consume(1, TokenType::Ops));
    }

    #[test]
    fn test_rate_limiter_apply() {
        use std::os::fd::AsRawFd;

        let mut source = RateLimiter::new(100, 0, 100_000, 10, 0, 100_000);
        assert!(source.consume(10, TokenType::Bytes));
        assert!(source.consume(3, TokenType::Ops));
        let state = source.save();
        let mut live = RateLimiter::new(100, 0, 100_000, 10, 0, 100_000);
        let fd = live.as_raw_fd();
        assert!(live.consume(100, TokenType::Bytes));
        assert!(live.consume(10, TokenType::Ops));
        assert!(!live.consume(1, TokenType::Bytes));
        assert!(live.is_blocked());

        let mut invalid = state.clone();
        invalid.bandwidth.as_mut().unwrap().refill_time = u64::MAX;
        assert_eq!(
            live.restore_in_place(&invalid, ()).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert!(live.is_blocked());
        assert_eq!(live.ops().unwrap().budget(), 0);
        assert_eq!(live.bandwidth().unwrap().budget(), 0);

        live.restore_in_place(&state, ()).unwrap();
        assert_eq!(live.as_raw_fd(), fd);
        assert!(!live.timer_fd.is_armed());
        assert_eq!(live.timer_fd.read(), 0);
        assert!(!live.is_blocked());
        assert!(live.consume(90, TokenType::Bytes));
        assert!(live.consume(7, TokenType::Ops));
        assert!(!live.consume(1, TokenType::Ops));

        for handle_expiry in [false, true] {
            let clock = MockClock::new();
            let mut live = RateLimiter::new_mocked(100, 0, 100_000, 10, 0, 100_000, &clock);
            assert!(live.consume(100, TokenType::Bytes));
            assert!(!live.consume(1, TokenType::Bytes));
            clock.advance(REFILL_TIMER_DURATION);
            if handle_expiry {
                live.event_handler().unwrap();
            }

            live.restore_in_place(&state, ()).unwrap();
            assert!(!live.timer_fd.is_armed());
            assert_eq!(live.timer_fd.read(), 0);
            assert!(!live.is_blocked());
            assert!(live.consume(90, TokenType::Bytes));
            assert!(live.consume(7, TokenType::Ops));
            assert!(!live.consume(1, TokenType::Ops));
        }
    }
}
