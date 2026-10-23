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

impl Persist<'_> for TokenBucket {
    type State = TokenBucketState;
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

    /// Applies the saved budget, one-time burst and idle time.
    ///
    /// This only works on a freshly constructed bucket: the saved `size` and `refill_time` are
    /// not applied, so the bucket must come from `TokenBucket::new(state.size,
    /// state.one_time_burst, state.refill_time)`, which also derives the refill rate and the
    /// replenishable burst limit from that configuration.
    fn apply_state(&mut self, state: &Self::State, _: Self::ApplyArgs) -> Result<(), Self::Error> {
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

impl Persist<'_> for RateLimiter {
    type State = RateLimiterState;
    type ApplyArgs = ();
    type Error = io::Error;

    fn save(&self) -> Self::State {
        RateLimiterState {
            ops: self.ops.as_ref().map(|ops| ops.save()),
            bandwidth: self.bandwidth.as_ref().map(|bw| bw.save()),
        }
    }

    /// Restores token buckets while retaining the EventManager-registered timer fd.
    fn apply_state(&mut self, state: &Self::State, _: Self::ApplyArgs) -> Result<(), Self::Error> {
        let apply_bucket = |state: &TokenBucketState| -> Result<TokenBucket, io::Error> {
            // The provisioned limiter has no buckets; construct the saved configuration.
            let mut bucket = TokenBucket::new(state.size, state.one_time_burst, state.refill_time)
                .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
            bucket.apply_state(state, ())?;
            Ok(bucket)
        };
        let ops = state.ops.as_ref().map(apply_bucket).transpose()?;
        let bandwidth = state.bandwidth.as_ref().map(apply_bucket).transpose()?;
        self.ops = ops;
        self.bandwidth = bandwidth;
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    fn reapply_bucket(state: &TokenBucketState) -> TokenBucket {
        let mut bucket =
            TokenBucket::new(state.size, state.one_time_burst, state.refill_time).unwrap();
        bucket.apply_state(state, ()).unwrap();
        bucket
    }

    fn reapply_limiter(state: &RateLimiterState) -> RateLimiter {
        let mut limiter = RateLimiter::default();
        limiter.apply_state(state, ()).unwrap();
        limiter
    }

    #[test]
    fn test_token_bucket_persistence() {
        let mut tb = TokenBucket::new(1000, 2000, 3000).unwrap();

        // Check that TokenBucket restores correctly if untouched.
        let restored_tb = reapply_bucket(&tb.save());
        assert!(tb.partial_eq(&restored_tb));

        // Check that TokenBucket restores correctly after partially consuming tokens.
        tb.reduce(100);
        let restored_tb = reapply_bucket(&tb.save());
        assert!(tb.partial_eq(&restored_tb));

        // Check that TokenBucket restores correctly after replenishing tokens.
        tb.force_replenish(100);
        let restored_tb = reapply_bucket(&tb.save());
        assert!(tb.partial_eq(&restored_tb));

        // Test serialization.
        let tb_state = tb.save();
        let serialized_data = bitcode::serialize(&tb_state).unwrap();

        let restored_state = bitcode::deserialize(&serialized_data).unwrap();
        let restored_tb = reapply_bucket(&restored_state);
        assert!(tb.partial_eq(&restored_tb));
    }

    #[test]
    fn test_rate_limiter_persistence() {
        let refill_time = 100_000;
        let mut rate_limiter = RateLimiter::new(100, 0, refill_time, 10, 0, refill_time);

        // Check that RateLimiter restores correctly if untouched.
        let restored_rate_limiter = reapply_limiter(&rate_limiter.save());

        assert!(
            rate_limiter
                .ops()
                .unwrap()
                .partial_eq(restored_rate_limiter.ops().unwrap())
        );
        assert!(
            rate_limiter
                .bandwidth()
                .unwrap()
                .partial_eq(restored_rate_limiter.bandwidth().unwrap())
        );
        assert!(!restored_rate_limiter.timer_fd.is_armed());

        // Check that RateLimiter restores correctly after partially consuming tokens.
        rate_limiter.consume(10, TokenType::Bytes);
        rate_limiter.consume(10, TokenType::Ops);
        let restored_rate_limiter = reapply_limiter(&rate_limiter.save());

        assert!(
            rate_limiter
                .ops()
                .unwrap()
                .partial_eq(restored_rate_limiter.ops().unwrap())
        );
        assert!(
            rate_limiter
                .bandwidth()
                .unwrap()
                .partial_eq(restored_rate_limiter.bandwidth().unwrap())
        );
        assert!(!restored_rate_limiter.timer_fd.is_armed());

        // Check that RateLimiter restores correctly after totally consuming tokens.
        rate_limiter.consume(1000, TokenType::Bytes);
        let restored_rate_limiter = reapply_limiter(&rate_limiter.save());

        assert!(
            rate_limiter
                .ops()
                .unwrap()
                .partial_eq(restored_rate_limiter.ops().unwrap())
        );
        assert!(
            rate_limiter
                .bandwidth()
                .unwrap()
                .partial_eq(restored_rate_limiter.bandwidth().unwrap())
        );

        // Test serialization.
        let rate_limiter_state = rate_limiter.save();
        let serialized_data = bitcode::serialize(&rate_limiter_state).unwrap();

        let restored_state = bitcode::deserialize(&serialized_data).unwrap();
        let restored_rate_limiter = reapply_limiter(&restored_state);

        assert!(
            rate_limiter
                .ops()
                .unwrap()
                .partial_eq(restored_rate_limiter.ops().unwrap())
        );
        assert!(
            rate_limiter
                .bandwidth()
                .unwrap()
                .partial_eq(restored_rate_limiter.bandwidth().unwrap())
        );
    }

    #[test]
    fn test_rate_limiter_apply() {
        let mut source = RateLimiter::new(100, 0, 100_000, 10, 0, 100_000);
        assert!(source.consume(10, TokenType::Bytes));
        assert!(source.consume(3, TokenType::Ops));
        let state = source.save();

        let mut limiter = RateLimiter::default();
        let fd = limiter.as_raw_fd();
        let mut invalid = state.clone();
        invalid.bandwidth.as_mut().unwrap().refill_time = u64::MAX;
        assert_eq!(
            limiter.apply_state(&invalid, ()).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert!(limiter.bandwidth().is_none());
        assert!(limiter.ops().is_none());

        limiter.apply_state(&state, ()).unwrap();
        assert_eq!(limiter.as_raw_fd(), fd);
        assert!(!limiter.is_blocked());
        assert_eq!(limiter.bandwidth().unwrap().budget(), 90);
        assert_eq!(limiter.ops().unwrap().budget(), 7);
    }
}
