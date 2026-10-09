// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Snapshot lifecycles for devices and their runtime state.
//!
//! Load provisions components, applies their saved state, then finishes load-only activation and
//! subscriptions. Managers complete each phase across their children before starting the next.

use crate::EventManager;

/// Resources that only snapshot load uses, after runtime state is applied.
pub struct LoadContext<'a> {
    /// The event loop that will drive the fully restored devices.
    pub event_manager: &'a mut EventManager,
}

impl std::fmt::Debug for LoadContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadContext").finish_non_exhaustive()
    }
}

/// Saving and applying the snapshot state of a device or component.
///
/// Components constructed by their owner need only this trait. Components with snapshot-specific
/// construction also implement [`Provision`].
pub trait Persist<'a>: Sized {
    /// The type of the object representing the state of the component.
    type State;
    /// Inputs needed to apply saved runtime state.
    type ApplyArgs;
    /// The type of the error that can occur while restoring the component.
    type Error;

    /// Returns the current state of the component.
    fn save(&self) -> Self::State;
    /// Applies saved runtime state to a provisioned component, retaining its host resources.
    ///
    /// Restore saved runtime fields without replacing resource owners or registrations.
    /// Saved fields that [`Provision::provision`] consumes, such as paths, addresses and
    /// device layout, are provisioning inputs and are left alone here. Component tests check
    /// the remaining fields with a save/apply/save round trip using `assert_apply_restores`.
    fn apply_state(
        &mut self,
        state: &Self::State,
        args: Self::ApplyArgs,
    ) -> Result<(), Self::Error>;
    /// Finishes load-only activation and subscriptions after runtime state is applied.
    fn post_restore(
        &mut self,
        _state: &Self::State,
        _load: &mut LoadContext<'_>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Snapshot-specific construction of a component's host resources and fixed configuration.
///
/// Saved paths, addresses and device layout may be needed during construction.
pub trait Provision<'a>: Persist<'a> {
    /// Inputs needed to construct the component's host resources and fixed configuration.
    type ProvisionArgs;

    /// Provisions resources and fixed configuration for subsequent state application.
    fn provision(args: Self::ProvisionArgs, state: &Self::State) -> Result<Self, Self::Error>;
}

/// Loads a fully initialized component from `state`.
pub fn restore<'a, T: Provision<'a>>(
    args: T::ProvisionArgs,
    state: &T::State,
    apply: T::ApplyArgs,
    load: &mut LoadContext<'_>,
) -> Result<T, T::Error> {
    let mut value = T::provision(args, state)?;
    value.apply_state(state, apply)?;
    value.post_restore(state, load)?;
    Ok(value)
}

#[cfg(test)]
pub(crate) fn restore_for_test<'a, T: Provision<'a>>(
    args: T::ProvisionArgs,
    state: &T::State,
    apply: T::ApplyArgs,
) -> Result<T, T::Error> {
    let mut event_manager = EventManager::new().unwrap();
    restore::<T>(
        args,
        state,
        apply,
        &mut LoadContext {
            event_manager: &mut event_manager,
        },
    )
}

/// Applies `state` to `live` and checks that saving again reproduces `state`, so a saved field
/// that `apply_state` ignores fails the test. `live` must start out with different values,
/// except for fields that provisioning consumes, which must already match `state`.
#[cfg(test)]
pub(crate) fn assert_apply_restores<'a, T>(live: &mut T, state: &T::State, args: T::ApplyArgs)
where
    T: Persist<'a>,
    T::State: serde::Serialize + std::fmt::Debug,
    T::Error: std::fmt::Debug,
{
    live.apply_state(state, args).unwrap();
    let restored = live.save();
    assert!(
        bitcode::serialize(&restored).unwrap() == bitcode::serialize(state).unwrap(),
        "apply_state did not restore every saved field\nrestored: {restored:?}\n   saved: {state:?}"
    );
}
