// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines snapshot lifecycles for devices and their runtime state.

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
    /// The no-op default is temporary scaffolding for unconverted components whose
    /// `provision` still applies their saved state. Converted components must implement this.
    fn apply_state(
        &mut self,
        _state: &Self::State,
        _args: Self::ApplyArgs,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
    /// Finishes load-only activation and subscriptions after runtime state is applied.
    fn post_restore(
        &mut self,
        _state: &Self::State,
        _load: &mut LoadContext<'_>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Construction capability for components that can be loaded from a snapshot.
pub trait Provision<'a>: Persist<'a> {
    /// The type of the object holding the provisioning arguments.
    type ProvisionArgs;

    /// Creates the component. Unconverted components temporarily apply saved state here.
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
