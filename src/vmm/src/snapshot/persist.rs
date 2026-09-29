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

/// The snapshot lifecycle of a device or a component made of devices.
pub trait Persist<'a>: Sized {
    /// The type of the object representing the state of the component.
    type State;
    /// The type of the object holding the constructor arguments. Components that their owner
    /// creates, such as virtio transports, use `std::convert::Infallible`.
    type ConstructorArgs;
    /// Inputs needed by both snapshot load and in-place reset.
    type ApplyArgs;
    /// The type of the error that can occur while restoring the component.
    type Error;

    /// Returns the current state of the component.
    fn save(&self) -> Self::State;
    /// Creates host resources and fixed configuration, without applying runtime state.
    fn create(
        constructor_args: Self::ConstructorArgs,
        state: &Self::State,
    ) -> Result<Self, Self::Error>;
    /// Applies all runtime state, retaining host resources on a live component.
    fn restore_in_place(
        &mut self,
        state: &Self::State,
        args: Self::ApplyArgs,
    ) -> Result<(), Self::Error>;
    /// Finishes load-only work after runtime state is applied. Reset never calls this.
    fn post_restore(
        &mut self,
        _state: &Self::State,
        _load: &mut LoadContext<'_>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Loads a component from `state`: creates it, applies its runtime state, then finishes the
/// load-only work. Device managers run each phase on all of their children before the next.
pub fn restore<'a, T: Persist<'a>>(
    args: T::ConstructorArgs,
    state: &T::State,
    apply: T::ApplyArgs,
    load: &mut LoadContext<'_>,
) -> Result<T, T::Error> {
    let mut value = T::create(args, state)?;
    value.restore_in_place(state, apply)?;
    value.post_restore(state, load)?;
    Ok(value)
}

#[cfg(test)]
pub(crate) fn restore_for_test<'a, T: Persist<'a>>(
    args: T::ConstructorArgs,
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
