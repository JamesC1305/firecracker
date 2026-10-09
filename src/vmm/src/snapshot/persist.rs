// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines snapshot lifecycles for devices and their runtime state.

/// The snapshot lifecycle of a device or a component made of devices.
pub trait Persist<'a>: Sized {
    /// The type of the object representing the state of the component.
    type State;
    /// The type of the error that can occur while restoring the component.
    type Error;

    /// Returns the current state of the component.
    fn save(&self) -> Self::State;
}

/// Construction capability for components that can be loaded from a snapshot.
pub trait Provision<'a>: Persist<'a> {
    /// The type of the object holding the provisioning arguments.
    type ProvisionArgs;

    /// Creates the component. Components without in-place restoration also apply state here.
    fn provision(args: Self::ProvisionArgs, state: &Self::State) -> Result<Self, Self::Error>;
}
