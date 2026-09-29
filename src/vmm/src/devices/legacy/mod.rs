// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

//! Implements legacy devices (UART, RTC etc).
mod i8042;
#[cfg(target_arch = "aarch64")]
pub mod rtc_pl031;
pub mod serial;

use std::io;
use std::ops::Deref;
use std::sync::Arc;

use serde::Serializer;
use serde::ser::SerializeMap;
use vm_superio::Trigger;
use vmm_sys_util::eventfd::EventFd;

pub use self::i8042::{I8042Device, I8042Error as I8042DeviceError};
#[cfg(target_arch = "aarch64")]
pub use self::rtc_pl031::RTCDevice;
pub use self::serial::{SerialDevice, SerialEventsWrapper, SerialWrapper};

/// Wrapper for implementing the trigger functionality for `EventFd`.
///
/// The trigger is used for handling events in the legacy devices.
#[derive(Debug)]
pub struct EventFdTrigger(Arc<EventFd>);

impl Trigger for EventFdTrigger {
    type E = io::Error;

    fn trigger(&self) -> io::Result<()> {
        self.write(1)
    }
}

impl Deref for EventFdTrigger {
    type Target = EventFd;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl EventFdTrigger {
    /// Clone an `EventFdTrigger` with a duplicated file descriptor.
    pub fn try_clone(&self) -> io::Result<Self> {
        Ok(EventFdTrigger::new((**self).try_clone()?))
    }

    /// Share the same file descriptor, preserving its event-loop registrations.
    pub fn shared_clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }

    /// Create an `EventFdTrigger`.
    pub fn new(evt: EventFd) -> Self {
        Self(Arc::new(evt))
    }

    /// Get the associated event fd out of an `EventFdTrigger`.
    pub fn get_event(&self) -> EventFd {
        self.0.try_clone().unwrap()
    }
}

/// Called by METRICS.flush(), this function facilitates serialization of aggregated metrics.
pub fn flush_metrics<S: Serializer>(serializer: S) -> Result<S::Ok, S::Error> {
    let mut seq = serializer.serialize_map(Some(1))?;
    seq.serialize_entry("i8042", &i8042::METRICS)?;
    #[cfg(target_arch = "aarch64")]
    seq.serialize_entry("rtc", &rtc_pl031::METRICS)?;
    seq.serialize_entry("uart", &serial::METRICS)?;
    seq.end()
}
