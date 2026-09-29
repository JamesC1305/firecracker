// Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::convert::TryInto;

use serde::Serialize;
use vm_superio::Rtc;
use vm_superio::rtc_pl031::RtcEvents;

use crate::logger::{IncMetric, SharedIncMetric, warn};

/// Metrics specific to the RTC device.
#[derive(Debug, Serialize, Default)]
pub struct RTCDeviceMetrics {
    /// Errors triggered while using the RTC device.
    pub error_count: SharedIncMetric,
    /// Number of superfluous read intents on this RTC device.
    pub missed_read_count: SharedIncMetric,
    /// Number of superfluous write intents on this RTC device.
    pub missed_write_count: SharedIncMetric,
}

impl RTCDeviceMetrics {
    /// Const default construction.
    pub const fn new() -> Self {
        Self {
            error_count: SharedIncMetric::new(),
            missed_read_count: SharedIncMetric::new(),
            missed_write_count: SharedIncMetric::new(),
        }
    }
}

impl RtcEvents for RTCDeviceMetrics {
    fn invalid_read(&self) {
        self.missed_read_count.inc();
        self.error_count.inc();
        warn!("Guest read at invalid offset.")
    }

    fn invalid_write(&self) {
        self.missed_write_count.inc();
        self.error_count.inc();
        warn!("Guest write at invalid offset.")
    }
}

impl RtcEvents for &'static RTCDeviceMetrics {
    fn invalid_read(&self) {
        RTCDeviceMetrics::invalid_read(self);
    }

    fn invalid_write(&self) {
        RTCDeviceMetrics::invalid_write(self);
    }
}

/// Stores aggregated metrics
pub static METRICS: RTCDeviceMetrics = RTCDeviceMetrics::new();

/// Wrapper over vm_superio's RTC implementation.
#[derive(Debug)]
pub struct RTCDevice(vm_superio::Rtc<&'static RTCDeviceMetrics>);

impl Default for RTCDevice {
    fn default() -> Self {
        RTCDevice(Rtc::with_events(&METRICS))
    }
}

impl RTCDevice {
    pub fn new() -> RTCDevice {
        Default::default()
    }

    /// Returns the guest-visible state to the values that [`Self::new`] sets, keeping
    /// the metrics backend. Snapshots do not save RTC state, so this matches snapshot
    /// load: the counter follows current host time without the guest's previous offset.
    pub fn reset_to_fresh(&mut self) {
        self.0 = Rtc::with_events(*self.0.events());
    }
}

impl std::ops::Deref for RTCDevice {
    type Target = vm_superio::Rtc<&'static RTCDeviceMetrics>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for RTCDevice {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// Implements Bus functions for AMBA PL031 RTC device
impl RTCDevice {
    pub fn bus_read(&mut self, offset: u64, data: &mut [u8]) {
        if let (Ok(offset), 4) = (u16::try_from(offset), data.len()) {
            // read() function from RTC implementation expects a slice of
            // len 4, and we just validated that this is the data length
            self.read(offset, data.try_into().unwrap())
        } else {
            warn!(
                "Found invalid data offset/length while trying to read from the RTC: {}, {}",
                offset,
                data.len()
            );
            METRICS.error_count.inc();
        }
    }

    pub fn bus_write(&mut self, offset: u64, data: &[u8]) {
        if let (Ok(offset), 4) = (u16::try_from(offset), data.len()) {
            // write() function from RTC implementation expects a slice of
            // len 4, and we just validated that this is the data length
            self.write(offset, data.try_into().unwrap())
        } else {
            warn!(
                "Found invalid data offset/length while trying to write to the RTC: {}, {}",
                offset,
                data.len()
            );
            METRICS.error_count.inc();
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl crate::vstate::bus::BusDevice for RTCDevice {
    fn read(&mut self, _base: u64, offset: u64, data: &mut [u8]) {
        self.bus_read(offset, data)
    }

    fn write(
        &mut self,
        _base: u64,
        offset: u64,
        data: &[u8],
    ) -> Option<std::sync::Arc<std::sync::Barrier>> {
        self.bus_write(offset, data);
        None
    }
}

#[cfg(test)]
mod tests {
    use vm_superio::Rtc;

    use super::*;
    use crate::logger::IncMetric;

    #[test]
    fn test_rtc_device() {
        static TEST_RTC_DEVICE_METRICS: RTCDeviceMetrics = RTCDeviceMetrics::new();
        let mut rtc_pl031 = RTCDevice(Rtc::with_events(&TEST_RTC_DEVICE_METRICS));
        let data = [0; 4];

        // Write to the DR register. Since this is a RO register, the write
        // function should fail.
        let invalid_writes_before = TEST_RTC_DEVICE_METRICS.missed_write_count.count();
        let error_count_before = TEST_RTC_DEVICE_METRICS.error_count.count();
        rtc_pl031.bus_write(0x000, &data);
        let invalid_writes_after = TEST_RTC_DEVICE_METRICS.missed_write_count.count();
        let error_count_after = TEST_RTC_DEVICE_METRICS.error_count.count();
        assert_eq!(invalid_writes_after - invalid_writes_before, 1);
        assert_eq!(error_count_after - error_count_before, 1);
    }

    #[test]
    fn test_reset_to_fresh() {
        use vm_superio::rtc_pl031::RtcState;

        let mut live = RTCDevice(Rtc::from_state(
            &RtcState {
                lr: 123,
                offset: -456,
                mr: 789,
                imsc: 1,
                ris: 1,
            },
            &METRICS,
        ));
        let mut fresh = RTCDevice::new();

        live.reset_to_fresh();

        // Compare the counter offset rather than RTCDR, which advances with host time.
        assert_eq!(live.state(), fresh.state());
        for offset in [0x004, 0x008, 0x00c, 0x010, 0x014, 0x018] {
            let mut live_value = [0; 4];
            let mut fresh_value = [0; 4];
            live.bus_read(offset, &mut live_value);
            fresh.bus_read(offset, &mut fresh_value);
            assert_eq!(live_value, fresh_value, "RTC register {offset:#x}");
        }
    }

    #[test]
    fn test_reset_to_fresh_preserves_metrics() {
        static TEST_RESET_METRICS: RTCDeviceMetrics = RTCDeviceMetrics::new();
        let mut rtc = RTCDevice(Rtc::with_events(&TEST_RESET_METRICS));
        rtc.bus_write(0x000, &[0; 4]);
        rtc.bus_read(0x01c, &mut [0; 4]);
        assert_eq!(TEST_RESET_METRICS.error_count.count(), 2);

        rtc.reset_to_fresh();

        assert!(std::ptr::eq(*rtc.events(), &TEST_RESET_METRICS));
        assert_eq!(TEST_RESET_METRICS.error_count.count(), 2);
        rtc.bus_write(0x000, &[0; 4]);
        rtc.bus_read(0x01c, &mut [0; 4]);
        assert_eq!(TEST_RESET_METRICS.missed_write_count.count(), 2);
        assert_eq!(TEST_RESET_METRICS.missed_read_count.count(), 2);
        assert_eq!(TEST_RESET_METRICS.error_count.count(), 4);
    }

    #[test]
    fn test_rtc_invalid_buf_len() {
        static TEST_RTC_INVALID_BUF_LEN_METRICS: RTCDeviceMetrics = RTCDeviceMetrics::new();
        let mut rtc_pl031 = RTCDevice(Rtc::with_events(&TEST_RTC_INVALID_BUF_LEN_METRICS));
        let write_data_good = 123u32.to_le_bytes();
        let mut data_bad = [0; 2];
        let mut read_data_good = [0; 4];

        rtc_pl031.bus_write(0x008, &write_data_good);
        rtc_pl031.bus_write(0x008, &data_bad);
        rtc_pl031.bus_read(0x008, &mut read_data_good);
        rtc_pl031.bus_read(0x008, &mut data_bad);
        assert_eq!(u32::from_le_bytes(read_data_good), 123);
        assert_eq!(u16::from_le_bytes(data_bad), 0);
    }
}
