// Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(target_arch = "x86_64")]
use acpi_tables::{Aml, aml};

use crate::devices::acpi::vmclock::{VmClock, VmClockError};
use crate::devices::acpi::vmgenid::{VmGenId, VmGenIdError};
use crate::vstate::vm::KvmVm;

#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum ACPIDeviceError {
    /// VMGenID: {0}
    VmGenId(#[from] VmGenIdError),
    /// VMClock: {0}
    VmClock(#[from] VmClockError),
    /// Could not register IRQ with KVM: {0}
    RegisterIrq(#[from] kvm_ioctls::Error),
    /// Resource allocator error: {0}
    ResourceAllocator(#[from] vm_allocator::Error),
}

// Although both VMGenID and VMClock devices are always present, they should be instantiated when
// they are attached to preserve the existing ordering of GSI allocation.
#[derive(Debug, Default)]
pub struct ACPIDeviceManager {
    /// VMGenID device
    pub(super) vmgenid: Option<VmGenId>,
    /// VMclock device
    pub(super) vmclock: Option<VmClock>,
}

impl ACPIDeviceManager {
    /// Create a new ACPIDeviceManager object
    pub fn new(vmgenid: VmGenId, vmclock: VmClock) -> Self {
        ACPIDeviceManager {
            vmgenid: Some(vmgenid),
            vmclock: Some(vmclock),
        }
    }

    pub fn attach_vmgenid(&mut self, vm: &KvmVm) -> Result<(), ACPIDeviceError> {
        self.vmgenid = Some(VmGenId::new(&mut vm.resource_allocator())?);
        Ok(())
    }

    pub fn attach_vmclock(&mut self, vm: &KvmVm) -> Result<(), ACPIDeviceError> {
        self.vmclock = Some(VmClock::new(&mut vm.resource_allocator())?);
        Ok(())
    }

    pub fn vmgenid(&self) -> &VmGenId {
        self.vmgenid.as_ref().expect("Missing VMGenID device")
    }

    pub fn vmclock(&self) -> &VmClock {
        self.vmclock.as_ref().expect("Missing VMClock device")
    }

    pub fn activate_vmgenid(&self, vm: &KvmVm) -> Result<(), ACPIDeviceError> {
        vm.register_irq(&self.vmgenid().interrupt_evt, self.vmgenid().gsi)?;
        self.vmgenid().activate(vm.guest_memory())?;
        Ok(())
    }

    pub fn activate_vmclock(&self, vm: &KvmVm) -> Result<(), ACPIDeviceError> {
        vm.register_irq(&self.vmclock().interrupt_evt, self.vmclock().gsi)?;
        self.vmclock().activate(vm.guest_memory())?;
        Ok(())
    }
}

#[cfg(target_arch = "x86_64")]
impl Aml for ACPIDeviceManager {
    fn append_aml_bytes(&self, v: &mut Vec<u8>) -> Result<(), aml::AmlError> {
        // AML for [`VmGenId`] device.
        self.vmgenid().append_aml_bytes(v)?;
        // AML for [`VmClock`] device.
        self.vmclock().append_aml_bytes(v)?;

        // Create the AML for the GED interrupt handler
        aml::Device::new(
            "_SB_.GED_".try_into()?,
            vec![
                &aml::Name::new("_HID".try_into()?, &"ACPI0013")?,
                &aml::Name::new(
                    "_CRS".try_into()?,
                    &aml::ResourceTemplate::new(vec![
                        &aml::Interrupt::new(true, true, false, false, self.vmgenid().gsi),
                        &aml::Interrupt::new(true, true, false, false, self.vmclock().gsi),
                    ]),
                )?,
                // We know that the maximum IRQ number fits in a u8. We have up to
                // 32 IRQs in x86 and up to 128 in ARM (look into
                // `vmm::crate::arch::layout::GSI_LEGACY_END`). Both `vmgenid.gsi`
                // and `vmclock.gsi` can safely be cast to `u8` without truncation,
                // so we let clippy know.
                &aml::Method::new(
                    "_EVT".try_into()?,
                    1,
                    true,
                    vec![
                        &aml::If::new(
                            #[allow(clippy::cast_possible_truncation)]
                            &aml::Equal::new(&aml::Arg(0), &(self.vmgenid().gsi as u8)),
                            vec![&aml::Notify::new(
                                &aml::Path::new("\\_SB_.VGEN")?,
                                &0x80usize,
                            )],
                        ),
                        &aml::If::new(
                            #[allow(clippy::cast_possible_truncation)]
                            &aml::Equal::new(&aml::Arg(0), &(self.vmclock().gsi as u8)),
                            vec![&aml::Notify::new(
                                &aml::Path::new("\\_SB_.VCLK")?,
                                &0x80usize,
                            )],
                        ),
                    ],
                ),
            ],
        )
        .append_aml_bytes(v)
    }
}

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use std::os::fd::AsRawFd;

    use vm_memory::{ByteValued, Bytes};

    use super::*;
    use crate::snapshot::Persist;
    use crate::utils::mib_to_bytes;
    use crate::vstate::resources::ResourceAllocator;
    use crate::vstate::vm::tests::setup_vm_with_memory;

    #[test]
    fn test_reset_republishes_generation_state() {
        let mut resource_allocator = ResourceAllocator::new();
        let mut state = ACPIDeviceManager::new(
            VmGenId::new(&mut resource_allocator).unwrap(),
            VmClock::new(&mut resource_allocator).unwrap(),
        )
        .save();
        state.vmclock.inner.seq_count = 14;
        state.vmclock.inner.disruption_marker = 7;
        state.vmclock.inner.vm_generation_counter = 7;

        let vm = setup_vm_with_memory(mib_to_bytes(1));
        vm.setup_irqchip().unwrap();
        let mut acpi = ACPIDeviceManager::restore(&vm, &state).unwrap();
        let genid_fd = acpi.vmgenid().interrupt_evt.as_raw_fd();
        let clock_fd = acpi.vmclock().interrupt_evt.as_raw_fd();
        let mem = vm.guest_memory();
        let clock = acpi.vmclock().save().inner;
        let clock_len = clock.as_slice().len();
        assert_eq!(clock.seq_count, 16);
        assert_eq!(clock.disruption_marker, 8);
        assert_eq!(clock.vm_generation_counter, 8);

        for generation in 9..=10 {
            let old_genid = acpi.vmgenid().gen_id;
            // Memory reversion can leave old or empty contents in both device pages.
            mem.write_slice(&[0; 16], acpi.vmgenid().guest_address)
                .unwrap();
            mem.write_slice(&vec![0; clock_len], acpi.vmclock().guest_address)
                .unwrap();
            acpi.restore_in_place(&state, mem).unwrap();

            let genid: u128 = mem.read_obj(acpi.vmgenid().guest_address).unwrap();
            assert_ne!(genid, old_genid);
            assert_eq!(genid, acpi.vmgenid().gen_id);
            // The guest sees the whole VMClock page again, with one more generation.
            let clock = acpi.vmclock().save().inner;
            let mut page = vec![0; clock_len];
            mem.read_slice(&mut page, acpi.vmclock().guest_address)
                .unwrap();
            assert_eq!(page, clock.as_slice());
            assert_eq!(u64::from(clock.seq_count), generation * 2);
            assert_eq!(clock.disruption_marker, generation);
            assert_eq!(clock.vm_generation_counter, generation);
            assert_eq!(acpi.vmgenid().interrupt_evt.as_raw_fd(), genid_fd);
            assert_eq!(acpi.vmclock().interrupt_evt.as_raw_fd(), clock_fd);
        }
    }
}
