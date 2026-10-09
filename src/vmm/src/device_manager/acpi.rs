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
    vmgenid: Option<VmGenId>,
    /// VMclock device
    vmclock: Option<VmClock>,
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

    pub fn vmgenid_mut(&mut self) -> &mut VmGenId {
        self.vmgenid.as_mut().expect("Missing VMGenID device")
    }

    pub fn vmclock(&self) -> &VmClock {
        self.vmclock.as_ref().expect("Missing VMClock device")
    }

    pub fn vmclock_mut(&mut self) -> &mut VmClock {
        self.vmclock.as_mut().expect("Missing VMClock device")
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
mod tests {
    use super::*;
    use crate::arch;
    use crate::snapshot::Persist;
    use crate::utils::u64_to_usize;
    use crate::vstate::resources::ResourceAllocator;
    use crate::vstate::vm::tests::setup_vm_with_memory;

    #[test]
    fn test_load_publishes_generation_state() {
        let vm = setup_vm_with_memory(
            u64_to_usize(arch::SYSTEM_MEM_START) + u64_to_usize(arch::SYSTEM_MEM_SIZE),
        );
        #[cfg(target_arch = "aarch64")]
        let mut vm = vm;
        #[cfg(target_arch = "x86_64")]
        vm.setup_irqchip().unwrap();
        #[cfg(target_arch = "aarch64")]
        vm.setup_irqchip(1).unwrap();

        let mut resource_allocator = ResourceAllocator::new();
        let mut source = ACPIDeviceManager::new(
            VmGenId::new(&mut resource_allocator).unwrap(),
            VmClock::new(&mut resource_allocator).unwrap(),
        );
        // Set up a source clock at generation seven before saving the manager.
        let mut clock_state = source.vmclock().save();
        clock_state.inner.seq_count = 12;
        clock_state.inner.disruption_marker = 6;
        clock_state.inner.vm_generation_counter = 6;
        source
            .vmclock_mut()
            .apply_state(&clock_state, vm.guest_memory())
            .unwrap();
        let state = source.save();
        let acpi =
            crate::snapshot::restore_for_test::<ACPIDeviceManager>(&vm, &state, vm.guest_memory())
                .unwrap();
        let clock = acpi.vmclock().save().inner;
        assert_eq!(clock.seq_count, 16);
        assert_eq!(clock.disruption_marker, 8);
        assert_eq!(clock.vm_generation_counter, 8);
    }
}
