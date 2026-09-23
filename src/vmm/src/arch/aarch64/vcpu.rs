// Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

use std::fmt::{Debug, Write};
use std::mem::offset_of;
use std::sync::Arc;

use kvm_bindings::*;
use kvm_ioctls::{VcpuExit, VcpuFd, VmFd};
use serde::{Deserialize, Serialize};
use vm_memory::GuestAddress;

use super::get_fdt_addr;
use super::regs::*;
use crate::arch::EntryPoint;
use crate::arch::aarch64::kvm::OptionalCapabilities;
use crate::arch::aarch64::regs::{Aarch64RegisterVec, KVM_REG_ARM64_SVE_VLS};
use crate::cpu_config::aarch64::custom_cpu_template::VcpuFeatures;
use crate::cpu_config::templates::CpuConfiguration;
use crate::logger::{IncMetric, METRICS, error};
use crate::vcpu::{VcpuConfig, VcpuError};
use crate::vstate::bus::Bus;
use crate::vstate::memory::{Address, GuestMemoryMmap};
use crate::vstate::vcpu::VcpuEmulation;
use crate::vstate::vm::KvmVm;

/// Errors thrown while setting aarch64 registers.
#[derive(Debug, PartialEq, Eq, thiserror::Error, displaydoc::Display)]
pub enum VcpuArchError {
    /// Failed to get register {0}: {1}
    GetOneReg(u64, kvm_ioctls::Error),
    /// Failed to set register {0:#x} to value {1}: {2}
    SetOneReg(u64, String, kvm_ioctls::Error),
    /// Failed to retrieve list of registers: {0}
    GetRegList(kvm_ioctls::Error),
    /// Failed to get multiprocessor state: {0}
    GetMp(kvm_ioctls::Error),
    /// Failed to set multiprocessor state: {0}
    SetMp(kvm_ioctls::Error),
    /// Failed FamStructWrapper operation: {0}
    Fam(vmm_sys_util::fam::Error),
    /// Failed to set/get device attributes for vCPU: {0}
    DeviceAttribute(kvm_ioctls::Error),
}

/// Extract the Manufacturer ID from the host.
/// The ID is found between bits 24-31 of MIDR_EL1 register.
pub fn get_manufacturer_id_from_host() -> Option<u32> {
    let midr_el1_path = "/sys/devices/system/cpu/cpu0/regs/identification/midr_el1";
    let midr_el1 = std::fs::read_to_string(midr_el1_path).ok()?;
    let midr_el1_trimmed = midr_el1.trim_end().trim_start_matches("0x");
    let manufacturer_id = u32::from_str_radix(midr_el1_trimmed, 16).ok()?;
    Some(manufacturer_id >> 24)
}

/// Saves states of registers into `state`.
///
/// # Arguments
///
/// * `ids` - Slice of registers ids to save.
/// * `regs` - Input/Output vector of registers.
pub fn get_registers(
    vcpu_fd: &VcpuFd,
    ids: &[u64],
    regs: &mut Aarch64RegisterVec,
) -> Result<(), VcpuArchError> {
    let mut big_reg = [0_u8; 256];
    for id in ids.iter() {
        let reg_size = vcpu_fd
            .get_one_reg(*id, &mut big_reg)
            .map_err(|e| VcpuArchError::GetOneReg(*id, e))?;
        let reg_ref = Aarch64RegisterRef::new(*id, &big_reg[0..reg_size]);
        regs.push(reg_ref);
    }
    Ok(())
}

/// Errors associated with the wrappers over KVM ioctls.
#[derive(Debug, PartialEq, Eq, thiserror::Error, displaydoc::Display)]
pub enum KvmVcpuError {
    /// Error configuring the vcpu registers: {0}
    ConfigureRegisters(VcpuArchError),
    /// Error creating vcpu: {0}
    CreateVcpu(kvm_ioctls::Error),
    /// Failed to dump CPU configuration: {0}
    DumpCpuConfig(VcpuArchError),
    /// Error getting the vcpu preferred target: {0}
    GetPreferredTarget(kvm_ioctls::Error),
    /// Error initializing the vcpu: {0}
    Init(kvm_ioctls::Error),
    /// Error applying template: {0}
    ApplyCpuTemplate(VcpuArchError),
    /// Cannot restore vcpu in place with a different target or feature set.
    IncompatibleVcpuInit,
    /// Cannot restore vcpu in place with a different pvtime IPA (current {0:?}, saved {1:?}).
    IncompatiblePvtimeIpa(Option<u64>, Option<u64>),
    /// Failed to restore the state of the vcpu: {0}
    RestoreState(VcpuArchError),
    /// Failed to save the state of the vcpu: {0}
    SaveState(VcpuArchError),
    /// Found unsupported KVM_ARM_VCPU_PMU_V3 bit set in vcpu features.
    UnsupportedPmuV3,
}

/// Error type for [`KvmVcpu::configure`].
pub type KvmVcpuConfigureError = KvmVcpuError;

/// A wrapper around creating and using a kvm aarch64 vcpu.
#[derive(Debug)]
pub struct KvmVcpu {
    /// Index of vcpu.
    pub index: u8,
    /// KVM vcpu fd.
    pub fd: VcpuFd,
    /// Vcpu peripherals, such as buses
    pub peripherals: Peripherals,
    kvi: kvm_vcpu_init,
    /// IPA of steal_time region
    pub pvtime_ipa: Option<GuestAddress>,
}

/// Vcpu peripherals
#[derive(Default, Debug)]
pub struct Peripherals {
    /// mmio bus.
    pub mmio_bus: Option<Arc<Bus>>,
}

impl KvmVcpu {
    /// Constructs a new kvm vcpu with arch specific functionality.
    ///
    /// # Arguments
    ///
    /// * `index` - Represents the 0-based CPU index between [0, max vcpus).
    /// * `vm` - The vm to which this vcpu will get attached.
    pub fn new(index: u8, vm: &KvmVm) -> Result<Self, KvmVcpuError> {
        let kvm_vcpu = vm
            .fd()
            .create_vcpu(index.into())
            .map_err(KvmVcpuError::CreateVcpu)?;

        let mut kvi = Self::default_kvi(vm.fd())?;
        // Secondary vcpus must be powered off for boot process.
        if 0 < index {
            kvi.features[0] |= 1 << KVM_ARM_VCPU_POWER_OFF;
        }

        Ok(KvmVcpu {
            index,
            fd: kvm_vcpu,
            peripherals: Default::default(),
            kvi,
            pvtime_ipa: None,
        })
    }

    /// Read the MPIDR - Multiprocessor Affinity Register.
    pub fn get_mpidr(&self) -> Result<u64, VcpuArchError> {
        // MPIDR register is 64 bit wide on aarch64
        let mut mpidr = [0_u8; 8];
        match self.fd.get_one_reg(MPIDR_EL1, &mut mpidr) {
            Err(err) => Err(VcpuArchError::GetOneReg(MPIDR_EL1, err)),
            Ok(_) => Ok(u64::from_le_bytes(mpidr)),
        }
    }

    /// Configures an aarch64 specific vcpu for booting Linux.
    ///
    /// # Arguments
    ///
    /// * `guest_mem` - The guest memory used by this microvm.
    /// * `kernel_entry_point` - Specifies the boot protocol and offset from `guest_mem` at which
    ///   the kernel starts.
    /// * `vcpu_config` - The vCPU configuration.
    pub fn configure(
        &mut self,
        guest_mem: &GuestMemoryMmap,
        kernel_entry_point: EntryPoint,
        vcpu_config: &VcpuConfig,
        optional_capabilities: &OptionalCapabilities,
    ) -> Result<(), KvmVcpuError> {
        for reg in vcpu_config.cpu_config.regs.iter() {
            self.fd.set_one_reg(reg.id, reg.as_slice()).map_err(|err| {
                KvmVcpuError::ApplyCpuTemplate(VcpuArchError::SetOneReg(
                    reg.id,
                    reg.value_str(),
                    err,
                ))
            })?;
        }

        self.setup_boot_regs(
            kernel_entry_point.entry_addr.raw_value(),
            guest_mem,
            optional_capabilities,
        )
        .map_err(KvmVcpuError::ConfigureRegisters)?;

        Ok(())
    }

    /// Initializes an aarch64 specific vcpu for booting Linux.
    ///
    /// # Arguments
    ///
    /// * `vm_fd` - The kvm `VmFd` for this microvm.
    pub fn init(&mut self, vcpu_features: &[VcpuFeatures]) -> Result<(), KvmVcpuError> {
        for feature in vcpu_features.iter() {
            let index = feature.index as usize;
            self.kvi.features[index] = feature.bitmap.apply(self.kvi.features[index]);
        }

        self.init_vcpu()?;
        self.finalize_vcpu()?;

        Ok(())
    }

    /// Creates default kvi struct based on vcpu index.
    pub fn default_kvi(vm_fd: &VmFd) -> Result<kvm_vcpu_init, KvmVcpuError> {
        let mut kvi = kvm_vcpu_init::default();
        // This reads back the kernel's preferred target type.
        vm_fd
            .get_preferred_target(&mut kvi)
            .map_err(KvmVcpuError::GetPreferredTarget)?;
        // We already checked that the capability is supported.
        kvi.features[0] |= 1 << KVM_ARM_VCPU_PSCI_0_2;

        Ok(kvi)
    }

    /// Save the KVM internal state.
    pub fn save_state(&self) -> Result<VcpuState, KvmVcpuError> {
        let mut state = VcpuState {
            mp_state: self.get_mpstate().map_err(KvmVcpuError::SaveState)?,
            ..Default::default()
        };
        self.get_all_registers(&mut state.regs)
            .map_err(KvmVcpuError::SaveState)?;
        state.mpidr = self.get_mpidr().map_err(KvmVcpuError::SaveState)?;

        state.kvi = self.kvi;
        // We don't save power off state in a snapshot, because
        // it was only needed during uVM boot process.
        // When uVM is restored, the kernel has already passed
        // the boot state and turned secondary vcpus on.
        state.kvi.features[0] &= !(1 << KVM_ARM_VCPU_POWER_OFF);

        state.pvtime_ipa = self.pvtime_ipa.map(|guest_addr| guest_addr.0);

        Ok(state)
    }

    /// Use provided state to populate KVM internal state.
    pub fn restore_state(&mut self, state: &VcpuState) -> Result<(), KvmVcpuError> {
        self.kvi = state.kvi;

        self.init_vcpu()?;

        // If KVM_REG_ARM64_SVE_VLS is present it needs to
        // be set before vcpu is finalized.
        if let Some(sve_vls_reg) = state
            .regs
            .iter()
            .find(|reg| reg.id == KVM_REG_ARM64_SVE_VLS)
        {
            self.set_register(sve_vls_reg)
                .map_err(KvmVcpuError::RestoreState)?;
        }

        self.finalize_vcpu()?;

        for reg in state.registers_after_finalize() {
            self.set_register(reg).map_err(KvmVcpuError::RestoreState)?;
        }
        self.set_mpstate(state.mp_state)
            .map_err(KvmVcpuError::RestoreState)?;

        // Assumes that steal time memory region was set up already
        if let Some(pvtime_ipa) = state.pvtime_ipa {
            self.enable_pvtime(GuestAddress(pvtime_ipa))
                .map_err(KvmVcpuError::RestoreState)?;
        }

        Ok(())
    }

    /// Restores an initialized vCPU without repeating one-shot setup.
    ///
    /// All vCPUs must remain paused until their state and the GIC have been restored.
    /// The caller must first complete pending userspace I/O without entering the guest.
    /// The VM must use per-register counter offsets, not KVM_ARM_SET_COUNTER_OFFSET.
    pub fn restore_state_in_place(&mut self, state: &VcpuState) -> Result<(), KvmVcpuError> {
        state.validate_in_place_restore(&self.kvi, self.pvtime_ipa)?;

        // kvm_vcpu_set_target() permits INIT to reset an initialized vCPU with unchanged features.
        // Reuse the original kvi, not save_state()'s POWER_OFF-stripped copy.
        // kvm_arch_vcpu_ioctl_vcpu_init() also handles stage-2/cache invalidation after KVM_RUN.
        self.init_vcpu()?;

        // kvm_reset_vcpu() preserves SVE finalization. kvm_arm_vcpu_finalize() and set_sve_vls()
        // reject attempts to repeat that setup. set_id_reg() and set_imp_id_reg() accept the
        // unchanged saved ID register values after KVM_RUN.
        // arch_timer_set_user() restores shared VM counter offsets, so keep the saved register
        // order and restore the GIC after kvm_timer_vcpu_reset() has cleared the timer IRQ state.
        for reg in state.registers_after_finalize() {
            self.set_register(reg).map_err(KvmVcpuError::RestoreState)?;
        }
        self.set_mpstate(state.mp_state)
            .map_err(KvmVcpuError::RestoreState)
    }

    /// Dumps CPU configuration.
    pub fn dump_cpu_config(&self) -> Result<CpuConfiguration, KvmVcpuError> {
        let mut regs = Aarch64RegisterVec::default();
        self.get_all_registers(&mut regs)
            .map_err(KvmVcpuError::DumpCpuConfig)?;
        Ok(CpuConfiguration { regs })
    }

    /// Initializes internal vcpufd.
    fn init_vcpu(&self) -> Result<(), KvmVcpuError> {
        // Setting KVM_ARM_VCPU_PMU_V3 without initialising the PMU causes KVM
        // to crash on KVM_RUN with EINVAL.
        //
        // To properly initialise the PMU, the KVM_SET_DEVICE_ATTR ioctl must
        // be made with the flag KVM_ARM_VCPU_PMU_V3_INIT set. Firecracker
        // currently does not handle this, so we should return an error instead.
        if (self.kvi.features[0] & (1 << KVM_ARM_VCPU_PMU_V3)) != 0 {
            return Err(KvmVcpuError::UnsupportedPmuV3);
        }

        self.fd.vcpu_init(&self.kvi).map_err(KvmVcpuError::Init)?;
        Ok(())
    }

    /// Checks for SVE feature and calls `vcpu_finalize` if
    /// it is enabled.
    fn finalize_vcpu(&self) -> Result<(), KvmVcpuError> {
        if (self.kvi.features[0] & (1 << KVM_ARM_VCPU_SVE)) != 0 {
            // KVM_ARM_VCPU_SVE has value 4 so casting to i32 is safe.
            #[allow(clippy::cast_possible_wrap)]
            let feature = KVM_ARM_VCPU_SVE as i32;
            self.fd.vcpu_finalize(&feature).unwrap();
        }
        Ok(())
    }

    /// Configure relevant boot registers for a given vCPU.
    ///
    /// # Arguments
    ///
    /// * `boot_ip` - Starting instruction pointer.
    /// * `mem` - Reserved DRAM for current VM.
    /// + `optional_capabilities` - which optional capabilities are enabled that might influence
    ///   vcpu configuration
    pub fn setup_boot_regs(
        &self,
        boot_ip: u64,
        mem: &GuestMemoryMmap,
        optional_capabilities: &OptionalCapabilities,
    ) -> Result<(), VcpuArchError> {
        let kreg_off = offset_of!(kvm_regs, regs);

        // Get the register index of the PSTATE (Processor State) register.
        let pstate = offset_of!(user_pt_regs, pstate) + kreg_off;
        let id = arm64_core_reg_id!(KVM_REG_SIZE_U64, pstate);
        self.fd
            .set_one_reg(id, &PSTATE_FAULT_BITS_64.to_le_bytes())
            .map_err(|err| {
                VcpuArchError::SetOneReg(id, format!("{PSTATE_FAULT_BITS_64:#x}"), err)
            })?;

        // Other vCPUs are powered off initially awaiting PSCI wakeup.
        if self.index == 0 {
            // Setting the PC (Processor Counter) to the current program address (kernel address).
            let pc = offset_of!(user_pt_regs, pc) + kreg_off;
            let id = arm64_core_reg_id!(KVM_REG_SIZE_U64, pc);
            self.fd
                .set_one_reg(id, &boot_ip.to_le_bytes())
                .map_err(|err| VcpuArchError::SetOneReg(id, format!("{boot_ip:#x}"), err))?;

            // Last mandatory thing to set -> the address pointing to the FDT (also called DTB).
            // "The device tree blob (dtb) must be placed on an 8-byte boundary and must
            // not exceed 2 megabytes in size." -> https://www.kernel.org/doc/Documentation/arm64/booting.txt.
            // We are choosing to place it the end of DRAM. See `get_fdt_addr`.
            let regs0 = offset_of!(user_pt_regs, regs) + kreg_off;
            let id = arm64_core_reg_id!(KVM_REG_SIZE_U64, regs0);
            let fdt_addr = get_fdt_addr(mem);
            self.fd
                .set_one_reg(id, &fdt_addr.to_le_bytes())
                .map_err(|err| VcpuArchError::SetOneReg(id, format!("{fdt_addr:#x}"), err))?;

            // Reset the physical counter for the guest. This way we avoid guest reading
            // host physical counter.
            // Resetting KVM_REG_ARM_PTIMER_CNT for single vcpu is enough because there is only
            // one timer struct with offsets per VM.
            // Because the access to KVM_REG_ARM_PTIMER_CNT is only present starting 6.4 kernel,
            // we only do the reset if KVM_CAP_COUNTER_OFFSET is present as it was added
            // in the same patch series as the ability to set the KVM_REG_ARM_PTIMER_CNT register.
            // Path series which introduced the needed changes:
            // https://lore.kernel.org/all/20230330174800.2677007-1-maz@kernel.org/
            // Note: the value observed by the guest will still be above 0, because there is a delta
            // time between this resetting and first call to KVM_RUN.
            if optional_capabilities.counter_offset {
                self.fd
                    .set_one_reg(KVM_REG_ARM_PTIMER_CNT, &[0; 8])
                    .map_err(|err| {
                        VcpuArchError::SetOneReg(id, format!("{KVM_REG_ARM_PTIMER_CNT:#x}"), err)
                    })?;
            }
        }
        Ok(())
    }

    /// Saves the states of the system registers into `state`.
    ///
    /// # Arguments
    ///
    /// * `regs` - Input/Output vector of registers.
    pub fn get_all_registers(&self, state: &mut Aarch64RegisterVec) -> Result<(), VcpuArchError> {
        get_registers(&self.fd, &self.get_all_registers_ids()?, state)
    }

    /// Returns all registers ids, including core and system
    pub fn get_all_registers_ids(&self) -> Result<Vec<u64>, VcpuArchError> {
        // Call KVM_GET_REG_LIST to get all registers available to the guest. For ArmV8 there are
        // less than 500 registers expected, resize to the reported size when necessary.
        let mut reg_list = RegList::new(500).map_err(VcpuArchError::Fam)?;

        match self.fd.get_reg_list(&mut reg_list) {
            Ok(_) => Ok(reg_list.as_slice().to_vec()),
            Err(e) => match e.errno() {
                libc::E2BIG => {
                    // resize and retry.
                    let size: usize = reg_list
                        .as_fam_struct_ref()
                        .n
                        .try_into()
                        // Safe to unwrap as Firecracker only targets 64-bit machines.
                        .unwrap();
                    reg_list = RegList::new(size).map_err(VcpuArchError::Fam)?;
                    self.fd
                        .get_reg_list(&mut reg_list)
                        .map_err(VcpuArchError::GetRegList)?;

                    Ok(reg_list.as_slice().to_vec())
                }
                _ => Err(VcpuArchError::GetRegList(e)),
            },
        }
    }

    /// Set the state of one system register.
    ///
    /// # Arguments
    ///
    /// * `reg` - Register to be set.
    pub fn set_register(&self, reg: Aarch64RegisterRef) -> Result<(), VcpuArchError> {
        self.fd
            .set_one_reg(reg.id, reg.as_slice())
            .map_err(|e| VcpuArchError::SetOneReg(reg.id, reg.value_str(), e))?;
        Ok(())
    }

    /// Get the multistate processor.
    ///
    /// # Arguments
    ///
    /// * `vcpu` - Structure for the VCPU that holds the VCPU's fd.
    pub fn get_mpstate(&self) -> Result<kvm_mp_state, VcpuArchError> {
        self.fd.get_mp_state().map_err(VcpuArchError::GetMp)
    }

    /// Set the state of the system registers.
    ///
    /// # Arguments
    ///
    /// * `state` - Structure for returning the state of the system registers.
    pub fn set_mpstate(&self, state: kvm_mp_state) -> Result<(), VcpuArchError> {
        self.fd.set_mp_state(state).map_err(VcpuArchError::SetMp)
    }

    /// Check if pvtime (steal time on ARM) is supported for vcpu
    pub fn supports_pvtime(&self) -> bool {
        let pvtime_device_attr = kvm_bindings::kvm_device_attr {
            group: kvm_bindings::KVM_ARM_VCPU_PVTIME_CTRL,
            attr: kvm_bindings::KVM_ARM_VCPU_PVTIME_IPA as u64,
            addr: 0,
            flags: 0,
        };

        // Use kvm_has_device_attr to check if PVTime is supported
        self.fd.has_device_attr(&pvtime_device_attr).is_ok()
    }

    /// Enables pvtime for vcpu
    pub fn enable_pvtime(&mut self, ipa: GuestAddress) -> Result<(), VcpuArchError> {
        self.pvtime_ipa = Some(ipa);

        // Use KVM syscall (kvm_set_device_attr) to register the vCPU with the steal_time region
        let vcpu_device_attr = kvm_bindings::kvm_device_attr {
            group: KVM_ARM_VCPU_PVTIME_CTRL,
            attr: KVM_ARM_VCPU_PVTIME_IPA as u64,
            addr: &ipa.0 as *const u64 as u64, // userspace address of attr data
            flags: 0,
        };

        self.fd
            .set_device_attr(&vcpu_device_attr)
            .map_err(VcpuArchError::DeviceAttribute)?;

        Ok(())
    }
}

impl Peripherals {
    /// Runs the vCPU in KVM context and handles the kvm exit reason.
    ///
    /// Returns error or enum specifying whether emulation was handled or interrupted.
    pub fn run_arch_emulation(&self, exit: VcpuExit) -> Result<VcpuEmulation, VcpuError> {
        METRICS.vcpu.failures.inc();
        error!("Unexpected exit reason on vcpu run: {:?}", exit);
        Err(VcpuError::UnhandledKvmExit(format!("{:?}", exit)))
    }
}

/// Structure holding VCPU kvm state.
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct VcpuState {
    /// Multiprocessing state.
    pub mp_state: kvm_mp_state,
    /// Vcpu registers.
    pub regs: Aarch64RegisterVec,
    /// We will be using the mpidr for passing it to the VmState.
    /// The VmState will give this away for saving restoring the icc and redistributor
    /// registers.
    pub mpidr: u64,
    /// kvi states for vcpu initialization.
    pub kvi: kvm_vcpu_init,
    /// ipa for steal_time region
    pub pvtime_ipa: Option<u64>,
}

impl VcpuState {
    fn validate_in_place_restore(
        &self,
        kvi: &kvm_vcpu_init,
        pvtime_ipa: Option<GuestAddress>,
    ) -> Result<(), KvmVcpuError> {
        // kvm_arch_vcpu_ioctl_vcpu_init() treats POWER_OFF as an ephemeral MP-state request.
        // save_state() strips it, but every other feature word and the target must match.
        if self.kvi.target != kvi.target
            || (self.kvi.features[0] ^ kvi.features[0]) & !(1 << KVM_ARM_VCPU_POWER_OFF) != 0
            || self.kvi.features[1..] != kvi.features[1..]
        {
            return Err(KvmVcpuError::IncompatibleVcpuInit);
        }

        // kvm_arm_pvtime_vcpu_init() runs at vCPU creation, not in kvm_reset_vcpu(). The IPA
        // survives INIT, and kvm_arm_pvtime_set_attr() rejects even an equal second write.
        // Neither enabling nor disabling pvtime is part of an in-place restore.
        let pvtime_ipa = pvtime_ipa.map(|ipa| ipa.0);
        if self.pvtime_ipa != pvtime_ipa {
            return Err(KvmVcpuError::IncompatiblePvtimeIpa(
                pvtime_ipa,
                self.pvtime_ipa,
            ));
        }
        Ok(())
    }

    fn registers_after_finalize(&self) -> impl Iterator<Item = Aarch64RegisterRef<'_>> {
        // set_sve_vls() rejects writes after SVE finalization, unlike the SVE data registers.
        self.regs
            .iter()
            .filter(|reg| reg.id != KVM_REG_ARM64_SVE_VLS)
    }
}

impl Debug for VcpuState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "kvm_mp_state: {:#x}", self.mp_state.mp_state)?;
        writeln!(f, "mpidr: {:#x}", self.mpidr)?;
        for reg in self.regs.iter() {
            writeln!(
                f,
                "{:#x} 0x{}",
                reg.id,
                reg.as_slice()
                    .iter()
                    .rev()
                    .fold(String::new(), |mut output, b| {
                        let _ = write!(output, "{b:x}");
                        output
                    })
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]
    use std::os::unix::io::AsRawFd;

    use kvm_bindings::{KVM_ARM_VCPU_PSCI_0_2, KVM_REG_SIZE_U64};
    use vm_memory::{Bytes, GuestAddress};

    use super::*;
    use crate::arch::BootProtocol;
    use crate::arch::aarch64::layout;
    use crate::arch::aarch64::regs::Aarch64RegisterRef;
    use crate::cpu_config::aarch64::CpuConfiguration;
    use crate::cpu_config::templates::RegisterValueFilter;
    use crate::test_utils::arch_mem;
    use crate::vcpu::VcpuConfig;
    use crate::vstate::vm::tests::setup_vm_with_memory;

    // KVM_REG_ARM_TIMER_CNT and KVM_REG_ARM_TIMER_CVAL have swapped architectural encodings.
    const KVM_REG_ARM_TIMER_CNT: u64 = SYS_CNTV_CVAL_EL0;
    const KVM_REG_ARM_TIMER_CVAL: u64 = 0x6030_0000_0013_df02;
    // KVM_REG_ARM64_SVE_ZREG(0, 0).
    const SVE_Z0: u64 = KVM_REG_ARM64 | KVM_REG_ARM64_SVE as u64 | KVM_REG_SIZE_U2048;
    const X0: u64 = arm64_core_reg_id!(
        KVM_REG_SIZE_U64,
        offset_of!(kvm_regs, regs) + offset_of!(user_pt_regs, regs)
    );
    const PSCI_SYSTEM_OFF: u64 = 0x8400_0008;

    fn setup_vcpu(mem_size: usize) -> (KvmVm, KvmVcpu) {
        let (mut vm, mut vcpu) = setup_vcpu_no_init(mem_size);
        vcpu.init(&[]).unwrap();
        vm.setup_irqchip(1).unwrap();
        (vm, vcpu)
    }

    fn setup_vcpu_no_init(mem_size: usize) -> (KvmVm, KvmVcpu) {
        let vm = setup_vm_with_memory(mem_size);
        let vcpu = KvmVcpu::new(0, &vm).unwrap();

        (vm, vcpu)
    }

    fn run_vcpu_to_shutdown(vm: &KvmVm, vcpu: &mut KvmVcpu) {
        // hvc #0 exits to userspace through PSCI SYSTEM_OFF without pending MMIO completion.
        vm.guest_memory()
            .write_slice(&0xd400_0002_u32.to_le_bytes(), GuestAddress(0))
            .unwrap();
        vcpu.fd.set_one_reg(PC, &0_u64.to_le_bytes()).unwrap();
        vcpu.fd
            .set_one_reg(X0, &PSCI_SYSTEM_OFF.to_le_bytes())
            .unwrap();
        assert!(matches!(
            vcpu.fd.run().unwrap(),
            VcpuExit::SystemEvent(KVM_SYSTEM_EVENT_SHUTDOWN, _)
        ));
    }

    #[test]
    fn test_in_place_restore_register_selection() {
        let regs = [
            Aarch64RegisterRef::new(KVM_REG_ARM_TIMER_CNT, &[0x11; 8]),
            Aarch64RegisterRef::new(KVM_REG_ARM64_SVE_VLS, &[0x22; 64]),
            Aarch64RegisterRef::new(PC, &[0x33; 8]),
            Aarch64RegisterRef::new(SVE_Z0, &[0x44; 256]),
            Aarch64RegisterRef::new(KVM_REG_ARM_PTIMER_CNT, &[0x55; 8]),
            Aarch64RegisterRef::new(KVM_REG_ARM_TIMER_CVAL, &[0x66; 8]),
        ];
        let mut state = VcpuState::default();
        for reg in regs {
            state.regs.push(reg);
        }

        let mut selected = state.registers_after_finalize();
        for expected in [regs[0], regs[2], regs[3], regs[4], regs[5]] {
            assert_eq!(selected.next(), Some(expected));
        }
        assert_eq!(selected.next(), None);
    }

    #[test]
    fn test_in_place_restore_init_compatibility() {
        let mut kvi = kvm_vcpu_init {
            target: KVM_ARM_TARGET_GENERIC_V8,
            ..Default::default()
        };
        kvi.features[0] = (1 << KVM_ARM_VCPU_PSCI_0_2) | (1 << KVM_ARM_VCPU_POWER_OFF);
        let mut state = VcpuState {
            kvi,
            ..Default::default()
        };
        state.kvi.features[0] &= !(1 << KVM_ARM_VCPU_POWER_OFF);
        assert_eq!(state.validate_in_place_restore(&kvi, None), Ok(()));
        assert_eq!(state.validate_in_place_restore(&state.kvi, None), Ok(()));

        state.kvi.target ^= 1;
        assert_eq!(
            state.validate_in_place_restore(&kvi, None),
            Err(KvmVcpuError::IncompatibleVcpuInit)
        );
        state.kvi.target = kvi.target;
        for index in 0..kvi.features.len() {
            state.kvi.features[index] ^= 1 << KVM_ARM_VCPU_PSCI_0_2;
            assert_eq!(
                state.validate_in_place_restore(&kvi, None),
                Err(KvmVcpuError::IncompatibleVcpuInit),
                "feature word {index}"
            );
            state.kvi.features[index] ^= 1 << KVM_ARM_VCPU_PSCI_0_2;
        }
    }

    #[test]
    fn test_in_place_restore_pvtime_compatibility() {
        let kvi = kvm_vcpu_init {
            target: KVM_ARM_TARGET_GENERIC_V8,
            ..Default::default()
        };
        let cases = [
            (None, None, Ok(())),
            (Some(0x1000), Some(0x1000), Ok(())),
            (
                Some(0x1000),
                Some(0x2000),
                Err(KvmVcpuError::IncompatiblePvtimeIpa(
                    Some(0x1000),
                    Some(0x2000),
                )),
            ),
            (
                Some(0x1000),
                None,
                Err(KvmVcpuError::IncompatiblePvtimeIpa(Some(0x1000), None)),
            ),
            (
                None,
                Some(0x1000),
                Err(KvmVcpuError::IncompatiblePvtimeIpa(None, Some(0x1000))),
            ),
        ];
        for (current_ipa, saved_ipa, expected) in cases {
            let state = VcpuState {
                kvi,
                pvtime_ipa: saved_ipa,
                ..Default::default()
            };
            assert_eq!(
                state.validate_in_place_restore(&kvi, current_ipa.map(GuestAddress)),
                expected
            );
        }
    }

    #[test]
    fn test_writable_imp_id_regs() {
        // MIDR_EL1: op0=3, op1=0, CRn=0, CRm=0, op2=0.
        const MIDR_EL1: u64 = 0x6030_0000_0013_c000;
        // An arbitrary valid MIDR value (implementer Arm, part Neoverse N1).
        const FAKE_MIDR: u64 = 0x410f_d0c0;

        // `KvmVm::new` enables KVM_CAP_ARM_WRITABLE_IMP_ID_REGS when the host
        // kernel offers it, before any vCPU is created.
        let vm = setup_vm_with_memory(0x1000);
        if vm
            .fd()
            .check_extension_raw(u64::from(kvm_bindings::KVM_CAP_ARM_WRITABLE_IMP_ID_REGS))
            != 1
        {
            // Host kernel predates writable implementation ID registers
            // (Linux 6.15); there is nothing further to verify.
            return;
        }
        let mut vcpu = KvmVcpu::new(0, &vm).unwrap();
        vcpu.init(&[]).unwrap();

        let mut val = [0u8; 8];
        vcpu.fd
            .set_one_reg(MIDR_EL1, &FAKE_MIDR.to_le_bytes())
            .unwrap();
        vcpu.fd.get_one_reg(MIDR_EL1, &mut val).unwrap();
        assert_eq!(u64::from_le_bytes(val), FAKE_MIDR);
    }

    #[test]
    fn test_create_vcpu() {
        let vm = setup_vm_with_memory(0x1000);

        unsafe { libc::close(vm.fd().as_raw_fd()) };

        let err = KvmVcpu::new(0, &vm);

        // dropping vm would double close the gic fd, so leak it
        // do the drop before assertion. Otherwise if assert fails,
        // we get IO runtime error instead of assert error.
        std::mem::forget(vm);

        assert_eq!(
            err.err().unwrap().to_string(),
            "Error creating vcpu: Bad file descriptor (os error 9)".to_string()
        );
    }

    #[test]
    fn test_configure_vcpu() {
        let (vm, mut vcpu) = setup_vcpu(0x10000);
        let optional_capabilities = vm.kvm().optional_capabilities();

        let vcpu_config = VcpuConfig {
            vcpu_count: 1,
            smt: false,
            cpu_config: CpuConfiguration::default(),
        };

        vcpu.configure(
            vm.guest_memory(),
            EntryPoint {
                entry_addr: GuestAddress(crate::arch::get_kernel_start()),
                protocol: BootProtocol::LinuxBoot,
            },
            &vcpu_config,
            &optional_capabilities,
        )
        .unwrap();

        unsafe { libc::close(vcpu.fd.as_raw_fd()) };

        let err = vcpu.configure(
            vm.guest_memory(),
            EntryPoint {
                entry_addr: GuestAddress(crate::arch::get_kernel_start()),
                protocol: BootProtocol::LinuxBoot,
            },
            &vcpu_config,
            &optional_capabilities,
        );

        // dropping vcpu would double close the gic fd, so leak it
        // do the drop before assertion. Otherwise if assert fails,
        // we get IO runtime error instead of assert error.
        std::mem::forget(vcpu);

        assert_eq!(
            err.unwrap_err(),
            KvmVcpuError::ConfigureRegisters(VcpuArchError::SetOneReg(
                0x6030000000100042,
                "0x3c5".to_string(),
                kvm_ioctls::Error::new(9)
            ))
        );
    }

    #[test]
    fn test_init_vcpu() {
        let mut vm = setup_vm_with_memory(0x1000);
        let mut vcpu = KvmVcpu::new(0, &vm).unwrap();
        vm.setup_irqchip(1).unwrap();

        // KVM_ARM_VCPU_PSCI_0_2 is set by default.
        // we check if we can remove it.
        let vcpu_features = vec![VcpuFeatures {
            index: 0,
            bitmap: RegisterValueFilter {
                filter: 1 << KVM_ARM_VCPU_PSCI_0_2,
                value: 0,
            },
        }];
        vcpu.init(&vcpu_features).unwrap();
        assert!((vcpu.kvi.features[0] & (1 << KVM_ARM_VCPU_PSCI_0_2)) == 0)
    }

    #[test]
    fn test_pmu_v3_feature_invalid() {
        let mut vm = setup_vm_with_memory(0x1000);
        let mut vcpu = KvmVcpu::new(0, &vm).unwrap();
        vm.setup_irqchip(1).unwrap();

        // Firecracker does not support KVM_ARM_VCPU_PMU_V3. Check that
        // attempting to enable this feature returns an error.
        let vcpu_features = vec![VcpuFeatures {
            index: 0,
            bitmap: RegisterValueFilter {
                filter: 1 << KVM_ARM_VCPU_PMU_V3,
                value: 1 << KVM_ARM_VCPU_PMU_V3,
            },
        }];

        let res = vcpu.init(&vcpu_features);
        assert!(matches!(res.unwrap_err(), KvmVcpuError::UnsupportedPmuV3));
    }

    #[test]
    fn test_vcpu_save_restore_state() {
        let mut vm = setup_vm_with_memory(0x1000);
        let mut vcpu = KvmVcpu::new(0, &vm).unwrap();
        vm.setup_irqchip(1).unwrap();

        // Calling KVM_GET_REGLIST before KVM_VCPU_INIT will result in error.
        let res = vcpu.save_state();
        assert!(matches!(
            res.unwrap_err(),
            KvmVcpuError::SaveState(VcpuArchError::GetRegList(_))
        ));

        // Try to restore the register using a faulty state.
        let mut faulty_vcpu_state = VcpuState::default();

        // Try faulty kvi state
        let res = vcpu.restore_state(&faulty_vcpu_state);
        assert!(matches!(res.unwrap_err(), KvmVcpuError::Init(_)));

        // Try faulty vcpu regs
        faulty_vcpu_state.kvi = KvmVcpu::default_kvi(vm.fd()).unwrap();
        let mut regs = Aarch64RegisterVec::default();
        let mut reg = Aarch64RegisterRef::new(KVM_REG_SIZE_U64, &[0; 8]);
        reg.id = 0;
        regs.push(reg);
        faulty_vcpu_state.regs = regs;
        let res = vcpu.restore_state(&faulty_vcpu_state);
        assert!(matches!(
            res.unwrap_err(),
            KvmVcpuError::RestoreState(VcpuArchError::SetOneReg(0, _, _))
        ));

        vcpu.init(&[]).unwrap();
        let state = vcpu.save_state().expect("Cannot save state of vcpu");
        assert!(!state.regs.is_empty());
        vcpu.restore_state(&state)
            .expect("Cannot restore state of vcpu");
    }

    #[test]
    fn test_vcpu_restore_state_in_place_rejects_incompatible_setup() {
        let (_vm, mut vcpu) = setup_vcpu(0x1000);
        vcpu.fd.set_one_reg(PC, &0x1234_u64.to_le_bytes()).unwrap();
        let mut state = vcpu.save_state().unwrap();
        state.kvi.target ^= 1;
        assert_eq!(
            vcpu.restore_state_in_place(&state),
            Err(KvmVcpuError::IncompatibleVcpuInit)
        );
        let mut pc = [0; 8];
        vcpu.fd.get_one_reg(PC, &mut pc).unwrap();
        assert_eq!(u64::from_le_bytes(pc), 0x1234);

        state.kvi = vcpu.kvi;
        state.pvtime_ipa = Some(0x100);
        assert_eq!(
            vcpu.restore_state_in_place(&state),
            Err(KvmVcpuError::IncompatiblePvtimeIpa(None, Some(0x100)))
        );
        vcpu.fd.get_one_reg(PC, &mut pc).unwrap();
        assert_eq!(u64::from_le_bytes(pc), 0x1234);
    }

    #[test]
    fn test_vcpu_restore_state_in_place_after_run() {
        let (vm, mut vcpu) = setup_vcpu(0x1000);
        if vm
            .fd()
            .check_extension_raw(KVM_CAP_ARM_WRITABLE_IMP_ID_REGS.into())
            == 1
        {
            vcpu.fd
                .set_one_reg(MIDR_EL1, &0x410f_d0c0_u64.to_le_bytes())
                .unwrap();
        }
        run_vcpu_to_shutdown(&vm, &mut vcpu);
        vcpu.fd
            .set_one_reg(KVM_REG_ARM_TIMER_CVAL, &0x1234_5678_u64.to_le_bytes())
            .unwrap();
        let mut state = vcpu.save_state().unwrap();
        state.mp_state.mp_state = KVM_MP_STATE_RUNNABLE;
        state
            .regs
            .iter_mut()
            .find(|reg| reg.id == PC)
            .unwrap()
            .set_value(0_u64);
        state
            .regs
            .iter_mut()
            .find(|reg| reg.id == X0)
            .unwrap()
            .set_value(PSCI_SYSTEM_OFF);

        // A large synthetic counter advance makes the rewind check independent of host timing.
        const COUNTER_ADVANCE: u64 = 1 << 40;
        for _ in 0..2 {
            vcpu.fd.set_one_reg(PC, &0x80_u64.to_le_bytes()).unwrap();
            vcpu.fd
                .set_one_reg(KVM_REG_ARM_TIMER_CVAL, &0x8765_4321_u64.to_le_bytes())
                .unwrap();
            for reg in state
                .regs
                .iter()
                .filter(|reg| matches!(reg.id, KVM_REG_ARM_TIMER_CNT | KVM_REG_ARM_PTIMER_CNT))
            {
                let advanced = reg.value::<u64, 8>().wrapping_add(COUNTER_ADVANCE);
                vcpu.fd
                    .set_one_reg(reg.id, &advanced.to_le_bytes())
                    .unwrap();
            }

            vcpu.restore_state_in_place(&state).unwrap();
            assert_eq!(vcpu.get_mpstate().unwrap(), state.mp_state);
            let mut actual = [0; 256];
            for reg in state.registers_after_finalize() {
                let len = vcpu.fd.get_one_reg(reg.id, &mut actual).unwrap();
                if matches!(reg.id, KVM_REG_ARM_TIMER_CNT | KVM_REG_ARM_PTIMER_CNT) {
                    let now = u64::from_le_bytes(actual[..8].try_into().unwrap());
                    assert!(
                        now.wrapping_sub(reg.value::<u64, 8>()) < COUNTER_ADVANCE / 2,
                        "counter {:#x} did not rewind",
                        reg.id
                    );
                } else {
                    assert_eq!(&actual[..len], reg.as_slice(), "register {:#x}", reg.id);
                }
            }
            assert!(matches!(
                vcpu.fd.run().unwrap(),
                VcpuExit::SystemEvent(KVM_SYSTEM_EVENT_SHUTDOWN, _)
            ));
        }
    }

    #[test]
    fn test_vcpu_restore_state_in_place_preserves_secondary_init() {
        let mut vm = setup_vm_with_memory(0x1000);
        let mut boot_vcpu = KvmVcpu::new(0, &vm).unwrap();
        let mut vcpu = KvmVcpu::new(1, &vm).unwrap();
        boot_vcpu.init(&[]).unwrap();
        vcpu.init(&[]).unwrap();
        vm.setup_irqchip(2).unwrap();
        let mut state = vcpu.save_state().unwrap();
        state.mp_state.mp_state = KVM_MP_STATE_RUNNABLE;

        for _ in 0..2 {
            vcpu.restore_state_in_place(&state).unwrap();
            assert_ne!(vcpu.kvi.features[0] & (1 << KVM_ARM_VCPU_POWER_OFF), 0);
            assert_eq!(vcpu.get_mpstate().unwrap(), state.mp_state);
        }
    }

    #[test]
    fn test_vcpu_restore_state_in_place_preserves_pvtime() {
        let (vm, mut vcpu) = setup_vcpu(0x1000);
        if !vcpu.supports_pvtime() {
            return;
        }
        let ipa = GuestAddress(0x100);
        vcpu.enable_pvtime(ipa).unwrap();
        run_vcpu_to_shutdown(&vm, &mut vcpu);
        let state = vcpu.save_state().unwrap();
        for _ in 0..2 {
            vcpu.restore_state_in_place(&state).unwrap();
        }

        let attr = kvm_device_attr {
            group: KVM_ARM_VCPU_PVTIME_CTRL,
            attr: KVM_ARM_VCPU_PVTIME_IPA as u64,
            addr: &ipa.0 as *const u64 as u64,
            flags: 0,
        };
        assert_eq!(
            vcpu.fd.set_device_attr(&attr),
            Err(kvm_ioctls::Error::new(libc::EEXIST))
        );
    }

    #[test]
    fn test_vcpu_restore_state_in_place_preserves_sve() {
        let (mut vm, mut vcpu) = setup_vcpu_no_init(0x1000);
        if vm.fd().check_extension_raw(KVM_CAP_ARM_SVE.into()) == 0 {
            return;
        }
        vcpu.init(&[VcpuFeatures {
            index: 0,
            bitmap: RegisterValueFilter {
                filter: 1 << KVM_ARM_VCPU_SVE,
                value: 1 << KVM_ARM_VCPU_SVE,
            },
        }])
        .unwrap();
        vm.setup_irqchip(1).unwrap();
        run_vcpu_to_shutdown(&vm, &mut vcpu);
        let mut state = vcpu.save_state().unwrap();
        let mut expected = [0; 256];
        expected[..16].fill(0x5a);
        state
            .regs
            .iter_mut()
            .find(|reg| reg.id == SVE_Z0)
            .unwrap()
            .set_value(expected);

        for _ in 0..2 {
            vcpu.restore_state_in_place(&state).unwrap();
            let mut actual = [0; 256];
            vcpu.fd.get_one_reg(SVE_Z0, &mut actual).unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_dump_cpu_config_before_init() {
        // Test `dump_cpu_config()` before `KVM_VCPU_INIT`.
        //
        // This should fail with ENOEXEC.
        // https://elixir.bootlin.com/linux/v5.10.176/source/arch/arm64/kvm/arm.c#L1165
        let mut vm = setup_vm_with_memory(0x1000);
        let vcpu = KvmVcpu::new(0, &vm).unwrap();
        vm.setup_irqchip(1).unwrap();

        vcpu.dump_cpu_config().unwrap_err();
    }

    #[test]
    fn test_dump_cpu_config_after_init() {
        // Test `dump_cpu_config()` after `KVM_VCPU_INIT`.
        let mut vm = setup_vm_with_memory(0x1000);
        let mut vcpu = KvmVcpu::new(0, &vm).unwrap();
        vm.setup_irqchip(1).unwrap();
        vcpu.init(&[]).unwrap();

        vcpu.dump_cpu_config().unwrap();
    }

    #[test]
    fn test_setup_non_boot_vcpu() {
        let vm = setup_vm_with_memory(0x1000);
        let mut vcpu1 = KvmVcpu::new(0, &vm).unwrap();
        vcpu1.init(&[]).unwrap();
        let mut vcpu2 = KvmVcpu::new(1, &vm).unwrap();
        vcpu2.init(&[]).unwrap();
    }

    #[test]
    fn test_get_valid_regs() {
        // Test `get_regs()` with valid register IDs.
        // - X0: 0x6030 0000 0010 0000
        // - X1: 0x6030 0000 0010 0002
        let (_, vcpu) = setup_vcpu(0x10000);
        let reg_list = Vec::<u64>::from([0x6030000000100000, 0x6030000000100002]);
        get_registers(&vcpu.fd, &reg_list, &mut Aarch64RegisterVec::default()).unwrap();
    }

    #[test]
    fn test_get_invalid_regs() {
        // Test `get_regs()` with invalid register IDs.
        let (_, vcpu) = setup_vcpu(0x10000);
        let reg_list = Vec::<u64>::from([0x6030000000100001, 0x6030000000100003]);
        get_registers(&vcpu.fd, &reg_list, &mut Aarch64RegisterVec::default()).unwrap_err();
    }

    #[test]
    fn test_setup_regs() {
        let (vm, vcpu) = setup_vcpu_no_init(0x10000);
        let mem = arch_mem(layout::FDT_MAX_SIZE + 0x1000);
        let optional_capabilities = vm.kvm().optional_capabilities();

        let res = vcpu.setup_boot_regs(0x0, &mem, &optional_capabilities);
        assert!(matches!(
            res.unwrap_err(),
            VcpuArchError::SetOneReg(0x6030000000100042, _, _)
        ));

        vcpu.init_vcpu().unwrap();

        vcpu.setup_boot_regs(0x0, &mem, &optional_capabilities)
            .unwrap();

        // Check that the register is reset on compatible kernels.
        // Because there is a delta in time between we reset the register and time we
        // read it, we cannot compare with 0. Instead we compare it with meaningfully
        // small value.
        if optional_capabilities.counter_offset {
            let mut reg_bytes = [0_u8; 8];
            vcpu.fd.get_one_reg(SYS_CNTPCT_EL0, &mut reg_bytes).unwrap();
            let counter_value = u64::from_le_bytes(reg_bytes);

            // We are reading the SYS_CNTPCT_EL0 right after resetting it.
            // If reset did happen successfully, the value should be quite small when we read it.
            // If the reset did not happen, the value will be same as on the host and it surely
            // will be more that `max_value`. Measurements show that usually value is close
            // to 1000. Use bigger `max_value` just in case.
            let max_value = 10_000;

            assert!(counter_value < max_value);
        }
    }

    #[test]
    fn test_read_mpidr() {
        let (_, vcpu) = setup_vcpu_no_init(0x10000);

        // Must fail when vcpu is not initialized yet.
        let res = vcpu.get_mpidr();
        assert!(matches!(
            res.unwrap_err(),
            VcpuArchError::GetOneReg(MPIDR_EL1, _)
        ));
        vcpu.init_vcpu().unwrap();

        assert_eq!(vcpu.get_mpidr().unwrap(), 0x8000_0000);
    }

    #[test]
    fn test_get_set_regs() {
        let (_, vcpu) = setup_vcpu_no_init(0x10000);

        // Must fail when vcpu is not initialized yet.
        let mut regs = Aarch64RegisterVec::default();
        let res = vcpu.get_all_registers(&mut regs);
        assert!(matches!(res.unwrap_err(), VcpuArchError::GetRegList(_)));
        vcpu.init_vcpu().unwrap();

        vcpu.get_all_registers(&mut regs).unwrap();
        for reg in regs.iter() {
            vcpu.set_register(reg).unwrap();
        }
    }

    #[test]
    fn test_mpstate() {
        use std::os::unix::io::AsRawFd;

        let (_, vcpu) = setup_vcpu(0x10000);

        let res = vcpu.get_mpstate();
        vcpu.set_mpstate(res.unwrap()).unwrap();

        unsafe { libc::close(vcpu.fd.as_raw_fd()) };

        let res = vcpu.get_mpstate();
        assert!(matches!(res, Err(VcpuArchError::GetMp(_))), "{:?}", res);

        let res = vcpu.set_mpstate(kvm_mp_state::default());

        // dropping vcpu would double close the fd, so leak it
        // do the drop before assertion. Otherwise if assert fails,
        // we get IO runtime error instead of assert error.
        std::mem::forget(vcpu);

        assert!(matches!(res, Err(VcpuArchError::SetMp(_))), "{:?}", res);
    }
}
