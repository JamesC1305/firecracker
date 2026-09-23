// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

mod dist_regs;
mod icc_regs;

use kvm_ioctls::{DeviceFd, VmFd};

use crate::arch::aarch64::gic::GicError;
use crate::arch::aarch64::gic::regs::{GicState, GicVcpuState};

/// Save the state of the GIC device.
pub fn save_state(fd: &DeviceFd, mpidrs: &[u64]) -> Result<GicState, GicError> {
    let mut vcpu_states = Vec::with_capacity(mpidrs.len());
    for mpidr in mpidrs {
        vcpu_states.push(GicVcpuState {
            rdist: Vec::new(),
            icc: icc_regs::get_icc_regs(fd, *mpidr)?,
        })
    }

    Ok(GicState {
        dist: dist_regs::get_dist_regs(fd)?,
        gic_vcpu_states: vcpu_states,
        ..Default::default()
    })
}

/// Restore the state of the GIC device.
pub fn restore_state(fd: &DeviceFd, mpidrs: &[u64], state: &GicState) -> Result<(), GicError> {
    dist_regs::set_dist_regs(fd, &state.dist)?;

    if mpidrs.len() != state.gic_vcpu_states.len() {
        return Err(GicError::InconsistentVcpuCount);
    }
    for (mpidr, vcpu_state) in mpidrs.iter().zip(&state.gic_vcpu_states) {
        icc_regs::set_icc_regs(fd, *mpidr, &vcpu_state.icc)?;
    }

    Ok(())
}

/// Restore a stopped GICv2 to the state produced by a fresh snapshot load.
///
/// Interrupt producers must be quiesced and kernel-owned IRQs reset by their owners first.
pub fn restore_state_in_place(
    fd: &DeviceFd,
    vm_fd: &VmFd,
    mpidrs: &[u64],
    state: &GicState,
) -> Result<(), GicError> {
    if mpidrs.len() != state.gic_vcpu_states.len() {
        return Err(GicError::InconsistentVcpuCount);
    }

    dist_regs::set_dist_regs_in_place(fd, vm_fd, mpidrs, &state.dist)?;
    // Packed MPIDRs can leave CPU-interface banks unselected by ordinary GICv2 replay.
    for cpuid in 0..mpidrs.len() {
        icc_regs::reset_to_fresh(fd, cpuid as u64)?;
    }
    for (mpidr, vcpu_state) in mpidrs.iter().zip(&state.gic_vcpu_states) {
        icc_regs::set_icc_regs(fd, *mpidr, &vcpu_state.icc)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]

    use kvm_bindings::{
        KVM_DEV_ARM_VGIC_GRP_CPU_REGS, KVM_DEV_ARM_VGIC_GRP_DIST_REGS, kvm_device_attr,
    };
    use kvm_ioctls::Kvm;

    use super::*;
    use crate::arch::aarch64::gic::{GICVersion, create_gic};

    pub(super) fn get_reg(fd: &DeviceFd, group: u32, cpuid: u64, offset: u64) -> u32 {
        let mut value = 0u32;
        let mut attr = kvm_device_attr {
            group,
            attr: (cpuid << 32) | offset,
            addr: &mut value as *mut u32 as u64,
            flags: 0,
        };
        unsafe { fd.get_device_attr(&mut attr).unwrap() };
        value
    }

    pub(super) fn set_reg(fd: &DeviceFd, group: u32, cpuid: u64, offset: u64, value: u32) {
        fd.set_device_attr(&kvm_device_attr {
            group,
            attr: (cpuid << 32) | offset,
            addr: &value as *const u32 as u64,
            flags: 0,
        })
        .unwrap();
    }

    #[test]
    fn test_vm_save_restore_state() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let gic_fd = match create_gic(&vm, 1, Some(GICVersion::GICV2)) {
            Ok(gic_fd) => gic_fd,
            Err(GicError::CreateGIC(_)) => return,
            _ => panic!("Failed to open setup GICv2"),
        };

        let mpidr = vec![0];
        let res = save_state(gic_fd.device_fd(), &mpidr);
        // We will receive an error if trying to call before creating vcpu.
        assert_eq!(
            format!("{:?}", res.unwrap_err()),
            "DeviceAttribute(Error(22), false, 2)"
        );

        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let _vcpu = vm.create_vcpu(0).unwrap();
        let gic = create_gic(&vm, 1, Some(GICVersion::GICV2)).expect("Cannot create gic");
        let gic_fd = gic.device_fd();

        let vm_state = save_state(gic_fd, &mpidr).unwrap();
        restore_state(gic_fd, &mpidr, &vm_state).unwrap();
    }

    fn check_cpu_interface_restore(mpidrs: [u64; 2], expected_pmr: [u32; 2]) {
        let kvm = Kvm::new().unwrap();
        let live_vm = kvm.create_vm().unwrap();
        let _live_vcpus = [
            live_vm.create_vcpu(0).unwrap(),
            live_vm.create_vcpu(1).unwrap(),
        ];
        let live_gic = match create_gic(&live_vm, 2, Some(GICVersion::GICV2)) {
            Ok(gic) => gic,
            Err(GicError::CreateGIC(_)) => return,
            err => panic!("Failed to set up GICv2: {err:?}"),
        };
        let fresh_vm = kvm.create_vm().unwrap();
        let _fresh_vcpus = [
            fresh_vm.create_vcpu(0).unwrap(),
            fresh_vm.create_vcpu(1).unwrap(),
        ];
        let fresh_gic = create_gic(&fresh_vm, 2, Some(GICVersion::GICV2)).unwrap();
        let live_fd = live_gic.device_fd();
        let fresh_fd = fresh_gic.device_fd();
        let group = KVM_DEV_ARM_VGIC_GRP_CPU_REGS;

        for cpuid in 0_u32..2 {
            for (offset, value) in [
                (0x00, 1 + cpuid),
                (0x04, 16 + cpuid),
                (0x08, 4 + cpuid),
                (0x1c, 5 + cpuid),
                (0xd0, 0x1111_1111 << cpuid),
                (0xd4, 0x1111_1111 << cpuid),
                (0xd8, 0x1111_1111 << cpuid),
                (0xdc, 0x1111_1111 << cpuid),
            ] {
                set_reg(live_fd, group, u64::from(cpuid), offset, value);
            }
        }
        let baseline = save_state(live_fd, &mpidrs).unwrap();

        for cpuid in 0_u32..2 {
            for (offset, value) in [
                (0x00, 3),
                (0x04, 2 + cpuid),
                (0x08, 7),
                (0x1c, 7),
                (0xd0, 0x4444_4444 << cpuid),
                (0xd4, 0x4444_4444 << cpuid),
                (0xd8, 0x4444_4444 << cpuid),
                (0xdc, 0x4444_4444 << cpuid),
            ] {
                set_reg(live_fd, group, u64::from(cpuid), offset, value);
            }
            assert_eq!(get_reg(live_fd, group, u64::from(cpuid), 0x04), 2 + cpuid);
        }

        restore_state(fresh_fd, &mpidrs, &baseline).unwrap();
        restore_state_in_place(live_fd, &live_vm, &mpidrs, &baseline).unwrap();

        for cpuid in 0..2 {
            for offset in [0x00, 0x04, 0x08, 0x1c, 0xd0, 0xd4, 0xd8, 0xdc] {
                assert_eq!(
                    get_reg(live_fd, group, cpuid, offset),
                    get_reg(fresh_fd, group, cpuid, offset),
                    "CPU {cpuid} interface register {offset:#x}"
                );
            }
            assert_eq!(
                get_reg(fresh_fd, group, cpuid, 0x04),
                expected_pmr[usize::try_from(cpuid).unwrap()]
            );
        }
    }

    #[test]
    fn test_restore_state_in_place_matches_fresh_cpu_interfaces() {
        check_cpu_interface_restore([0, 1], [16, 17]);
    }

    #[test]
    fn test_restore_state_in_place_defaults_omitted_cpu_interfaces() {
        check_cpu_interface_restore([0, 1 << 32], [16, 0]);
    }

    #[test]
    fn test_restore_state_in_place_rejects_vcpu_count_before_writes() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let _vcpus = [vm.create_vcpu(0).unwrap(), vm.create_vcpu(1).unwrap()];
        let gic = match create_gic(&vm, 2, Some(GICVersion::GICV2)) {
            Ok(gic) => gic,
            Err(GicError::CreateGIC(_)) => return,
            err => panic!("Failed to set up GICv2: {err:?}"),
        };
        let fd = gic.device_fd();
        let mpidrs = [0, 1];
        let mut baseline = save_state(fd, &mpidrs).unwrap();
        baseline.gic_vcpu_states.truncate(1);
        let group = KVM_DEV_ARM_VGIC_GRP_DIST_REGS;
        set_reg(fd, group, 0, 0x0104, 0xa5a5_a5a5);
        set_reg(fd, group, 1, 0x0400, 0x8080_8080);
        set_reg(fd, KVM_DEV_ARM_VGIC_GRP_CPU_REGS, 1, 0x04, 7);

        assert_eq!(
            restore_state_in_place(fd, &vm, &mpidrs, &baseline),
            Err(GicError::InconsistentVcpuCount)
        );
        assert_eq!(get_reg(fd, group, 0, 0x0104), 0xa5a5_a5a5);
        assert_eq!(get_reg(fd, group, 1, 0x0400), 0x8080_8080);
        assert_eq!(get_reg(fd, KVM_DEV_ARM_VGIC_GRP_CPU_REGS, 1, 0x04), 7);
    }
}
