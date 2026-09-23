// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

mod dist_regs;
mod icc_regs;
pub mod its_regs;
mod redist_regs;

use its_regs::{ItsRegisterState, its_reset, its_save_tables};
use kvm_ioctls::DeviceFd;

use crate::arch::aarch64::gic::GicError;
use crate::arch::aarch64::gic::regs::{GicState, GicVcpuState};

/// Save the state of the GIC device.
pub fn save_state(
    gic_device: &DeviceFd,
    its_device: &DeviceFd,
    mpidrs: &[u64],
) -> Result<GicState, GicError> {
    // vgic_v3_save_pending_tables and vgic_its_save_tables_v0 write kernel-owned state
    // into guest RAM. Capture the memory snapshot after these calls; calling them after
    // memory reversion would overwrite the checkpoint with the live interrupt state.
    super::save_pending_tables(gic_device)?;
    its_save_tables(its_device)?;

    let mut vcpu_states = Vec::with_capacity(mpidrs.len());
    for mpidr in mpidrs {
        vcpu_states.push(GicVcpuState {
            rdist: redist_regs::get_redist_regs(gic_device, *mpidr)?,
            icc: icc_regs::get_icc_regs(gic_device, *mpidr)?,
        })
    }

    let its_state = ItsRegisterState::save(its_device)?;

    Ok(GicState {
        dist: dist_regs::get_dist_regs(gic_device)?,
        gic_vcpu_states: vcpu_states,
        its_state: Some(its_state),
    })
}

/// Restore the state of the GIC device.
pub fn restore_state(
    gic_device: &DeviceFd,
    its_device: &DeviceFd,
    mpidrs: &[u64],
    state: &GicState,
) -> Result<(), GicError> {
    dist_regs::set_dist_regs(gic_device, &state.dist)?;

    if mpidrs.len() != state.gic_vcpu_states.len() {
        return Err(GicError::InconsistentVcpuCount);
    }
    for (mpidr, vcpu_state) in mpidrs.iter().zip(&state.gic_vcpu_states) {
        redist_regs::set_redist_regs(gic_device, *mpidr, &vcpu_state.rdist)?;
        icc_regs::set_icc_regs(gic_device, *mpidr, &vcpu_state.icc)?;
    }

    state
        .its_state
        .as_ref()
        .ok_or(GicError::MissingItsState)?
        .restore(its_device)
}

/// Restore an existing GIC after guest memory and vCPU state have been reverted.
pub fn restore_state_in_place(
    gic_device: &DeviceFd,
    its_device: &DeviceFd,
    mpidrs: &[u64],
    state: &GicState,
) -> Result<(), GicError> {
    if mpidrs.len() != state.gic_vcpu_states.len() {
        return Err(GicError::InconsistentVcpuCount);
    }
    let its_state = state.its_state.as_ref().ok_or(GicError::MissingItsState)?;

    // vgic_mmio_write_v3r_ctlr drops AP-list LPI references and invalidates ITS caches.
    // Disable every redistributor before vgic_its_reset drops the remaining ITE references.
    // Otherwise vgic_add_lpi can reuse a live LPI with stale target or pending state.
    for mpidr in mpidrs {
        redist_regs::disable_lpis(gic_device, *mpidr)?;
    }
    its_reset(its_device)?;

    dist_regs::set_dist_regs_in_place(gic_device, &state.dist)?;
    for (mpidr, vcpu_state) in mpidrs.iter().zip(&state.gic_vcpu_states) {
        redist_regs::set_redist_regs_in_place(gic_device, *mpidr, &vcpu_state.rdist)?;
        icc_regs::set_icc_regs(gic_device, *mpidr, &vcpu_state.icc)?;
    }

    // The redistributor bases and saved EnableLPIs bits must precede ITS table restore.
    // vgic_add_lpi imports each recreated LPI through vgic_v3_lpi_sync_pending_status,
    // which consumes its pending-table bit. Enabling LPIs afterward would reread that
    // cleared bit in vgic_enable_lpis and discard the restored pending interrupt.
    its_state.restore(its_device)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]

    use kvm_ioctls::Kvm;

    use super::*;
    use crate::arch::aarch64::gic::{GICVersion, create_gic};

    #[test]
    fn test_vm_save_restore_state() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let gic = create_gic(&vm, 1, Some(GICVersion::GICV3)).expect("Cannot create gic");
        let gic_fd = gic.device_fd();
        let its_fd = gic.its_fd().unwrap();

        let mpidr = vec![1];
        let res = save_state(gic_fd, its_fd, &mpidr);
        // We will receive an error if trying to call before creating vcpu.
        assert_eq!(
            format!("{:?}", res.unwrap_err()),
            "DeviceAttribute(Error(22), false, 5)"
        );

        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let _vcpu = vm.create_vcpu(0).unwrap();
        let gic = create_gic(&vm, 1, Some(GICVersion::GICV3)).expect("Cannot create gic");
        let gic_fd = gic.device_fd();
        let its_fd = gic.its_fd().unwrap();

        let vm_state = save_state(gic_fd, its_fd, &mpidr).unwrap();
        let val: u32 = 0;
        let gicd_statusr_off = 0x0010u64;
        let mut gic_dist_attr = kvm_bindings::kvm_device_attr {
            group: kvm_bindings::KVM_DEV_ARM_VGIC_GRP_DIST_REGS,
            attr: gicd_statusr_off,
            addr: &val as *const u32 as u64,
            flags: 0,
        };
        unsafe {
            gic_fd.get_device_attr(&mut gic_dist_attr).unwrap();
        }

        // The second value from the list of distributor registers is the value of the GICD_STATUSR
        // register. We assert that the one saved in the bitmap is the same with the one we
        // obtain with KVM_GET_DEVICE_ATTR.
        let gicd_statusr = &vm_state.dist[1];

        assert_eq!(gicd_statusr.chunks[0], val);
        assert_eq!(vm_state.dist.len(), 12);
        restore_state(gic_fd, its_fd, &mpidr, &vm_state).unwrap();
        restore_state(gic_fd, its_fd, &[1, 2], &vm_state).unwrap_err();
    }
}
