// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use kvm_bindings::*;
use kvm_ioctls::DeviceFd;

use crate::arch::aarch64::gic::GicError;
use crate::arch::aarch64::gic::regs::{GicRegState, SimpleReg, VgicRegEngine};

// Relevant PPI redistributor registers that we want to save/restore.
const GICR_CTLR: SimpleReg = SimpleReg::new(0x0000, 4);
const GICR_STATUSR: SimpleReg = SimpleReg::new(0x0010, 4);
const GICR_WAKER: SimpleReg = SimpleReg::new(0x0014, 4);
const GICR_PROPBASER: SimpleReg = SimpleReg::new(0x0070, 8);
const GICR_PENDBASER: SimpleReg = SimpleReg::new(0x0078, 8);

// Relevant SGI redistributor registers that we want to save/restore.
const GICR_SGI_OFFSET: u64 = 0x0001_0000;
const GICR_IGROUPR0: SimpleReg = SimpleReg::new(GICR_SGI_OFFSET + 0x0080, 4);
const GICR_ISENABLER0: SimpleReg = SimpleReg::new(GICR_SGI_OFFSET + 0x0100, 4);
const GICR_ICENABLER0: SimpleReg = SimpleReg::new(GICR_SGI_OFFSET + 0x0180, 4);
const GICR_ISPENDR0: SimpleReg = SimpleReg::new(GICR_SGI_OFFSET + 0x0200, 4);
const GICR_ICPENDR0: SimpleReg = SimpleReg::new(GICR_SGI_OFFSET + 0x0280, 4);
const GICR_ISACTIVER0: SimpleReg = SimpleReg::new(GICR_SGI_OFFSET + 0x0300, 4);
const GICR_ICACTIVER0: SimpleReg = SimpleReg::new(GICR_SGI_OFFSET + 0x0380, 4);
const GICR_IPRIORITYR0: SimpleReg = SimpleReg::new(GICR_SGI_OFFSET + 0x0400, 32);
const GICR_ICFGR0: SimpleReg = SimpleReg::new(GICR_SGI_OFFSET + 0x0C00, 8);

// List with relevant redistributor registers that we will be restoring.
static VGIC_RDIST_REGS: &[SimpleReg] = &[
    GICR_STATUSR,
    GICR_WAKER,
    GICR_PROPBASER,
    GICR_PENDBASER,
    GICR_CTLR,
];

// List with relevant SGI associated redistributor registers that we will be restoring.
static VGIC_SGI_REGS: &[SimpleReg] = &[
    GICR_IGROUPR0,
    GICR_ICENABLER0,
    GICR_ISENABLER0,
    GICR_ICFGR0,
    GICR_ICPENDR0,
    GICR_ISPENDR0,
    GICR_ICACTIVER0,
    GICR_ISACTIVER0,
    GICR_IPRIORITYR0,
];

struct RedistRegEngine {}

impl VgicRegEngine for RedistRegEngine {
    type Reg = SimpleReg;
    type RegChunk = u32;

    fn group() -> u32 {
        KVM_DEV_ARM_VGIC_GRP_REDIST_REGS
    }

    #[allow(clippy::cast_sign_loss)] // bit mask
    fn mpidr_mask() -> u64 {
        KVM_DEV_ARM_VGIC_V3_MPIDR_MASK as u64
    }
}

fn redist_regs() -> Box<dyn Iterator<Item = &'static SimpleReg>> {
    Box::new(VGIC_RDIST_REGS.iter().chain(VGIC_SGI_REGS))
}

pub(crate) fn get_redist_regs(
    fd: &DeviceFd,
    mpidr: u64,
) -> Result<Vec<GicRegState<u32>>, GicError> {
    RedistRegEngine::get_regs_data(fd, redist_regs(), mpidr)
}

pub(crate) fn set_redist_regs(
    fd: &DeviceFd,
    mpidr: u64,
    data: &[GicRegState<u32>],
) -> Result<(), GicError> {
    RedistRegEngine::set_regs_data(fd, redist_regs(), data, mpidr)
}

pub(crate) fn disable_lpis(fd: &DeviceFd, mpidr: u64) -> Result<(), GicError> {
    // vgic_mmio_write_v3r_ctlr() calls vgic_flush_pending_lpis() and
    // vgic_its_invalidate_all_caches() when EnableLPIs is cleared.
    RedistRegEngine::set_reg_value(fd, &GICR_CTLR, 0, mpidr)
}

pub(crate) fn set_redist_regs_in_place(
    fd: &DeviceFd,
    mpidr: u64,
    data: &[GicRegState<u32>],
) -> Result<(), GicError> {
    // vgic_uaccess_write_senable() and vgic_mmio_uaccess_write_sactive() only set 1 bits.
    // Clear live bits before applying the saved clear-before-set sequence.
    RedistRegEngine::set_reg_value(fd, &GICR_ICENABLER0, u32::MAX, mpidr)?;
    RedistRegEngine::set_reg_value(fd, &GICR_ICACTIVER0, u32::MAX, mpidr)?;
    // GicState omits line levels. vgic_write_irq_line_level_info() clears private PPI
    // inputs in block 0; SGI line levels are write-ignored.
    RedistRegEngine::set_line_level(fd, 0, 0, mpidr)?;
    // vgic_v3_uaccess_write_pending() replaces the pending word; ICPENDR is write-ignored.
    // The caller disables LPIs on every redistributor before ITS reset.
    // vgic_mmio_write_propbase() and vgic_mmio_write_pendbase() ignore writes while
    // LPIs are enabled, so keep the existing base-before-CTL restore order.
    set_redist_regs(fd, mpidr, data)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]
    use std::os::unix::io::AsRawFd;

    use kvm_ioctls::Kvm;

    use super::*;
    use crate::arch::aarch64::gic::{GICVersion, create_gic};

    #[test]
    fn test_access_redist_regs() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let _ = vm.create_vcpu(0).unwrap();
        let gic_fd = create_gic(&vm, 1, Some(GICVersion::GICV3)).expect("Cannot create gic");

        let gicr_typer = 123;
        let res = get_redist_regs(gic_fd.device_fd(), gicr_typer);
        let state = res.unwrap();
        assert_eq!(state.len(), 14);

        set_redist_regs(gic_fd.device_fd(), gicr_typer, &state).unwrap();

        unsafe { libc::close(gic_fd.device_fd().as_raw_fd()) };

        let res = set_redist_regs(gic_fd.device_fd(), gicr_typer, &state);
        assert_eq!(
            format!("{:?}", res.unwrap_err()),
            "DeviceAttribute(Error(9), true, 5)"
        );

        let res = get_redist_regs(gic_fd.device_fd(), gicr_typer);
        assert_eq!(
            format!("{:?}", res.unwrap_err()),
            "DeviceAttribute(Error(9), false, 5)"
        );

        // dropping gic_fd would double close the gic fd, so leak it
        std::mem::forget(gic_fd);
    }

    #[test]
    fn test_set_redist_regs_in_place_replaces_private_bitmaps() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let _vcpu = vm.create_vcpu(0).unwrap();
        let gic = create_gic(&vm, 1, Some(GICVersion::GICV3)).unwrap();
        let fd = gic.device_fd();
        let mpidr = 0;
        let bitmaps = [
            (&GICR_ISENABLER0, 0x0001_0001),
            (&GICR_ISPENDR0, 0x0002_0002),
            (&GICR_ISACTIVER0, 0x0004_0004),
        ];
        let live_only = 0x0008_0008;

        RedistRegEngine::set_reg_value(fd, &GICR_ICENABLER0, u32::MAX, mpidr).unwrap();
        RedistRegEngine::set_reg_value(fd, &GICR_ICACTIVER0, u32::MAX, mpidr).unwrap();
        RedistRegEngine::set_reg_value(fd, &GICR_ICFGR0, 0, mpidr).unwrap();
        for &(reg, saved_word) in &bitmaps {
            RedistRegEngine::set_reg_value(fd, reg, saved_word, mpidr).unwrap();
            assert_eq!(
                RedistRegEngine::get_reg_data(fd, reg, mpidr)
                    .unwrap()
                    .chunks,
                vec![saved_word]
            );
        }
        let saved = get_redist_regs(fd, mpidr).unwrap();

        RedistRegEngine::set_reg_value(fd, &GICR_ICENABLER0, u32::MAX, mpidr).unwrap();
        RedistRegEngine::set_reg_value(fd, &GICR_ICACTIVER0, u32::MAX, mpidr).unwrap();
        for &(reg, _) in &bitmaps {
            RedistRegEngine::set_reg_value(fd, reg, live_only, mpidr).unwrap();
            assert_eq!(
                RedistRegEngine::get_reg_data(fd, reg, mpidr)
                    .unwrap()
                    .chunks,
                vec![live_only]
            );
        }
        RedistRegEngine::set_line_level(fd, 0, live_only, mpidr).unwrap();
        let mut level = 0;
        unsafe {
            fd.get_device_attr(&mut RedistRegEngine::line_level_attr(0, &mut level, mpidr))
                .unwrap();
        }
        assert_eq!(level, live_only & 0xffff_0000);

        set_redist_regs_in_place(fd, mpidr, &saved).unwrap();

        for &(reg, saved_word) in &bitmaps {
            assert_eq!(
                RedistRegEngine::get_reg_data(fd, reg, mpidr)
                    .unwrap()
                    .chunks,
                vec![saved_word]
            );
        }
        unsafe {
            fd.get_device_attr(&mut RedistRegEngine::line_level_attr(0, &mut level, mpidr))
                .unwrap();
        }
        assert_eq!(level, 0);
    }

    #[test]
    fn test_set_redist_regs_in_place_restores_lpi_bases() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let _vcpu = vm.create_vcpu(0).unwrap();
        let gic = create_gic(&vm, 1, Some(GICVersion::GICV3)).unwrap();
        let fd = gic.device_fd();
        let mpidr = 0;
        let base_regs = [&GICR_PROPBASER, &GICR_PENDBASER];

        RedistRegEngine::set_reg_value(fd, &GICR_CTLR, 0, mpidr).unwrap();
        for (reg, chunks) in [
            (&GICR_PROPBASER, vec![0x0010_000f, 1]),
            (&GICR_PENDBASER, vec![0x0020_0000, 2]),
        ] {
            RedistRegEngine::set_reg_data(fd, reg, &GicRegState { chunks }, mpidr).unwrap();
        }
        let saved_bases =
            base_regs.map(|reg| RedistRegEngine::get_reg_data(fd, reg, mpidr).unwrap());
        RedistRegEngine::set_reg_value(fd, &GICR_CTLR, 1, mpidr).unwrap();
        let saved_ctlr = RedistRegEngine::get_reg_data(fd, &GICR_CTLR, mpidr).unwrap();
        assert_eq!(saved_ctlr.chunks[0] & 1, 1);
        let saved = get_redist_regs(fd, mpidr).unwrap();

        RedistRegEngine::set_reg_value(fd, &GICR_CTLR, 0, mpidr).unwrap();
        for (reg, base) in base_regs.iter().zip(&saved_bases) {
            let mut live_base = GicRegState {
                chunks: base.chunks.clone(),
            };
            live_base.chunks[0] ^= 0x0040_0000;
            live_base.chunks[1] ^= 4;
            RedistRegEngine::set_reg_data(fd, reg, &live_base, mpidr).unwrap();
            assert_eq!(
                RedistRegEngine::get_reg_data(fd, reg, mpidr)
                    .unwrap()
                    .chunks,
                live_base.chunks
            );
        }
        RedistRegEngine::set_reg_value(fd, &GICR_CTLR, 1, mpidr).unwrap();
        assert_eq!(
            RedistRegEngine::get_reg_data(fd, &GICR_CTLR, mpidr)
                .unwrap()
                .chunks,
            saved_ctlr.chunks
        );
        for (reg, base) in base_regs.iter().zip(&saved_bases) {
            let live_base = RedistRegEngine::get_reg_data(fd, reg, mpidr).unwrap();
            assert_ne!(live_base.chunks, base.chunks);
            RedistRegEngine::set_reg_data(fd, reg, base, mpidr).unwrap();
            assert_eq!(
                RedistRegEngine::get_reg_data(fd, reg, mpidr)
                    .unwrap()
                    .chunks,
                live_base.chunks
            );
        }

        disable_lpis(fd, mpidr).unwrap();
        assert_eq!(
            RedistRegEngine::get_reg_data(fd, &GICR_CTLR, mpidr)
                .unwrap()
                .chunks[0]
                & 1,
            0
        );
        set_redist_regs_in_place(fd, mpidr, &saved).unwrap();

        for (reg, base) in base_regs.iter().zip(&saved_bases) {
            assert_eq!(
                RedistRegEngine::get_reg_data(fd, reg, mpidr)
                    .unwrap()
                    .chunks,
                base.chunks
            );
        }
        assert_eq!(
            RedistRegEngine::get_reg_data(fd, &GICR_CTLR, mpidr)
                .unwrap()
                .chunks,
            saved_ctlr.chunks
        );
    }
}
