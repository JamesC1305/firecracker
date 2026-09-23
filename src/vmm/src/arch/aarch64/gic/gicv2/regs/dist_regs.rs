// Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;

use kvm_bindings::{
    KVM_ARM_IRQ_TYPE_PPI, KVM_ARM_IRQ_TYPE_SHIFT, KVM_ARM_IRQ_TYPE_SPI, KVM_ARM_IRQ_VCPU_SHIFT,
    KVM_DEV_ARM_VGIC_GRP_DIST_REGS, kvm_device_attr,
};
use kvm_ioctls::{DeviceFd, VmFd};

use crate::arch::aarch64::gic::GicError;
use crate::arch::aarch64::gic::regs::{GicRegState, MmioReg, SimpleReg, VgicRegEngine};
use crate::arch::{GSI_LEGACY_NUM, SPI_START};

// Distributor registers as detailed at page 75 from
// https://developer.arm.com/documentation/ihi0048/latest/.
// Address offsets are relative to the Distributor base address defined
// by the system memory map.
const GICD_CTLR: DistReg = DistReg::simple(0x0, 4);
const GICD_IIDR: DistReg = DistReg::simple(0x0008, 4);
const GICD_IGROUPR: DistReg = DistReg::shared_irq(0x0080, 1);
const GICD_ISENABLER: DistReg = DistReg::shared_irq(0x0100, 1);
const GICD_ICENABLER: DistReg = DistReg::shared_irq(0x0180, 1);
const GICD_ISPENDR: DistReg = DistReg::shared_irq(0x0200, 1);
const GICD_ICPENDR: DistReg = DistReg::shared_irq(0x0280, 1);
const GICD_ISACTIVER: DistReg = DistReg::shared_irq(0x0300, 1);
const GICD_ICACTIVER: DistReg = DistReg::shared_irq(0x0380, 1);
const GICD_IPRIORITYR: DistReg = DistReg::shared_irq(0x0400, 8);
const GICD_ITARGETSR: DistReg = DistReg::shared_irq(0x0800, 8);
const GICD_ICFGR: DistReg = DistReg::shared_irq(0x0C00, 2);
const GICD_CPENDSGIR: DistReg = DistReg::simple(0xF10, 16);
const GICD_SPENDSGIR: DistReg = DistReg::simple(0xF20, 16);

// List with relevant distributor registers that we will be restoring.
// Order is taken from qemu.
// Criteria for the present list of registers: only R/W registers, implementation specific registers
// are not saved.
static VGIC_DIST_REGS: &[DistReg] = &[
    GICD_CTLR,
    GICD_ICENABLER,
    GICD_ISENABLER,
    GICD_IGROUPR,
    GICD_ICFGR,
    GICD_ICPENDR,
    GICD_ISPENDR,
    GICD_ICACTIVER,
    GICD_ISACTIVER,
    GICD_IPRIORITYR,
    GICD_CPENDSGIR,
    GICD_SPENDSGIR,
];

/// Some registers have variable lengths since they dedicate a specific number of bits to
/// each interrupt. So, their length depends on the number of interrupts.
/// (i.e the ones that are represented as GICD_REG<n>) in the documentation mentioned above.
#[derive(PartialEq)]
pub struct SharedIrqReg {
    /// The offset from the component address. The register is memory mapped here.
    offset: u64,
    /// Number of bits per interrupt.
    bits_per_irq: u8,
}

impl MmioReg for SharedIrqReg {
    fn range(&self) -> Range<u64> {
        // The snapshot only contains SPIs. Banked SGI/PPI words are omitted.
        let start = self.offset + u64::from(SPI_START) * u64::from(self.bits_per_irq) / 8;

        let size_in_bits = u64::from(self.bits_per_irq) * u64::from(GSI_LEGACY_NUM);
        let mut size_in_bytes = size_in_bits / 8;
        if size_in_bits % 8 > 0 {
            size_in_bytes += 1;
        }

        start..start + size_in_bytes
    }
}

#[derive(PartialEq)]
enum DistReg {
    Simple(SimpleReg),
    SharedIrq(SharedIrqReg),
}

impl DistReg {
    const fn simple(offset: u64, size: u16) -> DistReg {
        DistReg::Simple(SimpleReg::new(offset, size))
    }

    const fn shared_irq(offset: u64, bits_per_irq: u8) -> DistReg {
        DistReg::SharedIrq(SharedIrqReg {
            offset,
            bits_per_irq,
        })
    }
}

impl MmioReg for DistReg {
    fn range(&self) -> Range<u64> {
        match self {
            DistReg::Simple(reg) => reg.range(),
            DistReg::SharedIrq(reg) => reg.range(),
        }
    }
}

struct DistRegEngine {}

impl VgicRegEngine for DistRegEngine {
    type Reg = DistReg;
    type RegChunk = u32;

    fn group() -> u32 {
        KVM_DEV_ARM_VGIC_GRP_DIST_REGS
    }

    fn kvm_device_attr(offset: u64, val: &mut Self::RegChunk, cpuid: u64) -> kvm_device_attr {
        kvm_device_attr {
            group: Self::group(),
            // vgic_v2_parse_attr selects banked registers by CPU index, not MPIDR affinity.
            // Ordinary shared-register save/restore always passes CPU 0.
            attr: ((cpuid & 0xff) << 32) | offset,
            addr: val as *mut Self::RegChunk as u64,
            flags: 0,
        }
    }
}

pub(crate) fn get_dist_regs(fd: &DeviceFd) -> Result<Vec<GicRegState<u32>>, GicError> {
    DistRegEngine::get_regs_data(fd, Box::new(VGIC_DIST_REGS.iter()), 0)
}

pub(crate) fn set_dist_regs(fd: &DeviceFd, state: &[GicRegState<u32>]) -> Result<(), GicError> {
    DistRegEngine::set_regs_data(fd, Box::new(VGIC_DIST_REGS.iter()), state, 0)
}

fn reset_private_regs(fd: &DeviceFd, vm_fd: &VmFd, cpuid: u64) -> Result<(), GicError> {
    // GICv2 CPU indices fit in the low IRQ_LINE CPU field; its bits 28..31 stay zero.
    let ppi_attr = (KVM_ARM_IRQ_TYPE_PPI << KVM_ARM_IRQ_TYPE_SHIFT)
        | (((cpuid & 0xff) as u32) << KVM_ARM_IRQ_VCPU_SHIFT);
    for intid in 16..SPI_START {
        // vgic_validate_injection ignores kernel-owned PPIs; their owners must reset them first.
        vm_fd
            .set_irq_line(ppi_attr | intid, false)
            .map_err(GicError::ResetInterrupt)?;
    }

    // vgic_mmio_write_sgipendc clears SGI sources and latches; ICPENDR0 only clears PPIs here.
    DistRegEngine::set_reg_value(fd, &GICD_CPENDSGIR, u32::MAX, cpuid)?;
    DistRegEngine::set_reg_value(fd, &DistReg::simple(0x0280, 4), 0xffff_0000, cpuid)?;
    DistRegEngine::set_reg_value(fd, &DistReg::simple(0x0380, 4), u32::MAX, cpuid)?;

    // vgic_setup_private_irq defaults to group 0, enabled SGIs and disabled PPIs; priority is zero.
    DistRegEngine::set_reg_value(fd, &DistReg::simple(0x0080, 4), 0, cpuid)?;
    DistRegEngine::set_reg_value(fd, &DistReg::simple(0x0400, 32), 0, cpuid)?;
    DistRegEngine::set_reg_value(fd, &DistReg::simple(0x0180, 4), 0xffff_0000, cpuid)?;
    DistRegEngine::set_reg_value(fd, &DistReg::simple(0x0100, 4), 0x0000_ffff, cpuid)?;
    // vgic_mmio_write_config and vgic_mmio_write_target keep private config and targets read-only.
    Ok(())
}

pub(crate) fn set_dist_regs_in_place(
    fd: &DeviceFd,
    vm_fd: &VmFd,
    mpidrs: &[u64],
    state: &[GicRegState<u32>],
) -> Result<(), GicError> {
    // vgic_mmio_uaccess_write_v2_misc must acknowledge IIDR before userspace can reset IGROUPR.
    let mut iidr = 0;
    // SAFETY: `iidr` is a writable u32, the width of a GICv2 distributor attribute.
    unsafe {
        fd.get_device_attr(&mut DistRegEngine::kvm_device_attr(
            GICD_IIDR.range().start,
            &mut iidr,
            0,
        ))
        .map_err(|err| GicError::DeviceAttribute(err, false, DistRegEngine::group()))?;
    }
    DistRegEngine::set_reg_value(fd, &GICD_IIDR, iidr, 0)?;

    // vgic_validate_injection ignores IRQ_LINE(false) for edge IRQs, even with an old level.
    // Temporarily make every SPI level-triggered before clearing its raw line state.
    DistRegEngine::set_reg_value(fd, &GICD_ICFGR, 0, 0)?;
    for intid in SPI_START..SPI_START + GSI_LEGACY_NUM {
        vm_fd
            .set_irq_line(
                (KVM_ARM_IRQ_TYPE_SPI << KVM_ARM_IRQ_TYPE_SHIFT) | intid,
                false,
            )
            .map_err(GicError::ResetInterrupt)?;
    }

    // vgic_uaccess_write_senable, vgic_uaccess_write_spending and
    // vgic_mmio_uaccess_write_sactive only set 1 bits.
    for reg in [GICD_ICENABLER, GICD_ICPENDR, GICD_ICACTIVER] {
        DistRegEngine::set_reg_value(fd, &reg, u32::MAX, 0)?;
    }
    // Packed snapshot MPIDRs do not identify GICv2's banked CPU indices.
    for cpuid in 0..mpidrs.len() {
        reset_private_regs(fd, vm_fd, cpuid as u64)?;
    }
    DistRegEngine::set_reg_value(fd, &GICD_ITARGETSR, 0, 0)?;
    DistRegEngine::set_reg_value(fd, &GICD_IGROUPR, 0, 0)?;

    for (reg, data) in VGIC_DIST_REGS.iter().zip(state) {
        // Fresh load never acknowledges IIDR, so vgic_mmio_uaccess_write_v2_group
        // ignores its saved group values.
        if reg != &GICD_IGROUPR {
            DistRegEngine::set_reg_data(fd, reg, data, 0)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]

    use std::os::unix::io::AsRawFd;

    use kvm_ioctls::Kvm;

    use super::super::tests::{get_reg, set_reg};
    use super::super::{restore_state, restore_state_in_place, save_state};
    use super::*;
    use crate::arch::aarch64::gic::{GICVersion, GicError, create_gic};

    #[test]
    fn test_access_dist_regs() {
        let kvm = Kvm::new().unwrap();
        let vm = kvm.create_vm().unwrap();
        let _ = vm.create_vcpu(0).unwrap();
        let gic_fd = match create_gic(&vm, 1, Some(GICVersion::GICV2)) {
            Ok(gic_fd) => gic_fd,
            Err(GicError::CreateGIC(_)) => return,
            _ => panic!("Failed to open setup GICv2"),
        };

        let res = get_dist_regs(gic_fd.device_fd());
        let state = res.unwrap();

        let res = set_dist_regs(gic_fd.device_fd(), &state);
        res.unwrap();

        unsafe { libc::close(gic_fd.device_fd().as_raw_fd()) };

        let res = get_dist_regs(gic_fd.device_fd());
        assert_eq!(
            format!("{:?}", res.unwrap_err()),
            "DeviceAttribute(Error(9), false, 1)"
        );

        // dropping gic_fd would double close the gic fd, so leak it
        std::mem::forget(gic_fd);
    }

    #[test]
    fn test_restore_dist_regs_in_place_matches_fresh_load() {
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
        let mpidrs = [0, 1 << 32];
        let group = KVM_DEV_ARM_VGIC_GRP_DIST_REGS;

        let iidr = get_reg(live_fd, group, 0, 0x0008);
        set_reg(live_fd, group, 0, 0x0008, iidr);
        for (reg, value) in [
            (GICD_CTLR, 1),
            (GICD_ISENABLER, 0x1111_1111),
            (GICD_IGROUPR, 0x5555_5555),
            (GICD_ICFGR, 0xaaaa_aaaa),
            (GICD_ISPENDR, 0x1111_1111),
            (GICD_ISACTIVER, 0x1111_1111),
            (GICD_IPRIORITYR, 0x4040_4040),
            (GICD_ITARGETSR, 0x0101_0101),
            (GICD_SPENDSGIR, 0x0001_0001),
        ] {
            for offset in reg.iter::<u32>() {
                set_reg(live_fd, group, 0, offset, value);
                assert_eq!(get_reg(live_fd, group, 0, offset), value);
            }
        }
        let baseline = save_state(live_fd, &mpidrs).unwrap();

        for reg in [GICD_ICENABLER, GICD_ICPENDR, GICD_ICACTIVER] {
            for offset in reg.iter::<u32>() {
                set_reg(live_fd, group, 0, offset, u32::MAX);
            }
        }
        for (reg, value) in [
            (GICD_CTLR, 0),
            (GICD_ISENABLER, 0x2222_2222),
            (GICD_IGROUPR, 0xaaaa_aaaa),
            (GICD_ICFGR, 0),
            (GICD_ISPENDR, 0x2222_2222),
            (GICD_ISACTIVER, 0x2222_2222),
            (GICD_IPRIORITYR, 0x8080_8080),
            (GICD_ITARGETSR, 0x0202_0202),
        ] {
            for offset in reg.iter::<u32>() {
                set_reg(live_fd, group, 0, offset, value);
            }
        }

        for cpuid in 0_u32..2 {
            for (offset, value) in [
                (0x0080, u32::MAX),
                (0x0180, u32::MAX),
                (0x0100, 0x0001_0000 << cpuid),
                (0x0280, u32::MAX),
                (0x0200, 0x0002_0000 << cpuid),
                (0x0300, 0x0004_0004 << cpuid),
            ] {
                set_reg(live_fd, group, u64::from(cpuid), offset, value);
            }
            for offset in (0x0400..0x0420).step_by(4) {
                let value = 0x8080_8080 | (cpuid * 0x0808_0808);
                set_reg(live_fd, group, u64::from(cpuid), offset, value);
            }
            for offset in GICD_SPENDSGIR.iter::<u32>() {
                let value = if cpuid == 0 { 0x0202_0202 } else { 0x0101_0101 };
                set_reg(live_fd, group, u64::from(cpuid), offset, value);
                assert_eq!(get_reg(live_fd, group, u64::from(cpuid), offset), value);
            }

            let intid = 16 + cpuid;
            live_vm
                .set_irq_line(
                    (KVM_ARM_IRQ_TYPE_PPI << KVM_ARM_IRQ_TYPE_SHIFT)
                        | (cpuid << KVM_ARM_IRQ_VCPU_SHIFT)
                        | intid,
                    true,
                )
                .unwrap();
            assert_ne!(
                get_reg(live_fd, group, u64::from(cpuid), 0x0200) & (1 << intid),
                0
            );
        }

        // Store asserted SPI levels, then hide them behind edge configuration.
        for intid in SPI_START..SPI_START + GSI_LEGACY_NUM {
            live_vm
                .set_irq_line(
                    (KVM_ARM_IRQ_TYPE_SPI << KVM_ARM_IRQ_TYPE_SHIFT) | intid,
                    true,
                )
                .unwrap();
        }
        for offset in GICD_ISPENDR.iter::<u32>() {
            assert_eq!(get_reg(live_fd, group, 0, offset), u32::MAX);
        }
        for offset in GICD_ICFGR.iter::<u32>() {
            set_reg(live_fd, group, 0, offset, 0xaaaa_aaaa);
        }
        for reg in [GICD_ISENABLER, GICD_ISPENDR, GICD_ISACTIVER] {
            for offset in reg.iter::<u32>() {
                assert_eq!(get_reg(live_fd, group, 0, offset), 0x2222_2222);
            }
        }

        restore_state(fresh_fd, &mpidrs, &baseline).unwrap();
        restore_state_in_place(live_fd, &live_vm, &mpidrs, &baseline).unwrap();

        for reg in [
            GICD_CTLR,
            GICD_IGROUPR,
            GICD_ISENABLER,
            GICD_ISPENDR,
            GICD_ISACTIVER,
            GICD_IPRIORITYR,
            GICD_ITARGETSR,
            GICD_ICFGR,
        ] {
            for offset in reg.iter::<u32>() {
                assert_eq!(
                    get_reg(live_fd, group, 0, offset),
                    get_reg(fresh_fd, group, 0, offset),
                    "shared distributor register {offset:#x}"
                );
            }
        }
        for reg in [GICD_IGROUPR, GICD_ITARGETSR] {
            for offset in reg.iter::<u32>() {
                assert_eq!(get_reg(fresh_fd, group, 0, offset), 0);
            }
        }

        for cpuid in 0..2 {
            for range in [
                0x0080..0x0084,
                0x0100..0x0104,
                0x0200..0x0204,
                0x0300..0x0304,
                0x0400..0x0420,
                0x0800..0x0820,
                0x0c00..0x0c08,
                0x0f20..0x0f30,
            ] {
                for offset in range.step_by(4) {
                    assert_eq!(
                        get_reg(live_fd, group, cpuid, offset),
                        get_reg(fresh_fd, group, cpuid, offset),
                        "CPU {cpuid} private distributor register {offset:#x}"
                    );
                }
            }
            assert_eq!(get_reg(fresh_fd, group, cpuid, 0x0080), 0);
            assert_eq!(get_reg(fresh_fd, group, cpuid, 0x0100), 0x0000_ffff);
            assert_eq!(get_reg(fresh_fd, group, cpuid, 0x0c00), 0xaaaa_aaaa);
            assert_eq!(get_reg(fresh_fd, group, cpuid, 0x0c04), 0);
        }

        // A later guest switch back to level must not expose an old raw line level.
        for offset in GICD_ICFGR.iter::<u32>() {
            set_reg(live_fd, group, 0, offset, 0);
            set_reg(fresh_fd, group, 0, offset, 0);
        }
        for offset in GICD_ISPENDR.iter::<u32>() {
            assert_eq!(
                get_reg(live_fd, group, 0, offset),
                get_reg(fresh_fd, group, 0, offset)
            );
            assert_eq!(get_reg(fresh_fd, group, 0, offset), 0x1111_1111);
        }
    }
}
