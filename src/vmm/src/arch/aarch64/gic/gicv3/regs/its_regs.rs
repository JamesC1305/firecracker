// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use kvm_bindings::{
    KVM_DEV_ARM_ITS_CTRL_RESET, KVM_DEV_ARM_ITS_RESTORE_TABLES, KVM_DEV_ARM_ITS_SAVE_TABLES,
    KVM_DEV_ARM_VGIC_GRP_CTRL, KVM_DEV_ARM_VGIC_GRP_ITS_REGS,
};
use kvm_ioctls::DeviceFd;
use serde::{Deserialize, Serialize};

use crate::arch::aarch64::gic::GicError;

// ITS registers that we want to preserve across snapshots
const GITS_CTLR: u32 = 0x0000;
const GITS_IIDR: u32 = 0x0004;
const GITS_CBASER: u32 = 0x0080;
const GITS_CWRITER: u32 = 0x0088;
const GITS_CREADR: u32 = 0x0090;
const GITS_BASER: u32 = 0x0100;

fn set_device_attribute(
    its_device: &DeviceFd,
    group: u32,
    attr: u32,
    val: u64,
) -> Result<(), GicError> {
    let gicv3_its_attr = kvm_bindings::kvm_device_attr {
        group,
        attr: attr as u64,
        addr: &val as *const u64 as u64,
        flags: 0,
    };

    its_device
        .set_device_attr(&gicv3_its_attr)
        .map_err(|err| GicError::DeviceAttribute(err, true, group))
}

fn get_device_attribute(its_device: &DeviceFd, group: u32, attr: u32) -> Result<u64, GicError> {
    let mut val = 0;

    let mut gicv3_its_attr = kvm_bindings::kvm_device_attr {
        group,
        attr: attr as u64,
        addr: &mut val as *mut u64 as u64,
        flags: 0,
    };

    // SAFETY: gicv3_its_attr.addr is safe to write to.
    unsafe { its_device.get_device_attr(&mut gicv3_its_attr) }
        .map_err(|err| GicError::DeviceAttribute(err, false, group))?;

    Ok(val)
}

fn its_read_register(its_fd: &DeviceFd, attr: u32) -> Result<u64, GicError> {
    get_device_attribute(its_fd, KVM_DEV_ARM_VGIC_GRP_ITS_REGS, attr)
}

fn its_set_register(its_fd: &DeviceFd, attr: u32, val: u64) -> Result<(), GicError> {
    set_device_attribute(its_fd, KVM_DEV_ARM_VGIC_GRP_ITS_REGS, attr, val)
}

pub fn its_save_tables(its_fd: &DeviceFd) -> Result<(), GicError> {
    set_device_attribute(
        its_fd,
        KVM_DEV_ARM_VGIC_GRP_CTRL,
        KVM_DEV_ARM_ITS_SAVE_TABLES,
        0,
    )
}

pub fn its_restore_tables(its_fd: &DeviceFd) -> Result<(), GicError> {
    set_device_attribute(
        its_fd,
        KVM_DEV_ARM_VGIC_GRP_CTRL,
        KVM_DEV_ARM_ITS_RESTORE_TABLES,
        0,
    )
}

pub(super) fn its_reset(its_fd: &DeviceFd) -> Result<(), GicError> {
    // vgic_its_reset disables the ITS and frees its device, ITE and collection lists while
    // retaining the table ABI. Clearing GITS_CTLR alone would leave those mappings behind.
    set_device_attribute(
        its_fd,
        KVM_DEV_ARM_VGIC_GRP_CTRL,
        KVM_DEV_ARM_ITS_CTRL_RESET,
        0,
    )
}

/// ITS registers that we save/restore during snapshot
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ItsRegisterState {
    iidr: u64,
    cbaser: u64,
    creadr: u64,
    cwriter: u64,
    baser: [u64; 8],
    ctlr: u64,
}

impl ItsRegisterState {
    /// Save ITS state
    pub fn save(its_fd: &DeviceFd) -> Result<Self, GicError> {
        let mut state = ItsRegisterState::default();

        for i in 0..8 {
            state.baser[i as usize] = its_read_register(its_fd, GITS_BASER + i * 8)?;
        }
        state.ctlr = its_read_register(its_fd, GITS_CTLR)?;
        state.cbaser = its_read_register(its_fd, GITS_CBASER)?;
        state.creadr = its_read_register(its_fd, GITS_CREADR)?;
        state.cwriter = its_read_register(its_fd, GITS_CWRITER)?;
        state.iidr = its_read_register(its_fd, GITS_IIDR)?;

        Ok(state)
    }

    /// Restore ITS state on a fresh or reset, disabled ITS.
    ///
    /// vgic_mmio_write_its_cbaser clears both queue offsets, so restore it before CREADR and
    /// CWRITER. vgic_mmio_write_its_baser ignores writes while enabled. Restore the table ABI
    /// and registers before vgic_its_restore_tables_v0, then GITS_CTLR last because
    /// vgic_mmio_write_its_ctlr can process queued commands when it enables the ITS.
    pub fn restore(&self, its_fd: &DeviceFd) -> Result<(), GicError> {
        its_set_register(its_fd, GITS_CBASER, self.cbaser)?;
        its_set_register(its_fd, GITS_IIDR, self.iidr)?;
        its_set_register(its_fd, GITS_CREADR, self.creadr)?;
        its_set_register(its_fd, GITS_CWRITER, self.cwriter)?;
        for i in 0..8 {
            its_set_register(its_fd, GITS_BASER + i * 8, self.baser[i as usize])?;
        }
        // We need to restore saved ITS tables before restoring GITS_CTLR
        its_restore_tables(its_fd)?;
        its_set_register(its_fd, GITS_CTLR, self.ctlr)
    }
}

#[cfg(test)]
mod tests {
    use kvm_bindings::{
        KVM_DEV_ARM_VGIC_GRP_DIST_REGS, KVM_DEV_ARM_VGIC_GRP_REDIST_REGS, KVM_MSI_VALID_DEVID,
        kvm_device_attr, kvm_msi, kvm_vcpu_init,
    };
    use vm_memory::{Bytes, GuestAddress};

    use super::super::super::save_pending_tables;
    use super::super::{redist_regs, restore_state_in_place, save_state};
    use super::*;
    use crate::arch::aarch64::gic::{GICVersion, create_gic};
    use crate::vstate::vm::tests::setup_vm_with_memory;

    #[test]
    fn test_restore_state_in_place_replays_enabled_its_with_live_lpis() {
        const MEMORY_SIZE: usize = 0x9_0000;
        const COMMAND_TABLE: u64 = 0x1_0000;
        const DEVICE_TABLE: u64 = 0x2_0000;
        const COLLECTION_TABLE: u64 = 0x3_0000;
        const ITT: u64 = 0x4_0000;
        const PROPERTY_TABLE: u64 = 0x5_0000;
        const PENDING_TABLES: [u64; 2] = [0x6_0000, 0x7_0000];
        const LIVE_COMMAND_TABLE: u64 = 0x8_0000;
        const MPIDRS: [u64; 2] = [0, 1 << 32];
        const VALID: u64 = 1 << 63;
        const BASER_PAGE_SIZE_64K: u64 = 2 << 8;
        const COLLECTION_ID: u64 = 7;
        const FIRST_LPI: u64 = 8192;
        const GICR_CTLR: u64 = 0x0000;
        const GICR_PROPBASER: u64 = 0x0070;
        const GICR_PENDBASER: u64 = 0x0078;
        const GITS_TRANSLATER: u64 = 0x1_0040;

        // No vCPU enters KVM_RUN and there are no asynchronous interrupt producers.
        // vgic_sanitise_its_baser fixes table pages at 64 KiB; each buffer is aligned
        // and backed by registered guest RAM, including both redistributor pending tables.
        let vm = setup_vm_with_memory(MEMORY_SIZE);
        let vcpus = [
            vm.fd().create_vcpu(0).unwrap(),
            vm.fd().create_vcpu(1).unwrap(),
        ];
        let mut target = kvm_vcpu_init::default();
        vm.fd().get_preferred_target(&mut target).unwrap();
        // kvm_reset_vcpu initializes each MPIDR used by the redistributor selectors.
        for vcpu in &vcpus {
            vcpu.vcpu_init(&target).unwrap();
        }
        let gic = create_gic(vm.fd(), 2, Some(GICVersion::GICV3)).unwrap();
        let gic_fd = gic.device_fd();
        let its_fd = gic.its_fd().unwrap();
        let memory = vm.guest_memory();
        let pending_addrs = PENDING_TABLES.map(|base| GuestAddress(base + FIRST_LPI / 8));
        let read_pending = || pending_addrs.map(|addr| memory.read_obj::<u8>(addr).unwrap());
        let write_entry = |addr, value: u64| {
            memory.write_obj(value.to_le(), GuestAddress(addr)).unwrap();
        };
        let read_entry = |addr| u64::from_le(memory.read_obj(GuestAddress(addr)).unwrap());
        let clear_table_entries = || {
            for addr in [DEVICE_TABLE, COLLECTION_TABLE, ITT, ITT + 8] {
                write_entry(addr, 0);
            }
        };
        let set_redist = |mpidr, offset, value: u32| {
            gic_fd
                .set_device_attr(&kvm_device_attr {
                    group: KVM_DEV_ARM_VGIC_GRP_REDIST_REGS,
                    attr: mpidr | offset,
                    addr: &value as *const u32 as u64,
                    flags: 0,
                })
                .unwrap();
        };
        let load_tables = |command_table, queue_offset| {
            // Install the fixture without using either production register-restore path.
            // vgic_mmio_write_its_cbaser clears both offsets, so set the empty queue last.
            its_set_register(its_fd, GITS_CBASER, VALID | command_table).unwrap();
            its_set_register(its_fd, GITS_CREADR, queue_offset).unwrap();
            its_set_register(its_fd, GITS_CWRITER, queue_offset).unwrap();
            for (reg, table) in [
                (GITS_BASER, DEVICE_TABLE),
                (GITS_BASER + 8, COLLECTION_TABLE),
            ] {
                its_set_register(its_fd, reg, VALID | BASER_PAGE_SIZE_64K | table).unwrap();
            }
            its_restore_tables(its_fd).unwrap();
            its_set_register(its_fd, GITS_CTLR, 1).unwrap();
        };

        // vgic_target_oracle needs the distributor enabled for pending, enabled LPIs
        // to acquire AP-list references. ITS reset alone does not release those references.
        set_device_attribute(gic_fd, KVM_DEV_ARM_VGIC_GRP_DIST_REGS, 0, 2).unwrap();
        assert_eq!(
            get_device_attribute(gic_fd, KVM_DEV_ARM_VGIC_GRP_DIST_REGS, 0).unwrap() & 2,
            2
        );
        let iidr = its_read_register(its_fd, GITS_IIDR).unwrap();
        its_set_register(its_fd, GITS_IIDR, iidr & !(0xf << 12)).unwrap();

        // ABI0 in Documentation/virt/kvm/devices/arm-vgic-its.rst uses 8-byte entries.
        // vgic_its_restore_dte interprets Size=0 as one EventID bit (events 0 and 1).
        // vgic_its_restore_cte uses the vCPU ID, not the packed MPIDR, as its target.
        let dte = VALID | ((ITT >> 8) << 5);
        let baseline_ite = (FIRST_LPI << 16) | COLLECTION_ID;
        let live_extra_ite = ((FIRST_LPI + 1) << 16) | COLLECTION_ID;
        write_entry(DEVICE_TABLE, dte);
        write_entry(COLLECTION_TABLE, VALID | COLLECTION_ID);
        write_entry(COLLECTION_TABLE + 8, 0);
        write_entry(ITT, baseline_ite);
        write_entry(ITT + 8, 0);
        // update_lpi_config reads enabled/priority from PROPBASER + INTID - 8192.
        memory
            .write_slice(&[0xa3, 0], GuestAddress(PROPERTY_TABLE))
            .unwrap();
        memory.write_obj(1u8, pending_addrs[0]).unwrap();
        memory.write_obj(0u8, pending_addrs[1]).unwrap();

        // vgic_mmio_write_propbase and vgic_mmio_write_pendbase require LPIs off.
        // IDbits=13 includes both LPIs; PTZ remains clear so pending RAM is imported.
        for (mpidr, pending_table) in MPIDRS.into_iter().zip(PENDING_TABLES) {
            set_redist(
                mpidr,
                GICR_PROPBASER,
                u32::try_from(PROPERTY_TABLE).unwrap() | 13,
            );
            set_redist(mpidr, GICR_PROPBASER + 4, 0);
            set_redist(mpidr, GICR_PENDBASER, u32::try_from(pending_table).unwrap());
            set_redist(mpidr, GICR_PENDBASER + 4, 0);
        }
        for mpidr in MPIDRS {
            set_redist(mpidr, GICR_CTLR, 1);
        }
        load_tables(COMMAND_TABLE, 0x40);
        // vgic_add_lpi calls vgic_v3_lpi_sync_pending_status, consuming the RAM bit.
        assert_eq!(read_pending(), [0, 0]);

        // Clear the encoded entries so capture must serialize real kernel objects.
        // Capture RAM only after save_state flushes both ITS and pending tables.
        clear_table_entries();
        let baseline = save_state(gic_fd, its_fd, &MPIDRS).unwrap();
        let saved_its = baseline.its_state.as_ref().unwrap();
        assert_eq!(saved_its.creadr, 0x40);
        assert_eq!(saved_its.cwriter, 0x40);
        assert_eq!(saved_its.ctlr & 1, 1);
        assert_eq!(saved_its.iidr & (0xf << 12), 0);
        assert_eq!(read_pending(), [1, 0]);
        assert_eq!(read_entry(DEVICE_TABLE), dte);
        assert_eq!(read_entry(COLLECTION_TABLE), VALID | COLLECTION_ID);
        assert_eq!(read_entry(ITT), baseline_ite);
        assert_eq!(read_entry(ITT + 8), 0);
        let mut baseline_ram = vec![0; MEMORY_SIZE];
        memory
            .read_slice(&mut baseline_ram, GuestAddress(0))
            .unwrap();

        let msi_address = gic.msi_properties().unwrap()[0] + GITS_TRANSLATER;
        let extra_msi = kvm_msi {
            address_lo: u32::try_from(msi_address & 0xffff_ffff).unwrap(),
            address_hi: u32::try_from(msi_address >> 32).unwrap(),
            data: 1,
            flags: KVM_MSI_VALID_DEVID,
            devid: 0,
            ..Default::default()
        };

        // Rewind twice. The second live state has a clear 8192 latch, so it must
        // regain the baseline pending bit as well as move back from CPU1 to CPU0.
        for (queue_offset, pending) in [(0x80, 0b11u8), (0xc0, 0b10)] {
            // Establish divergence independently: vgic_mmio_write_v3r_ctlr releases
            // AP-list/cache references before vgic_its_reset frees the ITE references.
            for mpidr in MPIDRS {
                redist_regs::disable_lpis(gic_fd, mpidr).unwrap();
            }
            its_reset(its_fd).unwrap();
            memory.write_slice(&baseline_ram, GuestAddress(0)).unwrap();
            write_entry(COLLECTION_TABLE, VALID | (1 << 16) | COLLECTION_ID);
            write_entry(ITT, (1 << 48) | baseline_ite);
            write_entry(ITT + 8, live_extra_ite);
            memory
                .write_obj(0xa3u8, GuestAddress(PROPERTY_TABLE + 1))
                .unwrap();
            memory.write_obj(0u8, pending_addrs[0]).unwrap();
            memory.write_obj(pending, pending_addrs[1]).unwrap();
            for mpidr in MPIDRS {
                set_redist(mpidr, GICR_CTLR, 1);
            }
            load_tables(LIVE_COMMAND_TABLE, queue_offset);
            assert_eq!(read_pending(), [0, 0]);
            assert_eq!(its_read_register(its_fd, GITS_CTLR).unwrap() & 1, 1);
            assert_eq!(
                its_read_register(its_fd, GITS_CREADR).unwrap(),
                queue_offset
            );
            assert_eq!(
                its_read_register(its_fd, GITS_CWRITER).unwrap(),
                queue_offset
            );
            assert_ne!(
                its_read_register(its_fd, GITS_CBASER).unwrap(),
                saved_its.cbaser
            );

            // vgic_its_resolve_lpi leaves a translation-cache reference on the first
            // injection; the second exercises vgic_its_inject_cached_translation.
            assert_eq!(vm.fd().signal_msi(extra_msi).unwrap(), 1);
            assert_eq!(vm.fd().signal_msi(extra_msi).unwrap(), 1);
            clear_table_entries();
            its_save_tables(its_fd).unwrap();
            save_pending_tables(gic_fd).unwrap();
            assert_eq!(read_pending(), [0, pending]);
            assert_eq!(read_entry(DEVICE_TABLE), dte);
            assert_eq!(
                read_entry(COLLECTION_TABLE),
                VALID | (1 << 16) | COLLECTION_ID
            );
            assert_eq!(read_entry(ITT), (1 << 48) | baseline_ite);
            assert_eq!(read_entry(ITT + 8), live_extra_ite);

            // Leave the divergent ITS enabled and its LPIs live. No save ioctl may
            // overwrite this RAM checkpoint before restore_state_in_place consumes it.
            memory.write_slice(&baseline_ram, GuestAddress(0)).unwrap();
            restore_state_in_place(gic_fd, its_fd, &MPIDRS, &baseline).unwrap();
            let restored_its = ItsRegisterState::save(its_fd).unwrap();
            assert_eq!(restored_its.iidr, saved_its.iidr);
            assert_eq!(restored_its.cbaser, saved_its.cbaser);
            assert_eq!(restored_its.creadr, saved_its.creadr);
            assert_eq!(restored_its.cwriter, saved_its.cwriter);
            assert_eq!(restored_its.baser, saved_its.baser);
            assert_eq!(restored_its.ctlr, saved_its.ctlr);
            assert_eq!(read_pending(), [0, 0]);

            // vgic_its_inject_msi returns 0 for an unmapped event. A stale translation
            // cache entry for event 1 must not resurrect the runtime-only 8193 LPI.
            assert_eq!(vm.fd().signal_msi(extra_msi).unwrap(), 0);
            clear_table_entries();
            its_save_tables(its_fd).unwrap();
            save_pending_tables(gic_fd).unwrap();
            // vgic_v3_save_pending_tables exports the kernel latches, not the input
            // RAM bits (which were consumed). Only 8192 on CPU0 may be pending.
            assert_eq!(read_pending(), [1, 0]);
            assert_eq!(read_entry(DEVICE_TABLE), dte);
            assert_eq!(read_entry(COLLECTION_TABLE), VALID | COLLECTION_ID);
            assert_eq!(read_entry(ITT), baseline_ite);
            assert_eq!(read_entry(ITT + 8), 0);
        }
    }
}
