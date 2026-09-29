// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines state structures for saving/restoring a Firecracker microVM.

use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::mem::forget;
use std::os::unix::io::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::utils::Version;
use serde::{Deserialize, Serialize};
use userfaultfd::{FeatureFlags, Uffd, UffdBuilder};
use vmm_sys_util::sock_ctrl_msg::ScmSocket;

#[cfg(target_arch = "aarch64")]
use crate::arch::aarch64::vcpu::get_manufacturer_id_from_host;
use crate::builder::{self, BuildMicrovmFromSnapshotError};
use crate::cpu_config::templates::StaticCpuTemplate;
#[cfg(target_arch = "x86_64")]
use crate::cpu_config::x86_64::cpuid::CpuidTrait;
#[cfg(target_arch = "x86_64")]
use crate::cpu_config::x86_64::cpuid::common::get_vendor_id_from_host;
use crate::device_manager::{DevicePersistError, DevicesState};
// Re-exported so external crates inspecting a `MicrovmState` snapshot can match on the
// serialised virtio transport variant.
pub use crate::device_manager::VirtioDevicesState;
use crate::devices::virtio::device::VirtioDeviceId;
use crate::logger::{info, warn};
use crate::mmds::data_store::MmdsData;
use crate::resources::VmResources;
use crate::seccomp::BpfThreadMap;
use crate::snapshot::Snapshot;
use crate::utils::u64_to_usize;
use crate::vmm_config::boot_source::BootSourceConfig;
use crate::vmm_config::instance_info::{InstanceInfo, VmState as InstanceState};
use crate::vmm_config::machine_config::{HugePageConfig, MachineConfigError, MachineConfigUpdate};
use crate::vmm_config::snapshot::{CreateSnapshotParams, LoadSnapshotParams, MemBackendType};
use crate::vstate::kvm::KvmState;
use crate::vstate::memory::{
    self, GuestMemoryExtension, GuestMemoryState, GuestRegionMmap, GuestRegionType, MemoryError,
};
use crate::vstate::vcpu::{VcpuSendEventError, VcpuState};
use crate::vstate::vm::{VmError, VmState};
use crate::{DirtyBitmap, EventManager, Vmm, vstate};

/// Holds information related to the VM that is not part of VmState.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
pub struct VmInfo {
    /// Guest memory size.
    pub mem_size_mib: u64,
    /// smt information
    pub smt: bool,
    /// CPU template type
    pub cpu_template: StaticCpuTemplate,
    /// Boot source information.
    pub boot_source: BootSourceConfig,
    /// Huge page configuration
    pub huge_pages: HugePageConfig,
}

impl From<&VmResources> for VmInfo {
    fn from(value: &VmResources) -> Self {
        Self {
            mem_size_mib: value.machine_config.mem_size_mib as u64,
            smt: value.machine_config.smt,
            cpu_template: StaticCpuTemplate::from(&value.machine_config.cpu_template),
            boot_source: value.boot_source.config.clone(),
            huge_pages: value.machine_config.huge_pages,
        }
    }
}

impl From<&Vmm> for VmInfo {
    fn from(value: &Vmm) -> Self {
        let machine_config = &value.machine_config;
        Self {
            mem_size_mib: machine_config.mem_size_mib as u64,
            smt: machine_config.smt,
            cpu_template: StaticCpuTemplate::from(&machine_config.cpu_template),
            boot_source: value.boot_source_config.clone(),
            huge_pages: machine_config.huge_pages,
        }
    }
}

/// Contains the necessary state for saving/restoring a microVM.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MicrovmState {
    /// Miscellaneous VM info.
    pub vm_info: VmInfo,
    /// KVM KVM state.
    pub kvm_state: KvmState,
    /// VM KVM state.
    pub vm_state: VmState,
    /// Vcpu states.
    pub vcpu_states: Vec<VcpuState>,
    /// Device states.
    pub device_states: DevicesState,
}

/// Load-time snapshot state that reset returns the microVM to.
#[derive(Debug)]
pub struct ResetContext {
    /// VM state to reapply without replacing the live allocator.
    pub vm_state: VmState,
    /// vCPU states copied to their owning threads on each reset.
    pub vcpu_states: Vec<VcpuState>,
    #[cfg(target_arch = "aarch64")]
    /// Saved vCPU affinities in KVM's GIC register attribute format.
    pub mpidrs: Vec<u64>,
    /// Device state to reapply while keeping host resources open.
    pub device_states: DevicesState,
    /// Whether each virtio device was activated after load. In-place restore cannot add,
    /// remove, activate or deactivate a device, so reset requires the same devices here.
    pub virtio_devices: HashMap<VirtioDeviceId, bool>,
    /// Where each PCI device mapped its BAR after load, and nothing with the MMIO transport.
    /// A mapping cannot move back in place, so reset requires the same addresses here.
    pub pci_bar_addresses: HashMap<VirtioDeviceId, u64>,
    /// A device was hot-plugged or unplugged since load. Reset cannot tell whether a device
    /// with the same ID is the one that load created.
    pub devices_hotplugged: bool,
    /// MMDS data after load.
    pub mmds_data: MmdsData,
    /// Whether snapshot load applied wall-clock time to kvmclock.
    pub clock_realtime: bool,
    /// Pages written since load whose dirty bits snapshot creation cleared.
    pub dirty_pages: DirtyBitmap,
    /// A reset failed part way. The microVM state is inconsistent, so it must not run, be
    /// snapshotted or be reset again.
    pub poisoned: bool,
}

/// This describes the mapping between Firecracker base virtual address and
/// offset in the buffer or file backend for a guest memory region. It is used
/// to tell an external process/thread where to populate the guest memory data
/// for this range.
///
/// E.g. Guest memory contents for a region of `size` bytes can be found in the
/// backend at `offset` bytes from the beginning, and should be copied/populated
/// into `base_host_address`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GuestRegionUffdMapping {
    /// Base host virtual address where the guest memory contents for this
    /// region should be copied/populated.
    pub base_host_virt_addr: u64,
    /// Region size.
    pub size: usize,
    /// Offset in the backend file/buffer where the region contents are.
    pub offset: u64,
    /// The configured page size for this memory region.
    pub page_size: usize,
    /// The configured page size **in bytes** for this memory region. The name is
    /// wrong but cannot be changed due to being API, so this field is deprecated,
    /// to be removed in 2.0.
    #[deprecated]
    pub page_size_kib: usize,
}

/// Errors related to saving and restoring Microvm state.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum MicrovmStateError {
    /// Operation not allowed: {0}
    NotAllowed(String),
    /// Cannot restore devices: {0}
    RestoreDevices(#[from] DevicePersistError),
    /// Cannot save Vcpu state: {0}
    SaveVcpuState(vstate::vcpu::VcpuError),
    /// Cannot save KvmVm state: {0}
    SaveVmState(vstate::vm::KvmVmError),
    /// Cannot signal Vcpu: {0}
    SignalVcpu(VcpuSendEventError),
    /// Vcpu is in unexpected state.
    UnexpectedVcpuResponse,
}

/// Errors associated with creating a snapshot.
#[rustfmt::skip]
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum CreateSnapshotError {
    /// A reset failed part way, so the microVM state is inconsistent.
    ResetPoisoned,
    /// Cannot get dirty bitmap: {0}
    DirtyBitmap(#[from] VmError),
    /// Cannot write memory file: {0}
    Memory(#[from] MemoryError),
    /// Cannot perform {0} on the memory backing file: {1}
    MemoryBackingFile(&'static str, io::Error),
    /// Cannot save the microVM state: {0}
    MicrovmState(MicrovmStateError),
    /// Cannot serialize the microVM state: {0}
    SerializeMicrovmState(#[from] crate::snapshot::SnapshotError),
    /// Cannot perform {0} on the snapshot backing file: {1}
    SnapshotBackingFile(&'static str, io::Error),
}

/// Errors associated with resetting a live microVM to its load-time snapshot.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum ResetSnapshotError {
    /// Reset requires the microVM to be paused.
    NotPaused,
    /// The microVM was not loaded from a snapshot with a file memory backend and dirty page
    /// tracking.
    NoResetContext,
    /// A previous reset failed part way, so the microVM state is inconsistent.
    Poisoned,
    /// Reset does not support {0}.
    Unsupported(&'static str),
    /// Virtio devices were added, removed, hot-plugged, unplugged, activated, reset or moved
    /// since the snapshot was loaded.
    DevicesChanged,
    /// Failed to create a new MMDS token key: {0}
    MmdsTokenKey(#[from] crate::mmds::data_store::MmdsDatastoreError),
    /// Failed to complete pending vCPU I/O: {0}
    CompleteVcpuIo(crate::vstate::vm::RestoreVcpuStatesError),
    /// Failed to complete in-flight block I/O: {0}
    DrainBlockIo(#[from] crate::devices::virtio::block::virtio::VirtioBlockError),
    /// Failed to get the dirty bitmap: {0}
    DirtyBitmap(#[from] VmError),
    /// Failed to revert guest memory: {0}
    RevertMemory(#[from] MemoryError),
    /// Failed to restore vCPU state: {0}
    RestoreVcpus(#[from] crate::vstate::vm::RestoreVcpuStatesError),
    /// Failed to restore VM state: {0}
    RestoreVm(#[from] crate::vstate::vm::KvmVmError),
    /// Failed to restore devices: {0}
    RestoreDevices(#[from] crate::device_manager::DeviceManagerPersistError),
}

/// Snapshot version
pub const SNAPSHOT_VERSION: Version = Version::new(12, 0, 0);

/// Creates a Microvm snapshot.
pub fn create_snapshot(
    vmm: &mut Vmm,
    vm_info: &VmInfo,
    params: &CreateSnapshotParams,
) -> Result<(), CreateSnapshotError> {
    if vmm.reset_poisoned() {
        return Err(CreateSnapshotError::ResetPoisoned);
    }
    let microvm_state = vmm
        .save_state(vm_info)
        .map_err(CreateSnapshotError::MicrovmState)?;

    snapshot_state_to_file(
        &microvm_state,
        &params.snapshot_path,
        params.sync_snapshot_files,
    )?;

    let kvm_vm = vmm.vm.as_kvm().ok_or_else(|| {
        CreateSnapshotError::MicrovmState(MicrovmStateError::NotAllowed(
            "snapshot requires KVM".into(),
        ))
    })?;
    kvm_vm.snapshot_memory_to_file(
        &params.mem_file_path,
        params.snapshot_type,
        params.sync_snapshot_files,
        vmm.reset_context
            .as_mut()
            .map(|context| &mut context.dirty_pages),
    )?;

    // We need to mark queues as dirty again for all activated devices. The reason we
    // do it here is that we don't mark pages as dirty during runtime
    // for queue objects.
    vmm.device_manager
        .mark_virtio_queue_memory_dirty(kvm_vm.guest_memory());

    Ok(())
}

/// Resets a paused microVM to the snapshot from which it was loaded, keeping its host
/// resources. The microVM stays paused.
///
/// An I/O-completion or preflight error leaves the original microVM usable. An error while
/// applying the snapshot poisons it: it cannot resume, be snapshotted or be reset again.
pub fn reset_to_snapshot(vmm: &mut Vmm) -> Result<(), ResetSnapshotError> {
    use crate::snapshot::Persist;

    // I/O completion refuses running vCPUs, and reversion must not race guest memory writes.
    if vmm.instance_info.state != InstanceState::Paused {
        return Err(ResetSnapshotError::NotPaused);
    }
    let mmds = vmm.get_mmds();
    let context = vmm
        .reset_context
        .as_mut()
        .ok_or(ResetSnapshotError::NoResetContext)?;
    if context.poisoned {
        return Err(ResetSnapshotError::Poisoned);
    }

    let kvm_vm = vmm
        .vm
        .as_kvm()
        .ok_or(ResetSnapshotError::Unsupported("non-KVM VMs"))?;
    // A completed I/O instruction can write RAM or change a device without guest entry.
    // Finish it before checking device state and before reverting memory.
    kvm_vm
        .complete_vcpu_io()
        .map_err(ResetSnapshotError::CompleteVcpuIo)?;
    // The per-device checks pair each live device with its saved state, so a device added
    // since load reports the topology change rather than its kind.
    if context.devices_hotplugged
        || vmm.device_manager.virtio_device_activation() != context.virtio_devices
        || vmm.device_manager.pci_bar_addresses() != context.pci_bar_addresses
    {
        return Err(ResetSnapshotError::DevicesChanged);
    }
    vmm.device_manager
        .check_reset(&context.device_states)
        .map_err(|error| ResetSnapshotError::Unsupported(error.0))?;
    // Create the new MMDS token key now, so that a failure leaves the microVM unchanged.
    let mmds_authority = mmds
        .as_ref()
        .map(|_| crate::mmds::data_store::Mmds::new_token_authority())
        .transpose()?;

    // In-flight asynchronous block I/O can write guest memory, so complete it before the
    // revert. Completing it changes nothing that the resumed microVM would not also see.
    vmm.device_manager.drain_block_io()?;

    // A failure from here on can leave the microVM partly reset, which poisons it.
    let result = apply_reset(kvm_vm, &mut vmm.device_manager, context);
    context.poisoned = result.is_err();
    result?;
    if let (Some(mmds), Some(authority)) = (mmds, mmds_authority) {
        mmds.lock().expect("Poisoned lock").restore_data(
            &context.mmds_data,
            authority,
            &vmm.instance_info.id,
        );
    }
    Ok(())
}

/// Returns guest memory, the vCPUs, the in-kernel VM state and the devices to the state in
/// `context`.
fn apply_reset(
    kvm_vm: &crate::vstate::vm::KvmVm,
    device_manager: &mut crate::device_manager::DeviceManager,
    context: &mut ResetContext,
) -> Result<(), ResetSnapshotError> {
    use crate::snapshot::Persist;
    revert_memory(kvm_vm, &mut context.dirty_pages)?;
    kvm_vm.restore_vcpu_states(&context.vcpu_states)?;
    #[cfg(target_arch = "x86_64")]
    kvm_vm.restore_kvm_state(&context.vm_state, context.clock_realtime)?;
    #[cfg(target_arch = "aarch64")]
    kvm_vm.restore_kvm_state(&context.mpidrs, &context.vm_state)?;
    device_manager.restore_in_place(&context.device_states, kvm_vm)?;
    Ok(())
}

/// Reverts the pages written since load to their contents in the snapshot file: the pages
/// that `dirty_since_load` or the dirty logs mark. Clears both afterwards.
fn revert_memory(
    kvm_vm: &crate::vstate::vm::KvmVm,
    dirty_since_load: &mut DirtyBitmap,
) -> Result<(), ResetSnapshotError> {
    let memory = kvm_vm.guest_memory();
    memory.accumulate_dirty(&kvm_vm.get_dirty_bitmap()?, dirty_since_load);
    memory.revert_to_file(dirty_since_load)?;
    memory.reset_dirty();
    dirty_since_load
        .values_mut()
        .for_each(|bitmap| bitmap.fill(0));
    Ok(())
}

fn snapshot_state_to_file(
    microvm_state: &MicrovmState,
    snapshot_path: &Path,
    sync_snapshot_files: bool,
) -> Result<(), CreateSnapshotError> {
    use self::CreateSnapshotError::*;
    let mut snapshot_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(snapshot_path)
        .map_err(|err| SnapshotBackingFile("open", err))?;

    let snapshot = Snapshot::new(microvm_state);
    snapshot.save(&mut snapshot_file)?;
    snapshot_file
        .flush()
        .map_err(|err| SnapshotBackingFile("flush", err))?;
    if sync_snapshot_files {
        snapshot_file
            .sync_all()
            .map_err(|err| SnapshotBackingFile("sync_all", err))?;
    }
    Ok(())
}

/// Validates that snapshot CPU vendor matches the host CPU vendor.
///
/// # Errors
///
/// When:
/// - Failed to read host vendor.
/// - Failed to read snapshot vendor.
#[cfg(target_arch = "x86_64")]
pub fn validate_cpu_vendor(microvm_state: &MicrovmState) {
    let host_vendor_id = get_vendor_id_from_host();
    let snapshot_vendor_id = microvm_state.vcpu_states[0].cpuid.vendor_id();
    match (host_vendor_id, snapshot_vendor_id) {
        (Ok(host_id), Some(snapshot_id)) => {
            info!("Host CPU vendor ID: {host_id:?}");
            info!("Snapshot CPU vendor ID: {snapshot_id:?}");
            if host_id != snapshot_id {
                warn!("Host CPU vendor ID differs from the snapshotted one",);
            }
        }
        (Ok(host_id), None) => {
            info!("Host CPU vendor ID: {host_id:?}");
            warn!("Snapshot CPU vendor ID: couldn't get from the snapshot");
        }
        (Err(_), Some(snapshot_id)) => {
            warn!("Host CPU vendor ID: couldn't get from the host");
            info!("Snapshot CPU vendor ID: {snapshot_id:?}");
        }
        (Err(_), None) => {
            warn!("Host CPU vendor ID: couldn't get from the host");
            warn!("Snapshot CPU vendor ID: couldn't get from the snapshot");
        }
    }
}

/// Validate that Snapshot Manufacturer ID matches
/// the one from the Host
///
/// The manufacturer ID for the Snapshot is taken from each VCPU state.
/// # Errors
///
/// When:
/// - Failed to read host vendor.
/// - Failed to read snapshot vendor.
#[cfg(target_arch = "aarch64")]
pub fn validate_cpu_manufacturer_id(microvm_state: &MicrovmState) {
    let host_cpu_id = get_manufacturer_id_from_host();
    let snapshot_cpu_id = microvm_state.vcpu_states[0].regs.manifacturer_id();
    match (host_cpu_id, snapshot_cpu_id) {
        (Some(host_id), Some(snapshot_id)) => {
            info!("Host CPU manufacturer ID: {host_id:?}");
            info!("Snapshot CPU manufacturer ID: {snapshot_id:?}");
            if host_id != snapshot_id {
                warn!("Host CPU manufacturer ID differs from the snapshotted one",);
            }
        }
        (Some(host_id), None) => {
            info!("Host CPU manufacturer ID: {host_id:?}");
            warn!("Snapshot CPU manufacturer ID: couldn't get from the snapshot");
        }
        (None, Some(snapshot_id)) => {
            warn!("Host CPU manufacturer ID: couldn't get from the host");
            info!("Snapshot CPU manufacturer ID: {snapshot_id:?}");
        }
        (None, None) => {
            warn!("Host CPU manufacturer ID: couldn't get from the host");
            warn!("Snapshot CPU manufacturer ID: couldn't get from the snapshot");
        }
    }
}
/// Error type for [`snapshot_state_sanity_check`].
#[derive(Debug, thiserror::Error, displaydoc::Display, PartialEq, Eq)]
pub enum SnapShotStateSanityCheckError {
    /// No memory region defined.
    NoMemory,
    /// No DRAM memory region defined.
    NoDramMemory,
    /// DRAM memory has more than a single slot.
    DramMemoryTooManySlots,
    /// DRAM memory is unplugged.
    DramMemoryUnplugged,
}

/// Performs sanity checks against the state file and returns specific errors.
pub fn snapshot_state_sanity_check(
    microvm_state: &MicrovmState,
) -> Result<(), SnapShotStateSanityCheckError> {
    // Check that the snapshot contains at least 1 mem region, that at least one is Dram,
    // and that Dram region contains a single plugged slot.
    // Upper bound check will be done when creating guest memory by comparing against
    // KVM max supported value kvm_context.max_memslots().
    let regions = &microvm_state.vm_state.memory.regions;

    if regions.is_empty() {
        return Err(SnapShotStateSanityCheckError::NoMemory);
    }

    if !regions
        .iter()
        .any(|r| r.region_type == GuestRegionType::Dram)
    {
        return Err(SnapShotStateSanityCheckError::NoDramMemory);
    }

    for dram_region in regions
        .iter()
        .filter(|r| r.region_type == GuestRegionType::Dram)
    {
        if dram_region.plugged.len() != 1 {
            return Err(SnapShotStateSanityCheckError::DramMemoryTooManySlots);
        }

        if !dram_region.plugged[0] {
            return Err(SnapShotStateSanityCheckError::DramMemoryUnplugged);
        }
    }

    #[cfg(target_arch = "x86_64")]
    validate_cpu_vendor(microvm_state);
    #[cfg(target_arch = "aarch64")]
    validate_cpu_manufacturer_id(microvm_state);

    Ok(())
}

/// Error type for [`restore_from_snapshot`].
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum RestoreFromSnapshotError {
    /// Failed to get snapshot state from file: {0}
    File(#[from] SnapshotStateFromFileError),
    /// Invalid snapshot state: {0}
    Invalid(#[from] SnapShotStateSanityCheckError),
    /// Failed to load guest memory: {0}
    GuestMemory(#[from] RestoreFromSnapshotGuestMemoryError),
    /// Failed to build microVM from snapshot: {0}
    Build(#[from] BuildMicrovmFromSnapshotError),
}
/// Sub-Error type for [`restore_from_snapshot`] to contain either [`GuestMemoryFromFileError`] or
/// [`GuestMemoryFromUffdError`] within [`RestoreFromSnapshotError`].
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum RestoreFromSnapshotGuestMemoryError {
    /// Error creating guest memory from file: {0}
    File(#[from] GuestMemoryFromFileError),
    /// Error creating guest memory from uffd: {0}
    Uffd(#[from] GuestMemoryFromUffdError),
}

/// Loads a Microvm snapshot producing a 'paused' Microvm.
pub fn restore_from_snapshot(
    instance_info: &InstanceInfo,
    event_manager: &mut EventManager,
    seccomp_filters: &BpfThreadMap,
    params: &LoadSnapshotParams,
    vm_resources: &mut VmResources,
) -> Result<Arc<Mutex<Vmm>>, RestoreFromSnapshotError> {
    let mut microvm_state = snapshot_state_from_file(&params.snapshot_path)?;
    for entry in &params.network_overrides {
        // Only the active transport carries virtio device state, so we look at whichever
        // variant this snapshot was saved with. The MMIO and PCI transports wrap their net
        // devices in distinct types, so we map down to the shared inner `NetState` in each arm.
        let device_state = match &mut microvm_state.device_states.virtio_state {
            VirtioDevicesState::Mmio(mmio_state) => mmio_state
                .net_devices
                .iter_mut()
                .map(|device| &mut device.device_state)
                .find(|x| x.id == entry.iface_id),
            VirtioDevicesState::Pci(pci_state) => pci_state
                .net_devices
                .iter_mut()
                .map(|device| &mut device.device_state)
                .find(|x| x.id == entry.iface_id),
        };
        device_state
            .map(|device_state| device_state.tap_if_name.clone_from(&entry.host_dev_name))
            .ok_or(SnapshotStateFromFileError::UnknownNetworkDevice)?;
    }

    if let Some(vsock_override) = &params.vsock_override {
        // There should only ever be at most one vsock device, therefore this
        // should correctly find it and modify the path if such a device exists.
        let device_state = match &mut microvm_state.device_states.virtio_state {
            VirtioDevicesState::Mmio(mmio_state) => mmio_state
                .vsock_device
                .as_mut()
                .map(|device| &mut device.device_state),
            VirtioDevicesState::Pci(pci_state) => pci_state
                .vsock_device
                .as_mut()
                .map(|device| &mut device.device_state),
        }
        .ok_or(SnapshotStateFromFileError::UnknownVsockDevice)?;

        device_state
            .backend
            .uds_path
            .clone_from(&vsock_override.uds_path);
    }

    let track_dirty_pages = params.track_dirty_pages;

    let vcpu_count = microvm_state
        .vcpu_states
        .len()
        .try_into()
        .map_err(|_| MachineConfigError::InvalidVcpuCount)
        .map_err(BuildMicrovmFromSnapshotError::VmUpdateConfig)?;

    // Due to questionable past API design decisions whether the restored VM
    // uses PCI is decided by the snapshot and not by the --enable-pci flag of
    // the process doing the restore. Set the option based on the snapshot
    // state.
    let pci_enabled = matches!(
        microvm_state.device_states.virtio_state,
        VirtioDevicesState::Pci(_)
    );
    if pci_enabled != vm_resources.pci_enabled {
        warn!(
            "The snapshot's PCI configuration does not match --enable-pci; \
             following the snapshot configuration."
        );
    }
    vm_resources.pci_enabled = pci_enabled;

    vm_resources
        .update_machine_config(&MachineConfigUpdate {
            vcpu_count: Some(vcpu_count),
            mem_size_mib: Some(u64_to_usize(microvm_state.vm_info.mem_size_mib)),
            smt: Some(microvm_state.vm_info.smt),
            cpu_template: Some(microvm_state.vm_info.cpu_template),
            track_dirty_pages: Some(track_dirty_pages),
            huge_pages: Some(params.huge_pages.resolve(microvm_state.vm_info.huge_pages)),
            #[cfg(feature = "gdb")]
            gdb_socket_path: None,
        })
        .map_err(BuildMicrovmFromSnapshotError::VmUpdateConfig)?;

    // Some sanity checks before building the microvm.
    snapshot_state_sanity_check(&microvm_state)?;

    let mem_backend_path = &params.mem_backend.backend_path;
    let mem_state = &microvm_state.vm_state.memory;

    let (guest_memory, uffd) = match params.mem_backend.backend_type {
        MemBackendType::File => {
            if vm_resources.machine_config.huge_pages.is_hugetlbfs() {
                return Err(RestoreFromSnapshotGuestMemoryError::File(
                    GuestMemoryFromFileError::HugetlbfsSnapshot,
                )
                .into());
            }
            (
                guest_memory_from_file(
                    mem_backend_path,
                    mem_state,
                    track_dirty_pages,
                    vm_resources.machine_config.huge_pages,
                )
                .map_err(RestoreFromSnapshotGuestMemoryError::File)?,
                None,
            )
        }
        MemBackendType::Uffd => guest_memory_from_uffd(
            mem_backend_path,
            mem_state,
            track_dirty_pages,
            vm_resources.machine_config.huge_pages,
        )
        .map_err(RestoreFromSnapshotGuestMemoryError::Uffd)?,
    };
    let vmm = builder::build_microvm_from_snapshot(
        instance_info,
        event_manager,
        &microvm_state,
        guest_memory,
        uffd,
        seccomp_filters,
        vm_resources,
        params.clock_realtime,
    )
    .map_err(RestoreFromSnapshotError::Build)?;

    // Reset needs the file mapping to refault snapshot pages, and dirty tracking to find
    // the pages that changed since load.
    if params.mem_backend.backend_type == MemBackendType::File && track_dirty_pages {
        let mut locked_vmm = vmm.lock().expect("Poisoned lock");
        let dirty_pages = locked_vmm
            .vm
            .as_kvm()
            .map(|vm| vm.guest_memory().clean_dirty_bitmap())
            .unwrap_or_default();
        let mmds_data = locked_vmm
            .get_mmds()
            .map(|mmds| mmds.lock().expect("Poisoned lock").save_data())
            .unwrap_or_default();
        locked_vmm.reset_context = Some(ResetContext {
            vm_state: microvm_state.vm_state,
            #[cfg(target_arch = "aarch64")]
            mpidrs: crate::construct_kvm_mpidrs(&microvm_state.vcpu_states),
            vcpu_states: microvm_state.vcpu_states,
            device_states: microvm_state.device_states,
            virtio_devices: locked_vmm.device_manager.virtio_device_activation(),
            pci_bar_addresses: locked_vmm.device_manager.pci_bar_addresses(),
            devices_hotplugged: false,
            mmds_data,
            clock_realtime: params.clock_realtime,
            dirty_pages,
            poisoned: false,
        });
    }
    Ok(vmm)
}

/// Error type for [`snapshot_state_from_file`]
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum SnapshotStateFromFileError {
    /// Failed to open snapshot file: {0}
    Open(#[from] std::io::Error),
    /// Failed to load snapshot state from file: {0}
    Load(#[from] crate::snapshot::SnapshotError),
    /// Unknown Network Device.
    UnknownNetworkDevice,
    /// Unknown Vsock Device.
    UnknownVsockDevice,
}

fn snapshot_state_from_file(
    snapshot_path: &Path,
) -> Result<MicrovmState, SnapshotStateFromFileError> {
    let mut snapshot_reader = File::open(snapshot_path)?;
    let snapshot = Snapshot::load(&mut snapshot_reader)?;

    Ok(snapshot.data)
}

/// Error type for [`guest_memory_from_file`].
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum GuestMemoryFromFileError {
    /// Failed to load guest memory: {0}
    File(#[from] std::io::Error),
    /// Failed to restore guest memory: {0}
    Restore(#[from] MemoryError),
    /// Cannot restore hugetlbfs backed snapshot by mapping the memory file. Please use uffd.
    HugetlbfsSnapshot,
}

fn guest_memory_from_file(
    mem_file_path: &Path,
    mem_state: &GuestMemoryState,
    track_dirty_pages: bool,
    huge_pages: HugePageConfig,
) -> Result<Vec<GuestRegionMmap>, GuestMemoryFromFileError> {
    let mem_file = File::open(mem_file_path)?;
    let guest_mem =
        memory::snapshot_file(mem_file, mem_state.regions(), track_dirty_pages, huge_pages)?;
    Ok(guest_mem)
}

/// Error type for [`guest_memory_from_uffd`]
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum GuestMemoryFromUffdError {
    /// Failed to restore guest memory: {0}
    Restore(#[from] MemoryError),
    /// Failed to UFFD object: {0}
    Create(userfaultfd::Error),
    /// Failed to register memory address range with the userfaultfd object: {0}
    Register(userfaultfd::Error),
    /// Failed to connect to UDS Unix stream: {0}
    Connect(#[from] std::io::Error),
    /// Failed to sends file descriptor: {0}
    Send(#[from] vmm_sys_util::errno::Error),
}

fn guest_memory_from_uffd(
    mem_uds_path: &Path,
    mem_state: &GuestMemoryState,
    track_dirty_pages: bool,
    huge_pages: HugePageConfig,
) -> Result<(Vec<GuestRegionMmap>, Option<Uffd>), GuestMemoryFromUffdError> {
    let (guest_memory, backend_mappings) =
        create_guest_memory(mem_state, track_dirty_pages, huge_pages)?;

    let mut uffd_builder = UffdBuilder::new();

    // We only make use of this if balloon devices are present, but we can enable it unconditionally
    // because the only place the kernel checks this is in a hook from madvise, e.g. it doesn't
    // actively change the behavior of UFFD, only passively. Without balloon devices
    // we never call madvise anyway, so no need to put this into a conditional.
    uffd_builder.require_features(FeatureFlags::EVENT_REMOVE);

    let uffd = uffd_builder
        .close_on_exec(true)
        .non_blocking(true)
        .user_mode_only(false)
        .create()
        .map_err(GuestMemoryFromUffdError::Create)?;

    for mem_region in guest_memory.iter() {
        uffd.register(mem_region.as_ptr().cast(), mem_region.size() as _)
            .map_err(GuestMemoryFromUffdError::Register)?;
    }

    send_uffd_handshake(mem_uds_path, &backend_mappings, &uffd)?;

    Ok((guest_memory, Some(uffd)))
}

fn create_guest_memory(
    mem_state: &GuestMemoryState,
    track_dirty_pages: bool,
    huge_pages: HugePageConfig,
) -> Result<(Vec<GuestRegionMmap>, Vec<GuestRegionUffdMapping>), GuestMemoryFromUffdError> {
    let guest_memory = memory::anonymous(mem_state.regions(), track_dirty_pages, huge_pages)?;
    let mut backend_mappings = Vec::with_capacity(guest_memory.len());
    let mut offset = 0;
    for mem_region in guest_memory.iter() {
        #[allow(deprecated)]
        backend_mappings.push(GuestRegionUffdMapping {
            base_host_virt_addr: mem_region.as_ptr() as u64,
            size: mem_region.size(),
            offset,
            page_size: huge_pages.page_size(),
            page_size_kib: huge_pages.page_size(),
        });
        offset += mem_region.size() as u64;
    }

    Ok((guest_memory, backend_mappings))
}

fn send_uffd_handshake(
    mem_uds_path: &Path,
    backend_mappings: &[GuestRegionUffdMapping],
    uffd: &impl AsRawFd,
) -> Result<(), GuestMemoryFromUffdError> {
    // This is safe to unwrap() because we control the contents of the vector
    // (i.e GuestRegionUffdMapping entries).
    let backend_mappings = serde_json::to_string(backend_mappings).unwrap();

    let socket = UnixStream::connect(mem_uds_path)?;
    socket.send_with_fd(
        backend_mappings.as_bytes(),
        // In the happy case we can close the fd since the other process has it open and is
        // using it to serve us pages.
        //
        // The problem is that if other process crashes/exits, firecracker guest memory
        // will simply revert to anon-mem behavior which would lead to silent errors and
        // undefined behavior.
        //
        // To tackle this scenario, the page fault handler can notify Firecracker of any
        // crashes/exits. There is no need for Firecracker to explicitly send its process ID.
        // The external process can obtain Firecracker's PID by calling `getsockopt` with
        // `libc::SO_PEERCRED` option like so:
        //
        // let mut val = libc::ucred { pid: 0, gid: 0, uid: 0 };
        // let mut ucred_size: u32 = mem::size_of::<libc::ucred>() as u32;
        // libc::getsockopt(
        //      socket.as_raw_fd(),
        //      libc::SOL_SOCKET,
        //      libc::SO_PEERCRED,
        //      &mut val as *mut _ as *mut _,
        //      &mut ucred_size as *mut libc::socklen_t,
        // );
        //
        // Per this linux man page: https://man7.org/linux/man-pages/man7/unix.7.html,
        // `SO_PEERCRED` returns the credentials (PID, UID and GID) of the peer process
        // connected to this socket. The returned credentials are those that were in effect
        // at the time of the `connect` call.
        //
        // Moreover, Firecracker holds a copy of the UFFD fd as well, so that even if the
        // page fault handler process does not tear down Firecracker when necessary, the
        // uffd will still be alive but with no one to serve faults, leading to guest freeze.
        uffd.as_raw_fd(),
    )?;

    // We prevent Rust from closing the socket file descriptor to avoid a potential race condition
    // between the mappings message and the connection shutdown. If the latter arrives at the UFFD
    // handler first, the handler never sees the mappings.
    forget(socket);

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixListener;

    use vmm_sys_util::tempfile::TempFile;

    use super::*;
    use crate::Vmm;
    #[cfg(target_arch = "x86_64")]
    use crate::builder::tests::insert_vmclock_device;
    #[cfg(target_arch = "x86_64")]
    use crate::builder::tests::insert_vmgenid_device;
    use crate::builder::tests::{
        CustomBlockConfig, default_kernel_cmdline, default_vmm, insert_balloon_device,
        insert_block_devices, insert_net_device, insert_vsock_device,
    };
    #[cfg(target_arch = "aarch64")]
    use crate::construct_kvm_mpidrs;
    use crate::devices::virtio::block::CacheType;
    use crate::snapshot::Persist;
    use crate::vmm_config::balloon::BalloonDeviceConfig;
    use crate::vmm_config::net::NetworkInterfaceConfig;
    use crate::vmm_config::vsock::tests::default_config;
    use crate::vstate::memory::{GuestMemoryRegionState, GuestRegionType};

    fn default_vmm_with_devices() -> Vmm {
        let mut event_manager = EventManager::new().expect("Cannot create EventManager");
        let mut vmm = default_vmm();
        let mut cmdline = default_kernel_cmdline();

        // Add a balloon device.
        let balloon_config = BalloonDeviceConfig {
            amount_mib: 0,
            deflate_on_oom: false,
            stats_polling_interval_s: 0,
            free_page_hinting: false,
            free_page_reporting: false,
        };
        insert_balloon_device(&mut vmm, &mut cmdline, &mut event_manager, balloon_config);

        // Add a block device.
        let drive_id = String::from("root");
        let block_configs = vec![CustomBlockConfig::new(
            drive_id,
            true,
            None,
            true,
            CacheType::Unsafe,
        )];
        insert_block_devices(&mut vmm, &mut cmdline, &mut event_manager, block_configs);

        // Add net device.
        let network_interface = NetworkInterfaceConfig {
            iface_id: String::from("netif"),
            host_dev_name: String::from("hostname"),
            guest_mac: None,
            mtu: None,
            rx_rate_limiter: None,
            tx_rate_limiter: None,
        };
        insert_net_device(
            &mut vmm,
            &mut cmdline,
            &mut event_manager,
            network_interface,
        );

        // Add vsock device.
        let mut tmp_sock_file = TempFile::new().unwrap();
        tmp_sock_file.remove().unwrap();
        let vsock_config = default_config(&tmp_sock_file);

        insert_vsock_device(&mut vmm, &mut cmdline, &mut event_manager, vsock_config);

        #[cfg(target_arch = "x86_64")]
        insert_vmgenid_device(&mut vmm);
        #[cfg(target_arch = "x86_64")]
        insert_vmclock_device(&mut vmm);

        vmm
    }

    fn snapshot_load_fixture(track_dirty_pages: bool) -> (Arc<Mutex<Vmm>>, EventManager) {
        snapshot_load_fixture_with(track_dirty_pages, false)
    }

    /// Loads a Full snapshot of an unbooted microVM that has no virtio devices.
    fn snapshot_load_fixture_with(
        track_dirty_pages: bool,
        pci_enabled: bool,
    ) -> (Arc<Mutex<Vmm>>, EventManager) {
        use crate::vmm_config::balloon::BalloonBuilder;

        let mut resources = fixture_resources(pci_enabled);
        // Test builds give the resources a balloon by default, and reset rejects balloons.
        resources.balloon = BalloonBuilder::new();
        snapshot_load_fixture_from(&resources, track_dirty_pages)
    }

    fn fixture_resources(pci_enabled: bool) -> VmResources {
        use crate::test_utils::mock_resources::{MockBootSourceConfig, MockVmResources};

        let boot_source = MockBootSourceConfig::new().with_default_boot_args().into();
        let mut resources: VmResources =
            MockVmResources::new().with_boot_source(boot_source).into();
        resources.pci_enabled = pci_enabled;
        resources
    }

    /// Loads a Full snapshot of an unbooted microVM built from `resources`.
    fn snapshot_load_fixture_from(
        resources: &VmResources,
        track_dirty_pages: bool,
    ) -> (Arc<Mutex<Vmm>>, EventManager) {
        use crate::builder::build_microvm_for_boot;
        use crate::seccomp::get_empty_filters;
        use crate::vmm_config::snapshot::{MemBackendConfig, SnapshotType};

        let mut source_events = EventManager::new().unwrap();
        let source = build_microvm_for_boot(
            &InstanceInfo::default(),
            resources,
            &mut source_events,
            &get_empty_filters(),
        )
        .unwrap();
        let state_file = TempFile::new().unwrap();
        let memory_file = TempFile::new().unwrap();
        {
            let mut source = source.lock().unwrap();
            let info = VmInfo::from(&*source);
            create_snapshot(
                &mut source,
                &info,
                &CreateSnapshotParams {
                    snapshot_type: SnapshotType::Full,
                    snapshot_path: state_file.as_path().to_path_buf(),
                    mem_file_path: memory_file.as_path().to_path_buf(),
                    sync_snapshot_files: false,
                },
            )
            .unwrap();
            source.stop(crate::FcExitCode::Ok);
        }

        let mut event_manager = EventManager::new().unwrap();
        let vmm = restore_from_snapshot(
            &InstanceInfo::default(),
            &mut event_manager,
            &get_empty_filters(),
            &LoadSnapshotParams {
                snapshot_path: state_file.as_path().to_path_buf(),
                mem_backend: MemBackendConfig {
                    backend_type: MemBackendType::File,
                    backend_path: memory_file.as_path().to_path_buf(),
                },
                track_dirty_pages,
                resume_vm: false,
                network_overrides: Vec::new(),
                vsock_override: None,
                clock_realtime: false,
                huge_pages: Default::default(),
            },
            &mut VmResources::default(),
        )
        .unwrap();
        (vmm, event_manager)
    }

    /// Returns a guest RAM address that the boot setup of the fixture leaves unused.
    fn scratch_address(memory: &crate::vstate::memory::GuestMemoryMmap) -> vm_memory::GuestAddress {
        use vm_memory::{Address, GuestMemoryBackend, GuestMemoryRegion};

        // 64 MiB into RAM lies past the kernel and boot data on both architectures.
        let ram_start = memory.iter().next().unwrap().start_addr();
        ram_start.unchecked_add(crate::utils::usize_to_u64(crate::utils::mib_to_bytes(64)))
    }

    #[test]
    fn test_snapshot_load_reset_eligibility() {
        for track_dirty_pages in [false, true] {
            let (vmm, _events) = snapshot_load_fixture(track_dirty_pages);
            assert_eq!(
                vmm.lock().unwrap().reset_context.is_some(),
                track_dirty_pages
            );
        }
    }

    #[test]
    fn test_reset_restores_loaded_state_repeatedly() {
        use vm_memory::{ByteValued, Bytes};

        use crate::snapshot::Persist;
        use crate::vmm_config::snapshot::SnapshotType;
        #[cfg(target_arch = "x86_64")]
        use crate::vstate::bus::BusDevice;

        let (vmm, _events) = snapshot_load_fixture(true);
        let mut vmm = vmm.lock().unwrap();
        let kvm = vmm.vm.as_kvm().unwrap().clone();
        let vm_fd = kvm.fd().as_raw_fd();
        #[cfg(target_arch = "x86_64")]
        let baseline_regs = vmm.reset_context.as_ref().unwrap().vcpu_states[0].regs;
        #[cfg(target_arch = "aarch64")]
        let baseline_pc = vmm.reset_context.as_ref().unwrap().vcpu_states[0]
            .regs
            .iter()
            .find(|reg| reg.id == crate::arch::aarch64::regs::PC)
            .unwrap()
            .value::<u64, 8>();
        // ARM has no serial device here, because the fixture's boot arguments have no console.
        #[cfg(target_arch = "x86_64")]
        let serial = vmm
            .device_manager
            .legacy_devices
            .as_ref()
            .unwrap()
            .stdio_serial
            .clone();
        let dirty_address = scratch_address(kvm.guest_memory());
        let baseline: u8 = kvm.guest_memory().read_obj(dirty_address).unwrap();
        let vmclock = |vmm: &Vmm| vmm.device_manager.acpi_devices.vmclock().save().inner;
        let initial_generation = vmclock(&vmm).vm_generation_counter;

        for generation in 1..=2 {
            kvm.guest_memory()
                .write_slice(&[!baseline], dirty_address)
                .unwrap();
            #[cfg(target_arch = "x86_64")]
            kvm.vcpus_handles()[0]
                .vcpu_fd
                .set_regs(&kvm_bindings::kvm_regs {
                    rax: 0x1234,
                    ..baseline_regs
                })
                .unwrap();
            #[cfg(target_arch = "aarch64")]
            kvm.vcpus_handles()[0]
                .vcpu_fd
                .set_one_reg(
                    crate::arch::aarch64::regs::PC,
                    &baseline_pc.wrapping_add(generation * 4).to_le_bytes(),
                )
                .unwrap();
            let old_genid = vmm.device_manager.acpi_devices.vmgenid().gen_id;
            #[cfg(target_arch = "x86_64")]
            {
                serial.lock().unwrap().write(0, 7, &[0xa5]);
                let legacy = vmm.device_manager.legacy_devices.as_ref().unwrap();
                let mut i8042 = legacy.i8042.lock().unwrap();
                i8042.write(0, 4, &[0x60]);
                i8042.write(0, 0, &[0]);
            }
            #[cfg(target_arch = "aarch64")]
            vmm.device_manager
                .mmio_platform_devices
                .rtc
                .as_ref()
                .unwrap()
                .inner
                .lock()
                .unwrap()
                .bus_write(0x008, &123_u32.to_le_bytes());

            // Snapshot creation clears current dirty logs but must retain reset dirt.
            let snapshot = TempFile::new().unwrap();
            let memory = TempFile::new().unwrap();
            let info = VmInfo::from(&*vmm);
            create_snapshot(
                &mut vmm,
                &info,
                &CreateSnapshotParams {
                    snapshot_type: SnapshotType::Diff,
                    snapshot_path: snapshot.as_path().to_path_buf(),
                    mem_file_path: memory.as_path().to_path_buf(),
                    sync_snapshot_files: false,
                },
            )
            .unwrap();
            reset_to_snapshot(&mut vmm).unwrap();

            assert_eq!(
                kvm.guest_memory().read_obj::<u8>(dirty_address).unwrap(),
                baseline
            );
            #[cfg(target_arch = "x86_64")]
            assert_eq!(
                kvm.vcpus_handles()[0].vcpu_fd.get_regs().unwrap(),
                baseline_regs
            );
            #[cfg(target_arch = "aarch64")]
            {
                let mut pc = [0_u8; 8];
                kvm.vcpus_handles()[0]
                    .vcpu_fd
                    .get_one_reg(crate::arch::aarch64::regs::PC, &mut pc)
                    .unwrap();
                assert_eq!(u64::from_le_bytes(pc), baseline_pc);
                let mut rtc_load = [0_u8; 4];
                vmm.device_manager
                    .mmio_platform_devices
                    .rtc
                    .as_ref()
                    .unwrap()
                    .inner
                    .lock()
                    .unwrap()
                    .bus_read(0x008, &mut rtc_load);
                assert_eq!(u32::from_le_bytes(rtc_load), 0);
            }
            assert_eq!(vmm.vm.as_kvm().unwrap().fd().as_raw_fd(), vm_fd);
            #[cfg(target_arch = "x86_64")]
            {
                let mut scratch = [0];
                serial.lock().unwrap().read(0, 7, &mut scratch);
                assert_eq!(scratch, [0]);
                vmm.send_ctrl_alt_del().unwrap();
            }
            let acpi = &vmm.device_manager.acpi_devices;
            let genid: u128 = kvm
                .guest_memory()
                .read_obj(acpi.vmgenid().guest_address)
                .unwrap();
            assert_ne!(genid, old_genid);
            assert_eq!(genid, acpi.vmgenid().gen_id);
            // The guest sees the whole VMClock page again, with one more generation.
            let clock = vmclock(&vmm);
            let mut page = vec![0; clock.as_slice().len()];
            kvm.guest_memory()
                .read_slice(&mut page, acpi.vmclock().guest_address)
                .unwrap();
            assert_eq!(page, clock.as_slice());
            assert_eq!(clock.vm_generation_counter, initial_generation + generation);
            assert_eq!(vmm.instance_info.state, InstanceState::Paused);
            assert!(!vmm.reset_poisoned());
            let dirty_since_load = &vmm.reset_context.as_ref().unwrap().dirty_pages;
            assert!(dirty_since_load.values().flatten().all(|&bits| bits == 0));
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_reset_reverts_ram_written_by_pending_mmio() {
        use kvm_ioctls::VcpuExit;
        use vm_memory::{Address, Bytes, GuestMemoryBackend};

        let (vmm, _events) = snapshot_load_fixture(true);
        let mut vmm = vmm.lock().unwrap();
        let kvm = vmm.vm.as_kvm().unwrap().clone();
        let code = scratch_address(kvm.guest_memory());
        let destination = code.unchecked_add(0x1000);
        // The boot setup identity maps the first 1 GiB, and RAM ends below that, so a read
        // just past RAM exits to the VMM as MMIO.
        let source = kvm.guest_memory().last_addr().unchecked_add(1);
        let baseline: u8 = kvm.guest_memory().read_obj(destination).unwrap();

        for _ in 0..2 {
            // movsb reads from RSI at an unmapped address and writes to RDI in RAM.
            // Its MMIO completion can write RAM even when immediate_exit prevents entry.
            kvm.guest_memory().write_slice(&[0xa4], code).unwrap();
            {
                let mut handles = kvm.vcpus_handles();
                let fd = &mut handles[0].vcpu_fd;
                let mut regs = fd.get_regs().unwrap();
                regs.rip = code.raw_value();
                regs.rsi = source.raw_value();
                regs.rdi = destination.raw_value();
                regs.rflags = 2;
                fd.set_regs(&regs).unwrap();
                fd.set_kvm_immediate_exit(0);
                match fd.run().unwrap() {
                    VcpuExit::MmioRead(address, data) => {
                        assert_eq!(address, source.raw_value());
                        assert_eq!(data.len(), 1);
                        data[0] = !baseline;
                    }
                    exit => panic!("Expected movsb MMIO read, got {exit:?}"),
                }
                // Leave the handled exit pending, as a Pause arriving before re-entry can.
                fd.set_kvm_immediate_exit(1);
            }
            assert_eq!(
                kvm.guest_memory().read_obj::<u8>(destination).unwrap(),
                baseline
            );

            reset_to_snapshot(&mut vmm).unwrap();

            assert_eq!(
                kvm.guest_memory().read_obj::<u8>(destination).unwrap(),
                baseline
            );
            assert_eq!(vmm.instance_info.state, InstanceState::Paused);
            assert!(!vmm.reset_poisoned());
        }
    }

    #[test]
    fn test_reset_prerequisites() {
        let mut vmm = default_vmm();
        assert!(matches!(
            reset_to_snapshot(&mut vmm),
            Err(ResetSnapshotError::NotPaused)
        ));
        vmm.instance_info.state = InstanceState::Paused;
        assert!(matches!(
            reset_to_snapshot(&mut vmm),
            Err(ResetSnapshotError::NoResetContext)
        ));
    }

    #[test]
    fn test_reset_rejects_device_changes_before_poison() {
        use crate::builder::tests::insert_entropy_device;

        // A device added after load.
        let (vmm, mut events) = snapshot_load_fixture(true);
        let mut vmm = vmm.lock().unwrap();
        insert_entropy_device(
            &mut vmm,
            &mut default_kernel_cmdline(),
            &mut events,
            Default::default(),
        );
        assert!(matches!(
            reset_to_snapshot(&mut vmm),
            Err(ResetSnapshotError::DevicesChanged)
        ));
        assert!(!vmm.reset_poisoned());

        // A snapshot-loaded device that reset cannot restore in place. The test build's
        // default resources include a balloon.
        let (vmm, _events) = snapshot_load_fixture_from(&fixture_resources(false), true);
        let mut vmm = vmm.lock().unwrap();
        assert!(matches!(
            reset_to_snapshot(&mut vmm),
            Err(ResetSnapshotError::Unsupported("balloon devices"))
        ));
        assert!(!vmm.reset_poisoned());
    }

    #[test]
    fn test_reset_rejects_hotplug_before_poison() {
        use crate::device_manager::tests::make_hotplug_block_cfg;
        use crate::devices::virtio::device::VirtioDeviceType;
        use crate::vmm_config::HotplugDeviceConfig;

        let (vmm, mut events) = snapshot_load_fixture_with(true, true);
        let mut vmm = vmm.lock().unwrap();
        let disk = TempFile::new().unwrap();
        // The PCI microVM resets while its devices are the ones that load created.
        reset_to_snapshot(&mut vmm).unwrap();

        // After a device is plugged in and unplugged again, the devices look as after load. A
        // device with the same ID could still be plugged in, which load did not create.
        let config = make_hotplug_block_cfg("scratch", &disk, false);
        vmm.hotplug_device(HotplugDeviceConfig::Block(config), &mut events)
            .unwrap();
        vmm.hot_unplug_device(
            (VirtioDeviceType::Block, "scratch".to_string()),
            &mut events,
        )
        .unwrap();
        assert!(matches!(
            reset_to_snapshot(&mut vmm),
            Err(ResetSnapshotError::DevicesChanged)
        ));
        assert!(!vmm.reset_poisoned());
    }

    #[test]
    fn test_reset_failure_poison_blocks_state_operations() {
        use vm_memory::Bytes;

        let (vmm, _events) = snapshot_load_fixture(true);
        let mut vmm = vmm.lock().unwrap();
        let kvm = vmm.vm.as_kvm().unwrap().clone();
        let dirty_address = scratch_address(kvm.guest_memory());
        let baseline: u8 = kvm.guest_memory().read_obj(dirty_address).unwrap();
        kvm.guest_memory()
            .write_slice(&[!baseline], dirty_address)
            .unwrap();
        let vcpu_state = &mut vmm.reset_context.as_mut().unwrap().vcpu_states[0];
        // Make the vCPU restore fail after reset has begun. x86 KVM checks the MP state only
        // since Linux 6.0, but every supported kernel rejects an XCR0 without x87 state.
        #[cfg(target_arch = "x86_64")]
        {
            vcpu_state.xcrs.nr_xcrs = 1;
            vcpu_state.xcrs.xcrs[0].xcr = 0;
            vcpu_state.xcrs.xcrs[0].value = 0;
        }
        #[cfg(target_arch = "aarch64")]
        {
            vcpu_state.mp_state.mp_state = u32::MAX;
        }
        assert!(matches!(
            reset_to_snapshot(&mut vmm),
            Err(ResetSnapshotError::RestoreVcpus(_))
        ));
        assert_eq!(
            kvm.guest_memory().read_obj::<u8>(dirty_address).unwrap(),
            baseline
        );
        assert!(vmm.reset_poisoned());
        assert!(matches!(vmm.resume_vm(), Err(crate::VmmError::ResetFailed)));
        assert!(matches!(
            reset_to_snapshot(&mut vmm),
            Err(ResetSnapshotError::Poisoned)
        ));

        let directory = vmm_sys_util::tempdir::TempDir::new().unwrap();
        let params = CreateSnapshotParams {
            snapshot_type: crate::vmm_config::snapshot::SnapshotType::Full,
            snapshot_path: directory.as_path().join("vmstate"),
            mem_file_path: directory.as_path().join("memory"),
            sync_snapshot_files: false,
        };
        assert!(matches!(
            create_snapshot(&mut vmm, &VmInfo::default(), &params),
            Err(CreateSnapshotError::ResetPoisoned)
        ));
        assert!(!params.snapshot_path.exists());
        assert!(!params.mem_file_path.exists());
    }

    #[test]
    fn test_microvm_state_snapshot() {
        let vmm = default_vmm_with_devices();
        let states = vmm.device_manager.save();

        // Only checking that all devices are saved, actual device state
        // is tested by that device's tests.
        let VirtioDevicesState::Mmio(mmio_state) = &states.virtio_state else {
            panic!("expected MMIO virtio device state");
        };
        assert_eq!(mmio_state.block_devices.len(), 1);
        assert_eq!(mmio_state.net_devices.len(), 1);
        assert!(mmio_state.vsock_device.is_some());
        assert!(mmio_state.balloon_device.is_some());

        let vcpu_states = vec![VcpuState::default()];
        #[cfg(target_arch = "aarch64")]
        let mpidrs = construct_kvm_mpidrs(&vcpu_states);
        let microvm_state = MicrovmState {
            device_states: states,
            vcpu_states,
            kvm_state: Default::default(),
            vm_info: VmInfo {
                mem_size_mib: 1u64,
                ..Default::default()
            },
            #[cfg(target_arch = "aarch64")]
            vm_state: vmm.vm.as_kvm().unwrap().save_state(&mpidrs).unwrap(),
            #[cfg(target_arch = "x86_64")]
            vm_state: vmm.vm.as_kvm().unwrap().save_state().unwrap(),
        };

        let serialized_data = bitcode::serialize(&microvm_state).unwrap();

        let restored_microvm_state: MicrovmState = bitcode::deserialize(&serialized_data).unwrap();

        assert_eq!(restored_microvm_state.vm_info, microvm_state.vm_info);
        let (VirtioDevicesState::Mmio(restored_mmio), VirtioDevicesState::Mmio(mmio)) = (
            &restored_microvm_state.device_states.virtio_state,
            &microvm_state.device_states.virtio_state,
        ) else {
            panic!("expected MMIO virtio device state");
        };
        assert_eq!(restored_mmio, mmio)
    }

    #[test]
    fn test_create_guest_memory() {
        let mem_state = GuestMemoryState {
            regions: vec![GuestMemoryRegionState {
                base_address: 0,
                size: 0x20000,
                region_type: GuestRegionType::Dram,
                plugged: vec![true],
            }],
        };

        let (_, uffd_regions) =
            create_guest_memory(&mem_state, false, HugePageConfig::None).unwrap();

        assert_eq!(uffd_regions.len(), 1);
        assert_eq!(uffd_regions[0].size, 0x20000);
        assert_eq!(uffd_regions[0].offset, 0);
        assert_eq!(uffd_regions[0].page_size, HugePageConfig::None.page_size());
    }

    #[test]
    fn test_send_uffd_handshake() {
        #[allow(deprecated)]
        let uffd_regions = vec![
            GuestRegionUffdMapping {
                base_host_virt_addr: 0,
                size: 0x100000,
                offset: 0,
                page_size: HugePageConfig::None.page_size(),
                page_size_kib: HugePageConfig::None.page_size(),
            },
            GuestRegionUffdMapping {
                base_host_virt_addr: 0x100000,
                size: 0x200000,
                offset: 0,
                page_size: HugePageConfig::Hugetlbfs2M.page_size(),
                page_size_kib: HugePageConfig::Hugetlbfs2M.page_size(),
            },
        ];

        let uds_path = TempFile::new().unwrap();
        let uds_path = uds_path.as_path();
        std::fs::remove_file(uds_path).unwrap();

        let listener = UnixListener::bind(uds_path).expect("Cannot bind to socket path");

        send_uffd_handshake(uds_path, &uffd_regions, &std::io::stdin()).unwrap();

        let (stream, _) = listener.accept().expect("Cannot listen on UDS socket");

        let mut message_buf = vec![0u8; 1024];
        let (bytes_read, _) = stream
            .recv_with_fd(&mut message_buf[..])
            .expect("Cannot recv_with_fd");
        message_buf.resize(bytes_read, 0);

        let deserialized: Vec<GuestRegionUffdMapping> =
            serde_json::from_slice(&message_buf).unwrap();

        assert_eq!(uffd_regions, deserialized);
    }
}
