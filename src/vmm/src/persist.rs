// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines state structures for saving/restoring a Firecracker microVM.

use std::fmt::Debug;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Write};
use std::mem::forget;
use std::os::unix::io::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use semver::Version;
use serde::{Deserialize, Serialize};
use userfaultfd::{FeatureFlags, RegisterMode, Uffd, UffdBuilder};
use vmm_sys_util::sock_ctrl_msg::ScmSocket;

use crate::arch::ArchVmError;
#[cfg(target_arch = "aarch64")]
use crate::arch::aarch64::vcpu::get_manufacturer_id_from_host;
use crate::builder::{self, BuildMicrovmFromSnapshotError};
use crate::cpu_config::templates::StaticCpuTemplate;
#[cfg(target_arch = "x86_64")]
use crate::cpu_config::x86_64::cpuid::CpuidTrait;
#[cfg(target_arch = "x86_64")]
use crate::cpu_config::x86_64::cpuid::common::get_vendor_id_from_host;
use crate::device_manager::persist::{
    ACPIDeviceManagerState, DevicePersistError, DeviceStates,
};
use crate::io_uring::IoUring;
use crate::io_uring::operation::{OpCode, Operation};
use crate::io_uring::restriction::Restriction;
use crate::logger::{info, warn};
use crate::resources::VmResources;
use crate::seccomp::BpfThreadMap;
use crate::snapshot::Snapshot;
use crate::utils::u64_to_usize;
use crate::vmm_config::boot_source::BootSourceConfig;
use crate::vmm_config::instance_info::InstanceInfo;
use crate::vmm_config::machine_config::{HugePageConfig, MachineConfigError, MachineConfigUpdate};
use crate::vmm_config::snapshot::{
    Checkpoint, CreateSnapshotParams, LoadSnapshotParams, MemBackendType, ResetSnapshotParams,
};
use crate::vstate::kvm::KvmState;
use crate::vstate::memory;
use crate::vstate::memory::{GuestMemoryState, GuestRegionMmap, MemoryError};
use crate::vstate::vcpu::{VcpuSendEventError, VcpuState};
use crate::vstate::vm::{VmError, VmState};
use crate::{EventManager, Vmm, vstate};

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

/// Contains the necesary state for saving/restoring a microVM.
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
    pub device_states: DeviceStates,
    /// ACPI devices state.
    pub acpi_dev_state: ACPIDeviceManagerState,
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
    /// The number of Vcpu states provided does not match the number of Vcpus present in the VMM.
    IncorrectVcpuStateCount,
    /// Operation not allowed: {0}
    NotAllowed(String),
    /// Cannot restore devices: {0}
    RestoreDevices(DevicePersistError),
    /// Cannot restore Vcpu state: {0}
    RestoreVcpuState(vstate::vcpu::VcpuError),
    /// Cannot save Vcpu state: {0}
    SaveVcpuState(vstate::vcpu::VcpuError),
    /// Cannot save Vm state: {0}
    SaveVmState(vstate::vm::ArchVmError),
    /// Cannot signal Vcpu: {0}
    SignalVcpu(VcpuSendEventError),
    /// Vcpu is in unexpected state.
    UnexpectedVcpuResponse,
}

/// Errors associated with creating a snapshot.
#[rustfmt::skip]
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum CreateSnapshotError {
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

/// Snapshot version
pub const SNAPSHOT_VERSION: Version = Version::new(8, 0, 0);

/// Creates a Microvm snapshot.
pub fn create_snapshot(
    vmm: &mut Vmm,
    vm_info: &VmInfo,
    params: &CreateSnapshotParams,
) -> Result<(), CreateSnapshotError> {
    let microvm_state = vmm
        .save_state(vm_info)
        .map_err(CreateSnapshotError::MicrovmState)?;

    snapshot_state_to_file(&microvm_state, &params.snapshot_path)?;

    vmm.vm
        .snapshot_memory_to_file(&params.mem_file_path, params.snapshot_type)?;

    // We need to mark queues as dirty again for all activated devices. The reason we
    // do it here is that we don't mark pages as dirty during runtime
    // for queue objects.
    // SAFETY:
    // This should never fail as we only mark pages only if device has already been activated,
    // and the address validation was already performed on device activation.
    vmm.mmio_device_manager
        .for_each_virtio_device(|_, _, _, dev| {
            let mut d = dev.lock().unwrap();
            if d.is_activated() {
                d.mark_queue_memory_dirty(vmm.vm.guest_memory())
            } else {
                Ok(())
            }
        })
        .unwrap();

    Ok(())
}

fn snapshot_state_to_file(
    microvm_state: &MicrovmState,
    snapshot_path: &Path,
) -> Result<(), CreateSnapshotError> {
    use self::CreateSnapshotError::*;
    let mut snapshot_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(snapshot_path)
        .map_err(|err| SnapshotBackingFile("open", err))?;

    let snapshot = Snapshot::new(SNAPSHOT_VERSION);
    snapshot.save(&mut snapshot_file, microvm_state)?;
    snapshot_file
        .flush()
        .map_err(|err| SnapshotBackingFile("flush", err))?;
    snapshot_file
        .sync_all()
        .map_err(|err| SnapshotBackingFile("sync_all", err))
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
}

/// Performs sanity checks against the state file and returns specific errors.
pub fn snapshot_state_sanity_check(
    microvm_state: &MicrovmState,
) -> Result<(), SnapShotStateSanityCheckError> {
    // Check if the snapshot contains at least 1 mem region.
    // Upper bound check will be done when creating guest memory by comparing against
    // KVM max supported value kvm_context.max_memslots().
    if microvm_state.vm_state.memory.regions.is_empty() {
        return Err(SnapShotStateSanityCheckError::NoMemory);
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
        let net_devices = &mut microvm_state.device_states.net_devices;
        if let Some(device) = net_devices
            .iter_mut()
            .find(|x| x.device_state.id == entry.iface_id)
        {
            device
                .device_state
                .tap_if_name
                .clone_from(&entry.host_dev_name);
        } else {
            return Err(SnapshotStateFromFileError::UnknownNetworkDevice.into());
        }
    }
    let track_dirty_pages = params.track_dirty_pages;

    let checkpoint = match params.enable_write_protection {
        true => {
            // Arbitrary 4096 ring slots
            let ring_size = 1 << 12;

            let io_uring = IoUring::<u64>::new(
                ring_size,
                vec![],
                // Madvise value is irrelevant. All madvise behaviour will be allowed.
                vec![Restriction::AllowOpCode(OpCode::Madvise(0))],
                None,
            )
            .map_err(BuildMicrovmFromSnapshotError::CreateResetIoUring)?;

            Some(Checkpoint {
                current_snapshot_path: params.snapshot_path.clone(),
                madvise_ring: io_uring,
            })
        }
        false => None,
    };

    let vcpu_count = microvm_state
        .vcpu_states
        .len()
        .try_into()
        .map_err(|_| MachineConfigError::InvalidVcpuCount)
        .map_err(BuildMicrovmFromSnapshotError::VmUpdateConfig)?;

    vm_resources
        .update_machine_config(&MachineConfigUpdate {
            vcpu_count: Some(vcpu_count),
            mem_size_mib: Some(u64_to_usize(microvm_state.vm_info.mem_size_mib)),
            smt: Some(microvm_state.vm_info.smt),
            cpu_template: Some(microvm_state.vm_info.cpu_template),
            track_dirty_pages: Some(track_dirty_pages),
            huge_pages: Some(microvm_state.vm_info.huge_pages),
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
                guest_memory_from_file(mem_backend_path, mem_state, track_dirty_pages)
                    .map_err(RestoreFromSnapshotGuestMemoryError::File)?,
                None,
            )
        }
        MemBackendType::Uffd => guest_memory_from_uffd(
            mem_backend_path,
            mem_state,
            track_dirty_pages,
            params.enable_write_protection,
            vm_resources.machine_config.huge_pages,
        )
        .map_err(RestoreFromSnapshotGuestMemoryError::Uffd)?,
    };
    builder::build_microvm_from_snapshot(
        instance_info,
        event_manager,
        microvm_state,
        guest_memory,
        uffd,
        seccomp_filters,
        vm_resources,
        checkpoint,
    )
    .map_err(RestoreFromSnapshotError::Build)
}

/// Error type for [`snapshot_state_from_file`]
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum SnapshotStateFromFileError {
    /// Failed to open snapshot file: {0}
    Open(std::io::Error),
    /// Failed to read snapshot file metadata: {0}
    Meta(std::io::Error),
    /// Failed to load snapshot state from file: {0}
    Load(#[from] crate::snapshot::SnapshotError),
    /// Unknown Network Device.
    UnknownNetworkDevice,
}

fn snapshot_state_from_file(
    snapshot_path: &Path,
) -> Result<MicrovmState, SnapshotStateFromFileError> {
    let snapshot = Snapshot::new(SNAPSHOT_VERSION);
    let mut snapshot_reader =
        File::open(snapshot_path).map_err(SnapshotStateFromFileError::Open)?;
    let metadata = std::fs::metadata(snapshot_path).map_err(SnapshotStateFromFileError::Meta)?;
    let snapshot_len = u64_to_usize(metadata.len());
    let state: MicrovmState = snapshot
        .load_with_version_check(&mut snapshot_reader, snapshot_len)
        .map_err(SnapshotStateFromFileError::Load)?;
    Ok(state)
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
) -> Result<Vec<GuestRegionMmap>, GuestMemoryFromFileError> {
    let mem_file = File::open(mem_file_path)?;
    let guest_mem = memory::snapshot_file(mem_file, mem_state.regions(), track_dirty_pages)?;
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
    enable_write_protection: bool,
    huge_pages: HugePageConfig,
) -> Result<(Vec<GuestRegionMmap>, Option<Uffd>), GuestMemoryFromUffdError> {
    let (guest_memory, backend_mappings) =
        create_guest_memory(mem_state, track_dirty_pages, huge_pages)?;

    let mut uffd_builder = UffdBuilder::new();

    // We only make use `EVENT_REMOVE` if balloon devices are present, but we can enable it
    // unconditionally because the only place the kernel checks this is in a hook from madvise,
    // e.g. it doesn't actively change the behavior of UFFD, only passively. Without balloon
    // devices we never call madvise anyway, so no need to put this into a conditional.
    //
    // If write-protection is enabled, we must require `PAGEFAULT_FLAG_WP` to
    // receive write-protected faults. We don't need to respond to REMOVE events,
    // as these will be handled by faulting the snapshot pages back in.
    let (features, register_mode) = match enable_write_protection {
        true => (
            FeatureFlags::PAGEFAULT_FLAG_WP,
            RegisterMode::MISSING | RegisterMode::WRITE_PROTECT,
        ),
        false => (FeatureFlags::EVENT_REMOVE, RegisterMode::MISSING),
    };

    uffd_builder.require_features(features);

    let uffd = uffd_builder
        .close_on_exec(true)
        .non_blocking(true)
        .user_mode_only(false)
        .create()
        .map_err(GuestMemoryFromUffdError::Create)?;

    for mem_region in guest_memory.iter() {
        uffd.register_with_mode(
            mem_region.as_ptr().cast(),
            mem_region.size() as _,
            register_mode,
        )
        .map_err(GuestMemoryFromUffdError::Register)?;

        if enable_write_protection {
            uffd.write_protect(mem_region.as_ptr().cast(), mem_region.size() as _)
                .map_err(GuestMemoryFromUffdError::Register)?;
        }
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
    // This is safe to unwrap() because we control the contents of the message
    // (i.e GuestRegionUffdMapping entries).
    let message = SnapshotMessage::LoadSnapshot {
        guest_mappings: Vec::from(backend_mappings),
    };
    let message = serde_json::to_string(&message).unwrap();

    let socket = UnixStream::connect(mem_uds_path)?;
    socket.send_with_fd(
        message.as_bytes(),
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

/// New file paths to be sent over to the UFFD handler to open during a reset.
#[derive(Debug, Serialize, Deserialize)]
pub struct ResetSnapshotFiles {
    /// The path of a new memory file to open.
    new_mem_file_path: PathBuf,
    /// The path of the diff between the current snapshot's memory and the new snapshot's memory.
    diff_file_path: PathBuf,
}
/// Message struct sent over UFFD socket.
#[derive(Debug, Serialize, Deserialize)]
pub enum SnapshotMessage {
    /// A request to load the handler's snapshot.
    LoadSnapshot {
        /// A description of the VMMs memory layout used by the UFFD handler to resolve page
        /// faults.
        guest_mappings: Vec<GuestRegionUffdMapping>,
    },
    /// A request to reset back to a snapshot.
    ResetSnapshot {
        /// New files for the handler to load.
        new_snapshot: Option<ResetSnapshotFiles>,
    },
}

/// Errors related to resetting to a snapshot.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum ResetSnapshotError {
    /// Failed to connect to Unix stream: {0}
    SocketConnect(std::io::Error),
    /// Failed to shutdown Unix stream: {0}
    SocketShutdown(std::io::Error),
    /// Error occurred with serialization/deserialization: {0}
    SerdeJson(#[from] serde_json::Error),
    /// Resetting is not enabled on this VM.
    ResettingNotEnabled,
    /// Failed to remove dirty page: {0}
    RemoveDirtyPages(std::io::Error),
    /// Failed to restore VM state: {0}
    RestoreVmState(ArchVmError),
    /// Failed to restore Vcpu states: {0}
    RestoreVcpuState(MicrovmStateError),
    /// Failed to load new snapshot state from file: {0}
    LoadNewState(SnapshotStateFromFileError),
}

/// Represents page ranges to be remove for resetting 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualAddressRange {
    /// The PFN that the range starts at.
    start_pfn: u64,
    // Number of PFNs included in the range.
    len: usize,
}

/// Reset the VMM back to a snapshotted state
pub fn reset_to_snapshot(
    vmm: &mut Vmm,
    params: ResetSnapshotParams,
) -> Result<(), ResetSnapshotError> {
    let checkpoint = vmm
        .checkpoint
        .as_mut()
        .ok_or(ResetSnapshotError::ResettingNotEnabled)?;

    let microvm_state = snapshot_state_from_file(&params.snapshot_path)
        .map_err(ResetSnapshotError::LoadNewState)?;

    // TODO: This is a really weak comparison to make. MicrovmState does not implement `PartialEq`,
    // so we cannot simply compare the states. This is part of a wider problem though, about moving
    // the VCPU states. If we can somehow pass them by reference to the thread, or implement
    // `Copy`, all of this would be unnecessary.
    let new_snapshot_files = {
        if *checkpoint.current_snapshot_path.as_path() == *params.snapshot_path.as_path() {
            None
        } else {
            checkpoint.current_snapshot_path = params.snapshot_path.clone();
            Some(ResetSnapshotFiles {
                new_mem_file_path: params.mem_file_path.unwrap(),
                diff_file_path: params.diff_file_path.unwrap(),
            })
        }
    };

    let stream =
        UnixStream::connect(params.reset_socket_path).map_err(ResetSnapshotError::SocketConnect)?;
    let reset_message = SnapshotMessage::ResetSnapshot {
        new_snapshot: new_snapshot_files
    };

    let start_time = Instant::now();
    {
        let writer = BufWriter::new(&stream);
        serde_json::to_writer(writer, &reset_message).map_err(ResetSnapshotError::SerdeJson)?;
        stream
            .shutdown(std::net::Shutdown::Write)
            .map_err(ResetSnapshotError::SocketShutdown)?;
    }

    let page_ranges: Vec<VirtualAddressRange> = {
        let reader = BufReader::new(&stream);
        serde_json::from_reader(reader).map_err(ResetSnapshotError::SerdeJson)?
    };
    info!("Time spent in handler: {:?}", start_time.elapsed());

    if !page_ranges.is_empty() {
        let io_uring = &mut checkpoint.madvise_ring;

        let page_size = microvm_state.vm_info.huge_pages.page_size();

        for range in page_ranges {
            let mut start_addr = range.start_pfn * page_size as u64;
            let mut size = range.len * page_size;

            // io_uring SQEs use u32 for `len` field. Therefore, if we have a range greater than 
            // u32::MAX, we must chunk it.
            while size > u32::MAX as usize {
                io_uring.push(Operation::madvise(
                    u64_to_usize(start_addr),
                    u32::MAX,
                    libc::MADV_DONTNEED,
                    start_addr,
                )).unwrap();
                start_addr += u32::MAX as u64;
                size -= u32::MAX as usize;
            }

            io_uring
                .push(Operation::madvise(
                    u64_to_usize(start_addr),
                    u32::try_from(size).unwrap(),
                    libc::MADV_DONTNEED,
                    start_addr,
                ))
                .unwrap();
        }

        let submitted = io_uring.submit_and_wait_all().unwrap();
        let mut done = 0;
        while done < submitted {
            match io_uring.pop() {
                Ok(Some(_)) => {
                    done += 1;
                }
                Ok(None) => (),
                Err(_) => panic!("Error removing from io_uring"), // How to handle this gracefully?
            }
        }
    }

    vmm.restore_vcpu_states(microvm_state.vcpu_states)
        .map_err(ResetSnapshotError::RestoreVcpuState)?;

    vmm.vm
        .restore_state(&microvm_state.vm_state)
        .map_err(ResetSnapshotError::RestoreVmState)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixListener;

    use vmm_sys_util::tempfile::TempFile;

    use super::*;
    use crate::Vmm;
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
    use crate::vstate::memory::GuestMemoryRegionState;

    fn default_vmm_with_devices() -> Vmm {
        let mut event_manager = EventManager::new().expect("Cannot create EventManager");
        let mut vmm = default_vmm();
        let mut cmdline = default_kernel_cmdline();

        // Add a balloon device.
        let balloon_config = BalloonDeviceConfig {
            amount_mib: 0,
            deflate_on_oom: false,
            stats_polling_interval_s: 0,
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

        vmm
    }

    #[test]
    fn test_microvm_state_snapshot() {
        let vmm = default_vmm_with_devices();
        let states = vmm.mmio_device_manager.save();

        // Only checking that all devices are saved, actual device state
        // is tested by that device's tests.
        assert_eq!(states.block_devices.len(), 1);
        assert_eq!(states.net_devices.len(), 1);
        assert!(states.vsock_device.is_some());
        assert!(states.balloon_device.is_some());

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
            vm_state: vmm.vm.save_state(&mpidrs).unwrap(),
            #[cfg(target_arch = "x86_64")]
            vm_state: vmm.vm.save_state().unwrap(),
            acpi_dev_state: vmm.acpi_device_manager.save(),
        };

        let mut buf = vec![0; 10000];
        Snapshot::serialize(&mut buf.as_mut_slice(), &microvm_state).unwrap();

        let restored_microvm_state: MicrovmState =
            Snapshot::deserialize(&mut buf.as_slice()).unwrap();

        assert_eq!(restored_microvm_state.vm_info, microvm_state.vm_info);
        assert_eq!(
            restored_microvm_state.device_states,
            microvm_state.device_states
        )
    }

    #[test]
    fn test_create_guest_memory() {
        let mem_state = GuestMemoryState {
            regions: vec![GuestMemoryRegionState {
                base_address: 0,
                size: 0x20000,
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
