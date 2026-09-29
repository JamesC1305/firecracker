// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use event_manager::{MutEventSubscriber, SubscriberOps};
use serde::{Deserialize, Serialize};

use super::persist::MmdsState;
use crate::EventManager;
use crate::device_manager::DevicePersistError;
use crate::devices::pci::PciSegment;
use crate::devices::virtio::balloon::Balloon;
use crate::devices::virtio::balloon::persist::BalloonState;
use crate::devices::virtio::block::device::Block;
use crate::devices::virtio::block::persist::BlockState;
use crate::devices::virtio::device::{VirtioDevice, VirtioDeviceId, VirtioDeviceType};
use crate::devices::virtio::mem::VirtioMem;
use crate::devices::virtio::mem::persist::{VirtioMemConstructorArgs, VirtioMemState};
use crate::devices::virtio::net::Net;
use crate::devices::virtio::net::persist::{NetConstructorArgs, NetState};
use crate::devices::virtio::pmem::device::Pmem;
use crate::devices::virtio::pmem::persist::{PmemConstructorArgs, PmemState};
use crate::devices::virtio::rng::Entropy;
use crate::devices::virtio::rng::persist::EntropyState;
use crate::devices::virtio::transport::pci::device::{
    CAPABILITY_BAR_SIZE, VirtioPciDevice, VirtioPciDeviceError, VirtioPciDeviceState,
};
use crate::devices::virtio::vsock::persist::{VsockConstructorArgs, VsockState};
use crate::devices::virtio::vsock::{Vsock, VsockError, VsockUnixBackend};
use crate::logger::{debug, warn};
use crate::pci::PciSBDF;
use crate::pci::bus::PciBusError;
use crate::resources::VmResources;
use crate::snapshot::{LoadContext, Persist, ResetUnsupported};
use crate::vmm_config::memory_hotplug::MemoryHotplugConfig;
use crate::vstate::bus::BusError;
use crate::vstate::interrupts::{InterruptError, MsixVectorGroup};
use crate::vstate::memory::GuestMemoryMmap;
use crate::vstate::vm::KvmVm;

#[derive(Debug)]
pub struct PciDevices {
    /// PCIe segment of the VMM. We currently support a single PCIe segment.
    pub pci_segment: PciSegment,
    /// All VirtIO PCI devices of the system
    pub virtio_devices: HashMap<VirtioDeviceId, Arc<Mutex<VirtioPciDevice>>>,
}

#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum PciManagerError {
    /// Resource allocation error: {0}
    ResourceAllocation(#[from] vm_allocator::Error),
    /// Bus error: {0}
    Bus(#[from] BusError),
    /// PCI bus error: {0}
    PciBus(#[from] PciBusError),
    /// MSI error: {0}
    Msi(#[from] InterruptError),
    /// VirtIO PCI device error: {0}
    VirtioPciDevice(#[from] VirtioPciDeviceError),
    /// KVM error: {0}
    Kvm(#[from] vmm_sys_util::errno::Error),
}

impl PciDevices {
    pub fn new(vm: &Arc<KvmVm>) -> Result<Self, PciManagerError> {
        // Currently we don't assign any IRQs to PCI devices. We will be using MSI-X interrupts
        // only.
        let pci_segment = PciSegment::new(0, vm, &[0u8; 32])?;

        Ok(Self {
            pci_segment,
            virtio_devices: HashMap::new(),
        })
    }

    fn attach_common(
        &mut self,
        vm: &KvmVm,
        device_type: VirtioDeviceType,
        id: String,
        sbdf: PciSBDF,
        virtio_device: Arc<Mutex<VirtioPciDevice>>,
    ) -> Result<&Arc<Mutex<VirtioPciDevice>>, PciManagerError> {
        let bar_address = {
            let device = virtio_device.lock().unwrap();

            device.register_notification_ioevents(vm)?;

            device.bar_address()
        };

        self.pci_segment
            .pci_bus
            .lock()
            .expect("Poisoned lock")
            .add_device(sbdf.device(), virtio_device.clone())?;

        debug!(
            "Inserting MMIO BAR region: {:#x}:{:#x}",
            bar_address, CAPABILITY_BAR_SIZE
        );
        vm.common
            .mmio_bus
            .insert(virtio_device.clone(), bar_address, CAPABILITY_BAR_SIZE)?;

        Ok(self
            .virtio_devices
            .entry((device_type, id))
            .insert_entry(virtio_device)
            .into_mut())
    }

    fn subscribe_device(device: &mut VirtioPciDevice, event_manager: &mut EventManager) {
        device.sub_id = Some(event_manager.add_subscriber(device.virtio_device()));
    }

    pub(crate) fn attach_pci_virtio_device(
        &mut self,
        vm: &Arc<KvmVm>,
        id: String,
        device: Arc<Mutex<dyn VirtioDevice>>,
        event_manager: &mut EventManager,
    ) -> Result<(), PciManagerError> {
        let sbdf = self.pci_segment.next_device_sbdf()?;
        debug!("Allocating SBDF: {sbdf:?} for device");

        let device_type = device.lock().expect("Poisoned lock").device_type();

        // Allocate one MSI vector per queue, plus one for configuration
        let msix_num =
            u16::try_from(device.lock().expect("Poisoned lock").queues().len() + 1).unwrap();

        let msix_vectors = KvmVm::create_msix_group(vm.clone(), msix_num)?;

        // Create the transport
        let mut virtio_device =
            VirtioPciDevice::new(id.clone(), vm, device, Arc::new(msix_vectors), sbdf);

        // Don't hold the resource allocator lock across attach_common()
        // below: a device access holds the bus lock and can take the allocator
        // lock, so the reverse order can deadlock.
        virtio_device.allocate_bars(&mut vm.resource_allocator().mmio32_memory);

        let virtio_device = Arc::new(Mutex::new(virtio_device));

        let device = self.attach_common(vm, device_type, id, sbdf, virtio_device)?;
        Self::subscribe_device(&mut device.lock().expect("Poisoned lock"), event_manager);
        Ok(())
    }

    pub(crate) fn pci_segment(&self) -> &PciSegment {
        &self.pci_segment
    }

    #[cfg(target_arch = "x86_64")]
    pub(crate) fn append_aml_bytes(
        &self,
        dsdt_data: &mut Vec<u8>,
    ) -> Result<(), acpi_tables::aml::AmlError> {
        use acpi_tables::Aml;

        self.pci_segment().append_aml_bytes(dsdt_data)
    }

    pub(crate) fn detach_pci_virtio_device(
        &mut self,
        vm: &KvmVm,
        device_id: VirtioDeviceId,
        event_manager: &mut EventManager,
    ) -> Result<(), PciManagerError> {
        let pci_device_arc = self
            .virtio_devices
            .remove(&device_id)
            .expect("device presence should be checked before detach");

        let sbdf_device = pci_device_arc.lock().expect("Poisoned lock").sbdf.device();

        // Remove the device from the PCI bus first. A config space access runs
        // with the PCI bus lock held and can relocate the BAR, so afterwards
        // the BAR address of the device can no longer change under us.
        self.pci_segment
            .pci_bus
            .lock()
            .expect("Poisoned lock")
            .remove_device(sbdf_device);

        // Next operations of removing device from mmio_bus and pci_bus need to wait for any other
        // user of the device to finish. This requires us to not hold the lock for the device in
        // case someone will try to access the device while we are in these several lines of code.
        let (bar_addr, sub_id) = {
            let pci_device = pci_device_arc.lock().expect("Poisoned lock");

            pci_device
                .unregister_notification_ioevents(vm)
                .map_err(PciManagerError::Kvm)?;
            (pci_device.bar_address(), pci_device.sub_id)
        };

        vm.common
            .mmio_bus
            .remove(bar_addr, CAPABILITY_BAR_SIZE)
            .map_err(PciManagerError::Bus)?;

        if let Some(sub_id) = sub_id
            && event_manager.remove_subscriber(sub_id).is_err()
        {
            warn!("Failed to remove event subscriber for device {device_id:?}");
        }

        pci_device_arc
            .lock()
            .expect("Poisoned lock")
            .free_bars(&mut vm.resource_allocator().mmio32_memory);

        // Ensure no other references to the device remain, so it is freed when
        // this function returns.
        assert_eq!(Arc::strong_count(&pci_device_arc), 1);

        Ok(())
    }

    fn create_pci_device<T: 'static + VirtioDevice + MutEventSubscriber + Debug>(
        &mut self,
        vm: &Arc<KvmVm>,
        device: Arc<Mutex<T>>,
        device_id: &str,
        transport_state: &VirtioPciDeviceState,
    ) -> Result<(), PciManagerError> {
        let device_type = device.lock().expect("Poisoned lock").device_type();

        let gsis = transport_state
            .msix_state
            .checked_gsis()
            .map_err(VirtioPciDeviceError::from)?;
        let expected_num_vectors = device.lock().expect("Poisoned lock").queues().len() + 1;
        if gsis.len() != expected_num_vectors {
            return Err(VirtioPciDeviceError::UnexpectedMsixVectorCount(
                gsis.len(),
                expected_num_vectors,
            )
            .into());
        }
        let vectors = Arc::new(
            MsixVectorGroup::from_gsis(vm.clone(), gsis).map_err(VirtioPciDeviceError::from)?,
        );
        let mut virtio_device = VirtioPciDevice::new(
            device_id.to_string(),
            vm,
            device,
            vectors,
            transport_state.sbdf,
        );
        virtio_device.set_restored_bar_address(transport_state.bar_address);
        let virtio_device = Arc::new(Mutex::new(virtio_device));

        self.attach_common(
            vm,
            device_type,
            device_id.to_string(),
            transport_state.sbdf,
            virtio_device,
        )?;

        Ok(())
    }

    /// Gets the specified device.
    pub fn get_virtio_device(
        &self,
        device_type: VirtioDeviceType,
        device_id: &str,
    ) -> Option<&Arc<Mutex<VirtioPciDevice>>> {
        self.virtio_devices
            .get(&(device_type, device_id.to_string()))
    }

    pub(crate) fn get_device(
        &self,
        device_type: VirtioDeviceType,
        device_id: &str,
    ) -> Option<Arc<Mutex<dyn VirtioDevice>>> {
        self.get_virtio_device(device_type, device_id)
            .map(|device| device.lock().expect("Poisoned lock").virtio_device())
    }

    pub(crate) fn contains_virtio_device(&self, device_id: &VirtioDeviceId) -> bool {
        self.virtio_devices.contains_key(device_id)
    }

    pub fn for_each_virtio_device(&self, mut f: impl FnMut(VirtioDeviceType, &dyn VirtioDevice)) {
        for ((device_type, _), pci_device) in &self.virtio_devices {
            let device_arc = pci_device.lock().expect("Poisoned lock").virtio_device();
            let device = device_arc.lock().expect("Poisoned lock");
            f(*device_type, &*device);
        }
    }

    pub(crate) fn for_each_virtio_device_mut(
        &self,
        mut f: impl FnMut(VirtioDeviceType, &mut dyn VirtioDevice),
    ) {
        for ((device_type, _), pci_device) in &self.virtio_devices {
            let device_arc = pci_device.lock().expect("Poisoned lock").virtio_device();
            let mut device = device_arc.lock().expect("Poisoned lock");
            f(*device_type, &mut *device);
        }
    }

    /// Installs the GSI routes that restoring the devices staged, then enables the unmasked
    /// MSI-X vectors of every device. Installing the routes after enabling an irqfd can panic
    /// older AMD/SVM kernels (see kernel commit a80ced6ea514).
    fn enable_msix_vectors(&self, vm: &KvmVm) -> Result<(), PciManagerError> {
        if self.virtio_devices.is_empty() {
            return Ok(());
        }
        vm.set_gsi_routes()?;
        for device in self.virtio_devices.values() {
            device
                .lock()
                .expect("Poisoned lock")
                .enable_unmasked_vectors()?;
        }
        Ok(())
    }

    fn restore_devices_in_place<'a, D>(
        &self,
        states: &[VirtioDeviceState<D::State>],
        mem: &'a GuestMemoryMmap,
    ) -> Result<(), DevicePersistError>
    where
        D: VirtioDevice + Persist<'a, ApplyArgs = &'a GuestMemoryMmap> + 'static,
        DevicePersistError: From<D::Error>,
    {
        for state in states {
            let device = self
                .get_virtio_device(D::const_device_type(), &state.device_id)
                .expect("snapshot load created a device for each state");
            let mut transport = device.lock().expect("Poisoned lock");
            transport
                .virtio_device()
                .lock()
                .expect("Poisoned lock")
                .as_mut_any()
                .downcast_mut::<D>()
                .expect("a device has the type of its state")
                .restore_in_place(&state.device_state, mem)?;
            transport
                .restore_in_place(&state.transport_state, ())
                .map_err(PciManagerError::from)?;
        }
        Ok(())
    }

    fn post_restore_devices<'a, D>(
        &self,
        states: &[VirtioDeviceState<D::State>],
        load: &mut LoadContext<'_>,
    ) -> Result<(), DevicePersistError>
    where
        D: VirtioDevice + Persist<'a> + 'static,
        DevicePersistError: From<D::Error>,
    {
        for state in states {
            let device = self
                .get_virtio_device(D::const_device_type(), &state.device_id)
                .expect("snapshot create made a device for each state");
            let mut transport = device.lock().expect("Poisoned lock");
            transport
                .virtio_device()
                .lock()
                .expect("Poisoned lock")
                .as_mut_any()
                .downcast_mut::<D>()
                .expect("a device has the type of its state")
                .post_restore(&state.device_state, load)?;
            transport
                .post_restore(&state.transport_state, load)
                .map_err(PciManagerError::from)?;
            // PCI init has always registered runtime watches after backend activation.
            Self::subscribe_device(&mut transport, load.event_manager);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtioDeviceState<T> {
    /// Device identifier
    pub device_id: String,
    /// Device SBDF
    pub sbdf: PciSBDF,
    /// Device state
    pub device_state: T,
    /// Transport state
    pub transport_state: VirtioPciDeviceState,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct PciDevicesState {
    /// Block device states.
    pub block_devices: Vec<VirtioDeviceState<BlockState>>,
    /// Net device states.
    pub net_devices: Vec<VirtioDeviceState<NetState>>,
    /// Vsock device state.
    pub vsock_device: Option<VirtioDeviceState<VsockState>>,
    /// Balloon device state.
    pub balloon_device: Option<VirtioDeviceState<BalloonState>>,
    /// Mmds state.
    pub mmds: Option<MmdsState>,
    /// Entropy device state.
    pub entropy_device: Option<VirtioDeviceState<EntropyState>>,
    /// Pmem device states.
    pub pmem_devices: Vec<VirtioDeviceState<PmemState>>,
    /// Memory device state.
    pub memory_device: Option<VirtioDeviceState<VirtioMemState>>,
}

pub struct PciDevicesConstructorArgs<'a> {
    pub vm: &'a Arc<KvmVm>,
    pub vm_resources: &'a mut VmResources,
    pub instance_id: &'a str,
}

impl<'a> Debug for PciDevicesConstructorArgs<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PciDevicesConstructorArgs")
            .field("vm", &self.vm)
            .field("vm_resources", &self.vm_resources)
            .field("instance_id", &self.instance_id)
            .finish()
    }
}

impl<'a> Persist<'a> for PciDevices {
    type State = PciDevicesState;
    type ConstructorArgs = PciDevicesConstructorArgs<'a>;
    type ApplyArgs = &'a KvmVm;
    type Error = DevicePersistError;

    fn save(&self) -> Self::State {
        let mut state = PciDevicesState::default();

        for pci_dev in self.virtio_devices.values() {
            let locked_pci_dev = pci_dev.lock().expect("Poisoned lock");
            let virtio_dev = locked_pci_dev.virtio_device();
            // We need to call `prepare_save()` on the device before saving the transport
            // so that, if we modify the transport state while preparing the device, e.g. sending
            // an interrupt to the guest, this is correctly captured in the saved transport state.
            let mut locked_virtio_dev = virtio_dev.lock().expect("Poisoned lock");
            locked_virtio_dev.prepare_save();
            let transport_state = locked_pci_dev.save();

            let sbdf = transport_state.sbdf;

            match locked_virtio_dev.device_type() {
                VirtioDeviceType::Balloon => {
                    let balloon_device = locked_virtio_dev
                        .as_any()
                        .downcast_ref::<Balloon>()
                        .unwrap();

                    let device_state = balloon_device.save();

                    state.balloon_device = Some(VirtioDeviceState {
                        device_id: balloon_device.id().to_string(),
                        sbdf,
                        device_state,
                        transport_state,
                    });
                }
                VirtioDeviceType::Block => {
                    let block_dev = locked_virtio_dev
                        .as_mut_any()
                        .downcast_mut::<Block>()
                        .unwrap();
                    if block_dev.is_vhost_user() {
                        warn!(
                            "Skipping vhost-user-block device. VhostUserBlock does not support \
                             snapshotting yet"
                        );
                    } else {
                        let device_state = block_dev.save();
                        state.block_devices.push(VirtioDeviceState {
                            device_id: block_dev.id().to_string(),
                            sbdf,
                            device_state,
                            transport_state,
                        });
                    }
                }
                VirtioDeviceType::Net => {
                    let net_dev = locked_virtio_dev
                        .as_mut_any()
                        .downcast_mut::<Net>()
                        .unwrap();
                    if let (Some(mmds_ns), None) = (net_dev.mmds_ns.as_ref(), state.mmds.as_ref()) {
                        let mmds_guard = mmds_ns.mmds.lock().expect("Poisoned lock");
                        state.mmds = Some(MmdsState {
                            version: mmds_guard.version(),
                            imds_compat: mmds_guard.imds_compat(),
                        });
                    }
                    let device_state = net_dev.save();

                    state.net_devices.push(VirtioDeviceState {
                        device_id: net_dev.id().to_string(),
                        sbdf,
                        device_state,
                        transport_state,
                    })
                }
                VirtioDeviceType::Vsock => {
                    let vsock_dev = locked_virtio_dev
                        .as_mut_any()
                        // Currently, VsockUnixBackend is the only implementation of VsockBackend.
                        .downcast_mut::<Vsock<VsockUnixBackend>>()
                        .unwrap();

                    // Save state after potential notification to the guest. This
                    // way we save changes to the queue the notification can cause.
                    let vsock_state = vsock_dev.save();

                    state.vsock_device = Some(VirtioDeviceState {
                        device_id: vsock_dev.id().to_string(),
                        sbdf,
                        device_state: vsock_state,
                        transport_state,
                    });
                }
                VirtioDeviceType::Rng => {
                    let rng_dev = locked_virtio_dev
                        .as_mut_any()
                        .downcast_mut::<Entropy>()
                        .unwrap();
                    let device_state = rng_dev.save();

                    state.entropy_device = Some(VirtioDeviceState {
                        device_id: rng_dev.id().to_string(),
                        sbdf,
                        device_state,
                        transport_state,
                    })
                }
                VirtioDeviceType::Pmem => {
                    let pmem_dev = locked_virtio_dev
                        .as_mut_any()
                        .downcast_mut::<Pmem>()
                        .unwrap();
                    let device_state = pmem_dev.save();
                    state.pmem_devices.push(VirtioDeviceState {
                        device_id: pmem_dev.config.id.clone(),
                        sbdf,
                        device_state,
                        transport_state,
                    });
                }
                VirtioDeviceType::Mem => {
                    let mem_dev = locked_virtio_dev
                        .as_mut_any()
                        .downcast_mut::<VirtioMem>()
                        .unwrap();
                    let device_state = mem_dev.save();

                    state.memory_device = Some(VirtioDeviceState {
                        device_id: mem_dev.id().to_string(),
                        sbdf,
                        device_state,
                        transport_state,
                    })
                }
            }
        }

        state
    }

    fn create(
        constructor_args: Self::ConstructorArgs,
        state: &Self::State,
    ) -> Result<Self, Self::Error> {
        let vm = constructor_args.vm;
        let mut pci_devices = PciDevices::new(vm)?;

        // Record each device in VmResources while its concrete type is still known.
        if let Some(saved) = &state.balloon_device {
            let device = Arc::new(Mutex::new(Balloon::create((), &saved.device_state)?));
            constructor_args
                .vm_resources
                .balloon
                .set_device(device.clone());
            pci_devices.create_pci_device(vm, device, &saved.device_id, &saved.transport_state)?;
        }
        for saved in &state.block_devices {
            let device = Arc::new(Mutex::new(Block::create((), &saved.device_state)?));
            constructor_args
                .vm_resources
                .block
                .add_virtio_device(device.clone());
            pci_devices.create_pci_device(vm, device, &saved.device_id, &saved.transport_state)?;
        }
        // Net devices share the MMDS datastore, so configure it before creating them.
        if let Some(mmds) = &state.mmds {
            constructor_args.vm_resources.set_mmds_basic_config(
                mmds.version,
                mmds.imds_compat,
                constructor_args.instance_id,
            )?;
        } else if state
            .net_devices
            .iter()
            .any(|dev| dev.device_state.mmds_ns.is_some())
        {
            // If there's at least one network device having an mmds_ns, it means
            // that we are restoring from a version that did not persist the `MmdsVersionState`.
            // Init with the default.
            constructor_args.vm_resources.mmds_or_default()?;
        }
        for saved in &state.net_devices {
            let device = Arc::new(Mutex::new(Net::create(
                NetConstructorArgs {
                    mmds: constructor_args.vm_resources.mmds.clone(),
                },
                &saved.device_state,
            )?));
            constructor_args
                .vm_resources
                .net_builder
                .add_device(device.clone());
            pci_devices.create_pci_device(vm, device, &saved.device_id, &saved.transport_state)?;
        }
        if let Some(saved) = &state.vsock_device {
            let device = Arc::new(Mutex::new(Vsock::<VsockUnixBackend>::create(
                VsockConstructorArgs {
                    backend: VsockUnixBackend::create(
                        (
                            saved.device_state.frontend.cid,
                            saved.device_state.backend.uds_path.clone(),
                        ),
                        &saved.device_state.backend,
                    )
                    .map_err(VsockError::VsockUdsBackend)?,
                },
                &saved.device_state,
            )?));
            constructor_args
                .vm_resources
                .vsock
                .set_device(device.clone());
            pci_devices.create_pci_device(vm, device, &saved.device_id, &saved.transport_state)?;
        }
        if let Some(saved) = &state.entropy_device {
            let device = Arc::new(Mutex::new(Entropy::create((), &saved.device_state)?));
            constructor_args
                .vm_resources
                .entropy
                .set_device(device.clone());
            pci_devices.create_pci_device(vm, device, &saved.device_id, &saved.transport_state)?;
        }
        for saved in &state.pmem_devices {
            let device = Arc::new(Mutex::new(Pmem::create(
                PmemConstructorArgs { vm: vm.clone() },
                &saved.device_state,
            )?));
            constructor_args
                .vm_resources
                .pmem
                .configs
                .push(saved.device_state.config.clone());
            pci_devices.create_pci_device(vm, device, &saved.device_id, &saved.transport_state)?;
        }
        if let Some(saved) = &state.memory_device {
            let device = VirtioMem::create(
                VirtioMemConstructorArgs::new(vm.clone()),
                &saved.device_state,
            )?;
            constructor_args.vm_resources.memory_hotplug = Some(MemoryHotplugConfig {
                total_size_mib: device.total_size_mib(),
                block_size_mib: device.block_size_mib(),
                slot_size_mib: device.slot_size_mib(),
            });
            pci_devices.create_pci_device(
                vm,
                Arc::new(Mutex::new(device)),
                &saved.device_id,
                &saved.transport_state,
            )?;
        }
        Ok(pci_devices)
    }

    /// Keeps bus mappings, ioeventfds and interrupt objects, then installs the restored routes
    /// before enabling unmasked vectors.
    fn restore_in_place(
        &mut self,
        state: &Self::State,
        vm: Self::ApplyArgs,
    ) -> Result<(), Self::Error> {
        let mem = vm.guest_memory();
        let PciDevicesState {
            block_devices,
            net_devices,
            vsock_device,
            balloon_device,
            // MMDS configuration is unchanged; the VMM restores its datastore and token key.
            mmds: _,
            entropy_device,
            pmem_devices,
            memory_device,
        } = state;
        self.restore_devices_in_place::<Balloon>(balloon_device.as_slice(), mem)?;
        self.restore_devices_in_place::<Block>(block_devices, mem)?;
        self.restore_devices_in_place::<Net>(net_devices, mem)?;
        self.restore_devices_in_place::<Vsock<VsockUnixBackend>>(vsock_device.as_slice(), mem)?;
        self.restore_devices_in_place::<Entropy>(entropy_device.as_slice(), mem)?;
        self.restore_devices_in_place::<Pmem>(pmem_devices, mem)?;
        self.restore_devices_in_place::<VirtioMem>(memory_device.as_slice(), mem)?;
        self.enable_msix_vectors(vm)?;
        Ok(())
    }

    fn post_restore(
        &mut self,
        state: &Self::State,
        load: &mut LoadContext<'_>,
    ) -> Result<(), Self::Error> {
        self.post_restore_devices::<Balloon>(state.balloon_device.as_slice(), load)?;
        self.post_restore_devices::<Block>(state.block_devices.as_slice(), load)?;
        self.post_restore_devices::<Net>(state.net_devices.as_slice(), load)?;
        self.post_restore_devices::<Vsock<VsockUnixBackend>>(state.vsock_device.as_slice(), load)?;
        self.post_restore_devices::<Entropy>(state.entropy_device.as_slice(), load)?;
        self.post_restore_devices::<Pmem>(state.pmem_devices.as_slice(), load)?;
        self.post_restore_devices::<VirtioMem>(state.memory_device.as_slice(), load)?;
        Ok(())
    }

    fn check_reset(&self, _state: &Self::State) -> Result<(), ResetUnsupported> {
        Err(ResetUnsupported("the PCI transport"))
    }
}

#[cfg(test)]
mod tests {
    use vmm_sys_util::tempfile::TempFile;

    use super::*;
    use crate::builder::tests::*;
    use crate::device_manager;
    use crate::devices::virtio::block::CacheType;
    use crate::mmds::data_store::MmdsVersion;
    use crate::resources::VmmConfig;
    use crate::vmm_config::balloon::BalloonDeviceConfig;
    use crate::vmm_config::entropy::EntropyDeviceConfig;
    use crate::vmm_config::memory_hotplug::MemoryHotplugConfig;
    use crate::vmm_config::net::NetworkInterfaceConfig;
    use crate::vmm_config::pmem::PmemConfig;
    use crate::vmm_config::vsock::VsockDeviceConfig;
    use crate::vstate::resources::ResourceAllocator;

    #[test]
    fn test_device_manager_persistence() {
        // These need to survive so the restored blocks find them.
        let _block_files;
        let _pmem_files;
        let mut tmp_sock_file = TempFile::new().unwrap();
        tmp_sock_file.remove().unwrap();

        let serialized_data;
        let saved_allocator;
        // Set up a vmm with one of each device, and get the serialized DeviceStates.
        {
            let mut event_manager = EventManager::new().expect("Unable to create EventManager");
            let mut vmm = default_vmm_with_pci();
            let mut cmdline = default_kernel_cmdline();

            // Add a balloon device.
            let balloon_cfg = BalloonDeviceConfig {
                amount_mib: 123,
                deflate_on_oom: false,
                stats_polling_interval_s: 1,
                free_page_hinting: false,
                free_page_reporting: false,
            };
            insert_balloon_device(&mut vmm, &mut cmdline, &mut event_manager, balloon_cfg);
            // Add a block device.
            let drive_id = String::from("root");
            let block_configs = vec![CustomBlockConfig::new(
                drive_id,
                true,
                None,
                true,
                CacheType::Unsafe,
            )];
            _block_files =
                insert_block_devices(&mut vmm, &mut cmdline, &mut event_manager, block_configs);
            // Add a net device.
            let network_interface = NetworkInterfaceConfig {
                iface_id: String::from("netif"),
                host_dev_name: String::from("hostname"),
                guest_mac: None,
                mtu: None,
                rx_rate_limiter: None,
                tx_rate_limiter: None,
            };
            insert_net_device_with_mmds(
                &mut vmm,
                &mut cmdline,
                &mut event_manager,
                network_interface,
                MmdsVersion::V2,
            );
            // Add a vsock device.
            let vsock_dev_id = "vsock";
            let vsock_config = VsockDeviceConfig {
                vsock_id: Some(vsock_dev_id.to_string()),
                guest_cid: 3,
                uds_path: tmp_sock_file.as_path().to_str().unwrap().to_string(),
            };
            insert_vsock_device(&mut vmm, &mut cmdline, &mut event_manager, vsock_config);
            // Add an entropy device.
            let entropy_config = EntropyDeviceConfig::default();
            insert_entropy_device(&mut vmm, &mut cmdline, &mut event_manager, entropy_config);
            // Add a pmem device.
            let pmem_id = String::from("pmem");
            let pmem_configs = vec![PmemConfig {
                id: pmem_id,
                path_on_host: "".into(),
                root_device: true,
                read_only: true,
                ..Default::default()
            }];
            _pmem_files =
                insert_pmem_devices(&mut vmm, &mut cmdline, &mut event_manager, pmem_configs);

            let memory_hotplug_config = MemoryHotplugConfig {
                total_size_mib: 1024,
                block_size_mib: 2,
                slot_size_mib: 128,
            };
            insert_virtio_mem_device(
                &mut vmm,
                &mut cmdline,
                &mut event_manager,
                memory_hotplug_config,
            );

            let device_state = vmm.device_manager.save();
            serialized_data = bitcode::serialize(&device_state).unwrap();
            saved_allocator = vmm.vm.as_kvm().unwrap().resource_allocator().save()
        }

        tmp_sock_file.remove().unwrap();

        let mut event_manager = EventManager::new().expect("Unable to create EventManager");
        // Keep in mind we are re-creating here an empty DeviceManager. Restoring later on
        // will create a new PciDevices manager different from vmm's virtio devices. We're
        // doing this to avoid restoring the whole Vmm, since what we really need from Vmm is the
        // KvmVm object and calling default_vmm() is the easiest way to create one.
        let vmm = default_vmm();
        // Restore the source allocator's state so the restored devices' GSIs match what their
        // `MsixVectorGroup::Drop` will try to free at end-of-test.
        *vmm.vm.as_kvm().unwrap().resource_allocator() =
            ResourceAllocator::from_state(&saved_allocator).unwrap();

        let device_manager_state: device_manager::DevicesState =
            bitcode::deserialize(&serialized_data).unwrap();
        let device_manager::VirtioDevicesState::Pci(pci_state) = &device_manager_state.virtio_state
        else {
            panic!("expected PCI virtio device state");
        };
        let vm_resources = &mut VmResources::default();
        let kvm_vm = vmm.vm.as_kvm().unwrap().clone();
        let restore_args = PciDevicesConstructorArgs {
            vm: &kvm_vm,
            vm_resources,
            instance_id: "microvm-id",
        };
        let _restored_dev_manager = crate::snapshot::restore::<PciDevices>(
            restore_args,
            pci_state,
            &kvm_vm,
            &mut LoadContext {
                event_manager: &mut event_manager,
            },
        )
        .unwrap();

        let expected_vm_resources = format!(
            r#"{{
  "balloon": {{
    "amount_mib": 123,
    "deflate_on_oom": false,
    "stats_polling_interval_s": 1,
    "free_page_hinting": false,
    "free_page_reporting": false
  }},
  "drives": [
    {{
      "drive_id": "root",
      "partuuid": null,
      "is_root_device": true,
      "cache_type": "Unsafe",
      "is_read_only": true,
      "discard": false,
      "path_on_host": "{}",
      "rate_limiter": null,
      "io_engine": "Sync",
      "blk_size": 512,
      "topology": {{
        "physical_block_exp": 0,
        "alignment_offset": 0,
        "min_io_size": 0,
        "opt_io_size": 128
      }},
      "socket": null
    }}
  ],
  "boot-source": {{
    "kernel_image_path": "",
    "initrd_path": null,
    "boot_args": null
  }},
  "cpu-config": null,
  "logger": null,
  "machine-config": {{
    "vcpu_count": 1,
    "mem_size_mib": 128,
    "smt": false,
    "track_dirty_pages": false,
    "huge_pages": "None"
  }},
  "metrics": null,
  "mmds-config": {{
    "version": "V2",
    "network_interfaces": [
      "netif"
    ],
    "ipv4_address": "169.254.169.254",
    "imds_compat": false
  }},
  "network-interfaces": [
    {{
      "iface_id": "netif",
      "host_dev_name": "hostname",
      "guest_mac": null,
      "mtu": null,
      "rx_rate_limiter": null,
      "tx_rate_limiter": null
    }}
  ],
  "vsock": {{
    "guest_cid": 3,
    "uds_path": "{}"
  }},
  "entropy": {{
    "rate_limiter": null
  }},
  "pmem": [
    {{
      "id": "pmem",
      "path_on_host": "{}",
      "root_device": true,
      "read_only": true,
      "rate_limiter": null
    }}
  ],
  "memory-hotplug": {{
    "total_size_mib": 1024,
    "block_size_mib": 2,
    "slot_size_mib": 128
  }}
}}"#,
            _block_files.last().unwrap().as_path().to_str().unwrap(),
            tmp_sock_file.as_path().to_str().unwrap(),
            _pmem_files.last().unwrap().as_path().to_str().unwrap(),
        );

        assert_eq!(
            vm_resources
                .mmds
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .version(),
            MmdsVersion::V2
        );
        assert_eq!(pci_state.mmds.as_ref().unwrap().version, MmdsVersion::V2);
        assert_eq!(
            expected_vm_resources,
            serde_json::to_string_pretty(&VmmConfig::from(&*vm_resources)).unwrap()
        );
    }
}
