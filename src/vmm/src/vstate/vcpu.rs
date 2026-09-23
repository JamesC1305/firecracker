// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

use std::os::fd::AsRawFd;
use std::sync::atomic::{Ordering, fence};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError, channel};
use std::sync::{Arc, Barrier};
use std::time::Duration;
use std::{fmt, io, thread};

use kvm_bindings::{KVM_SYSTEM_EVENT_RESET, KVM_SYSTEM_EVENT_SHUTDOWN};
use kvm_ioctls::{VcpuExit, VcpuFd};
use libc::{c_int, c_void, siginfo_t};
use vmm_sys_util::errno;
use vmm_sys_util::eventfd::EventFd;

use crate::FcExitCode;
pub use crate::arch::{KvmVcpu, KvmVcpuConfigureError, KvmVcpuError, Peripherals, VcpuState};
use crate::cpu_config::templates::{CpuConfiguration, GuestConfigError};
#[cfg(feature = "gdb")]
use crate::gdb::target::{GdbTargetError, get_raw_tid};
use crate::logger::{IncMetric, METRICS, error, info, warn};
use crate::seccomp::{BpfProgram, BpfProgramRef};
use crate::utils::signal::{Killable, register_signal_handler, sigrtmin};
use crate::vstate::bus::Bus;
use crate::vstate::vm::KvmVm;

/// Signal number (SIGRTMIN) used to kick Vcpus.
pub const VCPU_RTSIG_OFFSET: i32 = 0;

/// Maximum time to wait for a vCPU thread to exit when dropping its handle.
const VCPU_JOIN_TIMEOUT: Duration = Duration::from_secs(1);

/// Maximum KVM_RUN calls used to finish pending userspace I/O before reset.
const MAX_IO_COMPLETION_RUNS: usize = 16;

/// Errors associated with the wrappers over KVM ioctls.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum VcpuError {
    /// Error creating vcpu config: {0}
    VcpuConfig(GuestConfigError),
    /// Received error signaling kvm exit: {0}
    FaultyKvmExit(String),
    /// Failed to signal vcpu: {0}
    SignalVcpu(vmm_sys_util::errno::Error),
    /// Unexpected kvm exit received: {0}
    UnhandledKvmExit(String),
    /// Failed to run action on vcpu: {0}
    VcpuResponse(KvmVcpuError),
    /// Failed to complete pending userspace I/O: {0}
    CompletePendingIo(errno::Error),
    /// Unexpected KVM exit {0} while completing pending userspace I/O.
    UnexpectedIoCompletionExit(u32),
    /// Pending userspace I/O did not complete after {0} KVM_RUN calls.
    IoCompletionLimit(usize),
    /// Cannot spawn a new vCPU thread: {0}
    VcpuSpawn(io::Error),
    /// Vcpu not present in TLS
    VcpuTlsNotPresent,
    /// Error with gdb request sent
    #[cfg(feature = "gdb")]
    GdbRequest(GdbTargetError),
}

/// Encapsulates configuration parameters for the guest vCPUS.
#[derive(Debug)]
pub struct VcpuConfig {
    /// Number of guest VCPUs.
    pub vcpu_count: u8,
    /// Enable simultaneous multithreading in the CPUID configuration.
    pub smt: bool,
    /// Configuration for vCPU
    pub cpu_config: CpuConfiguration,
}

/// Error type for [`Vcpu::start_threaded`].
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum StartThreadedError {
    /// Failed to spawn vCPU thread: {0}
    Spawn(std::io::Error),
    /// Failed to clone kvm Vcpu fd: {0}
    CopyFd(CopyKvmFdError),
}

/// Error type for [`Vcpu::copy_kvm_vcpu_fd`].
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum CopyKvmFdError {
    /// Error with libc dup of kvm Vcpu fd
    DupError(#[from] std::io::Error),
    /// Error creating the Vcpu from the duplicated Vcpu fd
    CreateVcpuError(#[from] kvm_ioctls::Error),
}

/// A wrapper around creating and using a vcpu.
#[derive(Debug)]
pub struct Vcpu {
    /// Access to kvm-arch specific functionality.
    pub kvm_vcpu: KvmVcpu,

    /// File descriptor for vcpu to trigger exit event on vmm.
    exit_evt: EventFd,
    /// Debugger emitter for gdb events
    #[cfg(feature = "gdb")]
    gdb_event: Option<Sender<usize>>,
    /// The receiving end of events channel owned by the vcpu side.
    event_receiver: Receiver<VcpuEvent>,
    /// The transmitting end of the events channel which will be given to the handler.
    event_sender: Option<Sender<VcpuEvent>>,
    /// The receiving end of the responses channel which will be given to the handler.
    response_receiver: Option<Receiver<VcpuResponse>>,
    /// The transmitting end of the responses channel owned by the vcpu side.
    response_sender: Sender<VcpuResponse>,
}

/// States of the vCPU thread's run loop.
#[derive(Debug)]
enum VcpuRunState {
    /// The vCPU is executing guest code via `KVM_RUN`.
    Running,
    /// The vCPU is paused, waiting for events.
    Paused,
    /// The vCPU thread's run loop has finished; the thread will exit.
    Finished,
}

impl Vcpu {
    /// Registers a signal handler which kicks the vcpu running on the current thread, if there is
    /// one.
    fn register_kick_signal_handler(&mut self) {
        extern "C" fn handle_signal(_: c_int, _: *mut siginfo_t, _: *mut c_void) {
            // We write to the immediate_exit from other thread, so make sure the read in the
            // KVM_RUN sees the up to date value
            fence(Ordering::Acquire);
        }
        register_signal_handler(sigrtmin() + VCPU_RTSIG_OFFSET, handle_signal)
            .expect("Failed to register vcpu signal handler");
    }

    /// Constructs a new VCPU for `vm`.
    ///
    /// # Arguments
    ///
    /// * `index` - Represents the 0-based CPU index between [0, max vcpus).
    /// * `vm` - The vm to which this vcpu will get attached.
    /// * `exit_evt` - An `EventFd` that will be written into when this vcpu exits.
    pub fn new(index: u8, vm: &KvmVm, exit_evt: EventFd) -> Result<Self, VcpuError> {
        let (event_sender, event_receiver) = channel();
        let (response_sender, response_receiver) = channel();
        let kvm_vcpu = KvmVcpu::new(index, vm).unwrap();

        Ok(Vcpu {
            exit_evt,
            event_receiver,
            event_sender: Some(event_sender),
            response_receiver: Some(response_receiver),
            response_sender,
            #[cfg(feature = "gdb")]
            gdb_event: None,
            kvm_vcpu,
        })
    }

    /// Sets a MMIO bus for this vcpu.
    pub fn set_mmio_bus(&mut self, mmio_bus: Arc<Bus>) {
        self.kvm_vcpu.peripherals.mmio_bus = Some(mmio_bus);
    }

    /// Attaches the fields required for debugging
    #[cfg(feature = "gdb")]
    pub fn attach_debug_info(&mut self, gdb_event: Sender<usize>) {
        self.gdb_event = Some(gdb_event);
    }

    /// Obtains a copy of the VcpuFd
    pub fn copy_kvm_vcpu_fd(&self, vm: &KvmVm) -> Result<VcpuFd, CopyKvmFdError> {
        // SAFETY: We own this fd so it is considered safe to clone
        let r = unsafe { libc::dup(self.kvm_vcpu.fd.as_raw_fd()) };
        if r < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        // SAFETY: We assert this is a valid fd by checking the result from the dup
        unsafe { Ok(vm.fd().create_vcpu_from_rawfd(r)?) }
    }

    /// Moves the vcpu to its own thread and constructs a VcpuHandle.
    /// The handle can be used to control the remote vcpu.
    pub fn start_threaded(
        mut self,
        vm: &KvmVm,
        seccomp_filter: Arc<BpfProgram>,
        barrier: Arc<Barrier>,
    ) -> Result<VcpuHandle, StartThreadedError> {
        let event_sender = self.event_sender.take().expect("vCPU already started");
        let response_receiver = self.response_receiver.take().unwrap();
        let vcpu_fd = self
            .copy_kvm_vcpu_fd(vm)
            .map_err(StartThreadedError::CopyFd)?;
        let vcpu_thread = thread::Builder::new()
            .name(format!("fc_vcpu {}", self.kvm_vcpu.index))
            .spawn(move || {
                let filter = &*seccomp_filter;
                self.register_kick_signal_handler();
                // Synchronization to make sure thread local data is initialized.
                barrier.wait();
                self.run(filter);
            })
            .map_err(StartThreadedError::Spawn)?;

        Ok(VcpuHandle::new(
            event_sender,
            response_receiver,
            vcpu_fd,
            vcpu_thread,
        ))
    }

    /// Main loop of the vCPU thread.
    ///
    /// Runs the vCPU in KVM context in a loop. Handles KVM_EXITs then goes back in.
    /// Note that the state of the VCPU and associated VM must be setup first for this to do
    /// anything useful.
    pub fn run(&mut self, seccomp_filter: BpfProgramRef) {
        // Load seccomp filters for this vCPU thread.
        // Execution panics if filters cannot be loaded, use --no-seccomp if skipping filters
        // altogether is the desired behaviour.
        if let Err(err) = crate::seccomp::apply_filter(seccomp_filter) {
            panic!(
                "Failed to set the requested seccomp filters on vCPU {}: Error: {}",
                self.kvm_vcpu.index, err
            );
        }

        // Start running the machine state in the `Paused` state.
        let mut state = VcpuRunState::Paused;
        loop {
            state = match state {
                VcpuRunState::Running => self.running(),
                VcpuRunState::Paused => self.paused(),
                VcpuRunState::Finished => break,
            };
        }
    }

    // This is the main loop of the `Running` state.
    fn running(&mut self) -> VcpuRunState {
        // This loop is here just for optimizing the emulation path.
        // No point in ticking the state machine if there are no external events.
        loop {
            match self.run_emulation() {
                // Emulation ran successfully, continue.
                Ok(VcpuEmulation::Handled) => (),
                // Emulation was interrupted, check external events.
                Ok(VcpuEmulation::Interrupted) => break,
                // The guest requested a SHUTDOWN or RESET. This is ARM
                // specific. On x86 the i8042 emulation signals the main thread
                // directly without calling Vcpu::exit().
                Ok(VcpuEmulation::Stopped) => return self.exit(FcExitCode::Ok),
                // If the emulation requests a pause lets do this
                #[cfg(feature = "gdb")]
                Ok(VcpuEmulation::Paused) => {
                    #[cfg(target_arch = "x86_64")]
                    self.kvm_vcpu.kvmclock_ctrl();
                    return VcpuRunState::Paused;
                }
                // Emulation errors lead to vCPU exit.
                Err(_) => return self.exit(FcExitCode::GenericError),
            }
        }

        // By default don't change state.
        let mut state = VcpuRunState::Running;

        // Break this emulation loop on any transition request/external event.
        match self.event_receiver.try_recv() {
            // Running ---- Pause ----> Paused
            Ok(VcpuEvent::Pause) => {
                // Nothing special to do.
                self.response_sender
                    .send(VcpuResponse::Paused)
                    .expect("vcpu channel unexpectedly closed");

                #[cfg(target_arch = "x86_64")]
                self.kvm_vcpu.kvmclock_ctrl();

                // Move to 'paused' state.
                state = VcpuRunState::Paused;
            }
            Ok(VcpuEvent::Resume) => {
                self.response_sender
                    .send(VcpuResponse::Resumed)
                    .expect("vcpu channel unexpectedly closed");
            }
            // Saving or restoring state cannot be performed on a running vCPU.
            Ok(VcpuEvent::SaveState) | Ok(VcpuEvent::RestoreState(_)) => {
                self.response_sender
                    .send(VcpuResponse::NotAllowed(String::from(
                        "save/restore unavailable while running",
                    )))
                    .expect("vcpu channel unexpectedly closed");
            }
            Ok(VcpuEvent::CompleteIo) => {
                self.response_sender
                    .send(VcpuResponse::NotAllowed(String::from(
                        "I/O completion is unavailable while running",
                    )))
                    .expect("vcpu channel unexpectedly closed");
            }
            // DumpCpuConfig cannot be performed on a running Vcpu.
            Ok(VcpuEvent::DumpCpuConfig) => {
                self.response_sender
                    .send(VcpuResponse::NotAllowed(String::from(
                        "cpu config dump is unavailable while running",
                    )))
                    .expect("vcpu channel unexpectedly closed");
            }
            Ok(VcpuEvent::Finish) => return VcpuRunState::Finished,
            // Unhandled exit of the other end.
            Err(TryRecvError::Disconnected) => {
                // Move to 'exited' state.
                state = self.exit(FcExitCode::GenericError);
            }
            // All other events or lack thereof have no effect on current 'running' state.
            Err(TryRecvError::Empty) => (),
        }

        state
    }

    /// Completes pending userspace I/O without entering the guest.
    ///
    /// Completion can write guest RAM, so pending I/O must finish before reset reverts memory.
    fn complete_pending_io(&mut self) -> Result<(), VcpuError> {
        let immediate_exit = self.kvm_vcpu.fd.get_kvm_run().immediate_exit;
        self.kvm_vcpu.fd.set_kvm_immediate_exit(1);

        // Both x86 and arm64 kvm_arch_vcpu_ioctl_run() complete userspace I/O before honoring
        // immediate_exit. run_emulation() skips that ioctl when immediate_exit is set.
        let mut runs = 0;
        let result = loop {
            runs += 1;
            let emulation = match self.kvm_vcpu.fd.run() {
                Err(err) if err.errno() == libc::EINTR => break Ok(()),
                Err(err) => break Err(VcpuError::CompletePendingIo(err)),
                Ok(exit @ (VcpuExit::MmioRead(_, _) | VcpuExit::MmioWrite(_, _))) => {
                    handle_kvm_exit(&mut self.kvm_vcpu.peripherals, Ok(exit))
                }
                #[cfg(target_arch = "x86_64")]
                Ok(exit @ (VcpuExit::IoIn(_, _) | VcpuExit::IoOut(_, _))) => {
                    handle_kvm_exit(&mut self.kvm_vcpu.peripherals, Ok(exit))
                }
                Ok(_) => {
                    break Err(VcpuError::UnexpectedIoCompletionExit(
                        self.kvm_vcpu.fd.get_kvm_run().exit_reason,
                    ));
                }
            };
            if let Err(err) = emulation {
                break Err(err);
            }
            // Handle the last exit before stopping so a later RUN has valid response data.
            if runs == MAX_IO_COMPLETION_RUNS {
                break Err(VcpuError::IoCompletionLimit(MAX_IO_COMPLETION_RUNS));
            }
        };
        self.kvm_vcpu.fd.set_kvm_immediate_exit(immediate_exit);
        result
    }

    // This is the main loop of the `Paused` state.
    fn paused(&mut self) -> VcpuRunState {
        match self.event_receiver.recv() {
            // Paused ---- Resume ----> Running
            Ok(VcpuEvent::Resume) => {
                if self.kvm_vcpu.fd.get_kvm_run().immediate_exit == 1u8 {
                    warn!(
                        "Received a VcpuEvent::Resume message with immediate_exit enabled. \
                         immediate_exit was disabled before proceeding"
                    );
                    self.kvm_vcpu.fd.set_kvm_immediate_exit(0);
                }
                self.response_sender
                    .send(VcpuResponse::Resumed)
                    .expect("vcpu channel unexpectedly closed");
                // Move to 'running' state.
                VcpuRunState::Running
            }
            Ok(VcpuEvent::Pause) => {
                self.response_sender
                    .send(VcpuResponse::Paused)
                    .expect("vcpu channel unexpectedly closed");
                VcpuRunState::Paused
            }
            Ok(VcpuEvent::SaveState) => {
                // Save vcpu state.
                self.kvm_vcpu
                    .save_state()
                    .map(|vcpu_state| {
                        self.response_sender
                            .send(VcpuResponse::SavedState(Box::new(vcpu_state)))
                            .expect("vcpu channel unexpectedly closed");
                    })
                    .unwrap_or_else(|err| {
                        self.response_sender
                            .send(VcpuResponse::Error(VcpuError::VcpuResponse(err)))
                            .expect("vcpu channel unexpectedly closed");
                    });

                VcpuRunState::Paused
            }
            Ok(VcpuEvent::CompleteIo) => {
                let response = match self.complete_pending_io() {
                    Ok(()) => VcpuResponse::IoCompleted,
                    Err(err) => VcpuResponse::Error(err),
                };
                self.response_sender
                    .send(response)
                    .expect("vcpu channel unexpectedly closed");
                VcpuRunState::Paused
            }
            Ok(VcpuEvent::RestoreState(state)) => {
                // Reset completes old I/O before reverting RAM, with no intervening resume.
                // Keep this defensive completion before any saved KVM_SET operation.
                let result = self.complete_pending_io().and_then(|()| {
                    // On x86_64, snapshot load already set the TSC frequency, which nothing
                    // changes afterwards, so only the saved state needs restoring.
                    #[cfg(target_arch = "x86_64")]
                    let result = self.kvm_vcpu.restore_state(&state);
                    #[cfg(target_arch = "aarch64")]
                    let result = self.kvm_vcpu.restore_state_in_place(&state);
                    result.map_err(VcpuError::VcpuResponse)
                });
                let response = match result {
                    Ok(()) => VcpuResponse::RestoredState,
                    Err(err) => VcpuResponse::Error(err),
                };
                self.response_sender
                    .send(response)
                    .expect("vcpu channel unexpectedly closed");
                VcpuRunState::Paused
            }
            Ok(VcpuEvent::DumpCpuConfig) => {
                self.kvm_vcpu
                    .dump_cpu_config()
                    .map(|cpu_config| {
                        self.response_sender
                            .send(VcpuResponse::DumpedCpuConfig(Box::new(cpu_config)))
                            .expect("vcpu channel unexpectedly closed");
                    })
                    .unwrap_or_else(|err| {
                        self.response_sender
                            .send(VcpuResponse::Error(VcpuError::VcpuResponse(err)))
                            .expect("vcpu channel unexpectedly closed");
                    });

                VcpuRunState::Paused
            }
            Ok(VcpuEvent::Finish) => VcpuRunState::Finished,
            // Unhandled exit of the other end.
            Err(_) => {
                // Move to 'exited' state.
                self.exit(FcExitCode::GenericError)
            }
        }
    }

    // Transition to the exited state and finish on command.
    // Note that this function isn't called when the guest asks for a CPU
    // reset via the i8042 controller on x86.
    fn exit(&mut self, exit_code: FcExitCode) -> VcpuRunState {
        if let Err(err) = self.exit_evt.write(1) {
            METRICS.vcpu.failures.inc();
            error!("Failed signaling vcpu exit event: {}", err);
        }
        // From this state we only accept going to finished.
        loop {
            self.response_sender
                .send(VcpuResponse::Exited(exit_code))
                .expect("vcpu channel unexpectedly closed");
            // Wait for and only accept 'VcpuEvent::Finish'.
            if let Ok(VcpuEvent::Finish) = self.event_receiver.recv() {
                break;
            }
        }
        VcpuRunState::Finished
    }

    /// Runs the vCPU in KVM context and handles the kvm exit reason.
    ///
    /// Returns error or enum specifying whether emulation was handled or interrupted.
    pub fn run_emulation(&mut self) -> Result<VcpuEmulation, VcpuError> {
        if self.kvm_vcpu.fd.get_kvm_run().immediate_exit == 1u8 {
            warn!("Requested a vCPU run with immediate_exit enabled. The operation was skipped");
            self.kvm_vcpu.fd.set_kvm_immediate_exit(0);
            return Ok(VcpuEmulation::Interrupted);
        }

        match self.kvm_vcpu.fd.run() {
            Err(ref err) if err.errno() == libc::EINTR => {
                self.kvm_vcpu.fd.set_kvm_immediate_exit(0);
                // Notify that this KVM_RUN was interrupted.
                Ok(VcpuEmulation::Interrupted)
            }
            #[cfg(feature = "gdb")]
            Ok(VcpuExit::Debug(_)) => {
                if let Some(gdb_event) = &self.gdb_event {
                    gdb_event
                        .send(get_raw_tid(self.kvm_vcpu.index.into()))
                        .expect("Unable to notify gdb event");
                }

                Ok(VcpuEmulation::Paused)
            }
            emulation_result => handle_kvm_exit(&mut self.kvm_vcpu.peripherals, emulation_result),
        }
    }
}

/// Handle the return value of a call to [`VcpuFd::run`] and update our emulation accordingly
fn handle_kvm_exit(
    peripherals: &mut Peripherals,
    emulation_result: Result<VcpuExit, errno::Error>,
) -> Result<VcpuEmulation, VcpuError> {
    match emulation_result {
        Ok(run) => match run {
            VcpuExit::MmioRead(addr, data) => {
                data.fill(0);
                if let Some(mmio_bus) = &peripherals.mmio_bus {
                    let _metric = METRICS.vcpu.exit_mmio_read_agg.record_latency_metrics();
                    if let Err(err) = mmio_bus.read(addr, data) {
                        warn!("Invalid MMIO read @ {addr:#x}:{:#x}: {err}", data.len());
                    }
                    METRICS.vcpu.exit_mmio_read.inc();
                }
                Ok(VcpuEmulation::Handled)
            }
            VcpuExit::MmioWrite(addr, data) => {
                if let Some(mmio_bus) = &peripherals.mmio_bus {
                    let _metric = METRICS.vcpu.exit_mmio_write_agg.record_latency_metrics();
                    if let Err(err) = mmio_bus.write(addr, data) {
                        warn!("Invalid MMIO read @ {addr:#x}:{:#x}: {err}", data.len());
                    }
                    METRICS.vcpu.exit_mmio_write.inc();
                }
                Ok(VcpuEmulation::Handled)
            }
            // Documentation specifies that below kvm exits are considered
            // errors.
            VcpuExit::FailEntry(hardware_entry_failure_reason, cpu) => {
                // Hardware entry failure.
                METRICS.vcpu.failures.inc();
                error!(
                    "Received KVM_EXIT_FAIL_ENTRY signal: {} on cpu {}",
                    hardware_entry_failure_reason, cpu
                );
                Err(VcpuError::FaultyKvmExit(format!(
                    "{:?}",
                    VcpuExit::FailEntry(hardware_entry_failure_reason, cpu)
                )))
            }
            VcpuExit::InternalError => {
                // Failure from the Linux KVM subsystem rather than from the hardware.
                METRICS.vcpu.failures.inc();
                error!("Received KVM_EXIT_INTERNAL_ERROR signal");
                Err(VcpuError::FaultyKvmExit(format!(
                    "{:?}",
                    VcpuExit::InternalError
                )))
            }
            VcpuExit::SystemEvent(event_type, event_flags) => match event_type {
                KVM_SYSTEM_EVENT_RESET | KVM_SYSTEM_EVENT_SHUTDOWN => {
                    info!(
                        "Received KVM_SYSTEM_EVENT: type: {}, event: {:?}",
                        event_type, event_flags
                    );
                    Ok(VcpuEmulation::Stopped)
                }
                _ => {
                    METRICS.vcpu.failures.inc();
                    error!(
                        "Received KVM_SYSTEM_EVENT signal type: {}, flag: {:?}",
                        event_type, event_flags
                    );
                    Err(VcpuError::FaultyKvmExit(format!(
                        "{:?}",
                        VcpuExit::SystemEvent(event_type, event_flags)
                    )))
                }
            },
            arch_specific_reason => {
                // run specific architecture emulation.
                peripherals.run_arch_emulation(arch_specific_reason)
            }
        },
        // The unwrap on raw_os_error can only fail if we have a logic
        // error in our code in which case it is better to panic.
        Err(ref err) => match err.errno() {
            libc::EAGAIN => Ok(VcpuEmulation::Handled),
            libc::ENOSYS => {
                METRICS.vcpu.failures.inc();
                error!("Received ENOSYS error because KVM failed to emulate an instruction.");
                Err(VcpuError::FaultyKvmExit(
                    "Received ENOSYS error because KVM failed to emulate an instruction."
                        .to_string(),
                ))
            }
            _ => {
                METRICS.vcpu.failures.inc();
                error!("Failure during vcpu run: {}", err);
                Err(VcpuError::FaultyKvmExit(format!("{}", err)))
            }
        },
    }
}

/// List of events that the Vcpu can receive.
#[derive(Debug)]
pub enum VcpuEvent {
    /// The vCPU thread will end when receiving this message.
    Finish,
    /// Pause the Vcpu.
    Pause,
    /// Event to resume the Vcpu.
    Resume,
    /// Event to save the state of a paused Vcpu.
    SaveState,
    /// Complete pending userspace I/O on a paused vCPU before reverting memory or devices.
    CompleteIo,
    /// Event to restore the state of a paused vCPU.
    RestoreState(Box<VcpuState>),
    /// Event to dump CPU configuration of a paused Vcpu.
    DumpCpuConfig,
}

/// List of responses that the Vcpu reports.
pub enum VcpuResponse {
    /// Requested action encountered an error.
    Error(VcpuError),
    /// Vcpu is stopped.
    Exited(FcExitCode),
    /// Requested action not allowed.
    NotAllowed(String),
    /// Vcpu is paused.
    Paused,
    /// Vcpu is resumed.
    Resumed,
    /// Vcpu state is saved.
    SavedState(Box<VcpuState>),
    /// Pending userspace I/O has completed without guest entry.
    IoCompleted,
    /// Vcpu state is restored.
    RestoredState,
    /// Vcpu is in the state where CPU config is dumped.
    DumpedCpuConfig(Box<CpuConfiguration>),
}

impl fmt::Debug for VcpuResponse {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use crate::VcpuResponse::*;
        match self {
            Paused => write!(f, "VcpuResponse::Paused"),
            Resumed => write!(f, "VcpuResponse::Resumed"),
            Exited(code) => write!(f, "VcpuResponse::Exited({:?})", code),
            SavedState(_) => write!(f, "VcpuResponse::SavedState"),
            IoCompleted => write!(f, "VcpuResponse::IoCompleted"),
            RestoredState => write!(f, "VcpuResponse::RestoredState"),
            Error(err) => write!(f, "VcpuResponse::Error({:?})", err),
            NotAllowed(reason) => write!(f, "VcpuResponse::NotAllowed({})", reason),
            DumpedCpuConfig(_) => write!(f, "VcpuResponse::DumpedCpuConfig"),
        }
    }
}

/// Wrapper over Vcpu that hides the underlying interactions with the Vcpu thread.
#[derive(Debug)]
pub struct VcpuHandle {
    event_sender: Sender<VcpuEvent>,
    response_receiver: Receiver<VcpuResponse>,
    /// VcpuFd
    pub vcpu_fd: VcpuFd,
    // Rust JoinHandles have to be wrapped in Option if you ever plan on 'join()'ing them.
    // We want to be able to join these threads in tests.
    vcpu_thread: Option<thread::JoinHandle<()>>,
}

/// Error type for [`VcpuHandle::send_event`].
#[derive(Debug, derive_more::From, thiserror::Error)]
#[error("Failed to signal vCPU: {0}")]
pub struct VcpuSendEventError(pub vmm_sys_util::errno::Error);

impl VcpuHandle {
    /// Creates a new [`VcpuHandle`].
    ///
    /// # Arguments
    /// + `event_sender`: [`Sender`] to communicate [`VcpuEvent`] to control the vcpu.
    /// + `response_received`: [`Received`] from which the vcpu's responses can be read.
    /// + `vcpu_thread`: A [`JoinHandle`] for the vcpu thread.
    pub fn new(
        event_sender: Sender<VcpuEvent>,
        response_receiver: Receiver<VcpuResponse>,
        vcpu_fd: VcpuFd,
        vcpu_thread: thread::JoinHandle<()>,
    ) -> Self {
        Self {
            event_sender,
            response_receiver,
            vcpu_fd,
            vcpu_thread: Some(vcpu_thread),
        }
    }
    /// Sends event to vCPU.
    ///
    /// # Errors
    ///
    /// When [`vmm_sys_util::linux::signal::Killable::kill`] errors.
    pub fn send_event(&mut self, event: VcpuEvent) -> Result<(), VcpuSendEventError> {
        // Use expect() to crash if the other thread closed this channel.
        self.event_sender
            .send(event)
            .expect("event sender channel closed on vcpu end.");
        // Kick the vcpu so it picks up the message.
        // Add a fence to ensure the write is visible to the vpu thread
        self.vcpu_fd.set_kvm_immediate_exit(1);
        fence(Ordering::Release);
        self.vcpu_thread
            .as_ref()
            // Safe to unwrap since constructor make this 'Some'.
            .unwrap()
            .kill(sigrtmin() + VCPU_RTSIG_OFFSET)?;
        Ok(())
    }

    /// Returns a reference to the [`Received`] from which the vcpu's responses can be read.
    pub fn response_receiver(&self) -> &Receiver<VcpuResponse> {
        &self.response_receiver
    }
}

// Wait for the Vcpu thread to finish execution
impl Drop for VcpuHandle {
    fn drop(&mut self) {
        // The vCPU thread owns the response sender, so the channel disconnects
        // once it exits. Wait for that disconnect (draining any stale responses)
        // with a timeout rather than joining unconditionally, so a thread that
        // never finished (e.g. a missed Finish event) fails fast instead of
        // hanging teardown forever.
        let thread = self.vcpu_thread.take().unwrap();
        loop {
            match self.response_receiver.recv_timeout(VCPU_JOIN_TIMEOUT) {
                // Sender dropped: the thread has exited.
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {
                    let name = thread.thread().name().unwrap_or("<unnamed>");
                    panic!("Timed out waiting for vCPU thread '{name}' to exit")
                }
                // Unexpected: a response was still queued at teardown. Discard
                // it and keep waiting for the thread to exit.
                Ok(response) => {
                    warn!("Discarding unexpected vCPU response during teardown: {response:?}");
                }
            }
        }
        thread.join().unwrap();
    }
}

/// Vcpu emulation state.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum VcpuEmulation {
    /// Handled.
    Handled,
    /// Interrupted.
    Interrupted,
    /// Stopped.
    Stopped,
    /// Pause request
    #[cfg(feature = "gdb")]
    Paused,
}

#[cfg(test)]
pub(crate) mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]

    #[cfg(target_arch = "x86_64")]
    use std::collections::BTreeMap;
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Barrier, Mutex};

    use linux_loader::loader::KernelLoader;
    use vm_memory::Bytes;
    use vmm_sys_util::errno;

    use super::*;
    use crate::RECV_TIMEOUT_SEC;
    use crate::arch::{BootProtocol, EntryPoint};
    use crate::seccomp::get_empty_filters;
    use crate::utils::mib_to_bytes;
    use crate::utils::signal::validate_signal_num;
    use crate::vstate::bus::BusDevice;
    use crate::vstate::memory::{GuestAddress, GuestMemoryMmap};
    use crate::vstate::vcpu::VcpuError as EmulationError;
    use crate::vstate::vm::tests::setup_vm_with_memory;

    struct DummyDevice;

    impl BusDevice for DummyDevice {
        fn read(&mut self, _base: u64, _offset: u64, _data: &mut [u8]) {}

        fn write(&mut self, _base: u64, _offset: u64, _data: &[u8]) -> Option<Arc<Barrier>> {
            None
        }
    }

    #[test]
    fn test_handle_kvm_exit() {
        let (_, mut vcpu) = setup_vcpu(0x1000);
        let res = handle_kvm_exit(&mut vcpu.kvm_vcpu.peripherals, Ok(VcpuExit::Hlt));
        assert!(matches!(
            res,
            Err(EmulationError::UnhandledKvmExit(s)) if s == "Hlt",
        ));

        let res = handle_kvm_exit(&mut vcpu.kvm_vcpu.peripherals, Ok(VcpuExit::Shutdown));
        assert!(matches!(
            res,
            Err(EmulationError::UnhandledKvmExit(s)) if s == "Shutdown",
        ));

        let res = handle_kvm_exit(
            &mut vcpu.kvm_vcpu.peripherals,
            Ok(VcpuExit::FailEntry(0, 0)),
        );
        assert_eq!(
            format!("{:?}", res.unwrap_err()),
            format!(
                "{:?}",
                EmulationError::FaultyKvmExit("FailEntry(0, 0)".to_string())
            )
        );

        let res = handle_kvm_exit(&mut vcpu.kvm_vcpu.peripherals, Ok(VcpuExit::InternalError));
        assert_eq!(
            format!("{:?}", res.unwrap_err()),
            format!(
                "{:?}",
                EmulationError::FaultyKvmExit("InternalError".to_string())
            )
        );

        let res = handle_kvm_exit(
            &mut vcpu.kvm_vcpu.peripherals,
            Ok(VcpuExit::SystemEvent(2, &[])),
        );
        assert_eq!(res.unwrap(), VcpuEmulation::Stopped);

        let res = handle_kvm_exit(
            &mut vcpu.kvm_vcpu.peripherals,
            Ok(VcpuExit::SystemEvent(1, &[])),
        );
        assert_eq!(res.unwrap(), VcpuEmulation::Stopped);

        let res = handle_kvm_exit(
            &mut vcpu.kvm_vcpu.peripherals,
            Ok(VcpuExit::SystemEvent(3, &[])),
        );
        assert_eq!(
            format!("{:?}", res.unwrap_err()),
            format!(
                "{:?}",
                EmulationError::FaultyKvmExit("SystemEvent(3, [])".to_string())
            )
        );

        // Check what happens with an unhandled exit reason.
        let res = handle_kvm_exit(&mut vcpu.kvm_vcpu.peripherals, Ok(VcpuExit::Unknown));
        assert_eq!(
            res.unwrap_err().to_string(),
            "Unexpected kvm exit received: Unknown".to_string()
        );

        let res = handle_kvm_exit(
            &mut vcpu.kvm_vcpu.peripherals,
            Err(errno::Error::new(libc::EAGAIN)),
        );
        assert_eq!(res.unwrap(), VcpuEmulation::Handled);

        let res = handle_kvm_exit(
            &mut vcpu.kvm_vcpu.peripherals,
            Err(errno::Error::new(libc::ENOSYS)),
        );
        assert_eq!(
            format!("{:?}", res.unwrap_err()),
            format!(
                "{:?}",
                EmulationError::FaultyKvmExit(
                    "Received ENOSYS error because KVM failed to emulate an instruction."
                        .to_string()
                )
            )
        );

        let res = handle_kvm_exit(
            &mut vcpu.kvm_vcpu.peripherals,
            Err(errno::Error::new(libc::EINVAL)),
        );
        assert_eq!(
            format!("{:?}", res.unwrap_err()),
            format!(
                "{:?}",
                EmulationError::FaultyKvmExit("Invalid argument (os error 22)".to_string())
            )
        );

        let bus = Arc::new(Bus::new());
        let dummy = Arc::new(Mutex::new(DummyDevice));
        bus.insert(dummy, 0x10, 0x10).unwrap();
        vcpu.set_mmio_bus(bus);
        let addr = 0x10;

        let res = handle_kvm_exit(
            &mut vcpu.kvm_vcpu.peripherals,
            Ok(VcpuExit::MmioRead(addr, &mut [0, 0, 0, 0])),
        );
        assert_eq!(res.unwrap(), VcpuEmulation::Handled);

        let res = handle_kvm_exit(
            &mut vcpu.kvm_vcpu.peripherals,
            Ok(VcpuExit::MmioWrite(addr, &[0, 0, 0, 0])),
        );
        assert_eq!(res.unwrap(), VcpuEmulation::Handled);
    }

    impl PartialEq for VcpuResponse {
        fn eq(&self, other: &Self) -> bool {
            use crate::VcpuResponse::*;
            // Guard match with no wildcard to make sure we catch new enum variants.
            match self {
                Paused | Resumed | IoCompleted | RestoredState | Exited(_) => (),
                Error(_) | NotAllowed(_) | SavedState(_) | DumpedCpuConfig(_) => (),
            };
            match (self, other) {
                (Paused, Paused) | (Resumed, Resumed) => true,
                (Exited(code), Exited(other_code)) => code == other_code,
                (NotAllowed(_), NotAllowed(_))
                | (SavedState(_), SavedState(_))
                | (IoCompleted, IoCompleted)
                | (RestoredState, RestoredState)
                | (DumpedCpuConfig(_), DumpedCpuConfig(_)) => true,
                (Error(err), Error(other_err)) => {
                    format!("{:?}", err) == format!("{:?}", other_err)
                }
                _ => false,
            }
        }
    }

    // Auxiliary function being used throughout the tests.
    #[allow(unused_mut)]
    pub(crate) fn setup_vcpu(mem_size: usize) -> (KvmVm, Vcpu) {
        let mut vm = setup_vm_with_memory(mem_size);

        let mut vcpus = vm.create_vcpus(1).unwrap();
        let mut vcpu = vcpus.remove(0);

        #[cfg(target_arch = "aarch64")]
        vcpu.kvm_vcpu.init(&[]).unwrap();

        (vm, vcpu)
    }

    fn load_good_kernel(vm_memory: &GuestMemoryMmap) -> GuestAddress {
        use std::fs::File;
        use std::path::PathBuf;

        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        #[cfg(target_arch = "x86_64")]
        path.push("src/test_utils/mock_resources/test_elf.bin");
        #[cfg(target_arch = "aarch64")]
        path.push("src/test_utils/mock_resources/test_pe.bin");

        let mut kernel_file = File::open(path).expect("Cannot open kernel file");

        #[cfg(target_arch = "x86_64")]
        let entry_addr = linux_loader::loader::elf::Elf::load(
            vm_memory,
            Some(GuestAddress(crate::arch::get_kernel_start())),
            &mut kernel_file,
            Some(GuestAddress(crate::arch::get_kernel_start())),
        )
        .unwrap();
        #[cfg(target_arch = "aarch64")]
        let entry_addr =
            linux_loader::loader::pe::PE::load(vm_memory, None, &mut kernel_file, None).unwrap();
        entry_addr.kernel_load
    }

    fn vcpu_configured_for_boot() -> (KvmVm, VcpuHandle, EventFd) {
        // Need enough mem to boot linux.
        let mem_size = mib_to_bytes(64);
        let (vm, mut vcpu) = setup_vcpu(mem_size);

        let vcpu_exit_evt = vcpu.exit_evt.try_clone().unwrap();

        // Needs a kernel since we'll actually run this vcpu.
        let entry_point = EntryPoint {
            entry_addr: load_good_kernel(vm.guest_memory()),
            protocol: BootProtocol::LinuxBoot,
            // `setup_header` is only a field of `EntryPoint` on x86_64.
            #[cfg(target_arch = "x86_64")]
            setup_header: None,
        };

        #[cfg(target_arch = "x86_64")]
        {
            use crate::cpu_config::x86_64::cpuid::Cpuid;
            let cpuid = Cpuid::try_from(vm.kvm().supported_cpuid.clone()).unwrap();
            let configured_cpuid = vcpu
                .kvm_vcpu
                .configure_cpuid(&cpuid, 1, false)
                .expect("failed to configure vcpu CPUID");
            vcpu.kvm_vcpu
                .configure_msrs_for_boot(&BTreeMap::new(), &configured_cpuid)
                .expect("failed to configure vcpu MSRs");
            vcpu.kvm_vcpu
                .configure_boot_state(vm.guest_memory(), entry_point)
                .expect("failed to configure vcpu");
        }

        #[cfg(target_arch = "aarch64")]
        vcpu.kvm_vcpu
            .configure(
                vm.guest_memory(),
                entry_point,
                &VcpuConfig {
                    vcpu_count: 1,
                    smt: false,
                    cpu_config: crate::cpu_config::aarch64::CpuConfiguration::default(),
                },
                &vm.kvm().optional_capabilities(),
            )
            .expect("failed to configure vcpu");

        let mut seccomp_filters = get_empty_filters();
        let barrier = Arc::new(Barrier::new(2));
        let vcpu_handle = vcpu
            .start_threaded(
                &vm,
                seccomp_filters.remove("vcpu").unwrap(),
                barrier.clone(),
            )
            .expect("failed to start vcpu");
        // Wait for vCPUs to initialize their TLS before moving forward.
        barrier.wait();

        (vm, vcpu_handle, vcpu_exit_evt)
    }

    #[test]
    fn test_set_mmio_bus() {
        let (_, mut vcpu) = setup_vcpu(0x1000);
        assert!(vcpu.kvm_vcpu.peripherals.mmio_bus.is_none());
        vcpu.set_mmio_bus(Arc::new(Bus::new()));
        assert!(vcpu.kvm_vcpu.peripherals.mmio_bus.is_some());
    }

    #[test]
    fn test_vcpu_kick() {
        let (vm, mut vcpu) = setup_vcpu(0x1000);

        let mut kvm_run =
            kvm_ioctls::KvmRunWrapper::mmap_from_fd(&vcpu.kvm_vcpu.fd, vm.fd().run_size())
                .expect("cannot mmap kvm-run");
        let vcpu_kvm_run =
            kvm_ioctls::KvmRunWrapper::mmap_from_fd(&vcpu.kvm_vcpu.fd, vm.fd().run_size())
                .expect("cannot mmap kvm-run");
        let success = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let vcpu_success = success.clone();
        let barrier = Arc::new(Barrier::new(2));
        let vcpu_barrier = barrier.clone();
        // Start Vcpu thread which will be kicked with a signal.
        let handle = std::thread::Builder::new()
            .name("test_vcpu_kick".to_string())
            .spawn(move || {
                vcpu.register_kick_signal_handler();
                // Notify TLS was populated.
                vcpu_barrier.wait();
                // Loop for max 1 second to check if the signal handler has run.
                for _ in 0..10 {
                    if vcpu_kvm_run.as_ref().immediate_exit == 1 {
                        // Signal handler has run and set immediate_exit to 1.
                        vcpu_success.store(true, Ordering::Release);
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            })
            .expect("cannot start thread");
        barrier.wait();

        // Set immediate_exit and kick the Vcpu using the custom signal.
        kvm_run.as_mut_ref().immediate_exit = 1;
        handle
            .kill(sigrtmin() + VCPU_RTSIG_OFFSET)
            .expect("failed to signal thread");
        handle.join().expect("failed to join thread");
        // Verify that the Vcpu saw its kvm immediate-exit as set.
        assert!(success.load(Ordering::Acquire));
    }

    // Sends an event to a vcpu and expects a particular response.
    fn queue_event_expect_response(
        handle: &mut VcpuHandle,
        event: VcpuEvent,
        response: VcpuResponse,
    ) {
        handle
            .send_event(event)
            .expect("failed to send event to vcpu");
        assert_eq!(
            handle
                .response_receiver()
                .recv_timeout(RECV_TIMEOUT_SEC)
                .expect("did not receive event response from vcpu"),
            response
        );
    }

    #[test]
    fn test_immediate_exit_shortcircuits_execution() {
        let (_, mut vcpu) = setup_vcpu(0x1000);

        vcpu.kvm_vcpu.fd.set_kvm_immediate_exit(1);
        // Set a dummy value to be returned by the emulate call
        let result = vcpu.run_emulation().expect("Failed to run emulation");
        assert_eq!(
            result,
            VcpuEmulation::Interrupted,
            "The Immediate Exit short-circuit should have prevented the execution of emulate"
        );

        let event_sender = vcpu.event_sender.take().expect("vCPU already started");
        let _ = event_sender.send(VcpuEvent::Resume);
        vcpu.kvm_vcpu.fd.set_kvm_immediate_exit(1);
        // paused is expected to coerce immediate_exit to 0 when receiving a VcpuEvent::Resume
        let _ = vcpu.paused();
        assert_eq!(
            0,
            vcpu.kvm_vcpu.fd.get_kvm_run().immediate_exit,
            "Immediate Exit should have been disabled by sending Resume to a paused VM"
        )
    }

    // The restore test changes one register to prove that the saved state reached KVM.
    #[cfg(target_arch = "x86_64")]
    fn marker_register(state: &VcpuState) -> u64 {
        state.regs.rax
    }

    #[cfg(target_arch = "x86_64")]
    fn set_marker_register(state: &mut VcpuState, value: u64) {
        state.regs.rax = value;
    }

    #[cfg(target_arch = "aarch64")]
    fn marker_register(state: &VcpuState) -> u64 {
        use crate::arch::aarch64::regs::PC;
        state.regs.iter().find(|reg| reg.id == PC).unwrap().value()
    }

    #[cfg(target_arch = "aarch64")]
    fn set_marker_register(state: &mut VcpuState, value: u64) {
        use crate::arch::aarch64::regs::PC;
        let mut pc = state.regs.iter_mut().find(|reg| reg.id == PC).unwrap();
        pc.set_value(value);
    }

    fn save_vcpu_state(handle: &mut VcpuHandle) -> Box<VcpuState> {
        handle.send_event(VcpuEvent::SaveState).unwrap();
        match handle
            .response_receiver()
            .recv_timeout(RECV_TIMEOUT_SEC)
            .unwrap()
        {
            VcpuResponse::SavedState(state) => state,
            response => panic!("unexpected vCPU response: {response:?}"),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[derive(Default)]
    struct IoCompletionState {
        reads: std::sync::atomic::AtomicUsize,
        writes: std::sync::atomic::AtomicUsize,
        written_byte: std::sync::atomic::AtomicU8,
    }

    #[cfg(target_arch = "x86_64")]
    struct IoCompletionDevice {
        state: Arc<IoCompletionState>,
    }

    #[cfg(target_arch = "x86_64")]
    impl BusDevice for IoCompletionDevice {
        fn read(&mut self, _base: u64, offset: u64, data: &mut [u8]) {
            self.state.reads.fetch_add(1, Ordering::Relaxed);
            for (index, byte) in data.iter_mut().enumerate() {
                *byte = u8::try_from(offset + index as u64 + 1).unwrap();
            }
        }

        fn write(&mut self, _base: u64, _offset: u64, data: &[u8]) -> Option<Arc<Barrier>> {
            assert_eq!(data.len(), 1);
            self.state.writes.fetch_add(1, Ordering::Relaxed);
            self.state.written_byte.store(data[0], Ordering::Relaxed);
            None
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn vcpu_with_pending_string_io(
        program: &[u8],
        count: u64,
        state: Arc<IoCompletionState>,
    ) -> (KvmVm, Vcpu, Arc<Mutex<IoCompletionDevice>>) {
        let (vm, mut vcpu) = setup_vcpu(0x1000);
        vm.guest_memory()
            .write_slice(program, GuestAddress(0))
            .unwrap();
        vm.guest_memory()
            .write_slice(&[0xcc; 32], GuestAddress(0x400))
            .unwrap();
        vcpu.kvm_vcpu
            .fd
            .set_cpuid2(&vm.kvm().supported_cpuid)
            .unwrap();
        let mut sregs = vcpu.kvm_vcpu.fd.get_sregs().unwrap();
        sregs.cs.base = 0;
        sregs.cs.selector = 0;
        sregs.ds.base = 0;
        sregs.ds.selector = 0;
        sregs.es.base = 0;
        sregs.es.selector = 0;
        vcpu.kvm_vcpu.fd.set_sregs(&sregs).unwrap();
        set_pending_io_registers(&mut vcpu, 0, 0);
        let mut regs = vcpu.kvm_vcpu.fd.get_regs().unwrap();
        regs.rsi = 0x2000;
        regs.rdi = 0x400;
        regs.rcx = count;
        vcpu.kvm_vcpu.fd.set_regs(&regs).unwrap();

        let device = Arc::new(Mutex::new(IoCompletionDevice { state }));
        let mmio_bus = Arc::new(Bus::new());
        mmio_bus.insert(device.clone(), 0x2000, 0x100).unwrap();
        vcpu.set_mmio_bus(mmio_bus);
        let pio_bus = Arc::new(Bus::new());
        pio_bus.insert(device.clone(), 0x1234, 1).unwrap();
        vcpu.kvm_vcpu.peripherals.pio_bus = Some(pio_bus);

        let exit = vcpu.kvm_vcpu.fd.run().unwrap();
        assert!(matches!(exit, VcpuExit::MmioRead(0x2000, _)), "{exit:?}");
        handle_kvm_exit(&mut vcpu.kvm_vcpu.peripherals, Ok(exit)).unwrap();
        (vm, vcpu, device)
    }

    #[cfg(target_arch = "x86_64")]
    fn complete_io_event(vcpu: &mut Vcpu) -> VcpuResponse {
        vcpu.event_sender
            .as_ref()
            .unwrap()
            .send(VcpuEvent::CompleteIo)
            .unwrap();
        assert!(matches!(vcpu.paused(), VcpuRunState::Paused));
        vcpu.response_receiver
            .as_ref()
            .unwrap()
            .recv_timeout(RECV_TIMEOUT_SEC)
            .unwrap()
    }

    #[derive(Clone, Copy, Debug)]
    enum PendingIo {
        MmioRead,
        MmioWrite,
        #[cfg(target_arch = "x86_64")]
        PioIn,
    }

    #[cfg(target_arch = "x86_64")]
    fn pending_io_program(io: PendingIo) -> &'static [u8] {
        match io {
            // Real-mode mov eax, [0x2000] and mov [0x2000], eax.
            PendingIo::MmioRead => &[0x66, 0xa1, 0x00, 0x20],
            PendingIo::MmioWrite => &[0x66, 0xa3, 0x00, 0x20],
            // Real-mode in eax, dx.
            PendingIo::PioIn => &[0x66, 0xed],
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn pending_io_program(io: PendingIo) -> &'static [u8] {
        match io {
            // ldr w1, [x0] and str w1, [x0].
            PendingIo::MmioRead => &[0x01, 0x00, 0x40, 0xb9],
            PendingIo::MmioWrite => &[0x01, 0x00, 0x00, 0xb9],
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn set_pending_io_registers(vcpu: &mut Vcpu, pc: u64, value: u64) {
        let mut regs = vcpu.kvm_vcpu.fd.get_regs().unwrap();
        regs.rip = pc;
        regs.rax = value;
        regs.rdx = 0x1234;
        regs.rflags = 2;
        vcpu.kvm_vcpu.fd.set_regs(&regs).unwrap();
    }

    #[cfg(target_arch = "aarch64")]
    fn set_pending_io_registers(vcpu: &mut Vcpu, pc: u64, value: u64) {
        use crate::arch::aarch64::regs::PC;

        vcpu.kvm_vcpu.fd.set_one_reg(PC, &pc.to_le_bytes()).unwrap();
        // Core register IDs for X0 (MMIO address) and X1 (I/O data).
        vcpu.kvm_vcpu
            .fd
            .set_one_reg(0x6030_0000_0010_0000, &0x2000_u64.to_le_bytes())
            .unwrap();
        vcpu.kvm_vcpu
            .fd
            .set_one_reg(0x6030_0000_0010_0002, &value.to_le_bytes())
            .unwrap();
    }

    #[cfg(target_arch = "x86_64")]
    fn pending_io_registers(vcpu: &Vcpu) -> (u64, u64) {
        let regs = vcpu.kvm_vcpu.fd.get_regs().unwrap();
        (regs.rip, regs.rax)
    }

    #[cfg(target_arch = "aarch64")]
    fn pending_io_registers(vcpu: &Vcpu) -> (u64, u64) {
        use crate::arch::aarch64::regs::PC;

        let mut pc = [0; 8];
        let mut value = [0; 8];
        vcpu.kvm_vcpu.fd.get_one_reg(PC, &mut pc).unwrap();
        vcpu.kvm_vcpu
            .fd
            .get_one_reg(0x6030_0000_0010_0002, &mut value)
            .unwrap();
        (u64::from_le_bytes(pc), u64::from_le_bytes(value))
    }

    fn check_restore_completes_pending_io(
        io: PendingIo,
        saved_pc: u64,
        saved_mp_state: u32,
        immediate_exit: u8,
    ) {
        const SAVED_VALUE: u64 = 0x1122_3344;
        const IO_VALUE: u32 = 0xdead_beef;
        let (vm, mut vcpu) = setup_vcpu(0x1000);
        let program = pending_io_program(io);
        // A mistaken guest entry exits on another I/O instruction instead of hanging.
        for pc in [
            0,
            program.len() as u64,
            saved_pc,
            saved_pc + program.len() as u64,
        ] {
            vm.guest_memory()
                .write_slice(program, GuestAddress(pc))
                .unwrap();
        }
        #[cfg(target_arch = "x86_64")]
        {
            vcpu.kvm_vcpu
                .fd
                .set_cpuid2(&vm.kvm().supported_cpuid)
                .unwrap();
            let mut sregs = vcpu.kvm_vcpu.fd.get_sregs().unwrap();
            sregs.cs.base = 0;
            sregs.cs.selector = 0;
            sregs.ds.base = 0;
            sregs.ds.selector = 0;
            vcpu.kvm_vcpu.fd.set_sregs(&sregs).unwrap();
        }
        set_pending_io_registers(&mut vcpu, saved_pc, SAVED_VALUE);

        // Complete first-RUN setup before capturing the clean snapshot, without guest entry.
        vcpu.kvm_vcpu.fd.set_kvm_immediate_exit(1);
        assert_eq!(vcpu.kvm_vcpu.fd.run().unwrap_err().errno(), libc::EINTR);
        vcpu.kvm_vcpu.fd.set_kvm_immediate_exit(0);
        vcpu.kvm_vcpu
            .fd
            .set_mp_state(kvm_bindings::kvm_mp_state {
                mp_state: saved_mp_state,
            })
            .unwrap();
        let state = vcpu.kvm_vcpu.save_state().unwrap();

        vcpu.kvm_vcpu
            .fd
            .set_mp_state(kvm_bindings::kvm_mp_state {
                mp_state: kvm_bindings::KVM_MP_STATE_RUNNABLE,
            })
            .unwrap();
        set_pending_io_registers(&mut vcpu, 0, u64::from(IO_VALUE));
        match (io, vcpu.kvm_vcpu.fd.run().unwrap()) {
            (PendingIo::MmioRead, VcpuExit::MmioRead(address, data)) => {
                assert_eq!(address, 0x2000);
                data.copy_from_slice(&IO_VALUE.to_le_bytes());
            }
            (PendingIo::MmioWrite, VcpuExit::MmioWrite(address, data)) => {
                assert_eq!(address, 0x2000);
                assert_eq!(data, IO_VALUE.to_le_bytes());
            }
            #[cfg(target_arch = "x86_64")]
            (PendingIo::PioIn, VcpuExit::IoIn(port, data)) => {
                assert_eq!(port, 0x1234);
                data.copy_from_slice(&IO_VALUE.to_le_bytes());
            }
            (_, exit) => panic!("unexpected pending {io:?} exit: {exit:?}"),
        }

        // Leave the response pending, including when the ARM vCPU has been powered off.
        vcpu.kvm_vcpu
            .fd
            .set_mp_state(kvm_bindings::kvm_mp_state {
                mp_state: saved_mp_state,
            })
            .unwrap();
        vcpu.kvm_vcpu.fd.set_kvm_immediate_exit(immediate_exit);
        vcpu.event_sender
            .as_ref()
            .unwrap()
            .send(VcpuEvent::RestoreState(Box::new(state)))
            .unwrap();
        assert!(matches!(vcpu.paused(), VcpuRunState::Paused));
        let response = vcpu
            .response_receiver
            .as_ref()
            .unwrap()
            .recv_timeout(RECV_TIMEOUT_SEC)
            .unwrap();
        assert!(
            matches!(response, VcpuResponse::RestoredState),
            "{response:?}"
        );
        assert_eq!(
            vcpu.kvm_vcpu.fd.get_kvm_run().immediate_exit,
            immediate_exit
        );
        assert_eq!(
            vcpu.kvm_vcpu.fd.get_mp_state().unwrap().mp_state,
            saved_mp_state
        );
        assert_eq!(pending_io_registers(&vcpu), (saved_pc, SAVED_VALUE));

        // A leftover completion would overwrite the restored PC/data before returning EINTR.
        vcpu.kvm_vcpu.fd.set_kvm_immediate_exit(1);
        assert_eq!(vcpu.kvm_vcpu.fd.run().unwrap_err().errno(), libc::EINTR);
        assert_eq!(pending_io_registers(&vcpu), (saved_pc, SAVED_VALUE));
    }

    #[test]
    fn test_vcpu_restore_state_completes_pending_mmio_read() {
        check_restore_completes_pending_io(
            PendingIo::MmioRead,
            0x100,
            kvm_bindings::KVM_MP_STATE_RUNNABLE,
            1,
        );
    }

    #[test]
    fn test_vcpu_restore_state_completes_pending_mmio_write() {
        check_restore_completes_pending_io(
            PendingIo::MmioWrite,
            0x100,
            kvm_bindings::KVM_MP_STATE_RUNNABLE,
            0,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vcpu_restore_state_completes_pending_pio() {
        // Fast PIO completion checks RIP, so use an earlier snapshot at the same instruction.
        check_restore_completes_pending_io(
            PendingIo::PioIn,
            0,
            kvm_bindings::KVM_MP_STATE_RUNNABLE,
            0,
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_vcpu_restore_state_completes_pending_mmio_stopped() {
        check_restore_completes_pending_io(
            PendingIo::MmioRead,
            0x100,
            kvm_bindings::KVM_MP_STATE_STOPPED,
            1,
        );
    }

    #[test]
    fn test_vcpu_complete_io_events() {
        let (_vm, mut handle, _) = vcpu_configured_for_boot();
        queue_event_expect_response(
            &mut handle,
            VcpuEvent::CompleteIo,
            VcpuResponse::IoCompleted,
        );
        queue_event_expect_response(&mut handle, VcpuEvent::Resume, VcpuResponse::Resumed);
        queue_event_expect_response(
            &mut handle,
            VcpuEvent::CompleteIo,
            VcpuResponse::NotAllowed(String::new()),
        );
        queue_event_expect_response(&mut handle, VcpuEvent::Pause, VcpuResponse::Paused);
        queue_event_expect_response(
            &mut handle,
            VcpuEvent::CompleteIo,
            VcpuResponse::IoCompleted,
        );
        handle.send_event(VcpuEvent::Finish).unwrap();
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vcpu_complete_io_handles_pio_continuation() {
        // outsb from MMIO first exits for a memory read, then for the port write.
        // The following MMIO loop makes an unintended guest entry return to userspace.
        let state = Arc::new(IoCompletionState::default());
        let (_vm, mut vcpu, _device) = vcpu_with_pending_string_io(
            &[0x6e, 0x66, 0xa1, 0x00, 0x20, 0xeb, 0xfa],
            1,
            state.clone(),
        );
        let response = complete_io_event(&mut vcpu);
        assert!(
            matches!(response, VcpuResponse::IoCompleted),
            "{response:?}"
        );
        assert_eq!(state.writes.load(Ordering::Relaxed), 1);
        assert_eq!(state.written_byte.load(Ordering::Relaxed), 1);
        assert_eq!(vcpu.kvm_vcpu.fd.get_regs().unwrap().rip, 1);
        assert_eq!(vcpu.kvm_vcpu.fd.get_kvm_run().immediate_exit, 0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vcpu_complete_io_limit_keeps_response_valid() {
        // REP MOVSB reads MMIO and writes RAM one byte at a time without guest re-entry.
        // The following MMIO loop makes an unintended guest entry return to userspace.
        let state = Arc::new(IoCompletionState::default());
        let (vm, mut vcpu, _device) = vcpu_with_pending_string_io(
            &[0xf3, 0xa4, 0x66, 0xa1, 0x00, 0x20, 0xeb, 0xfa],
            32,
            state.clone(),
        );
        let response = complete_io_event(&mut vcpu);
        assert!(
            matches!(
                response,
                VcpuResponse::Error(VcpuError::IoCompletionLimit(16))
            ),
            "{response:?}"
        );
        assert_eq!(vcpu.kvm_vcpu.fd.get_kvm_run().immediate_exit, 0);

        let mut copied = [0; 32];
        vm.guest_memory()
            .read_slice(&mut copied, GuestAddress(0x400))
            .unwrap();
        let expected: [u8; 32] = std::array::from_fn(|index| u8::try_from(index + 1).unwrap());
        assert_eq!(&copied[..16], &expected[..16]);
        assert_eq!(&copied[16..], &[0xcc; 16]);
        // The first exit was handled before CompleteIo; the sixteenth continuation also has data.
        assert_eq!(state.reads.load(Ordering::Relaxed), 17);

        // A later completion consumes that last response, rather than stale data from byte 16.
        vcpu.kvm_vcpu.fd.set_kvm_immediate_exit(1);
        let response = complete_io_event(&mut vcpu);
        assert!(
            matches!(response, VcpuResponse::IoCompleted),
            "{response:?}"
        );
        assert_eq!(vcpu.kvm_vcpu.fd.get_kvm_run().immediate_exit, 1);
        vm.guest_memory()
            .read_slice(&mut copied, GuestAddress(0x400))
            .unwrap();
        assert_eq!(copied, expected);
        assert_eq!(state.reads.load(Ordering::Relaxed), 32);

        // Normal execution must finish REP and reach the following 32-bit MMIO load.
        vcpu.kvm_vcpu.fd.set_kvm_immediate_exit(0);
        let exit = vcpu.kvm_vcpu.fd.run().unwrap();
        match &exit {
            VcpuExit::MmioRead(address, data) => {
                assert_eq!(*address, 0x2000);
                assert_eq!(data.len(), 4);
            }
            _ => panic!("unexpected exit after completing REP MOVSB: {exit:?}"),
        }
        handle_kvm_exit(&mut vcpu.kvm_vcpu.peripherals, Ok(exit)).unwrap();
        let regs = vcpu.kvm_vcpu.fd.get_regs().unwrap();
        assert_eq!(regs.rip, 2);
        assert_eq!(regs.rcx, 0);
        assert_eq!(state.reads.load(Ordering::Relaxed), 33);
        vm.guest_memory()
            .read_slice(&mut copied, GuestAddress(0x400))
            .unwrap();
        assert_eq!(copied, expected);
    }

    #[test]
    fn test_vcpu_restore_state_events() {
        let (_vm, mut handle, _) = vcpu_configured_for_boot();

        let mut state = save_vcpu_state(&mut handle);
        set_marker_register(&mut state, 0x1234);
        queue_event_expect_response(
            &mut handle,
            VcpuEvent::RestoreState(state),
            VcpuResponse::RestoredState,
        );
        let state = save_vcpu_state(&mut handle);
        assert_eq!(marker_register(&state), 0x1234);

        // A running vCPU refuses the restore.
        queue_event_expect_response(&mut handle, VcpuEvent::Resume, VcpuResponse::Resumed);
        queue_event_expect_response(
            &mut handle,
            VcpuEvent::RestoreState(state),
            VcpuResponse::NotAllowed(String::new()),
        );
        handle.send_event(VcpuEvent::Finish).unwrap();
    }

    #[test]
    fn test_vcpu_pause_resume() {
        let (_vm, mut vcpu_handle, vcpu_exit_evt) = vcpu_configured_for_boot();

        // Queue a Resume event, expect a response.
        queue_event_expect_response(&mut vcpu_handle, VcpuEvent::Resume, VcpuResponse::Resumed);

        // Queue a Pause event, expect a response.
        queue_event_expect_response(&mut vcpu_handle, VcpuEvent::Pause, VcpuResponse::Paused);

        // Validate vcpu handled the EINTR gracefully and didn't exit.
        let err = vcpu_exit_evt.read().unwrap_err();
        assert_eq!(err.raw_os_error().unwrap(), libc::EAGAIN);

        // Queue another Pause event, expect a response.
        queue_event_expect_response(&mut vcpu_handle, VcpuEvent::Pause, VcpuResponse::Paused);

        // Queue a Resume event, expect a response.
        queue_event_expect_response(&mut vcpu_handle, VcpuEvent::Resume, VcpuResponse::Resumed);

        // Queue another Resume event, expect a response.
        queue_event_expect_response(&mut vcpu_handle, VcpuEvent::Resume, VcpuResponse::Resumed);

        // Queue another Pause event, expect a response.
        queue_event_expect_response(&mut vcpu_handle, VcpuEvent::Pause, VcpuResponse::Paused);

        // Queue a Resume event, expect a response.
        queue_event_expect_response(&mut vcpu_handle, VcpuEvent::Resume, VcpuResponse::Resumed);

        vcpu_handle.send_event(VcpuEvent::Finish).unwrap();
    }

    #[test]
    fn test_vcpu_save_state_events() {
        let (_vm, mut vcpu_handle, _vcpu_exit_evt) = vcpu_configured_for_boot();

        // Queue a Resume event, expect a response.
        queue_event_expect_response(&mut vcpu_handle, VcpuEvent::Resume, VcpuResponse::Resumed);

        // Queue a SaveState event, expect a response.
        queue_event_expect_response(
            &mut vcpu_handle,
            VcpuEvent::SaveState,
            VcpuResponse::NotAllowed(String::new()),
        );

        // Queue another Pause event, expect a response.
        queue_event_expect_response(&mut vcpu_handle, VcpuEvent::Pause, VcpuResponse::Paused);

        // Queue a SaveState event, get the response.
        vcpu_handle
            .send_event(VcpuEvent::SaveState)
            .expect("failed to send event to vcpu");
        match vcpu_handle
            .response_receiver()
            .recv_timeout(RECV_TIMEOUT_SEC)
            .expect("did not receive event response from vcpu")
        {
            VcpuResponse::SavedState(_) => {}
            _ => panic!("unexpected response"),
        };

        vcpu_handle.send_event(VcpuEvent::Finish).unwrap();
    }

    #[test]
    fn test_vcpu_dump_cpu_config() {
        let (_vm, mut vcpu_handle, _) = vcpu_configured_for_boot();

        // Queue a DumpCpuConfig event, expect a DumpedCpuConfig response.
        vcpu_handle
            .send_event(VcpuEvent::DumpCpuConfig)
            .expect("Failed to send an event to vcpu.");
        match vcpu_handle
            .response_receiver()
            .recv_timeout(RECV_TIMEOUT_SEC)
            .expect("Could not receive a response from vcpu.")
        {
            VcpuResponse::DumpedCpuConfig(_) => (),
            VcpuResponse::Error(err) => panic!("Got an error: {err}"),
            _ => panic!("Got an unexpected response."),
        }

        // Queue a Resume event, expect a response.
        queue_event_expect_response(&mut vcpu_handle, VcpuEvent::Resume, VcpuResponse::Resumed);

        // Queue a DumpCpuConfig event, expect a NotAllowed respoonse.
        // The DumpCpuConfig event is only allowed while paused.
        queue_event_expect_response(
            &mut vcpu_handle,
            VcpuEvent::DumpCpuConfig,
            VcpuResponse::NotAllowed(String::new()),
        );

        vcpu_handle.send_event(VcpuEvent::Finish).unwrap();
    }

    #[test]
    fn test_vcpu_rtsig_offset() {
        validate_signal_num(sigrtmin() + VCPU_RTSIG_OFFSET).unwrap();
    }
}
