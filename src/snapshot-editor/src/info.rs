// Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use clap::Subcommand;
use vmm::arch::VcpuState;
use vmm::persist::MicrovmState;
use vmm::snapshot::Snapshot;

use crate::utils::*;

#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum InfoVmStateError {
    /// {0}
    Utils(#[from] UtilsError),
    /// vCPU registers are not pinned in {0:?}.
    NotPinned(PathBuf),
    /// Expected {0} vCPUs, found {1}.
    VcpuCountMismatch(usize, usize),
    /// Pinned register state differs on vCPU {0}.
    RegisterPinMismatch(usize),
}

#[derive(Debug, Subcommand)]
pub enum InfoVmStateSubCommand {
    /// Print snapshot version.
    Version {
        /// Path to the vmstate file.
        #[arg(short, long)]
        vmstate_path: PathBuf,
    },
    /// Print info about vcpu states.
    VcpuStates {
        /// Path to the vmstate file.
        #[arg(short, long)]
        vmstate_path: PathBuf,
    },
    /// Check that two vmstates pin the same ordered register IDs and Arm Feature ID values.
    CheckVcpuRegisterPin {
        /// Path to the original pinned vmstate file.
        #[arg(long)]
        source_path: PathBuf,
        /// Path to the re-snapshotted vmstate file.
        #[arg(short, long)]
        vmstate_path: PathBuf,
    },
    /// Print readable MicroVM state.
    VmState {
        /// Path to the vmstate file.
        #[arg(short, long)]
        vmstate_path: PathBuf,
    },
}

pub fn info_vmstate_command(command: InfoVmStateSubCommand) -> Result<(), InfoVmStateError> {
    match command {
        InfoVmStateSubCommand::Version { vmstate_path } => info(&vmstate_path, info_version)?,
        InfoVmStateSubCommand::VcpuStates { vmstate_path } => {
            info(&vmstate_path, info_vcpu_states)?
        }
        InfoVmStateSubCommand::CheckVcpuRegisterPin {
            source_path,
            vmstate_path,
        } => check_vcpu_register_pin(source_path, vmstate_path)?,
        InfoVmStateSubCommand::VmState { vmstate_path } => info(&vmstate_path, info_vmstate)?,
    }
    Ok(())
}

fn info(
    vmstate_path: &PathBuf,
    f: impl Fn(&Snapshot<MicrovmState>) -> Result<(), InfoVmStateError>,
) -> Result<(), InfoVmStateError> {
    let snapshot = open_vmstate(vmstate_path)?;
    f(&snapshot)?;
    Ok(())
}

fn info_version(snapshot: &Snapshot<MicrovmState>) -> Result<(), InfoVmStateError> {
    println!("v{}", snapshot.version());
    Ok(())
}

fn info_vcpu_states(snapshot: &Snapshot<MicrovmState>) -> Result<(), InfoVmStateError> {
    for (i, state) in snapshot.data.vcpu_states.iter().enumerate() {
        println!("vcpu {i}:");
        println!("{state:#?}");
    }
    Ok(())
}

fn pinned_vcpu_states(path: PathBuf) -> Result<Vec<VcpuState>, InfoVmStateError> {
    let state = open_vmstate(&path)?.data;
    if !state.vcpu_registers_pinned {
        return Err(InfoVmStateError::NotPinned(path));
    }
    Ok(state.vcpu_states)
}

#[cfg(target_arch = "x86_64")]
fn pinned_registers(vcpu: &VcpuState) -> impl Iterator<Item = u32> {
    vcpu.saved_msrs
        .iter()
        .flat_map(|chunk| chunk.as_slice())
        .map(|msr| msr.index)
}

#[cfg(target_arch = "aarch64")]
fn pinned_registers(vcpu: &VcpuState) -> impl Iterator<Item = (u64, Option<u64>)> {
    use vmm::arch::aarch64::regs::is_feature_id_reg;

    vcpu.regs.iter().map(|reg| {
        let value = is_feature_id_reg(reg.id).then(|| reg.value::<u64, 8>());
        (reg.id, value)
    })
}

fn check_vcpu_register_pin(
    source_path: PathBuf,
    vmstate_path: PathBuf,
) -> Result<(), InfoVmStateError> {
    let source = pinned_vcpu_states(source_path)?;
    let current = pinned_vcpu_states(vmstate_path)?;
    if source.len() != current.len() {
        return Err(InfoVmStateError::VcpuCountMismatch(
            source.len(),
            current.len(),
        ));
    }
    for (i, (expected, actual)) in source.iter().zip(&current).enumerate() {
        if pinned_registers(expected).ne(pinned_registers(actual)) {
            return Err(InfoVmStateError::RegisterPinMismatch(i));
        }
    }
    Ok(())
}

fn info_vmstate(snapshot: &Snapshot<MicrovmState>) -> Result<(), InfoVmStateError> {
    println!("{:#?}", snapshot.data);
    Ok(())
}
