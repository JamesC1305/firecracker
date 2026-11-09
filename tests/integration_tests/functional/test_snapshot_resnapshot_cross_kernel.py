# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore and re-snapshot artifacts while retaining their vCPU register set."""

import pytest

from framework.artifacts import get_snapshot_dirs
from framework.microvm import Snapshot
from framework.vm_backend import VM_BACKEND_KVM, pin_vm_backend

pytestmark = [pytest.mark.nonci, pin_vm_backend(VM_BACKEND_KVM)]


@pytest.mark.timeout(600)
@pytest.mark.parametrize("snapshot_dir", get_snapshot_dirs())
def test_snap_resnapshot_from_artifacts(microvm_factory, snapshot_dir, results_dir):
    """Restore and re-snapshot a pinned artifact twice, keeping its register set.

    The artifact carries the pin, so nothing is passed here. The second
    generation restores a snapshot taken on this host, which checks that the
    pin is inherited.
    """
    snapshot = Snapshot.load_from(snapshot_dir)
    for _ in range(2):
        vm = microvm_factory.build(monitor_memory=False)
        vm.time_api_requests = False
        vm.spawn(log_level="Info")
        vm.restore_from_snapshot(snapshot)
        vm.resume()

        assert vm.state == "Running"
        for idx in range(len(vm.iface)):
            vm.ssh_iface(idx).check_output("true")

        # Record MONOTONIC before each snapshot, so the restore test measures
        # only the final restore hop.
        vm.ssh.check_output(
            "python3 -c 'import time; print(time.monotonic())' > /tmp/monotonic-before"
        )
        snapshot = vm.snapshot_full()
        vm.kill()

    snapshot_artifacts_dir = results_dir / snapshot_dir.name
    snapshot_artifacts_dir.mkdir(parents=True)
    snapshot.save_to(snapshot_artifacts_dir)
