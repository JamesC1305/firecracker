# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare fresh restore and in-place reset lifecycle and memory access costs."""

import signal
import time
from contextlib import closing
from pathlib import Path

from framework.artifacts import GUEST_KERNEL_DEFAULT, pin_guest_kernel
from framework.utils import start_fast_page_fault_helper
from host_tools.network import SSHConnection
from integration_tests.performance import test_snapshot

ITERATIONS = 10
MS_PER_SECOND = 1000
NS_PER_MS = 1_000_000

pytestmark = pin_guest_kernel(GUEST_KERNEL_DEFAULT)


def _time_ms(operation, *args, **kwargs):
    """Time a synchronous host operation."""
    start = time.monotonic()
    operation(*args, **kwargs)
    return (time.monotonic() - start) * MS_PER_SECOND


def _minor_faults(vm):
    """Read minflt, field 10, allowing spaces in the process name."""
    stat = Path(f"/proc/{vm.firecracker_pid}/stat").read_text(encoding="utf-8")
    fields = stat.rsplit(")", 1)[1].split()
    return int(fields[7])


def _spawn_vm(microvm_factory, guest_kernel, rootfs, pci_enabled):
    """Include factory construction, jailer setup and API readiness in spawn."""
    start = time.monotonic()
    vm = microvm_factory.build(
        guest_kernel, rootfs, monitor_memory=False, pci=pci_enabled
    )
    vm.spawn(log_level="Info", emit_metrics=True)
    return vm, (time.monotonic() - start) * MS_PER_SECOND


def _probe_tcp(vm, snapshot):
    """Wait for port 22 with one network namespace invocation."""
    iface = snapshot.net_ifaces[0]
    vm.netns.check_output(f"ip neigh flush dev {iface.tap_name}")
    vm.netns.check_output(
        "bash -c 'until timeout 0.2 bash -c "
        f'"</dev/tcp/{iface.guest_ip}/22" 2>/dev/null; do :; done\''
    )


def _ready_and_workloads(vm, snapshot, helper_pid):
    """Probe SSH, read 256 MiB, then touch each page in a 128 MiB region."""
    values = {}
    start = time.monotonic()
    vm.netns.check_output(f"ip neigh flush dev {snapshot.net_ifaces[0].tap_name}")
    with closing(
        SSHConnection(
            netns=vm.netns.id,
            ssh_key=vm.ssh_key,
            control_path=Path(vm.chroot()) / "lifecycle-ssh.sock",
            host=snapshot.net_ifaces[0].guest_ip,
            user="root",
        )
    ) as ssh:
        ssh.check_output("true")
        values["ready_ms"] = (time.monotonic() - start) * MS_PER_SECOND

        before = _minor_faults(vm)
        _, duration, _ = ssh.check_output(
            "t0=$(date +%s%N); "
            "dd if=/dev/shm/lifecycle-blob of=/dev/null bs=1M status=none || exit 1; "
            't1=$(date +%s%N); echo "$((t1 - t0))"'
        )
        values["minflt_read"] = _minor_faults(vm) - before
        values["read_ms"] = int(duration) / NS_PER_MS

        before = _minor_faults(vm)
        ssh.check_output(f"kill -s {signal.SIGUSR1} {helper_pid}")
        _, duration, _ = ssh.check_output(
            "while [ ! -s /tmp/fast_page_fault_helper.out ]; do sleep 0.01; done; "
            "cat /tmp/fast_page_fault_helper.out"
        )
        values["minflt_write"] = _minor_faults(vm) - before
        values["fault_write_ms"] = int(duration) / NS_PER_MS
        ssh.check_output("true")
    return values


def _emit(metrics, path, iteration, values):
    """Keep each path's samples separate in the fixture's single metrics file."""
    for name, value in values.items():
        if name.startswith("minflt_"):
            assert value >= 0
            unit = "Count"
        else:
            assert value > 0
            unit = "Milliseconds"
        metrics.put_metric(f"{path}_{name}", value, unit)
    summary = " ".join(f"{name}={value:.3f}" for name, value in values.items())
    print(f"lifecycle path={path} iteration={iteration} {summary}", flush=True)


def test_reset_vs_fresh_restore(
    microvm_factory, guest_kernel, rootfs, pci_enabled, metrics
):
    """Measure lifecycle and guest memory access after load and reset."""
    source = test_snapshot.SnapshotRestoreTest(mem=1024, vcpus=2).boot_vm(
        microvm_factory, guest_kernel, rootfs, pci_enabled, track_dirty_pages=True
    )
    # MetricsWrapper retains only one dimension set, so path prefixes name metrics.
    metrics.set_dimensions(
        {"performance_test": "test_reset_vs_fresh_restore", **source.dimensions}
    )
    source.ssh.check_output(
        "mount -o remount,size=512M /dev/shm && "
        "dd if=/dev/zero of=/dev/shm/lifecycle-blob bs=1M count=256 status=none && sync"
    )
    helper_pid = start_fast_page_fault_helper(source.ssh)
    snapshot = source.snapshot_diff()
    source.kill()

    for iteration in range(ITERATIONS):
        vm, spawn_ms = _spawn_vm(microvm_factory, guest_kernel, rootfs, pci_enabled)
        values = {"spawn_ms": spawn_ms}
        # The framework hardlinks memory/vmstate into the jail on the same device,
        # otherwise copies them. It also prepares disks and tap devices here.
        values["load_ms"] = _time_ms(vm.restore_from_snapshot, snapshot, resume=False)
        values["load_fc_ms"] = (
            vm.flush_metrics()["latencies_us"]["load_snapshot"] / 1000
        )
        values["resume_ms"] = _time_ms(vm.resume)
        values["ready_tcp_ms"] = _time_ms(_probe_tcp, vm, snapshot)
        values.update(_ready_and_workloads(vm, snapshot, helper_pid))
        values["total_to_ready_ms"] = sum(
            values[name] for name in ("spawn_ms", "load_ms", "resume_ms", "ready_ms")
        )
        values["total_to_tcp_ms"] = sum(
            values[name]
            for name in ("spawn_ms", "load_ms", "resume_ms", "ready_tcp_ms")
        )
        values["teardown_ms"] = _time_ms(vm.kill)
        _emit(metrics, "fresh", iteration, values)

    vm, _ = _spawn_vm(microvm_factory, guest_kernel, rootfs, pci_enabled)
    values = {"load_ms": _time_ms(vm.restore_from_snapshot, snapshot, resume=False)}
    values["load_fc_ms"] = vm.flush_metrics()["latencies_us"]["load_snapshot"] / 1000
    values["resume_ms"] = _time_ms(vm.resume)
    values["ready_tcp_ms"] = _time_ms(_probe_tcp, vm, snapshot)
    values.update(_ready_and_workloads(vm, snapshot, helper_pid))
    values["total_to_ready_ms"] = sum(
        values[name] for name in ("load_ms", "resume_ms", "ready_ms")
    )
    values["total_to_tcp_ms"] = sum(
        values[name] for name in ("load_ms", "resume_ms", "ready_tcp_ms")
    )
    _emit(metrics, "reset_first_load", 0, values)

    for iteration in range(1, ITERATIONS + 1):
        values = {"pause_ms": _time_ms(vm.pause)}
        values["reset_ms"] = _time_ms(vm.api.snapshot_reset.put)
        values["reset_fc_ms"] = (
            vm.flush_metrics()["latencies_us"]["reset_snapshot"] / 1000
        )
        values["resume_ms"] = _time_ms(vm.resume)
        values["ready_tcp_ms"] = _time_ms(_probe_tcp, vm, snapshot)
        values.update(_ready_and_workloads(vm, snapshot, helper_pid))
        values["total_to_ready_ms"] = sum(
            values[name] for name in ("pause_ms", "reset_ms", "resume_ms", "ready_ms")
        )
        values["total_to_tcp_ms"] = sum(
            values[name]
            for name in ("pause_ms", "reset_ms", "resume_ms", "ready_tcp_ms")
        )
        _emit(metrics, "reset", iteration, values)
    vm.kill()
