# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Performance benchmarks for snapshot restore and reset."""

import re
import signal
import tempfile
import time
from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest

import host_tools.drive as drive_tools
from framework.artifacts import GUEST_KERNEL_DEFAULT, pin_guest_kernel
from framework.microvm import Microvm, SnapshotType
from framework.utils import start_fast_page_fault_helper
from framework.utils_hugepages import HugePagesConfig
from host_tools.network import SSHConnection

USEC_IN_MSEC = 1000
NS_IN_MSEC = 1_000_000
ITERATIONS = 30
RESET_ITERATIONS = 10

pytestmark = pin_guest_kernel(GUEST_KERNEL_DEFAULT)


@lru_cache
def get_scratch_drives():
    """Create an array of scratch disks."""
    scratchdisks = ["vdb", "vdc", "vdd", "vde"]
    return [
        (drive, drive_tools.FilesystemFile(tempfile.mktemp(), size=64))
        for drive in scratchdisks
    ]


@dataclass
class SnapshotRestoreTest:
    """Dataclass encapsulating properties of snapshot restore tests"""

    vcpus: int = 1
    mem: int = 128
    nets: int = 3
    blocks: int = 3
    all_devices: bool = False
    huge_pages: HugePagesConfig = HugePagesConfig.NONE

    @property
    def id(self):
        """Computes a unique id for this test instance"""
        return "all_dev" if self.all_devices else f"{self.vcpus}vcpu_{self.mem}mb"

    def boot_vm(
        self,
        microvm_factory,
        guest_kernel,
        rootfs,
        pci_enabled,
        *,
        track_dirty_pages=False,
    ) -> Microvm:
        """Creates the initial snapshot that will be loaded repeatedly to sample latencies"""
        vm = microvm_factory.build(
            guest_kernel,
            rootfs,
            monitor_memory=False,
            pci=pci_enabled,
        )
        vm.spawn(log_level="Info", emit_metrics=True)
        vm.time_api_requests = False
        vm.basic_config(
            vcpu_count=self.vcpus,
            mem_size_mib=self.mem,
            rootfs_io_engine="Sync",
            huge_pages=self.huge_pages,
            track_dirty_pages=track_dirty_pages,
        )

        for _ in range(self.nets):
            vm.add_net_iface()

        if self.blocks > 1:
            scratch_drives = get_scratch_drives()
            for name, diskfile in scratch_drives[: (self.blocks - 1)]:
                vm.add_drive(name, diskfile.path, io_engine="Sync")

        if self.all_devices:
            vm.api.balloon.put(
                amount_mib=0, deflate_on_oom=True, stats_polling_interval_s=1
            )
            vm.api.vsock.put(vsock_id="vsock0", guest_cid=3, uds_path="/v.sock")

        vm.start()

        return vm


@pytest.mark.nonci
@pytest.mark.parametrize(
    "test_setup",
    [
        SnapshotRestoreTest(mem=128, vcpus=1),
        SnapshotRestoreTest(mem=1024, vcpus=1),
        SnapshotRestoreTest(mem=2048, vcpus=2),
        SnapshotRestoreTest(mem=4096, vcpus=3),
        SnapshotRestoreTest(mem=6144, vcpus=4),
        SnapshotRestoreTest(mem=8192, vcpus=5),
        SnapshotRestoreTest(mem=10240, vcpus=6),
        SnapshotRestoreTest(mem=12288, vcpus=7),
        SnapshotRestoreTest(all_devices=True),
    ],
    ids=lambda x: x.id,
)
def test_restore_latency(
    microvm_factory, guest_kernel, rootfs, pci_enabled, test_setup, metrics
):
    """
    Restores snapshots with vcpu/memory configuration, roughly scaling according to mem = (vcpus - 1) * 2048MB,
    which resembles firecracker production setups. Also contains a test case for restoring a snapshot will all devices
    attached to it.

    We only test a single guest kernel, as the guest kernel does not "participate" in snapshot restore.
    """
    vm = test_setup.boot_vm(microvm_factory, guest_kernel, rootfs, pci_enabled)

    metrics.set_dimensions(
        {
            "net_devices": str(test_setup.nets),
            "block_devices": str(test_setup.blocks),
            "vsock_devices": str(int(test_setup.all_devices)),
            "balloon_devices": str(int(test_setup.all_devices)),
            "huge_pages_config": str(test_setup.huge_pages),
            "performance_test": "test_restore_latency",
            "uffd_handler": "None",
            **vm.dimensions,
        }
    )

    snapshot = vm.snapshot_full()
    vm.kill()
    for microvm in microvm_factory.build_n_from_snapshot(
        snapshot, ITERATIONS, no_netns_reuse=True
    ):
        value = 0
        # Parse all metric data points in search of load_snapshot time.
        microvm.flush_metrics()
        for data_point in microvm.get_all_metrics():
            cur_value = data_point["latencies_us"]["load_snapshot"]
            if cur_value > 0:
                value = cur_value / USEC_IN_MSEC
                break
        assert value > 0
        metrics.put_metric("latency", value, "Milliseconds")


@pytest.mark.parametrize(
    ("test_setup", "dirty_mib"),
    [
        pytest.param(
            SnapshotRestoreTest(mem=mem),
            dirty_mib,
            id=f"1vcpu_{mem}mb_{dirty_mib}mb_dirty",
        )
        for mem, dirty_sizes in [
            (128, [0, 32]),
            (1024, [0, 64, 512]),
            (4096, [0, 64, 512]),
        ]
        for dirty_mib in dirty_sizes
    ],
)
def test_reset_latency(
    microvm_factory, guest_kernel, rootfs, pci_enabled, test_setup, dirty_mib, metrics
):
    """Measure in-place reset latency after dirtying guest memory."""
    vm = test_setup.boot_vm(
        microvm_factory, guest_kernel, rootfs, pci_enabled, track_dirty_pages=True
    )
    metrics.set_dimensions(
        {
            "net_devices": str(test_setup.nets),
            "block_devices": str(test_setup.blocks),
            "vsock_devices": "0",
            "balloon_devices": "0",
            "huge_pages_config": str(test_setup.huge_pages),
            "performance_test": "test_reset_latency",
            "uffd_handler": "None",
            "dirty_mib": str(dirty_mib),
            **vm.dimensions,
        }
    )

    # The default tmpfs limit can be smaller than half the configured guest RAM.
    vm.ssh.check_output(f"mount -o remount,size={max(dirty_mib, 1)}M /dev/shm")
    snapshot = vm.snapshot_diff()
    vm.kill()

    vm = microvm_factory.build(
        guest_kernel, rootfs, monitor_memory=False, pci=pci_enabled
    )
    vm.spawn(log_level="Info", emit_metrics=True)
    vm.restore_from_snapshot(snapshot)
    vm.flush_metrics()
    load_latency = max(
        point["latencies_us"]["load_snapshot"] for point in vm.get_all_metrics()
    )
    assert load_latency > 0
    metrics.put_metric("load_latency", load_latency / USEC_IN_MSEC, "Milliseconds")

    ssh_args = {
        "netns": vm.netns.id,
        "ssh_key": vm.ssh_key,
        "control_path": Path(vm.chroot()) / "reset-ssh.sock",
        "host": snapshot.net_ifaces[0].guest_ip,
        "user": "root",
    }
    reset_latencies = []
    vmm_reset_latencies = []
    for _ in range(RESET_ITERATIONS):
        # Reset restores the old tap's MAC in the guest ARP cache.
        vm.netns.check_output(f"ip neigh flush dev {snapshot.net_ifaces[0].tap_name}")
        vm.resume()
        # Connections opened after the snapshot cannot survive a reset.
        with closing(SSHConnection(**ssh_args)) as ssh:
            if dirty_mib:
                ssh.check_output(
                    f"dd if=/dev/zero of=/dev/shm/reset-dirty bs=1M count={dirty_mib} status=none"
                )
            else:
                ssh.check_output("true")

        vm.pause()
        vm.api.snapshot_reset.put()
        latencies = vm.flush_metrics()["latencies_us"]
        reset_latencies.append(latencies["reset_snapshot"])
        vmm_reset_latencies.append(latencies["vmm_reset_snapshot"])
        metrics.put_metric(
            "reset_latency", reset_latencies[-1] / USEC_IN_MSEC, "Milliseconds"
        )
        metrics.put_metric(
            "vmm_reset_latency", vmm_reset_latencies[-1] / USEC_IN_MSEC, "Milliseconds"
        )

    assert any(value > 0 for value in reset_latencies)
    assert any(value > 0 for value in vmm_reset_latencies)
    vm.netns.check_output(f"ip neigh flush dev {snapshot.net_ifaces[0].tap_name}")
    vm.resume()
    with closing(SSHConnection(**ssh_args)) as ssh:
        ssh.check_output("test ! -e /dev/shm/reset-dirty")


# When using the fault-all handler, all guest memory will be faulted in way before the helper tool
# wakes up, because it gets faulted in on the first page fault. In this scenario, we are not measuring UFFD
# latencies, but KVM latencies of setting up missing EPT entries.
@pytest.mark.nonci
@pytest.mark.parametrize("uffd_handler", [None, "on_demand", "fault_all"])
@pytest.mark.parametrize("huge_pages", HugePagesConfig)
def test_post_restore_latency(
    microvm_factory,
    rootfs,
    guest_kernel,
    pci_enabled,
    metrics,
    uffd_handler,
    huge_pages,
):
    """Collects latency metric of post-restore memory accesses done inside the guest"""
    if huge_pages != HugePagesConfig.NONE and uffd_handler is None:
        pytest.skip("huge page snapshots can only be restored using uffd")

    test_setup = SnapshotRestoreTest(mem=1024, vcpus=2, huge_pages=huge_pages)
    vm = test_setup.boot_vm(microvm_factory, guest_kernel, rootfs, pci_enabled)

    metrics.set_dimensions(
        {
            "net_devices": str(test_setup.nets),
            "block_devices": str(test_setup.blocks),
            "vsock_devices": str(int(test_setup.all_devices)),
            "balloon_devices": str(int(test_setup.all_devices)),
            "huge_pages_config": str(test_setup.huge_pages),
            "performance_test": "test_post_restore_latency",
            "uffd_handler": str(uffd_handler),
            **vm.dimensions,
        }
    )

    # Starts the helper and blocks until it has touched its memory and is
    # waiting in sigwait, so the snapshot below captures it in that state.
    start_fast_page_fault_helper(vm.ssh)

    snapshot = vm.snapshot_full()
    vm.kill()

    for microvm in microvm_factory.build_n_from_snapshot(
        snapshot, ITERATIONS, uffd_handler_name=uffd_handler
    ):
        _, pid, _ = microvm.ssh.check_output("pidof fast_page_fault_helper")

        microvm.ssh.check_output(f"kill -s {signal.SIGUSR1} {pid}")

        _, duration, _ = microvm.ssh.check_output(
            "while [ ! -f /tmp/fast_page_fault_helper.out ]; do sleep 1; done; cat /tmp/fast_page_fault_helper.out"
        )

        metrics.put_metric("fault_latency", int(duration) / NS_IN_MSEC, "Milliseconds")


@pytest.mark.nonci
@pytest.mark.parametrize("huge_pages", HugePagesConfig)
@pytest.mark.parametrize(
    ("vcpus", "mem"), [(1, 128), (1, 1024), (2, 2048), (3, 4096), (4, 6144)]
)
def test_population_latency(
    microvm_factory,
    rootfs,
    guest_kernel,
    pci_enabled,
    metrics,
    huge_pages,
    vcpus,
    mem,
):
    """Collects population latency metrics (e.g. how long it takes UFFD handler to fault in all memory)"""
    test_setup = SnapshotRestoreTest(mem=mem, vcpus=vcpus, huge_pages=huge_pages)
    vm = test_setup.boot_vm(microvm_factory, guest_kernel, rootfs, pci_enabled)

    metrics.set_dimensions(
        {
            "net_devices": str(test_setup.nets),
            "block_devices": str(test_setup.blocks),
            "vsock_devices": str(int(test_setup.all_devices)),
            "balloon_devices": str(int(test_setup.all_devices)),
            "huge_pages_config": str(test_setup.huge_pages),
            "performance_test": "test_population_latency",
            "uffd_handler": "fault_all",
            **vm.dimensions,
        }
    )

    snapshot = vm.snapshot_full()
    vm.kill()

    for microvm in microvm_factory.build_n_from_snapshot(
        snapshot, ITERATIONS, uffd_handler_name="fault_all"
    ):
        # API response times are unreliable while the uffd handler is
        # faulting in all pages — skip the timing validation.
        microvm.time_api_requests = False
        # do _something_ to trigger a pagefault, which will then cause the UFFD handler to fault in _everything_
        microvm.ssh.check_output("true")

        for _ in range(5):
            time.sleep(1)

            match = re.match(
                r"Finished Faulting All: (\d+)us", microvm.uffd_handler.log_data
            )

            if match:
                latency_us = int(match.group(1))

                metrics.put_metric(
                    "populate_latency", latency_us / 1000, "Milliseconds"
                )
                break
        else:
            raise RuntimeError("UFFD handler did not print population latency after 5s")


@pytest.mark.nonci
def test_snapshot_create_latency(
    uvm,
    metrics,
    snapshot_type,
):
    """Measure the latency of creating a Full snapshot"""

    vm = uvm
    vm.spawn()
    vm.basic_config(
        vcpu_count=2,
        mem_size_mib=512,
        track_dirty_pages=snapshot_type.needs_dirty_page_tracking,
    )
    vm.start()
    vm.pin_threads(0)

    metrics.set_dimensions(
        {
            **vm.dimensions,
            "performance_test": "test_snapshot_create_latency",
            "snapshot_type": str(snapshot_type),
        }
    )

    match snapshot_type:
        case SnapshotType.FULL:
            metric = "full_create_snapshot"
        case SnapshotType.DIFF | SnapshotType.DIFF_MINCORE:
            metric = "diff_create_snapshot"

    for _ in range(ITERATIONS):
        vm.make_snapshot(snapshot_type)
        fc_metrics = vm.flush_metrics()

        value = fc_metrics["latencies_us"][metric] / USEC_IN_MSEC
        metrics.put_metric("latency", value, "Milliseconds")
