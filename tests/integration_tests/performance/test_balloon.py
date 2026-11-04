# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for guest-side operations on /balloon resources."""

import concurrent
import signal
import time
from pathlib import Path

import pytest

from framework.artifacts import GUEST_KERNEL_DEFAULT, pin_guest_kernel, pin_pci
from framework.utils import (
    get_stable_rss_mem,
    get_threads,
    start_fast_page_fault_helper,
    track_cpu_utilization,
    wait_for_balloon_actual,
    wait_for_fast_page_fault_helper_output,
)
from framework.utils_hugepages import HugePagesConfig, supports_hugetlbfs_discard

# Every test in this module exercises all huge_pages variants.
pytestmark = pytest.mark.parametrize(
    "huge_pages",
    HugePagesConfig,
    indirect=True,
)

NS_IN_MSEC = 1_000_000
MIB = 1024 * 1024

# The sustained-churn test runs the helper back to back in chunks, so the
# per-chunk VMM CPU time and reported memory give one sample per chunk. A
# chunk spans about one 2 s reporting delay.
CHURN_CHUNKS = 10
CHURN_RUNS_PER_CHUNK = 40


def trigger_page_fault_run(vm):
    """
    Clears old data and starts the fast_page_fault_helper script
    """
    vm.ssh.check_output(
        "rm -f /tmp/fast_page_fault_helper.out && /usr/local/bin/fast_page_fault_helper -s"
    )


def get_page_fault_duration(vm):
    """
    Waits for the performance data to be available and will read the duration
    """
    return wait_for_fast_page_fault_helper_output(vm.ssh)


@pin_guest_kernel(GUEST_KERNEL_DEFAULT)
@pytest.mark.parametrize("method", ["reporting", "hinting"])
@pytest.mark.nonci
def test_hinting_reporting_cpu(
    microvm_factory,
    guest_kernel,
    rootfs,
    method,
    metrics,
    huge_pages,
):
    """
    Measure the CPU usage when running free page reporting and hinting
    """
    test_microvm = microvm_factory.build(
        guest_kernel,
        rootfs,
        pci=True,
        monitor_memory=False,
    )
    test_microvm.spawn(emit_metrics=False)
    test_microvm.basic_config(vcpu_count=2, mem_size_mib=1024, huge_pages=huge_pages)
    test_microvm.add_net_iface()

    free_page_reporting = method == "reporting"
    free_page_hinting = method == "hinting"
    # Add a deflated memory balloon.
    test_microvm.api.balloon.put(
        amount_mib=0,
        deflate_on_oom=False,
        stats_polling_interval_s=0,
        free_page_reporting=free_page_reporting,
        free_page_hinting=free_page_hinting,
    )
    test_microvm.start()
    test_microvm.pin_threads(0)

    metrics.set_dimensions(
        {
            "performance_test": "test_balloon_cpu",
            # "huge_pages": str(huge_pages),
            "method": method,
            "huge_pages": str(huge_pages),
            **test_microvm.dimensions,
        }
    )

    # Blocks until the helper has touched its memory and is waiting in sigwait.
    pid = start_fast_page_fault_helper(test_microvm.ssh)
    test_microvm.ssh.check_output(f"kill -s {signal.SIGUSR1} {pid}")

    cpu_util = None

    if free_page_reporting:
        cpu_util = track_cpu_utilization(test_microvm.firecracker_pid, 5, 0)
    else:
        wait_for_fast_page_fault_helper_output(test_microvm.ssh)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            cpu_load_future = executor.submit(
                track_cpu_utilization,
                test_microvm.firecracker_pid,
                2,
                omit=0,
            )
            test_microvm.api.balloon_hinting_start.patch()
            cpu_util = cpu_load_future.result()

    for thread_name, values in cpu_util.items():
        for value in values:
            metrics.put_metric(f"cpu_utilization_{thread_name}", value, "Percent")


@pin_guest_kernel(GUEST_KERNEL_DEFAULT)
@pytest.mark.parametrize("sleep_duration", [0, 1, 5])
@pytest.mark.nonci
def test_hinting_fault_latency(
    microvm_factory,
    guest_kernel,
    rootfs,
    metrics,
    sleep_duration,
    huge_pages,
):
    """
    Measure the overhead of running free page reporting with allocation heavy
    workloads.

    Test with different sleep intervals to measure the effect
    depending on frequenecy
    """
    runs = 5
    test_microvm = microvm_factory.build(
        guest_kernel,
        rootfs,
        pci=True,
        monitor_memory=False,
    )
    test_microvm.spawn(emit_metrics=False)
    test_microvm.basic_config(vcpu_count=2, mem_size_mib=1024, huge_pages=huge_pages)
    test_microvm.add_net_iface()

    # Add a deflated memory balloon.
    test_microvm.api.balloon.put(
        amount_mib=0,
        deflate_on_oom=False,
        stats_polling_interval_s=0,
        free_page_reporting=True,
    )
    test_microvm.start()
    test_microvm.pin_threads(0)

    metrics.set_dimensions(
        {
            "performance_test": "test_hinting_fault_latency",
            "sleep_duration": str(sleep_duration),
            "huge_pages": str(huge_pages),
            **test_microvm.dimensions,
        }
    )

    for i in range(runs):
        trigger_page_fault_run(test_microvm)
        reporting_duration = int(get_page_fault_duration(test_microvm)) / NS_IN_MSEC
        metrics.put_metric("latency", reporting_duration, "Milliseconds")

        if sleep_duration > 0 and (i + 1 < runs):
            time.sleep(sleep_duration)


def vmm_thread_cpu_ns(vm):
    """Return the CPU time, in nanoseconds, used so far by the VMM thread"""
    return sum(
        int(Path(f"/proc/{tid}/schedstat").read_text("ascii").split()[0])
        for name, tids in get_threads(vm.firecracker_pid).items()
        if name.startswith("firecracker")
        for tid in tids
    )


def wait_for_reporting_idle(vm, idle_s=3, timeout_s=60):
    """Wait until no free page report has arrived for `idle_s` seconds"""
    idle = 0
    for _ in range(timeout_s):
        time.sleep(1)
        reports = vm.flush_metrics()["balloon"]["free_page_report_count"]
        idle = idle + 1 if reports == 0 else 0
        if idle >= idle_s:
            return
    raise TimeoutError("free page reporting did not go idle")


@pin_guest_kernel(GUEST_KERNEL_DEFAULT)
@pytest.mark.parametrize("uffd_handler", [None, "on_demand"])
@pytest.mark.nonci
def test_reporting_churn(
    microvm_factory,
    guest_kernel,
    rootfs,
    metrics,
    uffd_handler,
    huge_pages,
):
    """
    Measure fault latency and VMM CPU time while free page reporting runs
    alongside an allocation-heavy workload.

    The guest runs the fault helper back to back, so it is always allocating
    or freeing memory when a reporting pass starts. With a UFFD handler, the
    VM is snapshotted and restored first.
    """
    vm = microvm_factory.build(guest_kernel, rootfs, pci=True, monitor_memory=False)
    vm.spawn(emit_metrics=False)
    vm.basic_config(vcpu_count=2, mem_size_mib=1024, huge_pages=huge_pages)
    vm.add_net_iface()
    vm.api.balloon.put(
        amount_mib=0,
        deflate_on_oom=False,
        stats_polling_interval_s=0,
        free_page_reporting=True,
    )
    vm.start()
    dimensions = vm.dimensions

    if uffd_handler:
        snapshot = vm.snapshot_full()
        vm.kill()
        vm = microvm_factory.build(monitor_memory=False)
        vm.spawn(emit_metrics=False)
        vm.restore_from_snapshot(snapshot, resume=True, uffd_handler_name=uffd_handler)
    vm.pin_threads(0)

    metrics.set_dimensions(
        {
            "performance_test": "test_reporting_churn",
            "huge_pages": str(huge_pages),
            "uffd_handler": str(uffd_handler),
            **dimensions,
        }
    )

    # Start from a quiet state, after the boot-time frees have been reported.
    wait_for_reporting_idle(vm)
    vm.ssh.check_output("rm -f /tmp/churn.lat")
    for _ in range(CHURN_CHUNKS):
        first_metric = len(vm.get_all_metrics())
        cpu_start = vmm_thread_cpu_ns(vm)
        vm.ssh.check_output(
            f"for i in $(seq {CHURN_RUNS_PER_CHUNK}); do "
            "/usr/local/bin/fast_page_fault_helper -s || exit 1; "
            "{ cat /tmp/fast_page_fault_helper.out; echo; } >> /tmp/churn.lat; "
            "done"
        )
        cpu_end = vmm_thread_cpu_ns(vm)
        vm.flush_metrics()
        reported = sum(
            point["balloon"]["free_page_report_freed"]
            for point in vm.get_all_metrics()[first_metric:]
        )
        metrics.put_metric(
            "vmm_thread_cpu_time", (cpu_end - cpu_start) / NS_IN_MSEC, "Milliseconds"
        )
        metrics.put_metric("reported", reported / MIB, "Megabytes")

    _, latencies, _ = vm.ssh.check_output("cat /tmp/churn.lat")
    for latency in latencies.split():
        metrics.put_metric("latency", int(latency) / NS_IN_MSEC, "Milliseconds")


# pylint: disable=C0103
@pytest.mark.parametrize("backing", ["anonymous", "memfd"])
@pytest.mark.parametrize("method", ["traditional", "hinting", "reporting"])
# The transport does not affect the reduction.
@pin_pci(True)
def test_size_reduction(uvm, rootfs, method, backing, huge_pages):
    """
    Verify that ballooning reduces RSS usage on a newly booted guest.
    """
    traditional_balloon = method == "traditional"
    free_page_reporting = method == "reporting"
    free_page_hinting = method == "hinting"
    memfd_backing = backing == "memfd"

    if huge_pages != HugePagesConfig.NONE:
        if not memfd_backing and not supports_hugetlbfs_discard():
            pytest.skip("Host does not support hugetlb discard")

        if traditional_balloon:
            pytest.skip("Traditional balloon device won't reduce RSS")

    test_microvm = uvm
    test_microvm.spawn()
    test_microvm.basic_config(huge_pages=huge_pages, add_root_device=not memfd_backing)
    if memfd_backing:
        # Guest memory is only backed by a memfd when a vhost-user device is attached.
        test_microvm.add_vhost_user_drive(
            "rootfs", rootfs, is_root_device=True, is_read_only=True
        )
    test_microvm.add_net_iface()

    # Add a memory balloon.
    test_microvm.api.balloon.put(
        amount_mib=0,
        deflate_on_oom=True,
        # wait_for_balloon_actual reads the balloon size from its statistics.
        stats_polling_interval_s=1 if traditional_balloon else 0,
        free_page_reporting=free_page_reporting,
        free_page_hinting=free_page_hinting,
    )

    # Start the microvm.
    test_microvm.start()

    get_stable_rss_mem(test_microvm)

    # Blocks until the helper has touched its 128 MiB and is waiting in sigwait.
    pid = start_fast_page_fault_helper(test_microvm.ssh)

    first_reading = get_stable_rss_mem(test_microvm)

    # Kill the application which will free the held memory
    test_microvm.ssh.check_output(f"kill -s {signal.SIGUSR1} {pid}")

    # Sleep to allow guest to clean up
    time.sleep(1)
    # Have the guest drop its caches.
    test_microvm.ssh.run("sync; echo 3 > /proc/sys/vm/drop_caches")
    time.sleep(2)

    # The balloon may first take guest memory that was never touched, which
    # frees nothing on the host. Inflate past all of it plus 20 MiB, twice the
    # reduction required below, leaving room for Firecracker's own memory in the
    # reading and for memory the guest touches in the meantime.
    inflate_size = 256 - int(first_reading / 1024) + 20

    if traditional_balloon:
        # Now inflate the balloon
        test_microvm.api.balloon.patch(amount_mib=inflate_size)
        wait_for_balloon_actual(test_microvm, inflate_size)
    elif free_page_hinting:
        test_microvm.api.balloon_hinting_start.patch()

    _ = get_stable_rss_mem(test_microvm)

    if traditional_balloon:
        # Deflate the balloon completely.
        test_microvm.api.balloon.patch(amount_mib=0)

    # Check memory usage again.
    second_reading = get_stable_rss_mem(test_microvm)

    # There should be a reduction of at least 10 MiB.
    reduction = first_reading - second_reading
    assert reduction >= 10 * 1024, (
        f"resident memory fell by {reduction} KiB "
        f"({first_reading} -> {second_reading}), expected at least 10 MiB"
    )
