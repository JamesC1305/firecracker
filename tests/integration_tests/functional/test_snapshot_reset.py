# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for resetting a microVM to its load-time snapshot in place."""

from contextlib import closing
from http import HTTPStatus
from pathlib import Path

import pytest

from framework.artifacts import GUEST_KERNEL_DEFAULT, pin_guest_kernel, pin_rootfs_mode
from framework.utils import check_network_data_integrity
from framework.utils_vsock import (
    ECHO_SERVER_PORT,
    VSOCK_UDS_PATH,
    start_guest_echo_server,
    vsock_connect_to_guest,
)
from host_tools.network import SSHConnection

pytestmark = [
    pin_guest_kernel(GUEST_KERNEL_DEFAULT),
    pin_rootfs_mode("ro"),
]


def test_snapshot_reset(uvm, microvm_factory, io_engine):
    """Three resets restore tmpfs state without breaking net, block or vsock."""
    uvm.spawn()
    uvm.basic_config(track_dirty_pages=True, rootfs_io_engine=io_engine)
    uvm.add_net_iface()
    uvm.api.vsock.put(vsock_id="vsock0", guest_cid=3, uds_path=VSOCK_UDS_PATH)
    uvm.start()
    uvm.ssh.check_output("echo baseline > /dev/shm/reset-baseline")
    start_guest_echo_server(uvm)

    # The first diff snapshot contains all writes since boot. Loading it enables
    # dirty tracking, which reset needs to discard subsequent guest writes.
    snapshot = uvm.snapshot_diff()
    uvm.kill()
    vm = microvm_factory.build()
    vm.spawn()
    vm.restore_from_snapshot(snapshot)
    vm.resume()

    # Check the initial load, then the same baseline after three resets.
    for cycle in range(4):
        if cycle:
            vm.pause()
            vm.api.snapshot_reset.put()
            assert vm.state == "Paused"
            # Reset returns the guest ARP cache to the snapshot, which holds the MAC
            # of the tap that the snapshot was taken with. Make the host ask for the
            # guest MAC again, so that the guest learns the MAC of the current tap.
            vm.netns.check_output(
                f"ip neigh flush dev {snapshot.net_ifaces[0].tap_name}"
            )
            vm.resume()

        # A reset invalidates connections opened after the snapshot. Do not use
        # the cached Microvm.ssh connection across resets.
        with closing(
            SSHConnection(
                netns=vm.netns.id,
                ssh_key=vm.ssh_key,
                control_path=Path(vm.chroot()) / "reset-ssh.sock",
                host=snapshot.net_ifaces[0].guest_ip,
                user="root",
            )
        ) as ssh:
            _, stdout, _ = ssh.check_output("cat /dev/shm/reset-baseline")
            assert stdout.strip() == "baseline"
            ssh.check_output("test ! -e /dev/shm/reset-created")
            check_network_data_integrity(ssh, size_bytes=4096)
            # Bypass the guest page cache so a broken block queue cannot hide
            # behind data already read before the snapshot.
            ssh.check_output("dd if=/dev/vda of=/dev/null bs=4096 count=1 iflag=direct")

            uds_path = str(Path(vm.chroot()) / VSOCK_UDS_PATH)
            with vsock_connect_to_guest(uds_path, ECHO_SERVER_PORT) as sock:
                sock.settimeout(5)
                payload = f"reset cycle {cycle}\n".encode()
                sock.sendall(payload)
                with sock.makefile("rb") as received:
                    assert received.read(len(payload)) == payload

            ssh.check_output(
                f"echo cycle-{cycle} > /dev/shm/reset-baseline && "
                "touch /dev/shm/reset-created"
            )
            _, stdout, _ = ssh.check_output("cat /dev/shm/reset-baseline")
            assert stdout.strip() == f"cycle-{cycle}"


@pytest.mark.parametrize("case", ["running", "booted", "dirty-tracking-disabled"])
def test_snapshot_reset_rejected(uvm, microvm_factory, case):
    """Rejected resets leave the VM's execution state and guest memory intact."""
    uvm.spawn()
    uvm.basic_config(track_dirty_pages=case != "dirty-tracking-disabled")
    uvm.add_net_iface()
    uvm.start()
    vm = uvm
    if case != "booted":
        snapshot = uvm.snapshot_diff() if case == "running" else uvm.snapshot_full()
        uvm.kill()
        vm = microvm_factory.build_from_snapshot(snapshot)

    vm.ssh.check_output("echo unchanged > /dev/shm/reset-rejected")
    if case != "running":
        vm.pause()
    state = vm.state

    with pytest.raises(RuntimeError) as exc:
        vm.api.snapshot_reset.put()
    assert exc.value.args[2].status_code == HTTPStatus.BAD_REQUEST
    assert vm.state == state

    if case != "running":
        vm.resume()
    _, stdout, _ = vm.ssh.check_output("cat /dev/shm/reset-rejected")
    assert stdout.strip() == "unchanged"
    check_network_data_integrity(vm.ssh, size_bytes=4096)
