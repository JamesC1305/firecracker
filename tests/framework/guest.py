# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guest distro detection and distro-specific properties."""

from pathlib import Path


class GuestDistro:
    """Distro-specific guest properties, inferred from rootfs filename."""

    def __init__(self, rootfs_path: Path):
        name = rootfs_path.stem.lower()
        if "ubuntu" in name:
            self.hostname = "ubuntu-fc-uvm"
            self.ssh_service = "ssh.service"
            self.os_release_token = "ID=ubuntu"
            self.shell_prompt = f"{self.hostname}:~#"
        elif "amazon" in name or "al2023" in name:
            self.hostname = "al2023-fc-uvm"
            self.ssh_service = "sshd.service"
            self.os_release_token = 'ID="amzn"'
            self.shell_prompt = f"[root@{self.hostname} ~]#"
        else:
            raise ValueError(f"Unknown guest distro for rootfs: {rootfs_path}")
