#!/usr/bin/env python3
# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scratch: run only the cross-kernel snapshot round trip. Not for merging.

The personal PR pipeline always uploads this script's output, so it emits the
cross-kernel pipeline instead, keeping the build step, the linux_6.1 snapshot
create steps the round trip depends on, and the round-trip group. It also runs
the unit tests and the vCPU register pinning functional test on two aarch64
instances and both host kernels, because the aarch64 code cannot run on an
x86_64 development host.
"""

import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path

SOURCE_KV = "linux_6.1"
CREATE_GROUPS = ("snapshot-create", "snapshot-create-aarch64")
DROPPED_GROUPS = ("snapshot-restore-across-instances-and-kernels",)
UNITTEST_INSTANCES = ("m7g.metal", "m9g.metal-48xl")
UNITTEST_KVS = ("linux_6.1", "linux_6.18")
UNITTESTS = "./tools/devtool -y test --no-build -- integration_tests/build/test_unittests.py"
VCPU_REGISTERS_TEST = (
    "./tools/devtool -y test --no-build --no-archive -- "
    "integration_tests/functional/test_snapshot_vcpu_registers.py"
)


def unittest_group(create_steps):
    """Test steps that reuse the agents of the aarch64 create steps."""
    steps = []
    for create in create_steps:
        instance = create["agents"]["instance"]
        if instance not in UNITTEST_INSTANCES:
            continue
        # The commands that fetch the build artifacts come before the devtool one.
        fetch_build = list(
            itertools.takewhile(
                lambda command: not command.startswith("./tools/devtool"),
                create["command"],
            )
        )
        for kv in UNITTEST_KVS:
            # The unit tests build what they need; neither test uploads snapshots.
            unittests = copy.deepcopy(create)
            del unittests["depends_on"]
            unittests["command"] = [UNITTESTS]
            unittests["label"] = f"unittests-{instance}-{kv}"

            functional = copy.deepcopy(create)
            functional["command"] = [*fetch_build, VCPU_REGISTERS_TEST]
            functional["label"] = f"vcpu-registers-{instance}-{kv}"

            for step in (unittests, functional):
                del step["key"]
                step["agents"]["kv"] = kv
                step["artifact_paths"] = ["./test_results/**/*"]
                steps.append(step)
    return {"group": "unittests-aarch64", "steps": steps}


def roundtrip_only(cross):
    """Filter the cross-kernel pipeline down to the round trip."""
    kept = []
    for step in cross["steps"]:
        group = step.get("group")
        if group in DROPPED_GROUPS:
            continue
        if group in CREATE_GROUPS:
            step["steps"] = [s for s in step["steps"] if s["agents"]["kv"] == SOURCE_KV]
        kept.append(step)
        if group == "snapshot-create-aarch64":
            kept.append(unittest_group(step["steps"]))
    cross["steps"] = kept
    return cross


def main():
    """Print the filtered pipeline."""
    script = Path(__file__).parent / "pipeline_cross.py"
    cross = json.loads(subprocess.check_output([sys.executable, str(script)]))
    print(json.dumps(roundtrip_only(cross), indent=4))


if __name__ == "__main__":
    main()
