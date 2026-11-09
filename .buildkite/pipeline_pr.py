#!/usr/bin/env python3
# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scratch: run only the cross-kernel snapshot round trip. Not for merging.

The personal PR pipeline always uploads this script's output, so it emits the
cross-kernel pipeline instead, keeping the build step, the linux_6.1 snapshot
create steps the round trip depends on, and the round-trip group.
"""

import json
import subprocess
import sys
from pathlib import Path

SOURCE_KV = "linux_6.1"
CREATE_GROUPS = ("snapshot-create", "snapshot-create-aarch64")
DROPPED_GROUPS = ("snapshot-restore-across-instances-and-kernels",)


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
    cross["steps"] = kept
    return cross


def main():
    """Print the filtered pipeline."""
    script = Path(__file__).parent / "pipeline_cross.py"
    cross = json.loads(subprocess.check_output([sys.executable, str(script)]))
    print(json.dumps(roundtrip_only(cross), indent=4))


if __name__ == "__main__":
    main()
