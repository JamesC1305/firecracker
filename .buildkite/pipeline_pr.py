#!/usr/bin/env python3
# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scratch: run only the cross-kernel snapshot round trip. Not for merging.

The personal PR pipeline always uploads this script's output, so it emits the
cross-kernel pipeline instead, keeping the build step, the round-trip group and
the snapshot create steps the round trip depends on.
"""

import json
import subprocess
import sys
from pathlib import Path

ROUNDTRIP_GROUP = "snapshot-roundtrip-across-kernels"
CREATE_GROUPS = ("snapshot-create", "snapshot-create-aarch64")
DROPPED_GROUPS = ("snapshot-restore-across-instances-and-kernels",)


def roundtrip_only(cross):
    """Filter the cross-kernel pipeline down to the round trip."""
    roundtrip = next(s for s in cross["steps"] if s.get("group") == ROUNDTRIP_GROUP)
    needed = {key for step in roundtrip["steps"] for key in step["depends_on"]}
    kept = []
    for step in cross["steps"]:
        group = step.get("group")
        if group in DROPPED_GROUPS:
            continue
        if group in CREATE_GROUPS:
            step["steps"] = [s for s in step["steps"] if s["key"] in needed]
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
