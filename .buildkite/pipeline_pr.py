#!/usr/bin/env python3
# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scratch pipeline: run only the reset lifecycle comparison on AL2023 6.18 hosts."""

import sys

sys.argv[1:] = ["--platforms", "al2023-linux_6.18", "--no-kani"]

# pylint: disable=wrong-import-position
from common import BKPipeline

pipeline = BKPipeline(priority=1, timeout_in_minutes=90, with_build_step=True)
pipeline.build_group(
    "performance",
    pipeline.devtool_test(
        devtool_opts="--performance -c 1-10 -m 0",
        pytest_opts="../tests/integration_tests/performance/test_reset_lifecycle.py",
    ),
    priority=2,
    agents={"ag": 1},
)
print(pipeline.to_json())
