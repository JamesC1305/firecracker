#!/usr/bin/env python3
# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Throwaway: A/B the free page THP failures against the baseline test.

Runs the new test_size_reduction and the baseline one (test_balloon_baseline.py)
on the same host, for the hinting and reporting variants with anonymous memory
and transparent huge pages on 6.1 guests, which failed on AMD hosts running the
AL2023 6.1 host kernel. Both print diagnostics after their final reading.
"""

import sys

from common import BKPipeline

sys.argv += [
    "--instances",
    "m6a.metal",
    "m7a.metal-48xl",
    "--platforms",
    "al2023-linux_6.1",
    "--no-kani",
]

pipeline = BKPipeline(priority=1, timeout_in_minutes=120)
pipeline.build_group(
    "balloon-thp-ab",
    pipeline.devtool_test(
        devtool_opts="--performance -c 1-10 -m 0",
        pytest_opts="--count=20 -rA "
        "-k 'test_size_reduction and 6.1.188 and anonymous and Transparent "
        "and not traditional' "
        "../tests/integration_tests/performance/test_balloon.py "
        "../tests/integration_tests/performance/test_balloon_baseline.py",
    ),
    priority=2,
    agents={"ag": 1},
)
print(pipeline.to_json())
