#!/usr/bin/env python3
# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Throwaway: run the traditional test_size_reduction variants many times.

Measures the failure rate on the x86 instance types and host kernels where CI
saw it fail, for guests that boot the AL2023 rootfs (6.1 and 6.18).
"""

import sys

from common import BKPipeline

sys.argv += [
    "--instances",
    "m6i.metal",
    "m6a.metal",
    "m7a.metal-48xl",
    "m7i.metal-48xl",
    "m8i.metal-48xl",
    "--platforms",
    "al2-linux_5.10",
    "al2023-linux_6.1",
    "al2023-linux_6.18",
    "--no-kani",
]

pipeline = BKPipeline(priority=1, timeout_in_minutes=90)
# Same settings as the PR pipeline's "performance" group, which is where CI
# runs test_size_reduction: one agent per host, pinned CPUs.
pipeline.build_group(
    "balloon-size-reduction",
    pipeline.devtool_test(
        devtool_opts="--performance -c 1-10 -m 0",
        pytest_opts="--count=10 -k 'test_size_reduction and traditional and not 5.10' "
        "../tests/integration_tests/performance/test_balloon.py",
    ),
    priority=2,
    agents={"ag": 1},
)
print(pipeline.to_json())
