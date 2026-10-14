#!/usr/bin/env python3
# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Throwaway: verify test_size_reduction with the pagemap-based guest reader.

Group 1 repeats every variant on x86 and Graviton hosts. Group 2 repeats the
variants that failed with the smaps-based reader (hinting and reporting,
anonymous memory, transparent huge pages, 6.1 guests) on the AMD hosts running
the AL2023 6.1 kernel, and shows each reading from pagemap next to smaps.
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
    "m7g.metal",
    "m8g.metal-24xl",
    "--platforms",
    "al2-linux_5.10",
    "al2023-linux_6.1",
    "al2023-linux_6.18",
    "--no-kani",
]

pipeline = BKPipeline(priority=1, timeout_in_minutes=120)
test_file = "../tests/integration_tests/performance/test_balloon.py"
pipeline.build_group(
    "balloon-size-reduction",
    pipeline.devtool_test(
        devtool_opts="--performance -c 1-10 -m 0",
        pytest_opts=f"--count=3 -k 'test_size_reduction and not 5.10' {test_file}",
    ),
    priority=2,
    agents={"ag": 1},
)
pipeline.build_group(
    "balloon-thp-amd",
    pipeline.devtool_test(
        devtool_opts="--performance -c 1-10 -m 0",
        pytest_opts="--count=20 -rA -k 'test_size_reduction and 6.1.188 and anonymous "
        f"and Transparent and not traditional' {test_file}",
    ),
    priority=2,
    agents={"ag": 1},
    instances=["m6a.metal", "m7a.metal-48xl"],
    platforms=[("al2023", "linux_6.1")],
)
print(pipeline.to_json())
