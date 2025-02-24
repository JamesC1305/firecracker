# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests ensuring desired style for commit messages."""

import os

import pytest

from framework import utils
from framework.ab_test import DEFAULT_A_REVISION


@pytest.mark.skipif(
    os.environ.get("BUILDKITE_PULL_REQUEST") == "5048",
    reason="PR of a feature branch from before this test was modified to follow Linux sign-off rules for co-authors.",
)
def test_gitlint():
    """
    Test that all commit messages pass the gitlint rules.
    """
    os.environ["LC_ALL"] = "C.UTF-8"
    os.environ["LANG"] = "C.UTF-8"

    rc, _, stderr = utils.run_cmd(
        f"gitlint --commits origin/{DEFAULT_A_REVISION}..HEAD -C ../.gitlint --extra-path framework/gitlint_rules.py",
    )
    assert rc == 0, "Commit message violates gitlint rules: {}".format(stderr)
