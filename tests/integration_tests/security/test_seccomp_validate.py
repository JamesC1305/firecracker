# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test that validates that seccompiler filters work as expected"""

import json
import platform
import resource
import signal
from pathlib import Path

import pytest
import seccomp

from framework import utils
from host_tools.cargo_build import expand_seccomp_filters

ARCH = platform.machine()

FC_FILTER_PATH = Path(f"../resources/seccomp/{ARCH}-unknown-linux-musl.json")
FILTER_THREADS = list(
    expand_seccomp_filters(json.loads(FC_FILTER_PATH.read_text(encoding="ascii")))
)

SECCOMP_ARCH = {
    "x86_64": seccomp.Arch.X86_64,
    "aarch64": seccomp.Arch.AARCH64,
}[ARCH]


@pytest.fixture
def bin_test_syscall(tmp_path):
    """Build the test_syscall binary."""
    test_syscall_bin = tmp_path / "test_syscall"
    compile_cmd = f"musl-gcc -static host_tools/test_syscalls.c -o {test_syscall_bin}"
    utils.check_output(compile_cmd)
    assert test_syscall_bin.exists()
    yield test_syscall_bin.resolve()


def _filter(*syscalls):
    """Build a small allowlist filter for compiler regression coverage."""
    return {
        "default_action": "trap",
        "filter_action": "allow",
        "filter": [{"syscall": syscall} for syscall in syscalls],
    }


def test_structured_policy_matches_flat_policy(seccompiler, bin_test_syscall, tmp_path):
    """A rule group included by two filters compiles exactly like two flat filters.

    The filter names are the same in both policies, so the compiled bytes can be
    compared directly.
    """
    structured = {
        "rule_groups": {"common": [{"syscall": "read"}, {"syscall": "exit_group"}]},
        "thread_filters": {
            "main": {
                "default_action": "trap",
                "filter_action": "allow",
                "include": ["common"],
                "rules": [{"syscall": "write"}],
            },
            "main_b": {
                "default_action": "trap",
                "filter_action": "allow",
                "include": ["common"],
                "rules": [{"syscall": "close"}],
            },
        },
    }
    flat = {
        "main": _filter("read", "exit_group", "write"),
        "main_b": _filter("read", "exit_group", "close"),
    }
    # The shipped-policy tests read rules through this helper, so it has to expand
    # the policy exactly as seccompiler does.
    assert expand_seccomp_filters(structured) == flat

    seccompiler.compile(structured, split_output=True)
    structured_bpf = {name: (tmp_path / f"{name}.bpf").read_bytes() for name in flat}
    seccompiler.compile(flat, split_output=True)
    assert structured_bpf == {
        name: (tmp_path / f"{name}.bpf").read_bytes() for name in flat
    }

    def probe(name, syscall):
        # fd=-1 keeps an admitted syscall harmless (EBADF); a rejected one
        # never reaches the kernel and ends the probe with SIGSYS.
        return utils.run_cmd(
            [
                str(bin_test_syscall),
                str(tmp_path / f"{name}.bpf"),
                str(seccomp.resolve_syscall(SECCOMP_ARCH, syscall)),
                "-1",
            ],
            shell=False,
        ).returncode

    for name, kept, dropped in [
        ("main", "write", "close"),
        ("main_b", "close", "write"),
    ]:
        assert probe(name, "read") == 0
        assert probe(name, kept) == 0
        assert probe(name, dropped) == -signal.SIGSYS


def test_structured_policy_rejects_unknown_keys(seccompiler):
    """A misspelled key in the structured layout is an error, not a silent no-op."""
    policy = {
        "thread_filters": {
            "main": {
                "default_action": "trap",
                "filter_action": "allow",
                "rules": [{"syscall": "read", "onyl": ["main"]}],
            }
        }
    }
    with pytest.raises(ChildProcessError, match="unknown field `onyl`"):
        seccompiler.compile(policy)


def test_included_rule_groups_must_exist(seccompiler):
    """A filter cannot include a rule group the policy does not define."""
    policy = {
        "thread_filters": {
            "main": {
                "default_action": "trap",
                "filter_action": "allow",
                "include": ["missing"],
                "rules": [],
            }
        }
    }
    with pytest.raises(ChildProcessError, match="MissingRuleGroup"):
        seccompiler.compile(policy)


def test_rule_groups_must_be_included(seccompiler):
    """A rule group no filter includes is dead policy and is rejected."""
    policy = {
        "rule_groups": {"spare": [{"syscall": "read"}]},
        "thread_filters": {
            "main": {"default_action": "trap", "filter_action": "allow", "rules": []}
        },
    }
    with pytest.raises(ChildProcessError, match="UnusedRuleGroup"):
        seccompiler.compile(policy)


@pytest.mark.parametrize("thread", FILTER_THREADS)
def test_validate_filter(seccompiler, bin_test_syscall, monkeypatch, tmp_path, thread):
    """Assert that the seccomp filter for one thread matches the JSON description."""

    fc_filter = json.loads(FC_FILTER_PATH.read_text(encoding="ascii"))

    # cd to a tmp dir because we may generate a bunch of intermediate files
    monkeypatch.chdir(tmp_path)
    # prevent coredumps
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    seccompiler.compile(fc_filter, split_output=True)

    # With split_output=True, individual .bpf files are created for each thread
    arch = SECCOMP_ARCH
    filter_data = expand_seccomp_filters(fc_filter)[thread]
    filter_path = Path(f"{thread}.bpf")
    assert filter_path.exists(), f"Expected {filter_path} to be created by seccompiler"

    # for each rule, run the helper program and execute a syscall
    for rule in filter_data["filter"]:
        print(filter_path, rule)
        syscall = rule["syscall"]
        # this one cannot be called directly
        if syscall in ["rt_sigreturn"]:
            continue
        syscall_id = seccomp.resolve_syscall(arch, syscall)
        cmd = f"{bin_test_syscall} {filter_path} {syscall_id}"
        if "args" not in rule:
            # syscall should be allowed with any arguments and exit 0
            assert utils.run_cmd(cmd).returncode == 0
        else:
            allowed_args = [0] * 4
            # if we call it with allowed args, it should exit 0
            for arg in rule["args"]:
                allowed_args[arg["index"]] = arg["val"]
            allowed_str = " ".join(str(x) for x in allowed_args)
            assert utils.run_cmd(f"{cmd} {allowed_str}").returncode == 0
            # for each allowed arg try a different number
            for arg in rule["args"]:
                bad_args = allowed_args.copy()
                if isinstance(arg["op"], dict) and "masked_eq" in arg["op"]:
                    # For masked_eq, flip the mask bit to violate the check
                    bad_args[arg["index"]] = str(arg["val"] ^ arg["op"]["masked_eq"])
                else:
                    # We just add 1000000 to the allowed arg and assume it
                    # is not something we allow in another rule. While not
                    # perfect it works in practice.
                    bad_args[arg["index"]] = str(arg["val"] + 1_000_000)
                unallowed_str = " ".join(str(x) for x in bad_args)
                outcome = utils.run_cmd(f"{cmd} {unallowed_str}")
                # if we call it with unallowed args, it should exit 159
                # 159 = 128 (abnormal termination) + 31 (SIGSYS)
                assert outcome.returncode == 159
