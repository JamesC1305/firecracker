# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for telling a Firecracker process's guest memory from its own."""

import platform

# If guest memory is >3GiB, it is split in a 2nd region
# Gap starts at 3GiBs and is 1GiB long
X86_32BIT_MEMORY_GAP_START = 3 << 30
X86_32BIT_MEMORY_GAP_SIZE = 1 << 30
# If guest memory is >255GiB, it is split in a 3rd region
# Gap starts at 256 GiB and is 256GiB long
X86_64BIT_MEMORY_GAP_START = 256 << 30
# On ARM64 we just have a single gap, but memory starts at an offset
# Gap starts at 256 GiB and is GiB long
# Memory starts at 2GiB
ARM64_64BIT_MEMORY_GAP_START = 256 << 30
ARM64_MEMORY_START = 2 << 30


def is_guest_mem_x86(size, guest_mem_bytes):
    """
    Checks if a region is a guest memory region based on
    x86_64 physical memory layout
    """
    # it could be bigger if hotplugging is enabled
    # if it's bigger, it's likely not from FC because we don't have big allocations
    return size >= guest_mem_bytes or size in (
        # memory fits before the first gap
        guest_mem_bytes,
        # guest memory spans at least two regions & memory fits before the second gap
        X86_32BIT_MEMORY_GAP_START,
        # guest memory spans exactly two regions
        guest_mem_bytes - X86_32BIT_MEMORY_GAP_START,
        # guest memory fills the space between the two gaps
        X86_64BIT_MEMORY_GAP_START
        - X86_32BIT_MEMORY_GAP_START
        - X86_32BIT_MEMORY_GAP_SIZE,
        # guest memory spans 3 regions, this is what remains past the second gap
        guest_mem_bytes - X86_64BIT_MEMORY_GAP_START + X86_32BIT_MEMORY_GAP_SIZE,
    )


def is_guest_mem_arch64(size, guest_mem_bytes):
    """
    Checks if a region is a guest memory region based on
    ARM64 physical memory layout
    """
    # it could be bigger if hotplugging is enabled
    # if it's bigger, it's likely not from FC because we don't have big allocations
    return size >= guest_mem_bytes or size in (
        # guest memory fits before the gap
        guest_mem_bytes,
        # guest memory fills the space before the gap
        ARM64_64BIT_MEMORY_GAP_START - ARM64_MEMORY_START,
        # guest memory spans 2 regions, this is what remains past the gap
        guest_mem_bytes - ARM64_64BIT_MEMORY_GAP_START + ARM64_MEMORY_START,
    )


def is_guest_mem(size, guest_mem_bytes):
    """
    If the address is recognised as a guest memory region,
    return True, otherwise return False.
    """

    if platform.machine() == "x86_64":
        return is_guest_mem_x86(size, guest_mem_bytes)

    return is_guest_mem_arch64(size, guest_mem_bytes)
