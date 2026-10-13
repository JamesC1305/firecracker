# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for measuring memory utilization for a process."""

import time
from threading import Thread

import psutil

from framework.utils_memory import is_guest_mem


class MemoryUsageExceededError(Exception):
    """A custom exception containing details on excessive memory usage."""

    def __init__(self, usage, threshold, *args):
        """Compose the error message containing the memory consumption."""
        super().__init__(
            f"Memory usage ({usage / (1 << 20):.2f} MiB) exceeded maximum threshold "
            f"({threshold / (1 << 20)} MiB)",
            *args,
        )


class MemoryMonitor(Thread):
    """Class to represent an RSS memory monitor for a Firecracker process.

    The guest's memory region is skipped, as the main interest is the
    VMM memory usage.
    """

    def __init__(
        self,
        vm,
        threshold_booted=5 << 20,
        threshold_snapshot=7 << 20,
        threshold_restored=5 << 20,
        period_s=0.01,
    ):
        """Initialize monitor attributes."""
        Thread.__init__(self)
        self._vm = vm
        self.threshold_booted = threshold_booted
        self.threshold_snapshot = threshold_snapshot
        self.threshold_restored = threshold_restored
        # Start with booted threshold by default
        self.threshold = threshold_booted
        self._exceeded = None
        self._period_s = period_s
        self._should_stop = False
        self._current_rss = 0
        self.daemon = True

    def signal_stop(self):
        """Signal that the thread should stop."""
        self._should_stop = True

    def set_threshold_for_restored_vm(self):
        """Set threshold for a restored VM."""
        self.threshold = self.threshold_restored

    def set_threshold_for_snapshot(self):
        """Set threshold for snapshot creation."""
        self.threshold = self.threshold_snapshot

    def stop(self):
        """Stop the thread"""
        if self.is_alive():
            self.signal_stop()
            self.join(timeout=1)

    def run(self):
        """Thread for monitoring the RSS memory usage of a Firecracker process.

        If overhead memory exceeds the maximum value, it is saved and memory
        monitoring ceases. It is up to the caller to check.
        """

        guest_mem_bytes = self._vm.mem_size_bytes
        try:
            ps = psutil.Process(self._vm.firecracker_pid)
        except (psutil.NoSuchProcess, FileNotFoundError):
            return
        while not self._should_stop:
            try:
                mmaps = ps.memory_maps(grouped=False)
            except psutil.NoSuchProcess:
                return
            mem_total = 0
            for mmap in mmaps:
                if is_guest_mem(mmap.size, guest_mem_bytes):
                    continue

                mem_total += mmap.rss
            self._current_rss = mem_total
            if mem_total > self.threshold:
                self._exceeded = ps
                return

            time.sleep(self._period_s)

    def check_samples(self):
        """Check that there are no samples over the threshold."""
        if self._exceeded is not None:
            raise MemoryUsageExceededError(
                self._current_rss, self.threshold, self._exceeded
            )

    @property
    def current_rss(self):
        """Obtain current RSS for Firecracker's overhead."""
        # This is to ensure that the monitor has updated itself.
        time.sleep(2 * self._period_s)
        return self._current_rss

    def __enter__(self):
        """To use it as a Context Manager

        >>> mm = MemoryMonitor(vm, threshold=10*1024)
        >>> with mm:
        >>>    # do stuff
        """

        self.start()

    def __exit__(self, _type, _value, _traceback):
        """Exit context"""

        if self.is_alive():
            self.signal_stop()
            self.join(timeout=1)
        self.check_samples()
