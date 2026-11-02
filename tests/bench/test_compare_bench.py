# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""Checks of bench/compare_bench.py: keys, ratios, flags and one-sided rows."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bench"))

import compare_bench  # noqa: E402


def row(workflow, space, median, mad=0.0, threads=1, mesh="hex_32", device="Cuda"):
    return {
        "benchmark": "backends",
        "workflow": workflow,
        "mesh": mesh,
        "space": space,
        "ranks": 1,
        "threads": threads,
        "median_sec": median,
        "mad_sec": mad,
        "host_backend": "OpenMP",
        "device_backend": device,
    }


class CompareBench(unittest.TestCase):
    def test_matched_row_reports_the_ratio_baseline_over_current(self):
        out = compare_bench.compare([row("derive", "OpenMP", 2.0)], [row("derive", "OpenMP", 1.0)])
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["ratio"], 2.0)
        self.assertEqual(out[0]["flag"], "+")

    def test_a_change_inside_the_spread_is_flagged_as_noise(self):
        out = compare_bench.compare([row("derive", "OpenMP", 1.0, mad=0.2)],
                                    [row("derive", "OpenMP", 0.9, mad=0.05)])
        self.assertEqual(out[0]["flag"], "~")
        self.assertAlmostEqual(out[0]["spread"], 0.2)

    def test_slower_rows_are_flagged_minus(self):
        out = compare_bench.compare([row("walk", "Serial", 1.0)], [row("walk", "Serial", 1.5)])
        self.assertEqual(out[0]["flag"], "-")

    def test_rows_on_one_side_only_are_kept_as_new_or_gone(self):
        out = compare_bench.compare([row("derive", "OpenMP", 1.0)], [row("walk", "OpenMP", 1.0)])
        flags = sorted(r["flag"] for r in out)
        self.assertEqual(flags, ["gone", "new"])

    def test_host_backend_rows_are_keyed_by_thread_count(self):
        base = [row("derive", "OpenMP", 1.0, threads=1), row("derive", "OpenMP", 0.5, threads=8)]
        cur = [row("derive", "OpenMP", 1.0, threads=1), row("derive", "OpenMP", 0.25, threads=8)]
        out = compare_bench.compare(base, cur)
        self.assertEqual(len(out), 2)
        by_threads = {r["threads"]: r["ratio"] for r in out}
        self.assertAlmostEqual(by_threads[1], 1.0)
        self.assertAlmostEqual(by_threads[8], 2.0)

    def test_serial_and_device_rows_collapse_across_thread_counts(self):
        base = [row("derive", "Serial", 1.0, threads=1), row("derive", "Serial", 1.1, threads=8),
                row("derive", "Cuda", 0.2, threads=1), row("derive", "Cuda", 0.3, threads=8)]
        cur = [row("derive", "Serial", 1.0, threads=4), row("derive", "Cuda", 0.1, threads=4)]
        out = compare_bench.compare(base, cur)
        self.assertEqual(len(out), 2)
        by_space = {r["space"]: r for r in out}
        self.assertIsNone(by_space["Serial"]["threads"])
        self.assertAlmostEqual(by_space["Serial"]["baseline"], 1.0)  # the smallest median wins
        self.assertAlmostEqual(by_space["Cuda"]["ratio"], 2.0)

    def test_min_change_hides_small_changes_but_not_one_sided_rows(self):
        base = [row("derive", "OpenMP", 1.0), row("walk", "OpenMP", 1.0)]
        cur = [row("derive", "OpenMP", 0.99)]
        out = compare_bench.compare(base, cur, min_change=0.05)
        self.assertEqual([r["flag"] for r in out], ["gone"])

    def test_table_lists_every_result(self):
        out = compare_bench.compare([row("derive", "OpenMP", 2.0)], [row("derive", "OpenMP", 1.0)])
        text = compare_bench.format_table(out)
        self.assertIn("derive", text)
        self.assertIn("2.00", text)


if __name__ == "__main__":
    unittest.main()
