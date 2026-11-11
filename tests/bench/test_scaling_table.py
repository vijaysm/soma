# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""Checks of bench/scaling_table.py: speedups, efficiencies and size scaling."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bench"))

import scaling_table  # noqa: E402


def row(workflow, space, median, threads=1, mesh="hex_32", ops=1000, mad=0.0):
    return {
        "benchmark": "backends",
        "workflow": workflow,
        "mesh": mesh,
        "space": space,
        "ranks": 1,
        "threads": threads,
        "repeats": 5,
        "median_sec": median,
        "mad_sec": mad,
        "ops": ops,
        "ns_per_op": median * 1e9 / ops,
        "host_backend": "OpenMP",
        "device_backend": "Cuda",
    }


class ScalingTable(unittest.TestCase):
    def test_speedup_and_efficiency_are_against_serial(self):
        rows = [row("derive", "Serial", 8.0), row("derive", "OpenMP", 8.0, threads=1),
                row("derive", "OpenMP", 2.0, threads=4), row("derive", "Cuda", 0.5, threads=4)]
        out = scaling_table.summarize(rows)
        self.assertEqual(len(out), 1)
        rec = out[0]
        self.assertEqual(rec["workflow"], "derive")
        self.assertEqual(rec["mesh"], "hex_32")
        self.assertAlmostEqual(rec["serial_sec"], 8.0)
        self.assertEqual([t for t, _, _ in rec["host"]], [1, 4])
        self.assertAlmostEqual(rec["host"][1][1], 4.0)  # speedup at 4 threads
        self.assertAlmostEqual(rec["host"][1][2], 1.0)  # efficiency
        self.assertAlmostEqual(rec["device_speedup"], 16.0)
        self.assertAlmostEqual(rec["device_over_best_host"], 4.0)

    def test_repeated_serial_and_device_rows_keep_the_fastest(self):
        rows = [row("walk", "Serial", 2.0, threads=1), row("walk", "Serial", 1.0, threads=8),
                row("walk", "Cuda", 0.4, threads=1), row("walk", "Cuda", 0.2, threads=8)]
        rec = scaling_table.summarize(rows)[0]
        self.assertAlmostEqual(rec["serial_sec"], 1.0)
        self.assertAlmostEqual(rec["device_sec"], 0.2)
        self.assertAlmostEqual(rec["device_speedup"], 5.0)

    def test_a_run_without_serial_uses_one_host_thread_as_the_reference(self):
        rows = [row("walk", "OpenMP", 4.0, threads=1), row("walk", "OpenMP", 1.0, threads=4)]
        rec = scaling_table.summarize(rows)[0]
        self.assertIsNone(rec["serial_sec"])
        self.assertAlmostEqual(rec["host"][1][1], 4.0)
        self.assertIsNone(rec["device_sec"])
        self.assertIsNone(rec["device_speedup"])

    def test_a_single_other_thread_count_has_no_reference(self):
        rows = [row("walk", "OpenMP", 1.0, threads=8), row("walk", "Cuda", 0.25, threads=8)]
        rec = scaling_table.summarize(rows)[0]
        self.assertEqual(rec["host"], [(8, None, None)])
        self.assertIsNone(rec["device_speedup"])
        self.assertAlmostEqual(rec["device_over_best_host"], 4.0)

    def test_memory_space_rows_are_reported_as_copies(self):
        rows = [row("mirror_in", "HostSpace", 1.0), row("mirror_in", "CudaSpace", 2.0)]
        rec = scaling_table.summarize(rows)[0]
        self.assertAlmostEqual(rec["serial_sec"], 1.0)
        self.assertAlmostEqual(rec["device_sec"], 2.0)
        self.assertAlmostEqual(rec["device_speedup"], 0.5)
        self.assertEqual(rec["host"], [])

    def test_size_scaling_lists_ns_per_op_by_mesh_in_size_order(self):
        rows = [row("traverse", "Cuda", 0.001, mesh="hex_64", ops=262144),
                row("traverse", "Cuda", 0.0005, mesh="hex_32", ops=32768),
                row("traverse", "Serial", 0.1, mesh="hex_64", ops=262144)]
        out = scaling_table.size_scaling(rows)
        key = ("traverse", "Cuda")
        self.assertEqual([m for m, _, _ in out[key]], ["hex_32", "hex_64"])
        self.assertAlmostEqual(out[key][0][2], 0.0005 * 1e9 / 32768)

    def test_tables_render(self):
        rows = [row("derive", "Serial", 8.0), row("derive", "OpenMP", 2.0, threads=4),
                row("derive", "Cuda", 0.5)]
        text = scaling_table.format_summary(scaling_table.summarize(rows))
        self.assertIn("derive", text)
        self.assertIn("16.00", text)
        text = scaling_table.format_sizes(scaling_table.size_scaling(rows))
        self.assertIn("hex_32", text)


if __name__ == "__main__":
    unittest.main()
