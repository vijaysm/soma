#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""Compares two JSON-lines benchmark runs (bench_backends --json, or the
other benchmarks with --json).

    compare_bench.py [--min-change F] [--workflow W] [--mesh M] [--space S]
                     baseline.jsonl current.jsonl

Rows are matched on benchmark, workflow, mesh, space and ranks, plus the
thread count except for Serial and device rows, which the sweep repeats per
thread count; of repeated rows the one with the smallest median is kept.
ratio is baseline / current: above 1 is faster now. A ratio within the
larger relative MAD of the two rows is noise and flagged "~"; rows present
on one side only are "new" or "gone". Standard library only.
"""

import argparse
import json
import sys


def _is_thread_keyed(r):
    return r["space"] != "Serial" and r["space"] != r.get("device_backend", "none")


def _key(r):
    threads = r["threads"] if _is_thread_keyed(r) else None
    return (r["benchmark"], r["workflow"], r["mesh"], r["space"], r.get("ranks", 1), threads)


def _index(rows):
    best = {}
    for r in rows:
        k = _key(r)
        if k not in best or r["median_sec"] < best[k]["median_sec"]:
            best[k] = r
    return best


def _spread(r):
    return r["mad_sec"] / r["median_sec"] if r["median_sec"] > 0 else 0.0


def compare(baseline_rows, current_rows, min_change=0.0):
    """Returns one record per matched key, sorted by key."""
    base = _index(baseline_rows)
    cur = _index(current_rows)
    out = []
    for k in sorted(set(base) | set(cur), key=lambda t: tuple(str(x) for x in t)):
        benchmark, workflow, mesh, space, ranks, threads = k
        rec = {"benchmark": benchmark, "workflow": workflow, "mesh": mesh, "space": space,
               "ranks": ranks, "threads": threads, "baseline": None, "current": None,
               "ratio": None, "spread": 0.0, "flag": ""}
        b = base.get(k)
        c = cur.get(k)
        if b is None:
            rec["current"] = c["median_sec"]
            rec["flag"] = "new"
        elif c is None:
            rec["baseline"] = b["median_sec"]
            rec["flag"] = "gone"
        else:
            rec["baseline"] = b["median_sec"]
            rec["current"] = c["median_sec"]
            rec["ratio"] = b["median_sec"] / c["median_sec"] if c["median_sec"] > 0 else 0.0
            rec["spread"] = max(_spread(b), _spread(c))
            change = abs(rec["ratio"] - 1.0)
            if change < min_change:
                continue
            if change <= rec["spread"]:
                rec["flag"] = "~"
            elif rec["ratio"] > 1.0:
                rec["flag"] = "+"
            else:
                rec["flag"] = "-"
        out.append(rec)
    return out


def format_table(records):
    head = f"{'workflow':<20} {'mesh':<12} {'space':<10} {'thr':>4} {'baseline_s':>12} " \
           f"{'current_s':>12} {'ratio':>7} {'spread':>7} flag"
    lines = [head]
    for r in records:
        thr = "" if r["threads"] is None else str(r["threads"])
        b = "" if r["baseline"] is None else f"{r['baseline']:.6f}"
        c = "" if r["current"] is None else f"{r['current']:.6f}"
        ratio = "" if r["ratio"] is None else f"{r['ratio']:.2f}"
        lines.append(f"{r['workflow']:<20} {r['mesh']:<12} {r['space']:<10} {thr:>4} {b:>12} "
                     f"{c:>12} {ratio:>7} {r['spread']:>7.3f} {r['flag']}")
    return "\n".join(lines)


def _load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("baseline")
    p.add_argument("current")
    p.add_argument("--min-change", type=float, default=0.0,
                   help="hide matched rows whose |ratio - 1| is below this")
    p.add_argument("--workflow")
    p.add_argument("--mesh")
    p.add_argument("--space")
    a = p.parse_args(argv)

    def keep(r):
        return ((a.workflow is None or r["workflow"] == a.workflow) and
                (a.mesh is None or r["mesh"] == a.mesh) and
                (a.space is None or r["space"] == a.space))

    base = [r for r in _load(a.baseline) if keep(r)]
    cur = [r for r in _load(a.current) if keep(r)]
    print(format_table(compare(base, cur, a.min_change)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
