#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""Summarizes one JSON-lines run of bench_backends as scaling tables.

    scaling_table.py [--workflow W] [--mesh M] run.jsonl

The first table has one row per (workflow, mesh): the Serial time, the host
backend's speedup and parallel efficiency against Serial at every thread
count, the device time, and the device speedup against Serial and against the
fastest host row. The second lists ns per op against the problem size for
every (workflow, space), so a departure from constant cost per op shows. Of
rows the sweep repeats per thread count (Serial, device, copies) the fastest
is kept. Without a Serial row the one-thread host row is the reference, and
a run at a single other thread count reports only the device-over-host ratio. A
copy workflow (mirror_in, mirror_out) is attributed to a memory space, so the
host row stands in for Serial and the device row for the device.
Standard library only.
"""

import argparse
import json
import sys
from collections import defaultdict


def _mesh_size(mesh):
    try:
        return int(mesh.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _classify(r):
    """Returns 'serial', 'host', 'device' or None for a row."""
    space = r["space"]
    device = r.get("device_backend", "none")
    host = r.get("host_backend", "none")
    if space == "Serial" or space == "HostSpace":
        return "serial"
    if space == device or space == device + "Space":
        return "device"
    if space == host:
        return "host"
    return None


def summarize(rows):
    """One record per (workflow, mesh), ordered by workflow then mesh size."""
    groups = defaultdict(lambda: {"serial": None, "device": None, "host": {}})
    for r in rows:
        kind = _classify(r)
        if kind is None:
            continue
        g = groups[(r["workflow"], r["mesh"])]
        if kind == "host":
            t = r["threads"]
            if t not in g["host"] or r["median_sec"] < g["host"][t]["median_sec"]:
                g["host"][t] = r
        elif g[kind] is None or r["median_sec"] < g[kind]["median_sec"]:
            g[kind] = r
    out = []
    for (workflow, mesh), g in sorted(groups.items(), key=lambda kv: (kv[0][0], _mesh_size(kv[0][1]), kv[0][1])):
        serial = g["serial"]["median_sec"] if g["serial"] else None
        host_rows = sorted(g["host"].items())
        # Without Serial, the one-thread host row is the reference; a run at
        # a single other thread count has none and reports no speedup.
        ref = serial
        if ref is None and host_rows and host_rows[0][0] == 1:
            ref = host_rows[0][1]["median_sec"]
        host = []
        best_host = None
        for t, r in host_rows:
            sec = r["median_sec"]
            speedup = ref / sec if ref and sec > 0 else None
            eff = speedup / t if speedup is not None and t > 0 else None
            host.append((t, speedup, eff))
            if best_host is None or sec < best_host:
                best_host = sec
        device = g["device"]["median_sec"] if g["device"] else None
        rec = {
            "workflow": workflow,
            "mesh": mesh,
            "serial_sec": serial,
            "host": host,
            "device_sec": device,
            "device_speedup": ref / device if device and ref else None,
            "device_over_best_host": best_host / device if device and best_host else None,
        }
        out.append(rec)
    return out


def size_scaling(rows):
    """{(workflow, space): [(mesh, ops, ns_per_op), ...]} in mesh size order."""
    best = {}
    for r in rows:
        k = (r["workflow"], r["space"], r["mesh"])
        if k not in best or r["median_sec"] < best[k]["median_sec"]:
            best[k] = r
    out = defaultdict(list)
    for (workflow, space, mesh), r in best.items():
        ns = r.get("ns_per_op")
        if ns is None and r.get("ops"):
            ns = r["median_sec"] * 1e9 / r["ops"]
        out[(workflow, space)].append((mesh, r.get("ops", 0), ns))
    for v in out.values():
        v.sort(key=lambda t: (_mesh_size(t[0]), t[0]))
    return dict(out)


def format_summary(records):
    threads = sorted({t for r in records for t, _, _ in r["host"]})
    head = f"{'workflow':<20} {'mesh':<10} {'serial_s':>10}"
    for t in threads:
        head += f" {'x' + str(t):>6} {'eff':>5}"
    head += f" {'device_s':>10} {'dev/ser':>8} {'dev/host':>8}"
    lines = [head]
    for r in records:
        line = f"{r['workflow']:<20} {r['mesh']:<10} "
        line += f"{r['serial_sec']:>10.6f}" if r["serial_sec"] is not None else f"{'':>10}"
        by_t = {t: (s, e) for t, s, e in r["host"]}
        for t in threads:
            if t in by_t and by_t[t][0] is not None:
                line += f" {by_t[t][0]:>6.2f} {by_t[t][1]:>5.2f}"
            else:
                line += f" {'':>6} {'':>5}"
        line += f" {r['device_sec']:>10.6f}" if r["device_sec"] is not None else f" {'':>10}"
        line += f" {r['device_speedup']:>8.2f}" if r["device_speedup"] is not None else f" {'':>8}"
        line += f" {r['device_over_best_host']:>8.2f}" if r["device_over_best_host"] is not None else f" {'':>8}"
        lines.append(line)
    return "\n".join(lines)


def format_sizes(sizes):
    lines = [f"{'workflow':<20} {'space':<10} {'mesh':<10} {'ops':>10} {'ns_per_op':>12}"]
    for (workflow, space), items in sorted(sizes.items()):
        for mesh, ops, ns in items:
            ns_text = f"{ns:>12.2f}" if ns is not None else f"{'':>12}"
            lines.append(f"{workflow:<20} {space:<10} {mesh:<10} {ops:>10} {ns_text}")
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
    p.add_argument("run")
    p.add_argument("--workflow")
    p.add_argument("--mesh")
    a = p.parse_args(argv)
    rows = _load(a.run)
    if a.workflow:
        rows = [r for r in rows if r["workflow"] == a.workflow]
    if a.mesh:
        rows = [r for r in rows if r["mesh"] == a.mesh]
    print(format_summary(summarize(rows)))
    print()
    print(format_sizes(size_scaling(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
