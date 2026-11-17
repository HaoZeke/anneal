#!/usr/bin/env python3
"""Strong, one-cell and weak CPU scaling tables for BatchEvaluator.

    python scripts/cpu_scaling.py --build-dir BUILD --out DIR [--plot FILE]

Writes strong.csv, onecell.csv and weak.csv to DIR. Each file starts with
``#`` lines naming the host and the build. Every row is a periodic Pt fcc
cell on the default MorsePot, so each call takes the linkcell and minimage
pair list. A timed sample is one whole ``evaluate_batch`` call: copying the
inputs, one potential per worker, the pair lists, the threads and the
returned arrays. Every result is compared bit for bit with a 1-thread call
on the same cells. Constants are derived in docs/orgmode/howto/cpu-scaling.org.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import rgpot

LATTICE = 3.92
SMALL_CELLS = 5
LARGE_CELLS = 14
JITTER = 0.015
SEED = 1000
PER_WORKER = 8
REPEATS = 7
MIN_REPEATS = 6
COLUMNS = (
    "table", "device", "atoms_per_cell", "cells", "threads", "repeats",
    "median_s", "q1_s", "q3_s", "iqr_s", "speedup", "efficiency", "match",
)


def _cpu_batch(cells, threads):
    pot = rgpot.MorsePot()
    pos, types, boxes = zip(*cells)
    return pot.evaluate_batch(list(pos), list(types), list(boxes), threads)


# A GPU backend registers an evaluator with the same signature here.
DEVICES = {"cpu": _cpu_batch}


def _run(*cmd):
    try:
        return subprocess.run(
            cmd, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unrecorded"


def _cores():
    out = _run("env", "-u", "OMP_NUM_THREADS", "nproc")
    return int(out) if out.isdigit() else os.cpu_count() or 1


def host_header(build_dir):
    model = "unrecorded"
    for line in _run("lscpu").splitlines():
        if line.startswith("Model name:"):
            model = line.split(":", 1)[1].strip()
    info = Path(build_dir) / "meson-info"
    compilers = json.loads((info / "intro-compilers.json").read_text())
    options = {
        item["name"]: item["value"]
        for item in json.loads((info / "intro-buildoptions.json").read_text())
    }
    wanted = ("buildtype", "optimization", "debug", "b_ndebug", "b_lto",
              "cpp_args", "fortran_args", "with_fast_nlist", "with_fortran_pots")
    flags = " ".join(f"{k}={options[k]}" for k in wanted if k in options)
    return {
        "uname_m": _run("uname", "-m"),
        "cpu_model": model,
        "cores": str(_cores()),
        "compilers": " ".join(
            f"{lang}={c['id']}-{c['version']}"
            for lang, c in sorted(compilers.get("host", {}).items())
        ),
        "build_flags": flags,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rgpot": rgpot.__version__,
    }


def pt_cell(repeat, seed):
    basis = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    grid = np.indices((repeat,) * 3).reshape(3, -1).T
    sites = (grid[:, None, :] + basis[None]).reshape(-1, 3) * LATTICE
    rng = np.random.default_rng(seed)
    pos = sites + rng.normal(scale=JITTER, size=sites.shape)
    box = np.eye(3) * repeat * LATTICE
    return (np.ascontiguousarray(pos), np.full(len(pos), 78, np.int32),
            np.ascontiguousarray(box))


def _bytes(rows):
    return [np.float64(r["energy"]).tobytes() + r["forces"].tobytes()
            for r in rows]


def quartiles(samples):
    q1, median, q3 = statistics.quantiles(samples, n=4, method="inclusive")
    return median, q1, q3


def measure(table, plan, repeats, device):
    """plan maps a thread count to its cells; widths are interleaved."""
    evaluate = DEVICES[device]
    refs = {p: _bytes(evaluate(cells, 1)) for p, cells in plan.items()}
    samples = {p: [] for p in plan}
    match = {p: True for p in plan}
    for _ in range(repeats + 1):
        for p, cells in plan.items():
            start = time.perf_counter()
            rows = evaluate(cells, p)
            samples[p].append(time.perf_counter() - start)
            match[p] &= _bytes(rows) == refs[p]
    out = []
    base = None
    for p, cells in plan.items():
        median, q1, q3 = quartiles(samples[p][1:])
        base = base or median
        speedup = base / median * (p if table == "weak" else 1)
        out.append({
            "table": table, "device": device,
            "atoms_per_cell": len(cells[0][0]), "cells": len(cells),
            "threads": p, "repeats": repeats, "median_s": f"{median:.6e}",
            "q1_s": f"{q1:.6e}", "q3_s": f"{q3:.6e}",
            "iqr_s": f"{q3 - q1:.6e}", "speedup": f"{speedup:.4f}",
            "efficiency": f"{speedup / p:.4f}", "match": str(match[p]).lower(),
        })
    return out


def widths(cores):
    out = [1 << k for k in range(cores.bit_length()) if 1 << k <= cores]
    return out + ([cores] if cores not in out else [])


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="ascii") as handle:
        for key, value in header.items():
            handle.write(f"# {key}: {value}\n")
        writer = csv.DictWriter(handle, fieldnames=COLUMNS,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(tables, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for name, rows in tables.items():
        p = [int(r["threads"]) for r in rows]
        ax.plot(p, [float(r["efficiency"]) for r in rows], "o-", label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(p, [str(x) for x in p])
    ax.set_xlabel("threads")
    ax.set_ylabel("parallel efficiency")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--threads", type=int, nargs="+")
    ap.add_argument("--device", choices=sorted(DEVICES), default="cpu")
    ap.add_argument("--plot", type=Path)
    args = ap.parse_args(argv)
    if args.repeats < MIN_REPEATS:
        ap.error(f"--repeats must be at least {MIN_REPEATS}")
    header = host_header(args.build_dir)
    header["device"] = args.device
    ps = sorted({1, *(args.threads or widths(int(header["cores"])))})
    small = [pt_cell(SMALL_CELLS, SEED + i)
             for i in range(PER_WORKER * max(ps))]
    large = [pt_cell(LARGE_CELLS, SEED)]
    plans = {
        "strong": {p: small for p in ps},
        "onecell": {p: large for p in ps},
        "weak": {p: small[:PER_WORKER * p] for p in ps},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, plan in plans.items():
        tables[name] = measure(name, plan, args.repeats, args.device)
        write_csv(args.out / f"{name}.csv", header, tables[name])
    if args.plot:
        plot(tables, args.plot)
    return 0 if all(r["match"] == "true"
                    for rows in tables.values() for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
