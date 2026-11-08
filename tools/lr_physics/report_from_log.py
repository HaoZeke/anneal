#!/usr/bin/env python3
"""Build the ladder tables from one bench log.

Medians and interquartile ranges are type-7 quantiles of the samples
on the following samples_us line. The printed values are %.17g.
"""

import math
import sys
from collections import defaultdict


def quantile7(values, p):
    xs = sorted(values)
    h = (len(xs) - 1) * p
    lo = math.floor(h)
    hi = math.ceil(h)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


def parse(path):
    rows = []
    pending = None
    header = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("samples_us") and pending is not None:
                samples = [float(x) for x in line.split()[1:]]
                pending["samples"] = samples
                pending["median_s"] = quantile7(samples, 0.5) / 1.0e6
                pending["iqr_us"] = quantile7(samples, 0.75) - quantile7(samples, 0.25)
                rows.append(pending)
                pending = None
                continue
            parts = line.split()
            if len(parts) >= 3 and parts[0] in ("strong", "weak128", "weak1024", "matched") and parts[1] in (
                "ewald",
                "pme",
                "d3",
                "d4",
            ):
                fields = {}
                for item in parts[2:]:
                    if "=" in item:
                        key, value = item.split("=", 1)
                        fields[key] = value
                pending = {
                    "ladder": parts[0],
                    "method": parts[1],
                    "n": int(fields["n"]),
                    "threads": int(fields["threads"]),
                    "energy": fields.get("energy_eV", ""),
                }
                continue
            if pending is None and not line.startswith("samples_us"):
                header.append(line)
    return header, rows


def emit(header, rows):
    print("# Long-range ladder")
    print()
    print("Tables are type-7 medians of the samples in the bench log.")
    print("Times are seconds. The interquartile range is microseconds.")
    print()
    for line in header:
        if line.startswith(("hostname ", "date ", "cpu ", "physical_cores ", "sockets ", "smt ", "gpu_device ")):
            print(line)
    print()
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["ladder"], row["method"])].append(row)
    for key in sorted(grouped):
        ladder, method = key
        print(f"## {ladder} {method}")
        print()
        print("| n | threads | median_s | iqr_us |")
        print("| --- | --- | --- | --- |")
        for row in grouped[key]:
            print(
                f"| {row['n']} | {row['threads']} | {row['median_s']:.17g} | {row['iqr_us']:.17g} |"
            )
        print()


def lower_bounds(path):
    """dftd4 cells killed at 1800 s. A bound is not a point time."""
    rows = []
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 6 or parts[0] != "dftd4" or parts[3] != "timeout":
                continue
            if not parts[4].startswith("limit_s=") or not parts[5].startswith("exit_code="):
                continue
            limit = int(parts[4].split("=", 1)[1])
            code = int(parts[5].split("=", 1)[1])
            if limit != 1800 or code != -9:
                continue
            rows.append((int(parts[1].split("=", 1)[1]), int(parts[2].split("=", 1)[1])))
    return rows


def emit_bounds(path):
    rows = lower_bounds(path)
    print("## dftd4 lower bounds")
    print()
    print("A cell killed at 1800 s is a lower bound. It is not a speedup.")
    print()
    print("| arm | n | threads | bound |")
    print("| --- | --- | --- | --- |")
    for n, threads in rows:
        print(f"| dftd4 | {n} | {threads} | >1800 s |")
    print()


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: report_from_log.py BENCH_LOG [REF_LOG]")
    header, rows = parse(sys.argv[1])
    if not rows:
        raise SystemExit("no samples in the log")
    emit(header, rows)
    if len(sys.argv) == 3:
        emit_bounds(sys.argv[2])


if __name__ == "__main__":
    main()
