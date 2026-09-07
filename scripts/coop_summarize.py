#!/usr/bin/env python3
"""Summarise a cooperative LJ campaign written by elja_coop_lj.sh.

Usage: coop_summarize.py <campaign-dir> [target-energy] [tolerance]

For each ensemble directory under <campaign-dir>/lj*/shared/ the script
reads every workers/replica-*.out, takes the deepest verified best energy
and the smallest first-target call count, and counts the ensemble solved
when any replica reached the target within the tolerance. It also counts
coordinator watchdog lines (slow requests) and whether TERMINAL_OK exists.
Defaults: LJ75 Marks -397.492331, tolerance 1e-4.
"""
import glob
import os
import re
import sys

root = sys.argv[1]
target = float(sys.argv[2]) if len(sys.argv) > 2 else -397.492331
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
best_re = re.compile(r"best (-?\d+\.\d+)")
first_re = re.compile(r"first_target (\d+)")
wall_re = re.compile(r"wall ([0-9.]+)s")

rows = []
for ens in sorted(glob.glob(os.path.join(root, "lj*", "shared", "*"))):
    outs = glob.glob(os.path.join(ens, "workers", "replica-*.out"))
    best = float("inf")
    firsts = []
    walls = []
    for path in outs:
        text = open(path, errors="replace").read()
        for line in text.splitlines():
            if "SOLVED" in line or "replica" in line and "best" in line and "charged" in line:
                m = best_re.search(line)
                if m:
                    best = min(best, float(m.group(1)))
                m = first_re.search(line)
                if m and "SOLVED" in line:
                    firsts.append(int(m.group(1)))
                m = wall_re.search(line)
                if m:
                    walls.append(float(m.group(1)))
    slow = 0
    err = os.path.join(ens, "coordinator.err")
    if os.path.exists(err):
        slow = sum(1 for l in open(err, errors="replace") if "slow request" in l)
    done = os.path.exists(os.path.join(ens, "TERMINAL_OK"))
    solved = best <= target + tol
    rows.append((os.path.basename(ens), len(outs), done, solved, best, min(firsts) if firsts else None, max(walls) if walls else None, slow))

print(f"{'ensemble':22s} workers done solved best          first_target wall_s slow")
for name, n, done, solved, best, first, wall, slow in rows:
    print(f"{name:22s} {n:7d} {str(done):5s} {str(solved):6s} {best:13.6f} {str(first):12s} {str(wall):6s} {slow}")
finished = [r for r in rows if r[2]]
print(f"ensembles {len(rows)} finished {len(finished)} solved {sum(1 for r in finished if r[3])} of finished; slow-request lines {sum(r[7] for r in rows)}")
