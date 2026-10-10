#!/usr/bin/env python3
"""Summarise a cooperative LJ campaign written by elja_coop_lj.sh.

Usage: coop_summarize.py <campaign-dir> [target-energy] [tolerance]

For each ensemble directory under <campaign-dir>/lj*/shared/ the script
reads every workers/replica-*.out, takes the deepest personal best energy
and the smallest hop count at which a replica reached the target, and counts the ensemble solved
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
best_re = re.compile(r"personal best (-?\d+\.\d+)\s+hops (\d+)")
final_re = re.compile(r"hops-per-core-hour [0-9.]+\s+hops (\d+)\s+wall ([0-9.]+)s")

rows = []
for ens in sorted(glob.glob(os.path.join(root, "lj*", "shared", "*"))):
    outs = glob.glob(os.path.join(ens, "workers", "replica-*.out"))
    best = float("inf")
    firsts = []
    walls = []
    for path in outs:
        text = open(path, errors="replace").read()
        for m in best_re.finditer(text):
            energy, hops = float(m.group(1)), int(m.group(2))
            if energy < best:
                best = energy
            if energy <= target + tol:
                firsts.append(hops)
        m = None
        for m in final_re.finditer(text):
            pass
        if m:
            walls.append(float(m.group(2)))
    slow = 0
    err = os.path.join(ens, "coordinator.err")
    if os.path.exists(err):
        slow = sum(1 for l in open(err, errors="replace") if "slow request" in l)
    done = os.path.exists(os.path.join(ens, "TERMINAL_OK"))
    solved = best <= target + tol
    rows.append((os.path.basename(ens), len(outs), done, solved, best, min(firsts) if firsts else None, max(walls) if walls else None, slow))

print(f"{'ensemble':22s} workers done solved best          first_hops   wall_s slow")
for name, n, done, solved, best, first, wall, slow in rows:
    print(f"{name:22s} {n:7d} {str(done):5s} {str(solved):6s} {best:13.6f} {str(first):12s} {str(wall):6s} {slow}")
finished = [r for r in rows if r[2]]
print(f"ensembles {len(rows)} finished {len(finished)} solved {sum(1 for r in finished if r[3])} of finished; slow-request lines {sum(r[7] for r in rows)}")
