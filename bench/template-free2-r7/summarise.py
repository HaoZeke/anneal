# Copyright (c) anneal developers
"""Hits, exact p values and outside time per call from a rows file.

    summarise.py ROWS BASE ARM [SKIP]

Fisher is two-sided on the hit counts. McNemar is the exact two-sided
binomial test on the seeds where exactly one arm hit, which is valid
because both arms start each seed from the same configuration. Outside
time per call is the type-7 median of outside_s / charged over the
cell's rows, and the ratio is ARM over BASE. A row whose elapsed time
exceeds its wall time by more than a second and 10 % is flagged as
paused. SKIP is an optional csv of cell,seed pairs whose rows are left
out of the timing, not the hits.
"""
import csv, sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from math import comb


def median(v):
    v = sorted(v)
    h = (len(v) - 1) * 0.5
    lo = int(h)
    return v[lo] + (h - lo) * (v[min(lo + 1, len(v) - 1)] - v[lo])


def fisher(a, n, b, m):
    total, hits = n + m, a + b
    def p(k):
        return comb(n, k) * comb(m, hits - k) / comb(total, hits)
    ks = range(max(0, hits - m), min(n, hits) + 1)
    obs = p(a)
    return min(1.0, sum(p(k) for k in ks if p(k) <= obs * (1 + 1e-9)))


def mcnemar(only_base, only_arm):
    d = only_base + only_arm
    if d == 0:
        return 1.0
    k = min(only_base, only_arm)
    return min(1.0, 2 * sum(comb(d, i) for i in range(k + 1)) / 2**d)


def main():
    rows_path, base, arm = sys.argv[1:4]
    skip = set()
    if len(sys.argv) > 4:
        skip = {(r["cell"], int(r["seed"])) for r in csv.DictReader(open(sys.argv[4]))}
    by = defaultdict(dict)
    paused = []
    for r in csv.DictReader(open(rows_path)):
        by[r["cell"]].setdefault(r["arm"], {})[int(r["seed"])] = r
        if float(r["elapsed_s"]) > float(r["wall_s"]) * 1.1 + 1.0:
            paused.append((r["cell"], r["arm"], r["seed"]))
    print("cell,n,n_timed,hits_base,hits_arm,fisher_p,base_only,arm_only,mcnemar_p,"
          "outside_per_call_base_us,outside_per_call_arm_us,outside_ratio")
    for cell, arms in by.items():
        if base not in arms or arm not in arms:
            continue
        seeds = sorted(set(arms[base]) & set(arms[arm]))
        hb = [int(arms[base][s]["hit"]) for s in seeds]
        ha = [int(arms[arm][s]["hit"]) for s in seeds]
        ob = sum(1 for x, y in zip(hb, ha) if x and not y)
        oa = sum(1 for x, y in zip(hb, ha) if y and not x)
        timed = [s for s in seeds if (cell, s) not in skip]
        def per_call(label):
            return median([float(arms[label][s]["outside_s"]) / float(arms[label][s]["charged"])
                           for s in timed]) * 1e6
        pb, pa = per_call(base), per_call(arm)
        ratio = format(Decimal(str(pa / pb)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP), "f")
        n = len(seeds)
        print(f"{cell},{n},{len(timed)},{sum(hb)},{sum(ha)},{fisher(sum(hb), n, sum(ha), n):.4g},"
              f"{ob},{oa},{mcnemar(ob, oa):.4g},{pb:.4f},{pa:.4f},{ratio}")
    print("paused rows:", paused or "none")


if __name__ == "__main__":
    main()
