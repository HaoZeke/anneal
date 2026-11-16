# Copyright (c) anneal developers
"""Where a cluster-hopping run stalls, read off its stderr traces.

The traces are the ones `cluster_hopping` writes under environment
switches: `ANNEAL_HOP_TRACE` (one JSON line per hop), `ANNEAL_ACC_TRACE`
(one line per adoption, with the action) and `ANNEAL_IMP_TRACE` (one line
per new best). Each trace file is read once and summarised per window of
charged calls:

- the chain's energy above the incumbent, as median and quartiles;
- acceptance, returns, screened and fully quenched trials;
- distinct quenched minima offered and distinct minima the chain stood on,
  keyed by energy to 1e-6;
- the stall actions taken (trail, climb, symmetrise, reseeds).

    diagnose.py --reference -191.0308528 trace-2002.txt.gz [...]
"""

import argparse
import collections
import gzip
import json
import statistics


def opener(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def quantiles(values):
    if not values:
        return (float("nan"),) * 3
    ordered = sorted(values)

    def at(q):
        pos = q * (len(ordered) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    return at(0.25), at(0.5), at(0.75)


def read(path):
    hops = []
    actions = []
    records = []
    for line in opener(path):
        if line.startswith("HOPTRACE "):
            row = json.loads(line[len("HOPTRACE ") :])
            row.pop("x", None)
            hops.append(row)
        elif line.startswith("ACCTRACE "):
            parts = line.split()
            actions.append((int(parts[2]), float(parts[4]), parts[6]))
        elif line.startswith("IMPTRACE "):
            parts = line.split()
            records.append((int(parts[2]), float(parts[4]), parts[6]))
    return hops, actions, records


def summarise(path, reference, window):
    hops, actions, records = read(path)
    if not hops:
        print(f"{path}: no HOPTRACE lines")
        return
    best = min(row["best"] for row in hops)
    last = max(hops, key=lambda row: row["spent"])
    first_at_best = min(row["spent"] for row in hops if row["best"] <= best + 1e-9)
    print(f"== {path}")
    print(
        f"best {best:.8f} gap {best - reference:.8f} hops {len(hops)} "
        f"calls {last['spent']} best first held at {first_at_best}"
    )
    print(
        "window      hops  acc  ret  scr  full   gap_q1   gap_med  gap_q3  "
        "offered stood  best_in_win"
    )
    by_window = collections.defaultdict(list)
    for row in hops:
        by_window[row["spent"] // window].append(row)
    hop_window = {row["hop"]: row["spent"] // window for row in hops}
    for key in sorted(by_window):
        rows = by_window[key]
        accepted = [row for row in rows if row["accept"]]
        returning = sum(row["returning"] for row in rows)
        screened = sum(row["screened"] and not row["returning"] for row in rows)
        full = sum(not row["unquenched"] for row in rows)
        gaps = [row["e"] - row["best"] for row in rows]
        q1, med, q3 = quantiles(gaps)
        offered = {round(row["e_new"], 6) for row in rows if not row["unquenched"]}
        stood = {round(row["e"], 6) for row in rows}
        low = min(row["e_new"] for row in rows if not row["unquenched"]) if full else float("nan")
        print(
            f"{key * window:>9} {len(rows):>6} {len(accepted) / len(rows):4.2f} "
            f"{returning / len(rows):4.2f} {screened / len(rows):4.2f} "
            f"{full / len(rows):4.2f} {q1:8.4f} {med:8.4f} {q3:8.4f} "
            f"{len(offered):>7} {len(stood):>5}  {low:.6f}"
        )
    arms = collections.defaultdict(lambda: [0, 0, 0])
    for row in hops:
        arm = arms[row["arm"]]
        arm[0] += 1
        arm[1] += row["accept"]
        arm[2] += (not row["unquenched"]) and row["e_new"] < row["e"] - 1e-6
    print("arm                proposed accepted downhill")
    for name, (proposed, accepted, down) in sorted(arms.items(), key=lambda item: -item[1][0]):
        print(f"{name:<18} {proposed:>8} {accepted:>8} {down:>8}")
    named = collections.Counter()
    for hop, energy, action in actions:
        if action.startswith(("stall", "csm", "motif", "continuous", "orbit", "seam", "prior")):
            named[action] += 1
    print("actions", dict(named))
    for hop, energy, action in actions:
        if action.startswith(("stall", "csm", "motif", "seam")):
            print(f"  hop {hop} window {hop_window.get(hop, '?')} {action} e {energy:.6f}")
    print("records")
    for hop, energy, arm in records:
        print(f"  hop {hop} e {energy:.6f} arm {arm}")
    energies = collections.Counter(round(row["e"], 6) for row in hops)
    print("most occupied minima (energy, hops)")
    for energy, count in energies.most_common(8):
        print(f"  {energy:.6f} {count}")
    tail = [row for row in hops if row["spent"] >= first_at_best]
    if tail:
        gaps = [row["e"] - best for row in tail]
        print(
            "after the final best: hops {} acceptance {:.3f} median gap {:.4f} "
            "distinct stood {} fraction within 0.05 {:.3f}".format(
                len(tail),
                sum(row["accept"] for row in tail) / len(tail),
                statistics.median(gaps),
                len({round(row["e"], 6) for row in tail}),
                sum(gap < 0.05 for gap in gaps) / len(gaps),
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=float, required=True)
    parser.add_argument("--window", type=int, default=100_000)
    parser.add_argument("traces", nargs="+")
    args = parser.parse_args()
    for path in args.traces:
        summarise(path, args.reference, args.window)


if __name__ == "__main__":
    main()
