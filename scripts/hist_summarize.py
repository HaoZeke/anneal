#!/usr/bin/env python3
"""Summarise an ensemble campaign directory written by elja_hist_campaign.sh.

Usage: hist_summarize.py <campaign-dir> [--org] [<control-arm> <treatment-arm>]

For every arm (the prefix before `_<seed>.out`) prints tasks, finished,
solved, first-target aggregate calls (median), mean wall per seed, mean
history seconds per replica, mean shared deposits, gossip rounds and
two-choice restarts per replica, and the solved seed list. Single-chain
arms (no `ensemble:` lines) report solved and hop counts. `--org` prints
an Org table.
"""
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ENSEMBLE = re.compile(
    r"^  seed (\d+) ensemble: best (\S+)\s+aggregate charged (\d+)\s+first_target_calls (\S+).*?wall ([0-9.]+)s(\s+SOLVED)?"
)
REPLICA = re.compile(
    r"^    seed (\d+) replica (\d+).*?hops (\d+)\s+charged (\d+)\s+basins (\d+)"
    r".*?history obs (\d+) new (\d+) refused (\d+) secs ([0-9.]+)\s+shared_deposits (\d+)"
    r"(?:\s+bias_published (\d+))?(?:\s+gossip (\d+))?(?:\s+gossip_interval (\d+))?(?:\s+two_choice_restarts (\d+))?"
)
SINGLE = re.compile(r"^  seed (\d+): best (\S+)\s+hops (\d+).*?(SOLVED)?$")
CROSSED = re.compile(r"crossed at hop \d+ of \d+ .*?, (\d+) charged")


def num(value):
    try:
        return float(value)
    except ValueError:
        return None


def summarise(directory):
    arms = defaultdict(lambda: {
        "tasks": 0, "done": 0, "solved": [], "first": [], "wall": [],
        "hsecs": [], "deposits": [], "gossip": [], "restarts": [], "hops": [],
    })
    for path in sorted(Path(directory).glob("*_*.out")):
        match = re.match(r"(.*)_(\d+)\.out$", path.name)
        if not match:
            continue
        arm, seed = match.group(1), int(match.group(2))
        record = arms[arm]
        record["tasks"] += 1
        text = path.read_text(errors="replace")
        if "gap to reference" in text:
            record["done"] += 1
        for line in text.splitlines():
            m = ENSEMBLE.match(line)
            if m:
                record["wall"].append(float(m.group(5)))
                if m.group(6):
                    record["solved"].append(seed)
                    first = num(m.group(4))
                    if first is not None:
                        record["first"].append(first)
                continue
            m = REPLICA.match(line)
            if m:
                record["hops"].append(int(m.group(3)))
                record["hsecs"].append(float(m.group(9)))
                record["deposits"].append(int(m.group(10)))
                if m.group(12):
                    record["gossip"].append(int(m.group(12)))
                if m.group(14):
                    record["restarts"].append(int(m.group(14)))
                continue
            m = SINGLE.match(line)
            if m and "replica" not in line:
                record["hops"].append(int(m.group(3)))
                if line.rstrip().endswith("SOLVED"):
                    record["solved"].append(seed)
            m = CROSSED.search(line)
            if m:
                record["first"].append(float(m.group(1)))
    return arms


def mean(values, fmt="{:.0f}"):
    return fmt.format(statistics.mean(values)) if values else "-"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    org = "--org" in sys.argv
    arms = summarise(sys.argv[1])
    header = ["arm", "tasks", "done", "solved", "first-target median",
              "wall s", "history s", "deposits", "gossip", "restarts", "hops"]
    rows = []
    for arm, r in sorted(arms.items()):
        rows.append([
            arm, r["tasks"], r["done"], len(r["solved"]),
            f"{statistics.median(r['first']):.3g}" if r["first"] else "-",
            mean(r["wall"]), mean(r["hsecs"], "{:.1f}"), mean(r["deposits"]),
            mean(r["gossip"]), mean(r["restarts"], "{:.2f}"), mean(r["hops"]),
        ])
    if org:
        print("| " + " | ".join(header) + " |")
        print("|" + "+".join("-" * (len(h) + 2) for h in header) + "|")
        for row in rows:
            print("| " + " | ".join(str(v) for v in row) + " |")
    else:
        widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
        for row in [header] + rows:
            print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))
    for arm, r in sorted(arms.items()):
        print(f"{arm} solved seeds: {' '.join(str(s) for s in sorted(r['solved']))}")
    pair = [a for a in sys.argv[2:] if not a.startswith("--")]
    if len(pair) == 2:
        compare(arms, pair[0], pair[1])


def compare(arms, control, treatment):
    """Paired comparison on the seeds both arms finished: gained, lost, sign test."""
    a, b = arms.get(control), arms.get(treatment)
    if a is None or b is None:
        sys.exit(f"unknown arm in comparison: {control} / {treatment}")
    sa, sb = set(a["solved"]), set(b["solved"])
    gained, lost = sorted(sb - sa), sorted(sa - sb)
    n = len(gained) + len(lost)
    # Two-sided exact sign test on the discordant seeds.
    from math import comb
    k = min(len(gained), len(lost))
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    print(
        f"{treatment} vs {control}: {len(sb)} vs {len(sa)} solved; "
        f"gained {len(gained)} {gained}; lost {len(lost)} {lost}; "
        f"discordant {n}, sign test p={p:.3f}"
    )


if __name__ == "__main__":
    main()
