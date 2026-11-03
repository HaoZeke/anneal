"""Score and plot a Sutton-Chen head-to-head over a seed range.

usage: plot_sc_hits.py <lo> <hi> <budget> <out.png> <label=dir>...

Each directory has one file per seed. A ``seed<k>.out`` file is the
output of ``examples/sc_h2h.rs``, and a ``seed<k>.json`` file from
another code needs the keys ``hit`` and ``calls_at_hit``.

Penalised average runtime with factor 2 (PAR-2) charges a miss twice
the budget. The paired test flips the sign of each per-seed PAR-2
difference against the first label, with 20000 draws from seed 0.
"""

import json
import random
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Sign-flip draws of the paired test.
SIGNFLIP_DRAWS = 20000
#: Slack that counts a draw tied with the observed mean as extreme.
TIE_TOLERANCE = 1e-15
#: Top of the hit-fraction scale, a little above one so a full curve shows.
FRACTION_TOP = 1.02
#: Figure size in inches.
FIGSIZE = (6, 3.5)
#: Resolution of the saved figure in dots per inch.
DPI = 150


def read(path):
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        return bool(d["hit"]), d["calls_at_hit"]
    d = dict(line.split(": ", 1) for line in path.read_text().splitlines() if ": " in line)
    hit = d["hit"] == "true"
    return hit, int(d["hit_calls"]) if hit else None


def runs(directory, seeds):
    out = {}
    for s in seeds:
        for suffix in (".out", ".json"):
            p = Path(directory) / f"seed{s}{suffix}"
            if p.exists():
                out[s] = read(p)
    return out


def signflip(diffs, draws=SIGNFLIP_DRAWS, seed=0):
    observed = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        total = sum(v if rng.randrange(2) == 0 else -v for v in diffs)
        if abs(total / len(diffs)) >= abs(observed) - TIE_TOLERANCE:
            extreme += 1
    return (extreme + 1) / (draws + 1)


def main():
    lo, hi, budget, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    seeds = range(lo, hi + 1)
    series = [(label, runs(path, seeds)) for label, path in (a.split("=", 1) for a in sys.argv[5:])]
    par2 = [{s: c if h else 2 * budget for s, (h, c) in r.items()} for _, r in series]
    print("label hits median_calls_to_hit mean_par2 paired_p")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for k, (label, r) in enumerate(series):
        calls = sorted(c for h, c in r.values() if h)
        med = f"{statistics.median(calls):.1f}".removesuffix(".0") if calls else "-"
        mean = sum(par2[k].values()) / len(r)
        p = "-"
        if k:
            common = sorted(set(par2[0]) & set(par2[k]))
            p = f"{signflip([par2[0][s] - par2[k][s] for s in common]):.4f}"
        print(f"{label} {len(calls)}/{len(r)} {med} {mean:.0f} {p}")
        xs = [0] + calls + [budget]
        ys = [i / len(r) for i in range(len(calls) + 1)] + [len(calls) / len(r)]
        ax.step(xs, ys, where="post", label=label)
    ax.set_xlabel("energy calls")
    ax.set_ylabel("fraction of seeds at the global minimum")
    ax.set_xlim(0, budget)
    ax.set_ylim(0, FRACTION_TOP)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)


if __name__ == "__main__":
    main()
