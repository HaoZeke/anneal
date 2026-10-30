#!/usr/bin/env python3
"""Per-seed rows and a paired comparison of two GFN2 water board reports.

A hit is a best structure assigned to a deposited isomer inside the IRA
gate. Usage: ``python experiments/water_board.py BEFORE/board.txt AFTER/board.txt OUT.csv``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hit_stats import fisher, mcnemar, wilson  # noqa: E402

FIELDS = ("seed", "isomer", "isomer_rmsd_A", "isomer_reason", "force_calls", "best_energy")
# Counters only newer boards print; summed per board, not compared per seed.
COUNTERS = ("refused_evaluations", "soap_fallbacks")


def seeds(path: Path) -> list[dict[str, str]]:
    rows = []
    current: dict[str, str] | None = None
    for line in path.read_text().splitlines():
        key, _, value = line.partition(": ")
        if key == "seed":
            current = {"seed": value}
            rows.append(current)
        elif key == "search_error":
            # The driver stops the board at the first seed with no finite start.
            print(f"{path}: search_error after seed {rows[-1]['seed'] if rows else 'none'}: {value}")
        elif current is not None and key in FIELDS + COUNTERS:
            current[key] = value
    return rows


def main() -> None:
    before, after, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    left = {row["seed"]: row for row in seeds(before)}
    right = {row["seed"]: row for row in seeds(after)}
    print(f"seeds before {len(left)} after {len(right)}")
    keys = sorted(set(left) & set(right), key=int)
    for name, rows in (("before", left), ("after", right)):
        tally: dict[str, int] = {}
        for key in keys:
            row = rows[key]
            label = row["isomer"] if row["isomer_reason"] == "inside gate" else "miss"
            tally[label] = tally.get(label, 0) + 1
        print(f"{name} isomers " + " ".join(f"{k} {v}" for k, v in sorted(tally.items())))
    same = sum(all(left[key][f] == right[key][f] for f in FIELDS) for key in keys)
    print(f"seeds_identical {same}/{len(keys)}")
    for name, rows in (("before", left), ("after", right)):
        for counter in COUNTERS:
            values = [int(rows[key][counter]) for key in keys if counter in rows[key]]
            if values:
                touched = sum(v > 0 for v in values)
                print(f"{name} {counter} total {sum(values)} seeds_nonzero {touched}/{len(values)}")
    hits = {"before": 0, "after": 0}
    only = {"before": 0, "after": 0}
    with out.open("w") as handle:
        handle.write(
            "seed,before_isomer,before_hit,before_calls,before_energy,"
            "after_isomer,after_hit,after_calls,after_energy\n"
        )
        for key in keys:
            a, b = left[key], right[key]
            hit_a = a["isomer_reason"] == "inside gate"
            hit_b = b["isomer_reason"] == "inside gate"
            hits["before"] += hit_a
            hits["after"] += hit_b
            only["before"] += hit_a and not hit_b
            only["after"] += hit_b and not hit_a
            handle.write(
                f"{key},{a['isomer']},{int(hit_a)},{a['force_calls']},{a['best_energy']},"
                f"{b['isomer']},{int(hit_b)},{b['force_calls']},{b['best_energy']}\n"
            )
    n = len(keys)
    for name in ("before", "after"):
        lo, hi = wilson(hits[name], n)
        print(f"{name} hits {hits[name]}/{n} wilson95 {lo:.3f}-{hi:.3f}")
    print(f"only_before {only['before']} only_after {only['after']}")
    print(
        "fisher_two_sided "
        f"{fisher(hits['after'], n - hits['after'], hits['before'], n - hits['before']):.4g}"
    )
    print(f"mcnemar_exact {mcnemar(only['after'], only['before']):.4g}")
    lost = [key for key in keys if left[key]["isomer_reason"] == "inside gate"
            and right[key]["isomer_reason"] != "inside gate"]
    print(f"seeds_losing_a_hit {' '.join(lost) if lost else 'none'}")


if __name__ == "__main__":
    main()
