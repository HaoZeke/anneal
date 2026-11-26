import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_rows(path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            charged = row["charged_calls"] or row["charged"]
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "hit": int(row["hit"]),
                    "calls_to_hit": int(float(row["calls_to_hit"])),
                    "charged": int(float(charged)),
                }
            )
    return rows


def aa_identical(path):
    fields = ("hit", "calls_to_hit", "total_calls", "polish_calls", "best_eV", "best_Eh")
    grouped = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["seed"], []).append(row)
    same = len(grouped) == 4
    for group in grouped.values():
        if {row["engine"] for row in group} != {"A", "B"}:
            same = False
        for field in fields:
            if len({row[field] for row in group}) != 1:
                same = False
    return same


def main():
    rows = load_rows(HERE / "results.csv")
    charged = np.asarray([row["charged"] for row in rows], dtype=float)
    hits = sum(row["hit"] for row in rows)
    print(f"hits {hits}/{len(rows)}")
    print(f"charged_median {float(np.median(charged)):.1f}")
    for row in rows:
        print(
            f"seed {row['seed']} hit {row['hit']} calls_to_hit {row['calls_to_hit']} charged {row['charged']}"
        )
    print(f"aa_identical {int(aa_identical(HERE / 'aa-flag-off.csv'))}")


if __name__ == "__main__":
    main()
