#!/usr/bin/env python3
"""Map sota_cutest long CSV rows to the schema expected by render_plots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sota_csv")
    p.add_argument("out_csv")
    args = p.parse_args()
    src = Path(args.sota_csv)
    dst = Path(args.out_csv)
    with src.open() as f, dst.open("w", newline="") as out:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "problem",
                "kind",
                "dim",
                "driver",
                "seed",
                "fevals",
                "best_val",
                "wall_time_s",
                "f_x0",
                "solved",
                "status",
            ],
        )
        writer.writeheader()
        for row in reader:
            status = (row.get("status") or "ok").strip()
            ok = status == "ok"
            best = row.get("best", "")
            try:
                bf = float(best)
                finite = bf == bf and abs(bf) != float("inf")
            except (TypeError, ValueError):
                finite = False
            writer.writerow(
                {
                    "problem": row["problem"],
                    "kind": "cutest",
                    "dim": row["dim"],
                    "driver": row["method"],
                    "seed": row["seed"],
                    "fevals": row.get("evals") or row.get("objective_evals") or "",
                    "best_val": best,
                    "wall_time_s": "",
                    "f_x0": row.get("initial", ""),
                    "solved": "1" if ok and finite else "0",
                    "status": status,
                }
            )
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
