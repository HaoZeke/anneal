#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summaries of the three anneal arms beside GMIN.

Per system and arm: hits out of six, the median calls to hit with a miss
counted as the full budget, and the median wall, inside (potential) and
outside seconds. Medians are Hyndman-Fan type 7. Pt-Cu and Cu38 hits are
within 0.001 of the lower of the recorded session reference and the best
energy this session reached on any arm. GMIN is this session's run where
gmin-rows.csv has the system and the recorded panel otherwise. A recorded
panel that stores only per-call times gives inside and outside as those
times the median charged calls, marked `~`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

HIT_TOL = 0.001
ARMS = ("default", "priors_off", "general_only")
ORDER = (
    "lj38", "lj75", "lj98", "lj104", "lj150",
    "m38r6", "m55r6", "m80r6", "m38r14", "m55r14", "m80r14",
    "sc126-38", "sc126-55",
    "au38", "au55", "cu38", "cu6au32", "cu27au28", "pt12cu11", "pt6cu17",
    "tip4p4", "tip4p8",
)
PTCU = "metal-cluster-h2h/ptcu-b3d03061/results.csv"
SESSION_REFERENCE = {"cu38": PTCU, "pt12cu11": PTCU, "pt6cu17": PTCU}
# system -> (file under bench/, kind, commit that wrote it)
GMIN_RECORDED = {
    "lj75": ("lj-outside/results.csv", "seeds", "2d00f044"),
    "lj98": ("lj-outside/results.csv", "seeds", "2d00f044"),
    "lj104": ("lj-outside/results.csv", "seeds", "2d00f044"),
    "lj150": ("lj-large/results.csv", "seeds", "3d5cbdee"),
    "m38r6": ("morse-h2h/results.csv", "medians", "770121ac"),
    "m55r6": ("morse-h2h/results.csv", "medians", "770121ac"),
    "m38r14": ("morse-h2h/results.csv", "medians", "770121ac"),
    "m55r14": ("morse-h2h/results.csv", "medians", "770121ac"),
    "m80r14": ("morse-h2h/results.csv", "medians", "770121ac"),
    "sc126-38": ("metal-cluster-h2h/sc-8e076b7f/results.csv", "per_eval", "290d5202"),
    "sc126-55": ("metal-cluster-h2h/sc-8e076b7f/results.csv", "per_eval", "290d5202"),
    "au38": ("metal-cluster-h2h/gupta-binary-e2865427/summary-v2.csv", "per_eval", "3a8134db"),
    "au55": ("metal-cluster-h2h/gupta-binary-e2865427/summary-v2.csv", "per_eval", "3a8134db"),
    "cu6au32": ("metal-cluster-h2h/gupta-binary-e2865427/summary-v2.csv", "per_eval", "3a8134db"),
    "cu27au28": ("metal-cluster-h2h/gupta-binary-e2865427/summary-v2.csv", "per_eval", "3a8134db"),
    "cu38": (PTCU, "per_eval", "453f5b7e"),
    "pt12cu11": (PTCU, "per_eval", "453f5b7e"),
    "pt6cu17": (PTCU, "per_eval", "453f5b7e"),
}


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def read(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def summarise(rows: list[dict[str, str]], hit_of) -> dict[str, str]:
    hits = [hit_of(r) for r in rows]
    evals = [float(r["evals_to_hit"]) if h else float(r["budget"]) for r, h in zip(rows, hits)]
    return {
        "hits": f"{sum(hits)}/{len(rows)}",
        "evals": type7(evals),
        "wall": type7([float(r["wall_s"]) for r in rows]),
        "inside": type7([float(r["pot_s"]) for r in rows]),
        "outside": type7([float(r["outside_s"]) for r in rows]),
        "approx": False,
    }


def recorded_gmin(bench: Path, system: str) -> dict[str, str] | None:
    if system not in GMIN_RECORDED:
        return None
    name, kind, commit = GMIN_RECORDED[system]
    rows = [r for r in read(bench / name) if r["system"] == system and r["arm"] == "gmin"]
    if kind == "seeds":
        for r in rows:
            wall, pot = float(r["wall_s"]), float(r["pot_s"])
            r["outside_s"] = f"{max(wall - pot, 0.0):.6f}"
        out = summarise(rows, lambda r: r["hit"] in ("1", "true"))
    else:
        (r,) = rows
        evals = float(r["evals_median"])
        if kind == "medians":
            inside, outside, approx = float(r["pot_median_s"]), float(r["outside_median_s"]), False
        else:
            inside = float(r["pot_per_eval_s"]) * evals
            outside = float(r["outside_per_eval_s"]) * evals
            approx = True
        out = {
            "hits": r["hits_over_n"],
            "evals": evals,
            "wall": float(r["wall_median_s"]),
            "inside": inside,
            "outside": outside,
            "approx": approx,
        }
    out["source"] = f"bench/{name} @ {commit}"
    return out


def cell(s: dict | None) -> str:
    if s is None:
        return "no row"
    mark = "~" if s.get("approx") else ""
    return (
        f"{s['hits']} · {s['evals']:.0f} · {s['wall']:.3f} s · "
        f"{mark}{s['inside']:.3f} / {mark}{s['outside']:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, required=True)
    args = parser.parse_args()
    here = args.bench / "priors-ablation"
    rows = read(here / "rows.csv")
    gmin_rows = read(here / "gmin-rows.csv") if (here / "gmin-rows.csv").is_file() else []
    out_rows = []
    lines = [
        "| system | GMIN | anneal default | anneal priors off | anneal general_only |",
        "|---|---|---|---|---|",
    ]
    for system in ORDER:
        mine = [r for r in rows if r["system"] == system]
        if not mine:
            continue
        reference = None
        if system in SESSION_REFERENCE:
            recorded = [r for r in read(args.bench / SESSION_REFERENCE[system]) if r["system"] == system]
            reference = min([float(recorded[0]["reference_eV"])] + [float(r["best"]) for r in mine])

        def hit_of(r, reference=reference):
            if reference is not None:
                return float(r["best"]) <= reference + HIT_TOL
            return r["hit_binary"] == "true"

        cells = {}
        for arm in ARMS:
            arm_rows = [r for r in mine if r["arm"] == arm]
            cells[arm] = summarise(arm_rows, hit_of) if arm_rows else None
        session_gmin = [r for r in gmin_rows if r["system"] == system]
        if session_gmin:
            gmin = summarise(session_gmin, lambda r: r["hit"] == "true")
            gmin["source"] = "bench/priors-ablation/gmin-rows.csv"
        else:
            gmin = recorded_gmin(args.bench, system)
        lines.append(
            f"| {system} | {cell(gmin)} | {cell(cells['default'])} | "
            f"{cell(cells['priors_off'])} | {cell(cells['general_only'])} |"
        )
        for arm, s in [("gmin", gmin), *cells.items()]:
            if s is None:
                continue
            out_rows.append(
                {
                    "system": system,
                    "arm": arm,
                    "hits": s["hits"],
                    "evals_median": f"{s['evals']:.1f}",
                    "wall_median_s": f"{s['wall']:.6f}",
                    "inside_median_s": f"{s['inside']:.6f}",
                    "outside_median_s": f"{s['outside']:.6f}",
                    "times_from_per_call": "yes" if s.get("approx") else "no",
                    "reference": "" if reference is None else f"{reference:.8f}",
                    "source": s.get("source", "bench/priors-ablation/rows.csv"),
                }
            )
    with (here / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    (here / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
