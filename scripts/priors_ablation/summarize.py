#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Priors on against priors off against GMIN.

Per system: hits out of six, the median calls to hit with a miss counted
as the full budget, and the median wall, inside (potential) and outside
seconds. Medians are Hyndman-Fan type 7. `p` is the two-sided Fisher exact
test of the on and off hit counts. GMIN is this session's run where
gmin-rows.csv has the system and the recorded panel otherwise. The
recorded priors-on row is the reference panel the earlier wins came from.
A recorded panel that stores only per-call times gives inside and outside
as those times the median charged calls, marked `~`; a blank is a panel
that did not time the potential.
"""

from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path

ORDER = (
    "lj38", "lj75", "lj98", "lj104", "lj150",
    "m55r6", "m80r6", "m55r14", "m80r14",
    "au55", "cu6au32", "cu27au28",
    "tip4p4", "tip4p8", "tip4p15",
)
GUPTA = "metal-cluster-h2h/gupta-binary-e2865427"
# system -> (file under bench/, label column value, commit that wrote it)
GMIN_RECORDED = {
    "lj75": ("lj-outside/results.csv", "gmin", "2d00f044"),
    "lj98": ("lj-outside/results.csv", "gmin", "2d00f044"),
    "lj104": ("lj-outside/results.csv", "gmin", "2d00f044"),
    "lj150": ("lj-large/results.csv", "gmin", "3d5cbdee"),
    "m55r6": ("morse-h2h/results.csv", "gmin", "770121ac"),
    "m55r14": ("morse-h2h/results.csv", "gmin", "770121ac"),
    "m80r14": ("morse-h2h/results.csv", "gmin", "770121ac"),
    "au55": (f"{GUPTA}/summary-v2.csv", "gmin", "3a8134db"),
    "cu6au32": (f"{GUPTA}/summary-v2.csv", "gmin", "3a8134db"),
    "cu27au28": (f"{GUPTA}/summary-v2.csv", "gmin", "3a8134db"),
    "tip4p4": ("tip4p-h2h/rows.csv", "gmin-control-same-start", "51523528"),
    "tip4p8": ("tip4p-h2h/rows.csv", "gmin-control-same-start", "51523528"),
    "tip4p15": ("tip4p-h2h/rows.csv", "gmin-control-same-start", "51523528"),
}
ON_RECORDED = {
    "lj38": ("growpal-h2h/results.csv", "anneal", "14c2cdbc"),
    "lj75": ("lj-outside/results.csv", "anneal", "2d00f044"),
    "lj98": ("lj-outside/results.csv", "anneal", "2d00f044"),
    "lj104": ("lj-outside/results.csv", "anneal", "2d00f044"),
    "lj150": ("lj-large/results.csv", "anneal", "3d5cbdee"),
    "m55r6": ("morse-h2h/seeds-rho6-cdb82168.csv", "anneal", "9ff2850e"),
    "m55r14": ("morse-h2h/seeds-cdb82168.csv", "anneal", "c1fe8188"),
    "m80r14": ("morse-h2h/seeds-cdb82168.csv", "anneal", "c1fe8188"),
    "au55": (f"{GUPTA}/summary-v2.csv", "anneal", "3a8134db"),
    "cu6au32": (f"{GUPTA}/spine-cu6-0b788212.csv", "spine", "0b788212"),
    "cu27au28": (f"{GUPTA}/mackay-2bc5333d.csv", "mackay", "0b788212"),
    "tip4p4": ("tip4p-h2h/rows.csv", "anneal-bf8c3fe1", "51523528"),
    "tip4p8": ("tip4p-h2h/rows.csv", "anneal-bf8c3fe1", "51523528"),
    "tip4p15": ("tip4p-h2h/rows.csv", "anneal-79c13d53", "51523528"),
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


def fisher(a: int, n1: int, b: int, n2: int) -> float:
    """Two-sided Fisher exact p for a/n1 against b/n2 hits."""
    k = a + b
    total = comb(n1 + n2, k)

    def prob(x: int) -> float:
        return comb(n1, x) * comb(n2, k - x) / total

    observed = prob(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def read(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def is_hit(value: str) -> bool:
    return value.strip().lower() in ("1", "true")


def label_of(row: dict[str, str]) -> str:
    return row.get("arm") or row.get("code") or ""


def seed_cells(rows: list[dict[str, str]]) -> dict:
    hits = [is_hit(r["hit"]) for r in rows]
    evals = [float(r["evals_to_hit"]) if h else float(r["budget"]) for r, h in zip(rows, hits)]
    walls = [float(r["wall_s"]) for r in rows]
    inside, outside, approx = [], [], False
    for r, wall in zip(rows, walls):
        if r.get("pot_s"):
            pot = float(r["pot_s"])
            inside.append(pot)
            outside.append(float(r["outside_s"]) if r.get("outside_s") else max(wall - pot, 0.0))
        elif r.get("pot_per_eval_s"):
            charged = float(r.get("charged") or r["evals_to_hit"])
            inside.append(float(r["pot_per_eval_s"]) * charged)
            outside.append(float(r["outside_per_eval_s"]) * charged)
            approx = True
    return {
        "hits": sum(hits),
        "n": len(rows),
        "evals": type7(evals),
        "wall": type7(walls),
        "inside": type7(inside) if inside else None,
        "outside": type7(outside) if outside else None,
        "approx": approx,
    }


def recorded(bench: Path, table: dict, system: str) -> dict | None:
    if system not in table:
        return None
    name, label, commit = table[system]
    rows = [r for r in read(bench / name) if r["system"] == system and label_of(r) == label]
    if not rows:
        raise SystemExit(f"no {label} row for {system} in {name}")
    if "hits_over_n" in rows[0]:
        (r,) = rows
        evals = float(r["evals_median"])
        if r.get("pot_median_s"):
            inside, outside, approx = float(r["pot_median_s"]), float(r["outside_median_s"]), False
        elif r.get("pot_per_eval_s"):
            inside = float(r["pot_per_eval_s"]) * evals
            outside = float(r["outside_per_eval_s"]) * evals
            approx = True
        else:
            inside = outside = None
            approx = False
        hits, n = (int(v) for v in r["hits_over_n"].split("/"))
        out = {
            "hits": hits,
            "n": n,
            "evals": evals,
            "wall": float(r["wall_median_s"]),
            "inside": inside,
            "outside": outside,
            "approx": approx,
        }
    else:
        out = seed_cells(rows)
    out["source"] = f"bench/{name} {label} @ {commit}"
    return out


def fmt_time(value, approx: bool) -> str:
    if value is None:
        return "–"
    return f"{'~' if approx else ''}{value:.3f}"


def cell(s: dict | None) -> str:
    if s is None:
        return "no row"
    return (
        f"{s['hits']}/{s['n']} · {s['evals']:.0f} · {s['wall']:.3f} s · "
        f"{fmt_time(s['inside'], s['approx'])} / {fmt_time(s['outside'], s['approx'])}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--gmin-rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = read(args.rows)
    for r in rows:
        r["hit"] = r["hit_binary"]
    gmin_rows = read(args.gmin_rows) if args.gmin_rows.is_file() else []
    lines = [
        "| system | GMIN | anneal priors on | anneal priors off | p (on vs off) | recorded priors on |",
        "|---|---|---|---|---|---|",
    ]
    out_rows = []
    for system in ORDER:
        mine = [r for r in rows if r["system"] == system]
        on_rows = [r for r in mine if r["arm"] == "default"]
        off_rows = [r for r in mine if r["arm"] == "priors_off"]
        on = seed_cells(on_rows) if on_rows else None
        off = seed_cells(off_rows) if off_rows else None
        session_gmin = [r for r in gmin_rows if r["system"] == system]
        if session_gmin:
            gmin = seed_cells(session_gmin)
            gmin["source"] = "bench/priors-ablation/gmin-rows.csv"
        else:
            gmin = recorded(args.bench, GMIN_RECORDED, system)
        ref = recorded(args.bench, ON_RECORDED, system)
        p = fisher(on["hits"], on["n"], off["hits"], off["n"]) if on and off else None
        ref_cell = f"{ref['hits']}/{ref['n']} · {ref['evals']:.0f}" if ref else "no row"
        lines.append(
            f"| {system} | {cell(gmin)} | {cell(on)} | {cell(off)} | "
            f"{'–' if p is None else f'{p:.3f}'} | {ref_cell} |"
        )
        for arm, s in (("gmin", gmin), ("priors_on", on), ("priors_off", off), ("recorded_priors_on", ref)):
            if s is None:
                continue
            out_rows.append(
                {
                    "system": system,
                    "arm": arm,
                    "hits": s["hits"],
                    "n": s["n"],
                    "evals_median": f"{s['evals']:.1f}",
                    "wall_median_s": f"{s['wall']:.6f}",
                    "inside_median_s": "" if s["inside"] is None else f"{s['inside']:.6f}",
                    "outside_median_s": "" if s["outside"] is None else f"{s['outside']:.6f}",
                    "times_from_per_call": "yes" if s["approx"] else "no",
                    "fisher_p_on_off": "" if p is None or arm not in ("priors_on", "priors_off") else f"{p:.6f}",
                    "source": s.get("source", "bench/priors-ablation/rows.csv"),
                }
            )
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    (args.out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
