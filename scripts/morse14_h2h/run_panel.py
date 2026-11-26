#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Morse rho = 14, M55 and M80, twelve seeds, paper budgets.

Anneal runs serially. A GMIN row already stored at the same budget is
copied and not launched again. Hits are a quenched energy within 0.001
of the Cambridge table. Medians are Hyndman-Fan type 7. Fisher is the
two-sided exact test on hits. Mann-Whitney is two-sided on the call
counts and on the wall, misses stored as the budget.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

for _candidate in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (_candidate / "target_tol.py").is_file() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break
from target_tol import DEFAULT_HIT_TOL  # noqa: E402
from machine_provenance import (  # noqa: E402
    aggregate_machine_columns,
    host_label,
    machine_columns,
    machine_fieldnames,
    read_built_gmin_provenance,
    require_csv_schema,
    require_gmin_build_key,
    validate_saved_gmin_provenance,
)

SYSTEMS = {
    "m55r14": {"n": 55, "rho": 14.0, "budget": 1_000_000, "reference": -220.646208},
    "m80r14": {"n": 80, "rho": 14.0, "budget": 4_000_000, "reference": -340.811371},
}
SEEDS = list(range(2000, 2012))
HIT_TOL = DEFAULT_HIT_TOL
SEED_FIELDS = [
    "system",
    "code",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "best",
    "pot_s",
    "outside_s",
]
ROW_FIELDS = machine_fieldnames(
    SEED_FIELDS + ["code_revision", "gmin_build_key", "host"]
)
SUMMARY_BASE_FIELDS = [
    "system",
    "anneal_hits",
    "anneal_n",
    "gmin_hits",
    "gmin_n",
    "gmin_reused",
    "fisher_p",
    "mannwhitney_p_evals",
    "mannwhitney_p_wall",
    "anneal_evals_median",
    "gmin_evals_median",
    "anneal_wall_median_s",
    "gmin_wall_median_s",
    "anneal_pot_median_s",
    "anneal_outside_median_s",
    "gmin_pot_median_s",
    "gmin_outside_median_s",
]
SUMMARY_FIELDS = machine_fieldnames(SUMMARY_BASE_FIELDS)


def type7(ordered: list[float], p: float) -> float:
    n = len(ordered)
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - frac) + ordered[index + 1] * frac


def median(values: list[float]) -> float:
    return type7(sorted(values), 0.5)


def comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def fisher_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    a, b = hits_a, n_a - hits_a
    c, d = hits_b, n_b - hits_b

    def prob(aa: int, bb: int, cc: int, dd: int) -> float:
        total = aa + bb + cc + dd
        return comb(aa + bb, aa) * comb(cc + dd, cc) / comb(total, aa + cc)

    base = prob(a, b, c, d)
    total = 0.0
    for aa in range(a + b + 1):
        cc = a + c - aa
        bb = a + b - aa
        dd = c + d - cc
        if min(aa, bb, cc, dd) < 0:
            continue
        p = prob(aa, bb, cc, dd)
        if p <= base + 1e-12:
            total += p
    return min(1.0, total)


def mann_whitney(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return float("nan")
    try:
        from scipy.stats import mannwhitneyu
    except ModuleNotFoundError:
        mannwhitneyu = None
    if mannwhitneyu is not None:
        return float(
            mannwhitneyu(
                left,
                right,
                alternative="two-sided",
                method="asymptotic",
                use_continuity=True,
            ).pvalue
        )
    n1, n2 = len(left), len(right)
    pooled = sorted(left + right)
    n = n1 + n2
    rank: dict[float, float] = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    u = sum(rank[v] for v in left) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return 1.0
    dev = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    return min(1.0, math.erfc(dev / math.sqrt(var) / math.sqrt(2.0)))


def binary_path(root: Path) -> Path:
    candidates = [
        root.parent / "tmp" / "anneal-target" / "release" / "examples" / "h2h_morse",
        root / "target" / "release" / "examples" / "h2h_morse",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit("h2h_morse release binary is not built")


def _resolve_gmin_provenance(
    svnrev: Path | None = None, gmin_source: Path | None = None
) -> tuple[str, str]:
    svnrev_path = svnrev or Path(
        os.environ.get(
            "GMIN_SVNREV_FILE", "/work/gmin-h2h/src/wales/GMIN/source/SVNREV"
        )
    )
    source_path = gmin_source or Path(
        os.environ.get("GMIN_SOURCE_FILE", "/work/gmin-h2h/gmin-source.txt")
    )
    return read_built_gmin_provenance(svnrev_path, source_path)


def _validate_caller_gmin_provenance(
    gmin_revision: str | None, gmin_build_key: str | None
) -> tuple[str, str]:
    if not isinstance(gmin_revision, str) or not gmin_revision.strip():
        raise ValueError("missing expected GMIN revision")
    return (
        gmin_revision.strip(),
        require_gmin_build_key(gmin_build_key, "expected gmin_build_key"),
    )


def load_gmin(
    archive: Path,
    *,
    gmin_revision: str | None = None,
    gmin_build_key: str | None = None,
) -> dict[tuple[str, int], dict[str, str]]:
    if gmin_revision is None and gmin_build_key is None:
        gmin_revision, gmin_build_key = _resolve_gmin_provenance()
    else:
        gmin_revision, gmin_build_key = _validate_caller_gmin_provenance(
            gmin_revision, gmin_build_key
        )
    rows = {}
    with archive.open() as handle:
        for row in csv.DictReader(handle):
            if row["code"] != "gmin":
                continue
            system = row["system"]
            if system not in SYSTEMS:
                continue
            if int(row["budget"]) != SYSTEMS[system]["budget"]:
                continue
            validate_saved_gmin_provenance(
                row,
                path=archive,
                revision=gmin_revision,
                build_key=gmin_build_key,
            )
            rows[(system, int(row["seed"]))] = row
    return rows


def correct_evals(hit: bool, evals: int, charged: int, budget: int) -> int:
    if not hit:
        return budget
    if evals < budget:
        return evals
    return min(charged, budget)


def parse_anneal(text: str, system: str) -> dict[str, str]:
    rows = [line for line in text.splitlines() if line.startswith(system + ",")]
    if not rows:
        raise SystemExit(f"no anneal row\n{text[-600:]}")
    cols = rows[-1].split(",")
    budget = int(cols[3])
    hit = cols[4] == "true"
    evals = correct_evals(hit, int(cols[5]), int(cols[6]), budget)
    return {
        "system": system,
        "code": "anneal",
        "seed": cols[2],
        "budget": cols[3],
        "hit": "1" if hit else "0",
        "evals_to_hit": str(evals),
        "charged": cols[6],
        "wall_s": cols[7],
        "best": cols[8],
        "pot_s": cols[9],
        "outside_s": cols[10],
    }


def run_anneal(binary: Path, system: str, seed: int) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    proc = subprocess.run(
        ["taskset", "-c", "0", str(binary), "run", system, str(seed)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    text = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise SystemExit(text[-800:])
    return parse_anneal(text, system)


def gmin_row(stored: dict[str, str]) -> dict[str, str]:
    return {field: stored.get(field, "") for field in ROW_FIELDS}


def write_rows(
    path: Path,
    rows: list[dict[str, str]],
    *,
    overwrite_legacy: bool = False,
) -> None:
    if not overwrite_legacy:
        require_csv_schema(path, ROW_FIELDS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarise(
    rows: list[dict[str, str]],
    path: Path,
    *,
    overwrite_legacy: bool = False,
) -> None:
    lines = []
    for system in SYSTEMS:
        anneal = [row for row in rows if row["system"] == system and row["code"] == "anneal"]
        gmin = [row for row in rows if row["system"] == system and row["code"] == "gmin"]
        shared = sorted(set(int(row["seed"]) for row in anneal) & set(int(row["seed"]) for row in gmin))
        anneal_shared = [row for row in anneal if int(row["seed"]) in shared]
        gmin_shared = [row for row in gmin if int(row["seed"]) in shared]

        def hits(group: list[dict[str, str]]) -> int:
            return sum(row["hit"] == "1" for row in group)

        def column(group: list[dict[str, str]], name: str) -> list[float]:
            return [float(row[name]) for row in group]

        both = anneal + gmin
        lines.append(
            {
                "system": system,
                "anneal_hits": str(hits(anneal)),
                "anneal_n": str(len(anneal)),
                "gmin_hits": str(hits(gmin_shared)),
                "gmin_n": str(len(gmin_shared)),
                "gmin_reused": " ".join(str(seed) for seed in shared),
                "fisher_p": f"{fisher_two_sided(hits(anneal_shared), len(anneal_shared), hits(gmin_shared), len(gmin_shared)):.6g}" if shared else "",
                "mannwhitney_p_evals": f"{mann_whitney(column(anneal_shared, 'evals_to_hit'), column(gmin_shared, 'evals_to_hit')):.6g}" if shared else "",
                "mannwhitney_p_wall": f"{mann_whitney(column(anneal_shared, 'wall_s'), column(gmin_shared, 'wall_s')):.6g}" if shared else "",
                "anneal_evals_median": f"{median(column(anneal, 'evals_to_hit')):.6f}",
                "gmin_evals_median": f"{median(column(gmin, 'evals_to_hit')):.6f}" if gmin else "",
                "anneal_wall_median_s": f"{median(column(anneal, 'wall_s')):.6f}",
                "gmin_wall_median_s": f"{median(column(gmin, 'wall_s')):.6f}" if gmin else "",
                "anneal_pot_median_s": f"{median(column(anneal, 'pot_s')):.6f}",
                "anneal_outside_median_s": f"{median(column(anneal, 'outside_s')):.6f}",
                "gmin_pot_median_s": f"{median(column(gmin, 'pot_s')):.6f}" if gmin else "",
                "gmin_outside_median_s": f"{median(column(gmin, 'outside_s')):.6f}" if gmin else "",
                **aggregate_machine_columns(both),
            }
        )
    fields = (
        SUMMARY_FIELDS
        if lines and "machine_id" in lines[0]
        else SUMMARY_BASE_FIELDS
    )
    if not overwrite_legacy:
        require_csv_schema(path, fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(lines)


def panel(
    root: Path,
    *,
    svnrev: Path | None = None,
    gmin_source: Path | None = None,
) -> list[dict[str, str]]:
    gmin_revision, gmin_build_key = _resolve_gmin_provenance(svnrev, gmin_source)
    archive = load_gmin(
        root / "bench" / "morse-h2h" / "seeds.csv",
        gmin_revision=gmin_revision,
        gmin_build_key=gmin_build_key,
    )
    binary = binary_path(root)
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    host = host_label(os.uname().nodename)
    machine = machine_columns()
    rows: list[dict[str, str]] = []
    for system in SYSTEMS:
        for seed in SEEDS:
            anneal = run_anneal(binary, system, seed)
            anneal.update(
                {
                    "code_revision": head,
                    "gmin_build_key": "",
                    "host": host,
                    **machine,
                }
            )
            print(",".join(anneal[field] for field in SEED_FIELDS), flush=True)
            rows.append(anneal)
            stored = archive.get((system, seed))
            if stored is not None:
                copied = gmin_row(stored)
                print("reused " + ",".join(copied[field] for field in SEED_FIELDS), flush=True)
                rows.append(copied)
    return rows


def one(
    root: Path,
    system: str,
    seed: int,
    arm: str,
    *,
    svnrev: Path | None = None,
    gmin_source: Path | None = None,
) -> None:
    if arm == "gmin":
        gmin_revision, gmin_build_key = _resolve_gmin_provenance(svnrev, gmin_source)
        stored = load_gmin(
            root / "bench" / "morse-h2h" / "seeds.csv",
            gmin_revision=gmin_revision,
            gmin_build_key=gmin_build_key,
        ).get((system, seed))
        if stored is None:
            raise SystemExit(
                f"no stored GMIN row for {system} seed {seed} at budget {SYSTEMS[system]['budget']}"
            )
        print(",".join(gmin_row(stored)[field] for field in SEED_FIELDS))
        return
    if arm != "anneal":
        raise SystemExit(f"unknown arm {arm}")
    print(",".join(run_anneal(binary_path(root), system, seed)[field] for field in SEED_FIELDS))


def _parse_argv(argv: Sequence[str] | None) -> list[str] | None:
    if argv is None:
        return None
    items = list(argv)
    if items and not items[0].startswith("-"):
        return items[1:]
    return items


def run_panel(
    root: Path,
    *,
    out_dir: Path | None = None,
    svnrev: Path | None = None,
    gmin_source: Path | None = None,
    overwrite_legacy: bool = False,
) -> tuple[Path, Path]:
    root = root.resolve()
    out = (out_dir or (root / "bench" / "morse14-h2h")).resolve()
    seeds_path = out / "seeds.csv"
    summary_path = out / "summary.csv"
    if not overwrite_legacy:
        require_csv_schema(seeds_path, ROW_FIELDS)
        require_csv_schema(summary_path, SUMMARY_FIELDS)
    rows = panel(root, svnrev=svnrev, gmin_source=gmin_source)
    write_rows(seeds_path, rows, overwrite_legacy=overwrite_legacy)
    summarise(rows, summary_path, overwrite_legacy=overwrite_legacy)
    return seeds_path, summary_path


def main(
    argv: Sequence[str] | None = None,
    *,
    out_dir: Path | None = None,
    overwrite_legacy: bool = False,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite-legacy", action="store_true")
    parser.add_argument("--system")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm")
    parser.add_argument("--svnrev", type=Path)
    parser.add_argument("--gmin-source", type=Path)
    args = parser.parse_args(_parse_argv(argv))
    root = args.root.resolve()
    if args.system or args.seed or args.arm:
        if not (args.system and args.seed is not None and args.arm):
            raise SystemExit("one job needs --system, --seed and --arm")
        one(
            root,
            args.system,
            args.seed,
            args.arm,
            svnrev=args.svnrev,
            gmin_source=args.gmin_source,
        )
        return 0
    run_panel(
        root,
        out_dir=out_dir if out_dir is not None else args.out,
        svnrev=args.svnrev,
        gmin_source=args.gmin_source,
        overwrite_legacy=bool(overwrite_legacy or args.overwrite_legacy),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
