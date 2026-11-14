#!/usr/bin/env python3
# Copyright (c) anneal developers
"""M70 and M80 at rho 3, 6, 10 and 14, plus M55 at rho 6 and 14.

Twelve seeds, paper budgets, one core. GMIN rows already stored at the
same seed and budget are copied. Anneal is this tree's recommended preset.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import socket
import subprocess
import sys
from pathlib import Path

SYSTEMS = (
    ("m55r6", 55, 6.0, 1_000_000, -250.286609),
    ("m55r14", 55, 14.0, 1_000_000, -220.646208),
    ("m70r3", 70, 3.0, 1_000_000, -577.739782),
    ("m70r6", 70, 6.0, 1_000_000, -325.887749),
    ("m70r10", 70, 10.0, 1_000_000, -296.412149),
    ("m70r14", 70, 14.0, 1_000_000, -292.462856),
    ("m80r3", 80, 3.0, 4_000_000, -690.577890),
    ("m80r6", 80, 6.0, 4_000_000, -378.333471),
    ("m80r10", 80, 10.0, 4_000_000, -344.911381),
    ("m80r14", 80, 14.0, 4_000_000, -340.811371),
)
SEEDS = tuple(range(2000, 2012))
FIELDS = [
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
SUMMARY_FIELDS = [
    "system",
    "anneal_hits",
    "anneal_n",
    "gmin_hits",
    "gmin_n",
    "fisher_p",
    "mannwhitney_p_evals",
    "mannwhitney_p_wall",
    "mannwhitney_p_pot",
    "mannwhitney_p_outside",
    "anneal_evals_median",
    "gmin_evals_median",
    "anneal_wall_median_s",
    "gmin_wall_median_s",
    "anneal_pot_median_s",
    "gmin_pot_median_s",
    "anneal_outside_median_s",
    "gmin_outside_median_s",
    "gmin_reused",
]


def type7(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * 0.5
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - frac) + ordered[index + 1] * frac


def fisher(h1: int, n1: int, h2: int, n2: int) -> float:
    from scipy.stats import fisher_exact

    table = [[h1, n1 - h1], [h2, n2 - h2]]
    return float(fisher_exact(table, alternative="two-sided").pvalue)


def mannwhitney(left: list[float], right: list[float]) -> float:
    from scipy.stats import mannwhitneyu

    if not left or not right:
        return float("nan")
    return float(mannwhitneyu(left, right, alternative="two-sided").pvalue)


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def have(rows: list[dict[str, str]], system: str, code: str, seed: int) -> bool:
    return any(
        row["system"] == system and row["code"] == code and int(row["seed"]) == seed
        for row in rows
    )


def canonical(row: dict[str, str]) -> dict[str, str]:
    out = {key: row[key] for key in FIELDS}
    out["hit"] = "1" if str(row["hit"]) in ("1", "true") else "0"
    return out


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        (canonical(row) for row in rows),
        key=lambda row: (row["system"], row["code"], int(row["seed"])),
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ordered)


def reuse_gmin(root: Path) -> list[dict[str, str]]:
    wanted = {name: budget for name, _n, _rho, budget, _ref in SYSTEMS}
    found = []
    for path in (
        root / "bench/morse14-h2h/seeds.csv",
        root / "bench/morse-h2h/seeds.csv",
    ):
        for row in load_rows(path):
            if row.get("code") != "gmin":
                continue
            budget = wanted.get(row["system"])
            if budget is None or int(row["budget"]) != budget:
                continue
            if int(row["seed"]) not in SEEDS:
                continue
            found.append({key: row[key] for key in FIELDS if key in row})
    # First file wins on a duplicate.
    kept = []
    seen = set()
    for row in found:
        key = (row["system"], int(row["seed"]))
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def anneal_line(text: str) -> dict[str, str]:
    data = [line for line in text.splitlines() if ",anneal," in line]
    if len(data) != 1:
        raise RuntimeError(f"no anneal row in {text[-300:]!r}")
    header = ",".join(FIELDS)
    return next(csv.DictReader([header, data[0]]))


def gmin_line(text: str) -> dict[str, str]:
    fields = {}
    for part in text.split()[1:]:
        key, value = part.split("=", 1)
        fields[key] = value
    wall = float(fields["wall_s"])
    pot = float(fields.get("pot_s") or 0.0)
    return {
        "system": fields["system"],
        "code": "gmin",
        "seed": fields["seed"],
        "budget": fields["budget"],
        "hit": "1" if fields["hit"] in ("1", "true") else "0",
        "evals_to_hit": fields["evals_to_hit"],
        "charged": fields.get("charged") or fields["evals_to_hit"],
        "wall_s": f"{wall:.6f}",
        "best": fields["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{max(wall - pot, 0.0):.6f}",
    }


def run_env(budget: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "H2H_BUDGET": str(budget),
            "OMP_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
    )
    return env


def run_anneal(binary: Path, name: str, seed: int, budget: int) -> dict[str, str]:
    proc = subprocess.run(
        ["taskset", "-c", "0", str(binary), "run", name, str(seed)],
        check=False,
        capture_output=True,
        text=True,
        env=run_env(budget),
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-400:] or proc.stdout[-400:])
    return anneal_line(proc.stdout)


def run_gmin(
    binary: Path,
    gmin: Path,
    data: str,
    name: str,
    rho: float,
    seed: int,
    budget: int,
    target: float,
    work: Path,
) -> dict[str, str]:
    work.mkdir(parents=True, exist_ok=True)
    coords = work / "coords"
    subprocess.run(
        [str(binary), "coords", name, str(seed), str(coords)],
        check=True,
        env=run_env(budget),
    )
    (work / "data").write_text(data + f"MORSE {rho:.1f}\nRANSEED {seed}\n")
    env = run_env(budget)
    env.update(
        {
            "H2H_SYSTEM": name,
            "H2H_SEED": str(seed),
            "H2H_TARGET": f"{target:.10f}",
            "H2H_TOL": "0.001",
            "H2H_ROW": str(work / "row.txt"),
        }
    )
    proc = subprocess.run(
        ["taskset", "-c", "0", str(gmin)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=work,
    )
    row_path = work / "row.txt"
    if not row_path.is_file():
        raise RuntimeError(
            f"gmin {name} {seed} exited {proc.returncode}\n{(proc.stdout or '')[-400:]}"
        )
    return gmin_line(row_path.read_text())


def summarise(rows: list[dict[str, str]], reused: set[tuple[str, int]]) -> list[dict[str, str]]:
    out = []
    for name, _n, _rho, _budget, _ref in SYSTEMS:
        anneal = [row for row in rows if row["system"] == name and row["code"] == "anneal"]
        gmin = [row for row in rows if row["system"] == name and row["code"] == "gmin"]
        if len(anneal) != 12 or len(gmin) != 12:
            continue

        def col(group, key):
            return [float(row[key]) for row in group]

        ah = sum(row["hit"] in ("1", "true") for row in anneal)
        gh = sum(row["hit"] in ("1", "true") for row in gmin)
        reused_seeds = " ".join(str(seed) for seed in SEEDS if (name, seed) in reused)
        out.append(
            {
                "system": name,
                "anneal_hits": ah,
                "anneal_n": 12,
                "gmin_hits": gh,
                "gmin_n": 12,
                "fisher_p": f"{fisher(ah, 12, gh, 12):.6g}",
                "mannwhitney_p_evals": f"{mannwhitney(col(anneal, 'evals_to_hit'), col(gmin, 'evals_to_hit')):.6g}",
                "mannwhitney_p_wall": f"{mannwhitney(col(anneal, 'wall_s'), col(gmin, 'wall_s')):.6g}",
                "mannwhitney_p_pot": f"{mannwhitney(col(anneal, 'pot_s'), col(gmin, 'pot_s')):.6g}",
                "mannwhitney_p_outside": f"{mannwhitney(col(anneal, 'outside_s'), col(gmin, 'outside_s')):.6g}",
                "anneal_evals_median": f"{type7(col(anneal, 'evals_to_hit')):.3f}",
                "gmin_evals_median": f"{type7(col(gmin, 'evals_to_hit')):.3f}",
                "anneal_wall_median_s": f"{type7(col(anneal, 'wall_s')):.6f}",
                "gmin_wall_median_s": f"{type7(col(gmin, 'wall_s')):.6f}",
                "anneal_pot_median_s": f"{type7(col(anneal, 'pot_s')):.6f}",
                "gmin_pot_median_s": f"{type7(col(gmin, 'pot_s')):.6f}",
                "anneal_outside_median_s": f"{type7(col(anneal, 'outside_s')):.6f}",
                "gmin_outside_median_s": f"{type7(col(gmin, 'outside_s')):.6f}",
                "gmin_reused": reused_seeds,
            }
        )
    return out


def main() -> None:
    if socket.gethostname() != "bench":
        raise SystemExit("hostname must be bench")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("/work/morse-panel"))
    parser.add_argument("--gmin", type=Path, default=Path("/work/morse-h2h/bin/GMIN"))
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--arm", choices=("anneal", "gmin", "both"), default="both")
    args = parser.parse_args()
    systems = [row for row in SYSTEMS if not args.system or row[0] in args.system]
    seeds = tuple(args.seed) if args.seed else SEEDS
    root = args.root
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    seeds_path = work / "seeds.csv"
    rows = load_rows(seeds_path)
    reused_gmin = reuse_gmin(root)
    reused_keys = {(row["system"], int(row["seed"])) for row in reused_gmin}
    for row in reused_gmin:
        if not have(rows, row["system"], "gmin", int(row["seed"])):
            rows.append(row)
    write_rows(seeds_path, rows)

    toolchain = "/usr/local/rustup/toolchains/1.88.0-x86_64-unknown-linux-gnu/bin"
    os.environ["PATH"] = toolchain + os.pathsep + os.environ.get("PATH", "")
    os.environ["CARGO_HOME"] = os.environ.get("CARGO_HOME", "/usr/local/cargo")
    target = work / "target"
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    env["TMPDIR"] = str(work / "rustc-tmp")
    (work / "rustc-tmp").mkdir(parents=True, exist_ok=True)
    cargo = os.environ.get(
        "CARGO",
        "/usr/local/rustup/toolchains/1.88.0-x86_64-unknown-linux-gnu/bin/cargo",
    )
    subprocess.run(
        [
            cargo,
            "build",
            "--release",
            "--locked",
            "--offline",
            "-j",
            "1",
            "--example",
            "h2h_morse",
        ],
        cwd=root,
        check=True,
        env=env,
    )
    binary = target / "release" / "examples" / "h2h_morse"
    data = (root / "examples/h2h_gmin_morse.data").read_text()
    for name, _n, rho, budget, ref in systems:
        for seed in seeds:
            if args.arm in ("anneal", "both") and not have(rows, name, "anneal", seed):
                print(f"anneal {name} {seed}", file=sys.stderr, flush=True)
                rows.append(run_anneal(binary, name, seed, budget))
                write_rows(seeds_path, rows)
            if args.arm == "anneal":
                continue
            if not have(rows, name, "gmin", seed):
                print(f"gmin {name} {seed}", file=sys.stderr, flush=True)
                rows.append(
                    run_gmin(
                        binary,
                        args.gmin,
                        data,
                        name,
                        rho,
                        seed,
                        budget,
                        ref,
                        work / "gmin" / name / str(seed),
                    )
                )
                write_rows(seeds_path, rows)

    summary = summarise(rows, reused_keys)
    # A one-seed timing must not replace the stored panel.
    full = (
        not args.system
        and not args.seed
        and args.arm == "both"
        and len(summary) == len(SYSTEMS)
    )
    if not full:
        print(
            f"panel incomplete ({len(summary)}/{len(SYSTEMS)}); bench rows left as they are",
            file=sys.stderr,
        )
        return
    out_dir = root / "bench/morse-panel"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "seeds.csv", rows)
    with (out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {out_dir / 'summary.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
