#!/usr/bin/env python3
# Copyright (c) anneal developers
"""OCCat against anneal and GMIN, one core, paper evaluation budgets.

Every system runs seeds 2000-2005. It continues through seed 2011 when
six more seeds of the slower code fit in six hours. A table of six or
twelve seeds carries a two-sided Fisher exact test on hits and a
two-sided Mann-Whitney test on evaluations, wall, time inside the
potential, and time outside it. A stored GMIN row at the same seed and
budget is copied. A miss is the full budget. Medians are Hyndman-Fan
type 7.
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

HIT_TOL = 0.001
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
EXTRA_SEEDS = (2006, 2007, 2008, 2009, 2010, 2011)
# Six further seeds of the slower code, when they fit in this much wall.
EXTEND_WALL_S = 6 * 3600.0
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OCCAT = Path("/work/occat-h2h/OCCat_release/bin/occat")
ANNEAL_LJ = Path("/work/occat-h2h/target/release/examples/h2h_gmin_proof")
ANNEAL_METAL = Path("/work/occat-h2h/target/release/examples/h2h_metal")
GMIN = Path("/work/gmin-h2h/bin/GMIN")
GMIN_DATA = ROOT / "examples" / "h2h_gmin_lj.data"

# A, xi, p, q, r0. Darby, Mortimer-Jones, Johnston and Roberts, 2002.
CU = (0.0855, 1.224, 10.960, 2.278, 2.556)
AU = (0.2061, 1.790, 10.229, 4.036, 2.884)
CU_AU = (0.1539, 1.5605, 11.05, 3.0475, 2.556)
# Cleri and Rosato, Phys. Rev. B 48, 22 (1993). Same order as OCCat.
PT = (0.2975, 2.695, 10.612, 4.004, 2.775)
# GMIN's BGUPTAT order is A, p, q, xi, r0.
CU_GMIN = (0.0855, 10.960, 2.278, 1.224, 2.556)
PT_GMIN = (0.2975, 10.612, 4.004, 2.695, 2.775)

SYSTEMS = {
    "lj38": {"kind": "lj", "n": 38, "budget": 400_000, "reference": -173.928427},
    "lj55": {"kind": "lj", "n": 55, "budget": 1_000_000, "reference": -279.248470},
    "lj75": {"kind": "lj", "n": 75, "budget": 4_000_000, "reference": -397.492331},
    # OCCat's published Gupta geometries, evaluated on this commit's potential.
    # Cobalt and Pd–Pt are not in that potential.
    "cu38": {"kind": "gupta", "n": 38, "budget": 400_000, "reference": -108.967590456248, "element": "Cu", "anneal": "cu38", "gmin": CU_GMIN},
    "cu55": {"kind": "gupta", "n": 55, "budget": 1_000_000, "reference": -163.213172694421, "element": "Cu", "anneal": "pt0cu55", "gmin": CU_GMIN},
    "cu75": {"kind": "gupta", "n": 75, "budget": 4_000_000, "reference": -225.309012274363, "element": "Cu", "anneal": "pt0cu75", "gmin": CU_GMIN},
    "cu98": {"kind": "gupta", "n": 98, "budget": 4_000_000, "reference": -299.078301345296, "element": "Cu", "anneal": "pt0cu98", "gmin": CU_GMIN},
    "cu104": {"kind": "gupta", "n": 104, "budget": 4_000_000, "reference": -318.367763340310, "element": "Cu", "anneal": "pt0cu104", "gmin": CU_GMIN},
    "pt38": {"kind": "gupta", "n": 38, "budget": 400_000, "reference": -199.303954795242, "element": "Pt", "anneal": "pt38", "gmin": PT_GMIN},
    "pt55": {"kind": "gupta", "n": 55, "budget": 1_000_000, "reference": -291.685913180440, "element": "Pt", "anneal": "pt55cu0", "gmin": PT_GMIN},
    "pt75": {"kind": "gupta", "n": 75, "budget": 4_000_000, "reference": -401.555054624194, "element": "Pt", "anneal": "pt75cu0", "gmin": PT_GMIN},
    "pt98": {"kind": "gupta", "n": 98, "budget": 4_000_000, "reference": -527.687703653275, "element": "Pt", "anneal": "pt98cu0", "gmin": PT_GMIN},
    "pt104": {"kind": "gupta", "n": 104, "budget": 4_000_000, "reference": -561.476078720705, "element": "Pt", "anneal": "pt104cu0", "gmin": PT_GMIN},
    "au55": {"kind": "gupta", "n": 55, "budget": 1_000_000, "reference": -191.03085280, "counts": (0, 55)},
    "cu6au32": {"kind": "gupta", "n": 38, "budget": 400_000, "reference": -130.79994836, "counts": (6, 32)},
    "cu19au19": {"kind": "gupta", "n": 38, "budget": 400_000, "reference": -122.99148391, "counts": (19, 19)},
    "cu27au28": {"kind": "gupta", "n": 55, "budget": 1_000_000, "reference": -182.93136675, "counts": (27, 28)},
    "lj98": {"kind": "lj", "n": 98, "budget": 4_000_000, "reference": -543.665361},
    "lj104": {"kind": "lj", "n": 104, "budget": 4_000_000, "reference": -582.086642},
}

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
    "code_revision",
    "host",
]

GMIN_SOURCES = [
    ROOT / "bench" / "lj-outside" / "results.csv",
    ROOT / "bench" / "au55-gupta" / "seeds.csv",
    ROOT
    / "bench"
    / "metal-cluster-h2h"
    / "gupta-binary-e2865427"
    / "general-v6-0baf811f.csv",
]


def type7(ordered: list[float], p: float) -> float:
    count = len(ordered)
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    index = lower - 1
    if frac == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - frac) + ordered[index + 1] * frac


def median(values: list[float]) -> float:
    return type7(sorted(values), 0.5)


def fisher_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    a, b = hits_a, n_a - hits_a
    c, d = hits_b, n_b - hits_b

    def prob(aa: int, bb: int, cc: int, dd: int) -> float:
        total = aa + bb + cc + dd
        return math.comb(aa + bb, aa) * math.comb(cc + dd, cc) / math.comb(total, aa + cc)

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
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return _mann_whitney_normal(left, right)
    if not left or not right:
        return float("nan")
    return float(mannwhitneyu(left, right, alternative="two-sided").pvalue)


def _mann_whitney_normal(x: list[float], y: list[float]) -> float:
    n1, n2 = len(x), len(y)
    pooled = sorted(x + y)
    ranks: dict[float, float] = {}
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1] == pooled[i]:
            j += 1
        ranks[pooled[i]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r1 = sum(ranks[v] for v in x)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2.0
    n = n1 + n2
    tie = 0
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1] == pooled[i]:
            j += 1
        t = j - i + 1
        tie += t**3 - t
        i = j + 1
    sigma2 = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if sigma2 <= 0.0:
        return 1.0
    z = (u - mu) / math.sqrt(sigma2)
    return max(0.0, min(1.0, math.erfc(abs(z) / math.sqrt(2.0))))


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes"}


def load_gmin() -> dict[tuple[str, int], dict[str, str]]:
    found: dict[tuple[str, int], dict[str, str]] = {}
    for path in GMIN_SOURCES:
        if not path.is_file():
            continue
        with path.open() as handle:
            for row in csv.DictReader(handle):
                code = row.get("code") or row.get("arm") or ""
                if code != "gmin":
                    continue
                system = row["system"]
                if system not in SYSTEMS:
                    continue
                seed = int(row["seed"])
                budget = int(float(row["budget"]))
                if budget != SYSTEMS[system]["budget"]:
                    continue
                if "pot_s" in row and row["pot_s"] not in ("", None):
                    pot = float(row["pot_s"])
                    wall = float(row["wall_s"])
                    if "outside_s" in row and row["outside_s"] not in ("", None):
                        outside = float(row["outside_s"])
                    else:
                        outside = max(0.0, wall - pot)
                elif "pot_per_eval_s" in row and row["pot_per_eval_s"]:
                    evals = float(row["evals_to_hit"])
                    pot = float(row["pot_per_eval_s"]) * evals
                    outside = float(row["outside_per_eval_s"]) * evals
                    wall = float(row["wall_s"])
                else:
                    continue
                hit = truthy(row["hit"])
                if row.get("evals_to_hit"):
                    evals = int(float(row["evals_to_hit"]))
                else:
                    evals = int(float(row["charged"]))
                charged = int(float(row["charged"])) if row.get("charged") else evals
                best = row.get("best") or ""
                found[(system, seed)] = {
                    "system": system,
                    "code": "gmin",
                    "seed": str(seed),
                    "budget": str(budget),
                    "hit": "1" if hit else "0",
                    "evals_to_hit": str(budget if not hit else evals),
                    "charged": str(charged),
                    "wall_s": f"{wall:.6f}",
                    "best": best,
                    "pot_s": f"{pot:.6f}",
                    "outside_s": f"{outside:.6f}",
                }
    return found


def occat_config(system: str, work: Path) -> Path:
    spec = SYSTEMS[system]
    work.mkdir(parents=True, exist_ok=True)
    if spec["kind"] == "lj":
        body = f"""[SwarmConfig]
PopSize = 30
MaxSteps = 100000000
NProc = 1
PrintEvery = 1000
NTerm = 99999999
[ClusterConfig]
NumberOfAtoms = [{spec['n']}]
NameOfAtoms = ["He"]
Vacancy = [16.0, 16.0, 16.0]
[FunctionConfig]
FunctionType = 0
FunctionParam = [[1.0, 1.0]]
"""
    elif "element" in spec:
        row = PT if spec["element"] == "Pt" else CU
        rendered = "[" + ", ".join(f"{value:.10g}" for value in row) + "]"
        body = f"""[SwarmConfig]
PopSize = 30
MaxSteps = 100000000
NProc = 1
PrintEvery = 1000
NTerm = 99999999
[ClusterConfig]
NumberOfAtoms = [{spec['n']}]
NameOfAtoms = ["{spec['element']}"]
Vacancy = [20.0, 20.0, 20.0]
[FunctionConfig]
FunctionType = 1
FunctionParam = [
  {rendered}
]
"""
    else:
        n_cu, n_au = spec["counts"]
        rows = []
        if n_cu and n_au:
            counts = f"[{n_cu}, {n_au}]"
            names = '["Cu", "Au"]'
            rows = [CU, CU_AU, AU]
        elif n_au:
            counts = f"[{n_au}]"
            names = '["Au"]'
            rows = [AU]
        else:
            counts = f"[{n_cu}]"
            names = '["Cu"]'
            rows = [CU]
        rendered = ",\n  ".join("[" + ", ".join(f"{value:.10g}" for value in row) + "]" for row in rows)
        body = f"""[SwarmConfig]
PopSize = 30
MaxSteps = 100000000
NProc = 1
PrintEvery = 1000
NTerm = 99999999
[ClusterConfig]
NumberOfAtoms = {counts}
NameOfAtoms = {names}
Vacancy = [20.0, 20.0, 20.0]
[FunctionConfig]
FunctionType = 1
FunctionParam = [
  {rendered}
]
"""
    path = work / "config.toml"
    path.write_text(body)
    return path


def run_occat(system: str, seed: int, work: Path) -> dict[str, str]:
    spec = SYSTEMS[system]
    config = occat_config(system, work)
    account = work / "account.bin"
    proc = subprocess.run(
        [
            sys.executable,
            str(HERE / "launch.py"),
            str(config),
            str(account),
            str(seed),
            str(spec["budget"]),
            f"{spec['reference']:.10f}",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=work,
    )
    (work / "launch.log").write_text(proc.stdout + proc.stderr)
    line = ""
    for candidate in (proc.stdout + "\n" + proc.stderr).splitlines():
        if candidate.startswith("occat_row "):
            line = candidate
    if not line:
        raise RuntimeError(f"occat {system} seed {seed} failed\n{(proc.stderr or proc.stdout)[-800:]}")
    fields = {}
    for part in line.split()[1:]:
        key, value = part.split("=", 1)
        fields[key] = value
    hit = fields["hit"] == "1"
    charged = int(float(fields["energy_evals"]))
    budget = spec["budget"]
    if int(fields.get("stop", "0")) not in (1, 2):
        raise RuntimeError(
            f"occat {system} seed {seed} stopped early\n{line}"
        )
    wall = float(fields["wall_s"])
    pot = float(fields["pot_s"])
    return {
        "system": system,
        "code": "occat",
        "seed": str(seed),
        "budget": str(budget),
        "hit": "1" if hit else "0",
        "evals_to_hit": str(charged if hit else budget),
        "charged": str(charged),
        "wall_s": f"{wall:.6f}",
        "best": fields["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{max(0.0, wall - pot):.6f}",
    }


def run_anneal(system: str, seed: int, work: Path) -> dict[str, str]:
    spec = SYSTEMS[system]
    env = os.environ.copy()
    env["H2H_BUDGET"] = str(spec["budget"])
    env["H2H_REFERENCE"] = f"{spec['reference']:.12f}"
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    binary = ANNEAL_LJ if spec["kind"] == "lj" else ANNEAL_METAL
    anneal_name = spec.get("anneal", system)
    proc = subprocess.run(
        ["taskset", "-c", "0", str(binary), "run", anneal_name, str(seed)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=work,
    )
    (work / "anneal.log").write_text(proc.stdout + proc.stderr)
    data = [
        line
        for line in proc.stdout.splitlines()
        if line.startswith(system + ",") or line.startswith(anneal_name + ",")
    ]
    if proc.returncode != 0 or not data:
        raise RuntimeError(f"anneal {system} seed {seed} failed\n{(proc.stderr or proc.stdout)[-800:]}")
    header = (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,"
        "head,tree,cpu,pins,pot_s"
    )
    # Metal rows carry outside columns after pot_s. Take the shared prefix.
    raw = data[-1].split(",")
    names = header.split(",")
    row = {names[i]: raw[i] for i in range(min(len(names), len(raw)))}
    if "pot_s" not in row and len(raw) >= 13:
        row["pot_s"] = raw[9] if spec["kind"] == "gupta" else raw[-1]
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    hit = truthy(row["hit"])
    return {
        "system": system,
        "code": "anneal",
        "seed": row["seed"],
        "budget": row["budget"],
        "hit": "1" if hit else "0",
        "evals_to_hit": row["evals_to_hit"],
        "charged": row["charged"],
        "wall_s": f"{wall:.6f}",
        "best": row["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{max(0.0, wall - pot):.6f}",
    }


def gmin_gupta_data(spec: dict, seed: int) -> str:
    row = " ".join(f"{value:.10g}" for value in spec["gmin"])
    n = spec["n"]
    return (
        f"BGUPTAT {n} {row}\n"
        f"BGUPTATAB {row}\n"
        f"BGUPTATBB {row}\n"
        "SLOPPYCONV 5.0D-3\n"
        "TIGHTCONV 1.0D-3\n"
        "EDIFF 0.003\n"
        "MAXIT 600 500\n"
        "SAVE 2\n"
        "CENTRE\n"
        "RADIUS 50.0\n"
        "TEMPERATURE 1.0\n"
        "STEPS 90000000 1.0\n"
        "STEP 0.40 0.51\n"
        "NEWRESTART 40\n"
        f"RANSEED {seed}\n"
    )


def run_gmin(system: str, seed: int, work: Path) -> dict[str, str]:
    spec = SYSTEMS[system]
    if spec["kind"] != "lj" and "gmin" not in spec:
        raise RuntimeError(f"no stored GMIN row for {system} seed {seed}")
    if not GMIN.is_file():
        raise RuntimeError("GMIN is not built")
    work.mkdir(parents=True, exist_ok=True)
    coords = work / "coords"
    if spec["kind"] == "lj":
        subprocess.run(
            ["taskset", "-c", "0", str(ANNEAL_LJ), "coords", system, str(seed), str(coords)],
            check=True,
            capture_output=True,
            text=True,
        )
        (work / "data").write_text(GMIN_DATA.read_text() + f"RANSEED {seed}\n")
    else:
        subprocess.run(
            [
                "taskset",
                "-c",
                "0",
                str(ANNEAL_METAL),
                "coords",
                spec["anneal"],
                str(seed),
                str(coords),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        (work / "data").write_text(gmin_gupta_data(spec, seed))
    env = os.environ.copy()
    env.update(
        {
            "H2H_BUDGET": str(spec["budget"]),
            "H2H_SYSTEM": system,
            "H2H_SEED": str(seed),
            "H2H_TARGET": f"{spec['reference']:.10f}",
            "H2H_TOL": f"{HIT_TOL}",
            "H2H_ROW": str(work / "row.txt"),
            "OMP_NUM_THREADS": "1",
        }
    )
    proc = subprocess.run(
        ["taskset", "-c", "0", str(GMIN)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=work,
    )
    (work / "gmin.log").write_text(proc.stdout + proc.stderr)
    text = (work / "row.txt").read_text() if (work / "row.txt").is_file() else ""
    fields = {}
    for part in text.split()[1:]:
        key, value = part.split("=", 1)
        fields[key] = value
    if "hit" not in fields:
        raise RuntimeError(f"gmin {system} seed {seed} produced no row\n{text[-400:]}")
    hit = fields["hit"] == "1"
    wall = float(fields["wall_s"])
    pot = float(fields["pot_s"])
    budget = spec["budget"]
    return {
        "system": system,
        "code": "gmin",
        "seed": str(seed),
        "budget": str(budget),
        "hit": "1" if hit else "0",
        "evals_to_hit": fields["evals_to_hit"] if hit else str(budget),
        "charged": fields["charged"],
        "wall_s": f"{wall:.6f}",
        "best": fields["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{max(0.0, wall - pot):.6f}",
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_seeds(path: Path, rows: list[dict[str, str]]) -> None:
    write_csv(path, SEED_FIELDS, rows)


def wall_allows(rows: list[dict[str, str]], system: str) -> bool:
    """True when six more seeds of the slower code fit in six hours."""
    worst = 0.0
    for code in ("anneal", "occat", "gmin"):
        group = [
            row
            for row in rows
            if row["system"] == system and row["code"] == code and int(row["seed"]) in SEEDS
        ]
        if len(group) < len(SEEDS):
            return False
        worst = max(worst, median([float(row["wall_s"]) for row in group]))
    return worst * len(EXTRA_SEEDS) <= EXTEND_WALL_S


def summarise(rows: list[dict[str, str]], out: Path, tests: Path) -> None:
    summary_fields = [
        "system",
        "code",
        "n",
        "hits",
        "hits_over_n",
        "budget",
        "evals_median",
        "wall_median_s",
        "pot_median_s",
        "outside_median_s",
        "wall_ratio",
    ]
    test_fields = [
        "system",
        "n",
        "anneal_hits",
        "occat_hits",
        "gmin_hits",
        "fisher_occat",
        "fisher_gmin",
        "mannwhitney_evals_occat",
        "mannwhitney_wall_occat",
        "mannwhitney_pot_occat",
        "mannwhitney_outside_occat",
        "mannwhitney_evals_gmin",
        "mannwhitney_wall_gmin",
        "mannwhitney_pot_gmin",
        "mannwhitney_outside_gmin",
    ]
    summary_rows = []
    test_rows = []
    for system, spec in SYSTEMS.items():
        grouped = {}
        for code in ("anneal", "occat", "gmin"):
            group = [row for row in rows if row["system"] == system and row["code"] == code]
            grouped[code] = group
        anneal_wall = median([float(row["wall_s"]) for row in grouped["anneal"]]) if grouped["anneal"] else float("nan")
        for code in ("anneal", "occat", "gmin"):
            group = grouped[code]
            if not group:
                continue
            wall = median([float(row["wall_s"]) for row in group])
            summary_rows.append(
                {
                    "system": system,
                    "code": code,
                    "n": str(len(group)),
                    "hits": str(sum(row["hit"] == "1" for row in group)),
                    "hits_over_n": f"{sum(row['hit'] == '1' for row in group)}/{len(group)}",
                    "budget": str(spec["budget"]),
                    "evals_median": f"{median([float(row['evals_to_hit']) for row in group]):.6f}",
                    "wall_median_s": f"{wall:.6f}",
                    "pot_median_s": f"{median([float(row['pot_s']) for row in group]):.6f}",
                    "outside_median_s": f"{median([float(row['outside_s']) for row in group]):.6f}",
                    "wall_ratio": f"{(wall / anneal_wall) if anneal_wall else float('nan'):.6f}",
                }
            )
        common = None
        if all(grouped[code] for code in ("anneal", "occat", "gmin")):
            common = set.intersection(
                *({int(row["seed"]) for row in grouped[code]} for code in ("anneal", "occat", "gmin"))
            )
        if common is not None and len(common) >= len(SEEDS):
            matched = {
                code: [row for row in grouped[code] if int(row["seed"]) in common]
                for code in ("anneal", "occat", "gmin")
            }

            def col(code: str, name: str) -> list[float]:
                return [float(row[name]) for row in matched[code]]

            def hits(code: str) -> int:
                return sum(row["hit"] == "1" for row in matched[code])

            test_rows.append(
                {
                    "system": system,
                    "n": str(len(common)),
                    "anneal_hits": str(hits("anneal")),
                    "occat_hits": str(hits("occat")),
                    "gmin_hits": str(hits("gmin")),
                    "fisher_occat": f"{fisher_two_sided(hits('anneal'), len(common), hits('occat'), len(common)):.6g}",
                    "fisher_gmin": f"{fisher_two_sided(hits('anneal'), len(common), hits('gmin'), len(common)):.6g}",
                    "mannwhitney_evals_occat": f"{mann_whitney(col('anneal','evals_to_hit'), col('occat','evals_to_hit')):.6g}",
                    "mannwhitney_wall_occat": f"{mann_whitney(col('anneal','wall_s'), col('occat','wall_s')):.6g}",
                    "mannwhitney_pot_occat": f"{mann_whitney(col('anneal','pot_s'), col('occat','pot_s')):.6g}",
                    "mannwhitney_outside_occat": f"{mann_whitney(col('anneal','outside_s'), col('occat','outside_s')):.6g}",
                    "mannwhitney_evals_gmin": f"{mann_whitney(col('anneal','evals_to_hit'), col('gmin','evals_to_hit')):.6g}",
                    "mannwhitney_wall_gmin": f"{mann_whitney(col('anneal','wall_s'), col('gmin','wall_s')):.6g}",
                    "mannwhitney_pot_gmin": f"{mann_whitney(col('anneal','pot_s'), col('gmin','pot_s')):.6g}",
                    "mannwhitney_outside_gmin": f"{mann_whitney(col('anneal','outside_s'), col('gmin','outside_s')):.6g}",
                }
            )
    write_csv(out, summary_fields, summary_rows)
    write_csv(tests, test_fields, test_rows)


def seeds_for(system: str) -> tuple[int, ...]:
    return SEEDS


def one(
    system: str,
    seed: int,
    code: str,
    stored: dict,
    work_root: Path,
) -> dict[str, str]:
    if code == "gmin" and (system, seed) in stored:
        print(f"reuse gmin {system} {seed}", flush=True)
        return stored[(system, seed)]
    work = work_root / system / f"{code}-{seed}"
    work.mkdir(parents=True, exist_ok=True)
    print(f"run {code} {system} {seed}", flush=True)
    if code == "occat":
        return run_occat(system, seed, work)
    if code == "anneal":
        return run_anneal(system, seed, work)
    return run_gmin(system, seed, work)


def campaign(seeds_path: Path, summary_path: Path, tests_path: Path, work_root: Path) -> None:
    if socket.gethostname() != "bench":
        raise SystemExit("refusing to run unless the hostname is bench")
    stored = load_gmin()
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "-c", "core.hooksPath=/dev/null", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    host = socket.gethostname()
    rows: list[dict[str, str]] = []
    if seeds_path.is_file():
        with seeds_path.open() as handle:
            rows.extend(csv.DictReader(handle))
    done = {(row["system"], row["code"], int(row["seed"])) for row in rows}
    for system in SYSTEMS:
        for seed in seeds_for(system):
            for code in ("anneal", "occat", "gmin"):
                if (system, code, seed) in done:
                    continue
                row = one(system, seed, code, stored, work_root)
                row["code_revision"] = revision
                row["host"] = host
                rows.append(row)
                done.add((system, code, seed))
                write_seeds(seeds_path, rows)
                print(
                    f"{system} {code} {seed} hit={row['hit']} evals={row['evals_to_hit']} "
                    f"wall={row['wall_s']} pot={row['pot_s']}",
                    flush=True,
                )
        if wall_allows(rows, system):
            print(f"wall allows {system}, extending to {EXTRA_SEEDS[-1]}", flush=True)
            for seed in EXTRA_SEEDS:
                for code in ("anneal", "occat", "gmin"):
                    if (system, code, seed) in done:
                        continue
                    row = one(system, seed, code, stored, work_root)
                    row["code_revision"] = revision
                    row["host"] = host
                    rows.append(row)
                    done.add((system, code, seed))
                    write_seeds(seeds_path, rows)
                    print(
                        f"{system} {code} {seed} hit={row['hit']} evals={row['evals_to_hit']} "
                        f"wall={row['wall_s']} pot={row['pot_s']}",
                        flush=True,
                    )
    write_seeds(seeds_path, rows)
    summarise(rows, summary_path, tests_path)
    write_readme(summary_path, tests_path, summary_path.parent / "README.md")


def write_readme(summary: Path, tests: Path, path: Path) -> None:
    lines = [
        "# OCCat head-to-head",
        "",
        "Wang, Xu, and Cheng, Journal of Computational Chemistry 47, e70583 (2026), doi:10.1002/jcc.70583.",
        "The release is https://github.com/LEDlamar/OCCat_release revision 64330de6a71259471d3a99a5d2afabb0a46a9ea0.",
        "Anneal and OCCat ran in one serial session under `sudo unshare --uts` with hostname bench, core 0.",
        "GMIN rows already stored at the same seed and budget were copied.",
        "A hit is a quenched energy within 0.001 of the Cambridge value.",
        "A miss is the full paper budget.",
        "Medians are Hyndman-Fan type 7.",
        "Inside is time in the potential. Outside is the rest of the wall.",
        "The wall ratio is that code's median wall divided by anneal's.",
        "",
        "| System | Code | Hits | Median evals | Median wall (s) | Median inside (s) | Median outside (s) | Ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    with summary.open() as handle:
        for row in csv.DictReader(handle):
            lines.append(
                "| {system} | {code} | {hits_over_n} | {evals_median} | {wall_median_s} | {pot_median_s} | {outside_median_s} | {wall_ratio} |".format(
                    **row
                )
            )
    if tests.is_file() and tests.stat().st_size > 0:
        body = list(csv.DictReader(tests.open()))
        if body:
            lines.extend(
                [
                    "",
                    "Fisher is the two-sided exact test on hits against anneal. Mann-Whitney is two-sided on evaluations, wall, time inside the potential, and time outside it. A miss is stored as the budget. A system runs seeds 2000-2005 and continues through 2011 when six more seeds of the slower code fit in six hours.",
                    "",
                    "| System | n | Anneal | OCCat | GMIN | Fisher OCCat | Fisher GMIN | MW evals OCCat | MW wall OCCat | MW inside OCCat | MW outside OCCat | MW evals GMIN | MW wall GMIN | MW inside GMIN | MW outside GMIN |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for row in body:
                lines.append(
                    "| {system} | {n} | {anneal_hits} | {occat_hits} | {gmin_hits} | {fisher_occat} | {fisher_gmin} | {mannwhitney_evals_occat} | {mannwhitney_wall_occat} | {mannwhitney_pot_occat} | {mannwhitney_outside_occat} | {mannwhitney_evals_gmin} | {mannwhitney_wall_gmin} | {mannwhitney_pot_gmin} | {mannwhitney_outside_gmin} |".format(
                        **row
                    )
                )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, default=ROOT / "bench" / "occat-h2h" / "seeds.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "bench" / "occat-h2h" / "summary.csv")
    parser.add_argument("--tests", type=Path, default=ROOT / "bench" / "occat-h2h" / "tests.csv")
    parser.add_argument("--work", type=Path, default=Path("/work/occat-h2h/runs"))
    args = parser.parse_args()
    campaign(args.seeds, args.summary, args.tests, args.work)


if __name__ == "__main__":
    main()
