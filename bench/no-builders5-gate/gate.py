#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Default-on paired gate for the no-builders5 series.

Reuses the Lennard-Jones panel and runner from commit 89172a22 (plus LJ55)
and the non-Lennard-Jones panel and drivers from commit 754a69f8, under
PAPER_BUDGETS and --host anneal-ci.

Execution protocol:
- One session, strictly serial (one run at a time), pinned to one core.
- Records core, CPU affinity, hashed machine id, code revision, tree SHA,
  and host label per run.
- Even seeds run A then B; odd seeds run B then A (ABBA across consecutive seeds).
- n >= 12 seeds (2000..2011) on every system, and n >= 24 (2000..2023) on
  LJ38, LJ55 and LJ75.
- A/A band (A vs A) on LJ38 and LJ55 at n = 12 (2000..2011).

Rule, declared before any run:
- PASS only if B loses nowhere:
  * no system where B has fewer hits with two-sided Fisher exact p < 0.05, and
  * no system where B's time-to-hit over hitting runs (energy calls primary,
    wall time secondary) is worse with two-sided Mann-Whitney p < 0.05 and
    outside the A/A band.
- Marginal losses (p < 0.10) are reported separately.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ALPHA = 0.05
MARGINAL_P = 0.10
HIT_TOL = 0.001
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261118

SEEDS_12 = tuple(range(2000, 2012))
SEEDS_24 = tuple(range(2000, 2024))
AA_SYSTEMS = ("lj38", "lj55")
FOCUS_24_SYSTEMS = ("lj38", "lj55", "lj75")

# Cambridge Cluster Database pair-well global minima for Lennard-Jones.
CCD_LJ = {
    38: -173.928427,
    55: -279.248470,
    75: -397.492331,
    76: -402.894866,
    77: -409.083517,
    78: -414.794401,
    79: -421.810897,
    80: -428.083564,
    81: -434.343643,
    82: -440.550425,
    83: -446.924094,
    84: -452.657214,
    85: -459.055799,
    86: -465.384493,
    87: -472.098165,
    88: -479.032630,
    89: -486.053911,
    90: -492.433908,
    91: -498.811060,
    92: -505.185309,
    93: -510.877688,
    94: -517.264131,
    95: -523.640211,
    96: -529.879146,
    97: -536.681383,
    98: -543.665361,
    99: -550.666526,
    100: -557.039820,
    101: -563.411308,
    102: -569.363652,
    103: -575.766131,
    104: -582.086642,
}
LJ_PANEL_SIZES = (38, 55, *range(75, 105))


@dataclass(frozen=True)
class SystemSpec:
    panel: str
    system: str
    driver: str
    budget: int
    reference: float
    source: str
    seeds: tuple[int, ...]
    kind: str = ""
    n: int = 0
    rho: float = 0.0


def morse(name: str, budget: int, ref: float, source: str) -> SystemSpec:
    return SystemSpec("nonlj", name, "h2h_morse", budget, ref, source, SEEDS_12)


def metal(name: str, budget: int, ref: float, source: str) -> SystemSpec:
    return SystemSpec("nonlj", name, "h2h_metal", budget, ref, source, SEEDS_12)


def blj(name: str, budget: int, ref: float) -> SystemSpec:
    return SystemSpec(
        "nonlj",
        name,
        "h2h_blj",
        budget,
        ref,
        "bench blj panel PAPER_BUDGETS and SYSTEMS",
        SEEDS_12,
    )


def water(n: int, ref: float) -> SystemSpec:
    return SystemSpec(
        "nonlj",
        f"tip4p{n}",
        "h2h_seed",
        2_000_000,
        ref,
        "scripts/tip4p_h2h run_h2h.py BUDGET, run_water_gmin.py CAMBRIDGE",
        SEEDS_12,
        kind="tip4p",
        n=n,
        rho=0.0,
    )


NONLJ_SYSTEMS: tuple[SystemSpec, ...] = (
    morse("m30r6", 400_000, -118.432844, "scripts/morse_gmin SYSTEMS"),
    morse("m30r14", 400_000, -106.835790, "scripts/morse_gmin SYSTEMS"),
    morse("m38r6", 400_000, -157.477108, "scripts/morse_gmin SYSTEMS"),
    morse("m38r14", 400_000, -144.321054, "scripts/morse_gmin SYSTEMS"),
    morse("m55r6", 1_000_000, -250.286609, "scripts/gmin_panel PAPER_BUDGETS"),
    morse("m55r14", 1_000_000, -220.646208, "scripts/gmin_panel PAPER_BUDGETS"),
    morse("m70r3", 1_000_000, -577.739782, "scripts/morse_panel SYSTEMS"),
    morse("m70r6", 1_000_000, -325.887749, "scripts/morse_panel SYSTEMS"),
    morse("m70r10", 1_000_000, -296.412149, "scripts/morse_panel SYSTEMS"),
    morse("m70r14", 1_000_000, -292.462856, "scripts/morse_panel SYSTEMS"),
    SystemSpec(
        "nonlj",
        "m75r14",
        "h2h_seed",
        4_000_000,
        -318.407330,
        "PairPotential::ccd_morse_minimum, 4e6 from the paper size map",
        SEEDS_12,
        kind="morse",
        n=75,
        rho=14.0,
    ),
    morse("m80r3", 4_000_000, -690.577890, "scripts/morse_panel SYSTEMS"),
    morse("m80r6", 4_000_000, -378.333471, "scripts/gmin_panel PAPER_BUDGETS"),
    morse("m80r10", 4_000_000, -344.911381, "scripts/morse_panel SYSTEMS"),
    morse("m80r14", 4_000_000, -340.811371, "scripts/gmin_panel PAPER_BUDGETS"),
    metal("au38", 400_000, -130.54001708, "scripts/metal_h2h run_darby.py PAPER_BUDGETS"),
    metal("cu6au32", 400_000, -130.79994836, "scripts/metal_h2h run_darby.py PAPER_BUDGETS"),
    metal("au55", 1_000_000, -191.03085280, "scripts/gmin_panel PAPER_BUDGETS"),
    metal("cu27au28", 1_000_000, -182.93136675, "scripts/metal_h2h run_darby.py PAPER_BUDGETS"),
    metal("cu19au19", 80_000, -122.99148391, "h2h_metal budget of the metal_h2h rule"),
    metal("zn38", 50_000, -50.499983, "h2h_metal budget of the metal_h2h rule"),
    metal("zn55", 80_000, -73.310032, "h2h_metal budget of the metal_h2h rule"),
    metal("zn75", 100_000, -100.203580, "h2h_metal budget of the metal_h2h rule"),
    metal("zn80", 100_000, -106.927940, "h2h_metal budget of the metal_h2h rule"),
    blj("blj105n38", 400_000, -177.2090846748),
    blj("blj115n38", 400_000, -187.3676867453),
    blj("blj130n38", 400_000, -191.1840192602),
    blj("blj105n55", 1_000_000, -283.1545313810),
    blj("blj115n55", 1_000_000, -289.2931853058),
    blj("blj130n55", 1_000_000, -306.7337643721),
    blj("blj130n98", 4_000_000, -590.7874132760),
    water(8, -305.5183221676),
    water(12, -492.9082357122),
    water(15, -628.3722525077),
    water(16, -681.1925945898),
    water(20, -872.9887544952),
    water(21, -916.7074302940),
)

NOT_RUN: tuple[tuple[str, str], ...] = (
    (
        "sc126-38 to sc96-80 (nine Sutton-Chen rows)",
        "needs the rgpot Sutton-Chen shared library through libloading; no such library is installed on this host, and h2h_sc exits at start",
    ),
    (
        "sw10, sw13, sw20, tc10, tc20 (Stillinger-Weber and Tersoff)",
        "needs the rgpot covalent shared library; not installed on this host",
    ),
    (
        "pt13, cu13, pt7cu6, pt12cu11, pt11cu12, pt6cu17, pt38, pt19cu19, cu38 (Pt-Cu Gupta)",
        "the repository holds no reference energy for these rows; the panel script takes the reference from a GMIN run, so no hit rule exists to score against",
    ),
    (
        "cu19au19 at the 4e5 Darby budget",
        "h2h_metal changes its budget only through an environment variable, which this gate does not set; cu19au19 runs at the 8e4 budget of the metal_h2h rule",
    ),
    (
        "tip4p25",
        "no Cambridge Cluster Database TIP4P minimum exists at 25 molecules, so no hit rule exists",
    ),
    (
        "(H2O)6 hexamer on GFN2-xTB, PBE and a machine-learned potential; water16, water20 and ammonia clusters on GFN2-xTB",
        "needs xtb, NWChem, a machine-learned potential or CREST; none is installed on this host",
    ),
    (
        "Si and organic covalent rows (covalent_h2h, openconf_h2h, tric_quench)",
        "need xtb, openconf or geomeTRIC; none is installed on this host",
    ),
    (
        "Pt(111) heptamer (OptBench)",
        "needs the OptBench data set and its engine; neither is available on this host",
    ),
)

ROW_FIELDS = [
    "session",
    "experiment",
    "panel",
    "system",
    "driver",
    "seed",
    "arm",
    "order",
    "budget",
    "reference",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "pot_s",
    "outside_s",
    "core",
    "affinity",
    "host",
    "machine_id",
    "code_revision",
    "tree",
    "utc",
]

HOST_FIELDS = [
    "host",
    "machine_id",
    "cpu_model",
    "logical_cpus",
    "os",
    "arch",
    "core",
    "affinity",
    "compiler",
]


def extract_braced(text: str, marker: str) -> str:
    start = text.index(marker)
    brace = text.index("{", start)
    depth = 0
    for index, char in enumerate(text[brace:], brace):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace : index + 1]
    raise RuntimeError(f"unclosed block at {marker}")


def paper_budgets(path: Path) -> dict[str, int]:
    block = extract_braced(path.read_text(), "PAPER_BUDGETS =")
    parsed = ast.literal_eval(block)
    if not isinstance(parsed, dict) or not parsed:
        raise RuntimeError(f"empty PAPER_BUDGETS in {path}")
    out = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, int):
            raise RuntimeError(f"bad PAPER_BUDGETS entry {key!r}")
        out[key] = value
    return out


def paper_seeds(path: Path) -> tuple[int, ...]:
    match = re.search(r"SEEDS = tuple\(range\((\d+), (\d+)\)\)", path.read_text())
    if match is None:
        raise RuntimeError(f"paper seed set missing in {path}")
    start, stop = int(match.group(1)), int(match.group(2))
    seeds = tuple(range(start, stop))
    if seeds != SEEDS_12:
        raise RuntimeError(f"paper seed set {seeds} != {SEEDS_12}")
    return seeds


def proof_references(repo: Path) -> dict[int, float]:
    proof = repo / "examples" / "h2h_gmin_proof.rs"
    text = proof.read_text()
    found = {}
    for name, count, energy in re.findall(
        r'"(lj\d+)" => \((\d+), (-?\d+\.\d+)\)', text
    ):
        size = int(name[2:])
        if size != int(count):
            raise RuntimeError(f"{name} does not match atom count {count}")
        found[size] = float(energy)
    if not found:
        raise RuntimeError(f"no Lennard-Jones references in {proof}")
    tol_rs = (repo / "src" / "target_tol.rs").read_text()
    tol_match = re.search(r"pub const DEFAULT_HIT_TOL: f64 = ([0-9.eE+-]+);", tol_rs)
    if tol_match is None or abs(float(tol_match.group(1)) - HIT_TOL) > 1e-15:
        raise RuntimeError(f"DEFAULT_HIT_TOL mismatch in {repo / 'src' / 'target_tol.rs'}")
    if "anneal_core::DEFAULT_HIT_TOL" not in text and "const HIT_TOL: f64 = 0.001;" not in text:
        raise RuntimeError(f"HIT_TOL definition unexpected in {proof}")
    return found


def build_lj_systems(repo: Path) -> tuple[SystemSpec, ...]:
    panel = repo / "scripts" / "gmin_panel" / "run_panel.py"
    other = repo / "scripts" / "paper_0120" / "run_panel.py"
    budgets = paper_budgets(panel)
    other_budgets = paper_budgets(other)
    for key, value in budgets.items():
        if key.startswith("lj") and key in other_budgets and other_budgets[key] != value:
            raise RuntimeError(f"{key} budget disagrees between panel scripts")
    paper_seeds(panel)
    paper_seeds(other)
    references = proof_references(repo)
    for size, energy in references.items():
        if size in CCD_LJ and abs(CCD_LJ[size] - energy) > 1e-12:
            raise RuntimeError(f"lj{size} reference disagrees with h2h_gmin_proof.rs")
    named_large = [
        budgets[f"lj{size}"]
        for size in range(75, 105)
        if f"lj{size}" in budgets
    ]
    if len(set(named_large)) != 1:
        raise RuntimeError("PAPER_BUDGETS has more than one value on LJ75..104")
    if budgets.get("lj38") != 400_000 or budgets.get("lj55") != 1_000_000:
        raise RuntimeError("unexpected PAPER_BUDGETS for lj38/lj55")
    out: list[SystemSpec] = []
    for size in LJ_PANEL_SIZES:
        key = f"lj{size}"
        if size not in CCD_LJ:
            raise RuntimeError(f"missing reference for {key}")
        if key in budgets:
            budget = budgets[key]
            source = "PAPER_BUDGETS"
        else:
            budget = named_large[0]
            source = "PAPER_BUDGETS (75-104 span)"
        seeds = SEEDS_24 if key in FOCUS_24_SYSTEMS else SEEDS_12
        out.append(
            SystemSpec(
                panel="lj",
                system=key,
                driver="lj_shelf_gate",
                budget=budget,
                reference=CCD_LJ[size],
                source=source,
                seeds=seeds,
                kind="lj",
                n=size,
                rho=0.0,
            )
        )
    return tuple(out)


def all_ab_systems(repo: Path) -> tuple[SystemSpec, ...]:
    lj = build_lj_systems(repo)
    return lj + NONLJ_SYSTEMS


def type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("empty sample")
    if count == 1:
        return float(ordered[0])
    position = 1.0 + (count - 1) * probability
    lower = math.floor(position)
    frac = position - lower
    index = int(lower) - 1
    if frac == 0.0:
        return float(ordered[index])
    return float(ordered[index] * (1.0 - frac) + ordered[index + 1] * frac)


def bootstrap_median_ci(
    values: list[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("empty sample")
    med = type7(values, 0.5)
    if len(values) == 1:
        return med, med, med
    rng = random.Random(seed)
    n = len(values)
    medians = [0.0] * resamples
    for i in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        medians[i] = type7(sample, 0.5)
    return med, type7(medians, 0.025), type7(medians, 0.975)


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    column = hits_a + hits_b
    total = n_a + n_b
    ways = math.comb(total, column)

    def probability(hits: int) -> float:
        return math.comb(n_a, hits) * math.comb(n_b, column - hits) / ways

    observed = probability(hits_a)
    low = max(0, column - n_b)
    high = min(n_a, column)
    tail = sum(
        probability(hits)
        for hits in range(low, high + 1)
        if probability(hits) <= observed * (1.0 + 1e-12)
    )
    return min(1.0, tail)


def midranks(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    scaled = [0] * len(values)
    cursor = 0
    count = len(values)
    while cursor < count:
        end = cursor
        while end + 1 < count and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        rank = cursor + end + 2
        for index in range(cursor, end + 1):
            scaled[order[index]] = rank
        cursor = end + 1
    return scaled


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float] | None:
    """Exact two-sided Mann-Whitney U test on pooled midranks.

    Returns ``(U_left, p_value)`` where larger ``U_left`` means ``left``
    tends to take larger values than ``right``.
    """
    if not left or not right:
        return None
    ranks = midranks(list(left) + list(right))
    n_left = len(left)
    n_right = len(right)
    observed = sum(ranks[:n_left])
    total_n = n_left + n_right
    mean = n_left * (total_n + 1)
    deviation = abs(observed - mean)
    by_count = [{0: 1}]
    for rank in ranks:
        nxt = [dict(layer) for layer in by_count]
        nxt.append({})
        for taken, layer in enumerate(by_count):
            if taken >= n_left:
                break
            dest = nxt[taken + 1]
            for size, count in layer.items():
                dest[size + rank] = dest.get(size + rank, 0) + count
        by_count = nxt[: n_left + 1]
    distribution = by_count[n_left]
    total = sum(distribution.values())
    extreme = sum(
        count
        for size, count in distribution.items()
        if abs(size - mean) + 1e-9 >= deviation
    )
    u_stat = observed / 2.0 - n_left * (n_left + 1) / 2.0
    return u_stat, extreme / total


def self_check() -> None:
    if abs(fisher(9, 12, 6, 12) - 0.40032305828509895) > 1e-12:
        raise RuntimeError("Fisher check failed")
    if abs(fisher(6, 6, 6, 6) - 1.0) > 1e-12:
        raise RuntimeError("Fisher tie check failed")
    tied = mann_whitney([1, 2, 3], [1, 2, 3])
    if tied is None or abs(tied[1] - 1.0) > 1e-12:
        raise RuntimeError("Mann-Whitney tie check failed")
    shift = mann_whitney([4, 5, 6], [1, 2, 3])
    if shift is None or abs(shift[0] - 9.0) > 1e-12 or abs(shift[1] - 0.1) > 1e-12:
        raise RuntimeError("Mann-Whitney shift check failed")
    if abs(type7([1, 2, 3, 4], 0.5) - 2.5) > 1e-12:
        raise RuntimeError("type 7 check failed")
    med, lo, hi = bootstrap_median_ci([10.0, 20.0, 30.0, 40.0, 50.0], resamples=200)
    if med != 30.0 or not (10.0 <= lo <= med <= hi <= 50.0):
        raise RuntimeError("bootstrap CI check failed")
    if set(CCD_LJ) != set(LJ_PANEL_SIZES):
        raise RuntimeError("CCD_LJ does not match LJ_PANEL_SIZES")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def now_session() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def hashed_machine_id() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if candidate.is_file():
            raw = candidate.read_text().strip()
            if raw:
                return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    raw = platform.node() or "unknown"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip().endswith(" name"):
                return value.strip().replace(",", " ")
    return "unknown"


def refuse_stray_environment() -> None:
    stray = sorted(
        k for k in os.environ if k.startswith(("H2H_", "PRIOR_", "RGPOT_", "ANNEAL_"))
    )
    if stray:
        raise RuntimeError(f"refusing to run with stray env vars set: {', '.join(stray)}")


def single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        env[key] = "1"
    return env


def git_info(tree: Path) -> dict[str, object]:
    head = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    tree_sha = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD^{tree}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "-C", str(tree), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "sha": head,
        "tree": tree_sha,
        "porcelain_clean": porcelain == "",
    }


def read_build_provenance(bin_dir: Path, tree: Path) -> dict[str, object]:
    ginfo = git_info(tree)
    if not ginfo["porcelain_clean"]:
        raise RuntimeError(f"source worktree is dirty: {ginfo}")
    rel = bin_dir / "release"
    for name in ("SOURCE_HEAD", "SOURCE_TREE", "PORCELAIN_STATUS", "RUSTC_VERSION"):
        path = rel / name
        if not path.is_file():
            raise RuntimeError(f"missing build provenance file {path.name} in build dir")
    head_file = (rel / "SOURCE_HEAD").read_text().strip()
    tree_file = (rel / "SOURCE_TREE").read_text().strip()
    porcelain_file = (rel / "PORCELAIN_STATUS").read_text().strip()
    rustc_file = (rel / "RUSTC_VERSION").read_text().strip()
    if head_file != ginfo["sha"] or tree_file != ginfo["tree"] or porcelain_file != "":
        raise RuntimeError("build provenance disagrees with clean worktree state")
    return {
        "sha": head_file,
        "tree": tree_file,
        "porcelain_clean": True,
        "rustc_version": rustc_file,
    }


def driver_path(bin_dir: Path, driver: str) -> Path:
    return bin_dir / "release" / "examples" / driver


def verify_drivers(bin_dirs: dict[str, Path], trees: dict[str, Path], systems: tuple[SystemSpec, ...]) -> None:
    for arm_name, bin_dir in bin_dirs.items():
        tree = trees[arm_name]
        for spec in systems:
            exe = driver_path(bin_dir, spec.driver)
            if not exe.is_file():
                raise RuntimeError(f"missing {spec.driver} for arm {arm_name}")
            if spec.driver in {"h2h_morse", "h2h_metal"}:
                out = subprocess.run(
                    [str(exe), "budget", spec.system],
                    cwd=tree,
                    env=single_thread_env(),
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.split()
                if int(out[0]) != spec.budget or abs(float(out[1]) - spec.reference) > 1e-6:
                    raise RuntimeError(
                        f"{spec.system} arm {arm_name}: driver says budget={out[0]} ref={out[1]}, "
                        f"expected {spec.budget} {spec.reference}"
                    )


def driver_command(spec: SystemSpec, bin_dir: Path, seed: int, scratch: str) -> list[str]:
    exe = str(driver_path(bin_dir, spec.driver))
    if spec.driver == "lj_shelf_gate":
        return [exe, str(spec.n), str(seed), str(spec.budget), f"{spec.reference:.12f}"]
    if spec.driver == "h2h_seed":
        stop = repr(spec.reference + HIT_TOL)
        return [
            exe,
            spec.kind,
            str(spec.n),
            repr(spec.rho),
            str(spec.budget),
            str(seed),
            "recommended",
            stop,
            os.path.join(scratch, "run"),
        ]
    return [exe, "run", spec.system, str(seed)]


def parse_driver_output(spec: SystemSpec, seed: int, stdout: str) -> dict[str, object]:
    if spec.driver == "lj_shelf_gate":
        lines = [line for line in stdout.splitlines() if line.strip()]
        if len(lines) != 2 or not lines[0].startswith("n,seed,budget,hit,"):
            raise RuntimeError(f"unexpected lj_shelf_gate output: {stdout[-500:]!r}")
        fields = lines[1].split(",")
        if len(fields) != 9:
            raise RuntimeError(f"unexpected lj_shelf_gate row: {lines[1]!r}")
        if int(fields[1]) != seed:
            raise RuntimeError(f"lj_shelf_gate seed mismatch: {fields[1]} != {seed}")
        hit = fields[3]
        if hit not in {"0", "1"}:
            raise RuntimeError(f"bad hit flag {hit}")
        wall_s = float(fields[7])
        pot_s = float(fields[8])
        return {
            "budget": int(fields[2]),
            "hit": int(hit),
            "evals_to_hit": int(fields[4]),
            "charged": int(fields[5]),
            "best": float(fields[6]),
            "best_str": fields[6],
            "wall_s": wall_s,
            "pot_s": pot_s,
            "outside_s": max(wall_s - pot_s, 0.0),
        }
    if spec.driver == "h2h_seed":
        line = next(x for x in stdout.splitlines() if x.startswith("h2h_row "))
        fields = dict(part.split("=", 1) for part in line.split()[1:])
        if int(fields["seed"]) != seed:
            raise RuntimeError(f"h2h_seed seed mismatch: {fields['seed']} != {seed}")
        wall_s = float(fields["wall_s"])
        pot_s = float(fields["pot_s"])
        outside_s = float(fields.get("outside_s", max(wall_s - pot_s, 0.0)))
        return {
            "budget": int(fields["budget"]),
            "hit": 1 if fields["hit"] == "1" else 0,
            "evals_to_hit": int(fields["evals_to_hit"]),
            "charged": int(fields["charged"]),
            "best": float(fields["best"]),
            "best_str": fields["best"],
            "wall_s": wall_s,
            "pot_s": pot_s,
            "outside_s": outside_s,
        }
    lines = stdout.splitlines()
    header_line = next(x for x in lines if x.startswith("system,"))
    rows = [x for x in lines if x.startswith(f"{spec.system},anneal,{seed},")]
    if len(rows) != 1:
        raise RuntimeError(f"expected one anneal row for {spec.system} seed {seed}, saw {len(rows)}")
    row = dict(zip(header_line.split(","), rows[0].split(",")))
    wall_s = float(row["wall_s"])
    pot_s = float(row.get("pot_s", "0"))
    outside_s = float(row["outside_s"]) if "outside_s" in row else max(wall_s - pot_s, 0.0)
    return {
        "budget": int(row["budget"]),
        "hit": 1 if row["hit"].strip().lower() in {"true", "1"} else 0,
        "evals_to_hit": int(row["evals_to_hit"]),
        "charged": int(row["charged"]),
        "best": float(row["best"]),
        "best_str": row["best"],
        "wall_s": wall_s,
        "pot_s": pot_s,
        "outside_s": outside_s,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    *,
    session: str,
    experiment: str,
    spec: SystemSpec,
    seed: int,
    arm: str,
    order: str,
    bin_dir: Path,
    tree: Path,
    arm_prov: dict[str, object],
    core: int,
    affinity: str,
    host: str,
    machine_id: str,
    log_path: Path,
) -> dict[str, str]:
    env = single_thread_env()
    with tempfile.TemporaryDirectory() as scratch:
        cmd = ["taskset", "-c", str(core), *driver_command(spec, bin_dir, seed, scratch)]
        proc = subprocess.run(
            cmd,
            cwd=tree,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"# experiment={experiment} panel={spec.panel} system={spec.system} "
        f"seed={seed} arm={arm} order={order} core={core} affinity={affinity}\n"
        f"# exit={proc.returncode}\n"
        + proc.stdout
        + proc.stderr,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{experiment} {spec.system} seed {seed} arm {arm} exited {proc.returncode}: "
            f"{proc.stderr[-1000:]}"
        )
    parsed = parse_driver_output(spec, seed, proc.stdout)
    if int(parsed["budget"]) != spec.budget:
        raise RuntimeError(
            f"{spec.system} arm {arm}: driver budget {parsed['budget']} != {spec.budget}"
        )
    if int(parsed["evals_to_hit"]) > spec.budget or int(parsed["charged"]) > spec.budget:
        raise RuntimeError(f"{spec.system} seed {seed} arm {arm} exceeded budget")
    return {
        "session": session,
        "experiment": experiment,
        "panel": spec.panel,
        "system": spec.system,
        "driver": spec.driver,
        "seed": str(seed),
        "arm": arm,
        "order": order,
        "budget": str(spec.budget),
        "reference": f"{spec.reference:.10f}",
        "hit": str(int(parsed["hit"])),
        "evals_to_hit": str(int(parsed["evals_to_hit"])),
        "charged": str(int(parsed["charged"])),
        "best": f"{float(parsed['best']):.10f}",
        "wall_s": f"{float(parsed['wall_s']):.6f}",
        "pot_s": f"{float(parsed['pot_s']):.6f}",
        "outside_s": f"{float(parsed['outside_s']):.6f}",
        "core": str(core),
        "affinity": affinity,
        "host": host,
        "machine_id": machine_id,
        "code_revision": str(arm_prov["sha"]),
        "tree": str(arm_prov["tree"]),
        "utc": now_utc(),
    }


def ordered_ab_systems(systems: tuple[SystemSpec, ...]) -> list[SystemSpec]:
    by_name = {s.system: s for s in systems}
    first_names = ["lj38", "lj55", "lj75", "lj98", "lj102", "lj103", "lj104"]
    first = [by_name[name] for name in first_names]
    nonlj = sorted(
        [s for s in systems if s.panel == "nonlj"],
        key=lambda s: (s.budget, s.system),
    )
    rest_lj = [
        s for s in systems if s.panel == "lj" and s.system not in set(first_names)
    ]
    return first + nonlj + rest_lj


def cmd_plan(args: argparse.Namespace) -> int:
    self_check()
    repo = Path(args.repo)
    systems = all_ab_systems(repo)
    print(f"A/A band: {', '.join(AA_SYSTEMS)} (n={len(SEEDS_12)}, seeds {SEEDS_12[0]}..{SEEDS_12[-1]})")
    print(f"A/B systems: {len(systems)} total")
    for spec in systems:
        print(
            f"  {spec.panel:5s} {spec.system:10s} {spec.driver:14s} "
            f"budget={spec.budget:>9d} ref={spec.reference:>15.8f} "
            f"n={len(spec.seeds):2d} ({spec.seeds[0]}..{spec.seeds[-1]}) [{spec.source}]"
        )
    print(f"Not run: {len(NOT_RUN)} entries")
    for name, reason in NOT_RUN:
        print(f"  {name}: {reason}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    self_check()
    refuse_stray_environment()
    if not args.host or not args.host.strip():
        raise RuntimeError("--host must be non-empty")
    host = args.host.strip()
    core = int(args.core)
    os.sched_setaffinity(0, {core})
    aff_set = os.sched_getaffinity(0)
    if aff_set != {core}:
        raise RuntimeError(f"failed to pin process to core {core}, got {sorted(aff_set)}")
    affinity = ",".join(str(c) for c in sorted(aff_set))

    a_tree = Path(args.a_tree).resolve()
    b_tree = Path(args.b_tree).resolve()
    a_bin = Path(args.a_bin).resolve()
    b_bin = Path(args.b_bin).resolve()
    out_dir = Path(args.out_dir).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    prov_a = read_build_provenance(a_bin, a_tree)
    prov_b = read_build_provenance(b_bin, b_tree)
    if prov_a["rustc_version"] != prov_b["rustc_version"]:
        raise RuntimeError(f"compiler mismatch: {prov_a['rustc_version']} vs {prov_b['rustc_version']}")

    systems = all_ab_systems(a_tree)
    verify_drivers({"A": a_bin, "B": b_bin}, {"A": a_tree, "B": b_tree}, systems)

    machine_id = hashed_machine_id()
    cpu = cpu_model()
    logical_cpus = os.cpu_count() or 1
    session = args.session or now_session()
    started_utc = now_utc()

    hosts_path = out_dir / "hosts.csv"
    write_csv(
        hosts_path,
        HOST_FIELDS,
        [
            {
                "host": host,
                "machine_id": machine_id,
                "cpu_model": cpu,
                "logical_cpus": str(logical_cpus),
                "os": platform.system(),
                "arch": platform.machine(),
                "core": str(core),
                "affinity": affinity,
                "compiler": str(prov_a["rustc_version"]),
            }
        ],
    )

    rows_path = out_dir / "rows.csv"
    existing_rows = read_csv(rows_path)
    done = {
        (r["experiment"], r["panel"], r["system"], int(r["seed"]), r["arm"])
        for r in existing_rows
    }

    by_name = {s.system: s for s in systems}
    aa_specs = [
        SystemSpec(
            panel="lj",
            system=name,
            driver=by_name[name].driver,
            budget=by_name[name].budget,
            reference=by_name[name].reference,
            source=by_name[name].source,
            seeds=SEEDS_12,
            kind=by_name[name].kind,
            n=by_name[name].n,
            rho=by_name[name].rho,
        )
        for name in AA_SYSTEMS
    ]
    ab_specs = ordered_ab_systems(systems)

    total_runs = 2 * sum(len(s.seeds) for s in aa_specs) + 2 * sum(len(s.seeds) for s in ab_specs)
    completed_count = len(done)

    # 1. A/A band
    for spec in aa_specs:
        for seed in spec.seeds:
            order = "A0A1" if seed % 2 == 0 else "A1A0"
            seq = ("A0", "A1") if seed % 2 == 0 else ("A1", "A0")
            for arm in seq:
                key = ("aa", spec.panel, spec.system, seed, arm)
                if key in done:
                    continue
                row = run_one(
                    session=session,
                    experiment="aa",
                    spec=spec,
                    seed=seed,
                    arm=arm,
                    order=order,
                    bin_dir=a_bin,
                    tree=a_tree,
                    arm_prov=prov_a,
                    core=core,
                    affinity=affinity,
                    host=host,
                    machine_id=machine_id,
                    log_path=logs_dir / "aa" / spec.panel / spec.system / f"{seed}-{arm}.log",
                )
                append_csv(rows_path, ROW_FIELDS, row)
                done.add(key)
                completed_count += 1
                print(
                    f"[{completed_count}/{total_runs}] aa {spec.system} seed={seed} arm={arm} "
                    f"hit={row['hit']} evals={row['evals_to_hit']} wall={row['wall_s']}",
                    flush=True,
                )

    # 2. A/B panels
    for spec in ab_specs:
        for seed in spec.seeds:
            order = "AB" if seed % 2 == 0 else "BA"
            seq = ("A", "B") if seed % 2 == 0 else ("B", "A")
            for arm in seq:
                key = ("ab", spec.panel, spec.system, seed, arm)
                if key in done:
                    continue
                bin_dir = a_bin if arm == "A" else b_bin
                tree = a_tree if arm == "A" else b_tree
                arm_prov = prov_a if arm == "A" else prov_b
                row = run_one(
                    session=session,
                    experiment="ab",
                    spec=spec,
                    seed=seed,
                    arm=arm,
                    order=order,
                    bin_dir=bin_dir,
                    tree=tree,
                    arm_prov=arm_prov,
                    core=core,
                    affinity=affinity,
                    host=host,
                    machine_id=machine_id,
                    log_path=logs_dir / "ab" / spec.panel / spec.system / f"{seed}-{arm}.log",
                )
                append_csv(rows_path, ROW_FIELDS, row)
                done.add(key)
                completed_count += 1
                print(
                    f"[{completed_count}/{total_runs}] ab {spec.panel}/{spec.system} seed={seed} "
                    f"arm={arm} hit={row['hit']} evals={row['evals_to_hit']} wall={row['wall_s']}",
                    flush=True,
                )

    finished_utc = now_utc()
    sessions_path = out_dir / "sessions.jsonl"
    session_entry = {
        "session": session,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "host": host,
        "machine_id": machine_id,
        "core": core,
        "affinity": affinity,
        "cpu_model": cpu,
        "logical_cpus": logical_cpus,
        "compiler": prov_a["rustc_version"],
        "arms": {"A": prov_a, "B": prov_b},
        "aa_systems": list(AA_SYSTEMS),
        "aa_n": len(SEEDS_12),
        "ab_systems": len(ab_specs),
        "ab_focus_n24": list(FOCUS_24_SYSTEMS),
        "ab_default_n": len(SEEDS_12),
        "schedule": "serial, one run at a time, pinned to one core, interleaved order-flipped ABBA per seed",
        "rows_written": completed_count,
        "status": "complete",
    }
    sessions_path.write_text(json.dumps(session_entry, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def summarize_metric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "ci_low": None, "ci_high": None, "iqr": None}
    med, ci_lo, ci_hi = bootstrap_median_ci(values)
    iqr = type7(values, 0.75) - type7(values, 0.25)
    return {
        "n": len(values),
        "median": med,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "iqr": iqr,
    }


def analyse_pair(
    left_rows: list[dict[str, str]],
    right_rows: list[dict[str, str]],
    expected_seeds: tuple[int, ...],
    left_label: str,
    right_label: str,
) -> dict[str, object]:
    by_l = {int(r["seed"]): r for r in left_rows}
    by_r = {int(r["seed"]): r for r in right_rows}
    if sorted(by_l) != list(expected_seeds) or sorted(by_r) != list(expected_seeds):
        raise RuntimeError(
            f"incomplete seeds for {left_rows[0]['system'] if left_rows else 'unknown'}: "
            f"{left_label}={len(by_l)} {right_label}={len(by_r)} expected={len(expected_seeds)}"
        )
    for seed in expected_seeds:
        rl, rr = by_l[seed], by_r[seed]
        if rl["core"] != rr["core"] or rl["affinity"] != rr["affinity"] or rl["order"] != rr["order"]:
            raise RuntimeError(f"seed {seed} split across cores/affinities/orders")

    n = len(expected_seeds)
    hits_l = sum(by_l[s]["hit"] == "1" for s in expected_seeds)
    hits_r = sum(by_r[s]["hit"] == "1" for s in expected_seeds)
    fp = fisher(hits_l, n, hits_r, n)

    calls_l = [float(by_l[s]["evals_to_hit"]) for s in expected_seeds if by_l[s]["hit"] == "1"]
    calls_r = [float(by_r[s]["evals_to_hit"]) for s in expected_seeds if by_r[s]["hit"] == "1"]
    wall_l = [float(by_l[s]["wall_s"]) for s in expected_seeds if by_l[s]["hit"] == "1"]
    wall_r = [float(by_r[s]["wall_s"]) for s in expected_seeds if by_r[s]["hit"] == "1"]

    calls_sum_l = summarize_metric(calls_l)
    calls_sum_r = summarize_metric(calls_r)
    wall_sum_l = summarize_metric(wall_l)
    wall_sum_r = summarize_metric(wall_r)

    mw_calls = mann_whitney(calls_r, calls_l)
    mw_wall = mann_whitney(wall_r, wall_l)

    med_cl = calls_sum_l["median"]
    med_cr = calls_sum_r["median"]
    calls_fold = (
        float(med_cr) / float(med_cl)
        if med_cl is not None and med_cr is not None and float(med_cl) > 0.0
        else None
    )

    med_wl = wall_sum_l["median"]
    med_wr = wall_sum_r["median"]
    wall_fold = (
        float(med_wr) / float(med_wl)
        if med_wl is not None and med_wr is not None and float(med_wl) > 0.0
        else None
    )

    identical = sum(
        by_l[s]["hit"] == by_r[s]["hit"]
        and by_l[s]["evals_to_hit"] == by_r[s]["evals_to_hit"]
        and by_l[s]["charged"] == by_r[s]["charged"]
        and by_l[s]["best"] == by_r[s]["best"]
        for s in expected_seeds
    )

    return {
        "n_seeds": n,
        "hits_a": hits_l,
        "hits_b": hits_r,
        "fisher_p": fp,
        "calls_a": calls_sum_l,
        "calls_b": calls_sum_r,
        "mw_u_calls": None if mw_calls is None else mw_calls[0],
        "mw_p_calls": None if mw_calls is None else mw_calls[1],
        "calls_fold": calls_fold,
        "wall_a": wall_sum_l,
        "wall_b": wall_sum_r,
        "mw_u_wall": None if mw_wall is None else mw_wall[0],
        "mw_p_wall": None if mw_wall is None else mw_wall[1],
        "wall_fold": wall_fold,
        "identical_seeds": identical,
    }


def fmt_p(p: float | None) -> str:
    if p is None:
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def fmt_calls_ci(summary: dict[str, object]) -> str:
    if not summary["n"]:
        return f"n/a (n=0)"
    return (
        f"{float(summary['median']):,.0f} "
        f"[{float(summary['ci_low']):,.0f}, {float(summary['ci_high']):,.0f}] "
        f"(n={summary['n']})"
    )


def fmt_wall_ci(summary: dict[str, object]) -> str:
    if not summary["n"]:
        return f"n/a (n=0)"
    return (
        f"{float(summary['median']):.3f} "
        f"[{float(summary['ci_low']):.3f}, {float(summary['ci_high']):.3f}] "
        f"(n={summary['n']})"
    )


def render_report(summary: dict[str, object]) -> str:
    sess = summary["session"]
    host_info = summary["host"]
    aa = summary["aa_band"]
    systems = summary["systems"]
    verdict = summary["verdict"]
    arms = sess["arms"]

    lines: list[str] = [
        "# Default-on paired gate for no-builders5",
        "",
        f"Verdict: **{verdict['status']}** — {verdict['summary']}",
        "",
        "## Declared rule",
        "",
        verdict["rule"],
        "",
        "## Session and build provenance",
        "",
        f"- Session: `{sess['session']}` ({sess['started_utc']} to {sess['finished_utc']})",
        f"- Host label: `{host_info['host']}`, hashed machine id `{host_info['machine_id']}`",
        f"- CPU: {host_info['cpu_model']} ({host_info['logical_cpus']} logical CPUs), pinned core `{host_info['core']}` (affinity `{host_info['affinity']}`)",
        f"- Compiler: `{host_info['compiler']}`",
        f"- Arm A: commit `{arms['A']['sha']}`, tree `{arms['A']['tree']}`, clean porcelain `{arms['A']['porcelain_clean']}`",
        f"- Arm B: commit `{arms['B']['sha']}`, tree `{arms['B']['tree']}`, clean porcelain `{arms['B']['porcelain_clean']}`",
        f"- Schedule: {sess['schedule']}; `n={len(SEEDS_24)}` seeds on {', '.join(sess['ab_focus_n24'])}, `n={len(SEEDS_12)}` seeds on all other systems; A/A band on {', '.join(sess['aa_systems'])} at `n={sess['aa_n']}`.",
        f"- Bootstrap CIs: {BOOTSTRAP_RESAMPLES:,} percentile resamples (seed `{BOOTSTRAP_SEED}`), medians and quantiles via Hyndman-Fan type 7 over hitting runs.",
        "",
        "## A/A calibration band (A vs A, n=12)",
        "",
        "| system | hits A0 | hits A1 | Fisher p | calls A0 median [95% CI] | calls A1 median [95% CI] | MW p calls | wall A0 median [95% CI] (s) | wall A1 median [95% CI] (s) | MW p wall |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: |",
    ]
    for row in aa["systems"]:
        lines.append(
            f"| {row['system']} | {row['hits_a']}/{row['n_seeds']} | {row['hits_b']}/{row['n_seeds']} "
            f"| {fmt_p(row['fisher_p'])} | {fmt_calls_ci(row['calls_a'])} | {fmt_calls_ci(row['calls_b'])} "
            f"| {fmt_p(row['mw_p_calls'])} | {fmt_wall_ci(row['wall_a'])} | {fmt_wall_ci(row['wall_b'])} "
            f"| {fmt_p(row['mw_p_wall'])} |"
        )
    lines.extend(
        [
            "",
            f"Observed A/A band bounds: calls max fold = `{aa['calls_max_fold']:.6f}`, "
            f"wall max fold = `{aa['wall_max_fold']:.6f}`, "
            f"calls min MW p = `{fmt_p(aa['calls_min_mw_p'])}`, "
            f"wall min MW p = `{fmt_p(aa['wall_min_mw_p'])}`.",
            "",
            "## Significant losses (rule triggers: p < 0.05 and outside A/A band)",
            "",
        ]
    )
    if not verdict["losses"]:
        lines.append("None.")
    else:
        for item in verdict["losses"]:
            lines.append(
                f"- {item['panel']}/{item['system']}: {', '.join(item['reasons'])} "
                f"(hits {item['hits_a']}/{item['n_seeds']} vs {item['hits_b']}/{item['n_seeds']}, Fisher p={fmt_p(item['fisher_p'])}, "
                f"MW p calls={fmt_p(item['mw_p_calls'])}, MW p wall={fmt_p(item['mw_p_wall'])})"
            )

    lines.extend(["", "## Marginal losses (p < 0.10, reported separately)", ""])
    if not verdict["marginal_losses"]:
        lines.append("None.")
    else:
        for item in verdict["marginal_losses"]:
            lines.append(
                f"- {item['panel']}/{item['system']}: {', '.join(item['marginal_reasons'])} "
                f"(hits {item['hits_a']}/{item['n_seeds']} vs {item['hits_b']}/{item['n_seeds']}, Fisher p={fmt_p(item['fisher_p'])}; "
                f"calls A={fmt_calls_ci(item['calls_a'])} vs B={fmt_calls_ci(item['calls_b'])}, MW p={fmt_p(item['mw_p_calls'])}; "
                f"wall A={fmt_wall_ci(item['wall_a'])} vs B={fmt_wall_ci(item['wall_b'])}, MW p={fmt_p(item['mw_p_wall'])})"
            )

    lines.extend(["", "## Focus systems (LJ38, LJ55, LJ75 at n=24)", ""])
    lines.append(
        "| system | budget | hits A | hits B | Fisher p | calls A median [95% CI] | calls B median [95% CI] | MW p calls | wall A median [95% CI] (s) | wall B median [95% CI] (s) | MW p wall | verdict |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | --- |")
    for s in [row for row in systems if row["system"] in FOCUS_24_SYSTEMS]:
        lines.append(
            f"| {s['system']} | {s['budget']:,} | {s['hits_a']}/{s['n_seeds']} | {s['hits_b']}/{s['n_seeds']} "
            f"| {fmt_p(s['fisher_p'])} | {fmt_calls_ci(s['calls_a'])} | {fmt_calls_ci(s['calls_b'])} "
            f"| {fmt_p(s['mw_p_calls'])} | {fmt_wall_ci(s['wall_a'])} | {fmt_wall_ci(s['wall_b'])} "
            f"| {fmt_p(s['mw_p_wall'])} | {'LOSS' if s['loss'] else 'ok'} |"
        )

    lines.extend(
        [
            "",
            "## Full per-system table",
            "",
            "| panel | system | driver | budget | n | hits A | hits B | Fisher p | calls A median [95% CI] | calls B median [95% CI] | MW p calls | wall A median [95% CI] (s) | wall B median [95% CI] (s) | MW p wall | loss |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | --- |",
        ]
    )
    for s in systems:
        lines.append(
            f"| {s['panel']} | {s['system']} | {s['driver']} | {s['budget']:,} | {s['n_seeds']} "
            f"| {s['hits_a']}/{s['n_seeds']} | {s['hits_b']}/{s['n_seeds']} | {fmt_p(s['fisher_p'])} "
            f"| {fmt_calls_ci(s['calls_a'])} | {fmt_calls_ci(s['calls_b'])} | {fmt_p(s['mw_p_calls'])} "
            f"| {fmt_wall_ci(s['wall_a'])} | {fmt_wall_ci(s['wall_b'])} | {fmt_p(s['mw_p_wall'])} "
            f"| {', '.join(s['loss_reasons']) if s['loss'] else 'no'} |"
        )

    lines.extend(["", "## Not run (outside verdict)", ""])
    for name, reason in NOT_RUN:
        lines.append(f"- {name}: {reason}.")
    lines.append("")
    return "\n".join(lines)


def cmd_summarize(args: argparse.Namespace) -> int:
    self_check()
    out_dir = Path(args.out_dir).resolve()
    repo = Path(args.repo).resolve()
    rows = read_csv(out_dir / "rows.csv")
    if not rows:
        raise RuntimeError("rows.csv is empty")
    sessions = [
        json.loads(line)
        for line in (out_dir / "sessions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(sessions) != 1:
        raise RuntimeError(f"expected single session in sessions.jsonl, found {len(sessions)}")
    sess = sessions[0]
    hosts = read_csv(out_dir / "hosts.csv")
    if len(hosts) != 1:
        raise RuntimeError(f"expected single host row in hosts.csv, found {len(hosts)}")
    host_info = hosts[0]

    # Verify single session and consistent host/machine_id
    for r in rows:
        if r["session"] != sess["session"]:
            raise RuntimeError(f"row session {r['session']} != {sess['session']}")
        if r["host"] != host_info["host"] or r["machine_id"] != host_info["machine_id"]:
            raise RuntimeError("row host or machine_id mismatch")

    systems = all_ab_systems(repo)
    by_name = {s.system: s for s in systems}

    # Analyse A/A band
    aa_results = []
    for name in AA_SYSTEMS:
        spec = by_name[name]
        l_rows = [r for r in rows if r["experiment"] == "aa" and r["system"] == name and r["arm"] == "A0"]
        r_rows = [r for r in rows if r["experiment"] == "aa" and r["system"] == name and r["arm"] == "A1"]
        res = analyse_pair(l_rows, r_rows, SEEDS_12, "A0", "A1")
        aa_results.append({"system": name, "budget": spec.budget, **res})

    calls_folds = []
    wall_folds = []
    calls_pvals = []
    wall_pvals = []
    for item in aa_results:
        c0 = item["calls_a"]["median"]
        c1 = item["calls_b"]["median"]
        if c0 and c1:
            calls_folds.append(max(float(c1) / float(c0), float(c0) / float(c1)))
        w0 = item["wall_a"]["median"]
        w1 = item["wall_b"]["median"]
        if w0 and w1:
            wall_folds.append(max(float(w1) / float(w0), float(w0) / float(w1)))
        if item["mw_p_calls"] is not None:
            calls_pvals.append(float(item["mw_p_calls"]))
        if item["mw_p_wall"] is not None:
            wall_pvals.append(float(item["mw_p_wall"]))

    aa_band = {
        "systems": aa_results,
        "calls_max_fold": max(calls_folds) if calls_folds else 1.0,
        "wall_max_fold": max(wall_folds) if wall_folds else 1.0,
        "calls_min_mw_p": min(calls_pvals) if calls_pvals else None,
        "wall_min_mw_p": min(wall_pvals) if wall_pvals else None,
    }

    # Analyse A/B systems
    ab_results = []
    losses = []
    marginal_losses = []
    for spec in systems:
        a_rows = [r for r in rows if r["experiment"] == "ab" and r["system"] == spec.system and r["arm"] == "A"]
        b_rows = [r for r in rows if r["experiment"] == "ab" and r["system"] == spec.system and r["arm"] == "B"]
        res = analyse_pair(a_rows, b_rows, spec.seeds, "A", "B")

        hits_drop = res["hits_b"] < res["hits_a"]
        lose_hit = hits_drop and float(res["fisher_p"]) < ALPHA

        med_ca = res["calls_a"]["median"]
        med_cb = res["calls_b"]["median"]
        calls_worse = med_ca is not None and med_cb is not None and float(med_cb) > float(med_ca)
        calls_outside_aa = res["calls_fold"] is not None and float(res["calls_fold"]) > aa_band["calls_max_fold"]
        lose_calls = (
            calls_worse
            and res["mw_p_calls"] is not None
            and float(res["mw_p_calls"]) < ALPHA
            and calls_outside_aa
        )

        med_wa = res["wall_a"]["median"]
        med_wb = res["wall_b"]["median"]
        wall_worse = med_wa is not None and med_wb is not None and float(med_wb) > float(med_wa)
        wall_outside_aa = res["wall_fold"] is not None and float(res["wall_fold"]) > aa_band["wall_max_fold"]
        lose_wall = (
            wall_worse
            and res["mw_p_wall"] is not None
            and float(res["mw_p_wall"]) < ALPHA
            and wall_outside_aa
        )

        loss_reasons = []
        if lose_hit:
            loss_reasons.append("hits (Fisher p<0.05)")
        if lose_calls:
            loss_reasons.append("calls to hit (MW p<0.05, outside A/A band)")
        if lose_wall:
            loss_reasons.append("wall time to hit (MW p<0.05, outside A/A band)")

        marginal_reasons = []
        if hits_drop and float(res["fisher_p"]) < MARGINAL_P:
            marginal_reasons.append(f"hits drop (Fisher p={fmt_p(float(res['fisher_p']))})")
        if calls_worse and res["mw_p_calls"] is not None and float(res["mw_p_calls"]) < MARGINAL_P:
            marginal_reasons.append(f"slower calls (MW p={fmt_p(float(res['mw_p_calls']))})")
        if wall_worse and res["mw_p_wall"] is not None and float(res["mw_p_wall"]) < MARGINAL_P:
            marginal_reasons.append(f"slower wall (MW p={fmt_p(float(res['mw_p_wall']))})")

        entry = {
            "panel": spec.panel,
            "system": spec.system,
            "driver": spec.driver,
            "budget": spec.budget,
            "reference": spec.reference,
            "source": spec.source,
            **res,
            "calls_outside_aa": calls_outside_aa,
            "wall_outside_aa": wall_outside_aa,
            "lose_hit": lose_hit,
            "lose_calls": lose_calls,
            "lose_wall": lose_wall,
            "loss": bool(loss_reasons),
            "loss_reasons": loss_reasons,
            "marginal_reasons": marginal_reasons,
        }
        ab_results.append(entry)
        if loss_reasons:
            losses.append(entry)
        if marginal_reasons:
            marginal_losses.append(entry)

    verdict_status = "PASS" if not losses else "FAIL"
    verdict_summary = (
        f"B loses nowhere across {len(ab_results)} systems"
        if verdict_status == "PASS"
        else f"B loses on {len(losses)} of {len(ab_results)} systems: "
        + ", ".join(f"{x['panel']}/{x['system']}" for x in losses)
    )
    rule_text = (
        "PASS only if B loses nowhere: no system where B has fewer hits with two-sided "
        "Fisher exact p < 0.05, and no system where B's time-to-hit over hitting runs "
        "(energy calls primary, wall time secondary) is worse with two-sided exact "
        "Mann-Whitney p < 0.05 and outside the observed A/A band."
    )

    summary = {
        "session": sess,
        "host": host_info,
        "aa_band": aa_band,
        "systems": ab_results,
        "not_run": [{"name": n, "reason": r} for n, r in NOT_RUN],
        "verdict": {
            "status": verdict_status,
            "summary": verdict_summary,
            "rule": rule_text,
            "alpha": ALPHA,
            "marginal_p": MARGINAL_P,
            "losses": [
                {
                    "panel": x["panel"],
                    "system": x["system"],
                    "reasons": x["loss_reasons"],
                    "n_seeds": x["n_seeds"],
                    "hits_a": x["hits_a"],
                    "hits_b": x["hits_b"],
                    "fisher_p": x["fisher_p"],
                    "mw_p_calls": x["mw_p_calls"],
                    "mw_p_wall": x["mw_p_wall"],
                }
                for x in losses
            ],
            "marginal_losses": marginal_losses,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path = Path(args.report)
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(f"verdict: {verdict_status} ({verdict_summary})", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--repo", required=True)
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run")
    run.add_argument("--a-tree", required=True)
    run.add_argument("--a-bin", required=True)
    run.add_argument("--b-tree", required=True)
    run.add_argument("--b-bin", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--logs-dir", required=True)
    run.add_argument("--host", required=True)
    run.add_argument("--core", type=int, default=0)
    run.add_argument("--session", default=None)
    run.set_defaults(func=cmd_run)

    summ = sub.add_parser("summarize")
    summ.add_argument("--repo", required=True)
    summ.add_argument("--out-dir", required=True)
    summ.add_argument("--report", required=True)
    summ.set_defaults(func=cmd_summarize)

    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"gate: {exc}", file=sys.stderr)
        raise SystemExit(1)
