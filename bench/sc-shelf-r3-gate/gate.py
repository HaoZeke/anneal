#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Paired gate for the shelf quarantine and stall-trail fixes.

Arm A and arm B are two release builds. The harness points at those
builds and does not change a budget, a seed, or a start. Both arms use
the recommended preset through the drivers of the existing shelf gates.

Panel. The Lennard-Jones panel of the shelf gate (LJ38 and LJ75 through
LJ104), plus the Morse, Gupta gold, and TIP4P rows of the non-LJ shelf
gate. A size named in ``PAPER_BUDGETS`` uses that budget. Every other
Lennard-Jones size from 75 through 104 uses the one budget that map
assigns throughout that range. Morse, gold, and water use the budgets
their drivers already publish. Every system uses the paper seed set,
twelve seeds.

Schedule. One session, one run at a time, pinned to one core. Within a
seed the systems run in panel order and the two arms of a system run
back to back. The first arm follows ABBA across seeds: AB, BA, BA, AB.
The same seeds feed both arms, so both arms draw the same start. Two
Lennard-Jones systems, LJ38 and LJ75, also run arm A against a second
copy of arm A at the same twelve seeds. That difference is the A/A band.

Rule, declared before any run. B loses nowhere, or the gate fails.
A system is a loss when B hits on fewer seeds and the two-sided Fisher
exact p is below 0.05, or when the median energy calls to a hit, taken
over hitting runs only, is higher for B, the two-sided exact
Mann-Whitney p is below 0.05, and that median difference is strictly
larger than the A/A band. The same tests at p below 0.1 are marginal
losses. A marginal loss is listed and does not change the verdict.
Medians are Hyndman-Fan type 7. Each median carries a 95 percent
percentile bootstrap interval from 10000 resamples of that median.
"""

from __future__ import annotations

import argparse
import ast
import csv
import errno
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HIT_TOL = 0.001
LOSS_P = 0.05
MARGINAL_P = 0.1
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 7619001
AA_SYSTEMS = ("lj38", "lj75")
ARM_A_COMMIT = "76a9001b6c76f7a1297d16d287301278f1876b9d"
ARM_B_TREE = "24e4303aa2ca9b80db48d42759b2b21e2aff3bb1"
HOST_LABEL = "anneal-ci"
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_DOMAIN_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "RAYON_NUM_THREADS": "1",
    "RUST_TEST_THREADS": "1",
    "TOKIO_WORKER_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
}
TRANSIENT_ERRNO = {errno.EAGAIN, errno.EWOULDBLOCK, errno.EINTR}
IO_NOTES: list[str] = []

# Cambridge Cluster Database pair-well global minima, the table the
# Lennard-Jones shelf gate scores against.
CCD = {
    38: -173.928427,
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

ROW_FIELDS = [
    "system",
    "family",
    "comparison",
    "arm",
    "seed",
    "seed_index",
    "order",
    "budget",
    "reference",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "pot_s",
    "core",
    "affinity",
    "machine_id",
    "host",
    "code_revision",
    "tree",
]

HOST_FIELDS = [
    "session",
    "host",
    "machine_id",
    "core",
    "affinity",
    "system",
    "comparison",
    "arm",
    "seed",
]

RULE = (
    "PASS only if B loses on no system. "
    "B loses a system when B hits on fewer seeds and the two-sided Fisher exact "
    "p is below 0.05, or when the median energy calls to a hit on hitting runs "
    "is higher for B than for A, the two-sided exact Mann-Whitney p is below 0.05, "
    "and that median difference is strictly larger than the A/A band. "
    "The A/A band is the larger of the two median-call differences between a second "
    "copy of arm A and the first, on LJ38 and LJ75, at the same twelve seeds. "
    "Marginal losses use the same tests at p below 0.1 and are listed separately. "
    "A marginal loss does not change the verdict."
)


@dataclass(frozen=True)
class System:
    name: str
    family: str
    driver: str
    budget: int
    reference: float
    n: int
    rho: float = 0.0
    kind: str = ""

    @property
    def level(self) -> float:
        return self.reference + HIT_TOL


def order_for(index: int) -> str:
    """First-arm pattern ABBA, repeated: AB, BA, BA, AB."""
    return "AB" if index % 4 in (0, 3) else "BA"


def sequence_for(order: str, labels: tuple[str, str]) -> tuple[str, str]:
    if order == "AB":
        return labels
    if order == "BA":
        return labels[1], labels[0]
    raise RuntimeError(f"unknown order {order}")


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
    if len(seeds) != 12:
        raise RuntimeError(f"paper seed set has {len(seeds)} seeds")
    return seeds


def proof_references(path: Path) -> dict[int, float]:
    text = path.read_text()
    found = {}
    for name, count, energy in re.findall(
        r'"(lj\d+)" => \((\d+), (-?\d+\.\d+)\)', text
    ):
        size = int(name[2:])
        if size != int(count):
            raise RuntimeError(f"{name} does not match atom count {count}")
        found[size] = float(energy)
    if not found:
        raise RuntimeError(f"no Lennard-Jones references in {path}")
    if "const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;" not in text:
        match = re.search(r"const HIT_TOL: f64 = ([0-9.]+);", text)
        if match is None or abs(float(match.group(1)) - HIT_TOL) > 1e-15:
            raise RuntimeError("hit tolerance is not 0.001")
    return found


def require_literal(path: Path, needle: str) -> None:
    if needle not in path.read_text():
        raise RuntimeError(f"{path.name} lacks {needle}")


def morse_row(name: str, n: int, rho: float, reference: float) -> System:
    if n <= 38:
        budget = 400_000
    elif n < 75:
        budget = 1_000_000
    else:
        budget = 4_000_000
    return System(name, "morse", "h2h_morse", budget, reference, n, rho)


def gold_row(name: str, n: int, budget: int, reference: float) -> System:
    return System(name, "gold", "h2h_metal", budget, reference, n)


def water_row(n: int, reference: float) -> System:
    return System(f"tip4p{n}", "water", "h2h_seed", 2_000_000, reference, n, 0.0, "tip4p")


def load_protocol(repo: Path) -> dict:
    panel = repo / "scripts" / "gmin_panel" / "run_panel.py"
    other = repo / "scripts" / "paper_0120" / "run_panel.py"
    proof = repo / "examples" / "h2h_gmin_proof.rs"
    budgets = paper_budgets(panel)
    other_budgets = paper_budgets(other)
    for key, value in budgets.items():
        if key.startswith("lj") and key in other_budgets and other_budgets[key] != value:
            raise RuntimeError(f"{key} budget disagrees between panel scripts")
    seeds = paper_seeds(panel)
    if paper_seeds(other) != seeds:
        raise RuntimeError("paper seed sets disagree")
    require_literal(repo / "src" / "target_tol.rs", "pub const DEFAULT_HIT_TOL: f64 = 1.0e-3;")
    references = proof_references(proof)
    for size, energy in references.items():
        if size in CCD and abs(CCD[size] - energy) > 1e-12:
            raise RuntimeError(f"lj{size} reference disagrees with the proof example")
    named_large = [
        budgets[f"lj{size}"] for size in range(75, 105) if f"lj{size}" in budgets
    ]
    if len(set(named_large)) != 1:
        raise RuntimeError("PAPER_BUDGETS has more than one value on LJ75 through LJ104")
    if "lj38" not in budgets:
        raise RuntimeError("PAPER_BUDGETS has no lj38 entry")
    morse_src = repo / "examples" / "h2h_morse.rs"
    metal_src = repo / "examples" / "h2h_metal.rs"
    potential_src = repo / "src" / "potentials.rs"
    water_src = repo / "scripts" / "tip4p_h2h" / "run_h2h.py"
    water_more = repo / "scripts" / "tip4p_h2h" / "run_water_gmin.py"
    require_literal(water_src, "BUDGET = 2_000_000")
    require_literal(potential_src, "(28, 75) => -318.407330")

    systems: list[System] = []
    for size in (38, *range(75, 105)):
        key = f"lj{size}"
        if size not in CCD:
            raise RuntimeError(f"missing reference for {key}")
        budget = budgets[key] if key in budgets else named_large[0]
        systems.append(System(key, "lj", "lj_shelf_gate", budget, CCD[size], size))

    morse = [
        morse_row("m30r6", 30, 6.0, -118.432844),
        morse_row("m30r14", 30, 14.0, -106.835790),
        morse_row("m38r6", 38, 6.0, -157.477108),
        morse_row("m38r14", 38, 14.0, -144.321054),
        morse_row("m55r6", 55, 6.0, -250.286609),
        morse_row("m55r14", 55, 14.0, -220.646208),
        morse_row("m70r3", 70, 3.0, -577.739782),
        morse_row("m70r6", 70, 6.0, -325.887749),
        morse_row("m70r10", 70, 10.0, -296.412149),
        morse_row("m70r14", 70, 14.0, -292.462856),
        System("m75r14", "morse", "h2h_seed", 4_000_000, -318.407330, 75, 14.0, "morse"),
        morse_row("m80r3", 80, 3.0, -690.577890),
        morse_row("m80r6", 80, 6.0, -378.333471),
        morse_row("m80r10", 80, 10.0, -344.911381),
        morse_row("m80r14", 80, 14.0, -340.811371),
    ]
    gold = [
        gold_row("au38", 38, 400_000, -130.54001708),
        gold_row("cu6au32", 38, 400_000, -130.79994836),
        gold_row("au55", 55, 1_000_000, -191.03085280),
        gold_row("cu27au28", 55, 1_000_000, -182.93136675),
        gold_row("cu19au19", 38, 80_000, -122.99148391),
    ]
    water = [
        water_row(8, -305.5183221676),
        water_row(12, -492.9082357122),
        water_row(15, -628.3722525077),
        water_row(16, -681.1925945898),
        water_row(20, -872.9887544952),
        water_row(21, -916.7074302940),
    ]
    source_needles = {
        "m30r6": (morse_src, "-118.432844"),
        "m30r14": (morse_src, "-106.835790"),
        "m38r6": (morse_src, "-157.477108"),
        "m38r14": (morse_src, "-144.321054"),
        "m55r6": (morse_src, "-250.286609"),
        "m55r14": (morse_src, "-220.646208"),
        "m70r3": (morse_src, "-577.739782"),
        "m70r6": (morse_src, "-325.887749"),
        "m70r10": (morse_src, "-296.412149"),
        "m70r14": (morse_src, "-292.462856"),
        "m75r14": (potential_src, "-318.407330"),
        "m80r3": (morse_src, "-690.577890"),
        "m80r6": (morse_src, "-378.333471"),
        "m80r10": (morse_src, "-344.911381"),
        "m80r14": (morse_src, "-340.811371"),
        "au38": (metal_src, "-130.54001708"),
        "cu6au32": (metal_src, "-130.79994836"),
        "au55": (metal_src, "-191.03085280"),
        "cu27au28": (metal_src, "-182.93136675"),
        "cu19au19": (metal_src, "-122.99148391"),
        "tip4p8": (water_src, "-305.5183221676"),
        "tip4p12": (water_more, "-492.9082357122"),
        "tip4p15": (water_src, "-628.3722525077"),
        "tip4p16": (water_more, "-681.1925945898"),
        "tip4p20": (water_src, "-872.9887544952"),
        "tip4p21": (water_src, "-916.7074302940"),
    }
    declared = {system.name: system for system in morse + gold + water}
    if set(declared) != set(source_needles):
        raise RuntimeError("reference needles do not cover the non-LJ rows")
    for system in morse + gold + water:
        path, needle = source_needles[system.name]
        require_literal(path, needle)
        if abs(system.reference - float(needle)) > 1e-12:
            raise RuntimeError(f"{system.name} reference does not match {needle}")
        if system.name in budgets and budgets[system.name] != system.budget:
            raise RuntimeError(f"{system.name} budget disagrees with PAPER_BUDGETS")
    systems.extend(morse)
    systems.extend(gold)
    systems.extend(water)
    names = [system.name for system in systems]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate system name")
    if any(name not in names for name in AA_SYSTEMS):
        raise RuntimeError("A/A systems are not on the panel")
    return {"systems": systems, "seeds": seeds, "budgets": budgets}


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
    offset = 0
    count = len(values)
    while offset < count:
        end = offset
        while end + 1 < count and values[order[end + 1]] == values[order[offset]]:
            end += 1
        rank = offset + end + 2
        for index in range(offset, end + 1):
            scaled[order[index]] = rank
        offset = end + 1
    return scaled


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float] | None:
    """Exact two-sided Mann-Whitney on the pooled midranks.

    The returned U belongs to ``left``.
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


def bootstrap_median_ci(values: list[float], key: str) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    digest = hashlib.sha256(f"{key}:{BOOTSTRAP_SEED}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "little"))
    count = len(values)
    medians = []
    for _ in range(BOOTSTRAP_N):
        sample = [values[rng.randrange(count)] for _ in range(count)]
        medians.append(type7(sample, 0.5))
    return [type7(medians, 0.025), type7(medians, 0.975)]


def judge(
    hits_a: int,
    hits_b: int,
    n_seeds: int,
    evals_a: list[float],
    evals_b: list[float],
    aa_band: float,
) -> dict:
    fisher_p = fisher(hits_a, n_seeds, hits_b, n_seeds)
    median_a = type7(evals_a, 0.5) if evals_a else None
    median_b = type7(evals_b, 0.5) if evals_b else None
    mw = mann_whitney(evals_b, evals_a)
    mw_p = None if mw is None else mw[1]
    mw_u = None if mw is None else mw[0]
    difference = None
    if median_a is not None and median_b is not None:
        difference = median_b - median_a
    outside = difference is not None and difference > aa_band
    more_calls = difference is not None and difference > 0.0

    def hit_loss(alpha: float) -> bool:
        return hits_b < hits_a and fisher_p < alpha

    def call_loss(alpha: float) -> bool:
        return bool(more_calls and outside and mw_p is not None and mw_p < alpha)

    reasons = []
    if hit_loss(LOSS_P):
        reasons.append("fewer hits")
    if call_loss(LOSS_P):
        reasons.append("more calls")
    marginal = []
    if hit_loss(MARGINAL_P) and not hit_loss(LOSS_P):
        marginal.append("fewer hits")
    if call_loss(MARGINAL_P) and not call_loss(LOSS_P):
        marginal.append("more calls")
    return {
        "fisher_p": fisher_p,
        "median_a": median_a,
        "median_b": median_b,
        "difference": difference,
        "outside_aa_band": outside,
        "mannwhitney_u_b": mw_u,
        "mannwhitney_p": mw_p,
        "loss": bool(reasons),
        "loss_reasons": reasons,
        "marginal": bool(marginal),
        "marginal_reasons": marginal,
    }


def self_check() -> None:
    if [order_for(index)[0] for index in range(8)] != list("ABBAABBA"):
        raise RuntimeError("ABBA pattern")
    if sequence_for("BA", ("A1", "A2")) != ("A2", "A1"):
        raise RuntimeError("order flip")
    if abs(fisher(9, 12, 6, 12) - 0.40032305828509895) > 1e-12:
        raise RuntimeError("Fisher check failed")
    if abs(fisher(6, 6, 6, 6) - 1.0) > 1e-12:
        raise RuntimeError("Fisher tie check failed")
    if fisher(12, 12, 0, 12) >= LOSS_P:
        raise RuntimeError("Fisher loss check failed")
    tied = mann_whitney([1, 2, 3], [1, 2, 3])
    if tied is None or abs(tied[1] - 1.0) > 1e-12:
        raise RuntimeError("Mann-Whitney tie check failed")
    shift = mann_whitney([1, 2, 3], [4, 5, 6])
    if shift is None or abs(shift[1] - 0.1) > 1e-12:
        raise RuntimeError("Mann-Whitney shift check failed")
    if abs(type7([1, 2, 3, 4], 0.5) - 2.5) > 1e-12:
        raise RuntimeError("type 7 check failed")
    quiet = judge(12, 12, 12, [100.0] * 12, [100.0] * 12, 0.0)
    if quiet["loss"] or quiet["marginal"] or abs(quiet["fisher_p"] - 1.0) > 1e-12:
        raise RuntimeError("quiet pair should pass")
    dropped = judge(12, 0, 12, [100.0] * 12, [], 0.0)
    if dropped["loss_reasons"] != ["fewer hits"]:
        raise RuntimeError("hit loss was not declared")
    costly = judge(12, 12, 12, [10.0] * 12, [20.0] * 12, 0.0)
    if "more calls" not in costly["loss_reasons"]:
        raise RuntimeError("call loss inside a zero band was not declared")
    inside = judge(12, 12, 12, [10.0] * 12, [20.0] * 12, 100.0)
    if inside["loss"] or inside["outside_aa_band"]:
        raise RuntimeError("a difference inside the A/A band was called a loss")
    shifted = [float(value) for value in range(12)]
    later = [float(value) for value in range(3, 15)]
    edge = judge(12, 12, 12, shifted, later, 0.0)
    if edge["loss"] or edge["marginal_reasons"] != ["more calls"]:
        raise RuntimeError("marginal call loss was not separated from a loss")
    covered = judge(12, 12, 12, shifted, later, 3.0)
    if covered["loss"] or covered["marginal"] or covered["outside_aa_band"]:
        raise RuntimeError("a difference equal to the A/A band was called outside it")
    interval = bootstrap_median_ci([1.0, 2.0, 3.0, 4.0], "self-check")
    if interval is None or not interval[0] <= 2.5 <= interval[1]:
        raise RuntimeError("bootstrap interval missed the median")


def git_sha(repo: Path, rev: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", rev],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def porcelain(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def machine_id() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    return hashlib.sha256(raw).hexdigest()


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name":
            return value.strip()
    return ""


def compiler_version() -> str:
    proc = subprocess.run(
        ["rustc", "+1.88.0", "-V"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def refuse_stray_environment() -> None:
    stray = sorted(key for key in os.environ if key.startswith(("H2H_", "PRIOR_", "RGPOT_")))
    if stray:
        raise RuntimeError(f"refusing to run with {', '.join(stray)} set")


def pin_thread_env() -> None:
    os.environ.update(THREAD_ENV)
    for key, value in THREAD_ENV.items():
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key} is not 1")


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(THREAD_ENV)
    return env


def shared_store_mounts() -> list[str]:
    mounts = []
    for line in Path("/proc/mounts").read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        if parts[2].startswith("fuse"):
            mounts.append(parts[1])
    return mounts


def assert_local_disk(path: Path) -> None:
    resolved = str(path.resolve())
    for mount in shared_store_mounts():
        root = mount.rstrip("/")
        if resolved == root or resolved.startswith(root + "/"):
            raise RuntimeError("refusing a path on a shared store mount")


def retry_io(action, label: str):
    delay = 0.05
    last = None
    for attempt in range(12):
        try:
            return action()
        except OSError as exc:
            last = exc
            if exc.errno not in TRANSIENT_ERRNO:
                raise
            IO_NOTES.append(label)
            if attempt == 11:
                break
            time.sleep(delay)
            delay = min(1.0, delay * 2)
    raise RuntimeError(f"transient write did not recover: {label}") from last


def write_bytes_retry(path: Path, payload: bytes) -> None:
    def op() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = f".{path.name}.{os.getpid()}.tmp"
        tmp = path.parent / tmp_name
        with tmp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    retry_io(op, path.name)


def repair_partial_tail(path: Path) -> bool:
    if not path.is_file():
        return False
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    cut = data.rfind(b"\n")
    kept = data[: cut + 1] if cut >= 0 else b""
    write_bytes_retry(path, kept)
    IO_NOTES.append(f"repaired {path.name}")
    return True


def binary(bin_dir: Path, driver: str) -> Path:
    return bin_dir / driver


def affinity_of(pid: int) -> str:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return ""
    for line in text.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    return ""


def pinned_affinity(core: int) -> str:
    proc = subprocess.run(
        [
            "taskset",
            "-c",
            str(core),
            sys.executable,
            "-c",
            "import os; print(','.join(str(item) for item in sorted(os.sched_getaffinity(0))))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def command(system: System, exe: Path, seed: int, scratch: Path) -> list[str]:
    if system.driver == "lj_shelf_gate":
        return [str(exe), str(system.n), str(seed), str(system.budget), f"{system.reference:.12f}"]
    if system.driver == "h2h_seed":
        prefix = scratch / "run"
        return [
            str(exe),
            system.kind,
            str(system.n),
            repr(system.rho),
            str(system.budget),
            str(seed),
            "recommended",
            repr(system.level),
            str(prefix),
        ]
    return [str(exe), "run", system.name, str(seed)]


def parse_output(system: System, text: str) -> dict:
    if system.driver == "lj_shelf_gate":
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) != 2 or not lines[0].startswith("n,seed,budget,hit,"):
            raise RuntimeError(f"unexpected runner output: {text!r}")
        fields = lines[1].split(",")
        if len(fields) != 9:
            raise RuntimeError(f"unexpected runner row: {lines[1]!r}")
        hit = fields[3]
        if hit not in {"0", "1"}:
            raise RuntimeError(f"bad hit flag {hit}")
        if fields[7] == "" or fields[8] == "":
            raise RuntimeError("runner omitted wall or potential time")
        return {
            "budget": int(fields[2]),
            "hit": int(hit),
            "evals_to_hit": int(fields[4]),
            "charged": int(fields[5]),
            "best": fields[6],
            "wall_s": fields[7],
            "pot_s": fields[8],
        }
    if system.driver == "h2h_seed":
        line = next(item for item in text.splitlines() if item.startswith("h2h_row "))
        fields = dict(part.split("=", 1) for part in line.split()[1:])
        hit = fields["hit"]
        if hit not in {"0", "1"}:
            raise RuntimeError(f"bad hit flag {hit}")
        if "pot_s" not in fields or "wall_s" not in fields:
            raise RuntimeError("seed driver omitted wall or potential time")
        return {
            "budget": int(fields["budget"]),
            "hit": int(hit),
            "evals_to_hit": int(fields["evals_to_hit"]),
            "charged": int(fields["charged"]),
            "best": fields["best"],
            "wall_s": fields["wall_s"],
            "pot_s": fields["pot_s"],
        }
    data = [line for line in text.splitlines() if ",anneal," in line]
    header = next(line for line in text.splitlines() if line.startswith("system,code,seed,"))
    if len(data) != 1:
        raise RuntimeError(f"expected one anneal row, saw {len(data)}")
    row = dict(zip(header.split(","), data[0].split(",")))
    hit = row["hit"].lower()
    if hit in {"true", "1"}:
        flag = 1
    elif hit in {"false", "0"}:
        flag = 0
    else:
        raise RuntimeError(f"bad hit flag {row['hit']}")
    if "pot_s" not in row or "wall_s" not in row:
        raise RuntimeError("driver omitted wall or potential time")
    return {
        "budget": int(row["budget"]),
        "hit": flag,
        "evals_to_hit": int(row["evals_to_hit"]),
        "charged": int(row["charged"]),
        "best": row["best"],
        "wall_s": row["wall_s"],
        "pot_s": row["pot_s"],
    }


def check_parsed(system: System, parsed: dict) -> None:
    if parsed["budget"] != system.budget:
        raise RuntimeError(
            f"{system.name}: driver budget {parsed['budget']} is not {system.budget}"
        )
    if parsed["charged"] > system.budget or parsed["evals_to_hit"] > system.budget:
        raise RuntimeError(f"{system.name}: calls exceed the budget")
    if parsed["hit"] == 0 and parsed["evals_to_hit"] != system.budget:
        raise RuntimeError(f"{system.name}: a miss was not entered at the budget")
    if parsed["hit"] == 1 and parsed["evals_to_hit"] > parsed["charged"]:
        raise RuntimeError(f"{system.name}: hit calls exceed charged calls")


def verify_drivers(protocol: dict, bin_a: Path, bin_b: Path) -> None:
    for system in protocol["systems"]:
        for directory in (bin_a, bin_b):
            exe = binary(directory, system.driver)
            if not exe.is_file():
                raise RuntimeError(f"{exe.name} is missing")
        if system.driver not in {"h2h_morse", "h2h_metal"}:
            continue
        for directory in (bin_a, bin_b):
            proc = subprocess.run(
                [str(binary(directory, system.driver)), "budget", system.name],
                check=True,
                capture_output=True,
                text=True,
                env=child_env(),
            )
            fields = proc.stdout.split()
            if int(fields[0]) != system.budget or abs(float(fields[1]) - system.reference) > 1e-6:
                raise RuntimeError(
                    f"{system.name}: driver says {fields[0]} {fields[1]}, "
                    f"the gate declares {system.budget} and {system.reference}"
                )


def verify_starts(protocol: dict, bin_a: Path, bin_b: Path) -> int:
    checked = 0
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        for system in protocol["systems"]:
            if system.driver not in {"h2h_morse", "h2h_metal"}:
                continue
            for seed in protocol["seeds"]:
                left = root / "a"
                right = root / "b"
                for directory, path in ((bin_a, left), (bin_b, right)):
                    proc = subprocess.run(
                        [
                            str(binary(directory, system.driver)),
                            "coords",
                            system.name,
                            str(seed),
                            str(path),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        env=child_env(),
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"{system.name} seed {seed} coords exited {proc.returncode}: "
                            f"{proc.stderr[-400:]}"
                        )
                if left.read_bytes() != right.read_bytes():
                    raise RuntimeError(f"{system.name} seed {seed} starts differ")
                checked += 1
    return checked


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, fields: list[str], row: dict) -> None:
    def op() -> None:
        new_file = not path.is_file() or path.stat().st_size == 0
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            if new_file:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())

    retry_io(op, path.name)


def append_jsonl(path: Path, row: dict) -> None:
    text = json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"

    def op() -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    retry_io(op, path.name)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def session_id(protocol: dict, arms: dict) -> str:
    payload = {
        "seeds": list(protocol["seeds"]),
        "systems": [
            [system.name, system.budget, system.reference] for system in protocol["systems"]
        ],
        "aa": list(AA_SYSTEMS),
        "arms": {name: value["tree"] for name, value in arms.items()},
        "order": "ABBA",
        "host": HOST_LABEL,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def arm_record(src: Path, bin_dir: Path, drivers: list[str]) -> dict:
    commit = git_sha(src, "HEAD")
    tree = git_sha(src, "HEAD^{tree}")
    status = porcelain(src)
    if status.strip():
        raise RuntimeError(f"arm worktree is not clean:\n{status}")
    binaries = {
        driver: file_sha256(binary(bin_dir, driver))
        for driver in drivers
    }
    return {
        "commit": commit,
        "tree": tree,
        "porcelain": "",
        "clean": True,
        "binaries": binaries,
    }


def jobs(protocol: dict) -> list[dict]:
    planned = []
    for index, seed in enumerate(protocol["seeds"]):
        order = order_for(index)
        for system in protocol["systems"]:
            for arm in sequence_for(order, ("A", "B")):
                planned.append(
                    {
                        "system": system,
                        "comparison": "AB",
                        "arm": arm,
                        "seed": seed,
                        "seed_index": index,
                        "order": order,
                    }
                )
            if system.name in AA_SYSTEMS:
                for arm in sequence_for(order, ("A1", "A2")):
                    planned.append(
                        {
                            "system": system,
                            "comparison": "AA",
                            "arm": arm,
                            "seed": seed,
                            "seed_index": index,
                            "order": order,
                        }
                    )
    return planned


def run_one(
    job: dict,
    bin_dir: Path,
    core: int,
    revision: str,
    tree: str,
    host: str,
    machine: str,
    log_dir: Path,
) -> dict:
    system: System = job["system"]
    exe = binary(bin_dir, system.driver)
    log_path = log_dir / f"{job['comparison']}_{system.name}_{job['arm']}_{job['seed']}.log"
    with tempfile.TemporaryDirectory() as scratch:
        argv = [
            "taskset",
            "-c",
            str(core),
            *command(system, exe, job["seed"], Path(scratch)),
        ]
        proc = subprocess.Popen(
            argv,
            cwd=scratch,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=child_env(),
        )
        affinity = ""
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            current = affinity_of(proc.pid)
            if current == str(core):
                affinity = current
                break
            if current:
                affinity = current
            if proc.poll() is not None:
                break
            time.sleep(0.01)
        stdout, stderr = proc.communicate()
    payload = stdout + ("\n" if stdout and stderr else "") + stderr
    write_bytes_retry(log_path, payload.encode())
    if proc.returncode != 0:
        raise RuntimeError(
            f"{system.name} {job['comparison']} {job['arm']} seed {job['seed']} "
            f"exited {proc.returncode}: {stderr[-800:]}"
        )
    if affinity != str(core):
        raise RuntimeError(
            f"{system.name} seed {job['seed']} affinity {affinity!r} is not core {core}"
        )
    parsed = parse_output(system, stdout)
    check_parsed(system, parsed)
    return {
        "system": system.name,
        "family": system.family,
        "comparison": job["comparison"],
        "arm": job["arm"],
        "seed": str(job["seed"]),
        "seed_index": str(job["seed_index"]),
        "order": job["order"],
        "budget": str(parsed["budget"]),
        "reference": f"{system.reference:.10f}",
        "hit": str(parsed["hit"]),
        "evals_to_hit": str(parsed["evals_to_hit"]),
        "charged": str(parsed["charged"]),
        "best": parsed["best"],
        "wall_s": parsed["wall_s"],
        "pot_s": parsed["pot_s"],
        "core": str(core),
        "affinity": affinity,
        "machine_id": machine,
        "host": host,
        "code_revision": revision,
        "tree": tree,
    }


def numeric(value: str) -> float:
    return float(value)


def outside_s(row: dict) -> float:
    if row.get("pot_s", "") == "" or row.get("wall_s", "") == "":
        raise RuntimeError(f"{row['system']} has no potential time")
    return max(0.0, numeric(row["wall_s"]) - numeric(row["pot_s"]))


def rows_for(rows: list[dict], system: str, comparison: str, arm: str, seeds: list[int]) -> list[dict] | None:
    found = {}
    for row in rows:
        if row["system"] == system and row["comparison"] == comparison and row["arm"] == arm:
            found[int(row["seed"])] = row
    if [seed for seed in seeds if seed not in found]:
        return None
    extra = [seed for seed in found if seed not in seeds]
    if extra:
        raise RuntimeError(f"{system} {comparison} {arm} has unexpected seeds {extra}")
    return [found[seed] for seed in seeds]


def audit_rows(rows: list[dict], protocol: dict, host: str) -> None:
    seeds = list(protocol["seeds"])
    for row in rows:
        index = int(row["seed_index"])
        if row["order"] != order_for(index):
            raise RuntimeError(f"{row['system']} seed {row['seed']} broke the ABBA order")
        if int(row["seed"]) != seeds[index]:
            raise RuntimeError(f"{row['system']} seed index does not match the seed")
        if row["host"] != host:
            raise RuntimeError(f"{row['system']} recorded host {row['host']}")
        if row["affinity"] != row["core"]:
            raise RuntimeError(f"{row['system']} affinity does not match its core")
    cores = {row["core"] for row in rows}
    machines = {row["machine_id"] for row in rows}
    if len(cores) != 1 or len(machines) != 1:
        raise RuntimeError("rows do not share one core and one machine id")


def summarise(rows: list[dict], protocol: dict, session: dict) -> dict:
    seeds = list(protocol["seeds"])
    audit_rows(rows, protocol, session["host"])
    aa = []
    deltas = []
    for name in AA_SYSTEMS:
        system = next(item for item in protocol["systems"] if item.name == name)
        left = rows_for(rows, name, "AA", "A1", seeds)
        right = rows_for(rows, name, "AA", "A2", seeds)
        if left is None or right is None:
            raise RuntimeError(f"{name} A/A rows are incomplete")
        evals_1 = [numeric(row["evals_to_hit"]) for row in left if row["hit"] == "1"]
        evals_2 = [numeric(row["evals_to_hit"]) for row in right if row["hit"] == "1"]
        if not evals_1 or not evals_2:
            raise RuntimeError(f"{name} A/A produced no hitting runs to set the band")
        median_1 = type7(evals_1, 0.5)
        median_2 = type7(evals_2, 0.5)
        delta = median_2 - median_1
        deltas.append(delta)
        hits_1 = sum(row["hit"] == "1" for row in left)
        hits_2 = sum(row["hit"] == "1" for row in right)
        mw = mann_whitney(evals_2, evals_1)
        aa.append(
            {
                "system": name,
                "budget": system.budget,
                "n": len(seeds),
                "hits_a1": hits_1,
                "hits_a2": hits_2,
                "fisher_p": fisher(hits_1, len(seeds), hits_2, len(seeds)),
                "median_a1": median_1,
                "median_a2": median_2,
                "ci_a1": bootstrap_median_ci(evals_1, f"{name}:AA:A1"),
                "ci_a2": bootstrap_median_ci(evals_2, f"{name}:AA:A2"),
                "difference": delta,
                "mannwhitney_p": None if mw is None else mw[1],
                "total_calls_a1": sum(int(row["charged"]) for row in left),
                "total_calls_a2": sum(int(row["charged"]) for row in right),
            }
        )
    aa_band = max(deltas)
    systems_out = []
    losses = []
    marginal = []
    for system in protocol["systems"]:
        left = rows_for(rows, system.name, "AB", "A", seeds)
        right = rows_for(rows, system.name, "AB", "B", seeds)
        if left is None or right is None:
            raise RuntimeError(f"{system.name} A/B rows are incomplete")
        for one, other in zip(left, right):
            if one["order"] != other["order"] or one["core"] != other["core"]:
                raise RuntimeError(f"{system.name} seed {one['seed']} split the pair")
        evals_a = [numeric(row["evals_to_hit"]) for row in left if row["hit"] == "1"]
        evals_b = [numeric(row["evals_to_hit"]) for row in right if row["hit"] == "1"]
        hits_a = sum(row["hit"] == "1" for row in left)
        hits_b = sum(row["hit"] == "1" for row in right)
        decision = judge(hits_a, hits_b, len(seeds), evals_a, evals_b, aa_band)
        entry = {
            "system": system.name,
            "family": system.family,
            "budget": system.budget,
            "reference": system.reference,
            "n": len(seeds),
            "hits_a": hits_a,
            "hits_b": hits_b,
            "fisher_p": decision["fisher_p"],
            "calls_a": {
                "n": len(evals_a),
                "median": decision["median_a"],
                "ci95": bootstrap_median_ci(evals_a, f"{system.name}:AB:A"),
            },
            "calls_b": {
                "n": len(evals_b),
                "median": decision["median_b"],
                "ci95": bootstrap_median_ci(evals_b, f"{system.name}:AB:B"),
            },
            "mannwhitney_u_b": decision["mannwhitney_u_b"],
            "mannwhitney_p": decision["mannwhitney_p"],
            "median_difference": decision["difference"],
            "outside_aa_band": decision["outside_aa_band"],
            "total_calls_a": sum(int(row["charged"]) for row in left),
            "total_calls_b": sum(int(row["charged"]) for row in right),
            "outside_s_a": sum(outside_s(row) for row in left),
            "outside_s_b": sum(outside_s(row) for row in right),
            "loss": decision["loss"],
            "loss_reasons": decision["loss_reasons"],
            "marginal": decision["marginal"],
            "marginal_reasons": decision["marginal_reasons"],
        }
        systems_out.append(entry)
        if decision["loss"]:
            losses.append(
                {
                    "system": system.name,
                    "reasons": decision["loss_reasons"],
                    "fisher_p": decision["fisher_p"],
                    "mannwhitney_p": decision["mannwhitney_p"],
                }
            )
        if decision["marginal"]:
            marginal.append(
                {
                    "system": system.name,
                    "reasons": decision["marginal_reasons"],
                    "fisher_p": decision["fisher_p"],
                    "mannwhitney_p": decision["mannwhitney_p"],
                }
            )
    verdict = "PASS" if not losses else "FAIL"
    return {
        "rule": RULE,
        "verdict": verdict,
        "alpha": LOSS_P,
        "marginal_alpha": MARGINAL_P,
        "aa_band_calls": aa_band,
        "aa": aa,
        "losses": losses,
        "marginal": marginal,
        "host": session["host"],
        "machine_id": session["machine_id"],
        "core": session["core"],
        "affinity": session["affinity"],
        "compiler": session["compiler"],
        "cpu": session["cpu"],
        "seeds": seeds,
        "order": "ABBA",
        "hit_tolerance": HIT_TOL,
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "interval": 0.95},
        "arms": session["arms"],
        "starts_checked": session.get("starts_checked", 0),
        "systems": systems_out,
        "total_calls_a": sum(item["total_calls_a"] for item in systems_out),
        "total_calls_b": sum(item["total_calls_b"] for item in systems_out),
        "outside_s_a": sum(item["outside_s_a"] for item in systems_out),
        "outside_s_b": sum(item["outside_s_b"] for item in systems_out),
        "io_resumes": int(session.get("io_resumes", 0)),
    }


def fmt_p(value: float | None) -> str:
    if value is None:
        return ""
    if value < 1e-4:
        return f"{value:.1e}"
    return f"{value:.4f}"


def fmt_num(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}"


def fmt_ci(summary: dict) -> str:
    median = fmt_num(summary["median"])
    if not median:
        return ""
    interval = summary["ci95"]
    if not interval:
        return median
    return f"{median} [{fmt_num(interval[0])}, {fmt_num(interval[1])}]"


def markdown(report: dict) -> str:
    lines = [
        "# Shelf quarantine paired gate",
        "",
        report["rule"],
        "",
        f"Verdict: **{report['verdict']}**.",
        "",
        "## Record",
        "",
        f"- Host label: {report['host']}",
        f"- Machine id: `{report['machine_id']}`",
        f"- Core {report['core']}, affinity {report['affinity']}",
        f"- Compiler: {report['compiler']}",
        f"- CPU: {report['cpu']}",
        f"- Arm A commit `{report['arms']['A']['commit']}` tree `{report['arms']['A']['tree']}`",
        f"- Arm B commit `{report['arms']['B']['commit']}` tree `{report['arms']['B']['tree']}`",
        "- Porcelain was empty on both arm worktrees before the session.",
        f"- Seeds: {' '.join(str(seed) for seed in report['seeds'])}.",
        "- Schedule: one run at a time on the pinned core, with every thread variable set to 1. Within a seed, systems run in panel order and the two arms run back to back. The first arm follows ABBA.",
        f"- A/A band: {fmt_num(report['aa_band_calls'])} calls, from LJ38 and LJ75.",
        "- A hit is a quenched energy at or below the reference plus 0.001. A miss is entered at the budget. Medians use hitting runs only.",
        f"- Bootstrap: {report['bootstrap']['n']} resamples, percentile interval {report['bootstrap']['interval']}.",
        f"- Starts compared on the Morse and gold drivers: {report['starts_checked']} seed pairs.",
        "",
        "## A/A band",
        "",
        "| system | hits A1 | hits A2 | Fisher p | median A1 (95% CI) | median A2 (95% CI) | difference | MW p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in report["aa"]:
        lines.append(
            "| {system} | {h1}/{n} | {h2}/{n} | {fp} | {m1} [{c1}] | {m2} [{c2}] | {diff} | {mp} |".format(
                system=system["system"],
                n=system["n"],
                h1=system["hits_a1"],
                h2=system["hits_a2"],
                fp=fmt_p(system["fisher_p"]),
                m1=fmt_num(system["median_a1"]),
                c1=", ".join(fmt_num(item) for item in system["ci_a1"]),
                m2=fmt_num(system["median_a2"]),
                c2=", ".join(fmt_num(item) for item in system["ci_a2"]),
                diff=fmt_num(system["difference"]),
                mp=fmt_p(system["mannwhitney_p"]),
            )
        )
    lines.extend(
        [
            "",
            "## Systems",
            "",
            "| system | family | budget | hits A | hits B | Fisher p | median calls A (95% CI) | median calls B (95% CI) | MW p | total calls A | total calls B | outside s A | outside s B | loss |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for system in report["systems"]:
        loss = ", ".join(system["loss_reasons"])
        lines.append(
            "| {system} | {family} | {budget} | {ha}/{n} | {hb}/{n} | {fp} | {ma} | {mb} | {mp} | {ta} | {tb} | {oa} | {ob} | {loss} |".format(
                system=system["system"],
                family=system["family"],
                budget=system["budget"],
                n=system["n"],
                ha=system["hits_a"],
                hb=system["hits_b"],
                fp=fmt_p(system["fisher_p"]),
                ma=fmt_ci(system["calls_a"]),
                mb=fmt_ci(system["calls_b"]),
                mp=fmt_p(system["mannwhitney_p"]),
                ta=system["total_calls_a"],
                tb=system["total_calls_b"],
                oa=f"{system['outside_s_a']:.6f}",
                ob=f"{system['outside_s_b']:.6f}",
                loss=loss,
            )
        )
    lines.extend(
        [
            "",
            f"Total charged calls: A {report['total_calls_a']}, B {report['total_calls_b']}.",
            f"Outside-potential time: A {report['outside_s_a']:.6f} s, B {report['outside_s_b']:.6f} s.",
            "",
        ]
    )
    if report.get("io_resumes"):
        lines.extend(
            [
                f"The runner continued the same session after {report['io_resumes']} transient write failures.",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Losses",
            "",
        ]
    )
    if not report["losses"]:
        lines.append("None.")
    for item in report["losses"]:
        lines.append(
            f"- {item['system']}: {', '.join(item['reasons'])}; "
            f"Fisher p {fmt_p(item['fisher_p'])}; Mann-Whitney p {fmt_p(item['mannwhitney_p'])}"
        )
    lines.extend(["", "## Marginal losses", ""])
    if not report["marginal"]:
        lines.append("None.")
    for item in report["marginal"]:
        lines.append(
            f"- {item['system']}: {', '.join(item['reasons'])}; "
            f"Fisher p {fmt_p(item['fisher_p'])}; Mann-Whitney p {fmt_p(item['mannwhitney_p'])}"
        )
    lines.extend(["", f"Verdict: **{report['verdict']}**.", ""])
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    )
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, path)
    except Exception:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)
        raise


def rewrite_hosts(out: Path) -> None:
    rows = read_rows(out / "rows.csv")
    session = load_session(out / "sessions.jsonl")["session"]
    path = out / "hosts.csv"
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=out, delete=False, prefix=".hosts."
    )
    try:
        writer = csv.DictWriter(handle, fieldnames=HOST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "session": session,
                    "host": row["host"],
                    "machine_id": row["machine_id"],
                    "core": row["core"],
                    "affinity": row["affinity"],
                    "system": row["system"],
                    "comparison": row["comparison"],
                    "arm": row["arm"],
                    "seed": row["seed"],
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, path)
    except Exception:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)
        raise


def load_session(path: Path) -> dict:
    rows = [row for row in read_jsonl(path) if row.get("kind") == "session"]
    if len(rows) != 1:
        raise RuntimeError(f"expected one session record, found {len(rows)}")
    return rows[0]


def resume_count(path: Path) -> int:
    return sum(1 for row in read_jsonl(path) if row.get("kind") == "resume")


def note_resume(path: Path, ident: str, detail: str) -> None:
    append_jsonl(
        path,
        {
            "kind": "resume",
            "session": ident,
            "detail": detail,
        },
    )


def emit_report(out: Path, protocol: dict) -> dict:
    rows = read_rows(out / "rows.csv")
    session = load_session(out / "sessions.jsonl")
    session["io_resumes"] = resume_count(out / "sessions.jsonl")
    report = summarise(rows, protocol, session)
    atomic_write(out / "summary.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_write(out / "verdict.md", markdown(report))
    print(f"verdict {report['verdict']}", flush=True)
    return report


def cmd_self_check(_args: argparse.Namespace) -> int:
    self_check()
    print("self-check ok")
    return 0


def cmd_systems(args: argparse.Namespace) -> int:
    self_check()
    protocol = load_protocol(Path(args.repo))
    print(f"seeds {protocol['seeds'][0]} to {protocol['seeds'][-1]} ({len(protocol['seeds'])})")
    print(f"systems {len(protocol['systems'])}")
    print(f"jobs {len(jobs(protocol))}")
    for system in protocol["systems"]:
        print(
            f"{system.name}\t{system.family}\t{system.driver}\t"
            f"budget {system.budget}\treference {system.reference}"
        )
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    src = Path(args.src)
    runner = Path(args.runner)
    target = Path(args.target_dir)
    example = src / "examples" / "lj_shelf_gate.rs"
    if example.exists():
        raise RuntimeError(f"{example.name} already exists in the source tree")
    example.write_bytes(runner.read_bytes())
    try:
        subprocess.run(
            [
                args.cargo,
                f"+{args.toolchain}",
                "build",
                "--release",
                "--locked",
                "--example",
                "lj_shelf_gate",
                "--example",
                "h2h_morse",
                "--example",
                "h2h_metal",
                "--example",
                "h2h_seed",
                "-j",
                str(args.jobs),
                "--target-dir",
                str(target),
            ],
            cwd=src,
            check=True,
        )
    finally:
        example.unlink(missing_ok=True)
    if porcelain(src).strip():
        raise RuntimeError("build left the source tree dirty")
    print(git_sha(src, "HEAD"))
    print(git_sha(src, "HEAD^{tree}"))
    return 0


def pending_path(out: Path, row: dict) -> Path:
    name = f"{row['system']}_{row['comparison']}_{row['arm']}_{row['seed']}.json"
    return out / "pending" / name


def remember_row(out: Path, row: dict) -> None:
    write_bytes_retry(
        pending_path(out, row),
        (json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n").encode(),
    )


def recorded_keys(path: Path) -> set[tuple]:
    return {
        (row["system"], row["comparison"], row["arm"], int(row["seed"]))
        for row in read_rows(path)
        if row.get("system") and row.get("seed")
    }


def recorded_runs(path: Path) -> set[tuple]:
    found = set()
    for row in read_jsonl(path):
        if row.get("kind") != "run":
            continue
        found.add((row["system"], row["comparison"], row["arm"], int(row["seed"])))
    return found


def commit_row(out: Path, sessions: Path, ident: str, row: dict, done: set) -> None:
    key = (row["system"], row["comparison"], row["arm"], int(row["seed"]))
    if key not in recorded_keys(out / "rows.csv"):
        append_csv(out / "rows.csv", ROW_FIELDS, row)
    if key not in recorded_keys(out / "hosts.csv"):
        append_csv(
            out / "hosts.csv",
            HOST_FIELDS,
            {
                "session": ident,
                "host": row["host"],
                "machine_id": row["machine_id"],
                "core": row["core"],
                "affinity": row["affinity"],
                "system": row["system"],
                "comparison": row["comparison"],
                "arm": row["arm"],
                "seed": row["seed"],
            },
        )
    if key not in recorded_runs(sessions):
        append_jsonl(
            sessions,
            {
                "kind": "run",
                "session": ident,
                "system": row["system"],
                "comparison": row["comparison"],
                "arm": row["arm"],
                "seed": int(row["seed"]),
                "order": row["order"],
                "core": int(row["core"]),
                "affinity": row["affinity"],
                "machine_id": row["machine_id"],
                "host": row["host"],
                "hit": int(row["hit"]),
                "evals_to_hit": int(row["evals_to_hit"]),
                "charged": int(row["charged"]),
                "best": row["best"],
                "wall_s": row["wall_s"],
                "code_revision": row["code_revision"],
                "tree": row["tree"],
            },
        )
    done.add(key)
    path = pending_path(out, row)
    if path.is_file():
        retry_io(path.unlink, path.name)


def flush_pending(out: Path, sessions: Path, ident: str, done: set) -> int:
    directory = out / "pending"
    if not directory.is_dir():
        return 0
    flushed = 0
    for path in sorted(directory.glob("*.json")):
        row = json.loads(path.read_text())
        key = (row["system"], row["comparison"], row["arm"], int(row["seed"]))
        before = key in done
        commit_row(out, sessions, ident, row, done)
        if not before:
            flushed += 1
    return flushed


def cmd_run(args: argparse.Namespace) -> int:
    self_check()
    if args.host != HOST_LABEL:
        raise RuntimeError(f"host label must be {HOST_LABEL}")
    refuse_stray_environment()
    pin_thread_env()
    repo = Path(args.repo)
    protocol = load_protocol(repo)
    bin_a = Path(args.bin_a)
    bin_b = Path(args.bin_b)
    out = Path(args.out)
    log_dir = Path(args.log_dir)
    assert_local_disk(out)
    assert_local_disk(log_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    drivers = sorted({system.driver for system in protocol["systems"]})
    arms = {
        "A": arm_record(Path(args.src_a), bin_a, drivers),
        "B": arm_record(Path(args.src_b), bin_b, drivers),
    }
    if arms["A"]["commit"] != ARM_A_COMMIT:
        raise RuntimeError(f"arm A commit is {arms['A']['commit']}")
    if arms["B"]["tree"] != ARM_B_TREE:
        raise RuntimeError(f"arm B tree is {arms['B']['tree']}")
    if arms["A"]["tree"] == arms["B"]["tree"]:
        raise RuntimeError("arms share a tree")
    verify_drivers(protocol, bin_a, bin_b)
    checked = verify_starts(protocol, bin_a, bin_b)
    affinity = pinned_affinity(args.core)
    if affinity != str(args.core):
        raise RuntimeError(f"pinned affinity {affinity} is not core {args.core}")
    machine = machine_id()
    ident = session_id(protocol, arms)
    sessions = out / "sessions.jsonl"
    repaired = [
        name
        for name in ("rows.csv", "hosts.csv", "sessions.jsonl")
        if repair_partial_tail(out / name)
    ]
    existing = [row for row in read_jsonl(sessions) if row.get("kind") == "session"]
    session = {
        "kind": "session",
        "session": ident,
        "host": args.host,
        "machine_id": machine,
        "core": args.core,
        "affinity": affinity,
        "compiler": compiler_version(),
        "cpu": cpu_model(),
        "seeds": list(protocol["seeds"]),
        "order": "ABBA",
        "threads": 1,
        "aa_systems": list(AA_SYSTEMS),
        "arms": arms,
        "starts_checked": checked,
        "started": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if existing:
        prior = existing[0]
        if prior["session"] != ident or prior["machine_id"] != machine:
            raise RuntimeError("existing session does not match this arm pair")
        session = prior
        session["starts_checked"] = checked
    else:
        append_jsonl(sessions, session)
    if repaired:
        note_resume(sessions, ident, "repaired a partial row and continued the same session")
    done = {
        (row["system"], row["comparison"], row["arm"], int(row["seed"]))
        for row in read_rows(out / "rows.csv")
    }
    flushed = flush_pending(out, sessions, ident, done)
    if flushed:
        note_resume(sessions, ident, f"flushed {flushed} completed rows")
    planned = jobs(protocol)
    revisions = {
        "A": arms["A"],
        "A1": arms["A"],
        "A2": arms["A"],
        "B": arms["B"],
    }
    directories = {"A": bin_a, "A1": bin_a, "A2": bin_a, "B": bin_b}
    pending = [
        job
        for job in planned
        if (job["system"].name, job["comparison"], job["arm"], job["seed"]) not in done
    ]
    print(
        f"session {ident} jobs {len(planned)} pending {len(pending)} starts {checked}",
        flush=True,
    )
    started = time.monotonic()
    for number, job in enumerate(pending, start=1):
        arm_meta = revisions[job["arm"]]
        key = (job["system"].name, job["comparison"], job["arm"], job["seed"])
        attempts = 0
        while key not in done:
            attempts += 1
            notes_before = len(IO_NOTES)
            try:
                row = run_one(
                    job,
                    directories[job["arm"]],
                    args.core,
                    arm_meta["commit"],
                    arm_meta["tree"],
                    args.host,
                    machine,
                    log_dir,
                )
                remember_row(out, row)
                commit_row(out, sessions, ident, row, done)
            except OSError as exc:
                if exc.errno not in TRANSIENT_ERRNO or attempts >= 5:
                    raise
                note_resume(sessions, ident, "continued the same session")
                done = {
                    (item["system"], item["comparison"], item["arm"], int(item["seed"]))
                    for item in read_rows(out / "rows.csv")
                }
                flush_pending(out, sessions, ident, done)
                continue
            except RuntimeError as exc:
                if "transient write did not recover" not in str(exc) or attempts >= 5:
                    raise
                note_resume(sessions, ident, "continued the same session")
                done = {
                    (item["system"], item["comparison"], item["arm"], int(item["seed"]))
                    for item in read_rows(out / "rows.csv")
                }
                flush_pending(out, sessions, ident, done)
                continue
            if len(IO_NOTES) > notes_before:
                note_resume(sessions, ident, "continued the same session")
            break
        elapsed = time.monotonic() - started
        recorded = next(
            item
            for item in read_rows(out / "rows.csv")
            if (item["system"], item["comparison"], item["arm"], int(item["seed"])) == key
        )
        print(
            f"done {number}/{len(pending)} {recorded['system']} {recorded['comparison']} "
            f"{recorded['arm']} seed {recorded['seed']} hit {recorded['hit']} "
            f"calls {recorded['evals_to_hit']} wall {recorded['wall_s']} "
            f"elapsed {elapsed:.0f}s",
            flush=True,
        )
    rewrite_hosts(out)
    report = emit_report(out, protocol)
    return 0 if report["verdict"] == "PASS" else 1


def cmd_report(args: argparse.Namespace) -> int:
    self_check()
    protocol = load_protocol(Path(args.repo))
    report = emit_report(Path(args.out), protocol)
    return 0 if report["verdict"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("self-check")
    check.set_defaults(func=cmd_self_check)

    systems = sub.add_parser("systems")
    systems.add_argument("--repo", required=True)
    systems.set_defaults(func=cmd_systems)

    build = sub.add_parser("build")
    build.add_argument("--src", required=True)
    build.add_argument("--runner", required=True)
    build.add_argument("--target-dir", required=True)
    build.add_argument("--cargo", default="cargo")
    build.add_argument("--toolchain", default="1.88.0")
    build.add_argument("--jobs", type=int, default=8)
    build.set_defaults(func=cmd_build)

    run = sub.add_parser("run")
    run.add_argument("--repo", required=True)
    run.add_argument("--src-a", required=True)
    run.add_argument("--src-b", required=True)
    run.add_argument("--bin-a", required=True)
    run.add_argument("--bin-b", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--log-dir", required=True)
    run.add_argument("--host", required=True)
    run.add_argument("--core", type=int, default=0)
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report")
    report.add_argument("--repo", required=True)
    report.add_argument("--out", required=True)
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"gate: {exc}", file=sys.stderr)
        sys.exit(1)
