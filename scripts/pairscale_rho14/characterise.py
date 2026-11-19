#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Pre-declared comparison of the pair-scaled Morse preset on rho = 14.

Arm A is the default preset at the base commit, with no ``H2H_`` variable
set. Arm B is the tree that adds the opt-in pair scaling, run with
``H2H_PAIR_SCALE=1`` and no other ``H2H_`` variable. The cells are
``m55r14`` (budget 1_000_000, reference -220.646208) and ``m80r14``
(budget 4_000_000, reference -340.811371). A hit is a finite best energy
at or below the reference plus 0.001.

Seeds are 2100 through 2123, twenty-four per arm per cell. Both arms of a
seed receive the geometry arm A writes for that seed. The order inside
each cell is ABBA: an even offset from 2100 runs A then B, an odd offset
runs B then A. Slot 0 is the first arm of a seed. Each ``(slot, arm)``
pair is counted and the run stops unless every pair is balanced.

Runs are serial. The process tree is pinned to core 0 and every thread-pool
variable is 1. Binaries are release builds. There is no shortened budget
and no warm-up search.

The declared tests, per cell, are a two-sided exact Fisher test on hits
and a two-sided Mann-Whitney test on calls to the hit. A miss is ranked
worse than every hit. Bonferroni uses m = 2 inside each test family, so
the threshold is 0.025. Medians are Hyndman-Fan type 7.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

BASE_COMMIT = "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"
BASE_TREE = "7a2ba49b37241a8fe42b2c45c226374de6d92274"
ARM_B_TREE = "70871f7307271806926b2aec9870e838dffbe924"
HIT_TOL = 1.0e-3
SEEDS = tuple(range(2100, 2124))
CELLS = (
    ("m55r14", 1_000_000, -220.646208),
    ("m80r14", 4_000_000, -340.811371),
)
CELL_BY_NAME = {name: (budget, reference) for name, budget, reference in CELLS}
ARMS = ("A", "B")
ALPHA = 0.025

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "TBB_NUM_THREADS",
    "JULIA_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "TOKIO_WORKER_THREADS",
)

ROW_FIELDS = (
    "cell",
    "arm",
    "seed",
    "slot",
    "core",
    "hit",
    "calls",
    "charged",
    "wall",
    "best",
    "best_bits",
    "budget",
    "machine",
    "base_commit",
    "base_tree",
)
SESSION_FIELDS = ROW_FIELDS + (
    "session",
    "order_index",
    "reference",
    "arm_commit",
    "arm_tree",
    "start_sha256",
    "pair_scale",
    "rustc",
)

HEADER = "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s"
WALL_TOKEN = re.compile(r"^-?\d+\.\d{6}$")
BEST_TOKEN = re.compile(r"^-?\d+\.\d{8}$")


def arm_sequence(seed: int) -> tuple[str, str]:
    """ABBA. Even offsets run A then B. Odd offsets run B then A."""
    if (seed - SEEDS[0]) % 2 == 0:
        return ("A", "B")
    return ("B", "A")


def schedule() -> list[dict]:
    jobs = []
    for cell, budget, reference in CELLS:
        for seed in SEEDS:
            for slot, arm in enumerate(arm_sequence(seed)):
                jobs.append(
                    {
                        "cell": cell,
                        "budget": budget,
                        "reference": reference,
                        "seed": seed,
                        "slot": slot,
                        "arm": arm,
                    }
                )
    return jobs


def pair_counts(jobs: list[dict]) -> Counter:
    """``(cell, slot, arm)`` counts. Slot 0 is the first arm of a seed."""
    return Counter((job["cell"], job["slot"], job["arm"]) for job in jobs)


def require_balance(jobs: list[dict]) -> None:
    if len(SEEDS) != 24 or len(jobs) != len(CELLS) * len(SEEDS) * 2:
        raise SystemExit("schedule length")
    opening = [job["arm"] for job in jobs if job["cell"] == "m55r14"][:4]
    if opening != ["A", "B", "B", "A"]:
        raise SystemExit(f"ABBA opening {opening}")
    counts = pair_counts(jobs)
    for cell, _budget, _reference in CELLS:
        for slot in (0, 1):
            for arm in ARMS:
                found = counts[(cell, slot, arm)]
                if found != len(SEEDS) // 2:
                    raise SystemExit(f"unbalanced {(cell, slot, arm)} = {found}")
    pooled = Counter((slot, arm) for (_cell, slot, arm) in counts.elements())
    for slot in (0, 1):
        for arm in ARMS:
            if pooled[(slot, arm)] != len(CELLS) * len(SEEDS) // 2:
                raise SystemExit(f"pooled {(slot, arm)} = {pooled[(slot, arm)]}")


def type7(values: list[float], probability: float = 0.5) -> float:
    """Hyndman-Fan type 7 quantile."""
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("type 7 needs a sample")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * probability
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def fisher_exact(hits_a: int, n_a: int, hits_b: int, n_b: int) -> tuple[int, int]:
    """Two-sided exact Fisher as ``(numerator, denominator)``.

    The denominator is the number of ways to place the observed hits. The
    numerator sums the placements whose probability is at most the observed
    probability. Comparison uses the integer cross product, so ties of
    probability are exact.
    """
    if not (0 <= hits_a <= n_a and 0 <= hits_b <= n_b):
        raise ValueError("fisher counts")
    hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, hits)

    def numer(k: int) -> int:
        return math.comb(n_a, k) * math.comb(n_b, hits - k)

    observed = numer(hits_a)
    lo = max(0, hits - n_b)
    hi = min(n_a, hits)
    total = sum(numer(k) for k in range(lo, hi + 1) if numer(k) <= observed)
    return total, ways


def fisher_clears(total: int, ways: int) -> bool:
    """True when the exact p is strictly below 0.025 = 1/40."""
    return total * 40 < ways


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float, float, float]:
    """U of ``left``, two-sided p, and the two mean ranks.

    Normal approximation with the tie correction and the continuity
    correction. Every value tied gives p = 1. A lower mean rank is a
    lower call score.
    """
    n1, n2 = len(left), len(right)
    if n1 < 1 or n2 < 1:
        raise ValueError("mann-whitney needs two non-empty samples")
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
    rank_a = sum(rank[value] for value in left)
    rank_b = sum(rank[value] for value in right)
    u = rank_a - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return u, 1.0, rank_a / n1, rank_b / n2
    dev = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    p = min(1.0, math.erfc(dev / math.sqrt(var) / math.sqrt(2.0)))
    return u, p, rank_a / n1, rank_b / n2


def call_score(hit: int, calls: int, budget: int) -> float:
    """Hits keep their call count. Every miss scores one past the budget."""
    if hit == 1:
        if calls > budget:
            raise ValueError("hit calls exceed the budget")
        return float(calls)
    return float(budget) + 1.0


def energy_hits(energy: float, reference: float, tol: float = HIT_TOL) -> bool:
    return math.isfinite(energy) and math.isfinite(reference) and math.isfinite(tol) and tol >= 0.0 and energy <= reference + tol


def full_precision(value: float) -> str:
    """Shortest plain decimal that parses back to the same binary64."""
    if not math.isfinite(value):
        raise ValueError("non-finite best")
    for digits in range(1, 18):
        text = format(value, f".{digits}f")
        if float(text) == value:
            return text
    text = repr(value)
    if float(text) != value:
        raise ValueError("best does not round-trip")
    return text


def best_bits(value: float) -> str:
    return f"{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


def show_p(p: float) -> str:
    return f"{p:.8g}"


def show_num(value: float) -> str:
    if value == math.trunc(value):
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def direction_hits(hits_a: int, hits_b: int) -> str:
    if hits_a == hits_b:
        return "equal"
    if hits_b > hits_a:
        return "B more hits"
    return "A more hits"


def direction_lower(left: float | None, right: float | None) -> str:
    """Direction of a lower-is-smaller quantity. ``None`` is an empty sample."""
    if left is None or right is None:
        return "undefined"
    if left == right:
        return "equal"
    if right < left:
        return "B lower"
    return "A lower"


def signed(left: float | None, right: float | None) -> str:
    if left is None or right is None:
        return "undefined"
    return show_num(right - left)


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return type7(values)


def select(rows: list[dict], cell: str, arm: str) -> list[dict]:
    found = [row for row in rows if row["cell"] == cell and row["arm"] == arm]
    if [int(row["seed"]) for row in found] != list(SEEDS):
        raise SystemExit(f"{cell} {arm} seeds {[row['seed'] for row in found]}")
    return found


def summarise_cell(rows: list[dict], cell: str) -> dict:
    budget, reference = CELL_BY_NAME[cell]
    left = select(rows, cell, "A")
    right = select(rows, cell, "B")
    for row in left + right:
        if int(row["budget"]) != budget:
            raise SystemExit(f"{cell} budget {row['budget']}")
        if int(row["core"]) != 0 or int(row["slot"]) not in (0, 1):
            raise SystemExit(f"{cell} core or slot {row}")
    hits_a = sum(int(row["hit"]) == 1 for row in left)
    hits_b = sum(int(row["hit"]) == 1 for row in right)
    fisher_n, fisher_d = fisher_exact(hits_a, len(left), hits_b, len(right))
    scores_a = [call_score(int(row["hit"]), int(row["calls"]), budget) for row in left]
    scores_b = [call_score(int(row["hit"]), int(row["calls"]), budget) for row in right]
    u, p_calls, mean_a, mean_b = mann_whitney(scores_a, scores_b)
    calls_hit_a = [float(row["calls"]) for row in left if int(row["hit"]) == 1]
    calls_hit_b = [float(row["calls"]) for row in right if int(row["hit"]) == 1]
    calls_all_a = [float(row["calls"]) for row in left]
    calls_all_b = [float(row["calls"]) for row in right]
    wall_a = [float(row["wall"]) for row in left]
    wall_b = [float(row["wall"]) for row in right]
    med_hit_a = median_or_none(calls_hit_a)
    med_hit_b = median_or_none(calls_hit_b)
    med_all_a = type7(calls_all_a)
    med_all_b = type7(calls_all_b)
    med_wall_a = type7(wall_a)
    med_wall_b = type7(wall_b)
    return {
        "cell": cell,
        "budget": budget,
        "reference": reference,
        "n": len(left),
        "hits_a": hits_a,
        "hits_b": hits_b,
        "hits_direction": direction_hits(hits_a, hits_b),
        "hits_diff_b_minus_a": hits_b - hits_a,
        "fisher_n": fisher_n,
        "fisher_d": fisher_d,
        "fisher_p": fisher_n / fisher_d,
        "fisher_clears": fisher_clears(fisher_n, fisher_d),
        "median_calls_hits_a": med_hit_a,
        "median_calls_hits_b": med_hit_b,
        "median_calls_hits_direction": direction_lower(med_hit_a, med_hit_b),
        "median_calls_hits_diff": signed(med_hit_a, med_hit_b),
        "median_calls_all_a": med_all_a,
        "median_calls_all_b": med_all_b,
        "median_calls_all_direction": direction_lower(med_all_a, med_all_b),
        "median_calls_all_diff": signed(med_all_a, med_all_b),
        "median_wall_a": med_wall_a,
        "median_wall_b": med_wall_b,
        "median_wall_direction": direction_lower(med_wall_a, med_wall_b),
        "median_wall_diff": signed(med_wall_a, med_wall_b),
        "u_a": u,
        "mannwhitney_p": p_calls,
        "mean_rank_a": mean_a,
        "mean_rank_b": mean_b,
        "rank_direction": direction_lower(mean_a, mean_b),
        "rank_diff": signed(mean_a, mean_b),
        "mannwhitney_clears": p_calls < ALPHA,
        "per_seed": [
            {
                "seed": int(a["seed"]),
                "hit_a": int(a["hit"]),
                "hit_b": int(b["hit"]),
                "calls_a": int(a["calls"]),
                "calls_b": int(b["calls"]),
            }
            for a, b in zip(left, right)
        ],
    }


def clears_text(flag: bool) -> str:
    if flag:
        return "clears the Bonferroni threshold 0.025"
    return "does not clear the Bonferroni threshold 0.025"


def render_report(rows: list[dict], sessions: list[dict]) -> str:
    require_balance(schedule())
    check_agreement(rows, sessions)
    counts = pair_counts(
        [
            {"cell": row["cell"], "slot": int(row["slot"]), "arm": row["arm"]}
            for row in sessions
        ]
    )
    lines = [
        "# Morse rho = 14, pair-scaled preset",
        "",
        f"Arm A is commit `{BASE_COMMIT}` tree `{BASE_TREE}`, the default preset, with no H2H variable set.",
        f"Arm B is tree `{ARM_B_TREE}` with `H2H_PAIR_SCALE=1` and no other H2H variable. That switch selects the pair-scaled preset.",
        "Seeds 2100-2123 are twenty-four fresh seeds per arm per cell. Both arms of a seed run the geometry arm A writes for that seed.",
        "m55r14 has budget 1000000 and reference -220.646208. m80r14 has budget 4000000 and reference -340.811371.",
        "A hit is a finite best energy at or below the reference plus 0.001.",
        "Inside each cell the order is ABBA. An even offset from 2100 runs A then B, and an odd offset runs B then A. Slot 0 is the first arm of the seed. m55r14 runs before m80r14.",
        "The measured process tree is pinned to core 0. Every thread-pool variable is 1. The binaries are release builds. No run uses a shortened budget, and there is no warm-up search.",
        "Medians are Hyndman-Fan type 7. The all-seeds call median uses the recorded calls, and a miss is recorded at its budget.",
        "Fisher is the two-sided exact test on the hit table. Mann-Whitney is two-sided, using the normal approximation with the tie correction and the continuity correction.",
        "For that ranking a hit keeps its call count and a miss scores one past the budget, so every miss ranks worse than every hit.",
        "Bonferroni uses m = 2 inside the Fisher family and m = 2 inside the Mann-Whitney family. The threshold in each family is 0.025.",
        "The recorded best is the example's printed energy, eight digits after the decimal, reparsed to binary64 and written so the decimal round-trips. best_bits is the IEEE-754 layout of that reparsed value.",
        "",
        "## Balance",
        "",
        "Counts of (cell, slot, arm) in the executed sessions:",
        "",
        "| cell | slot | arm | count |",
        "| --- | --- | --- | --- |",
    ]
    for cell, _budget, _reference in CELLS:
        for slot in (0, 1):
            for arm in ARMS:
                found = counts[(cell, slot, arm)]
                if found != 12:
                    raise SystemExit(f"session balance {(cell, slot, arm)} = {found}")
                lines.append(f"| {cell} | {slot} | {arm} | {found} |")
    pooled = Counter((int(row["slot"]), row["arm"]) for row in sessions)
    lines.extend(["", "Pooled over both cells:", ""])
    for slot in (0, 1):
        for arm in ARMS:
            lines.append(f"- slot {slot}, arm {arm}: {pooled[(slot, arm)]}")
    machines = sorted({row["machine"] for row in rows})
    if len(machines) != 1:
        raise SystemExit(f"machine ids {machines}")
    lines.extend(["", f"Hashed machine id `{machines[0]}`.", ""])
    for cell, _budget, _reference in CELLS:
        block = summarise_cell(rows, cell)
        lines.extend(render_cell(block))
    lines.extend(["## Per-seed hits", ""])
    for cell, _budget, _reference in CELLS:
        block = summarise_cell(rows, cell)
        lines.extend(
            [
                f"### {cell}",
                "",
                "| seed | A hit | B hit | A calls | B calls |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in block["per_seed"]:
            lines.append(
                f"| {item['seed']} | {item['hit_a']} | {item['hit_b']} | {item['calls_a']} | {item['calls_b']} |"
            )
        lines.append("")
    arm_commits = {arm: sorted({row["arm_commit"] for row in sessions if row["arm"] == arm}) for arm in ARMS}
    arm_trees = {arm: sorted({row["arm_tree"] for row in sessions if row["arm"] == arm}) for arm in ARMS}
    for arm in ARMS:
        if len(arm_commits[arm]) != 1 or len(arm_trees[arm]) != 1:
            raise SystemExit(f"{arm} identity {arm_commits} {arm_trees}")
    lines.extend(
        [
            "## Identity",
            "",
            f"- base commit `{BASE_COMMIT}`",
            f"- base tree `{BASE_TREE}`",
            f"- arm A commit `{arm_commits['A'][0]}` tree `{arm_trees['A'][0]}`",
            f"- arm B commit `{arm_commits['B'][0]}` tree `{arm_trees['B'][0]}`",
            f"- rustc `{sessions[0]['rustc']}`",
            "",
        ]
    )
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    forbidden = ("Co-authored-by", "Signed-off-by", "Generated with", "PASS", "FAIL")
    for token in forbidden:
        if token in text:
            raise SystemExit(f"report contains {token}")
    return text


def render_cell(block: dict) -> list[str]:
    lines = [
        f"## {block['cell']}",
        "",
        f"Budget {block['budget']}. Reference {block['reference']}. n = {block['n']} per arm.",
        "",
        "| quantity | A | B | B minus A | direction |",
        "| --- | --- | --- | --- | --- |",
        f"| hits | {block['hits_a']} | {block['hits_b']} | {block['hits_diff_b_minus_a']} | {block['hits_direction']} |",
        "| median calls, hits only | "
        + " | ".join(
            (
                show_optional(block["median_calls_hits_a"]),
                show_optional(block["median_calls_hits_b"]),
                block["median_calls_hits_diff"],
                block["median_calls_hits_direction"],
            )
        )
        + " |",
        "| median calls, all seeds | "
        + " | ".join(
            (
                show_num(block["median_calls_all_a"]),
                show_num(block["median_calls_all_b"]),
                block["median_calls_all_diff"],
                block["median_calls_all_direction"],
            )
        )
        + " |",
        "| median wall, seconds | "
        + " | ".join(
            (
                f"{block['median_wall_a']:.6f}",
                f"{block['median_wall_b']:.6f}",
                block["median_wall_diff"],
                block["median_wall_direction"],
            )
        )
        + " |",
        "",
        f"Fisher p = {block['fisher_n']}/{block['fisher_d']} = {show_p(block['fisher_p'])}. "
        f"The hit difference is {block['hits_direction']}. "
        f"The Fisher test {clears_text(block['fisher_clears'])}.",
        "",
        f"Mann-Whitney U of A = {show_num(block['u_a'])}, p = {show_p(block['mannwhitney_p'])}. "
        f"Mean rank A = {block['mean_rank_a']:.4f}, mean rank B = {block['mean_rank_b']:.4f}, "
        f"B minus A = {block['rank_diff']}. The call-rank difference is {block['rank_direction']}. "
        f"The Mann-Whitney test {clears_text(block['mannwhitney_clears'])}.",
        "",
    ]
    return lines


def show_optional(value: float | None) -> str:
    if value is None:
        return "undefined"
    return show_num(value)


def check_agreement(rows: list[dict], sessions: list[dict]) -> None:
    if len(rows) != len(sessions):
        raise SystemExit(f"{len(rows)} rows and {len(sessions)} sessions")
    jobs = schedule()
    if len(rows) != len(jobs):
        raise SystemExit(f"expected {len(jobs)} rows, found {len(rows)}")
    by_key = {(row["cell"], row["arm"], int(row["seed"])): row for row in rows}
    seen = []
    for session, job in zip(sessions, jobs):
        if session["cell"] != job["cell"] or session["arm"] != job["arm"] or int(session["seed"]) != job["seed"]:
            raise SystemExit("session order differs from the declared schedule")
        if int(session["slot"]) != job["slot"] or int(session["budget"]) != job["budget"]:
            raise SystemExit("session slot or budget")
        if session["reference"] != f"{job['reference']:.6f}":
            raise SystemExit("session reference")
        key = (session["cell"], session["arm"], int(session["seed"]))
        row = by_key[key]
        seen.append(key)
        for field in ("slot", "core", "hit", "calls", "charged", "wall", "best", "best_bits", "budget", "machine", "base_commit", "base_tree"):
            if row[field] != session[field]:
                raise SystemExit(f"{key} {field} row {row[field]} session {session[field]}")
        if row["base_commit"] != BASE_COMMIT or row["base_tree"] != BASE_TREE:
            raise SystemExit("base identity")
        if session["arm"] == "A" and session["pair_scale"] != "0":
            raise SystemExit("arm A pair scale")
        if session["arm"] == "B" and session["pair_scale"] != "1":
            raise SystemExit("arm B pair scale")
        if session["arm"] == "B" and session["arm_tree"] != ARM_B_TREE:
            raise SystemExit("arm B tree")
        if session["arm"] == "A" and session["arm_tree"] != BASE_TREE:
            raise SystemExit("arm A tree")
        best = float(row["best"])
        if full_precision(best) != row["best"] or best_bits(best) != row["best_bits"]:
            raise SystemExit(f"{key} best precision")
        reference = job["reference"]
        hit = energy_hits(best, reference)
        if int(row["hit"]) != int(hit):
            raise SystemExit(f"{key} hit flag disagrees with the recorded best")
        if int(row["hit"]) == 0 and int(row["calls"]) != int(row["budget"]):
            raise SystemExit(f"{key} miss is not charged at the budget")
        digest = session["start_sha256"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise SystemExit("start hash")
    if seen != [(job["cell"], job["arm"], job["seed"]) for job in jobs]:
        raise SystemExit("row keys")
    starts = {}
    for session in sessions:
        key = (session["cell"], int(session["seed"]))
        starts.setdefault(key, set()).add(session["start_sha256"])
    for key, hashes in starts.items():
        if len(hashes) != 1:
            raise SystemExit(f"starts differ {key}")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, fields: tuple[str, ...], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({field: row[field] for field in fields})
        handle.flush()
        os.fsync(handle.fileno())


def machine_hash() -> str:
    token = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
    if not token or any(char.isspace() for char in token):
        raise SystemExit("machine id token")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def git_identity(tree: Path) -> tuple[str, str]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tree, check=True, capture_output=True, text=True
    ).stdout.strip()
    digest = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=tree, check=True, capture_output=True, text=True
    ).stdout.strip()
    return head, digest


def thread_env(base: dict[str, str], pair_scale: bool) -> dict[str, str]:
    env = dict(base)
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_", "PRIOR_")):
            del env[key]
        elif "NUM_THREADS" in key or key.endswith("_THREADS") or key in THREAD_VARS:
            env[key] = "1"
    for key in THREAD_VARS:
        env[key] = "1"
    if pair_scale:
        env["H2H_PAIR_SCALE"] = "1"
    h2h = sorted(key for key in env if key.startswith("H2H_"))
    if pair_scale and h2h != ["H2H_PAIR_SCALE"]:
        raise SystemExit(f"arm B environment {h2h}")
    if not pair_scale and h2h:
        raise SystemExit(f"arm A environment {h2h}")
    if env.get("H2H_PAIR_SCALE") not in (None, "1"):
        raise SystemExit("pair scale value")
    for key in THREAD_VARS:
        if env[key] != "1":
            raise SystemExit(f"{key}={env[key]}")
    return env


def ensure_core0() -> None:
    affinity = set(os.sched_getaffinity(0))
    if affinity == {0}:
        return
    if os.environ.get("PAIRSCALE_RHO14_PINNED") == "1":
        raise SystemExit(f"affinity {sorted(affinity)}")
    os.environ["PAIRSCALE_RHO14_PINNED"] = "1"
    os.execvp("taskset", ["taskset", "-c", "0", sys.executable, *sys.argv])


def rustc_version() -> str:
    proc = subprocess.run(["rustc", "--version"], check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def run_budget(binary: Path, tree: Path, env: dict[str, str], cell: str) -> tuple[int, float, float]:
    proc = subprocess.run(
        [str(binary), "budget", cell],
        cwd=tree,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"{cell} budget failed\n{proc.stderr}")
    lines = [line for line in proc.stdout.splitlines() if line]
    if len(lines) != 3:
        raise SystemExit(f"{cell} budget lines {lines}")
    budget = int(lines[0])
    reference = float(lines[1])
    rho = float(lines[2])
    return budget, reference, rho


def write_start(binary: Path, tree: Path, env: dict[str, str], cell: str, seed: int, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(binary), "coords", cell, str(seed), str(path)],
        cwd=tree,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not path.is_file():
        raise SystemExit(f"{cell} {seed} coords failed\n{proc.stderr}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def parse_stdout(text: str, cell: str, seed: int, budget: int) -> dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    headers = [i for i, line in enumerate(lines) if line == HEADER]
    if len(headers) != 1:
        raise SystemExit(f"{cell} {seed}: expected one result header\n{text[-2000:]}")
    index = headers[0]
    if index + 1 >= len(lines):
        raise SystemExit(f"{cell} {seed}: missing result row")
    data = lines[index + 1]
    row = dict(zip(HEADER.split(","), data.split(","), strict=True))
    if row["system"] != cell or int(row["seed"]) != seed or int(row["budget"]) != budget:
        raise SystemExit(f"{cell} {seed}: row {data}")
    if row["code"] != "anneal":
        raise SystemExit(f"{cell} {seed}: code {row['code']}")
    if not WALL_TOKEN.fullmatch(row["wall_s"]) or not BEST_TOKEN.fullmatch(row["best"]):
        raise SystemExit(f"{cell} {seed}: wall or best token {data}")
    return row


def execute(binary: Path, tree: Path, env: dict[str, str], cell: str, seed: int, budget: int, coords: Path, logs: Path, arm: str) -> dict[str, str]:
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{cell}-{arm}-{seed}.stdout"
    stderr_path = logs / f"{cell}-{arm}-{seed}.stderr"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(
            [str(binary), "run", cell, str(seed), str(coords)],
            cwd=tree,
            env=env,
            check=False,
            stdout=stdout,
            stderr=stderr,
        )
    if proc.returncode != 0:
        tail = stderr_path.read_text(encoding="utf-8")[-2000:]
        raise SystemExit(f"{cell} {arm} {seed} exited {proc.returncode}\n{tail}")
    return parse_stdout(stdout_path.read_text(encoding="utf-8"), cell, seed, budget)


def completed_prefix(path: Path) -> list[tuple[str, str, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = read_csv(path)
    jobs = schedule()
    keys = [(row["cell"], row["arm"], int(row["seed"])) for row in rows]
    expected = [(job["cell"], job["arm"], job["seed"]) for job in jobs]
    if keys != expected[: len(keys)]:
        raise SystemExit("existing rows are not a prefix of the declared schedule")
    return keys


def run_campaign(argv: list[str]) -> int:
    ensure_core0()
    if set(os.sched_getaffinity(0)) != {0}:
        raise SystemExit("core")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--a-bin", type=Path, required=True)
    parser.add_argument("--a-tree", type=Path, required=True)
    parser.add_argument("--b-bin", type=Path, required=True)
    parser.add_argument("--b-tree", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    for path in (args.logs, args.work):
        resolved = path.resolve()
        if resolved == root or root in resolved.parents:
            raise SystemExit(f"{resolved} is inside the repository")
    jobs = schedule()
    require_balance(jobs)
    counts = pair_counts(jobs)
    for key in sorted(counts):
        print(f"count cell={key[0]} slot={key[1]} arm={key[2]} n={counts[key]}", flush=True)
    a_commit, a_tree = git_identity(args.a_tree)
    b_commit, b_tree = git_identity(args.b_tree)
    if (a_commit, a_tree) != (BASE_COMMIT, BASE_TREE):
        raise SystemExit(f"arm A is {a_commit} {a_tree}")
    if b_tree != ARM_B_TREE:
        raise SystemExit(f"arm B tree {b_tree}")
    parent = dict(os.environ)
    env_a = thread_env(parent, pair_scale=False)
    env_b = thread_env(parent, pair_scale=True)
    for cell, budget, reference in CELLS:
        for binary, tree, env, arm in (
            (args.a_bin, args.a_tree, env_a, "A"),
            (args.b_bin, args.b_tree, env_b, "B"),
        ):
            got_budget, got_reference, rho = run_budget(binary, tree, env, cell)
            if got_budget != budget or f"{got_reference:.6f}" != f"{reference:.6f}" or rho != 14.0:
                raise SystemExit(f"{arm} {cell} budget {got_budget} reference {got_reference} rho {rho}")
    machine = machine_hash()
    compiler = rustc_version()
    done = completed_prefix(args.rows)
    session_rows = read_csv(args.sessions) if args.sessions.exists() and args.sessions.stat().st_size else []
    if len(session_rows) != len(done):
        raise SystemExit("sessions and rows have different lengths")
    session = session_rows[0]["session"] if session_rows else time_session()
    print(f"session {session} machine {machine} rustc {compiler}", flush=True)
    for index, job in enumerate(jobs):
        key = (job["cell"], job["arm"], job["seed"])
        if index < len(done):
            if done[index] != key:
                raise SystemExit("prefix key")
            continue
        if index != len(done):
            raise SystemExit("gap in the schedule")
        coords = args.work / job["cell"] / f"{job['seed']}.xyz"
        start_hash = write_start(args.a_bin, args.a_tree, env_a, job["cell"], job["seed"], coords)
        binary = args.a_bin if job["arm"] == "A" else args.b_bin
        tree = args.a_tree if job["arm"] == "A" else args.b_tree
        env = env_a if job["arm"] == "A" else env_b
        arm_commit = a_commit if job["arm"] == "A" else b_commit
        arm_tree = a_tree if job["arm"] == "A" else b_tree
        print(f"start {job['cell']} {job['arm']} {job['seed']} slot {job['slot']}", flush=True)
        parsed = execute(binary, tree, env, job["cell"], job["seed"], job["budget"], coords, args.logs, job["arm"])
        best = float(parsed["best"])
        binary_hit = parsed["hit"].lower()
        if binary_hit not in ("true", "false"):
            raise SystemExit(f"hit token {parsed['hit']}")
        hit = energy_hits(best, job["reference"])
        if hit != (binary_hit == "true"):
            raise SystemExit(
                f"{job['cell']} {job['arm']} {job['seed']}: printed hit {binary_hit} "
                f"disagrees with best {parsed['best']}"
            )
        calls = int(parsed["evals_to_hit"])
        charged = int(parsed["charged"])
        if hit and not 0 <= calls <= job["budget"]:
            raise SystemExit("hit calls")
        if not hit and calls != job["budget"]:
            raise SystemExit("miss calls")
        record = {
            "cell": job["cell"],
            "arm": job["arm"],
            "seed": str(job["seed"]),
            "slot": str(job["slot"]),
            "core": "0",
            "hit": "1" if hit else "0",
            "calls": str(calls),
            "charged": str(charged),
            "wall": parsed["wall_s"],
            "best": full_precision(best),
            "best_bits": best_bits(best),
            "budget": str(job["budget"]),
            "machine": machine,
            "base_commit": BASE_COMMIT,
            "base_tree": BASE_TREE,
            "session": session,
            "order_index": str(index),
            "reference": f"{job['reference']:.6f}",
            "arm_commit": arm_commit,
            "arm_tree": arm_tree,
            "start_sha256": start_hash,
            "pair_scale": "1" if job["arm"] == "B" else "0",
            "rustc": compiler,
        }
        if float(record["best"]) != best or best_bits(float(record["best"])) != record["best_bits"]:
            raise SystemExit("recorded best does not round-trip")
        append_csv(args.rows, ROW_FIELDS, record)
        append_csv(args.sessions, SESSION_FIELDS, record)
        done.append(key)
        print(
            f"done {job['cell']} {job['arm']} {job['seed']} hit={record['hit']} "
            f"calls={record['calls']} charged={record['charged']} wall={record['wall']} best={record['best']}",
            flush=True,
        )
    return 0


def time_session() -> str:
    import time

    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def self_test() -> None:
    jobs = schedule()
    require_balance(jobs)
    if fisher_exact(2, 2, 0, 2) != (2, 6):
        raise SystemExit("fisher 2 vs 0")
    if fisher_exact(0, 24, 0, 24) != (1, 1):
        raise SystemExit("fisher zeros")
    total, ways = fisher_exact(5, 12, 0, 12)
    if abs(total / ways - 0.0372671) > 5e-8:
        raise SystemExit(f"fisher 5 vs 0 of 12 is {total}/{ways}")
    if fisher_clears(total, ways):
        raise SystemExit("5 vs 0 of 12 must stay above 0.025")
    if not fisher_clears(*fisher_exact(24, 24, 0, 24)):
        raise SystemExit("24 vs 0 should clear")
    u, p, mean_a, mean_b = mann_whitney([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    if p != 1.0 or mean_a != mean_b:
        raise SystemExit("tied mann-whitney")
    u, p, mean_a, mean_b = mann_whitney([10.0, 20.0, 30.0], [101.0, 101.0, 101.0])
    if not (mean_a < mean_b and u == 0.0 and 0.0 < p < 0.1):
        raise SystemExit(f"ordered mann-whitney {u} {p} {mean_a} {mean_b}")
    if call_score(1, 1_000_000, 1_000_000) >= call_score(0, 1_000_000, 1_000_000):
        raise SystemExit("a miss must rank worse than a full-budget hit")
    if type7([float(v) for v in range(1, 25)]) != 12.5:
        raise SystemExit("type 7 median")
    sample = float("-220.64620756")
    text = full_precision(sample)
    if float(text) != sample or best_bits(float(text)) != best_bits(sample):
        raise SystemExit("best round-trip")
    if not energy_hits(-220.646208 + HIT_TOL, -220.646208):
        raise SystemExit("boundary hit")
    if energy_hits(math.nextafter(-220.646208 + HIT_TOL, math.inf), -220.646208):
        raise SystemExit("boundary miss")
    if energy_hits(math.nan, -220.646208) or energy_hits(math.inf, -220.646208):
        raise SystemExit("non-finite hit")
    env = thread_env({"H2H_BRANCH": "1", "OMP_NUM_THREADS": "8", "PATH": "/usr/bin"}, pair_scale=False)
    if "H2H_BRANCH" in env or env["OMP_NUM_THREADS"] != "1" or any(env[key] != "1" for key in THREAD_VARS):
        raise SystemExit("arm A env")
    env_b = thread_env({"H2H_PLATEAU_HEAT": "1", "PATH": "/usr/bin"}, pair_scale=True)
    if env_b["H2H_PAIR_SCALE"] != "1" or "H2H_PLATEAU_HEAT" in env_b:
        raise SystemExit("arm B env")
    if direction_hits(3, 5) != "B more hits" or direction_lower(4.0, 2.0) != "B lower":
        raise SystemExit("direction")
    if direction_lower(None, 1.0) != "undefined" or direction_hits(4, 4) != "equal":
        raise SystemExit("tie direction")
    text = render_report(synthetic_rows(), synthetic_sessions())
    if text.count("does not clear the Bonferroni threshold 0.025") != 4:
        raise SystemExit("tied synthetic report")
    if "equal" not in text or "B more hits" in text:
        raise SystemExit("synthetic direction")
    print("ok")


def synthetic_record(job: dict, index: int) -> dict:
    best = -1.0
    return {
        "cell": job["cell"],
        "arm": job["arm"],
        "seed": str(job["seed"]),
        "slot": str(job["slot"]),
        "core": "0",
        "hit": "0",
        "calls": str(job["budget"]),
        "charged": str(job["budget"]),
        "wall": "1.000000",
        "best": full_precision(best),
        "best_bits": best_bits(best),
        "budget": str(job["budget"]),
        "machine": "ab" * 32,
        "base_commit": BASE_COMMIT,
        "base_tree": BASE_TREE,
        "session": "20260101T000000Z",
        "order_index": str(index),
        "reference": f"{job['reference']:.6f}",
        "arm_commit": BASE_COMMIT if job["arm"] == "A" else "f14713c3ad2115ba3134dc7b04de553b9695b946",
        "arm_tree": BASE_TREE if job["arm"] == "A" else ARM_B_TREE,
        "start_sha256": "cd" * 32,
        "pair_scale": "1" if job["arm"] == "B" else "0",
        "rustc": "rustc 1.88.0 (6b00bc388 2025-06-23)",
    }


def synthetic_rows() -> list[dict]:
    return [{field: record[field] for field in ROW_FIELDS} for record in synthetic_sessions()]


def synthetic_sessions() -> list[dict]:
    return [synthetic_record(job, index) for index, job in enumerate(schedule())]


def main(argv: list[str]) -> int:
    if len(argv) == 1 or argv[1] == "--self-test":
        self_test()
        return 0
    if argv[1] == "run":
        return run_campaign(argv[2:])
    if argv[1] == "report":
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--rows", type=Path, required=True)
        parser.add_argument("--sessions", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)
        args = parser.parse_args(argv[2:])
        text = render_report(read_csv(args.rows), read_csv(args.sessions))
        args.out.write_text(text, encoding="utf-8", newline="\n")
        return 0
    raise SystemExit(f"unknown command {argv[1]}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
