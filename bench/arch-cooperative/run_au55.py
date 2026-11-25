# Copyright (c) anneal developers
"""Serial au55 cooperative arm.

Reads PAPER_BUDGETS from the paper panel, runs seeds 2000-2011 one at
a time on one pinned CPU, and appends rows. A seed already stored for
the same system, arm, seed and revision is left untouched.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "paper_0120"))
from run_panel import PAPER_BUDGETS  # noqa: E402

SEEDS = tuple(range(2000, 2012))
ARM = "cooperative"
SYSTEM = "au55"
CHAINS = 4
REFERENCE = "-191.03085280"
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261123
KINDS = (
    "local_proposal",
    "remote_proposal",
    "descriptor_evaluation",
    "accepted_quench",
    "rejected_quench",
    "fresh_validation",
    "retry",
    "rpc_fallback",
    "auxiliary_evaluation",
    "basin_escape",
    "saddle_ride",
)
HEADER = [
    "system",
    "arm",
    "seed",
    "chains",
    "oracle_calls",
    "call_budget",
    "hit",
    "calls_to_hit",
    "hit_clock",
    "best_energy",
    "wall_s",
    "cpu_s",
    "peak_rss_anon",
]
for _kind in KINDS:
    HEADER.append(f"{_kind}_events")
    HEADER.append(f"{_kind}_calls")
HEADER.extend(
    [
        "charge_sum",
        "ensemble_total",
        "server_aggregate",
        "terms_match",
        "binary",
        "features",
        "shape_keyed",
        "libira_sha256",
        "revision",
        "nproc",
        "cpu_model",
    ]
)
ROWS = ROOT / "bench" / "arch-cooperative" / "au55" / "rows.csv"
FEATURES = "bank-rpc ira"
BINARY_NAME = "arch_cooperative"
REPORT = ROOT / "bench" / "arch-cooperative" / "au55" / "REPORT.md"
BASELINE = ROOT / "bench" / "paper-0120" / "rows.csv"


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("this harness takes no arguments")
    budget = int(PAPER_BUDGETS[SYSTEM])
    if budget <= 0 or budget % CHAINS != 0:
        raise SystemExit("au55 budget is not a positive multiple of four")
    metal = (ROOT / "examples" / "h2h_metal.rs").read_text(encoding="utf-8")
    if REFERENCE not in metal:
        raise SystemExit("au55 reference string is absent from the metal example")
    binary = release_binary()
    libira = libira_sha256()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise SystemExit("HEAD is not a 40-character revision")
    done = stored_seeds(revision)
    walls: list[float] = []
    finished: list[dict[str, str]] = []
    if ROWS.is_file():
        finished.extend(rows_for(revision))
        walls.extend(float(row["wall_s"]) for row in finished)
    minima = tempfile.mkdtemp(prefix="au55-minima-")
    for seed in SEEDS:
        if seed in done:
            continue
        row = run_seed(seed, budget, revision, minima, binary, libira)
        append_row(row)
        finished.append(row)
        walls.append(float(row["wall_s"]))
        done.add(seed)
        print(
            f"seed {seed} hit={row['hit']} calls={row['calls_to_hit']} "
            f"wall_s={row['wall_s']} terms_match={row['terms_match']}",
            flush=True,
        )
    write_report(finished, budget, revision, libira)


def stored_seeds(revision: str) -> set[int]:
    return {int(row["seed"]) for row in rows_for(revision)}


def rows_for(revision: str) -> list[dict[str, str]]:
    if not ROWS.is_file() or ROWS.stat().st_size == 0:
        return []
    with ROWS.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HEADER:
            raise SystemExit("row header does not match the protocol")
        rows = []
        for row in reader:
            if row["system"] == SYSTEM and row["arm"] == ARM and row["revision"] == revision:
                rows.append(row)
        return rows


def release_binary() -> Path:
    target = ROOT / "target"
    config = ROOT / ".cargo" / "config.toml"
    if config.is_file():
        for line in config.read_text(encoding="utf-8").splitlines():
            stripped = line.split("#", 1)[0].strip()
            if stripped.startswith("target-dir"):
                value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                target = Path(value)
                if not target.is_absolute():
                    target = (ROOT / target).resolve()
                break
    binary = target / "release" / "examples" / "arch_cooperative"
    if not binary.is_file():
        raise SystemExit("release example is not built")
    return binary


def libira_sha256() -> str:
    lib_dir = os.environ.get("IRA_LIB_DIR", "")
    path = Path(lib_dir) / "libira.so"
    if not path.is_file():
        raise SystemExit("libira.so is not available")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(digest) != 64:
        raise SystemExit("libira.so digest is not 64 hex characters")
    return digest


def run_seed(
    seed: int,
    budget: int,
    revision: str,
    minima: str,
    binary: Path,
    libira: str,
) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        env[key] = "1"
    env["CATALOG_CONNECT_TIMEOUT_SECS"] = "10"
    env["CATALOG_IO_TIMEOUT_SECS"] = "120"
    env["MINIMA_ROOT"] = minima
    with tempfile.NamedTemporaryFile(prefix="au55-cooperative-", suffix=".log", delete=False) as log:
        log_path = log.name
    with open(log_path, "w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            [
                "taskset",
                "-c",
                "0",
                str(binary),
                "seed",
                str(seed),
                str(budget),
                revision,
                libira,
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        tail = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        print("\n".join(tail), file=sys.stderr)
        raise SystemExit(f"seed {seed} exited {completed.returncode}")
    line = completed.stdout.strip()
    if "\n" in line or not line:
        raise SystemExit(f"seed {seed} did not print one row")
    fields = line.split(",")
    if len(fields) != len(HEADER):
        raise SystemExit(f"seed {seed} row has {len(fields)} fields")
    row = dict(zip(HEADER, fields, strict=True))
    check_row(row, seed, budget, revision, libira)
    return row


def check_row(
    row: dict[str, str], seed: int, budget: int, revision: str, libira: str
) -> None:
    if "/" in row["cpu_model"] or contains_forbidden(row["cpu_model"]):
        raise SystemExit(f"seed {seed} cpu model is not a bare model string")
    if row["system"] != SYSTEM or row["arm"] != ARM:
        raise SystemExit(f"seed {seed} identity does not match")
    if int(row["seed"]) != seed or int(row["chains"]) != CHAINS:
        raise SystemExit(f"seed {seed} chain record does not match")
    if int(row["call_budget"]) != budget or row["revision"] != revision:
        raise SystemExit(f"seed {seed} budget or revision does not match")
    if row["binary"] != BINARY_NAME or row["features"] != FEATURES:
        raise SystemExit(f"seed {seed} binary or features do not match")
    if row["shape_keyed"] != "true" or row["libira_sha256"] != libira:
        raise SystemExit(f"seed {seed} shape keying or library digest does not match")
    wall = float(row["wall_s"])
    if not math.isfinite(wall) or wall < 0.05:
        raise SystemExit(f"seed {seed} wall time is below 50 ms")
    oracle = int(row["oracle_calls"])
    charge = int(row["charge_sum"])
    ensemble = int(row["ensemble_total"])
    server = int(row["server_aggregate"])
    kind_calls = sum(int(row[f"{kind}_calls"]) for kind in KINDS)
    if kind_calls != charge:
        raise SystemExit(f"seed {seed} kind calls do not add up")
    matched = charge == ensemble == oracle == server
    if matched != (row["terms_match"] == "1"):
        raise SystemExit(f"seed {seed} terms_match disagrees with the totals")
    hit = int(row["hit"])
    calls = int(row["calls_to_hit"])
    clock = row["hit_clock"]
    if hit == 0:
        if clock != "miss" or calls != budget + 1:
            raise SystemExit(f"seed {seed} miss clock is inconsistent")
    elif hit == 1 and clock == "quench" and calls > budget:
        raise SystemExit(f"seed {seed} hit clock exceeds the budget")
    elif hit == 1 and clock == "end" and calls != oracle:
        raise SystemExit(f"seed {seed} end clock is not the charged sum")
    elif hit == 1 and clock not in ("quench", "end"):
        raise SystemExit(f"seed {seed} hit clock is unknown")
    elif hit not in (0, 1):
        raise SystemExit(f"seed {seed} hit flag is unknown")


def append_row(row: dict[str, str]) -> None:
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    new = not ROWS.is_file() or ROWS.stat().st_size == 0
    if not new:
        with ROWS.open(encoding="utf-8", newline="") as handle:
            header = handle.readline().rstrip("\n")
        if header != ",".join(HEADER):
            raise SystemExit("row header does not match the protocol")
    line = ",".join(row[name] for name in HEADER)
    with ROWS.open("a", encoding="utf-8", newline="\n") as handle:
        if new:
            handle.write(",".join(HEADER) + "\n")
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_report(
    rows: list[dict[str, str]],
    budget: int,
    revision: str,
    libira: str,
) -> None:
    if not rows:
        return
    hits = [int(row["hit"]) for row in rows]
    calls = [int(row["calls_to_hit"]) for row in rows]
    walls = [float(row["wall_s"]) for row in rows]
    rng = random.Random(BOOTSTRAP_SEED)
    hit_lo, hit_hi = bootstrap_ci(hits, mean, rng)
    call_lo, call_hi = bootstrap_ci(calls, median, rng)
    lines = [
        "# Au55 cooperative rows",
        "",
        f"Revision `{revision}`. Seeds finished: {len(rows)}.",
        f"Hits: {sum(hits)} of {len(rows)}.",
        (
            f"Hit rate {mean(hits):.4f}, bootstrap 95 percent interval "
            f"[{hit_lo:.4f}, {hit_hi:.4f}]."
        ),
        (
            f"Calls to hit, median {median(calls):.1f}, bootstrap 95 percent "
            f"interval [{call_lo:.1f}, {call_hi:.1f}]. A miss is the budget "
            f"plus one ({budget + 1})."
        ),
        f"Wall sum {sum(walls):.3f} s. Mean wall {mean(walls):.3f} s.",
        f"CPU sum {sum(float(row['cpu_s']) for row in rows):.3f} s.",
        f"Oracle-call sum {sum(int(row['oracle_calls']) for row in rows)}.",
        f"Ledger charge sum {sum(int(row['charge_sum']) for row in rows)}.",
        f"Ensemble total sum {sum(int(row['ensemble_total']) for row in rows)}.",
        f"Server aggregate sum {sum(int(row['server_aggregate']) for row in rows)}.",
        f"Binary {BINARY_NAME}. Features {FEATURES}. shape_keyed true.",
        f"libira.so sha256 {libira}.",
        "",
        "Per seed:",
        "",
        "| seed | hit | calls_to_hit | wall_s | charge_sum | terms_match |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['hit']} | {row['calls_to_hit']} | "
            f"{row['wall_s']} | {row['charge_sum']} | {row['terms_match']} |"
        )
    mismatches = [row for row in rows if row["terms_match"] != "1"]
    lines.append("")
    if mismatches:
        lines.append("Ledger mismatches:")
        lines.append("")
        for row in mismatches:
            lines.append(
                f"Seed {row['seed']}: charge_sum {row['charge_sum']}, "
                f"ensemble_total {row['ensemble_total']}, "
                f"oracle_calls {row['oracle_calls']}, "
                f"server_aggregate {row['server_aggregate']}."
            )
    else:
        lines.append("Named charge terms match the charged total on every finished seed.")
    ends = [row["seed"] for row in rows if row["hit_clock"] == "end"]
    if ends:
        lines.append("")
        lines.append(
            "Hit energy without a quench snapshot on seeds " + ", ".join(ends) + "."
        )
        lines.append("Those calls_to_hit values are the end charged sum.")
    comparison(lines, rows, budget)
    lines.append("")
    lines.append(
        "`rows-bankrpc-only.csv` keeps twelve Au55 rows from a `bank-rpc`"
    )
    lines.append("build with `shape_keyed` false. That file is not a paper cell.")
    lines.append("Its values are unchanged.")
    lines.append("")
    lines.append(
        "Intervals use 10000 resamples and generator seed 20261123. "
        "The hit-rate interval is drawn first, then the calls-to-hit median. "
        "Fisher and Mann-Whitney are computed in this harness without an "
        "external statistics package."
    )
    text = "\n".join(lines) + "\n"
    if contains_forbidden(text):
        raise SystemExit("report text contains a forbidden token")
    REPORT.write_text(text, encoding="utf-8", newline="\n")


def comparison(lines: list[str], rows: list[dict[str, str]], budget: int) -> None:
    seeds = [int(row["seed"]) for row in rows]
    base = baseline_for(seeds)
    arm_hits = sum(int(row["hit"]) for row in rows)
    base_hits = sum(int(base[seed]["hit"]) for seed in seeds)
    arm_calls = [float(row["calls_to_hit"]) for row in rows]
    base_calls = []
    for seed in seeds:
        item = base[seed]
        if int(item["hit"]) == 1:
            base_calls.append(float(item["evals_to_hit"]))
        else:
            base_calls.append(float(item["budget"]) + 1.0)
    misses_arm = len(rows) - arm_hits
    misses_base = len(seeds) - base_hits
    fisher = fisher_two_sided(arm_hits, misses_arm, base_hits, misses_base)
    mwu = mann_whitney(arm_calls, base_calls)
    lines.append("")
    lines.append(
        f"Against stored au55 `h2h_metal` rows for the same seeds: "
        f"hits {arm_hits} versus {base_hits}, Fisher two-sided p {fisher:.4g}."
    )
    lines.append(
        f"Calls to hit, Mann-Whitney two-sided p {mwu:.4g}. "
        f"Stored misses were recoded from the budget to the budget plus one."
    )
    if any(int(base[seed]["budget"]) != budget for seed in seeds):
        lines.append("A stored row used a different call budget.")


def baseline_for(seeds: list[int]) -> dict[int, dict[str, str]]:
    found: dict[int, dict[str, str]] = {}
    with BASELINE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["system"] != SYSTEM or row["driver"] != "h2h_metal" or row["code"] != "anneal":
                continue
            seed = int(row["seed"])
            if seed not in seeds:
                continue
            if seed in found:
                raise SystemExit(f"duplicate stored au55 seed {seed}")
            found[seed] = row
    missing = [seed for seed in seeds if seed not in found]
    if missing:
        raise SystemExit("stored au55 rows are missing for a finished seed")
    return found


def contains_forbidden(text: str) -> bool:
    return ("G" + "MIN") in text


def mean(values: list[float] | list[int]) -> float:
    return sum(values) / len(values)


def median(values: list[float] | list[int]) -> float:
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def percentile(ordered: list[float], probability: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    rank = probability * (len(ordered) - 1)
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_ci(values: list[int], statistic, rng: random.Random) -> tuple[float, float]:
    size = len(values)
    samples = []
    for _ in range(BOOTSTRAP_DRAWS):
        draw = [values[rng.randrange(size)] for _ in range(size)]
        samples.append(float(statistic(draw)))
    samples.sort()
    return percentile(samples, 0.025), percentile(samples, 0.975)


def fisher_two_sided(hits_a: int, miss_a: int, hits_b: int, miss_b: int) -> float:
    row1 = hits_a + miss_a
    col1 = hits_a + hits_b
    total = row1 + hits_b + miss_b

    def ways(hits: int) -> int:
        other = col1 - hits
        if hits < 0 or other < 0 or hits > row1 or other > total - row1:
            return 0
        return math.comb(row1, hits) * math.comb(total - row1, other)

    observed = ways(hits_a)
    extreme = sum(ways(hits) for hits in range(row1 + 1) if ways(hits) <= observed)
    return extreme / math.comb(total, col1)


def mann_whitney(left: list[float], right: list[float]) -> float:
    pooled = left + right
    ranks = average_ranks(pooled)
    n1 = len(left)
    n2 = len(right)
    rank_sum = sum(ranks[:n1])
    u1 = rank_sum - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    tie_sum = 0
    previous = None
    run = 0
    for value in sorted(pooled):
        if previous is not None and value != previous:
            tie_sum += run**3 - run
            run = 0
        run += 1
        previous = value
    tie_sum += run**3 - run
    tie_term = tie_sum / (n * (n - 1)) if n > 1 else 0.0
    variance = n1 * n2 / 12.0 * ((n + 1) - tie_term)
    if variance <= 0.0:
        return 1.0
    z = (abs(u1 - mu) - 0.5) / math.sqrt(variance)
    if z < 0.0:
        z = 0.0
    return min(1.0, 2.0 * 0.5 * math.erfc(z / math.sqrt(2.0)))


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    mark = 0
    while mark < len(order):
        end = mark
        while end + 1 < len(order) and values[order[end + 1]] == values[order[mark]]:
            end += 1
        rank = 0.5 * ((mark + 1) + (end + 1))
        for index in range(mark, end + 1):
            ranks[order[index]] = rank
        mark = end + 1
    return ranks


if __name__ == "__main__":
    main()
