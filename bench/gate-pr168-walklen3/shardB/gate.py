#!/usr/bin/env python3
"""Run and score the PR 168 walklen3 shard B gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

A_REV = "7e1cdb925588d4ab8a92e420f6ac7cf70357b1ec"
A_TREE = "e5ebda63f9f0117226d50c13abf33243ccf452cc"
M_REV = "8824564b2479f0917c1e4927cd5ac938a86c2ac0"
M_TREE = "991d5afc6d7b7f7e6c392cb68aa351af1fc50179"
O_REV = "c96807c54e8cc602322255c75be61425c7bf722a"
O_TREE = "7f2625597a167d3c7841b2266eec844997e53266"
O_PAYLOAD = "fd5449766b4f1f693de2245933afa609a93a5c6b"
O_PAYLOAD_TREE = "491819895e46a8fccbfc2449c18868780d1a090b"
N12 = tuple(range(2000, 2012))
LJ75 = tuple(range(2000, 2108))
OOS_N12 = tuple(range(2012, 2024))
OOS_LJ75 = tuple(range(2012, 2120))
BOOTSTRAP_SEED = 20261123
BOOTSTRAP_DRAWS = 10_000
ENERGY_AGREEMENT = 1.0e-7
CLAIM_GEO = 0.932
ARM_ORDER = (
    ("A", "M", "A2"),
    ("A2", "M", "A"),
    ("M", "A2", "A"),
    ("A", "A2", "M"),
    ("A2", "A", "M"),
    ("M", "A", "A2"),
)
OOS_ORDER = (
    ("A", "M", "O"),
    ("O", "M", "A"),
    ("M", "A", "O"),
    ("O", "A", "M"),
    ("M", "O", "A"),
    ("A", "O", "M"),
)
OOS_CELLS = (
    "lj75-seed",
    "ase-lj98",
    "ase-lj104",
    "au55-seed",
    "lj38",
    "m38r6",
    "m55r14",
)
OOS_PAIRS = (("O", "M"), ("O", "A"), ("M", "A"))
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TBB_NUM_THREADS": "1",
    "TOKIO_WORKER_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}
FIELDS = (
    "cell",
    "arm",
    "seed",
    "round",
    "slot",
    "budget",
    "hit",
    "calls_to_hit",
    "charged",
    "walks",
    "best_energy",
    "wall_s",
    "potential_s",
    "outside_s",
    "outside_per_call_s",
    "peak_rss_anon_kb",
    "inner_reps",
    "trace_sha256",
    "arm_revision",
    "arm_tree",
    "binary_sha256",
)
BINARIES = ("ase_panel", "h2h_metal", "h2h_morse", "h2h_seed")
ORDER = (
    "lj38",
    "m38r6",
    "m55r14",
    "au55-seed",
    "ase-lj98",
    "ase-lj104",
    "lj75-seed",
)


@dataclass(frozen=True)
class Cell:
    name: str
    binary: str
    kind: str
    budget: int
    reference: str
    seeds: tuple[int, ...]
    system: str = ""
    n: int = 0
    rho: str = "0"


def _cell(name: str, binary: str, kind: str, budget: int, reference: str, seeds, **kwargs):
    return Cell(name, binary, kind, budget, reference, seeds, **kwargs)


CELLS = {
    cell.name: cell
    for cell in (
        _cell("lj38", "h2h_seed", "seed", 400_000, "-173.928427", N12, n=38),
        _cell("m38r6", "h2h_morse", "driver", 400_000, "-157.477108", N12, system="m38r6"),
        _cell("m55r14", "h2h_morse", "driver", 1_000_000, "-220.646208", N12, system="m55r14"),
        _cell("au55-seed", "h2h_metal", "driver", 1_000_000, "-191.03085280", N12, system="au55"),
        _cell("ase-lj98", "ase_panel", "ase", 4_000_000, "-543.665361", N12, system="lj98"),
        _cell("ase-lj104", "ase_panel", "ase", 4_000_000, "-582.086642", N12, system="lj104"),
        _cell("lj75-seed", "h2h_seed", "seed", 4_000_000, "-397.492331", LJ75, n=75),
    )
}


def stop_energy(reference: str) -> str:
    whole, frac = reference.split(".")
    scale = 10 ** len(frac)
    value = int(whole + frac) + scale // 1000
    sign = "-" if value < 0 else ""
    value = abs(value)
    text = f"{value:0{len(frac) + 1}d}"
    return sign + text[:-len(frac)] + "." + text[-len(frac):]


class Arm:
    def __init__(self, name: str, source: Path, binaries: Path, revision: str, tree: str) -> None:
        self.name = name
        self.revision = git(source, "rev-parse", "HEAD")
        self.tree = git(source, "rev-parse", "HEAD^{tree}")
        if (self.revision, self.tree) != (revision, tree):
            raise RuntimeError(f"{name} source identity mismatch")
        if git(source, "status", "--porcelain"):
            raise RuntimeError(f"{name} source tree is not clean")
        self.hashes = {binary: sha256(binaries / binary) for binary in BINARIES}
        self.binaries = binaries


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_")):
            del env[key]
    env.update(THREAD_ENV)
    return env


def command(cell: Cell, binary: Path, seed: int, scratch: Path) -> list[str]:
    if cell.kind == "ase":
        args = [str(binary), cell.system, str(cell.budget), str(seed), "recommended"]
    elif cell.kind == "seed":
        args = [
            str(binary),
            "lj",
            str(cell.n),
            cell.rho,
            str(cell.budget),
            str(seed),
            "recommended",
            stop_energy(cell.reference),
            str(scratch / "run"),
        ]
    else:
        args = [str(binary), "run", cell.system, str(seed)]
    return ["taskset", "-c", "0", *args]


def key_values(line: str) -> dict[str, str]:
    values = {}
    for token in line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = value
    return values


def parse_memory(stdout: str) -> int:
    rows = [key_values(line) for line in stdout.splitlines() if line.startswith("gate_memory ")]
    if len(rows) != 1:
        raise RuntimeError("expected one memory row")
    value = int(rows[0]["peak_rss_anon_kb"])
    if value <= 0:
        raise RuntimeError("RssAnon was not sampled")
    return value


def parse_walks(stdout: str) -> int:
    values = []
    for line in stdout.splitlines():
        if line.startswith("gate_walks "):
            values.append(int(key_values(line)["hops"]))
    if len(values) != 1:
        raise RuntimeError("expected one walk count")
    return values[0]


def parse_result(cell: Cell, stdout: str) -> dict[str, float | int]:
    if cell.kind == "ase":
        lines = [line for line in stdout.splitlines() if line.startswith("system=")]
        if len(lines) != 1:
            raise RuntimeError("expected one ASE result row")
        fields = dict(token.split("=", 1) for token in lines[0].split())
        hit = int(fields["hit"])
        calls = cell.budget + 1 if not hit else int(fields["first_calls"])
        return {
            "hit": hit,
            "calls": calls,
            "charged": int(fields["charged"]),
            "best": float(fields["best"]),
            "wall": float(fields["wall_s"]),
            "potential": float(fields["inside_s"]),
            "outside": float(fields["outside_s"]),
        }
    if cell.kind == "seed":
        lines = [key_values(line) for line in stdout.splitlines() if line.startswith("h2h_row ")]
        if len(lines) != 1:
            raise RuntimeError("expected one seeded result row")
        fields = lines[0]
    else:
        headers = [line for line in stdout.splitlines() if line.startswith("system,")]
        data = [line for line in stdout.splitlines() if line.startswith(cell.system + ",")]
        if len(headers) != 1 or len(data) != 1:
            raise RuntimeError("expected one driver result row")
        fields = dict(zip(headers[0].split(","), data[0].split(","), strict=True))
    hit = int(str(fields["hit"]).lower() in ("1", "true"))
    best = float(fields["best"])
    charged = int(fields["charged"])
    calls = cell.budget + 1 if not hit else int(fields["evals_to_hit"])
    wall = float(fields["wall_s"])
    potential = float(fields.get("pot_s", fields.get("inside_s", "0")))
    outside = float(fields.get("outside_s", str(max(0.0, wall - potential))))
    return {
        "hit": hit,
        "calls": calls,
        "charged": charged,
        "best": best,
        "wall": wall,
        "potential": potential,
        "outside": outside,
    }


def trace_digest(cell: Cell, scratch: Path) -> str:
    if cell.kind != "seed":
        return ""
    path = scratch / "run.trace.csv"
    if not path.is_file():
        raise RuntimeError("seeded run wrote no trace")
    return sha256(path)


def one_launch(cmd: list[str], env: dict[str, str]) -> str:
    proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    text = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(f"run exited {proc.returncode}\n{text[-2000:]}")
    return text


def run_sample(cell: Cell, arm: Arm, seed: int, scratch: Path, env: dict[str, str]) -> tuple[dict, str, int, str, int]:
    cmd = command(cell, arm.binaries / cell.binary, seed, scratch)
    text = one_launch(cmd, env)
    parsed = parse_result(cell, text)
    rss = parse_memory(text)
    walks = parse_walks(text)
    digest = trace_digest(cell, scratch)
    samples = [(parsed, rss)]
    transcript = [" ".join(cmd), text]
    if float(parsed["wall"]) < 0.05:
        repetitions = max(5, math.ceil(0.25 / max(float(parsed["wall"]), 0.001)))
        for _ in range(repetitions - 1):
            current_text = one_launch(cmd, env)
            current = parse_result(cell, current_text)
            if (
                int(current["hit"]) != int(parsed["hit"])
                or int(current["calls"]) != int(parsed["calls"])
                or abs(float(current["best"]) - float(parsed["best"])) > ENERGY_AGREEMENT
                or parse_walks(current_text) != walks
                or trace_digest(cell, scratch) != digest
            ):
                raise RuntimeError("inner repetitions changed the deterministic result")
            samples.append((current, parse_memory(current_text)))
            transcript.append(current_text)
    combined = dict(parsed)
    for name in ("wall", "potential", "outside"):
        combined[name] = statistics.fmean(float(row[name]) for row, _ in samples)
    return combined, "\n".join(transcript), max(value for _, value in samples), digest, walks


def append_csv(path: Path, fields: tuple[str, ...], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def append_session(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def durable_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def arms_from(args: argparse.Namespace) -> dict[str, Arm]:
    return {
        "A": Arm("A", Path(args.source_a), Path(args.bin_a), A_REV, A_TREE),
        "A2": Arm("A2", Path(args.source_a2), Path(args.bin_a2), A_REV, A_TREE),
        "M": Arm("M", Path(args.source_m), Path(args.bin_m), M_REV, M_TREE),
    }


def run_cell(args: argparse.Namespace, cell: Cell, deadline: float, built=None, cycle=None) -> int:
    if built is None:
        built = arms_from(args)
    if cycle is None:
        cycle = ARM_ORDER
    rows_path = Path(args.rows)
    sessions_path = Path(args.sessions)
    logs = Path(args.logs)
    scratch_root = Path(args.scratch)
    rows = load_rows(rows_path)
    done = {(row["cell"], row["arm"], row["seed"]) for row in rows}
    for row in rows:
        if row["cell"] != cell.name:
            continue
        arm = built[row["arm"]]
        if row["binary_sha256"] != arm.hashes[cell.binary]:
            raise RuntimeError("binary digest differs from the earlier row")
        if (row["arm_revision"], row["arm_tree"]) != (arm.revision, arm.tree):
            raise RuntimeError("source identity differs from the earlier row")
    env = child_env()
    for round_index, seed in enumerate(cell.seeds):
        if any((cell.name, arm_name, str(seed)) not in done for arm_name in cycle[round_index % len(cycle)]):
            if time.time() >= deadline:
                print(f"{cell.name} stopped before seed {seed}: deadline", flush=True)
                return 0
        order = cycle[round_index % len(cycle)]
        for slot, arm_name in enumerate(order, start=1):
            key = (cell.name, arm_name, str(seed))
            if key in done:
                continue
            arm = built[arm_name]
            scratch = scratch_root / cell.name / f"{seed}-{arm_name}"
            scratch.mkdir(parents=True, exist_ok=True)
            result, transcript, rss, digest, walks = run_sample(cell, arm, seed, scratch, env)
            log = logs / cell.name / f"{seed}-{arm_name}.log"
            durable_text(log, transcript)
            charged = int(result["charged"])
            outside = float(result["outside"])
            row = {
                "cell": cell.name,
                "arm": arm_name,
                "seed": str(seed),
                "round": str(round_index + 1),
                "slot": str(slot),
                "budget": str(cell.budget),
                "hit": str(int(result["hit"])),
                "calls_to_hit": str(int(result["calls"])),
                "charged": str(charged),
                "walks": str(walks),
                "best_energy": f"{float(result['best']):.10f}",
                "wall_s": f"{float(result['wall']):.9f}",
                "potential_s": f"{float(result['potential']):.9f}",
                "outside_s": f"{outside:.9f}",
                "outside_per_call_s": f"{outside / max(1, charged):.12e}",
                "peak_rss_anon_kb": str(rss),
                "inner_reps": str(transcript.count("gate_memory ")),
                "trace_sha256": digest,
                "arm_revision": arm.revision,
                "arm_tree": arm.tree,
                "binary_sha256": arm.hashes[cell.binary],
            }
            append_csv(rows_path, FIELDS, row)
            session = dict(row)
            session["utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            append_session(sessions_path, session)
            done.add(key)
            print(
                f"{cell.name} round={round_index + 1} arm={arm_name} "
                f"hit={row['hit']} calls={row['calls_to_hit']} "
                f"outside={row['outside_s']} rss={rss}",
                flush=True,
            )
    return 0


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_values(draw) -> tuple[float, float, float]:
    point = draw(None)
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [draw(rng) for _ in range(BOOTSTRAP_DRAWS)]
    return point, quantile(draws, 0.025), quantile(draws, 0.975)


def safe_geomean(left: list[float], right: list[float]) -> tuple[float, float, float, int]:
    try:
        return ratio_geomean(left, right)
    except RuntimeError:
        return (float("nan"), float("nan"), float("nan"), 0)


def ratio_geomean(left: list[float], right: list[float]) -> tuple[float, float, float, int]:
    pairs = [(a, b) for a, b in zip(left, right, strict=True) if a > 0 and b > 0]
    if not pairs:
        raise RuntimeError("no positive pairs")

    def draw(rng: random.Random | None) -> float:
        chosen = pairs
        if rng is not None:
            chosen = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        logs = [math.log(a / b) for a, b in chosen]
        return math.exp(statistics.fmean(logs))

    estimate, low, high = bootstrap_values(draw)
    return estimate, low, high, len(pairs)


def ratio_of_medians(left: list[float], right: list[float]) -> tuple[float, float, float]:
    if len(left) != len(right) or not left:
        raise RuntimeError("median ratio has incomplete rounds")

    def draw(rng: random.Random | None) -> float:
        if rng is None:
            sample_l, sample_r = left, right
        else:
            idx = [rng.randrange(len(left)) for _ in left]
            sample_l = [left[i] for i in idx]
            sample_r = [right[i] for i in idx]
        return statistics.median(sample_l) / statistics.median(sample_r)

    return bootstrap_values(draw)


def log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_less(hits_m: int, n_m: int, hits_a: int, n_a: int) -> float:
    col1 = hits_m + hits_a
    lo = max(0, col1 - n_a)
    hi = min(n_m, col1)
    logs = [log_comb(n_m, i) + log_comb(n_a, col1 - i) for i in range(lo, hi + 1)]
    top = max(logs)
    weights = [math.exp(value - top) for value in logs]
    total = sum(weights)
    keep = sum(weight for i, weight in zip(range(lo, hi + 1), weights) if i <= hits_m)
    return keep / total


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def mann_whitney_worse(m_values: list[float], a_values: list[float]) -> tuple[float, float]:
    pooled = m_values + a_values
    ranks = rankdata(pooled)
    n1 = len(m_values)
    n2 = len(a_values)
    u = sum(ranks[:n1]) - n1 * (n1 + 1) / 2.0
    counts: dict[float, int] = {}
    for value in pooled:
        counts[value] = counts.get(value, 0) + 1
    total = n1 + n2
    tie = sum(count**3 - count for count in counts.values())
    variance = n1 * n2 / 12.0 * (total + 1.0 - tie / (total * (total - 1.0)))
    if variance <= 0:
        return u, 1.0
    z = (u - n1 * n2 / 2.0 - 0.5) / math.sqrt(variance)
    return u, 0.5 * math.erfc(z / math.sqrt(2.0))


def ci_text(ci: tuple[float, float, float]) -> str:
    return f"{ci[0]:.4f} [{ci[1]:.4f}, {ci[2]:.4f}]"


def grouped(rows: list[dict[str, str]], cell: str, arm: str) -> list[dict[str, str]]:
    return sorted((row for row in rows if row["cell"] == cell and row["arm"] == arm), key=lambda row: int(row["seed"]))


def judge(complete: bool, hit_loss: bool, fisher: float, calls_loss: bool, calls_real: bool, outside_real: bool, rss_real: bool, outside_miss: bool, rss_miss: bool) -> str:
    if not complete:
        return "UNRESOLVED"
    real = (hit_loss and fisher < 0.05) or calls_real or outside_real or rss_real
    missed = hit_loss or calls_loss or outside_miss or rss_miss
    if real:
        return "FIX"
    if missed:
        return "UNRESOLVED"
    return "PASS"


def analyze(args: argparse.Namespace) -> int:
    rows = load_rows(Path(args.rows))
    lines = [
        "# PR 168 walklen3 gate, shard B",
        "",
        f"Measured tree `{M_TREE}`.",
        f"Develop `{A_REV}` (tree `{A_TREE}`) is the measured base.",
        "It is not an ancestor of payload `4ab2681ae6bf72d61c633197e88047f00cdd7efa`,",
        "and the payload is not an ancestor of develop. `git merge-tree --write-tree`",
        f"exited 0 and wrote `{M_TREE}`. A is that develop commit, built twice.",
        f"M is scratch merge `{M_REV}` of that tree. The scratch merge is not in the result history.",
        "",
        "Bootstrap: 10000 paired resamples, seed 20261123, type-7 percentiles.",
        "A fresh generator is used for each interval. A miss counts as budget plus one.",
        "",
        "| cell | n | hits A/A2/M | lost A hits | Fisher p | MWU p | geo calls M/A | median calls A/A2/M | median ratio M/A | outside M/A | outside A2/A | RSS M/A | calls/walk M/A | trajectories | verdict |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    details: list[str] = []
    overall = "PASS"
    rank = {"PASS": 0, "UNRESOLVED": 1, "FIX": 2}
    for name in ORDER:
        cell = CELLS[name]
        groups = {arm: grouped(rows, cell.name, arm) for arm in ("A", "A2", "M")}
        counts = {arm: len(groups[arm]) for arm in groups}
        target = len(cell.seeds)
        complete = all(counts[arm] == target for arm in counts) and all(
            int(row["peak_rss_anon_kb"]) > 0 for group in groups.values() for row in group
        )
        reached = min(counts.values()) if counts else 0
        if not groups["A"]:
            lines.append(f"| {cell.name} | 0/{target} |  |  |  |  |  |  |  |  |  |  |  |  | UNRESOLVED |")
            claim = ""
            if cell.name == "lj75-seed":
                claim = " The calls-to-hit geometric mean at seeds 2000-2011 was not measured, so the claim 0.932 is unchecked."
            details.extend(
                [
                    "",
                    f"## {cell.name}",
                    "",
                    "Verdict: UNRESOLVED.",
                    "",
                    f"Rounds reached: 0 of {target}.{claim}",
                    "",
                ]
            )
            overall = "UNRESOLVED" if rank[overall] < rank["UNRESOLVED"] else overall
            continue
        # Score the longest aligned prefix of complete rounds. A trailing
        # partial round is not included.
        aligned = []
        for seed in (str(seed) for seed in cell.seeds):
            if all(any(row["seed"] == seed for row in groups[arm]) for arm in groups):
                aligned.append(seed)
            else:
                break
        if not aligned:
            lines.append(
                f"| {cell.name} | 0/{target} |  |  |  |  |  |  |  |  |  |  |  |  | UNRESOLVED |"
            )
            details.extend(
                ["", f"## {cell.name}", "", "Verdict: UNRESOLVED.", "", f"Rounds reached: 0 of {target}.", ""]
            )
            overall = "UNRESOLVED" if rank[overall] < rank["UNRESOLVED"] else overall
            continue
        picked = {
            arm: [next(row for row in groups[arm] if row["seed"] == seed) for seed in aligned]
            for arm in groups
        }
        n = len(aligned)
        hits = {arm: sum(int(row["hit"]) for row in picked[arm]) for arm in picked}
        lost = [
            a_row["seed"]
            for a_row, m_row in zip(picked["A"], picked["M"], strict=True)
            if int(a_row["hit"]) == 1 and int(m_row["hit"]) == 0
        ]
        fisher = fisher_less(hits["M"], n, hits["A"], n) if n else float("nan")
        calls = {arm: [float(row["calls_to_hit"]) for row in picked[arm]] for arm in picked}
        u_stat, mw = mann_whitney_worse(calls["M"], calls["A"]) if n else (float("nan"), float("nan"))
        geo = safe_geomean(calls["M"], calls["A"])
        med = {arm: statistics.median(calls[arm]) for arm in calls}
        med_ratio = ratio_of_medians(calls["M"], calls["A"])
        med_ratio_a2 = ratio_of_medians(calls["A2"], calls["A"])
        outside_m = safe_geomean(
            [float(row["outside_s"]) for row in picked["M"]],
            [float(row["outside_s"]) for row in picked["A"]],
        )
        outside_a2 = safe_geomean(
            [float(row["outside_s"]) for row in picked["A2"]],
            [float(row["outside_s"]) for row in picked["A"]],
        )
        rss_m = safe_geomean(
            [float(row["peak_rss_anon_kb"]) for row in picked["M"]],
            [float(row["peak_rss_anon_kb"]) for row in picked["A"]],
        )
        rss_a2 = safe_geomean(
            [float(row["peak_rss_anon_kb"]) for row in picked["A2"]],
            [float(row["peak_rss_anon_kb"]) for row in picked["A"]],
        )
        walk_pairs = []
        for a_row, m_row in zip(picked["A"], picked["M"], strict=True):
            a_walks = float(a_row["walks"])
            m_walks = float(m_row["walks"])
            if a_walks > 0 and m_walks > 0:
                walk_pairs.append((float(m_row["charged"]) / m_walks, float(a_row["charged"]) / a_walks))
        if walk_pairs:
            calls_per_walk = safe_geomean(
                [left for left, _ in walk_pairs],
                [right for _, right in walk_pairs],
            )
        else:
            calls_per_walk = (float("nan"), float("nan"), float("nan"), 0)
        hit_loss = hits["M"] < hits["A"]
        calls_loss = med["M"] > med["A"]
        calls_real = calls_loss and med_ratio[1] > 1 and med_ratio[1] > med_ratio_a2[2]
        outside_miss = outside_m[2] > 1.10
        rss_miss = rss_m[2] > 1.10
        outside_real = outside_miss and outside_m[1] > 1 and outside_m[1] > outside_a2[2]
        rss_real = rss_miss and rss_m[1] > 1 and rss_m[1] > rss_a2[2]
        full = complete and n == target and outside_m[3] > 0 and rss_m[3] > 0
        verdict = judge(full, hit_loss, fisher, calls_loss, calls_real, outside_real, rss_real, outside_miss, rss_miss)
        if rank[verdict] > rank[overall]:
            overall = verdict
        identical = 0
        comparable = 0
        differ = []
        for a_row, m_row in zip(picked["A"], picked["M"], strict=True):
            if a_row["trace_sha256"] and m_row["trace_sha256"]:
                comparable += 1
                if a_row["trace_sha256"] == m_row["trace_sha256"]:
                    identical += 1
                else:
                    differ.append(a_row["seed"])
        if comparable:
            trajectory = f"{identical}/{comparable} bit-identical"
        else:
            trajectory = "no trace file"
        lines.append(
            f"| {cell.name} | {n}/{target} | {hits['A']}/{hits['A2']}/{hits['M']} | {len(lost)} | "
            f"{fisher:.4g} | {mw:.4g} | {ci_text(geo[:3])} | "
            f"{med['A']:.1f}/{med['A2']:.1f}/{med['M']:.1f} | {ci_text(med_ratio)} | "
            f"{ci_text(outside_m[:3])} | {ci_text(outside_a2[:3])} | {ci_text(rss_m[:3])} | "
            f"{'undefined' if calls_per_walk[3] == 0 else ci_text(calls_per_walk[:3])} | {trajectory} | {verdict} |"
        )
        reasons = []
        if not full:
            reasons.append(f"rounds {n} of {target}")
        if hit_loss:
            reasons.append("hit loss" + (" real" if fisher < 0.05 else " not real"))
        if calls_loss:
            reasons.append("calls-to-hit loss" + (" real" if calls_real else " not real"))
        if outside_miss:
            reasons.append("outside upper above 1.10" + (" real" if outside_real else " not real"))
        if rss_miss:
            reasons.append("RSS upper above 1.10" + (" real" if rss_real else " not real"))
        details.extend(
            [
                "",
                f"## {cell.name}",
                "",
                f"Verdict: {verdict}.",
                "",
                "Notes: " + ("; ".join(reasons) if reasons else "every bar holds") + ".",
                "",
                f"Lost A hits: {', '.join(lost) if lost else 'none'}.",
                f"Mann-Whitney U (M larger than A): {u_stat:.1f}.",
                f"RSS A2/A: {ci_text(rss_a2[:3])}.",
                f"Calls-to-hit median ratio A2/A: {ci_text(med_ratio_a2)}.",
                f"Calls per walk uses {calls_per_walk[3]} seeds with a positive hop count on A and M.",
                "Binary sha256 "
                + " ".join(f"{arm} {picked[arm][0]['binary_sha256']}" for arm in ("A", "A2", "M"))
                + ".",
                "Revision "
                + " ".join(f"{arm} {picked[arm][0]['arm_revision']}" for arm in ("A", "M"))
                + ".",
            ]
        )
        if cell.name == "lj75-seed" and n >= 12:
            early_m = calls["M"][:12]
            early_a = calls["A"][:12]
            early = safe_geomean(early_m, early_a)
            rounded = f"{early[0]:.3f}"
            details.extend(
                [
                    "",
                    f"Seeds 2000-2011 calls-to-hit geometric mean M/A: {ci_text(early[:3])}.",
                    f"The claim 0.932 {'matches' if rounded == '0.932' else 'does not match'} this value at three decimals "
                    f"(measured {early[0]:.6f}).",
                ]
            )
        both = [
            (a_row, m_row)
            for a_row, m_row in zip(picked["A"], picked["M"], strict=True)
            if int(a_row["hit"]) and int(m_row["hit"])
        ]
        if cell.name == "lj75-seed" and both:
            to_hit = safe_geomean(
                [float(m_row["outside_s"]) for _, m_row in both],
                [float(a_row["outside_s"]) for a_row, _ in both],
            )
            one_sided = [
                a_row["seed"]
                for a_row, m_row in zip(picked["A"], picked["M"], strict=True)
                if a_row["hit"] != m_row["hit"]
            ]
            details.extend(
                [
                    "",
                    "Paired outside time to hit, seeds hit by both A and M: "
                    + ci_text(to_hit[:3])
                    + f" (n={to_hit[3]}).",
                    "Seeds hit by only one of A and M: " + (", ".join(one_sided) if one_sided else "none") + ".",
                ]
            )
        if comparable:
            a2_same = sum(
                a_row["trace_sha256"] == a2_row["trace_sha256"]
                for a_row, a2_row in zip(picked["A"], picked["A2"], strict=True)
                if a_row["trace_sha256"] and a2_row["trace_sha256"]
            )
            details.extend(
                [
                    "",
                    f"Trace files bit-identical to A: {identical} of {comparable}.",
                    "Seeds whose trace differs: " + (", ".join(differ) if differ else "none") + ".",
                    f"A2 traces bit-identical to A: {a2_same} of {comparable}.",
                ]
            )
        else:
            energy_same = sum(
                a_row["best_energy"] == m_row["best_energy"]
                for a_row, m_row in zip(picked["A"], picked["M"], strict=True)
            )
            details.extend(
                [
                    "",
                    "This driver does not write a call trajectory.",
                    f"Printed best-energy strings match on {energy_same} of {n} seeds.",
                ]
            )
        header = (
            "| seed | A best | A hit | M best | M hit | A outside | M outside | M/A | trace | moved |"
            if cell.name == "lj75-seed"
            else "| seed | A best | A hit | M best | M hit | trace | moved |"
        )
        rule = "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |" if cell.name == "lj75-seed" else "| ---: | ---: | ---: | ---: | ---: | --- | --- |"
        details.extend(["", header, rule])
        for a_row, m_row in zip(picked["A"], picked["M"], strict=True):
            moved = a_row["hit"] != m_row["hit"] or abs(float(a_row["best_energy"]) - float(m_row["best_energy"])) > ENERGY_AGREEMENT
            if a_row["trace_sha256"] and m_row["trace_sha256"]:
                trace = "identical" if a_row["trace_sha256"] == m_row["trace_sha256"] else "different"
            else:
                trace = "none"
            if cell.name == "lj75-seed":
                a_time = float(a_row["outside_s"]) if int(a_row["hit"]) else None
                m_time = float(m_row["outside_s"]) if int(m_row["hit"]) else None
                ratio = f"{m_time / a_time:.4f}" if a_time not in (None, 0.0) and m_time is not None else "-"
                details.append(
                    f"| {a_row['seed']} | {a_row['best_energy']} | {a_row['hit']} | {m_row['best_energy']} | {m_row['hit']} | "
                    f"{'-' if a_time is None else f'{a_time:.6f}'} | {'-' if m_time is None else f'{m_time:.6f}'} | {ratio} | {trace} | "
                    f"{'yes' if moved else 'no'} |"
                )
            else:
                details.append(
                    f"| {a_row['seed']} | {a_row['best_energy']} | {a_row['hit']} | {m_row['best_energy']} | {m_row['hit']} | {trace} | "
                    f"{'yes' if moved else 'no'} |"
                )
        details.append("")
    lines.extend(["", f"Overall: **{overall}**.", "", "## Sampler", ""])
    lines.extend(
        [
            "A, A2, and M carry the same measurement-only sampler. It reads",
            "`RssAnon` from `/proc/self/status` at search start, at powers of",
            "two in the basin count, at search end, and at process exit, and",
            "reports the largest value. The three source files are restored",
            "after compilation. The sampler is the resident-set measurement",
            "on every arm.",
            "",
        ]
    )
    lines.extend(details)
    durable_text(Path(args.report), "\n".join(lines) + "\n")
    print(overall)
    return 0


def wilson(hits: int, n: int) -> tuple[float, float]:
    if n <= 0:
        raise RuntimeError("wilson interval needs rounds")
    z = 1.959963984540054
    proportion = hits / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (proportion + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt(proportion * (1.0 - proportion) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def oos_cell(name: str) -> Cell:
    cell = CELLS[name]
    seeds = OOS_LJ75 if name == "lj75-seed" else OOS_N12
    return Cell(
        cell.name,
        cell.binary,
        cell.kind,
        cell.budget,
        cell.reference,
        seeds,
        system=cell.system,
        n=cell.n,
        rho=cell.rho,
    )


def oos_arms(args: argparse.Namespace) -> dict[str, Arm]:
    return {
        "A": Arm("A", Path(args.source_a), Path(args.bin_a), A_REV, A_TREE),
        "M": Arm("M", Path(args.source_m), Path(args.bin_m), M_REV, M_TREE),
        "O": Arm("O", Path(args.source_o), Path(args.bin_o), O_REV, O_TREE),
    }


def pair_scores(picked: dict[str, list[dict[str, str]]], challenger: str, reference: str) -> dict[str, object]:
    left = picked[challenger]
    right = picked[reference]
    n = len(left)
    hits_left = sum(int(row["hit"]) for row in left)
    hits_right = sum(int(row["hit"]) for row in right)
    calls_left = [float(row["calls_to_hit"]) for row in left]
    calls_right = [float(row["calls_to_hit"]) for row in right]
    wall_left = [float(row["outside_s"]) for row in left]
    wall_right = [float(row["outside_s"]) for row in right]
    rss_left = [float(row["peak_rss_anon_kb"]) for row in left]
    rss_right = [float(row["peak_rss_anon_kb"]) for row in right]
    _u_calls, calls_p = mann_whitney_worse(calls_left, calls_right)
    _u_wall, wall_p = mann_whitney_worse(wall_left, wall_right)
    _u_rss, rss_p = mann_whitney_worse(rss_left, rss_right)
    calls_ratio = ratio_of_medians(calls_left, calls_right)
    wall_ratio = ratio_of_medians(wall_left, wall_right)
    rss_ratio = ratio_of_medians(rss_left, rss_right)
    hit_loss = hits_left < hits_right
    calls_loss = statistics.median(calls_left) > statistics.median(calls_right)
    wall_miss = wall_ratio[2] > 1.10
    rss_miss = rss_ratio[2] > 1.10
    return {
        "n": n,
        "fisher": fisher_less(hits_left, n, hits_right, n),
        "calls_p": calls_p,
        "wall_p": wall_p,
        "rss_p": rss_p,
        "calls_ratio": calls_ratio,
        "wall_ratio": wall_ratio,
        "rss_ratio": rss_ratio,
        "hit_loss": hit_loss,
        "calls_loss": calls_loss,
        "calls_real": calls_loss and calls_ratio[1] > 1,
        "wall_miss": wall_miss,
        "rss_miss": rss_miss,
        "wall_real": wall_miss and wall_ratio[1] > 1,
        "rss_real": rss_miss and rss_ratio[1] > 1,
    }


def analyze_oos(args: argparse.Namespace) -> int:
    rows = load_rows(Path(args.rows))
    for row in rows:
        if int(row["seed"]) < 2012 or row["arm"] not in ("A", "M", "O"):
            raise RuntimeError("out-of-sample file contains an in-sample row")
    lines = [
        "# PR 168 walklen3 gate, shard B, out of sample",
        "",
        f"Measured base `{A_REV}`, tree `{A_TREE}`.",
        f"M is scratch merge `{M_REV}`, tree `{M_TREE}`.",
        f"`git merge-tree --write-tree` of the base and payload `{O_PAYLOAD}`",
        f"(tree `{O_PAYLOAD_TREE}`) exited 0 and wrote `{O_TREE}`.",
        f"O is scratch merge `{O_REV}` of that tree. Neither scratch merge is in the result history.",
        "",
        "Seeds 2000-2011 stay in the in-sample A-versus-M record and are not inputs here.",
        "This file uses seeds 2012-2119 for lj75-seed and seeds 2012-2023 for the other cells.",
        "lj75-seed runs first. The arm order repeats AMO, OMA, MAO, OAM, MOA, AOM.",
        "",
        "Bootstrap: 10000 paired resamples, seed 20261123, type-7 percentiles.",
        "A fresh generator is used for each interval. A miss counts as budget plus one.",
        "The ratio is the median of the first arm divided by the median of the second.",
        "The bar is no hit loss, no calls-to-hit loss, and an upper interval at most 1.10",
        "on outside time and on RSS. A hit loss is real when the one-sided Fisher",
        "probability is below 0.05. A calls, outside, or RSS loss is real only when",
        "its lower interval is above 1. This round has no second build of the base.",
        "",
        "| cell | arm | hits | Wilson 95% | median calls | median outside | median RSS |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    details: list[str] = []
    pair_lines = [
        "",
        "| cell | pair | Fisher p | MWU calls | MWU outside | MWU RSS | calls ratio | outside ratio | RSS ratio | verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    overall = "PASS"
    rank = {"PASS": 0, "UNRESOLVED": 1, "FIX": 2}
    for name in OOS_CELLS:
        cell = oos_cell(name)
        groups = {arm: grouped(rows, cell.name, arm) for arm in ("A", "M", "O")}
        target = len(cell.seeds)
        complete = all(len(groups[arm]) == target for arm in groups) and all(
            int(row["peak_rss_anon_kb"]) > 0 for group in groups.values() for row in group
        )
        aligned = []
        for seed in (str(seed) for seed in cell.seeds):
            if all(any(row["seed"] == seed for row in groups[arm]) for arm in groups):
                aligned.append(seed)
            else:
                break
        if not aligned:
            for arm in ("A", "M", "O"):
                lines.append(f"| {cell.name} | {arm} | 0/{target} |  |  |  |  |")
            for challenger, reference in OOS_PAIRS:
                pair_lines.append(
                    f"| {cell.name} | {challenger}/{reference} |  |  |  |  |  |  |  | UNRESOLVED |"
                )
            details.extend(
                [
                    "",
                    f"## {cell.name}",
                    "",
                    "Verdict A: reference.",
                    "Verdict M versus A: UNRESOLVED.",
                    "Verdict O versus A: UNRESOLVED.",
                    "Verdict O versus M: UNRESOLVED.",
                    "",
                    f"Rounds reached: 0 of {target}.",
                    "",
                ]
            )
            if rank["UNRESOLVED"] > rank[overall]:
                overall = "UNRESOLVED"
            continue
        picked = {
            arm: [next(row for row in groups[arm] if row["seed"] == seed) for seed in aligned]
            for arm in groups
        }
        n = len(aligned)
        for arm in ("A", "M", "O"):
            hits = sum(int(row["hit"]) for row in picked[arm])
            low, high = wilson(hits, n)
            calls = statistics.median(float(row["calls_to_hit"]) for row in picked[arm])
            wall = statistics.median(float(row["outside_s"]) for row in picked[arm])
            rss = statistics.median(float(row["peak_rss_anon_kb"]) for row in picked[arm])
            lines.append(
                f"| {cell.name} | {arm} | {hits}/{n} | [{low:.4f}, {high:.4f}] | "
                f"{calls:.1f} | {wall:.6f} | {rss:.1f} |"
            )
        full = complete and n == target
        scored = {}
        for challenger, reference in OOS_PAIRS:
            scored[(challenger, reference)] = pair_scores(picked, challenger, reference)
            item = scored[(challenger, reference)]
            verdict = judge(
                full,
                bool(item["hit_loss"]),
                float(item["fisher"]),
                bool(item["calls_loss"]),
                bool(item["calls_real"]),
                bool(item["wall_real"]),
                bool(item["rss_real"]),
                bool(item["wall_miss"]),
                bool(item["rss_miss"]),
            )
            item["verdict"] = verdict
            if rank[verdict] > rank[overall]:
                overall = verdict
            pair_lines.append(
                f"| {cell.name} | {challenger}/{reference} | {float(item['fisher']):.4g} | "
                f"{float(item['calls_p']):.4g} | {float(item['wall_p']):.4g} | {float(item['rss_p']):.4g} | "
                f"{ci_text(item['calls_ratio'])} | {ci_text(item['wall_ratio'])} | "
                f"{ci_text(item['rss_ratio'])} | {verdict} |"
            )
        details.extend(["", f"## {cell.name}", ""])
        for label, key in (
            ("M versus A", ("M", "A")),
            ("O versus A", ("O", "A")),
            ("O versus M", ("O", "M")),
        ):
            item = scored[key]
            reasons = []
            if not full:
                reasons.append(f"rounds {n} of {target}")
            if item["hit_loss"]:
                reasons.append("hit loss" + (" real" if float(item["fisher"]) < 0.05 else " not real"))
            if item["calls_loss"]:
                reasons.append("calls-to-hit loss" + (" real" if item["calls_real"] else " not real"))
            if item["wall_miss"]:
                reasons.append("outside upper above 1.10" + (" real" if item["wall_real"] else " not real"))
            if item["rss_miss"]:
                reasons.append("RSS upper above 1.10" + (" real" if item["rss_real"] else " not real"))
            details.append(f"Verdict {label}: {item['verdict']}.")
            details.append("Notes: " + ("; ".join(reasons) if reasons else "every bar holds") + ".")
        details.append("")
        details.append(
            "Binary sha256 "
            + " ".join(f"{arm} {picked[arm][0]['binary_sha256']}" for arm in ("A", "M", "O"))
            + "."
        )
        details.append(
            "Revision " + " ".join(f"{arm} {picked[arm][0]['arm_revision']}" for arm in ("A", "M", "O")) + "."
        )
        if cell.kind == "seed":
            for challenger in ("M", "O"):
                identical = 0
                differ = []
                for base, other in zip(picked["A"], picked[challenger], strict=True):
                    if base["trace_sha256"] and other["trace_sha256"]:
                        if base["trace_sha256"] == other["trace_sha256"]:
                            identical += 1
                        else:
                            differ.append(base["seed"])
                details.append(f"{challenger} traces bit-identical to A: {identical} of {n}.")
                if differ:
                    details.append("Seeds whose trace differs: " + ", ".join(differ) + ".")
        else:
            same = 0
            for base, other_m, other_o in zip(picked["A"], picked["M"], picked["O"], strict=True):
                if base["best_energy"] == other_m["best_energy"] == other_o["best_energy"]:
                    same += 1
            details.append("This driver does not write a call trajectory.")
            details.append(f"Printed best-energy strings match on {same} of {n} seeds.")
        details.append("")
    lines.extend(pair_lines)
    lines.extend(["", f"Overall: **{overall}**.", ""])
    lines.extend(details)
    durable_text(Path(args.report), "\n".join(lines) + "\n")
    print(overall)
    return 0


def self_test() -> int:
    assert stop_energy("-397.492331") == "-397.491331"
    assert stop_energy("-173.928427") == "-173.927427"
    assert stop_energy("-157.477108") == "-157.476108"
    assert stop_energy("-220.646208") == "-220.645208"
    assert abs(quantile([1.0, 2.0, 3.0], 0.5) - 2.0) < 1e-12
    assert abs(fisher_less(0, 12, 12, 12) - 1 / math.comb(24, 12)) < 1e-12
    assert mann_whitney_worse([1, 2, 3], [1, 2, 3])[1] > 0.4
    assert rankdata([2, 1, 2]) == [2.5, 1.0, 2.5]
    assert ARM_ORDER[0] == ("A", "M", "A2")
    assert ARM_ORDER[1] == tuple(reversed(ARM_ORDER[0]))
    assert judge(False, False, 1, False, False, False, False, False, False) == "UNRESOLVED"
    assert judge(True, False, 1, False, False, False, False, False, False) == "PASS"
    assert judge(True, True, 0.01, False, False, False, False, False, False) == "FIX"
    assert judge(True, False, 1, False, False, False, False, True, False) == "UNRESOLVED"
    assert judge(True, False, 1, False, False, True, False, True, False) == "FIX"
    low, high = wilson(5, 10)
    assert abs(low - 0.236593) < 1e-4
    assert abs(high - 0.763407) < 1e-4
    assert wilson(0, 12)[0] < 1e-12
    assert OOS_ORDER[0] == ("A", "M", "O")
    assert OOS_ORDER[1] == ("O", "M", "A")
    assert len({perm for perm in OOS_ORDER}) == 6
    assert OOS_CELLS[0] == "lj75-seed"
    assert OOS_LJ75[0] == 2012 and OOS_LJ75[-1] == 2119 and len(OOS_LJ75) == 108
    assert OOS_N12[0] == 2012 and OOS_N12[-1] == 2023 and len(OOS_N12) == 12
    slots = {0: [], 1: [], 2: []}
    for perm in OOS_ORDER:
        for index, arm in enumerate(perm):
            slots[index].append(arm)
    for slot in slots.values():
        assert slot.count("A") == slot.count("M") == slot.count("O") == 2
    geo = ratio_geomean([2, 8], [1, 2])
    assert abs(geo[0] - 2.0 ** 1.5) < 1e-9
    print("ok")
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    sub = top.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run-all")
    run.add_argument("--source-a", required=True)
    run.add_argument("--source-a2", required=True)
    run.add_argument("--source-m", required=True)
    run.add_argument("--bin-a", required=True)
    run.add_argument("--bin-a2", required=True)
    run.add_argument("--bin-m", required=True)
    run.add_argument("--rows", required=True)
    run.add_argument("--sessions", required=True)
    run.add_argument("--logs", required=True)
    run.add_argument("--scratch", required=True)
    run.add_argument("--hours", type=float, default=4.0)
    report = sub.add_parser("analyze")
    report.add_argument("--rows", required=True)
    report.add_argument("--report", required=True)
    oos = sub.add_parser("run-oos")
    oos.add_argument("--source-a", required=True)
    oos.add_argument("--source-m", required=True)
    oos.add_argument("--source-o", required=True)
    oos.add_argument("--bin-a", required=True)
    oos.add_argument("--bin-m", required=True)
    oos.add_argument("--bin-o", required=True)
    oos.add_argument("--rows", required=True)
    oos.add_argument("--sessions", required=True)
    oos.add_argument("--logs", required=True)
    oos.add_argument("--scratch", required=True)
    oos.add_argument("--hours", type=float, default=12.0)
    oos_report = sub.add_parser("analyze-oos")
    oos_report.add_argument("--rows", required=True)
    oos_report.add_argument("--report", required=True)
    sub.add_parser("self-test")
    return top


def main() -> int:
    args = parser().parse_args()
    if args.action == "self-test":
        return self_test()
    if args.action == "analyze":
        return analyze(args)
    if args.action == "analyze-oos":
        return analyze_oos(args)
    if args.action == "run-oos":
        if Path(args.rows).name == "rows.csv":
            raise RuntimeError("out-of-sample rows go in their own file")
        built = oos_arms(args)
        deadline = time.time() + args.hours * 3600
        for name in OOS_CELLS:
            run_cell(args, oos_cell(name), deadline, built, OOS_ORDER)
        return 0
    deadline = time.time() + args.hours * 3600
    for name in ORDER:
        run_cell(args, CELLS[name], deadline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
