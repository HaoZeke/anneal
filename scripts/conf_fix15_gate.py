#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Run and report the conf-fix15 same-session A/B gate.

The runner has four commands:

* ``panel1`` runs the molecular A/B and A/A-prime comparisons.
* ``score-panel1`` applies the repository's conformer polishing and hit rule.
* ``panel2`` runs cluster identity checks plus five A placement comparisons.
* ``report`` validates the schedule and writes the committed records.

Every benchmark subprocess is serial, pinned to core zero, and receives one
thread for each common numerical runtime. Run directories and process logs are
scratch data; only the sanitized records written by ``report`` are committed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SEEDS = tuple(range(2000, 2012))
SYSTEMS1 = ("rq3", "915", "xld", "fluvastatin", "z34", "amprenavir", "lopinavir", "lorlatinib")
SYSTEMS2 = ("lj38", "lj75", "m80r6", "au38", "tip4p20")
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "RUST_TEST_THREADS",
    "TBB_NUM_THREADS",
    "POLARS_MAX_THREADS",
)
ROW_FIELDS = (
    "panel",
    "system",
    "comparison",
    "arm",
    "variant",
    "seed",
    "slot",
    "core",
    "budget",
    "hit",
    "calls",
    "charged",
    "raw_callbacks",
    "wall_s",
    "inside_s",
    "outside_s",
    "outside_per_call_s",
    "best_energy",
    "best_bits",
    "coordinates_sha256",
    "bit_identity",
    "reference",
    "error",
    "machine_id",
    "session",
    "source_commit",
    "source_tree",
    "artifact_sha256",
    "harness_blobs",
)


def load_provenance(path: Path) -> dict:
    data = json.loads(path.read_text())
    required = ("session", "machine_id", "source_root", "counter", "kit_sha256", "kit_tree", "arms")
    missing = [key for key in required if key not in data]
    if missing:
        raise SystemExit(f"provenance is missing {', '.join(missing)}")
    if len(data["machine_id"]) < 12 or any(ch not in "0123456789abcdef" for ch in data["machine_id"]):
        raise SystemExit("machine_id must be a lowercase hexadecimal digest")
    return data


def one_thread_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_")):
            del env[key]
    for key in THREAD_VARS:
        env[key] = "1"
    env["PYTHONHASHSEED"] = "0"
    if extra:
        env.update(extra)
    return env


def require_core_zero() -> None:
    if hasattr(os, "sched_getaffinity") and os.sched_getaffinity(0) != {0}:
        raise SystemExit(f"runner affinity is {sorted(os.sched_getaffinity(0))}, expected [0]")


def ordered_pair(left: str, right: str, index: int) -> tuple[str, str]:
    """ABBA over each four consecutive seeds."""
    return (right, left) if index % 4 in (1, 2) else (left, right)


def arm_record(provenance: dict, arm: str) -> dict:
    try:
        return provenance["arms"][arm]
    except KeyError as err:
        raise SystemExit(f"provenance has no arm {arm}") from err


def run_checked(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["taskset", "-c", "0", *command],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    log.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark subprocess exited {proc.returncode}; see its scratch log")
    return proc.stdout


def read_budgets(source_root: Path) -> dict[str, int]:
    path = source_root / "bench" / "openconf-h2h" / "budgets.csv"
    with path.open(newline="") as handle:
        return {row["molecule"]: int(row["anneal_budget"]) for row in csv.DictReader(handle)}


def panel1(args: argparse.Namespace, provenance: dict) -> None:
    require_core_zero()
    source_root = Path(provenance["source_root"])
    job_script = source_root / "scripts" / "openconf_h2h" / "h2h_job.py"
    budgets = read_budgets(source_root)
    comparisons = (("ab", "A", "B"), ("aa", "A", "Aprime"))
    for system in SYSTEMS1:
        for comparison, left, right in comparisons:
            for index, seed in enumerate(SEEDS):
                for slot, arm in enumerate(ordered_pair(left, right, index), 1):
                    out = args.work / "panel1" / comparison / system / str(seed) / arm
                    done = out / "job.json"
                    if done.is_file():
                        continue
                    out.mkdir(parents=True, exist_ok=True)
                    record = arm_record(provenance, arm)
                    env = one_thread_env(
                        {
                            "PYTHONPATH": record["python_path"],
                            "LD_PRELOAD": provenance["counter"],
                        }
                    )
                    command = [
                        sys.executable,
                        "-u",
                        str(job_script),
                        "anneal",
                        system,
                        str(seed),
                        "--out",
                        str(out),
                        "--budget",
                        str(budgets[system]),
                    ]
                    run_checked(
                        command,
                        cwd=source_root,
                        env=env,
                        log=args.logs / "panel1" / comparison / system / f"{seed}-{slot}-{arm}.log",
                    )
                    metadata = {
                        "comparison": comparison,
                        "logical_arm": arm,
                        "slot": slot,
                        "seed": seed,
                    }
                    (out / "gate.json").write_text(json.dumps(metadata, sort_keys=True) + "\n")


def import_scorer(source_root: Path):
    scripts = source_root / "scripts" / "openconf_h2h"
    sys.path.insert(0, str(scripts))
    import h2h_common as hc
    from h2h_score import Molecule

    return hc, Molecule


def panel1_jobs(work: Path) -> list[tuple[Path, dict, dict]]:
    jobs = []
    for gate_path in sorted((work / "panel1").glob("*/*/*/*/gate.json")):
        job_dir = gate_path.parent
        gate = json.loads(gate_path.read_text())
        info = json.loads((job_dir / "job.json").read_text())
        jobs.append((job_dir, gate, info))
    expected = len(SYSTEMS1) * len(SEEDS) * 4
    if len(jobs) != expected:
        raise SystemExit(f"panel1 has {len(jobs)} jobs, expected {expected}")
    return jobs


def score_panel1(args: argparse.Namespace, provenance: dict) -> None:
    require_core_zero()
    source_root = Path(provenance["source_root"])
    hc, Molecule = import_scorer(source_root)
    molecules = {name: Molecule(hc.molecules()[name]) for name in SYSTEMS1}
    jobs = panel1_jobs(args.work)
    scored = []
    for job_dir, gate, info in jobs:
        molecule = molecules[info["molecule"]]
        cache = job_dir / "gate-polished.npz"
        if cache.is_file():
            import numpy as np

            data = np.load(cache)
            xs, energies, valid = data["x"], data["e"], data["ok"]
        else:
            import numpy as np

            frames = hc.read_xyz(job_dir / "candidates.xyz")
            polished = [molecule.polish(coords) for _symbols, coords, _comment in frames]
            xs = np.asarray([item[0] for item in polished]).reshape(-1, molecule.mol.GetNumAtoms(), 3)
            energies = np.asarray([item[1] for item in polished])
            valid = np.asarray([item[2] for item in polished], dtype=bool)
            np.savez(cache, x=xs, e=energies, ok=valid)
        polished_all = [
            (coords, float(energy), bool(ok))
            for coords, energy, ok in zip(xs, energies, valid, strict=True)
        ]
        deduped = molecule.dedupe(
            [(coords, energy) for coords, energy, ok in polished_all if ok]
        )
        scored.append((job_dir, gate, info, polished_all, deduped))

    references = {}
    for system in SYSTEMS1:
        pool = [
            item
            for _job_dir, _gate, info, _polished_all, deduped in scored
            if info["molecule"] == system
            for item in deduped
        ]
        if not pool:
            raise SystemExit(f"panel1 {system} has no valid conformer")
        references[system] = min(pool, key=lambda item: item[1])

    budgets = read_budgets(source_root)
    rows = []
    for job_dir, gate, info, polished_all, deduped in scored:
        system = info["molecule"]
        molecule = molecules[system]
        reference = references[system]
        hit = any(molecule.same(reference, item) for item in deduped)
        trace = hc.read_xyz(job_dir / "trace.xyz")
        first_raw = None
        if hit:
            for index, (_symbols, _coords, comment) in enumerate(trace):
                if index >= len(polished_all):
                    break
                coords, energy, ok = polished_all[index]
                if ok and molecule.same(reference, (coords, energy)):
                    first_raw = int(comment.split()[0])
                    break
        budget = budgets[system]
        calls = min(budget, (first_raw + 1) // 2) if first_raw is not None else budget
        charged = min(budget, max(int(info["energy_calls"]), int(info["grad_calls"])))
        raw_callbacks = int(info["energy_calls"]) + int(info["grad_calls"])
        outside = float(info["outside_s"])
        arm = gate["logical_arm"]
        record = arm_record(provenance, arm)
        rows.append(
            {
                "panel": "molecular",
                "system": system,
                "comparison": gate["comparison"],
                "arm": arm,
                "variant": record.get("variant", arm),
                "seed": str(gate["seed"]),
                "slot": str(gate["slot"]),
                "core": "0",
                "budget": str(budget),
                "hit": str(int(hit)),
                "calls": str(calls),
                "charged": str(charged),
                "raw_callbacks": str(raw_callbacks),
                "wall_s": f"{float(info['wall_s']):.9f}",
                "inside_s": f"{float(info['pot_s']):.9f}",
                "outside_s": f"{outside:.9f}",
                "outside_per_call_s": f"{outside / max(charged, 1):.12f}",
                "best_energy": f"{min((item[1] for item in deduped), default=math.inf):.12f}",
                "best_bits": "",
                "coordinates_sha256": "",
                "bit_identity": "",
                "reference": f"{reference[1]:.12f}",
                "error": info.get("error") or "",
                "machine_id": provenance["machine_id"],
                "session": provenance["session"],
                "source_commit": record["commit"],
                "source_tree": record["tree"],
                "artifact_sha256": record["artifact_sha256"],
                "harness_blobs": ";".join(provenance["harness_blobs"]),
            }
        )
    write_csv(args.work / "panel1-rows.csv", ROW_FIELDS, rows)


def parse_identity_output(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    headers = [line for line in lines if line.startswith("system,seed,budget,")]
    data = [line for line in lines if not line.startswith("system,seed,budget,")]
    if len(headers) != 1 or len(data) != 1:
        raise RuntimeError("identity runner did not emit one header and row")
    row = dict(zip(headers[0].split(","), data[0].split(","), strict=True))
    return row


def identity_comparisons(provenance: dict) -> list[tuple[str, str, str]]:
    comparisons = [("ab", "A", "B"), ("aa", "A", "Aprime")]
    placements = sorted(
        (name for name in provenance["arms"] if name.startswith("P")),
        key=lambda name: int(name[1:]),
    )
    if placements != [f"P{index}" for index in range(1, 6)]:
        raise SystemExit("provenance needs placement arms P1 through P5")
    comparisons.extend((f"placement{index}", "A", name) for index, name in enumerate(placements, 1))
    return comparisons


def panel2(args: argparse.Namespace, provenance: dict) -> None:
    require_core_zero()
    for system in SYSTEMS2:
        for comparison, left, right in identity_comparisons(provenance):
            for index, seed in enumerate(SEEDS):
                pair = []
                for slot, arm in enumerate(ordered_pair(left, right, index), 1):
                    run_dir = args.work / "panel2" / comparison / system / str(seed) / arm
                    result_path = run_dir / "result.json"
                    bits_path = run_dir / "coordinates.bin"
                    if result_path.is_file() and bits_path.is_file():
                        result = json.loads(result_path.read_text())
                    else:
                        run_dir.mkdir(parents=True, exist_ok=True)
                        record = arm_record(provenance, arm)
                        stdout = run_checked(
                            [record["identity_binary"], system, str(seed), str(bits_path)],
                            cwd=Path(provenance["source_root"]),
                            env=one_thread_env(),
                            log=args.logs
                            / "panel2"
                            / comparison
                            / system
                            / f"{seed}-{slot}-{arm}.log",
                        )
                        result = parse_identity_output(stdout)
                        result.update(
                            {
                                "comparison": comparison,
                                "logical_arm": arm,
                                "slot": slot,
                            }
                        )
                        result_path.write_text(json.dumps(result, sort_keys=True) + "\n")
                    pair.append((arm, run_dir, result))
                left_run = next(item for item in pair if item[0] == left)
                right_run = next(item for item in pair if item[0] == right)
                same = all(
                    (
                        left_run[2]["best_bits"] == right_run[2]["best_bits"],
                        left_run[2]["calls"] == right_run[2]["calls"],
                        left_run[2]["charged"] == right_run[2]["charged"],
                        (left_run[1] / "coordinates.bin").read_bytes()
                        == (right_run[1] / "coordinates.bin").read_bytes(),
                    )
                )
                for _arm, run_dir, result in pair:
                    result["bit_identity"] = int(same)
                    (run_dir / "result.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    score_panel2(args.work, provenance)


def score_panel2(work: Path, provenance: dict) -> None:
    rows = []
    for result_path in sorted((work / "panel2").glob("*/*/*/*/result.json")):
        result = json.loads(result_path.read_text())
        arm = result["logical_arm"]
        record = arm_record(provenance, arm)
        bits = (result_path.parent / "coordinates.bin").read_bytes()
        charged = int(result["charged"])
        outside = float(result["outside_s"])
        rows.append(
            {
                "panel": "cluster",
                "system": result["system"],
                "comparison": result["comparison"],
                "arm": arm,
                "variant": record.get("variant", arm),
                "seed": result["seed"],
                "slot": str(result["slot"]),
                "core": "0",
                "budget": result["budget"],
                "hit": result["hit"],
                "calls": result["calls"],
                "charged": result["charged"],
                "raw_callbacks": result["charged"],
                "wall_s": result["wall_s"],
                "inside_s": result["inside_s"],
                "outside_s": result["outside_s"],
                "outside_per_call_s": f"{outside / max(charged, 1):.12f}",
                "best_energy": result["best"],
                "best_bits": result["best_bits"],
                "coordinates_sha256": hashlib.sha256(bits).hexdigest(),
                "bit_identity": str(result["bit_identity"]),
                "reference": result["reference"],
                "error": "",
                "machine_id": provenance["machine_id"],
                "session": provenance["session"],
                "source_commit": record["commit"],
                "source_tree": record["tree"],
                "artifact_sha256": record["artifact_sha256"],
                "harness_blobs": ";".join(provenance["harness_blobs"]),
            }
        )
    expected = len(SYSTEMS2) * len(SEEDS) * len(identity_comparisons(provenance)) * 2
    if len(rows) != expected:
        raise SystemExit(f"panel2 has {len(rows)} rows, expected {expected}")
    write_csv(work / "panel2-rows.csv", ROW_FIELDS, rows)


def type7(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("type 7 needs a sample")
    if len(ordered) == 1:
        return ordered[0]
    position = 1.0 + (len(ordered) - 1) * quantile
    lower = int(position)
    fraction = position - lower
    if fraction == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - fraction) + ordered[lower] * fraction


def median_iqr(rows: list[dict[str, str]], field: str) -> tuple[float, float]:
    values = [float(row[field]) for row in rows]
    return type7(values, 0.5), type7(values, 0.75) - type7(values, 0.25)


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    from scipy.stats import fisher_exact

    return float(fisher_exact([[hits_a, n_a - hits_a], [hits_b, n_b - hits_b]])[1])


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    from scipy.stats import mannwhitneyu

    if len(set(left + right)) == 1:
        return len(left) * len(right) / 2.0, 1.0
    result = mannwhitneyu(
        left,
        right,
        alternative="two-sided",
        method="asymptotic",
        use_continuity=True,
    )
    return float(result.statistic), float(result.pvalue)


def ratio_band(groups: list[tuple[list[dict[str, str]], list[dict[str, str]]]]) -> float:
    band = 1.0
    for left, right in groups:
        left_median = type7([float(row["wall_s"]) for row in left], 0.5)
        right_median = type7([float(row["wall_s"]) for row in right], 0.5)
        if left_median > 0.0 and right_median > 0.0:
            band = max(band, left_median / right_median, right_median / left_median)
    return band


def aggregate_outside(rows: list[dict[str, str]]) -> float:
    charged = sum(int(row["charged"]) for row in rows)
    return sum(float(row["outside_s"]) for row in rows) / max(charged, 1)


def comparison_groups(
    rows: list[dict[str, str]], panel: str, system: str, comparison: str, left: str, right: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    selected = [
        row
        for row in rows
        if row["panel"] == panel and row["system"] == system and row["comparison"] == comparison
    ]
    a = sorted((row for row in selected if row["arm"] == left), key=lambda row: int(row["seed"]))
    b = sorted((row for row in selected if row["arm"] == right), key=lambda row: int(row["seed"]))
    if [row["seed"] for row in a] != [row["seed"] for row in b] or len(a) != len(SEEDS):
        raise SystemExit(f"{panel} {system} {comparison} is not a complete paired cell")
    return a, b


def external_crest(source_root: Path) -> dict[str, dict[str, str]]:
    path = source_root / "bench" / "openconf-h2h" / "results.csv"
    with path.open(newline="") as handle:
        return {row["molecule"]: row for row in csv.DictReader(handle) if row["arm"] == "crest"}


def summarize_cell(
    system: str,
    a: list[dict[str, str]],
    b: list[dict[str, str]],
    band: float,
    alpha: float,
) -> dict[str, str]:
    a_hits = sum(int(row["hit"]) for row in a)
    b_hits = sum(int(row["hit"]) for row in b)
    fisher_p = fisher(a_hits, len(a), b_hits, len(b))
    a_calls, a_calls_iqr = median_iqr(a, "calls")
    b_calls, b_calls_iqr = median_iqr(b, "calls")
    _u_calls, p_calls = mann_whitney(
        [float(row["calls"]) for row in a],
        [float(row["calls"]) for row in b],
    )
    a_wall, _a_wall_iqr = median_iqr(a, "wall_s")
    b_wall, _b_wall_iqr = median_iqr(b, "wall_s")
    _u_wall, p_wall = mann_whitney(
        [float(row["wall_s"]) for row in a],
        [float(row["wall_s"]) for row in b],
    )
    wall_ratio = b_wall / a_wall if a_wall > 0.0 else math.inf
    reasons = []
    if b_hits < a_hits and fisher_p < alpha:
        reasons.append("hits")
    if b_calls > a_calls and p_calls < alpha:
        reasons.append("calls")
    if b_wall > a_wall and p_wall < alpha and wall_ratio > band:
        reasons.append("wall")
    return {
        "system": system,
        "n": str(len(a)),
        "a_hits": f"{a_hits}/{len(a)}",
        "b_hits": f"{b_hits}/{len(b)}",
        "fisher_p": f"{fisher_p:.9g}",
        "a_calls_median": f"{a_calls:.3f}",
        "a_calls_iqr": f"{a_calls_iqr:.3f}",
        "b_calls_median": f"{b_calls:.3f}",
        "b_calls_iqr": f"{b_calls_iqr:.3f}",
        "mannwhitney_p_calls": f"{p_calls:.9g}",
        "a_wall_median_s": f"{a_wall:.9f}",
        "b_wall_median_s": f"{b_wall:.9f}",
        "mannwhitney_p_wall": f"{p_wall:.9g}",
        "a_outside_per_call_s": f"{aggregate_outside(a):.12f}",
        "b_outside_per_call_s": f"{aggregate_outside(b):.12f}",
        "wall_ratio": f"{wall_ratio:.6f}",
        "wall_band_upper": f"{band:.6f}",
        "verdict": "FAIL" if reasons else "PASS",
        "reason": " ".join(reasons),
    }


def schedule_counts(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    for row in rows:
        key = (row["panel"], row["system"], row["comparison"], row["slot"], row["arm"])
        counts[key] += 1
    out = []
    for key, count in sorted(counts.items()):
        panel, system, comparison, slot, arm = key
        out.append(
            {
                "panel": panel,
                "system": system,
                "comparison": comparison,
                "slot": slot,
                "arm": arm,
                "count": str(count),
                "expected": "6",
                "verified": str(int(count == 6)),
            }
        )
    if not out or any(row["verified"] != "1" for row in out):
        raise SystemExit("schedule is not balanced by (slot, arm)")
    return out


def sessions_rows(provenance: dict, schedule: list[dict[str, str]]) -> list[dict[str, str]]:
    balanced = int(all(row["verified"] == "1" for row in schedule))
    rows = []
    for arm, record in sorted(provenance["arms"].items()):
        rows.append(
            {
                "session": provenance["session"],
                "machine_id": provenance["machine_id"],
                "arm": arm,
                "variant": record.get("variant", arm),
                "source_commit": record["commit"],
                "source_tree": record["tree"],
                "artifact_sha256": record["artifact_sha256"],
                "core": "0",
                "threads": "1",
                "seeds": " ".join(map(str, SEEDS)),
                "n": str(len(SEEDS)),
                "abba_balanced": str(balanced),
                "warmup": "none",
                "paper_budgets": "1",
                "kit_sha256": provenance["kit_sha256"],
                "kit_tree": provenance["kit_tree"],
                "harness_blobs": ";".join(provenance["harness_blobs"]),
            }
        )
    return rows


def report(args: argparse.Namespace, provenance: dict) -> None:
    require_core_zero()
    rows = read_csv(args.work / "panel1-rows.csv") + read_csv(args.work / "panel2-rows.csv")
    rows.sort(
        key=lambda row: (
            row["panel"],
            row["system"],
            row["comparison"],
            int(row["seed"]),
            int(row["slot"]),
        )
    )
    if {row["session"] for row in rows} != {provenance["session"]}:
        raise SystemExit("rows do not belong to one session")
    schedule = schedule_counts(rows)
    alpha = 0.05 / (len(SYSTEMS1) + len(SYSTEMS2))
    crest = external_crest(Path(provenance["source_root"]))
    panel1_summary = []
    for system in SYSTEMS1:
        a, b = comparison_groups(rows, "molecular", system, "ab", "A", "B")
        aa, aprime = comparison_groups(rows, "molecular", system, "aa", "A", "Aprime")
        line = summarize_cell(system, a, b, ratio_band([(aa, aprime)]), alpha)
        external = crest.get(system)
        line.update(
            {
                "crest_hits": external["hits_0p1"] if external else "",
                "crest_calls_median": external["grad_calls_median"] if external else "",
                "crest_outside_per_call_s": external["outside_per_grad_s"] if external else "",
            }
        )
        panel1_summary.append(line)

    panel2_summary = []
    for system in SYSTEMS2:
        a, b = comparison_groups(rows, "cluster", system, "ab", "A", "B")
        band_groups = [comparison_groups(rows, "cluster", system, "aa", "A", "Aprime")]
        band_groups.extend(
            comparison_groups(rows, "cluster", system, f"placement{index}", "A", f"P{index}")
            for index in range(1, 6)
        )
        line = summarize_cell(system, a, b, ratio_band(band_groups), alpha)
        identity = sum(row["bit_identity"] == "1" for row in a)
        placement_rows = [row for pair in band_groups for group in pair for row in group]
        placement_identity = sum(row["bit_identity"] == "1" for row in placement_rows)
        line.update(
            {
                "bit_identity": f"{identity}/{len(a)}",
                "placement_bit_identity": f"{placement_identity}/{len(placement_rows)}",
            }
        )
        if identity != len(a) or placement_identity != len(placement_rows):
            line["verdict"] = "FAIL"
            line["reason"] = (line["reason"] + " identity").strip()
        panel2_summary.append(line)

    overall = "PASS"
    if any(row["verdict"] != "PASS" for row in panel1_summary + panel2_summary):
        overall = "FAIL"
    args.records.mkdir(parents=True, exist_ok=True)
    write_csv(args.records / "rows.csv", ROW_FIELDS, rows)
    write_csv(
        args.records / "schedule-counts.csv",
        ("panel", "system", "comparison", "slot", "arm", "count", "expected", "verified"),
        schedule,
    )
    write_csv(
        args.records / "sessions.csv",
        (
            "session",
            "machine_id",
            "arm",
            "variant",
            "source_commit",
            "source_tree",
            "artifact_sha256",
            "core",
            "threads",
            "seeds",
            "n",
            "abba_balanced",
            "warmup",
            "paper_budgets",
            "kit_sha256",
            "kit_tree",
            "harness_blobs",
        ),
        sessions_rows(provenance, schedule),
    )
    panel1_fields = tuple(panel1_summary[0])
    panel2_fields = tuple(panel2_summary[0])
    write_csv(args.records / "panel1-summary.csv", panel1_fields, panel1_summary)
    write_csv(args.records / "panel2-summary.csv", panel2_fields, panel2_summary)
    (args.records / "README.md").write_text(
        render_report(panel1_summary, panel2_summary, overall, alpha, provenance)
    )


def show(value: str) -> str:
    return value if value else "-"


def render_report(
    panel1: list[dict[str, str]],
    panel2: list[dict[str, str]],
    overall: str,
    alpha: float,
    provenance: dict,
) -> str:
    lines = [
        "Copyright (c) anneal developers",
        "",
        "# conf-fix15 same-session A/B gate",
        "",
        f"Overall verdict: **{overall}**.",
        "",
        "A and B are release, locked builds from the recorded commits and trees.",
        "A-prime is a second A build. Five byte-identical A artifact placements",
        "measure the cluster wall-time placement band. All runs are serial on core",
        "zero with numerical runtime thread counts set to one, seeds 2000--2011,",
        "no warm-up, and repository paper budgets. Pair order is balanced ABBA;",
        "`schedule-counts.csv` verifies every `(slot, arm)` count.",
        "",
        f"The family-wise threshold is 0.05/13 = {alpha:.9g}. A cell fails when B",
        "has significantly fewer hits, significantly larger calls, or significantly",
        "larger wall time beyond its measured A/A or placement band. A cluster cell",
        "also fails on any unexplained bit-identity break in best energy, calls, or",
        "final-coordinate bits.",
        "",
        "The external CREST columns reproduce the available repository harness rows.",
        "Those rows use that campaign's six-seed 0.1 kcal/mol energy-hit summary;",
        "a dash means the repository has no completed CREST row for that molecule.",
        "",
        "## Molecular conformer panel",
        "",
        "| System | A hits | B hits | CREST hits | A calls (IQR) | B calls (IQR) | CREST calls | A outside/call (s) | B outside/call (s) | CREST outside/call (s) | Fisher p | MW p calls | MW p wall | Wall band | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in panel1:
        lines.append(
            "| {system} | {a_hits} | {b_hits} | {crest} | {ac} ({ai}) | {bc} ({bi}) | "
            "{cc} | {ao} | {bo} | {co} | {fp} | {pc} | {pw} | {band} | {verdict} |".format(
                system=row["system"],
                a_hits=row["a_hits"],
                b_hits=row["b_hits"],
                crest=show(row["crest_hits"]),
                ac=row["a_calls_median"],
                ai=row["a_calls_iqr"],
                bc=row["b_calls_median"],
                bi=row["b_calls_iqr"],
                cc=show(row["crest_calls_median"]),
                ao=row["a_outside_per_call_s"],
                bo=row["b_outside_per_call_s"],
                co=show(row["crest_outside_per_call_s"]),
                fp=row["fisher_p"],
                pc=row["mannwhitney_p_calls"],
                pw=row["mannwhitney_p_wall"],
                band=row["wall_band_upper"],
                verdict=row["verdict"],
            )
        )
    lines.extend(
        [
            "",
            "## Cluster identity panel",
            "",
            "| System | Identity A/B | A hits | B hits | A calls (IQR) | B calls (IQR) | A outside/call (s) | B outside/call (s) | Fisher p | MW p calls | MW p wall | Placement band | Verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in panel2:
        lines.append(
            "| {system} | {identity} | {a_hits} | {b_hits} | {ac} ({ai}) | {bc} ({bi}) | "
            "{ao} | {bo} | {fp} | {pc} | {pw} | {band} | {verdict} |".format(
                system=row["system"],
                identity=row["bit_identity"],
                a_hits=row["a_hits"],
                b_hits=row["b_hits"],
                ac=row["a_calls_median"],
                ai=row["a_calls_iqr"],
                bc=row["b_calls_median"],
                bi=row["b_calls_iqr"],
                ao=row["a_outside_per_call_s"],
                bo=row["b_outside_per_call_s"],
                fp=row["fisher_p"],
                pc=row["mannwhitney_p_calls"],
                pw=row["mannwhitney_p_wall"],
                band=row["wall_band_upper"],
                verdict=row["verdict"],
            )
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Input kit SHA-256: `{provenance['kit_sha256']}`",
            f"- B source tree: `{provenance['kit_tree']}`",
            f"- Session: `{provenance['session']}`",
            f"- Hashed machine id: `{provenance['machine_id']}`",
            "- Per-run source commits, source trees, artifact hashes, and harness",
            "  blob ids are in `rows.csv`; build/session records are in `sessions.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    order = [ordered_pair("A", "B", index) for index in range(12)]
    counts = defaultdict(int)
    for pair in order:
        for slot, arm in enumerate(pair, 1):
            counts[(slot, arm)] += 1
    if set(counts.values()) != {6}:
        raise SystemExit(f"ABBA counts {dict(counts)}")
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if type7(sample, 0.25) != 2.25 or type7(sample, 0.5) != 3.5:
        raise SystemExit("type 7")
    print("ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    for name in ("panel1", "score-panel1", "panel2"):
        command = sub.add_parser(name)
        command.add_argument("--provenance", type=Path, required=True)
        command.add_argument("--work", type=Path, required=True)
        command.add_argument("--logs", type=Path)
    command = sub.add_parser("report")
    command.add_argument("--provenance", type=Path, required=True)
    command.add_argument("--work", type=Path, required=True)
    command.add_argument("--records", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return 0
    provenance = load_provenance(args.provenance)
    if args.command in ("panel1", "panel2") and args.logs is None:
        parser.error(f"{args.command} needs --logs")
    commands = {
        "panel1": panel1,
        "score-panel1": score_panel1,
        "panel2": panel2,
        "report": report,
    }
    commands[args.command](args, provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
