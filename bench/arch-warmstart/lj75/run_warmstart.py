#!/usr/bin/env python3
"""Build LJ75 MinimaCorpus (seeds 1000-1011) and run warm vs cold (2000-2011)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "paper_0120"))
import run_panel  # noqa: E402

BUDGET = run_panel.PAPER_BUDGETS["lj75"]
CORPUS_SEEDS = list(range(1000, 1012))
EVAL_SEEDS = list(range(2000, 2012))

ROW_FIELDS = [
    "system",
    "arm",
    "seed",
    "workers",
    "total_oracle_calls",
    "call_budget",
    "hit",
    "calls_to_hit",
    "wall_to_hit_s",
    "best_energy",
    "wall_s",
    "cpu_s",
    "peak_rss_anon_kib",
    "replica_charged",
    "hops",
    "basins",
    "screened_out",
    "returned",
    "history_obs",
    "history_new",
    "ledger_sum",
    "evals_observed",
    "ckpt_calls_to_hit",
    "warm_seed_energy",
    "corpus_recorded",
    "slot",
    "revision",
    "nproc",
    "cpu_model",
    "pinned_cores",
    "oversubscribed",
]


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def target_binary() -> Path:
    meta = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            cwd=REPO,
            text=True,
        )
    )
    return Path(meta["target_directory"]) / "release" / "examples" / "arch_lj75_bench"


def load_done(path: Path) -> set[tuple[str, int]]:
    if not path.is_file():
        return set()
    with path.open(newline="") as handle:
        return {(r["arm"], int(r["seed"])) for r in csv.DictReader(handle)}


def append_row(path: Path, row: dict[str, object]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def corpus_stats(corpus_dir: Path) -> tuple[int, dict[str, str]]:
    total = 0
    hashes: dict[str, str] = {}
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(corpus_dir))
            data = p.read_bytes()
            total += len(data)
            hashes[rel] = hashlib.sha256(data).hexdigest()
    return total, hashes


def run_cmd(
    binary: Path,
    args: list[str],
    arm: str,
    seed: int,
    slot: int,
    revision: str,
    nproc: int,
    model: str,
    rows_path: Path,
) -> None:
    cmd = ["taskset", "-c", "0", str(binary), *args]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{arm} seed={seed} failed ({proc.returncode}): {proc.stderr}")
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            payload = json.loads(line)
    if payload is None:
        raise RuntimeError(f"missing JSON output for {arm} seed={seed}")
    if float(payload["wall_s"]) < 0.050:
        raise RuntimeError(
            f"sample wall_s={payload['wall_s']} < 0.050 s for {arm} seed={seed}"
        )
    if int(payload["ledger_sum"]) != int(payload["total_oracle_calls"]):
        raise RuntimeError(f"ledger mismatch for {arm} seed={seed}: {payload}")
    row = {
        "system": payload["system"],
        "arm": arm,
        "seed": seed,
        "workers": 1,
        "total_oracle_calls": payload["total_oracle_calls"],
        "call_budget": payload["call_budget"],
        "hit": payload["hit"],
        "calls_to_hit": payload["calls_to_hit"],
        "wall_to_hit_s": (
            "" if payload["wall_to_hit_s"] is None else f"{payload['wall_to_hit_s']:.6f}"
        ),
        "best_energy": f"{payload['best_energy']:.10f}",
        "wall_s": f"{payload['wall_s']:.6f}",
        "cpu_s": f"{payload['cpu_s']:.2f}",
        "peak_rss_anon_kib": payload["peak_rss_anon_kib"],
        "replica_charged": payload["replica_charged"],
        "hops": payload["hops"],
        "basins": payload["basins"],
        "screened_out": payload["screened_out"],
        "returned": payload["returned"],
        "history_obs": payload["history_obs"],
        "history_new": payload["history_new"],
        "ledger_sum": payload["ledger_sum"],
        "evals_observed": payload["evals_observed"],
        "ckpt_calls_to_hit": (
            "" if payload["ckpt_calls_to_hit"] is None else payload["ckpt_calls_to_hit"]
        ),
        "warm_seed_energy": (
            ""
            if payload["warm_seed_energy"] is None
            else f"{payload['warm_seed_energy']:.10f}"
        ),
        "corpus_recorded": (
            "" if payload["corpus_recorded"] is None else payload["corpus_recorded"]
        ),
        "slot": slot,
        "revision": revision,
        "nproc": nproc,
        "cpu_model": model,
        "pinned_cores": "0",
        "oversubscribed": "false",
    }
    append_row(rows_path, row)
    print(
        f"{arm} seed={seed} slot={slot} hit={row['hit']} calls={row['calls_to_hit']} "
        f"wall={row['wall_s']}s rss={row['peak_rss_anon_kib']}KiB best={row['best_energy']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=REPO.parent / "tmp" / "lj75-minima-corpus",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path(__file__).resolve().parent / "rows.csv",
    )
    parser.add_argument(
        "--corpus-meta",
        type=Path,
        default=Path(__file__).resolve().parent / "corpus_meta.json",
    )
    args = parser.parse_args()

    binary = target_binary()
    if not binary.is_file():
        raise SystemExit(f"missing binary: {binary}")
    revision = git_head()
    nproc = os.cpu_count() or 1
    model = cpu_model()
    done = load_done(args.rows)

    for idx, seed in enumerate(CORPUS_SEEDS, 1):
        if ("corpus", seed) in done:
            continue
        run_cmd(
            binary,
            ["build-corpus", str(args.corpus_dir), str(seed), str(BUDGET)],
            "corpus",
            seed,
            idx,
            revision,
            nproc,
            model,
            args.rows,
        )
        done.add(("corpus", seed))

    size_bytes, hashes = corpus_stats(args.corpus_dir)
    args.corpus_meta.write_text(
        json.dumps(
            {
                "size_bytes": size_bytes,
                "files": hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"corpus size_bytes={size_bytes} files={list(hashes)}", flush=True)

    for idx, seed in enumerate(EVAL_SEEDS):
        order = ("cold", "warm") if (idx % 4 in (0, 3)) else ("warm", "cold")
        for slot, arm in enumerate(order, 1):
            if (arm, seed) in done:
                continue
            cmd_args = (
                ["single", "cold", str(seed), str(BUDGET)]
                if arm == "cold"
                else ["single", "warm", str(seed), str(BUDGET), str(args.corpus_dir)]
            )
            run_cmd(
                binary,
                cmd_args,
                arm,
                seed,
                slot,
                revision,
                nproc,
                model,
                args.rows,
            )
            done.add((arm, seed))


if __name__ == "__main__":
    main()
