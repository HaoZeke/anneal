#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial ABBA panel runner for anneal #179 r8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

EXPECTED_A_HEAD = "cb0cfbfa0cfae21b62cc32037defc5327dc52a51"
EXPECTED_B_TREE = "dd97e61fbe57f633fa131794918d647a3a20b6c3"

CELLS = (
    "lj38",
    "blj130n38",
    "sc108-80",
    "lj75",
    "lj98",
    "lj102",
    "lj104",
    "m80r14",
)
SEEDS = tuple(range(2000, 2012))

ROW_FIELDS = [
    "cell",
    "seed",
    "arm",
    "order_pos",
    "arm_order",
    "budget",
    "reference",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "pot_s",
    "outside_s",
    "outside_us",
    "rss_anon_kb",
    "symmetrised",
    "sym_gain",
    "bin_sha256",
    "code_revision",
    "tree_sha",
    "utc",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            h.update(chunk)
    return h.hexdigest()


def git_rev(tree: Path, expr: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", expr],
        text=True,
    ).strip()


def discover_cargo() -> str:
    for candidate in (
        os.environ.get("CARGO", ""),
        "/usr/local/cargo/bin/cargo",
        shutil.which("cargo") or "",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("cargo not found")


def single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("H2H_"):
            del env[key]
    for key in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "MKL_DOMAIN_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "GOTO_NUM_THREADS",
    ):
        env[key] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    return env


def build_driver(
    driver_dir: Path,
    wt_tree: Path,
    build_dir: Path,
    *,
    placement_pad: bool = False,
) -> dict[str, object]:
    wt_tree = wt_tree.resolve()
    build_dir = build_dir.resolve()
    src_dir = build_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    template = (driver_dir / "Cargo.toml.in").read_text()
    (build_dir / "Cargo.toml").write_text(
        template.replace("@ANNEAL_PATH@", str(wt_tree))
    )
    shutil.copy2(driver_dir / "src" / "main.rs", src_dir / "main.rs")

    target_dir = build_dir / "target"
    tmp_dir = build_dir / "tmp"
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cargo = discover_cargo()
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target_dir)
    env["TMPDIR"] = str(tmp_dir)
    env["CARGO_INCREMENTAL"] = "0"
    env["RUSTFLAGS"] = (
        f"--remap-path-prefix={wt_tree}=/src "
        f"--remap-path-prefix={build_dir}=/drv "
        f"--remap-path-prefix={Path.home()}=/user"
    )
    cmd = [cargo, "+1.88.0", "build", "--release"]
    if placement_pad:
        cmd.extend(["--features", "placement-pad"])
    proc = subprocess.run(
        cmd,
        cwd=build_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"build failed in {build_dir}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    bin_path = target_dir / "release" / "r8-gate-driver"
    return {
        "bin": bin_path,
        "bin_sha256": sha256_file(bin_path),
        "bin_bytes": bin_path.stat().st_size,
        "head": git_rev(wt_tree, "HEAD"),
        "tree": git_rev(wt_tree, "HEAD^{tree}"),
    }

def arm_schedule(arms: tuple[str, ...], offset: int) -> tuple[str, ...]:
    shift = (offset // 2) % len(arms)
    rotated = arms[shift:] + arms[:shift]
    if offset % 2 == 1:
        rotated = rotated[::-1]
    return rotated


def parse_gate_row(stdout: str) -> dict[str, str]:
    for line in stdout.splitlines():
        if line.startswith("gate_row "):
            parsed: dict[str, str] = {}
            for tok in line.split()[1:]:
                k, v = tok.split("=", 1)
                parsed[k] = v
            return parsed
    raise RuntimeError(f"missing gate_row in stdout:\n{stdout[-1500:]}")


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def append_journal(path: Path, entry: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="\n") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def probe_budget(bin_path: Path, cell: str, env: dict[str, str]) -> tuple[int, float]:
    out = subprocess.check_output(
        [str(bin_path), "budget", cell],
        env=env,
        text=True,
    ).strip()
    parts = out.split()
    return int(parts[0]), float(parts[1])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wt-a", type=Path, default=Path("/tmp/r8-wt-a"))
    parser.add_argument("--wt-b", type=Path, default=Path("/tmp/r8-wt-b"))
    parser.add_argument("--work", type=Path, default=Path("/tmp/r8-work"))
    parser.add_argument("--logs", type=Path, default=Path("/tmp/r8-logs"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cells", default=",".join(CELLS))
    parser.add_argument("--no-ap", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args(argv)

    a_head = git_rev(args.wt_a, "HEAD")
    if a_head != EXPECTED_A_HEAD:
        raise SystemExit(f"Arm A HEAD {a_head} != {EXPECTED_A_HEAD}")
    b_tree = git_rev(args.wt_b, "HEAD^{tree}")
    if b_tree != EXPECTED_B_TREE:
        raise SystemExit(f"Arm B tree {b_tree} != {EXPECTED_B_TREE}")

    driver_dir = Path(__file__).resolve().parent / "driver"
    build_a = build_driver(driver_dir, args.wt_a, args.work / "build-a", placement_pad=False)
    build_b = build_driver(driver_dir, args.wt_b, args.work / "build-b", placement_pad=False)
    arms_meta: dict[str, dict[str, object]] = {
        "A": build_a,
        "AA": build_a,
        "B": build_b,
    }
    arm_names: tuple[str, ...] = ("A", "B", "AA")
    if not args.no_ap:
        build_ap = build_driver(driver_dir, args.wt_a, args.work / "build-ap", placement_pad=True)
        arms_meta["AP"] = build_ap
        arm_names = ("A", "B", "AA", "AP")

    env = single_thread_env()
    args.logs.mkdir(parents=True, exist_ok=True)
    probe_lines: list[str] = []
    probe_targets = [("A", build_a), ("B", build_b)]
    if "AP" in arms_meta:
        probe_targets.append(("AP", arms_meta["AP"]))
    for label, meta in probe_targets:
        probe_out = subprocess.check_output(
            ["taskset", "-c", "0", str(meta["bin"]), "probe"],
            env=env,
            text=True,
        ).strip()
        probe_lines.append(
            f"arm={label} head={meta['head']} tree={meta['tree']} "
            f"bin_bytes={meta['bin_bytes']} bin_sha256={meta['bin_sha256']} {probe_out}"
        )
    (args.logs / "probe.log").write_text("\n".join(probe_lines) + "\n")
    for line in probe_lines:
        print(line, flush=True)

    if args.build_only:
        return 0

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    budgets: dict[str, tuple[int, float]] = {}
    for cell in cells:
        ba, ra = probe_budget(Path(build_a["bin"]), cell, env)
        bb, rb = probe_budget(Path(build_b["bin"]), cell, env)
        if ba != bb or abs(ra - rb) > 1e-12:
            raise SystemExit(f"{cell}: A/B budget or ref mismatch ({ba},{ra}) vs ({bb},{rb})")
        budgets[cell] = (ba, ra)

    rows_path = args.out_dir / "rows.csv"
    journal_path = args.out_dir / "journal.jsonl"
    existing = load_rows(rows_path)
    done = {(r["cell"], int(r["seed"]), r["arm"]) for r in existing}

    written = 0
    for cell in cells:
        budget, reference = budgets[cell]
        for offset, seed in enumerate(SEEDS):
            order = arm_schedule(arm_names, offset)
            order_str = "-".join(order)
            for pos, arm_name in enumerate(order):
                if (cell, seed, arm_name) in done:
                    continue
                meta = arms_meta[arm_name]
                bin_path = Path(meta["bin"])
                cmd = ["taskset", "-c", "0", str(bin_path), "run", cell, str(seed)]
                proc = subprocess.run(
                    cmd,
                    cwd=args.work,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                log_path = args.logs / cell / f"{seed}-{arm_name}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(
                    f"arm={arm_name} cell={cell} seed={seed} exit={proc.returncode}\n"
                    + proc.stdout
                    + proc.stderr
                )
                if proc.returncode != 0:
                    raise SystemExit(f"{cell} seed={seed} arm={arm_name} failed exit={proc.returncode}")
                parsed = parse_gate_row(proc.stdout)
                if parsed["cell"] != cell or int(parsed["seed"]) != seed or int(parsed["budget"]) != budget:
                    raise SystemExit(f"mismatched row output: {parsed}")
                charged = int(parsed["charged"])
                outside_s = float(parsed["outside_s"])
                outside_us = outside_s / max(charged, 1) * 1.0e6
                utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                row = {
                    "cell": cell,
                    "seed": str(seed),
                    "arm": arm_name,
                    "order_pos": str(pos),
                    "arm_order": order_str,
                    "budget": str(budget),
                    "reference": f"{reference:.10f}",
                    "hit": parsed["hit"],
                    "evals_to_hit": parsed["evals_to_hit"],
                    "charged": str(charged),
                    "best": parsed["best"],
                    "wall_s": parsed["wall_s"],
                    "pot_s": parsed["pot_s"],
                    "outside_s": parsed["outside_s"],
                    "outside_us": f"{outside_us:.6f}",
                    "rss_anon_kb": parsed["rss_anon_kb"],
                    "symmetrised": parsed["symmetrised"],
                    "sym_gain": parsed["sym_gain"],
                    "bin_sha256": str(meta["bin_sha256"]),
                    "code_revision": str(meta["head"]),
                    "tree_sha": str(meta["tree"]),
                    "utc": utc,
                }
                append_row(rows_path, row)
                append_journal(journal_path, {"utc": utc, "cell": cell, "seed": seed, "arm": arm_name, "order_pos": pos})
                done.add((cell, seed, arm_name))
                written += 1
                print(
                    f"[{written}] {cell} seed={seed} arm={arm_name} hit={row['hit']} "
                    f"evals={row['evals_to_hit']} charged={row['charged']} "
                    f"out_us={row['outside_us']} rss_kb={row['rss_anon_kb']}",
                    flush=True,
                )

    print(f"completed new_rows={written}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
