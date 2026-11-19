#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Short recommended trajectory prefix for git bisect.

Exit 0 when the quench-energy prefix matches the saved arm A trace.
Exit 1 when it differs. Exit 125 when the commit cannot be scored.

The example is patched only for this run, then restored. The patch turns
on the existing energy trace and prints it. Scored gate binaries are not
patched.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

TRACE_ON = "    cfg.energy_trace = true;\n"
PRINT_TRACE = """    if let Some(trace) = outcome.energy_trace.as_ref() {
        for (step, energy) in trace.iter().enumerate() {
            println!("trace step={step} energy={energy:.16e}");
        }
    }
"""


def patch_example(text: str) -> str:
    cfg = "    let cfg = match preset {\n"
    after = "    };\n    let potential = PairPotential::morse(n, rho);\n"
    finish = '    .expect("finite start");\n    let wall = started.elapsed().as_secs_f64();\n'
    for anchor in (cfg, after, finish):
        if text.count(anchor) != 1:
            raise SystemExit(125)
    text = text.replace(cfg, "    let mut cfg = match preset {\n", 1)
    text = text.replace(after, "    };\n" + TRACE_ON + "    let potential = PairPotential::morse(n, rho);\n", 1)
    text = text.replace(finish, '    .expect("finite start");\n' + PRINT_TRACE + "    let wall = started.elapsed().as_secs_f64();\n", 1)
    return text


def restore(worktree: Path) -> None:
    subprocess.run(
        ["git", "-C", str(worktree), "checkout", "--", "examples/ase_nonlj.rs"],
        check=False,
        capture_output=True,
        text=True,
    )


def parse_trace(stdout: str) -> list[str]:
    return [line.split(" energy=", 1)[1] for line in stdout.splitlines() if line.startswith("trace step=")]


def first_difference(expected: list[str], found: list[str]) -> int | None:
    limit = min(len(expected), len(found))
    for index in range(limit):
        if expected[index] != found[index]:
            return index
    if len(expected) != len(found):
        return limit
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--system", default="m38r14")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=400_000)
    parser.add_argument("--cap", default="11.142690")
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args(argv)
    example = args.worktree / "examples" / "ase_nonlj.rs"
    original = example.read_text()
    try:
        example.write_text(patch_example(original))
        script = Path(__file__).with_name("build_arm.sh")
        build = subprocess.run(["bash", str(script), str(args.worktree), str(args.target)], text=True)
        if build.returncode != 0:
            print("build failed", file=sys.stderr)
            return 125
        binary = args.binary or (args.target / "release" / "examples" / "ase_nonlj")
        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "OMP_THREAD_LIMIT": "1",
                "OMP_DYNAMIC": "FALSE",
                "OMP_PROC_BIND": "TRUE",
                "OMP_PLACES": "cores",
                "MKL_NUM_THREADS": "1",
                "MKL_DOMAIN_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "BLIS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMBA_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "RAYON_NUM_THREADS": "1",
                "GOTO_NUM_THREADS": "1",
                "TBB_NUM_THREADS": "1",
                "POLARS_MAX_THREADS": "1",
                "JULIA_NUM_THREADS": "1",
                "LJOS_SEAT": "benchbot",
            }
        )
        proc = subprocess.run(
            [
                "taskset",
                "-c",
                "0",
                str(binary),
                "run",
                args.system,
                str(args.budget),
                str(args.seed),
                "recommended",
                args.cap,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            print(proc.stderr[-500:], file=sys.stderr)
            return 125
        trace = parse_trace(proc.stdout)
        if args.record is not None:
            args.record.write_text("".join(f"{energy}\n" for energy in trace))
            print(f"recorded {len(trace)} steps", flush=True)
            return 0
        if args.expected is None:
            print("expected trace is required", file=sys.stderr)
            return 125
        expected = [line.strip() for line in args.expected.read_text().splitlines() if line.strip()]
        step = first_difference(expected, trace)
        if step is None:
            print(f"match steps={len(trace)}", flush=True)
            return 0
        print(f"differ step={step} expected_len={len(expected)} found_len={len(trace)}", flush=True)
        return 1
    finally:
        example.write_text(original)
        restore(args.worktree)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
