#!/usr/bin/env python3
"""Evidence for the engine buffer adaptation, one short search per case.

The real GFN2 engine hands back the force, minus dE/dx. Code A publishes that
buffer unchanged as a force. Code B negates it as if it were dE/dx, so it
needs the shim in gradient mode. This script prints the finite-difference
sign check of the engine and of both shim modes, then runs one seed of each
code at a short budget with the buffer it is written for and with the other
one.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def run(argv, env, cwd):
    return subprocess.run([str(a) for a in argv], env=env, cwd=cwd, capture_output=True, text=True)


def report_fields(text):
    fields, inside = {}, False
    for line in text.splitlines():
        if line.startswith("seed: "):
            inside = True
        if inside and ": " in line:
            key, value = line.split(": ", 1)
            fields.setdefault(key, value.strip())
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--shim", required=True)
    parser.add_argument("--binary-a", required=True)
    parser.add_argument("--binary-b", required=True)
    parser.add_argument("--geometry", required=True, help="xyz of a minimum for the finite differences")
    parser.add_argument("--references-dir", required=True)
    parser.add_argument("--water-xyz", required=True)
    parser.add_argument("--budget", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name in ("engine", "shim", "binary_a", "binary_b", "geometry", "references_dir", "water_xyz"):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))

    here = Path(__file__).resolve().parent
    base = {k: v for k, v in os.environ.items() if not k.startswith(("WATER_", "PAPER_", "RGPOT_", "XTB_SHIM_"))}
    base["OMP_NUM_THREADS"] = "1"

    print("finite-difference sign check of the buffer")
    cases = [
        ("real engine", ["--library", args.engine]),
        ("shim, force mode", ["--library", args.shim, "--real-engine", args.engine, "--buffer", "force"]),
        ("shim, gradient mode", ["--library", args.shim, "--real-engine", args.engine, "--buffer", "gradient"]),
    ]
    for label, extra in cases:
        done = run([sys.executable, here / "sign_check.py", "--geometry", args.geometry, *extra], base, here)
        print(f"\n{label}")
        print(done.stdout.rstrip() if done.returncode == 0 else f"failed: {done.stderr.strip()[-200:]}")

    print("\none seed, budget %d" % args.budget)
    print("code | engine buffer | best energy eV | monomers intact | force calls | isomer | reason")
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        for code, binary in (("A", args.binary_a), ("B", args.binary_b)):
            for label, library, mode in (
                ("real engine", args.engine, None),
                ("shim force", args.shim, "force"),
                ("shim gradient", args.shim, "gradient"),
            ):
                env = dict(base)
                env["RGPOT_XTB_ENGINE"] = str(library)
                if mode:
                    env["XTB_SHIM_REAL"] = args.engine
                    env["XTB_SHIM_BUFFER"] = mode
                    env["XTB_SHIM_STATS"] = str(scratch / "stats.bin")
                env["WATER_ISOMER_DIR"] = args.references_dir
                env["WATER_FORCE_BUDGET"] = str(args.budget)
                env["WATER_SEED_FIRST"] = str(args.seed)
                env["WATER_SEED_COUNT"] = "1"
                if code == "B":
                    env["WATER_START_XYZ"] = args.water_xyz
                done = run([binary, "gfn2", "recommended"], env, scratch)
                fields = report_fields(done.stdout)
                print(
                    f"{code} | {label} | {fields.get('best_energy', 'n/a')} | "
                    f"{fields.get('best_monomers_intact', 'n/a')} | {fields.get('force_calls', 'n/a')} | "
                    f"{fields.get('isomer', 'n/a')} | {fields.get('isomer_reason', 'n/a')}"
                )


if __name__ == "__main__":
    main()
