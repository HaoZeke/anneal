#!/usr/bin/env python3
# Copyright (c) anneal developers
"""GFN2 water against CREST at one call cap per seed.

Sizes 6, 8 and 12, seeds 0 through 5. CREST runs on the anneal start.
Anneal then spends that many potential calls. The stall rows are the
paper budget of 4000 on the two seeds that previously sat in an SCC
rejection: hexamer seed 5 and (H2O)12 seed 5.
"""

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

EH_EV = 27.211386245988
SIZES = (6, 8, 12)
SEEDS = range(6)
PAPER_BUDGET = 4000
STALL_CASES = ((6, 5), (12, 5))

STALL_HEADER = (
    "molecules,seed,code,budget,wall_s,calls,scc_rejected,"
    "potential_s,outside_s,energy_eV,status\n"
)
CREST_HEADER = (
    "molecules,seed,arm,calls,wall_s,potential_s,outside_s,energy_eV\n"
)


def hostname_is_bench():
    name = os.uname().nodename
    if name != "bench":
        raise SystemExit(f"hostname is {name}, want bench")


def run(cmd, cwd, env, stdout_path, stderr_path):
    started = time.perf_counter()
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=out, stderr=err)
    return proc.returncode, time.perf_counter() - started


def field(text, key):
    match = re.search(rf"(?m)^{re.escape(key)}: (.*)$", text)
    return match.group(1).strip() if match else ""


def write_xtb_wrapper(path, real, counter, inside):
    path.write_text(
        "#!/bin/sh\n"
        f"real='{real}'\n"
        "out=$(mktemp)\n"
        'start=$(date +%s.%N)\n'
        '"$real" "$@" >"$out" 2>&1\n'
        "rc=$?\n"
        'end=$(date +%s.%N)\n'
        "cat \"$out\"\n"
        "python3 - \"$out\" \"$start\" \"$end\" << 'PY'\n"
        "import re, sys\n"
        "text = open(sys.argv[1], errors='ignore').read()\n"
        "cycles = len(re.findall(r'(?m)^\\s*\\* total energy\\b', text))\n"
        "if not cycles:\n"
        "    cycles = len(re.findall(r'(?m):: total energy\\b', text))\n"
        f"open('{counter}', 'a').write(str(cycles) + '\\n')\n"
        "start, end = float(sys.argv[2]), float(sys.argv[3])\n"
        f"open('{inside}', 'a').write(f'{{end - start:.6f}}\\n')\n"
        "PY\n"
        'rm -f "$out"\n'
        "exit $rc\n"
    )
    path.chmod(0o755)


def geometry_calls(counter):
    """Converged xtb energies. One `* total energy` is one geometry."""
    if not counter.is_file():
        return 0
    total = 0
    for line in counter.read_text().splitlines():
        if line.strip():
            total += int(line)
    return total


def md_steps(text):
    """Finished meta-MD steps. Each block prints `MD time /ps` then `dt /fs`."""
    times = []
    dts = []
    for line in text.splitlines():
        if "MD time /ps" in line:
            times.append(float(line.split(":")[1]))
        elif re.search(r"^\s*dt /fs\s*:", line):
            dts.append(float(line.split(":")[1]))
    finished = len(re.findall(r"Meta-MTD .* finished", text))
    steps = 0
    for duration, step in list(zip(times, dts))[:finished]:
        if step > 0:
            steps += int(round(duration * 1000.0 / step))
    return steps


def crest_inside(path):
    if not path.is_file():
        return 0.0
    return sum(float(line) for line in path.read_text().splitlines() if line.strip())


def crest_energy_ev(text):
    found = re.findall(r"CREGEN> E lowest\s*:\s*([+-]?\d+\.\d+)", text)
    if not found:
        return ""
    return f"{float(found[-1]) * EH_EV:.10f}"


def crest_runtime(text):
    match = re.search(
        r"CREST runtime \(total\)\s+(\d+) d,\s+(\d+) h,\s+(\d+) min,\s+([0-9.]+) sec",
        text,
    )
    if not match:
        return None
    days, hours, minutes, seconds = (float(part) for part in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def anneal_start(binary, molecules, seed, dest, env):
    dest.mkdir(parents=True, exist_ok=True)
    child = env.copy()
    child["H2H_START_ONLY"] = "1"
    subprocess.run(
        [str(binary), str(molecules), "1", str(seed), str(dest)],
        cwd=dest,
        env=child,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    start = dest / "start.xyz"
    if not start.is_file():
        raise SystemExit(f"no start for m={molecules} seed={seed}")
    return start


def append_row(path, header, row):
    new = not path.exists()
    with path.open("a") as handle:
        if new:
            handle.write(header)
        handle.write(row + "\n")


def run_stall(binary, code, molecules, seed, runs, out, env):
    dest = runs / "stall" / code / f"m{molecules}" / f"seed{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    code_rc, _ = run(
        [str(binary), str(molecules), str(PAPER_BUDGET), str(seed), str(dest)],
        dest,
        env,
        dest / "stdout.txt",
        dest / "stderr.txt",
    )
    meta = (dest / "run.txt").read_text() if (dest / "run.txt").is_file() else ""
    status = "ran" if code_rc == 0 and field(meta, "wall_s") else "stopped"
    append_row(
        out / "stall.csv",
        STALL_HEADER,
        ",".join(
            [
                str(molecules),
                str(seed),
                code,
                str(PAPER_BUDGET),
                field(meta, "wall_s"),
                field(meta, "energy_calls"),
                field(meta, "scc_rejected") or "0",
                field(meta, "potential_s"),
                field(meta, "outside_s"),
                field(meta, "engine_best_eV"),
                status,
            ]
        ),
    )


def recorded(path, molecules, seed, arm):
    if not path.is_file():
        return False
    for line in path.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 3 and parts[0] == str(molecules) and parts[1] == str(seed) and parts[2] == arm:
            return True
    return False


def run_pair(binary, crest, xtb, molecules, seed, runs, out, env):
    if recorded(out / "crest.csv", molecules, seed, "crest") and recorded(
        out / "crest.csv", molecules, seed, "anneal"
    ):
        return
    base = runs / "crest" / f"m{molecules}" / f"seed{seed}"
    start_dir = base / "start"
    start = anneal_start(binary, molecules, seed, start_dir, env)
    crest_dir = base / "crest"
    crest_dir.mkdir(parents=True, exist_ok=True)
    (crest_dir / "start.xyz").write_bytes(start.read_bytes())
    counter = crest_dir / "xtb-calls"
    inside = crest_dir / "xtb-inside"
    wrapper = crest_dir / "xtb-counted"
    write_xtb_wrapper(wrapper, xtb, counter, inside)
    child = env.copy()
    child["OMP_NUM_THREADS"] = "1"
    child["MKL_NUM_THREADS"] = "1"
    child["OPENBLAS_NUM_THREADS"] = "1"
    already = (crest_dir / "crest.out").is_file() and "CREST runtime" in (
        crest_dir / "crest.out"
    ).read_text(errors="ignore")
    if already:
        crest_rc, crest_wall = 0, 0.0
    else:
        crest_rc, crest_wall = run(
            [str(crest), "start.xyz", "--gfn2", "-T", "1", "--legacy", "-xnam", str(wrapper)],
            crest_dir,
            child,
            crest_dir / "crest.out",
            crest_dir / "crest.err",
        )
    text = (crest_dir / "crest.out").read_text(errors="ignore") if (crest_dir / "crest.out").is_file() else ""
    calls = geometry_calls(counter) + md_steps(text)
    if calls <= 0:
        raise SystemExit(f"CREST counted no evaluations for m={molecules} seed={seed} exit {crest_rc}")
    own = crest_runtime(text)
    wall_s = own if own is not None else crest_wall
    # The wrapper's date deltas drift and can sum past CREST's own runtime.
    pot_s = min(crest_inside(inside), wall_s)
    outside_s = wall_s - pot_s
    wall = f"{wall_s:.6f}"
    pot = f"{pot_s:.6f}"
    outside = f"{outside_s:.6f}"
    append_row(
        out / "crest.csv",
        CREST_HEADER,
        ",".join(
            [
                str(molecules),
                str(seed),
                "crest",
                str(calls),
                wall,
                pot,
                outside,
                crest_energy_ev(text),
            ]
        ),
    )
    anneal_dir = base / "anneal"
    anneal_dir.mkdir(parents=True, exist_ok=True)
    run(
        [str(binary), str(molecules), str(calls), str(seed), str(anneal_dir)],
        anneal_dir,
        env,
        anneal_dir / "stdout.txt",
        anneal_dir / "stderr.txt",
    )
    meta = (anneal_dir / "run.txt").read_text()
    append_row(
        out / "crest.csv",
        CREST_HEADER,
        ",".join(
            [
                str(molecules),
                str(seed),
                "anneal",
                field(meta, "energy_calls"),
                field(meta, "wall_s"),
                field(meta, "potential_s"),
                field(meta, "outside_s"),
                field(meta, "engine_best_eV"),
            ]
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--unfixed", default="")
    parser.add_argument("--crest", required=True)
    parser.add_argument("--xtb", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stall-only", action="store_true")
    parser.add_argument("--crest-only", action="store_true")
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()
    hostname_is_bench()
    runs = Path(args.runs)
    out = Path(args.out)
    runs.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    if not args.crest_only and args.unfixed:
        for molecules, seed in STALL_CASES:
            if args.size and molecules != args.size:
                continue
            if args.seed >= 0 and seed != args.seed:
                continue
            run_stall(args.unfixed, "unfixed", molecules, seed, runs, out, env)
            run_stall(args.binary, "fixed", molecules, seed, runs, out, env)
    if args.stall_only:
        return
    sizes = (args.size,) if args.size else SIZES
    seeds = (args.seed,) if args.seed >= 0 else SEEDS
    for molecules in sizes:
        for seed in seeds:
            run_pair(args.binary, args.crest, args.xtb, molecules, seed, runs, out, env)


if __name__ == "__main__":
    main()
