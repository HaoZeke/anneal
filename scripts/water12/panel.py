#!/usr/bin/env python3
# Copyright (c) anneal developers
"""CREST-capped (H2O)12 seeds, plain arm and plateau arm.

Appends one row per finished arm. The call cap is CREST's counted
evaluations on that seed's anneal start. Hostname must be bench.
"""

import os
import re
import subprocess
import time
from pathlib import Path

EH_EV = 27.211386245988
ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get("WATER12_BIN", "/work/water12/bin/gfn2_water_h2h"))
CREST = Path(os.environ.get("CREST_BIN", "/work/crestenv/.pixi/envs/default/bin/crest"))
XTB = Path(os.environ.get("XTB_BIN", "/work/xtb-gfn2"))
RUNS = Path(os.environ.get("WATER12_RUNS", "/work/water12/panel"))
OUT = Path(os.environ.get("WATER12_OUT", str(ROOT / "bench" / "water12" / "panel.csv")))
REVISION = os.environ.get("WATER12_REVISION", "unset")

HEADER = (
    "molecules,seed,arm,calls,wall_s,potential_s,outside_s,"
    "energy_eV,code_revision,host\n"
)


def hostname_is_bench():
    name = os.uname().nodename
    if name != "bench":
        raise SystemExit(f"hostname is {name}, want bench")
    return name


def run(cmd, cwd, env, stdout_path, stderr_path):
    started = time.perf_counter()
    cwd.mkdir(parents=True, exist_ok=True)
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
        'cat "$out"\n'
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
    if not counter.is_file():
        return 0
    total = 0
    for line in counter.read_text().splitlines():
        if line.strip():
            total += int(line)
    return total


def md_steps(text):
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


def crest_runtime(text):
    match = re.search(
        r"CREST runtime \(total\)\s+(\d+) d,\s+(\d+) h,\s+(\d+) min,\s+([0-9.]+) sec",
        text,
    )
    if not match:
        return None
    days, hours, minutes, seconds = (float(part) for part in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def crest_energy_ev(text):
    found = re.findall(r"CREGEN> E lowest\s*:\s*([+-]?\d+\.\d+)", text)
    if not found:
        return ""
    return f"{float(found[-1]) * EH_EV:.10f}"


def recorded(seed, arm):
    if not OUT.is_file():
        return False
    for line in OUT.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 3 and parts[0] == "12" and parts[1] == str(seed) and parts[2] == arm:
            return True
    return False


def append_row(row):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    with OUT.open("a") as handle:
        if new:
            handle.write(HEADER)
        handle.write(row + "\n")


def anneal_start(seed, dest, env):
    dest.mkdir(parents=True, exist_ok=True)
    child = env.copy()
    child["H2H_START_ONLY"] = "1"
    child.pop("H2H_PLATEAU", None)
    subprocess.run(
        [str(BINARY), "12", "1", str(seed), str(dest)],
        cwd=dest,
        env=child,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    start = dest / "start.xyz"
    if not start.is_file():
        raise SystemExit(f"no start for seed {seed}")
    return start


def crest_cap(seed, start, env):
    crest_dir = RUNS / f"seed{seed}" / "crest"
    crest_dir.mkdir(parents=True, exist_ok=True)
    (crest_dir / "start.xyz").write_bytes(start.read_bytes())
    text_path = crest_dir / "crest.out"
    if text_path.is_file() and "CREST runtime" in text_path.read_text(errors="ignore"):
        text = text_path.read_text(errors="ignore")
    else:
        counter = crest_dir / "xtb-calls"
        inside = crest_dir / "xtb-inside"
        wrapper = crest_dir / "xtb-counted"
        write_xtb_wrapper(wrapper, XTB, counter, inside)
        child = env.copy()
        child["OMP_NUM_THREADS"] = "1"
        run(
            [str(CREST), "start.xyz", "--gfn2", "-T", "1", "--legacy", "-xnam", str(wrapper)],
            crest_dir,
            child,
            text_path,
            crest_dir / "crest.err",
        )
        text = text_path.read_text(errors="ignore") if text_path.is_file() else ""
    calls = geometry_calls(crest_dir / "xtb-calls") + md_steps(text)
    if calls <= 0:
        raise SystemExit(f"CREST counted no evaluations for seed {seed}")
    own = crest_runtime(text)
    wall = own if own is not None else 0.0
    inside = 0.0
    inside_path = crest_dir / "xtb-inside"
    if inside_path.is_file():
        inside = sum(float(line) for line in inside_path.read_text().splitlines() if line.strip())
    pot = min(inside, wall) if wall else inside
    outside = max(wall - pot, 0.0)
    if not recorded(seed, "crest"):
        append_row(
            ",".join(
                [
                    "12",
                    str(seed),
                    "crest",
                    str(calls),
                    f"{wall:.6f}",
                    f"{pot:.6f}",
                    f"{outside:.6f}",
                    crest_energy_ev(text),
                    REVISION,
                    "bench",
                ]
            )
        )
    return calls


def anneal(seed, calls, arm, start, env):
    if recorded(seed, arm):
        return
    dest = RUNS / f"seed{seed}" / arm
    dest.mkdir(parents=True, exist_ok=True)
    child = env.copy()
    child["OMP_NUM_THREADS"] = "1"
    child["MKL_NUM_THREADS"] = "1"
    child["OPENBLAS_NUM_THREADS"] = "1"
    child["RAYON_NUM_THREADS"] = "1"
    if arm == "plateau":
        child["H2H_PLATEAU"] = "1"
    else:
        child.pop("H2H_PLATEAU", None)
    child["H2H_START_XYZ"] = str(start)
    run(
        [str(BINARY), "12", str(calls), str(seed), str(dest)],
        dest,
        child,
        dest / "stdout.txt",
        dest / "stderr.txt",
    )
    meta = (dest / "run.txt").read_text()
    append_row(
        ",".join(
            [
                "12",
                str(seed),
                arm,
                field(meta, "energy_calls"),
                field(meta, "wall_s"),
                field(meta, "potential_s"),
                field(meta, "outside_s"),
                field(meta, "engine_best_eV"),
                REVISION,
                "bench",
            ]
        )
    )


def main():
    host = hostname_is_bench()
    assert host == "bench"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    for seed in range(6, 12):
        start = anneal_start(seed, RUNS / f"seed{seed}" / "start", env)
        calls = crest_cap(seed, start, env)
        anneal(seed, calls, "plain", start, env)
        anneal(seed, calls, "plateau", start, env)


if __name__ == "__main__":
    main()
