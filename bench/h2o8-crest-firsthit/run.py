#!/usr/bin/env python3
# Copyright (c) anneal developers
"""CREST first-hit calls for the GFN2 water octamer.

The invocation is the recorded one: crest start.xyz --gfn2 -T 1 --legacy,
with xtb 6.7.1 behind -xnam and one thread. A counted call is a finished
MD step (MD time over the timestep) or one optimization cycle, the same
accounting as the recorded full-run totals. The cycle energy is the
full-precision value from the optimization log. The run stops at the
budget. A miss is budget + 1.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REFERENCE_EH = -40.681336013150
TOLERANCE_EH = 1e-6
THRESHOLD_EH = REFERENCE_EH + TOLERANCE_EH
BUDGET = 151738
MISS = BUDGET + 1
RECORDED_17000 = 127141
HEAVY = {"O": 2}
CUTOFF_A = 1.2

STAR_RE = re.compile(r"(?m)^\s*\* total energy\s*:\s*([+-]?\d+\.\d+)")
COLON_RE = re.compile(r"(?m)^\s*:: total energy\s+([+-]?\d+\.\d+)")
LOG_ENERGY_RE = re.compile(r"^ energy:\s+([+-]?\d+\.\d+)")
MD_TIME_RE = re.compile(r"MD time /ps\s*:\s*([0-9.]+)")
MD_DT_RE = re.compile(r"dt /fs\s*:\s*([0-9.]+)")
MD_DUMP_RE = re.compile(r"dumpstep\(trj\) /fs\s*:\s*([0-9.]+)")
MD_STEPS_RE = re.compile(r"max steps\s*:\s*(\d+)")


def intact(atoms: list[tuple[str, list[float]]], expect: dict[str, int] = HEAVY) -> bool:
    if not atoms:
        return False
    hydrogens = [xyz for sym, xyz in atoms if sym.upper() == "H"]
    heavies = [(sym.upper(), xyz) for sym, xyz in atoms if sym.upper() in expect]
    if not heavies:
        return False

    def near(a: list[float], b: list[float]) -> bool:
        return math.dist(a, b) <= CUTOFF_A

    return all(sum(near(xyz, h) for h in hydrogens) == expect[sym] for sym, xyz in heavies)


def read_frames(path: Path) -> list[tuple[float | None, list[tuple[str, list[float]]]]]:
    if not path.is_file():
        return []
    lines = path.read_text(errors="replace").splitlines()
    frames = []
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if not parts:
            i += 1
            continue
        try:
            count = int(parts[0])
        except ValueError:
            i += 1
            continue
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        body = lines[i + 2 : i + 2 + count]
        if len(body) < count:
            break
        energy = None
        match = LOG_ENERGY_RE.match(comment) or re.search(r"energy:\s+([+-]?\d+\.\d+)", comment)
        if match:
            energy = float(match.group(1))
        atoms = []
        for row in body:
            item = row.split()
            if len(item) < 4:
                continue
            atoms.append((item[0], [float(v) for v in item[1:4]]))
        frames.append((energy, atoms))
        i += 2 + count
    return frames


def md_blocks_of(text: str) -> list[int]:
    """Finished MD lengths, each MD time in ps times 1000 over dt in fs.

    A timing trial that never prints a finished block is absent here, which
    is the accounting used for the recorded totals.
    """
    rows = text.splitlines()
    blocks = []
    i = 0
    while i < len(rows):
        if "MD time /ps" not in rows[i]:
            i += 1
            continue
        try:
            length = float(rows[i].split(":")[1].split()[0])
        except (IndexError, ValueError):
            i += 1
            continue
        dt = None
        finished = False
        j = i + 1
        while j < len(rows) and "MD time /ps" not in rows[j]:
            if "dt /fs" in rows[j] and dt is None:
                try:
                    dt = float(rows[j].split(":")[1].split()[0])
                except (IndexError, ValueError):
                    dt = None
            if "finished" in rows[j].lower():
                finished = True
                break
            j += 1
        if finished and dt:
            blocks.append(int(round(length * 1000.0 / dt)))
        i = j
    return blocks


def md_steps_of(text: str) -> int:
    return sum(md_blocks_of(text))


def kind_of(args: list[str]) -> str:
    text = " ".join(args)
    if "--md" in text or "--omd" in text:
        return "md"
    if "--opt" in text:
        return "opt"
    return "other"


def gfn2_args(args: list[str]) -> bool:
    text = " ".join(args)
    return "gfn0" not in text and "gfnff" not in text and "gfn-ff" not in text


def first_intact(energies: list[float], frames: list, threshold: float) -> int:
    for index, energy in enumerate(energies, start=1):
        if energy > threshold:
            continue
        atoms = frames[index - 1] if index - 1 < len(frames) else None
        if atoms is None or intact(atoms):
            return index
    return 0


def kill_crest() -> None:
    pid = os.getpid()
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_bytes()
        except OSError:
            return
        end = stat.rfind(b")")
        ppid = int(stat[end + 2 :].split()[1])
        pid = ppid
        if pid <= 1:
            return
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        if b"crest" in cmd and b"start.xyz" in cmd:
            os.kill(pid, signal.SIGTERM)
            return


def load_count(state: Path) -> int:
    path = state / "count.txt"
    if not path.is_file():
        return 0
    return int(path.read_text().strip() or "0")


def store_count(state: Path, count: int) -> None:
    path = state / "count.txt"
    path.write_text(f"{count}\n")


def append_event(state: Path, event: dict) -> None:
    path = state / "events.jsonl"
    with path.open("a") as handle:
        handle.write(json.dumps(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def wrap_main() -> None:
    state = Path(os.environ["H2O8_STATE"])
    xtb = os.environ["H2O8_XTB"]
    budget = int(os.environ.get("H2O8_BUDGET", str(BUDGET)))
    threshold = float(os.environ.get("H2O8_THRESHOLD", str(THRESHOLD_EH)))
    args = sys.argv[1:]
    kind = kind_of(args)
    gfn2 = gfn2_args(args)
    count = load_count(state)
    if count >= budget:
        append_event(state, {"kind": kind, "n": 0, "skipped": True, "argv": args})
        kill_crest()
        raise SystemExit(0)

    run_args = [arg for arg in args if arg != "--ceasefiles"]
    cwd = Path.cwd()
    present = {path.name for path in cwd.iterdir()}
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    proc = subprocess.run([xtb, *run_args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    sys.stdout.buffer.write(proc.stdout)
    sys.stdout.buffer.flush()
    text = proc.stdout.decode("utf-8", errors="replace")
    room = budget - count
    event: dict = {"kind": kind, "argv": args, "gfn2": gfn2, "rc": proc.returncode}

    if kind == "opt":
        stars = [float(value) for value in STAR_RE.findall(text)]
        colons = [float(value) for value in COLON_RE.findall(text)]
        frames = read_frames(cwd / "xtbopt.log")
        log_energies = [energy for energy, _atoms in frames if energy is not None]
        geometries = [atoms for energy, atoms in frames if energy is not None]
        if stars and len(log_energies) == len(stars):
            energies = log_energies
        elif stars:
            energies = stars
            geometries = []
        else:
            energies = colons
            geometries = []
        event["log_match"] = bool(stars) and len(log_energies) == len(stars)
        event["n_full"] = len(energies)
        kept = energies[:room]
        kept_geom = geometries[: len(kept)] if geometries else []
        event["n"] = len(kept)
        event["energies"] = kept
        event["min_e"] = min(kept) if kept else None
        event["first_local"] = first_intact(kept, kept_geom, threshold) if gfn2 else 0
        final = colons[-1] if stars and colons else None
        event["final_e"] = final
        event["final_intact"] = False
        if gfn2 and final is not None and final <= threshold and len(kept) == len(energies):
            final_frames = read_frames(cwd / "xtbopt.xyz")
            atoms = final_frames[0][1] if final_frames else []
            event["final_intact"] = intact(atoms) if atoms else False
        # Drop files the ceasefiles switch would not have left behind.
        for path in list(cwd.iterdir()):
            if path.name in present or path.name in {"xtbopt.xyz", ".xtboptok"}:
                continue
            if path.is_file():
                path.unlink()
    elif kind == "md":
        steps = md_steps_of(text)
        dump_match = MD_DUMP_RE.search(text)
        dt_match = MD_DT_RE.search(text)
        dump_fs = float(dump_match.group(1)) if dump_match else None
        dt_fs = float(dt_match.group(1)) if dt_match else None
        frames = read_frames(cwd / "xtb.trj")
        min_e = None
        hit_step = 0
        for index, (energy, atoms) in enumerate(frames):
            if energy is None:
                continue
            if min_e is None or energy < min_e:
                min_e = energy
            if index == 0 or not gfn2 or energy > threshold:
                continue
            if dump_fs and dt_fs:
                step = int(round(index * dump_fs / dt_fs))
            else:
                step = index
            if step < 1 or step > steps:
                continue
            if intact(atoms):
                hit_step = step
                break
        event["n_full"] = steps
        # The short timing trial is not a finished block in the crest log.
        counted = steps if steps >= 400 else 0
        event["n"] = min(counted, room)
        event["min_e"] = min_e
        event["hit_step"] = hit_step if counted and hit_step and hit_step <= event["n"] else 0
        event["dump_fs"] = dump_fs
        event["dt_fs"] = dt_fs
    else:
        event["n"] = 0
        event["min_e"] = None
        event["first_local"] = 0

    append_event(state, event)
    done = count + int(event["n"])
    store_count(state, done)
    if done % 1000 < int(event["n"]) or done >= budget:
        (state / "progress.txt").write_text(f"{done}\n")
    if done >= budget:
        kill_crest()
    raise SystemExit(proc.returncode)


def crest_identity(crest: Path) -> tuple[str, str]:
    proc = subprocess.run([str(crest), "--version"], capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    version = re.search(r"Version\s+(\d+\.\d+\.\d+)", text)
    revision = re.search(r"commit\s+\(([0-9a-fA-F]+)\)", text)
    if not version or not revision:
        raise SystemExit("crest --version did not report a version and commit")
    return version.group(1), revision.group(1)


def parse_time(path: Path) -> tuple[float, float]:
    text = path.read_text(errors="replace")
    user = re.search(r"User time \(seconds\):\s+([0-9.]+)", text)
    system = re.search(r"System time \(seconds\):\s+([0-9.]+)", text)
    elapsed = re.search(r"Elapsed \(wall clock\) time \(.*\):\s+(\S+)", text)
    if not user or not system or not elapsed:
        raise SystemExit(f"missing times in {path}")
    cpu_s = float(user.group(1)) + float(system.group(1))
    stamp = elapsed.group(1)
    parts = stamp.split(":")
    if len(parts) == 3:
        wall_s = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        wall_s = int(parts[0]) * 60 + float(parts[1])
    else:
        wall_s = float(parts[0])
    return wall_s, cpu_s


def summarize(events: list[dict], crest_out: str) -> dict:
    blocks = md_blocks_of(crest_out)
    block_at = 0
    offset = 0
    first = None
    best = None
    final_only = None
    opt_cycles = 0
    wrapper_md = 0
    for event in events:
        kind = event.get("kind")
        n = int(event.get("n") or 0)
        if kind == "md":
            full = int(event.get("n_full") or n)
            if block_at < len(blocks) and full == blocks[block_at]:
                block_at += 1
                n = min(n if n else full, blocks[block_at - 1])
            elif n and full and n < full:
                n = min(n, BUDGET - offset)
            else:
                continue
            wrapper_md += n
            step = int(event.get("hit_step") or 0)
            if step and first is None and step <= n:
                first = offset + step
            energy = event.get("min_e")
            if energy is not None and (best is None or energy < best):
                best = energy
        elif kind == "opt":
            opt_cycles += n
            local = int(event.get("first_local") or 0)
            if local and first is None and local <= n:
                first = offset + local
            for energy in event.get("energies") or []:
                if best is None or energy < best:
                    best = energy
            final = event.get("final_e")
            if isinstance(final, float) and (best is None or final < best):
                best = final
            if (
                final_only is None
                and event.get("final_intact")
                and isinstance(final, float)
                and final <= THRESHOLD_EH
                and not local
            ):
                final_only = offset + n
        else:
            n = 0
        offset += n
        if offset >= BUDGET:
            offset = BUDGET
            break
    return {
        "total_calls": min(offset, BUDGET),
        "first_hit_call": first if first is not None else MISS,
        "best_Eh": best,
        "crest_out_md_steps": sum(blocks),
        "wrapper_md": wrapper_md,
        "opt_cycles": opt_cycles,
        "final_only_at": final_only,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = [
        "seed",
        "first_hit_call",
        "total_calls",
        "hit",
        "best_Eh",
        "wall_s",
        "cpu_s",
        "version",
        "revision",
    ]
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row[name]) for name in fields))
    path.write_text("\n".join(lines) + "\n")


def load_finished(out: Path) -> list[dict]:
    rows = []
    for path in sorted(out.glob("*/row.json"), key=lambda item: int(item.parent.name)):
        rows.append(json.loads(path.read_text()))
    return rows


def generate_start(binary: Path, seed: int, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["H2H_START_ONLY"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [str(binary), "h2o-8", "4000", str(seed), str(dest)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0 or not (dest / "start.xyz").is_file():
        raise SystemExit(f"start {seed} failed\n{proc.stdout}")


def run_seed(
    seed: int,
    out: Path,
    binary: Path,
    crest: Path,
    xtb: Path,
    runner: Path,
    version: str,
    revision: str,
) -> dict:
    seed_dir = out / str(seed)
    if (seed_dir / "row.json").is_file():
        return json.loads((seed_dir / "row.json").read_text())
    seed_dir.mkdir(parents=True, exist_ok=True)
    generate_start(binary, seed, seed_dir / "startgen")
    work = seed_dir / "crest"
    if work.exists():
        for path in sorted(work.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    work.mkdir(parents=True, exist_ok=True)
    (work / "start.xyz").write_text((seed_dir / "startgen" / "start.xyz").read_text())
    for name in ("events.jsonl", "count.txt"):
        path = seed_dir / name
        if path.exists():
            path.unlink()
    wrapper = seed_dir / "xtb-count"
    wrapper.write_text(
        "#!/bin/sh\n"
        "export H2O8_CREST_WRAP=1\n"
        f"export H2O8_STATE='{seed_dir}'\n"
        f"export H2O8_XTB='{xtb}'\n"
        f"export H2O8_BUDGET={BUDGET}\n"
        f"export H2O8_THRESHOLD='{THRESHOLD_EH:.12f}'\n"
        "export OMP_NUM_THREADS=1\n"
        "export MKL_NUM_THREADS=1\n"
        "export OPENBLAS_NUM_THREADS=1\n"
        "export OMP_DYNAMIC=FALSE\n"
        f"exec python3 '{runner}' \"$@\"\n"
    )
    wrapper.chmod(0o755)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    time_path = (seed_dir / "time.txt").resolve()
    stdout_path = seed_dir / "crest.out"
    stderr_path = seed_dir / "crest.err"
    command = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(time_path),
        "taskset",
        "-c",
        "0",
        str(crest),
        "start.xyz",
        "--gfn2",
        "-T",
        "1",
        "--legacy",
        "-xnam",
        str(wrapper),
    ]
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.run(command, cwd=work, env=env, stdout=stdout, stderr=stderr)
    wall_s, cpu_s = parse_time(time_path)
    events = []
    events_path = seed_dir / "events.jsonl"
    if events_path.is_file():
        for line in events_path.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))
    summary = summarize(events, stdout_path.read_text(errors="replace"))
    hit_call = int(summary["first_hit_call"])
    if hit_call > BUDGET:
        hit_call = MISS
    best = summary["best_Eh"]
    row = {
        "seed": seed,
        "first_hit_call": hit_call,
        "total_calls": int(summary["total_calls"]),
        "hit": 1 if hit_call <= BUDGET else 0,
        "best_Eh": f"{best:.12f}" if best is not None else "",
        "wall_s": f"{wall_s:.6f}",
        "cpu_s": f"{cpu_s:.6f}",
        "version": version,
        "revision": revision,
        "crest_out_md_steps": summary["crest_out_md_steps"],
        "wrapper_md": summary["wrapper_md"],
        "opt_cycles": summary["opt_cycles"],
        "final_only_at": summary["final_only_at"],
        "exit_code": proc.returncode,
    }
    (seed_dir / "row.json").write_text(json.dumps(row) + "\n")
    print(
        f"seed {seed} first_hit_call={row['first_hit_call']} total_calls={row['total_calls']} "
        f"hit={row['hit']} best_Eh={row['best_Eh']} wall_s={row['wall_s']} "
        f"md_steps={row['crest_out_md_steps']} wrapper_md={row['wrapper_md']} "
        f"opt_cycles={row['opt_cycles']} "
        f"final_only_at={row['final_only_at']} exit={proc.returncode}",
        flush=True,
    )
    return row


def campaign(argv: list[str]) -> None:
    out = Path(argv[argv.index("--out") + 1] if "--out" in argv else "runs").resolve()
    raw = argv[argv.index("--seeds") + 1]
    seeds = []
    for part in raw.split(","):
        if "-" in part:
            left, right = part.split("-")
            seeds.extend(range(int(left), int(right) + 1))
        else:
            seeds.append(int(part))
    hours = float(argv[argv.index("--hours") + 1]) if "--hours" in argv else 3.8
    binary = Path(os.environ["H2H_BIN"])
    crest = Path(os.environ["CREST_BIN"])
    xtb = Path(os.environ["XTB_BIN"])
    for path in (binary, crest, xtb):
        if not path.is_file():
            raise SystemExit(f"missing {path.name}")
    out.mkdir(parents=True, exist_ok=True)
    version, revision = crest_identity(crest)
    runner = Path(__file__).resolve()
    started = time.time()
    deadline = started + hours * 3600.0
    rows_path = out.parent / "rows.csv" if out.name == "runs" else out / "rows.csv"
    if "--rows" in argv:
        rows_path = Path(argv[argv.index("--rows") + 1]).resolve()
    for seed in seeds:
        if (out / str(seed) / "row.json").is_file():
            print(f"seed {seed} already finished", flush=True)
            continue
        if time.time() > deadline:
            print(f"deadline reached before seed {seed}", flush=True)
            break
        run_seed(seed, out, binary, crest, xtb, runner, version, revision)
        public = []
        for item in load_finished(out):
            public.append({key: item[key] for key in (
                "seed",
                "first_hit_call",
                "total_calls",
                "hit",
                "best_Eh",
                "wall_s",
                "cpu_s",
                "version",
                "revision",
            )})
        write_rows(rows_path, public)
        done = next(item for item in load_finished(out) if item["seed"] == seed)
        if seed == 17000:
            match = done["total_calls"] == RECORDED_17000
            logged = done["crest_out_md_steps"] + done["opt_cycles"]
            print(
                f"CHECK seed 17000 total_calls={done['total_calls']} "
                f"recorded={RECORDED_17000} match={int(match)} "
                f"crest_out_plus_opt={logged}",
                flush=True,
            )


def main() -> None:
    if os.environ.get("H2O8_CREST_WRAP") == "1":
        wrap_main()
        return
    campaign(sys.argv[1:])


if __name__ == "__main__":
    main()
