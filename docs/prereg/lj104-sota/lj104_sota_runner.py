#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial gate for the pre-registered LJ104 comparison.

A scored start takes an exclusive lock and will not start while another
scored start holds it. Every row this process writes cites the git blob
of this file, HEAD, and the tree of HEAD. Arm B stays UNFILLED.
Arm D is the lattice v3.2 tip. The held-out seeds are 5000-5029.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HELD_OUT_LO = 5000
HELD_OUT_HI = 5029
PRESET_ID = "UNFILLED"
D_V3_TIP = "28ece242da192e769865bc1b1d93fa8e3a50b7a2"
LJ104_N = 104
LJ104_BUDGET = 4_000_000
E_KNOWN_LJ104 = -582.086642
HIT_LINE = E_KNOWN_LJ104 + 1e-6
DLS_TIE = "lowest-index"
DLS_STOP = "-582.086641"
# Stated before any panel run. Each process stops at this wall-clock cap.
WALL_CLOCK_CAP_SECONDS = 600
COLUMNS = (
    "runner_blob",
    "head",
    "tree",
    "seed",
    "code",
    "hit",
    "energy",
    "calls",
    "wall_seconds",
)
# examples/h2h_arm3.rs lines 46-49, and PAPER_BUDGETS lines 16-19.
PANEL_TOL = 0.001
PANEL_SEEDS = (6000, 6001, 6002, 6003, 6004, 6005)
PANEL_SYSTEMS = (
    (38, 400_000, -173.928427),
    (55, 1_000_000, -279.248470),
    (75, 4_000_000, -397.492331),
    (98, 4_000_000, -543.665361),
)


def script_path() -> Path:
    return Path(__file__).resolve()


def repo_root() -> Path:
    here = script_path().parent
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    print("runner is not inside a git work tree", file=sys.stderr)
    raise SystemExit(1)


def git_out(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root()), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr.strip() or "git failed", file=sys.stderr)
        raise SystemExit(1)
    return proc.stdout.strip()


def provenance() -> dict[str, str]:
    """Git blob of this runner, HEAD, and the tree of HEAD."""
    fields = {
        "runner_blob": git_out("hash-object", "--", str(script_path())),
        "head": git_out("rev-parse", "HEAD"),
        "tree": git_out("rev-parse", "HEAD^{tree}"),
    }
    missing = [name for name in ("runner_blob", "head", "tree") if not fields[name]]
    if missing:
        print("scored row is missing " + ", ".join(missing), file=sys.stderr)
        raise SystemExit(1)
    return fields


def default_lock_file() -> Path:
    return script_path().parent / ".scored-run.lock"


def acquire_scored_run(lock_file: Path):
    """Exclusive non-blocking lock. Exit 2 when another scored run is live."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_file.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        print("another scored run is live", file=sys.stderr)
        raise SystemExit(2)
    return handle


def release_scored_run(handle) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def iter_seeds(spec: str):
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_text, hi_text = part.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                print(f"seed range {part} is reversed", file=sys.stderr)
                raise SystemExit(2)
            yield from range(lo, hi + 1)
        else:
            yield int(part)


def touches_held_out(spec: str) -> bool:
    return any(HELD_OUT_LO <= seed <= HELD_OUT_HI for seed in iter_seeds(spec))


def write_row(path: Path, fields: dict[str, str]) -> None:
    for name in ("runner_blob", "head", "tree"):
        if not fields.get(name):
            print("scored row is missing " + name, file=sys.stderr)
            raise SystemExit(1)
    header = "\t".join(COLUMNS)
    body = "\t".join(fields.get(column, "") for column in COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + body + "\n", encoding="utf-8")


def record_serial_boundary(dest: Path, label: str) -> None:
    """Start or end receipt for one strictly serial process.

    The receipt is the kernel boot id, the uptime reading, and lscpu.
    A scored run keeps the start receipt and the end receipt. Two
    different boot ids mean the process did not stay on one boot.
    """
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    uptime = Path("/proc/uptime").read_text(encoding="utf-8").strip()
    rustc = subprocess.run(["rustc", "-V"], check=False, capture_output=True, text=True)
    cargo = subprocess.run(["cargo", "-V"], check=False, capture_output=True, text=True)
    rustc_line = (rustc.stdout if rustc.returncode == 0 else rustc.stderr).strip()
    cargo_line = (cargo.stdout if cargo.returncode == 0 else cargo.stderr).strip()
    mhz = [
        line
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        if line.lower().startswith("cpu mhz")
    ]
    listing = subprocess.run(
        ["lscpu"],
        check=False,
        capture_output=True,
        text=True,
    )
    lscpu_text = listing.stdout if listing.returncode == 0 else listing.stderr
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{label}.receipt").write_text(
        "boot_id "
        + boot_id
        + "\n"
        + "uptime "
        + uptime
        + "\n"
        + rustc_line
        + "\n"
        + cargo_line
        + "\n"
        + "\n".join(mhz)
        + "\n"
        + lscpu_text,
        encoding="utf-8",
    )


def panel_jobs():
    """The 24 default-panel runs, one size at a time, seeds 6000-6005."""
    jobs = []
    for n, budget, reference in PANEL_SYSTEMS:
        for seed in PANEL_SEEDS:
            if HELD_OUT_LO <= seed <= HELD_OUT_HI:
                raise SystemExit("panel seed list overlaps 5000-5029")
            jobs.append(
                {
                    "n": n,
                    "budget": budget,
                    "reference": reference,
                    "seed": seed,
                    "stop": reference + PANEL_TOL,
                }
            )
    return jobs


def parse_h2h(text: str) -> dict[str, str]:
    fields = {}
    for line in text.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        fields[key] = value
    return fields


def cmd_panel(args: argparse.Namespace) -> int:
    jobs = panel_jobs()
    if len(jobs) != 24:
        print("the default panel is 24 runs", file=sys.stderr)
        return 2
    root = repo_root()
    binary = Path(args.binary)
    if not binary.is_absolute():
        binary = root / binary
    if not binary.is_file():
        print("missing h2h_seed binary", file=sys.stderr)
        return 2
    rel = Path(args.output_dir)
    if rel.is_absolute():
        print("output dir must be relative", file=sys.stderr)
        return 2
    out = root / rel
    out.mkdir(parents=True, exist_ok=True)
    rustc = subprocess.run(["rustc", "-V"], check=False, capture_output=True, text=True)
    cargo = subprocess.run(["cargo", "-V"], check=False, capture_output=True, text=True)
    rustc_line = (rustc.stdout if rustc.returncode == 0 else rustc.stderr).strip()
    cargo_line = (cargo.stdout if cargo.returncode == 0 else cargo.stderr).strip()
    if rustc_line != "rustc 1.88.0 (6b00bc388 2025-06-23)":
        print("panel requires rustc 1.88.0", file=sys.stderr)
        return 2
    if not cargo_line.startswith("cargo 1.88.0 "):
        print("panel requires cargo 1.88.0", file=sys.stderr)
        return 2
    stamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%dT%H:%M:%S%z")
    t_note = "\n".join(
        [
            "Copyright (c) anneal developers",
            "",
            "T is the clock time at the start of this panel.",
            stamp + ".",
            "The rustc version line follows.",
            rustc_line + ".",
            "The cargo version line follows.",
            cargo_line + ".",
            "The wall-clock cap is 600 seconds for each process.",
            "That cap was stated before this run.",
            "The panel logs cover seeds 6000, 6001, 6002, 6003, 6004, and 6005 only.",
            "",
        ]
    )
    (Path(__file__).resolve().parent / "T.txt").write_text(t_note + "\n", encoding="utf-8")
    handle = acquire_scored_run(Path(args.lock_file))
    session_start = time.monotonic()
    try:
        fields = provenance()
        record_serial_boundary(out, "start")
        rows = out / "rows.tsv"
        rows.write_text("\t".join(COLUMNS) + "\n", encoding="utf-8")
        for index, job in enumerate(jobs, start=1):
            stem = f"{index:02d}-lj{job['n']}-{job['seed']}"
            prefix = rel / stem
            log_path = out / f"{stem}.log"
            err_path = out / f"{stem}.stderr"
            span_path = out / f"{stem}.span"
            started = time.monotonic()
            proc = subprocess.run(
                [
                    os.path.relpath(binary, root),
                    "lj",
                    str(job["n"]),
                    "0",
                    str(job["budget"]),
                    str(job["seed"]),
                    "recommended",
                    f"{job['stop']:.10f}",
                    prefix.as_posix(),
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=WALL_CLOCK_CAP_SECONDS,
            )
            finished = time.monotonic()
            log_path.write_text(proc.stdout, encoding="utf-8")
            err_path.write_text(proc.stderr, encoding="utf-8")
            span_path.write_text(
                f"{started - session_start:.6f} {finished - session_start:.6f}\n",
                encoding="utf-8",
            )
            if proc.returncode != 0:
                print(f"{stem} exited {proc.returncode}", file=sys.stderr)
                return proc.returncode or 1
            parsed = parse_h2h(proc.stdout)
            energy = float(parsed["lowest"])
            hit = energy <= job["stop"]
            row = dict(fields)
            row.update(
                {
                    "seed": str(job["seed"]),
                    "code": f"default-lj{job['n']}",
                    "hit": "true" if hit else "false",
                    "energy": f"{energy:.10f}",
                    "calls": parsed.get("calls", ""),
                    "wall_seconds": parsed.get("wall_seconds", ""),
                }
            )
            with rows.open("a", encoding="utf-8") as handle_rows:
                handle_rows.write("\t".join(row.get(column, "") for column in COLUMNS) + "\n")
            print(
                f"{stem} hit={row['hit']} energy={row['energy']} calls={row['calls']} wall={row['wall_seconds']}",
                flush=True,
            )
        record_serial_boundary(out, "end")
        elapsed = time.monotonic() - session_start
        (out / "session.txt").write_text(
            f"elapsed_seconds {elapsed:.6f}\nruns {len(jobs)}\n",
            encoding="utf-8",
        )
    finally:
        release_scored_run(handle)
    return 0


def cmd_hold(args: argparse.Namespace) -> int:
    handle = acquire_scored_run(Path(args.lock_file))
    try:
        print("holding", flush=True)
        time.sleep(args.seconds)
    finally:
        release_scored_run(handle)
    return 0


def cmd_emit_row(args: argparse.Namespace) -> int:
    handle = acquire_scored_run(Path(args.lock_file))
    try:
        fields = provenance()
        fields.update(
            {
                "seed": "unscored",
                "code": "schema",
                "hit": "unscored",
                "energy": "",
                "calls": "",
            }
        )
        write_row(Path(args.output), fields)
    finally:
        release_scored_run(handle)
    return 0


def held_out_seeds() -> list[int]:
    return list(range(HELD_OUT_LO, HELD_OUT_HI + 1))


def _require_tool(path: Path, label: str) -> Path:
    if not path.is_file():
        print(f"missing {label}", file=sys.stderr)
        raise SystemExit(2)
    return path


def _run_capped(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> tuple[int, str, str, float]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=WALL_CLOCK_CAP_SECONDS,
        )
        return proc.returncode, _public(proc.stdout), _public(proc.stderr), time.monotonic() - started
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        err = err + "\nwall-clock cap reached\n"
        return 124, _public(out), _public(err), float(WALL_CLOCK_CAP_SECONDS)


def _public(text: str) -> str:
    return re.sub(r"(^|\s)(/[^\s]+)", r"\1PATH", text)


def _parse_kv(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        fields[key.strip()] = value.strip()
    return fields


def _requench(binary: Path, xyz: Path, cwd: Path) -> tuple[str, str]:
    code, out, err, _wall = _run_capped([str(binary), str(xyz)], cwd, None)
    (xyz.parent / (xyz.stem + ".requench.err")).write_text(err, encoding="utf-8")
    if code != 0:
        return "", ""
    fields = _parse_kv(out)
    return fields.get("energy", ""), fields.get("calls", "")


def _xyz_from_gmin_lowest(lowest: Path, xyz: Path, n_atoms: int) -> bool:
    if not lowest.is_file():
        return False
    lines = [line for line in lowest.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    coords = []
    for line in lines:
        parts = line.split()
        nums = []
        for part in parts:
            try:
                nums.append(float(part))
            except ValueError:
                continue
        if len(nums) >= 3:
            coords.append(nums[-3:])
    if len(coords) < n_atoms:
        return False
    body = [f"{n_atoms}", "gmin lowest"]
    for x, y, z in coords[:n_atoms]:
        body.append(f"X {x:.10f} {y:.10f} {z:.10f}")
    xyz.write_text("\n".join(body) + "\n", encoding="utf-8")
    return True


def _write_heldout_t(stamp: str, rustc_line: str, cargo_line: str) -> None:
    note = "\n".join(
        [
            "Copyright (c) anneal developers",
            "",
            "T is the clock time at the start of the held-out run.",
            stamp + ".",
            "Arm D is the lattice v3.2 tip.",
            D_V3_TIP + ".",
            "Arm B stays UNFILLED.",
            "The rustc version line follows.",
            rustc_line + ".",
            "The cargo version line follows.",
            cargo_line + ".",
            "The wall-clock cap is 600 seconds for each process.",
            "That cap was stated before this run.",
            "The seeds are 5000 through 5029.",
            "The call budget is 4000000.",
            "The tie cut is lowest-index.",
            "",
        ]
    )
    (script_path().parent / "T-heldout.txt").write_text(note + "\n", encoding="utf-8")


def cmd_heldout(args: argparse.Namespace) -> int:
    if PRESET_ID != "UNFILLED":
        print("arm B was filled; this command leaves it UNFILLED", file=sys.stderr)
        return 2
    if not D_V3_TIP.startswith("28ece242"):
        print("arm D tip is not the lattice v3.2 tip", file=sys.stderr)
        return 2
    root = repo_root()
    dls = _require_tool(Path(args.dls), "lj_dls")
    requench = _require_tool(Path(args.requench), "lj_requench")
    gmin = Path(args.gmin) if args.gmin else None
    if gmin is not None:
        gmin = _require_tool(gmin, "GMIN")
    coords_bin = Path(args.coords) if args.coords else None
    if gmin is not None:
        coords_bin = _require_tool(Path(args.coords), "coords helper")
    rel = Path(args.output_dir)
    if rel.is_absolute():
        print("output dir must be relative", file=sys.stderr)
        return 2
    out = root / rel
    out.mkdir(parents=True, exist_ok=True)
    rustc = subprocess.run(["rustc", "-V"], check=False, capture_output=True, text=True)
    cargo = subprocess.run(["cargo", "-V"], check=False, capture_output=True, text=True)
    rustc_line = (rustc.stdout if rustc.returncode == 0 else rustc.stderr).strip()
    cargo_line = (cargo.stdout if cargo.returncode == 0 else cargo.stderr).strip()
    if rustc_line != "rustc 1.88.0 (6b00bc388 2025-06-23)":
        print("held-out run requires rustc 1.88.0", file=sys.stderr)
        return 2
    if not cargo_line.startswith("cargo 1.88.0 "):
        print("held-out run requires cargo 1.88.0", file=sys.stderr)
        return 2
    keywords = (script_path().parent / "gmin_best.keywords").read_text(encoding="utf-8").rstrip() + "\n\n"
    stamp = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_heldout_t(stamp, rustc_line, cargo_line)
    handle = acquire_scored_run(Path(args.lock_file))
    session_start = time.monotonic()
    try:
        fields = provenance()
        record_serial_boundary(out, "start")
        rows = out / "rows.tsv"
        rows.write_text("\t".join(COLUMNS) + "\n", encoding="utf-8")
        env_base = os.environ.copy()
        env_base["OMP_NUM_THREADS"] = "1"
        env_base["RAYON_NUM_THREADS"] = "1"
        arms = ["dls"]
        if gmin is not None:
            arms.append("gmin")
        for seed in held_out_seeds():
            for code in arms:
                stem = f"{code}-{seed}"
                print(f"start {stem}", flush=True)
                row = dict(fields)
                row.update({"seed": str(seed), "code": code, "hit": "false", "energy": "", "calls": "", "wall_seconds": ""})
                if code == "dls":
                    prefix = out / "dls" / str(seed)
                    prefix.parent.mkdir(parents=True, exist_ok=True)
                    code_rc, stdout, stderr, wall = _run_capped(
                        [
                            "taskset",
                            "-c",
                            "0",
                            str(dls),
                            str(LJ104_N),
                            str(LJ104_BUDGET),
                            str(seed),
                            DLS_TIE,
                            DLS_STOP,
                            str(prefix),
                        ],
                        root,
                        env_base,
                    )
                    (out / "dls" / f"{seed}.log").write_text(stdout, encoding="utf-8")
                    (out / "dls" / f"{seed}.err").write_text(stderr, encoding="utf-8")
                    parsed = _parse_kv(stdout)
                    row["calls"] = parsed.get("charged", "")
                    row["wall_seconds"] = parsed.get("wall_seconds", f"{wall:.3f}")
                    xyz = Path(str(prefix) + ".xyz")
                    energy, rq_calls = ("", "")
                    if xyz.is_file():
                        energy, rq_calls = _requench(requench, xyz, root)
                        (out / "requench").mkdir(parents=True, exist_ok=True)
                        (out / "requench" / f"dls-{seed}.txt").write_text(
                            f"energy: {energy}\ncalls: {rq_calls}\n",
                            encoding="utf-8",
                        )
                    if energy:
                        row["energy"] = energy
                        row["hit"] = "true" if float(energy) <= HIT_LINE else "false"
                    if code_rc not in (0,):
                        (out / "dls" / f"{seed}.status").write_text(f"exit {code_rc}\n", encoding="utf-8")
                else:
                    work = out / "gmin" / str(seed)
                    work.mkdir(parents=True, exist_ok=True)
                    coords = work / "coords"
                    helper = _run_capped(
                        [str(coords_bin), "coords", "lj104", str(seed), str(coords)],
                        root,
                        env_base,
                    )
                    (work / "coords.err").write_text(helper[2], encoding="utf-8")
                    (work / "data").write_text(keywords + f"RANSEED {seed}\n", encoding="utf-8")
                    gmin_env = dict(env_base)
                    gmin_env.update(
                        {
                            "H2H_SYSTEM": "lj104",
                            "H2H_SEED": str(seed),
                            "H2H_BUDGET": str(LJ104_BUDGET),
                            "H2H_TARGET": f"{E_KNOWN_LJ104:.6f}",
                            "H2H_TOL": "1e-6",
                            "H2H_ROW": "row.txt",
                        }
                    )
                    code_rc, stdout, stderr, wall = _run_capped(
                        ["taskset", "-c", "0", str(gmin)],
                        work,
                        gmin_env,
                    )
                    (work / "stdout.txt").write_text(stdout, encoding="utf-8")
                    (work / "stderr.txt").write_text(stderr, encoding="utf-8")
                    charged = ""
                    for line in stdout.splitlines():
                        if line.startswith("h2h_row "):
                            parts = {}
                            for item in line.split()[1:]:
                                if "=" in item:
                                    key, value = item.split("=", 1)
                                    parts[key] = value
                            charged = parts.get("charged", "")
                            row["wall_seconds"] = parts.get("wall_s", f"{wall:.3f}")
                    row["calls"] = charged
                    lowest = work / "lowest"
                    xyz = work / "lowest.xyz"
                    energy = ""
                    if _xyz_from_gmin_lowest(lowest, xyz, LJ104_N):
                        energy, rq_calls = _requench(requench, xyz, root)
                        (out / "requench").mkdir(parents=True, exist_ok=True)
                        (out / "requench" / f"gmin-{seed}.txt").write_text(
                            f"energy: {energy}\ncalls: {rq_calls}\n",
                            encoding="utf-8",
                        )
                    if energy:
                        row["energy"] = energy
                        row["hit"] = "true" if float(energy) <= HIT_LINE else "false"
                    if code_rc not in (0,):
                        (work / "status.txt").write_text(f"exit {code_rc}\n", encoding="utf-8")
                with rows.open("a", encoding="utf-8") as handle_rows:
                    handle_rows.write("\t".join(row.get(column, "") for column in COLUMNS) + "\n")
                print(
                    f"{stem} hit={row['hit']} energy={row['energy']} calls={row['calls']} wall={row['wall_seconds']}",
                    flush=True,
                )
        record_serial_boundary(out, "end")
        elapsed = time.monotonic() - session_start
        (out / "session.txt").write_text(
            f"elapsed_seconds {elapsed:.6f}\nruns {len(held_out_seeds()) * 2}\n",
            encoding="utf-8",
        )
    finally:
        release_scored_run(handle)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    code = args.code.strip().lower()
    if code in {"b", "candidate", "preset"} or PRESET_ID == "UNFILLED":
        print("arm B preset id is UNFILLED; no scored search is started", file=sys.stderr)
        return 3
    print("use the heldout command for a scored search", file=sys.stderr)
    return 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", default=str(default_lock_file()))
    sub = parser.add_subparsers(dest="command", required=True)

    hold = sub.add_parser("hold", help="hold the scored-run lock")
    hold.add_argument("--seconds", type=float, required=True)
    hold.set_defaults(func=cmd_hold)

    emit = sub.add_parser("emit-row", help="write one provenance row")
    emit.add_argument("--output", required=True)
    emit.set_defaults(func=cmd_emit_row)

    run = sub.add_parser("run", help="refused while arm B is UNFILLED")
    run.add_argument("--code", required=True)
    run.add_argument("--seeds", required=True)
    run.add_argument("--receipt-dir", default="lj104-sota-receipts")
    run.set_defaults(func=cmd_run)

    held = sub.add_parser("heldout", help="score arm D and GMIN on seeds 5000-5029")
    held.add_argument("--output-dir", required=True)
    held.add_argument("--dls", required=True)
    held.add_argument("--requench", required=True)
    held.add_argument("--gmin", default="")
    held.add_argument("--coords", default="")
    held.set_defaults(func=cmd_heldout)

    panel = sub.add_parser("panel", help="run the default no-regression panel serially")
    panel.add_argument("--output-dir", required=True)
    panel.add_argument("--binary", default="target/release/examples/h2h_seed")
    panel.set_defaults(func=cmd_panel)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
