#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Same-runner no-regression gate for the PR #160 revert.

This is the previously committed three-arm gate, adapted to the fixed
develop and revert trees. TIP4P9 through TIP4P21 use seeds 2000--2023.
The recorded Lennard-Jones and Morse cells use seeds 2000--2011.

Every process is fresh, serial, pinned to core 0, and receives a clean
single-thread environment. The six arm permutations rotate by seed. There
are no retries, warm-up exclusions, or result-dependent substitutions.

Hits are derived from recorded energies at reference plus 0.001. A miss
has infinite time to hit. For every cell, B must have at least as many hits
as A. The B/A median outside-time-per-call ratio must not exceed 1.10 and
is judged beside the A2/A same-tree band. Fisher, Mann-Whitney, and a
deterministic paired bootstrap interval are reported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

PANEL_SEEDS = tuple(range(2000, 2012))
TIP4P_SEEDS = tuple(range(2000, 2024))
HIT_TOL = 1.0e-3
TIP4P_BUDGET = 2_000_000
TIP4P_SIZES = tuple(range(9, 22))
ARMS = ("A", "A2", "B")
ARM_COMMITS = {
    "A": ("61fa67cf380e46876b569da45e7cc091647a775b", "cff796c507a8fa1d0bfb28888c3570b61988055e"),
    "A2": ("61fa67cf380e46876b569da45e7cc091647a775b", "cff796c507a8fa1d0bfb28888c3570b61988055e"),
    "B": ("9fcc707237e62692d3aabe0ec6ba5b5d37008522", "0e469d0f9e49deb0083332c72316d29918626350"),
}
ARM_SLUG = {"A": "a", "A2": "a2", "B": "b"}
ORDERS = (
    ("A", "B", "A2"),
    ("A2", "B", "A"),
    ("B", "A2", "A"),
    ("A", "A2", "B"),
    ("A2", "A", "B"),
    ("B", "A", "A2"),
)

# Verdict thresholds, fixed before the first run.
MAX_OUTSIDE_RATIO = 1.10
MIN_HIT_DELTA = 0

# scripts/gmin_panel/run_panel.py: PAPER_BUDGETS and SYSTEMS.
PAPER_BUDGETS = {
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj102": 4_000_000,
    "lj103": 4_000_000,
    "lj104": 4_000_000,
    "lj150": 4_000_000,
    "lj75-seed": 4_000_000,
    "m55r6": 1_000_000,
    "m55r14": 1_000_000,
    "m80r6": 4_000_000,
    "m80r14": 4_000_000,
}
PAIR_REFERENCES = {
    "lj38": -173.928427,
    "lj55": -279.248470,
    "lj75": -397.492331,
    "lj98": -543.665361,
    "lj102": -569.363652,
    "lj103": -575.766131,
    "lj104": -582.086642,
    "lj150": -893.310258,
    "lj75-seed": -397.492331,
    "m55r6": -250.286609,
    "m55r14": -220.646208,
    "m80r6": -378.333471,
    "m80r14": -340.811371,
}
PAIR_SYSTEMS = tuple(PAPER_BUDGETS)

ROW_FIELDS = (
    "cell", "arm", "seed", "slot", "tree", "exit_code", "energy", "hit",
    "calls", "charged", "wall_s", "inside_s", "outside_s", "per_call_s",
    "core", "affinity", "rss_anon_peak_kb", "ru_maxrss_kb", "energy_bits",
    "final_sha256", "budget", "reference", "machine_id", "binary_sha256",
)
START_FIELDS = ("system", "seed", "arm", "source", "start_sha256")
CORE = "0"
THREAD_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "GOTO_NUM_THREADS",
    "OMP_THREAD_LIMIT", "MKL_DOMAIN_NUM_THREADS", "JULIA_NUM_THREADS",
    "TBB_NUM_THREADS", "POLARS_MAX_THREADS", "OPENMP_NUM_THREADS",
)


def tip4p_references() -> dict[int, float]:
    with (ROOT / "data" / "tip4p_ccd.csv").open() as handle:
        return {int(r["n"]): float(r["energy"]) for r in csv.DictReader(handle)}


def cells() -> list[tuple[str, int, float, tuple[int, ...]]]:
    """(system, budget, reference, seeds) in run order."""
    refs = tip4p_references()
    out = [
        (f"tip4p{n}", TIP4P_BUDGET, refs[n], TIP4P_SEEDS)
        for n in TIP4P_SIZES
    ]
    out += [
        (s, PAPER_BUDGETS[s], PAIR_REFERENCES[s], PANEL_SEEDS)
        for s in PAIR_SYSTEMS
    ]
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def machine_id() -> str:
    return hashlib.sha256(Path("/etc/machine-id").read_bytes()).hexdigest()


def bits(text: str) -> str:
    return struct.pack(">d", float(text)).hex()


def child_env() -> dict[str, str]:
    env = {
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    }
    for name in THREAD_VARS:
        env[name] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    return env


def driver(system: str) -> str:
    if system.startswith("tip4p"):
        return "h2h_seed"
    if system == "lj75-seed":
        return "h2h_seed"
    if system.startswith("lj"):
        return "h2h_gmin_proof"
    return "h2h_morse"


class Arm:
    def __init__(self, name: str, spec: str) -> None:
        bin_dir, src_dir = spec.split(":", 1)
        self.name = name
        self.bin = Path(bin_dir).resolve()
        self.src = Path(src_dir).resolve()
        self.sha = {d: sha256_file(self.bin / d) for d in ("h2h_seed", "h2h_gmin_proof", "h2h_morse")}
        self.driver_sha = {
            d: sha256_file(self.src / "examples" / f"{d}.rs")
            for d in ("h2h_seed", "h2h_gmin_proof", "h2h_morse")
        }

    def state(self) -> tuple[str, str, int]:
        def git(*args: str) -> str:
            return subprocess.run(["git", "-C", str(self.src), *args], check=True,
                                  capture_output=True, text=True).stdout.strip()
        dirty = len([l for l in git("status", "--porcelain").splitlines() if l])
        return git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}"), dirty


def parse_arms(specs: list[str]) -> dict[str, Arm]:
    arms = {}
    for spec in specs:
        name, rest = spec.split("=", 1)
        if name not in ARMS:
            raise SystemExit(f"unknown arm {name}")
        arms[name] = Arm(name, rest)
    if set(arms) != set(ARMS):
        raise SystemExit(f"need every arm of {ARMS}")
    return arms


def session_line(path: Path, *fields: object) -> None:
    with path.open("a", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow([str(f) for f in fields])


def check_arms(arms: dict[str, Arm], session: Path, when: str) -> None:
    for source in ("h2h_seed", "h2h_gmin_proof", "h2h_morse"):
        digests = {arms[name].driver_sha[source] for name in ARMS}
        if len(digests) != 1:
            raise SystemExit(f"{source} source differs across arms")
    for name in ARMS:
        arm = arms[name]
        head, tree, dirty = arm.state()
        want_head, want_tree = ARM_COMMITS[name]
        ok = head == want_head and tree == want_tree and dirty == 0
        session_line(session, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), when,
                     name, head, tree, f"dirty={dirty}", "ok" if ok else "MISMATCH")
        for d, sha in arm.sha.items():
            now = sha256_file(arm.bin / d)
            session_line(session, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), when,
                         name, d, now, "sha256", "ok" if now == sha else "MISMATCH")
            if now != sha:
                raise SystemExit(f"{name} {d} changed")
        for d, sha in arm.driver_sha.items():
            session_line(session, when, name, d, sha, "driver_source", "ok")
        if not ok:
            raise SystemExit(f"{name} checkout is not {want_head} / {want_tree} clean")


def command(arm: Arm, system: str, budget: int, ref: float, seed: int, prefix: Path,
            starts: Path) -> list[str]:
    pin = ["taskset", "-c", CORE]
    if system.startswith("tip4p"):
        n = system.removeprefix("tip4p")
        return pin + [str(arm.bin / "h2h_seed"), "tip4p", n, "0", str(budget), str(seed),
                      "recommended", repr(ref + HIT_TOL), str(prefix)]
    if system == "lj75-seed":
        return pin + [
            str(arm.bin / "h2h_seed"), "lj", "75", "0", str(budget),
            str(seed), "recommended", repr(ref + HIT_TOL), str(prefix),
        ]
    if system.startswith("lj"):
        return pin + [str(arm.bin / "h2h_gmin_proof"), "run", system, str(seed),
                      f"{prefix}.start"]
    return pin + [str(arm.bin / "h2h_morse"), "run", system, str(seed),
                  str(starts / f"{system}-s{seed}.start")]


def parse(system: str, text: str) -> dict[str, str]:
    if system.startswith("tip4p") or system == "lj75-seed":
        line = next(l for l in text.splitlines() if l.startswith("h2h_row "))
        row = dict(tok.split("=", 1) for tok in line.split()[1:])
        calls = int(re.search(r"^calls: (\d+)$", text, re.M).group(1))
        return {"calls_raw": str(calls), "evals": row["evals_to_hit"], "charged": row["charged"],
                "wall": row["wall_s"], "inside": row["pot_s"], "outside": row["outside_s"],
                "best": row["best"], "budget": row["budget"]}
    lines = text.splitlines()
    idx = next(i for i, l in enumerate(lines) if l.startswith("system,code,seed,"))
    row = dict(zip(lines[idx].split(","), lines[idx + 1].split(","), strict=True))
    wall, inside = float(row["wall_s"]), float(row["pot_s"])
    outside = row.get("outside_s") or f"{max(wall - inside, 0.0):.6f}"
    return {"evals": row["evals_to_hit"], "charged": row["charged"], "wall": row["wall_s"],
            "inside": row["pot_s"], "outside": outside, "best": row["best"],
            "budget": row["budget"]}


def write_starts(arms: dict[str, Arm], starts: Path, rows: Path) -> None:
    starts.mkdir(parents=True, exist_ok=True)
    base = arms["A"]
    for system in PAIR_SYSTEMS:
        if system == "lj75-seed":
            continue
        exe = base.bin / driver(system)
        for seed in PANEL_SEEDS:
            path = starts / f"{system}-s{seed}.start"
            subprocess.run(["taskset", "-c", CORE, str(exe), "coords", system, str(seed), str(path)],
                           check=True, env=child_env())
            append(rows, START_FIELDS, dict(system=system, seed=seed, arm="A",
                                            source=f"{driver(system)} coords",
                                            start_sha256=sha256_file(path)))


def append(path: Path, fields: tuple[str, ...], row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def process_status(pid: int) -> tuple[str, int]:
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except FileNotFoundError:
        return "", 0
    affinity = ""
    rss_anon = 0
    for line in lines:
        if line.startswith("Cpus_allowed_list"):
            affinity = line.split(":", 1)[1].strip()
        elif line.startswith("RssAnon:"):
            rss_anon = int(line.split()[1])
    return affinity, rss_anon


def run_measured(
    command_line: list[str],
    cwd: Path,
    env: dict[str, str],
    out_path: Path,
    err_path: Path,
) -> tuple[int, str, str, str, int, int, float]:
    start = time.monotonic()
    pid = os.fork()
    if pid == 0:
        try:
            os.chdir(cwd)
            out_fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            err_fd = os.open(err_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.dup2(out_fd, 1)
            os.dup2(err_fd, 2)
            os.close(out_fd)
            os.close(err_fd)
            os.execvpe(command_line[0], command_line, env)
        except BaseException as err:  # noqa: BLE001
            os.write(2, f"exec failed: {err}\n".encode())
            os._exit(127)
    affinity = ""
    peak = 0
    while True:
        current, rss_anon = process_status(pid)
        if current == CORE:
            affinity = current
        peak = max(peak, rss_anon)
        waited, status, usage = os.wait4(pid, os.WNOHANG)
        if waited == pid:
            break
        time.sleep(0.002)
    elapsed = time.monotonic() - start
    stdout = out_path.read_text(errors="replace")
    stderr = err_path.read_text(errors="replace")
    return (
        os.waitstatus_to_exitcode(status),
        stdout,
        stderr,
        affinity,
        peak,
        int(usage.ru_maxrss),
        elapsed,
    )


def cmd_run(args: argparse.Namespace) -> None:
    if os.sched_getaffinity(0) != {0}:
        os.sched_setaffinity(0, {0})
    arms = parse_arms(args.arm)
    work = args.work.resolve()
    logs, starts = work / "logs", work / "starts"
    logs.mkdir(parents=True, exist_ok=True)
    out = args.out.resolve()
    rows = out / "rows.csv"
    start_rows = out / "starts.csv"
    session = out / "sessions.csv"
    if rows.exists():
        raise SystemExit("rows.csv exists: one session only, no reruns")
    mid = machine_id()
    stamp = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())  # noqa: E731
    session_line(session, "utc", "event", "arm_or_key", "a", "b", "c", "d")
    session_line(session, stamp(), "machine", mid, f"cpus={os.cpu_count()}",
                 f"core={CORE}", "", "")
    session_line(session, stamp(), "env", "clean", "threads=1",
                 "dynamic=false", "", "")
    check_arms(arms, session, "before_first_run")
    for name in ARMS:
        for d, sha in arms[name].sha.items():
            session_line(session, stamp(), "binary", name, d, sha, "", "")
    write_starts(arms, starts, start_rows)
    session_line(session, stamp(), "starts_written", "A", "panel coords",
                 (len(PAIR_SYSTEMS) - 1) * len(PANEL_SEEDS), "", "")
    env = child_env()
    slot = 0
    for system, budget, ref, seeds in cells():
        session_line(session, stamp(), "cell_start", system, budget, repr(ref), "", "")
        for k, seed in enumerate(seeds):
            for name in ORDERS[k % len(ORDERS)]:
                arm = arms[name]
                slot += 1
                tag = f"{system}-s{seed}-{ARM_SLUG[name]}"
                prefix = logs / tag
                cmd = command(arm, system, budget, ref, seed, prefix, starts)
                out_path = Path(f"{prefix}.out")
                err_path = Path(f"{prefix}.err")
                rc, stdout, stderr, affinity, rss_anon, ru_maxrss, elapsed = (
                    run_measured(cmd, logs, env, out_path, err_path)
                )
                Path(f"{prefix}.rc").write_text(f"{rc}\n")
                row = dict(
                    cell=system, arm=name, seed=seed, slot=slot,
                    tree=ARM_COMMITS[name][1], exit_code=rc, core=CORE,
                    affinity=affinity, rss_anon_peak_kb=rss_anon,
                    ru_maxrss_kb=ru_maxrss, budget=budget, reference=repr(ref),
                    machine_id=mid, binary_sha256=arm.sha[driver(system)],
                    final_sha256="",
                )
                try:
                    if rc != 0:
                        raise RuntimeError(f"exit {rc}")
                    if affinity != CORE:
                        raise RuntimeError(f"affinity {affinity!r}")
                    got = parse(system, stdout)
                    if int(got["budget"]) != budget:
                        raise RuntimeError(f"driver budget {got['budget']}")
                    best = got["best"]
                    hit = float(best) <= ref + HIT_TOL
                    calls = int(got["evals"]) if hit else budget
                    per_call = float(got["outside"]) / int(got["charged"])
                    final_path = Path(f"{prefix}.xyz")
                    row.update(
                        hit=int(hit), calls=calls, charged=got["charged"],
                        wall_s=got["wall"], inside_s=got["inside"],
                        outside_s=got["outside"], per_call_s=f"{per_call:.12g}",
                        energy=best, energy_bits=bits(best),
                        final_sha256=(
                            sha256_file(final_path) if final_path.exists() else ""
                        ),
                    )
                except Exception as err:  # noqa: BLE001
                    row.update(
                        hit=0, calls=budget, charged="", wall_s=f"{elapsed:.6f}",
                        inside_s="", outside_s="", per_call_s="", energy="",
                        energy_bits="",
                    )
                    session_line(
                        session, stamp(), "failed", name, system, seed,
                        str(err), stderr[-500:],
                    )
                append(rows, ROW_FIELDS, row)
                start_file = (
                    Path(f"{prefix}.start.gmin")
                    if system.startswith("tip4p") or system == "lj75-seed"
                    else Path(f"{prefix}.start")
                    if system.startswith("lj")
                    else starts / f"{system}-s{seed}.start"
                )
                append(start_rows, START_FIELDS, dict(
                    system=system, seed=seed, arm=name, source=f"{driver(system)} run",
                    start_sha256=sha256_file(start_file) if start_file.exists() else ""))
        session_line(session, stamp(), "cell_end", system, "", "", "", "")
    check_arms(arms, session, "after_last_run")
    session_line(session, stamp(), "done", slot, "", "", "", "")


def cmd_hosts(args: argparse.Namespace) -> None:
    cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                if l.startswith("model name")), "")
    path = args.out.resolve() / "hosts.csv"
    with path.open("w", newline="") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(["machine_id", "cpu_model", "logical_cpus", "kernel", "arch"])
        uname = os.uname()
        w.writerow([machine_id(), cpu, os.cpu_count(), uname.release, uname.machine])


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--arm", action="append", required=True, metavar="NAME=BIN_DIR:SRC_DIR")
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--out", type=Path, default=HERE)
    hosts = sub.add_parser("hosts")
    hosts.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args(argv[1:])
    {"run": cmd_run, "hosts": cmd_hosts}[args.command](args)


if __name__ == "__main__":
    main(sys.argv)
