#!/usr/bin/env python3
"""Serial same-session runner for the r-next8 gate.

Paths, cores, and library locations come from the command line. The
scientific choices are those in PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BASE_REV = "bd73f89d2879eeff2fb1323ce2eec8ac13d70ce3"
BASE_TREE = "fb5ae7cc254401140f67bfc9188078fe07f110ca"
PAYLOAD_TREE = "210f9611263fa07119ef78e3ccbc23cd3c2c0f4d"
SEEDS = tuple(range(2000, 2012))
GFN2_WALL_S = 1500.0
GFN2_BUDGET = 100_000
FORCE_TOL_EV_A = 1.0e-3
HIT_TOL = 1.0e-3
BOND_TOL = 1.25
RADII = {6: 0.76, 14: 1.11}
HARTREE_TO_EV = 27.21138505
BOHR_TO_A = 0.529177210903
ORDERS = (
    ("A", "B", "B2", "A2"),
    ("A2", "B2", "B", "A"),
)
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENMP_NUM_THREADS",
)
CELLS = (
    ("gfn2-si10", "gfn2", "si10", 14, GFN2_BUDGET, -460.68373809),
    ("sw10", "covalent", "sw10", 14, 100_000, -13.79658026),
    ("sw13", "covalent", "sw13", 14, 100_000, -18.50399759),
    ("tc10", "covalent", "tc10", 6, 100_000, -55.08979464),
    ("sw20", "covalent", "sw20", 14, 400_000, -30.18725174),
    ("tc20", "covalent", "tc20", 6, 400_000, -120.69889105),
    ("gfn2-c10", "gfn2", "c10", 6, GFN2_BUDGET, -571.32273783),
    ("gfn2-si13", "gfn2", "si13", 14, GFN2_BUDGET, -599.34546694),
)
ROW_FIELDS = (
    "cell",
    "phase",
    "kind",
    "arm",
    "position",
    "seed",
    "budget",
    "reference",
    "revision",
    "tree",
    "binary_sha256",
    "hit",
    "physical_hit",
    "calls",
    "calls_to_hit",
    "outside_s",
    "wall_s",
    "rss_anon_peak_kb",
    "trajectory_sha256",
    "energy_sha256",
    "best",
    "censored",
    "nn_min_A",
    "nn_max_A",
    "detached",
    "connected",
    "sp_energy_eV",
    "scc_converged",
    "max_force_eV_A",
    "force_ok",
    "scc_rejected",
    "geometry_rejected",
    "engine_calls",
    "ledger_calls",
)
RSS_RE = re.compile(r"^rss_anon_self peak_kb=(\d+) end_kb=(\d+)$", re.M)
CUH2_RE = re.compile(r"^cuh2_engine_calls (\d+)$", re.M)
ARM_RE = re.compile(r"^arm \S+ draws \d+ accepts \d+ best -?\d")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def blank_clocks(fields: list[str], indexes: tuple[int, ...]) -> str:
    copied = list(fields)
    for index in indexes:
        if index < len(copied):
            copied[index] = "*"
    return ",".join(copied)


@dataclass
class Arm:
    name: str
    source: Path
    covalent: Path
    gfn2: Path
    slab: Path
    revision: str
    tree: str
    hashes: dict[str, str]

    def binary(self, kind: str) -> Path:
        return {"covalent": self.covalent, "gfn2": self.gfn2, "slab": self.slab}[kind]


def load_arms(specs: list[str]) -> dict[str, Arm]:
    if len(specs) % 5 != 0:
        raise SystemExit("--arm expects name source covalent gfn2 slab")
    arms: dict[str, Arm] = {}
    for offset in range(0, len(specs), 5):
        name, source_text, covalent_text, gfn2_text, slab_text = specs[offset : offset + 5]
        source = Path(source_text).resolve()
        tree = git(source, "write-tree")
        revision = git(source, "rev-parse", "HEAD")
        expected = PAYLOAD_TREE if name.startswith("B") else BASE_TREE
        if tree != expected:
            raise SystemExit(f"{name} tree {tree} is not {expected}")
        if name.startswith("A"):
            if revision != BASE_REV:
                raise SystemExit(f"{name} revision {revision} is not {BASE_REV}")
            dirty = git(source, "status", "--porcelain")
            if dirty:
                raise SystemExit(f"{name} source is not clean")
        covalent = Path(covalent_text).resolve()
        gfn2 = Path(gfn2_text).resolve()
        slab = Path(slab_text).resolve()
        arms[name] = Arm(
            name,
            source,
            covalent,
            gfn2,
            slab,
            BASE_REV,
            tree,
            {
                "covalent": sha256_file(covalent),
                "gfn2": sha256_file(gfn2),
                "slab": sha256_file(slab),
            },
        )
    missing = {"A", "A2", "B", "B2"} - set(arms)
    if missing:
        raise SystemExit(f"missing arms {sorted(missing)}")
    return arms


def sw_sigma(library: Path) -> float:
    lib = ctypes.CDLL(str(library))
    lib.rgpot_covalent_sw_units.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    sigma = ctypes.c_double()
    epsilon = ctypes.c_double()
    lib.rgpot_covalent_sw_units(ctypes.byref(sigma), ctypes.byref(epsilon))
    if sigma.value <= 0.0:
        raise SystemExit("Stillinger-Weber sigma is not positive")
    return float(sigma.value)


def read_xyz_or_raw(path: Path) -> list[tuple[float, float, float]]:
    lines = path.read_text().splitlines()
    coords: list[tuple[float, float, float]] = []
    numeric = []
    for line in lines:
        fields = line.split()
        if len(fields) == 4 and not _is_float(fields[0]):
            numeric.append(fields[1:])
        elif len(fields) >= 3 and all(_is_float(item) for item in fields[:3]):
            if len(fields) == 3 or _is_float(fields[0]):
                numeric.append(fields[:3] if _is_float(fields[0]) else fields[1:4])
    for item in numeric:
        coords.append((float(item[0]), float(item[1]), float(item[2])))
    return coords


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def geometry_report(coords: list[tuple[float, float, float]], atomic_number: int) -> dict[str, str]:
    if len(coords) < 2:
        return {
            "nn_min_A": "",
            "nn_max_A": "",
            "detached": "true",
            "connected": "false",
        }
    radius = RADII[atomic_number]
    cut = BOND_TOL * (radius + radius)
    parent = list(range(len(coords)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    nearest: list[float] = []
    for left, xyz in enumerate(coords):
        best = float("inf")
        for right, other in enumerate(coords):
            if left == right:
                continue
            distance = sum((a - b) ** 2 for a, b in zip(xyz, other)) ** 0.5
            best = min(best, distance)
            if distance < cut:
                parent[find(right)] = find(left)
        nearest.append(best)
    components = len({find(index) for index in range(len(coords))})
    connected = components == 1
    return {
        "nn_min_A": f"{min(nearest):.6f}",
        "nn_max_A": f"{max(nearest):.6f}",
        "detached": "false" if connected else "true",
        "connected": "true" if connected else "false",
    }


def fresh_single_point(xtb: Path, xyz: Path, work: Path, library_path: str) -> dict[str, str]:
    work.mkdir(parents=True, exist_ok=True)
    for stale in work.iterdir():
        if stale.is_file():
            stale.unlink()
    local = work / "geom.xyz"
    local.write_bytes(xyz.read_bytes())
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = library_path
    env["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [
            str(xtb),
            "geom.xyz",
            "--gfn",
            "2",
            "--etemp",
            "300",
            "--acc",
            "0.01",
            "--iterations",
            "500",
            "--grad",
            "--norestart",
        ],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    (work / "xtb.out").write_text(result.stdout)
    (work / "xtb.err").write_text(result.stderr)
    failed_file = (work / ".sccnotconverged").exists()
    text = (result.stdout + "\n" + result.stderr).lower()
    scc_ok = result.returncode == 0 and not failed_file and "not converged" not in text
    gradient = work / "gradient"
    max_force = ""
    energy = ""
    force_ok = "false"
    if gradient.is_file():
        components: list[float] = []
        for line in gradient.read_text().splitlines():
            if "SCF energy" in line:
                tokens = line.replace("=", " ").split()
                try:
                    energy = f"{float(tokens[tokens.index('energy') + 1]) * HARTREE_TO_EV:.8f}"
                except (ValueError, IndexError):
                    energy = ""
            parts = line.split()
            if len(parts) == 3 and all(_is_float(part) for part in parts):
                scale = HARTREE_TO_EV / BOHR_TO_A
                components.extend(abs(float(part)) * scale for part in parts)
        if components:
            peak = max(components)
            max_force = f"{peak:.8e}"
            force_ok = "true" if peak < FORCE_TOL_EV_A else "false"
    return {
        "sp_energy_eV": energy,
        "scc_converged": "true" if scc_ok else "false",
        "max_force_eV_A": max_force,
        "force_ok": force_ok if max_force else "false",
    }


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in ROW_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def done_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            (row["cell"], row["phase"], row["arm"], row["seed"])
            for row in csv.DictReader(handle)
        }


def base_env(library_path: str, probe: Path, extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for name in THREAD_VARS:
        env[name] = "1"
    env.pop("RGPOT_XTB_ETEMP", None)
    env.pop("H2H_BUDGET", None)
    env.pop("H2H_TARGET", None)
    env.pop("H2H_GTOL", None)
    env.pop("H2H_WALL_S", None)
    env["LD_LIBRARY_PATH"] = library_path
    env["LD_PRELOAD"] = str(probe)
    env.update(extra)
    return env


def run_process(cmd: list[str], env: dict[str, str], log: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        log.write_text(stdout + "\n---- stderr ----\n" + stderr + "\nTIMEOUT\n")
        return subprocess.CompletedProcess(cmd, 124, stdout, stderr + "\nTIMEOUT\n")
    log.write_text(result.stdout + "\n---- stderr ----\n" + result.stderr)
    return result


def rss_peak(stderr: str) -> str:
    found = RSS_RE.findall(stderr)
    if not found:
        return ""
    return found[-1][0]


def engine_calls(stderr: str) -> str:
    found = CUH2_RE.findall(stderr)
    return found[-1] if found else ""


def parse_csv_line(stdout: str, system: str) -> list[str] | None:
    for line in stdout.splitlines():
        if line.startswith(system + ","):
            return line.split(",")
    return None


def arm_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if ARM_RE.match(line)]


def empty_row(cell: str, phase: str, kind: str, arm: Arm, position: int, seed: str, budget: str, reference: str, binary_kind: str) -> dict[str, str]:
    return {
        "cell": cell,
        "phase": phase,
        "kind": kind,
        "arm": arm.name,
        "position": str(position),
        "seed": seed,
        "budget": budget,
        "reference": reference,
        "revision": arm.revision,
        "tree": arm.tree,
        "binary_sha256": arm.hashes[binary_kind],
        "hit": "false",
        "physical_hit": "false",
        "calls": "",
        "calls_to_hit": "",
        "outside_s": "",
        "wall_s": "",
        "rss_anon_peak_kb": "",
        "trajectory_sha256": "",
        "energy_sha256": "",
        "best": "",
        "censored": "true",
        "nn_min_A": "",
        "nn_max_A": "",
        "detached": "",
        "connected": "",
        "sp_energy_eV": "",
        "scc_converged": "",
        "max_force_eV_A": "",
        "force_ok": "",
        "scc_rejected": "",
        "geometry_rejected": "",
        "engine_calls": "",
        "ledger_calls": "",
    }


def finish_physical(row: dict[str, str], kind: str) -> None:
    hit = row["hit"] == "true"
    connected = row["connected"] == "true"
    if kind == "gfn2":
        ok = (
            hit
            and connected
            and row["scc_converged"] == "true"
            and row["force_ok"] == "true"
            and row["censored"] != "true"
        )
    elif kind == "covalent":
        ok = hit and connected and row["best"] not in {"", "inf", "nan"} and row["censored"] != "true"
    else:
        ok = False
    row["physical_hit"] = "true" if ok else "false"


def search_jobs() -> list[tuple[tuple[str, str, str, int, int, float], int]]:
    by_name = {cell[0]: cell for cell in CELLS}
    jobs: list[tuple[tuple[str, str, str, int, int, float], int]] = [
        (by_name["gfn2-si10"], 2000)
    ]
    for name in ("sw10", "sw13", "tc10", "sw20", "tc20"):
        for seed in SEEDS:
            jobs.append((by_name[name], seed))
    for seed in SEEDS:
        if seed != 2000:
            jobs.append((by_name["gfn2-si10"], seed))
    for name in ("gfn2-c10", "gfn2-si13"):
        for seed in SEEDS:
            jobs.append((by_name[name], seed))
    return jobs


def run_search(args: argparse.Namespace, arms: dict[str, Arm], sigma: float, seen: set[tuple[str, str, str, str]]) -> None:
    for (cell, kind, system, atomic_number, budget, reference), seed in search_jobs():
            order = ORDERS[seed % 2]
            for position, name in enumerate(order, start=1):
                key = (cell, "", name, str(seed))
                if key in seen:
                    continue
                arm = arms[name]
                row = empty_row(cell, "", kind, arm, position, str(seed), str(budget), f"{reference:.8f}", kind)
                work = args.log_dir / cell / f"{name}-s{seed}"
                work.mkdir(parents=True, exist_ok=True)
                coords = work / ("best.xyz" if kind == "gfn2" else "best.raw")
                if kind == "covalent":
                    cmd = [
                        "taskset",
                        "-c",
                        str(args.core),
                        str(arm.covalent),
                        "run",
                        system,
                        str(seed),
                        str(coords),
                    ]
                    env = base_env(
                        args.ld_library_path,
                        args.probe,
                        {"H2H_ARM": "general", "RGPOT_COVALENT_LIB": str(args.covalent_lib)},
                    )
                    timeout = 6 * 3600
                else:
                    cmd = [
                        "taskset",
                        "-c",
                        str(args.core),
                        str(arm.gfn2),
                        system,
                        str(seed),
                        str(budget),
                        f"{reference:.8f}",
                        str(coords),
                    ]
                    env = base_env(
                        args.ld_library_path,
                        args.probe,
                        {
                            "RGPOT_XTB_ENGINE": str(args.xtb_engine),
                            "H2H_WALL_S": str(int(GFN2_WALL_S)),
                        },
                    )
                    timeout = GFN2_WALL_S + 180
                result = run_process(cmd, env, work / "run.log", timeout)
                row["rss_anon_peak_kb"] = rss_peak(result.stderr)
                fields = parse_csv_line(result.stdout, system)
                if result.returncode != 0 or fields is None:
                    row["trajectory_sha256"] = sha256_text(result.stdout)
                    append_row(args.rows, row)
                    seen.add(key)
                    print(f"{cell} {name} {seed} failed rc={result.returncode}", flush=True)
                    continue
                if kind == "covalent":
                    # system,code,seed,budget,hit,evals,charged,wall,best,pot,outside,...
                    hit = fields[4] == "true"
                    charged = int(fields[6])
                    wall = float(fields[7])
                    best = fields[8]
                    outside = fields[10]
                    clocks = (7, 9, 10, 11, 12)
                    digest_body = blank_clocks(fields, clocks) + "\n" + "\n".join(arm_lines(result.stdout))
                    scc_rejected = ""
                    geometry_rejected = ""
                    censored = False
                else:
                    hit = fields[4] == "true"
                    charged = int(fields[6])
                    wall = float(fields[8])
                    best = fields[9]
                    outside = fields[11]
                    clocks = (8, 10, 11, 12, 13)
                    digest_body = blank_clocks(fields, clocks)
                    scc_rejected = fields[14] if len(fields) > 14 else ""
                    geometry_rejected = fields[15] if len(fields) > 15 else ""
                    censored = (not hit) and charged < budget
                reported_budget = int(fields[3])
                if reported_budget != budget:
                    raise SystemExit(f"{cell} reported budget {reported_budget}")
                row.update(
                    {
                        "hit": "true" if hit else "false",
                        "calls": str(charged),
                        "outside_s": outside,
                        "wall_s": f"{wall:.6f}",
                        "best": best,
                        "censored": "true" if censored else "false",
                        "trajectory_sha256": sha256_text(digest_body + "\n"),
                        "scc_rejected": scc_rejected,
                        "geometry_rejected": geometry_rejected,
                        "ledger_calls": str(charged),
                    }
                )
                if hit:
                    row["calls_to_hit"] = fields[5]
                elif censored:
                    row["calls_to_hit"] = ""
                else:
                    row["calls_to_hit"] = str(budget + 1)
                if coords.is_file() and coords.stat().st_size > 0:
                    raw = read_xyz_or_raw(coords)
                    scale = sigma if system.startswith("sw") else 1.0
                    scaled = [(x * scale, y * scale, z * scale) for x, y, z in raw]
                    row.update(geometry_report(scaled, atomic_number))
                    row["energy_sha256"] = sha256_text(best + "\n" + coords.read_text())
                    if kind == "gfn2":
                        point = fresh_single_point(
                            args.xtb,
                            coords,
                            work / "sp",
                            args.ld_library_path,
                        )
                        row.update(point)
                else:
                    row["energy_sha256"] = sha256_text(best + "\n")
                    row["detached"] = "true"
                    row["connected"] = "false"
                finish_physical(row, kind)
                append_row(args.rows, row)
                seen.add(key)
                print(
                    f"{cell} {name} {seed} hit={row['hit']} physical={row['physical_hit']} "
                    f"calls={row['calls']} censored={row['censored']}",
                    flush=True,
                )


def prior_engine(path: Path) -> dict[tuple[str, str], int]:
    totals: dict[tuple[str, str], int] = {}
    if not path.exists():
        return totals
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["cell"] == "slab" and row["engine_calls"]:
                totals[(row["phase"], row["arm"])] = int(row["engine_calls"])
    return totals


def run_slab(args: argparse.Namespace, arms: dict[str, Arm], seen: set[tuple[str, str, str, str]]) -> None:
    phases = (("combined", "1"), ("withgrad", "2"))
    totals = prior_engine(args.rows)
    for phase, budget in phases:
        order = ORDERS[0]
        for position, name in enumerate(order, start=1):
            key = ("slab", phase, name, "2000")
            if key in seen:
                continue
            arm = arms[name]
            work = args.log_dir / "slab" / f"{phase}-{name}"
            row = empty_row("slab", phase, "slab", arm, position, "2000", budget, "", "slab")
            row["censored"] = "false"
            row["hit"] = ""
            row["physical_hit"] = ""
            cmd = [
                "taskset",
                "-c",
                str(args.core),
                str(arm.slab),
                str(args.slab_con),
                budget,
                "2000",
            ]
            env = base_env(
                args.ld_library_path,
                args.probe,
                {
                    "RGPOT_CUH2_LIBRARY": str(args.cuh2_counter),
                    "CUH2_REAL": str(args.cuh2_lib),
                    "RELAX_STEPS": "0",
                },
            )
            result = run_process(cmd, env, work / "run.log", 600)
            calls = engine_calls(result.stderr)
            row["rss_anon_peak_kb"] = rss_peak(result.stderr)
            row["engine_calls"] = calls
            row["trajectory_sha256"] = sha256_text(
                "\n".join(
                    line
                    for line in result.stdout.splitlines()
                    if not line.startswith("rss_anon_self") and not line.startswith("cuh2_engine_calls")
                )
                + "\n"
            )
            charged = ""
            for line in result.stdout.splitlines():
                if line.startswith("SUMMARY "):
                    for token in line.split():
                        if token.startswith("charged="):
                            charged = token.split("=", 1)[1]
            row["ledger_calls"] = charged
            row["calls"] = calls
            if calls != "" and result.returncode == 0:
                totals[(phase, name)] = int(calls)
            if result.returncode != 0 or calls == "" or charged != budget:
                row["censored"] = "true"
            if phase == "withgrad" and ("combined", name) in totals and calls != "":
                row["calls_to_hit"] = str(int(calls) - totals[("combined", name)])
            append_row(args.rows, row)
            seen.add(key)
            print(f"slab {phase} {name} engine={calls} ledger={charged} rc={result.returncode}", flush=True)


def check_covalent_budgets(arm: Arm) -> None:
    for cell, kind, system, _atomic_number, budget, reference in CELLS:
        if kind != "covalent":
            continue
        result = subprocess.run(
            [str(arm.covalent), "budget", system],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = result.stdout.splitlines()
        got_budget = int(lines[0])
        got_reference = float(lines[1])
        if got_budget != budget or abs(got_reference - reference) > 1.0e-8:
            raise SystemExit(
                f"{arm.name} {cell} budget {got_budget} reference {got_reference}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", nargs=5, metavar=("NAME", "SOURCE", "COVALENT", "GFN2", "SLAB"))
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--covalent-lib", type=Path, required=True)
    parser.add_argument("--xtb-engine", type=Path, required=True)
    parser.add_argument("--xtb", type=Path, required=True)
    parser.add_argument("--cuh2-lib", type=Path, required=True)
    parser.add_argument("--cuh2-counter", type=Path, required=True)
    parser.add_argument("--slab-con", type=Path, required=True)
    parser.add_argument("--ld-library-path", required=True)
    parser.add_argument("--core", type=int, default=0)
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-slab", action="store_true")
    args = parser.parse_args()
    if not args.arm:
        raise SystemExit("at least the four arms are required")
    flat = [item for group in args.arm for item in group]
    arms = load_arms(flat)
    sigma = sw_sigma(args.covalent_lib)
    print(f"sw_sigma {sigma:.8f}", flush=True)
    check_covalent_budgets(arms["A"])
    check_covalent_budgets(arms["B"])
    seen = done_keys(args.rows)
    if not args.skip_slab:
        run_slab(args, arms, seen)
    if not args.skip_search:
        run_search(args, arms, sigma, seen)


if __name__ == "__main__":
    sys.exit(main())
