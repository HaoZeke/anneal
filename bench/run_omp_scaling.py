#!/usr/bin/env python3
"""OpenMP scaling harness for rgpot classical potentials.

The harness enforces:
- release builds
- pinned OpenMP placement
- interleaved rounds with per-round order reversal
- n >= 12 samples per reported cell
- median and IQR reporting
- A/A noise bands and Mann-Whitney p-values for Morse A-vs-develop checks
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_POTENTIALS = (
    "lj",
    "morse",
    "eam",
    "emt",
    "sw",
    "tersoff",
    "zbl",
    "sutton-chen",
)

TEAM_MIN_ATOMS = 8192
EMT_MIN_CELL_ATOMS = 256

KERNEL_SOURCE_SUFFIXES = (
    "CppCore/rgpot/LennardJones/LJPot.cc",
    "CppCore/rgpot/Morse/MorsePot.cc",
    "CppCore/rgpot/EAM/EAMPot.cc",
    "CppCore/rgpot/StillingerWeber/SWSi.cc",
    "CppCore/rgpot/Tersoff/TersoffSiC.cc",
    "CppCore/rgpot/ZBL/ZBLPot.cc",
    "CppCore/rgpot/SuttonChen/SuttonChenPot.cc",
    "CppCore/rgpot/EMT/EMTPot.cc",
    "CppCore/rgpot/nlist/NeighbourList.cc",
)


@dataclass(frozen=True)
class BuildContext:
    repo: Path
    build_dir: Path
    worker: Path
    sha: str


@dataclass(frozen=True)
class LscpuInfo:
    logical_cores: int
    physical_cores: int
    threads_per_core: int
    sockets: int


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("empty percentile input")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def iqr(values: list[float]) -> float:
    return percentile(values, 75.0) - percentile(values, 25.0)


def mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def parse_cpu_list(token: str) -> set[int]:
    found: set[int] = set()
    for part in token.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            found.update(range(lo, hi + 1))
        else:
            found.add(int(part))
    return found


def sibling_key(cpu: int) -> str:
    path = Path("/sys/devices/system/cpu") / f"cpu{cpu}" / "topology" / "thread_siblings_list"
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError:
        return str(cpu)


def exclusive_cpus() -> list[int]:
    allowed = sorted(os.sched_getaffinity(0))
    chosen: list[int] = []
    seen: set[str] = set()
    for cpu in allowed:
        key = sibling_key(cpu)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(cpu)
    return chosen


def parse_leading_int(token: str, default: int) -> int:
    match = re.match(r"^\s*([0-9]+)", token)
    if not match:
        return default
    return int(match.group(1))


def lscpu_info() -> LscpuInfo:
    out = run(["lscpu"], capture_output=True)
    fields: dict[str, str] = {}
    for line in out.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    logical = parse_leading_int(fields.get("CPU(s)", "0"), 0)
    threads_per_core = max(1, parse_leading_int(fields.get("Thread(s) per core", "1"), 1))
    cores_per_socket = parse_leading_int(fields.get("Core(s) per socket", "0"), 0)
    sockets = max(1, parse_leading_int(fields.get("Socket(s)", "1"), 1))
    physical = cores_per_socket * sockets
    if physical <= 0 and logical > 0:
        physical = max(1, logical // threads_per_core)
    if physical <= 0:
        physical = 1
    if logical <= 0:
        logical = physical * threads_per_core
    return LscpuInfo(
        logical_cores=logical,
        physical_cores=physical,
        threads_per_core=threads_per_core,
        sockets=sockets,
    )


def cpu_model() -> str:
    model = "unknown"
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    return model


def compiler_version(cxx: str) -> str:
    out = run([cxx, "--version"], capture_output=True)
    line = out.stdout.splitlines()
    return line[0].strip() if line else "unknown"


def openmp_runtime(worker: Path) -> dict[str, str]:
    out = run(["ldd", str(worker)], capture_output=True)
    runtime_name = "unknown"
    runtime_path = ""
    for line in out.stdout.splitlines():
        if "libgomp" in line or "libomp" in line:
            lib_match = re.search(r"(libgomp\.so[^\s]*|libomp\.so[^\s]*)", line)
            if lib_match:
                runtime_name = Path(lib_match.group(1)).name
            match = re.search(r"=>\s+(\S+)", line)
            if match:
                runtime_path = match.group(1)
            break
    package_hint = "unknown"
    package_candidates = ("libgomp1", "libomp5")
    if runtime_name.startswith("libomp"):
        package_candidates = ("libomp5", "libgomp1")
    for package_name in package_candidates:
        try:
            pkg_out = run(
                ["dpkg-query", "-W", "-f=${Version}", package_name],
                capture_output=True,
            )
            version = pkg_out.stdout.strip()
        except subprocess.CalledProcessError:
            continue
        if version:
            package_hint = f"{package_name} {version}"
            break
    return {
        "library": runtime_name,
        "package_version": package_hint,
    }


def command_text(entry: dict[str, Any]) -> str:
    if "arguments" in entry and isinstance(entry["arguments"], list):
        return " ".join(str(arg) for arg in entry["arguments"])
    return str(entry.get("command", ""))


def openmp_compile_evidence(build_dir: Path, repo: Path) -> dict[str, Any]:
    compile_commands = build_dir / "compile_commands.json"
    if not compile_commands.is_file():
        return {"available": False, "entries": [], "all_sources_have_fopenmp": False}
    data = json.loads(compile_commands.read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []
    all_ok = True
    for suffix in KERNEL_SOURCE_SUFFIXES:
        expected = suffix.replace("\\", "/")
        found = None
        for entry in data:
            file_path = str(entry["file"]).replace("\\", "/")
            if file_path.endswith(expected):
                found = entry
                break
        if found is None:
            entries.append({"source": suffix, "found": False, "has_fopenmp": False})
            all_ok = False
            continue
        has_fopenmp = "-fopenmp" in command_text(found).split()
        entries.append({"source": suffix, "found": True, "has_fopenmp": has_fopenmp})
        all_ok = all_ok and has_fopenmp
    library = find_rgpot_library(build_dir)
    ldd = run(["ldd", str(library)], capture_output=True)
    links_openmp = any("libgomp" in line or "libomp" in line for line in ldd.stdout.splitlines())
    return {
        "available": True,
        "entries": entries,
        "all_sources_have_fopenmp": all_ok,
        "library_links_openmp_runtime": links_openmp,
    }


def configure_and_build(repo: Path, build_dir: Path) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    meson_exe = shutil.which("meson")
    if meson_exe:
        meson_prefix = [meson_exe]
    else:
        meson_prefix = [sys.executable, "-m", "mesonbuild.mesonmain"]

    setup = meson_prefix + [
        "setup",
        str(build_dir),
        str(repo),
        "-Dbuildtype=release",
        "-Dwith_tests=false",
        "-Dwith_examples=false",
        "-Dwith_rpc=false",
        "-Dwith_cache=false",
        "-Dwith_xtensor=false",
        "-Dwith_eigen=false",
        "-Dwith_python=false",
        "-Dwith_fortran_pots=disabled",
        "-Dwith_water=false",
        "-Dwith_ipi=disabled",
        "-Dwith_mpi=disabled",
        "-Dwith_dftd3=false",
        "-Dwith_dftd4=false",
        "-Dwith_expr=false",
        "-Dwith_tblite=false",
        "-Dwith_metatomic=false",
        "-Dwith_xckernel=false",
        "-Dwith_skala=false",
        "-Dwith_long_range=false",
    ]
    if (build_dir / "meson-private").is_dir():
        run(meson_prefix + ["setup", "--reconfigure", str(build_dir)])
    else:
        run(setup)
    run(meson_prefix + ["compile", "-C", str(build_dir)])


def find_rgpot_library(build_dir: Path) -> Path:
    candidates = sorted(build_dir.rglob("librgpot.so*"))
    if not candidates:
        candidates = sorted(build_dir.rglob("librgpot.dylib"))
    if not candidates:
        raise FileNotFoundError(f"librgpot shared library not found under {build_dir}")
    # Prefer an unversioned soname symlink when available.
    for candidate in candidates:
        if candidate.name in {"librgpot.so", "librgpot.dylib"}:
            return candidate
    return candidates[0]


def build_worker(repo: Path, build_dir: Path, cxx: str) -> Path:
    source = repo / "bench" / "rgpot_omp_worker.cc"
    if not source.is_file():
        raise FileNotFoundError(f"worker source missing: {source}")
    output_dir = build_dir / "bench"
    output_dir.mkdir(parents=True, exist_ok=True)
    worker = output_dir / "rgpot_omp_worker"
    library = find_rgpot_library(build_dir)
    lib_dir = library.parent
    compile = [
        cxx,
        "-O3",
        "-DNDEBUG",
        "-std=c++20",
        "-fopenmp",
        str(source),
        "-I" + str(repo / "CppCore"),
        "-I" + str(repo / "include"),
        "-L" + str(lib_dir),
        "-Wl,-rpath," + str(lib_dir),
        "-lrgpot",
        "-o",
        str(worker),
    ]
    run(compile, cwd=repo)
    return worker


def ensure_worker_source(target_repo: Path, source_repo: Path) -> None:
    target = target_repo / "bench" / "rgpot_omp_worker.cc"
    source = source_repo / "bench" / "rgpot_omp_worker.cc"
    if not source.is_file():
        raise FileNotFoundError(f"worker source missing in source repo: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.read_bytes() == source.read_bytes():
        return
    shutil.copy2(source, target)


def git_sha(repo: Path) -> str:
    out = run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True)
    return out.stdout.strip()


def ensure_commit(repo: Path, expected: str) -> None:
    got = git_sha(repo)
    if got != expected:
        raise RuntimeError(f"{repo} HEAD is {got}, expected {expected}")


def make_context(repo: Path, build_dir: Path, expected_sha: str, cxx: str) -> BuildContext:
    ensure_commit(repo, expected_sha)
    configure_and_build(repo, build_dir)
    worker = build_worker(repo, build_dir, cxx)
    return BuildContext(repo=repo, build_dir=build_dir, worker=worker, sha=expected_sha)


def taskset_list(cpus: list[int]) -> str:
    return ",".join(str(cpu) for cpu in cpus)


def parse_affinity_count(token: str) -> int:
    if not token:
        return 0
    return len([part for part in token.split(",") if part.strip()])


def parse_parallel_affinity_masks(token: str) -> list[str]:
    return [part.strip() for part in token.split(";") if part.strip()]


def run_worker_once(
    ctx: BuildContext,
    *,
    potential: str,
    atoms: int,
    threads: int,
    sample_seconds: float,
    cpus: list[int],
    dump_force: Path | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "false"
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = ",".join(f"{{{cpu}}}" for cpu in cpus)
    env["GOMP_CPU_AFFINITY"] = taskset_list(cpus)
    command = [
        "taskset",
        "-c",
        taskset_list(cpus),
        str(ctx.worker),
        "--potential",
        potential,
        "--atoms",
        str(atoms),
        "--threads",
        str(threads),
        "--samples",
        "1",
        "--sample-seconds",
        f"{sample_seconds:.12g}",
    ]
    if dump_force is not None:
        command += ["--dump-force", str(dump_force)]
    out = run(command, cwd=ctx.repo, env=env, capture_output=True)
    return json.loads(out.stdout)


def sample_cell(
    ctx: BuildContext,
    *,
    potential: str,
    atoms: int,
    thread_counts: list[int],
    rounds: int,
    sample_seconds: float,
    cpus_by_threads: dict[int, list[int]],
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {
        t: {
            "build_ns": [],
            "force_ns": [],
            "total_ns": [],
            "evals": [],
            "omp_max_threads": [],
            "omp_parallel_threads": [],
            "affinity": [],
            "parallel_affinity": [],
            "energy_bits": None,
            "force_hash": None,
        }
        for t in thread_counts
    }
    for round_index in range(rounds):
        order = thread_counts if round_index % 2 == 0 else list(reversed(thread_counts))
        for threads in order:
            row = run_worker_once(
                ctx,
                potential=potential,
                atoms=atoms,
                threads=threads,
                sample_seconds=sample_seconds,
                cpus=cpus_by_threads[threads],
            )
            rows[threads]["build_ns"].append(float(row["build_ns_per_eval"][0]))
            rows[threads]["force_ns"].append(float(row["force_ns_per_eval"][0]))
            rows[threads]["total_ns"].append(float(row["total_ns_per_eval"][0]))
            rows[threads]["evals"].append(int(row["evals_per_sample"][0]))
            rows[threads]["omp_max_threads"].append(int(row["omp_max_threads_per_sample"][0]))
            rows[threads]["omp_parallel_threads"].append(int(row["omp_parallel_threads_per_sample"][0]))
            rows[threads]["affinity"].append(str(row["affinity_per_sample"][0]))
            rows[threads]["parallel_affinity"].append(str(row["omp_parallel_affinity_per_sample"][0]))
            if rows[threads]["energy_bits"] is None:
                rows[threads]["energy_bits"] = row["energy_bits"]
            if rows[threads]["force_hash"] is None:
                rows[threads]["force_hash"] = row["force_hash"]
    return rows


def sample_weak_cell(
    ctx: BuildContext,
    *,
    potential: str,
    atoms_by_threads: dict[int, int],
    thread_counts: list[int],
    rounds: int,
    sample_seconds: float,
    cpus_by_threads: dict[int, list[int]],
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {
        t: {
            "build_ns": [],
            "force_ns": [],
            "total_ns": [],
            "evals": [],
            "omp_max_threads": [],
            "omp_parallel_threads": [],
            "affinity": [],
            "parallel_affinity": [],
            "energy_bits": None,
            "force_hash": None,
        }
        for t in thread_counts
    }
    for round_index in range(rounds):
        order = thread_counts if round_index % 2 == 0 else list(reversed(thread_counts))
        for threads in order:
            atoms = atoms_by_threads[threads]
            row = run_worker_once(
                ctx,
                potential=potential,
                atoms=atoms,
                threads=threads,
                sample_seconds=sample_seconds,
                cpus=cpus_by_threads[threads],
            )
            rows[threads]["build_ns"].append(float(row["build_ns_per_eval"][0]))
            rows[threads]["force_ns"].append(float(row["force_ns_per_eval"][0]))
            rows[threads]["total_ns"].append(float(row["total_ns_per_eval"][0]))
            rows[threads]["evals"].append(int(row["evals_per_sample"][0]))
            rows[threads]["omp_max_threads"].append(int(row["omp_max_threads_per_sample"][0]))
            rows[threads]["omp_parallel_threads"].append(int(row["omp_parallel_threads_per_sample"][0]))
            rows[threads]["affinity"].append(str(row["affinity_per_sample"][0]))
            rows[threads]["parallel_affinity"].append(str(row["omp_parallel_affinity_per_sample"][0]))
            if rows[threads]["energy_bits"] is None:
                rows[threads]["energy_bits"] = row["energy_bits"]
            if rows[threads]["force_hash"] is None:
                rows[threads]["force_hash"] = row["force_hash"]
    return rows


def summarize_cell(samples: list[float], atoms: int) -> dict[str, float]:
    med = statistics.median(samples)
    spread = iqr(samples)
    return {
        "median_ns_per_eval": med,
        "iqr_ns_per_eval": spread,
        "median_eval_per_s": 1.0e9 / med if med > 0.0 else 0.0,
        "median_ns_per_atom_eval": med / float(atoms),
    }


def ranks(values: list[tuple[float, int]]) -> list[float]:
    sorted_pairs = sorted(enumerate(values), key=lambda item: item[1][0])
    out = [0.0] * len(values)
    i = 0
    while i < len(sorted_pairs):
        j = i + 1
        while j < len(sorted_pairs) and sorted_pairs[j][1][0] == sorted_pairs[i][1][0]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        for k in range(i, j):
            original_index = sorted_pairs[k][0]
            out[original_index] = avg_rank
        i = j
    return out


def mann_whitney(dev_times: list[float], ref_times: list[float]) -> dict[str, float]:
    x = dev_times
    y = ref_times
    nx = len(x)
    ny = len(y)
    if nx == 0 or ny == 0:
        return {"u": float("nan"), "z": float("nan"), "p_two_sided": float("nan"), "p_dev_slower": float("nan")}
    merged: list[tuple[float, int]] = [(v, 0) for v in x] + [(v, 1) for v in y]
    rank_values = ranks(merged)
    rank_sum_x = sum(rank_values[i] for i in range(nx))
    u1 = rank_sum_x - nx * (nx + 1) / 2.0
    n = nx + ny
    mean_u = nx * ny / 2.0

    tie_counts: dict[float, int] = {}
    for value, _group in merged:
        tie_counts[value] = tie_counts.get(value, 0) + 1
    tie_term = sum(count**3 - count for count in tie_counts.values())
    if n <= 1:
        return {"u": u1, "z": float("nan"), "p_two_sided": float("nan"), "p_dev_slower": float("nan")}
    var_u = (nx * ny / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if var_u <= 0.0:
        return {"u": u1, "z": float("nan"), "p_two_sided": 1.0, "p_dev_slower": 1.0}
    # Continuity correction for one-sided "dev is slower" tail.
    correction = 0.5 if u1 > mean_u else -0.5
    z = (u1 - mean_u - correction) / math.sqrt(var_u)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p_dev_slower = 1.0 - cdf
    p_two = 2.0 * min(cdf, 1.0 - cdf)
    p_two = max(0.0, min(1.0, p_two))
    p_dev_slower = max(0.0, min(1.0, p_dev_slower))
    return {"u": u1, "z": z, "p_two_sided": p_two, "p_dev_slower": p_dev_slower}


def median_force_eval_per_s(force_samples: list[float]) -> float:
    med = statistics.median(force_samples)
    return 1.0e9 / med if med > 0.0 else 0.0


def safe_speedup(base: float, value: float) -> float:
    return base / value if value > 0.0 else 0.0


def worker_failure(
    ctx: BuildContext,
    *,
    potential: str,
    atoms: int,
    threads: int,
    sample_seconds: float,
    cpus: list[int],
) -> dict[str, Any] | None:
    try:
        run_worker_once(
            ctx,
            potential=potential,
            atoms=atoms,
            threads=threads,
            sample_seconds=sample_seconds,
            cpus=cpus,
        )
    except subprocess.CalledProcessError as err:
        return {
            "exit_code": err.returncode,
            "stderr": (err.stderr or "").strip(),
            "stdout": (err.stdout or "").strip(),
        }
    return None


def compare_morse_cell(
    *,
    develop_ctx: BuildContext,
    ref_ctx: BuildContext,
    atoms: int,
    threads: int,
    rounds: int,
    sample_seconds: float,
    cpus: list[int],
) -> dict[str, Any]:
    dev_ns: list[float] = []
    ref_ns: list[float] = []
    for round_index in range(rounds):
        order = ("ref", "dev") if round_index % 2 == 0 else ("dev", "ref")
        for arm in order:
            ctx = ref_ctx if arm == "ref" else develop_ctx
            row = run_worker_once(
                ctx,
                potential="morse",
                atoms=atoms,
                threads=threads,
                sample_seconds=sample_seconds,
                cpus=cpus,
            )
            value = float(row["total_ns_per_eval"][0])
            if arm == "ref":
                ref_ns.append(value)
            else:
                dev_ns.append(value)

    aa_a: list[float] = []
    aa_b: list[float] = []
    for round_index in range(rounds):
        order = ("aa_a", "aa_b") if round_index % 2 == 0 else ("aa_b", "aa_a")
        seen: dict[str, float] = {}
        for arm in order:
            row = run_worker_once(
                develop_ctx,
                potential="morse",
                atoms=atoms,
                threads=threads,
                sample_seconds=sample_seconds,
                cpus=cpus,
            )
            seen[arm] = float(row["total_ns_per_eval"][0])
        aa_a.append(seen["aa_a"])
        aa_b.append(seen["aa_b"])

    dev_thr = [1.0e9 / x for x in dev_ns]
    ref_thr = [1.0e9 / x for x in ref_ns]
    aa_a_thr = [1.0e9 / x for x in aa_a]
    aa_b_thr = [1.0e9 / x for x in aa_b]
    aa_ratios = [b / a for a, b in zip(aa_a_thr, aa_b_thr)]
    ratio = statistics.median(dev_thr) / statistics.median(ref_thr)
    band_low = min(aa_ratios)
    band_high = max(aa_ratios)
    stats = mann_whitney(dev_ns, ref_ns)
    return {
        "atoms": atoms,
        "threads": threads,
        "dev_total_ns": dev_ns,
        "ref_total_ns": ref_ns,
        "dev_ref_throughput_ratio": ratio,
        "aa_band_low": band_low,
        "aa_band_high": band_high,
        "regression_beyond_band": ratio < band_low or ratio > band_high,
        "p_dev_slower": stats["p_dev_slower"],
        "p_two_sided": stats["p_two_sided"],
        "u_statistic": stats["u"],
    }


def parse_force_dump(path: Path) -> tuple[float, array.array]:
    with path.open("rb") as handle:
        raw_n = handle.read(8)
        if len(raw_n) != 8:
            raise RuntimeError("force dump header is truncated")
        n_atoms = int.from_bytes(raw_n, byteorder="little", signed=False)
        energy_bytes = handle.read(8)
        if len(energy_bytes) != 8:
            raise RuntimeError("force dump energy is truncated")
        energy = array.array("d")
        energy.frombytes(energy_bytes)
        values = array.array("d")
        values.fromfile(handle, 3 * n_atoms)
    return energy[0], values


def max_abs_force_delta(reference: array.array, other: array.array) -> float:
    if len(reference) != len(other):
        raise RuntimeError("force vector length mismatch")
    delta = 0.0
    for a, b in zip(reference, other):
        diff = abs(a - b)
        if diff > delta:
            delta = diff
    return delta


def format_float(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.{digits}g}"


def single_line(text: str) -> str:
    return " ".join(text.split())


def emt_sizes(strong_sizes: list[int]) -> list[int]:
    out = list(strong_sizes)
    if not out:
        return out
    out[0] = max(EMT_MIN_CELL_ATOMS, out[0])
    return out


def find_row(
    rows: list[dict[str, Any]], *, potential: str, atoms: int, threads: int
) -> dict[str, Any] | None:
    for row in rows:
        if row["potential"] == potential and row["atoms"] == atoms and row["threads"] == threads:
            return row
    return None


def infer_cliff_cause(*, atoms: int, threads: int, build_median_ns: float, force_median_ns: float) -> str:
    if atoms < TEAM_MIN_ATOMS:
        return f"serial by design (below kTeamMinAtoms={TEAM_MIN_ATOMS})"
    atoms_per_thread = atoms / float(threads)
    if build_median_ns > 1.5 * force_median_ns:
        return "neighbor-list build dominates the step cost"
    if atoms_per_thread < 400.0:
        return "small atoms-per-thread leaves OpenMP overhead dominant"
    return "force kernel scaling is limited, likely by memory bandwidth/cache pressure"


def write_markdown(
    *,
    path: Path,
    metadata: dict[str, Any],
    strong: list[dict[str, Any]],
    weak: list[dict[str, Any]],
    morse: list[dict[str, Any]],
    determinism: list[dict[str, Any]],
    cliffs: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    emt_reproducer: dict[str, Any],
    notable_force_cliffs: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    strong_rows = sorted(strong, key=lambda row: (row["potential"], row["atoms"], row["threads"]))
    weak_rows = sorted(weak, key=lambda row: (row["potential"], row["threads"]))
    morse_rows = sorted(morse, key=lambda row: (row["atoms"], row["threads"]))
    determinism_rows = sorted(determinism, key=lambda row: (row["potential"], row["atoms"]))
    lines.append("# rgpot OpenMP scaling report")
    lines.append("")
    lines.append("## Build and machine")
    lines.append("")
    lines.append(f"- develop SHA: `{metadata['base_sha']}`")
    lines.append(f"- reference A SHA: `{metadata['ref_sha']}`")
    lines.append(f"- compiler: `{metadata['compiler_version']}`")
    lines.append(f"- build type: `{metadata['build_type']}`")
    lines.append(
        f"- OpenMP runtime library: `{metadata['openmp_runtime']['library']}` "
        f"(`{metadata['openmp_runtime']['package_version']}`)"
    )
    lines.append(f"- CPU model: `{metadata['cpu_model']}`")
    lines.append(
        f"- lscpu logical cores: `{metadata['lscpu_logical_cores']}`; "
        f"lscpu physical cores: `{metadata['lscpu_physical_cores']}`"
    )
    lines.append(
        f"- affinity logical cores: `{metadata['affinity_logical_cores']}`; "
        f"affinity physical cores: `{metadata['affinity_physical_cores']}`"
    )
    lines.append(f"- thread(s) per core from lscpu: `{metadata['lscpu_threads_per_core']}`")
    lines.append(f"- benchmark thread counts: `{metadata['thread_counts']}`")
    lines.append(
        "- pinning: `OMP_PROC_BIND=close`, explicit `OMP_PLACES` cpu list, "
        "`GOMP_CPU_AFFINITY` fixed, and taskset to the same cpu list"
    )
    lines.append(
        "- neighbour-list builder timed separately: "
        f"`{metadata['neighbour_list_builder']}`"
    )
    lines.append(
        "- neighbour-list parallel threshold: "
        f"`{metadata['neighbour_list_parallel_note']}`"
    )
    lines.append(
        f"- force kernels and neighbour-list build both stay serial below "
        f"`kTeamMinAtoms={TEAM_MIN_ATOMS}` (`CppCore/rgpot/Slices.hpp`)"
    )
    lines.append("")

    lines.append("## OpenMP build evidence (develop vs reference A)")
    lines.append("")
    lines.append(
        f"- develop all listed kernels compiled with `-fopenmp`: "
        f"`{metadata['openmp_build_evidence']['develop']['all_sources_have_fopenmp']}`"
    )
    lines.append(
        f"- reference A all listed kernels compiled with `-fopenmp`: "
        f"`{metadata['openmp_build_evidence']['reference']['all_sources_have_fopenmp']}`"
    )
    lines.append(
        f"- develop librgpot links OpenMP runtime: "
        f"`{metadata['openmp_build_evidence']['develop']['library_links_openmp_runtime']}`"
    )
    lines.append(
        f"- reference A librgpot links OpenMP runtime: "
        f"`{metadata['openmp_build_evidence']['reference']['library_links_openmp_runtime']}`"
    )
    lines.append("")
    lines.append("|Source|develop `-fopenmp`|reference A `-fopenmp`|")
    lines.append("|---|---|---|")
    dev_entries = {
        entry["source"]: entry
        for entry in metadata["openmp_build_evidence"]["develop"]["entries"]
    }
    ref_entries = {
        entry["source"]: entry
        for entry in metadata["openmp_build_evidence"]["reference"]["entries"]
    }
    for source in KERNEL_SOURCE_SUFFIXES:
        dev = dev_entries.get(source, {"found": False, "has_fopenmp": False})
        ref = ref_entries.get(source, {"found": False, "has_fopenmp": False})
        lines.append(
            f"|{source}|"
            f"{'yes' if dev.get('found') and dev.get('has_fopenmp') else 'no'}|"
            f"{'yes' if ref.get('found') and ref.get('has_fopenmp') else 'no'}|"
        )
    lines.append("")

    lines.append("## Strong scaling (develop, total = build + force)")
    lines.append("")
    lines.append(
        "|Potential|Atoms|Threads|SMT row|eval/s median|ns/atom/eval median|"
        "IQR ns/eval|Speedup|Efficiency|build ns/eval median|force ns/eval median|"
        "build share|"
    )
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in strong_rows:
        lines.append(
            "|{potential}|{atoms}|{threads}|{smt}|{eps}|{nsa}|{iqr}|{speedup}|"
            "{eff}|{build}|{force}|{share}|".format(
                potential=row["potential"],
                atoms=row["atoms"],
                threads=row["threads"],
                smt=row["smt_label"],
                eps=format_float(row["summary"]["median_eval_per_s"], 6),
                nsa=format_float(row["summary"]["median_ns_per_atom_eval"], 6),
                iqr=format_float(row["summary"]["iqr_ns_per_eval"], 6),
                speedup=format_float(row["speedup"], 5),
                eff=format_float(row["efficiency"], 5),
                build=format_float(row["build_median_ns"], 6),
                force=format_float(row["force_median_ns"], 6),
                share=format_float(row["build_share"], 6),
            )
        )
    lines.append("")

    lines.append("## Strong scaling (develop, force-evaluation only)")
    lines.append("")
    lines.append("|Potential|Atoms|Threads|force eval/s median|force speedup|force efficiency|")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in strong_rows:
        lines.append(
            "|{potential}|{atoms}|{threads}|{eps}|{speedup}|{eff}|".format(
                potential=row["potential"],
                atoms=row["atoms"],
                threads=row["threads"],
                eps=format_float(row["force_eval_per_s_median"], 6),
                speedup=format_float(row["force_speedup"], 5),
                eff=format_float(row["force_efficiency"], 5),
            )
        )
    lines.append("")

    lines.append("## Strong scaling (develop, neighbour-list build only)")
    lines.append("")
    lines.append("|Potential|Atoms|Threads|build ns/eval median|build speedup|build efficiency|")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in strong_rows:
        lines.append(
            "|{potential}|{atoms}|{threads}|{build}|{speedup}|{eff}|".format(
                potential=row["potential"],
                atoms=row["atoms"],
                threads=row["threads"],
                build=format_float(row["build_median_ns"], 6),
                speedup=format_float(row["build_speedup"], 5),
                eff=format_float(row["build_efficiency"], 5),
            )
        )
    lines.append("")

    lines.append("## OpenMP and affinity diagnostics from worker samples")
    lines.append("")
    lines.append(
        "|Potential|Atoms|Threads|omp_get_max_threads observed|parallel-team size observed|"
        "main-thread affinity cpu count observed|main-thread affinity sets|"
        "parallel-thread affinity cpu count observed|parallel-thread affinity masks|"
    )
    lines.append("|---|---:|---:|---|---|---|---|---|---|")
    for row in strong_rows:
        lines.append(
            "|{potential}|{atoms}|{threads}|{max_thr}|{team_thr}|{main_count}|{main_aff}|"
            "{par_count}|{par_aff}|".format(
                potential=row["potential"],
                atoms=row["atoms"],
                threads=row["threads"],
                max_thr=row["omp_max_threads_observed"],
                team_thr=row["omp_parallel_threads_observed"],
                main_count=row["affinity_cpu_count_observed"],
                main_aff=row["affinity_observed"],
                par_count=row["omp_parallel_thread_affinity_cpu_counts_observed"],
                par_aff=row["omp_parallel_thread_affinity_masks_observed"],
            )
        )
    lines.append("")

    lines.append("## Weak scaling (develop, total)")
    lines.append("")
    lines.append(
        "|Potential|Threads|SMT row|Atoms|eval/s median|ns/atom/eval median|"
        "Weak efficiency|"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for row in weak_rows:
        lines.append(
            "|{potential}|{threads}|{smt}|{atoms}|{eps}|{nsa}|{eff}|".format(
                potential=row["potential"],
                threads=row["threads"],
                smt=row["smt_label"],
                atoms=row["atoms"],
                eps=format_float(row["summary"]["median_eval_per_s"], 6),
                nsa=format_float(row["summary"]["median_ns_per_atom_eval"], 6),
                eff=format_float(row["weak_efficiency"], 5),
            )
        )
    lines.append("")

    lines.append("## Weak scaling (develop, force-evaluation only)")
    lines.append("")
    lines.append("|Potential|Threads|Atoms|force eval/s median|force weak efficiency|")
    lines.append("|---|---:|---:|---:|---:|")
    for row in weak_rows:
        lines.append(
            "|{potential}|{threads}|{atoms}|{eps}|{eff}|".format(
                potential=row["potential"],
                threads=row["threads"],
                atoms=row["atoms"],
                eps=format_float(row["force_eval_per_s_median"], 6),
                eff=format_float(row["force_weak_efficiency"], 5),
            )
        )
    lines.append("")

    lines.append("## Morse regression check: reference A vs develop")
    lines.append("")
    lines.append(
        "|Atoms|Threads|Throughput ratio develop/reference|A/A band low|A/A band high|"
        "Mann-Whitney p (develop slower)|Beyond A/A band?|"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for row in morse_rows:
        lines.append(
            "|{atoms}|{threads}|{ratio}|{low}|{high}|{p}|{flag}|".format(
                atoms=row["atoms"],
                threads=row["threads"],
                ratio=format_float(row["dev_ref_throughput_ratio"], 6),
                low=format_float(row["aa_band_low"], 6),
                high=format_float(row["aa_band_high"], 6),
                p=format_float(row["p_dev_slower"], 6),
                flag="yes" if row["regression_beyond_band"] else "no",
            )
        )
    lines.append("")

    lines.append("## Determinism across thread counts")
    lines.append("")
    lines.append(
        "|Potential|Atoms|Bit-identical energy+force hashes|"
        "Max abs force deviation when not bit-identical|"
    )
    lines.append("|---|---:|---|---:|")
    for row in determinism_rows:
        lines.append(
            "|{potential}|{atoms}|{bit}|{delta}|".format(
                potential=row["potential"],
                atoms=row["atoms"],
                bit="yes" if row["bit_identical"] else "no",
                delta=format_float(row["max_abs_force_deviation"], 6),
            )
        )
    lines.append("")

    lines.append("## EMT failure reproducer")
    lines.append("")
    lines.append(
        f"- reproducer command arguments: `--potential {emt_reproducer['potential']} "
        f"--atoms {emt_reproducer['atoms']} --threads {emt_reproducer['threads']} "
        f"--samples 1 --sample-seconds {emt_reproducer['sample_seconds']}`"
    )
    lines.append(
        f"- develop status: `{emt_reproducer['develop']['status']}`; "
        f"stderr: `{emt_reproducer['develop']['stderr']}`"
    )
    lines.append(
        f"- reference A status: `{emt_reproducer['reference']['status']}`; "
        f"stderr: `{emt_reproducer['reference']['stderr']}`"
    )
    lines.append("")

    lines.append("## Scaling cliffs (efficiency < 0.6)")
    lines.append("")
    if cliffs:
        lines.append("|Potential|Atoms|Threads|Efficiency|Likely cause|")
        lines.append("|---|---:|---:|---:|---|")
        for row in sorted(cliffs, key=lambda entry: (entry["potential"], entry["atoms"], entry["threads"])):
            lines.append(
                f"|{row['potential']}|{row['atoms']}|{row['threads']}|"
                f"{format_float(row['efficiency'], 6)}|{row['likely_cause']}|"
            )
    else:
        lines.append("No efficiency cliff below 0.6 was observed.")
    lines.append("")

    lines.append("## Notable force-kernel cliffs")
    lines.append("")
    if notable_force_cliffs:
        lines.append("|Potential|Atoms|Threads|Force speedup|Measured cause|")
        lines.append("|---|---:|---:|---:|---|")
        for row in notable_force_cliffs:
            lines.append(
                f"|{row['potential']}|{row['atoms']}|{row['threads']}|"
                f"{format_float(row['force_speedup'], 6)}|{row['cause']}|"
            )
    else:
        lines.append("No additional force-kernel-specific cliffs were tagged.")
    lines.append("")

    lines.append("## Potentials skipped")
    lines.append("")
    if skipped:
        lines.append("|Potential|Reason|")
        lines.append("|---|---|")
        for row in skipped:
            reason = row["reason"].replace("\n", "<br>")
            lines.append(f"|{row['potential']}|{reason}|")
    else:
        lines.append("None.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-develop", required=True)
    parser.add_argument("--repo-reference", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--ref-sha", required=True)
    parser.add_argument("--build-develop", required=True)
    parser.add_argument("--build-reference", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--samples", required=True, type=int)
    parser.add_argument("--sample-seconds", required=True, type=float)
    parser.add_argument("--weak-atoms-per-thread", required=True, type=int)
    parser.add_argument("--strong-sizes", required=True)
    parser.add_argument("--potentials", required=True)
    parser.add_argument("--cxx", required=True)
    args = parser.parse_args()

    strong_sizes = [int(part) for part in args.strong_sizes.split(",") if part.strip()]
    potentials = [part.strip() for part in args.potentials.split(",") if part.strip()]
    if args.samples < 12:
        raise RuntimeError("samples must be at least 12 per cell")

    repo_develop = Path(args.repo_develop).resolve()
    repo_ref = Path(args.repo_reference).resolve()
    build_develop = Path(args.build_develop).resolve()
    build_ref = Path(args.build_reference).resolve()
    out_json = Path(args.out_json).resolve()
    out_md = Path(args.out_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lscpu = lscpu_info()
    affinity_logical_cores = len(sorted(os.sched_getaffinity(0)))
    cpus = exclusive_cpus()
    affinity_physical_cores = len(cpus)
    if affinity_physical_cores < 1:
        raise RuntimeError("no CPU cores available")
    thread_ceiling = min(affinity_physical_cores, lscpu.physical_cores)
    if thread_ceiling < 1:
        raise RuntimeError("thread ceiling from lscpu/affinity is < 1")
    thread_counts_set: set[int] = set()
    for value in (1, 2, 4):
        if value <= thread_ceiling:
            thread_counts_set.add(value)
    if lscpu.physical_cores >= 8 and 8 <= thread_ceiling:
        thread_counts_set.add(8)
    thread_counts_set.add(thread_ceiling)
    thread_counts = sorted(thread_counts_set)
    cpus_by_threads = {t: cpus[:t] for t in thread_counts}

    ensure_worker_source(repo_ref, repo_develop)

    develop_ctx = make_context(
        repo_develop, build_develop, args.base_sha, args.cxx
    )
    ref_ctx = make_context(
        repo_ref, build_ref, args.ref_sha, args.cxx
    )

    openmp_evidence = {
        "develop": openmp_compile_evidence(build_develop, repo_develop),
        "reference": openmp_compile_evidence(build_ref, repo_ref),
    }

    emt_threads = thread_counts[0]
    emt_atoms = strong_sizes[0]
    emt_dev_failure = worker_failure(
        develop_ctx,
        potential="emt",
        atoms=emt_atoms,
        threads=emt_threads,
        sample_seconds=args.sample_seconds,
        cpus=cpus_by_threads[emt_threads],
    )
    emt_ref_failure = worker_failure(
        ref_ctx,
        potential="emt",
        atoms=emt_atoms,
        threads=emt_threads,
        sample_seconds=args.sample_seconds,
        cpus=cpus_by_threads[emt_threads],
    )
    emt_reproducer = {
        "potential": "emt",
        "atoms": emt_atoms,
        "threads": emt_threads,
        "sample_seconds": args.sample_seconds,
        "develop": {
            "status": "ok" if emt_dev_failure is None else f"failed (exit {emt_dev_failure['exit_code']})",
            "stderr": (
                "none"
                if emt_dev_failure is None
                else single_line(emt_dev_failure["stderr"] or emt_dev_failure["stdout"] or "no message")
            ),
        },
        "reference": {
            "status": "ok" if emt_ref_failure is None else f"failed (exit {emt_ref_failure['exit_code']})",
            "stderr": (
                "none"
                if emt_ref_failure is None
                else single_line(emt_ref_failure["stderr"] or emt_ref_failure["stdout"] or "no message")
            ),
        },
    }

    metadata = {
        "base_sha": args.base_sha,
        "ref_sha": args.ref_sha,
        "compiler_version": compiler_version(args.cxx),
        "build_type": "release (meson -Dbuildtype=release)",
        "openmp_runtime": openmp_runtime(develop_ctx.worker),
        "cpu_model": cpu_model(),
        "lscpu_logical_cores": lscpu.logical_cores,
        "lscpu_physical_cores": lscpu.physical_cores,
        "lscpu_threads_per_core": lscpu.threads_per_core,
        "affinity_logical_cores": affinity_logical_cores,
        "affinity_physical_cores": affinity_physical_cores,
        "thread_counts": thread_counts,
        "samples_per_cell": args.samples,
        "sample_seconds": args.sample_seconds,
        "strong_sizes": strong_sizes,
        "weak_atoms_per_thread": args.weak_atoms_per_thread,
        "openmp_build_evidence": openmp_evidence,
        "neighbour_list_builder": "rgpot::nlist::build (internal rgpot implementation)",
        "neighbour_list_parallel_note": (
            "rgpot::kTeamMinAtoms in CppCore/rgpot/Slices.hpp is 8192; "
            "below that threshold rgpot::nlist::build and the force kernels stay serial"
        ),
    }

    strong_rows: list[dict[str, Any]] = []
    weak_rows: list[dict[str, Any]] = []
    determinism_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for potential in potentials:
        sizes = emt_sizes(strong_sizes) if potential == "emt" else list(strong_sizes)
        # Preflight to avoid partial data for unavailable kernels.
        preflight_failure = worker_failure(
            develop_ctx,
            potential=potential,
            atoms=sizes[0],
            threads=thread_counts[0],
            sample_seconds=args.sample_seconds,
            cpus=cpus_by_threads[thread_counts[0]],
        )
        if preflight_failure is not None:
            reason = single_line(preflight_failure["stderr"] or preflight_failure["stdout"] or "no message")
            skipped.append({"potential": potential, "reason": reason})
            continue

        for atoms in sizes:
            cell = sample_cell(
                develop_ctx,
                potential=potential,
                atoms=atoms,
                thread_counts=thread_counts,
                rounds=args.samples,
                sample_seconds=args.sample_seconds,
                cpus_by_threads=cpus_by_threads,
            )
            base_total = statistics.median(cell[thread_counts[0]]["total_ns"])
            base_force = statistics.median(cell[thread_counts[0]]["force_ns"])
            base_build = statistics.median(cell[thread_counts[0]]["build_ns"])
            bit_identical = True
            max_abs_delta = 0.0

            for threads in thread_counts:
                summary = summarize_cell(cell[threads]["total_ns"], atoms)
                build_median = statistics.median(cell[threads]["build_ns"])
                force_median = statistics.median(cell[threads]["force_ns"])
                speedup = safe_speedup(base_total, summary["median_ns_per_eval"])
                efficiency = speedup / float(threads)
                force_eval_per_s = median_force_eval_per_s(cell[threads]["force_ns"])
                force_speedup = safe_speedup(base_force, force_median)
                build_speedup = safe_speedup(base_build, build_median)
                build_share = (
                    build_median / (build_median + force_median)
                    if (build_median + force_median) > 0.0
                    else 0.0
                )
                affinity_observed = sorted(set(cell[threads]["affinity"]))
                affinity_counts = sorted(set(parse_affinity_count(x) for x in affinity_observed))
                parallel_affinity_observed = sorted(set(cell[threads]["parallel_affinity"]))
                thread_masks: set[str] = set()
                for sample_affinity in parallel_affinity_observed:
                    for mask in parse_parallel_affinity_masks(sample_affinity):
                        thread_masks.add(mask)
                thread_mask_counts = sorted(set(parse_affinity_count(mask) for mask in thread_masks))
                strong_rows.append(
                    {
                        "potential": potential,
                        "atoms": atoms,
                        "threads": threads,
                        "summary": summary,
                        "build_median_ns": build_median,
                        "force_median_ns": force_median,
                        "build_share": build_share,
                        "speedup": speedup,
                        "efficiency": efficiency,
                        "force_eval_per_s_median": force_eval_per_s,
                        "force_speedup": force_speedup,
                        "force_efficiency": force_speedup / float(threads),
                        "build_speedup": build_speedup,
                        "build_efficiency": build_speedup / float(threads),
                        "omp_max_threads_observed": sorted(set(cell[threads]["omp_max_threads"])),
                        "omp_parallel_threads_observed": sorted(
                            set(cell[threads]["omp_parallel_threads"])
                        ),
                        "affinity_observed": affinity_observed,
                        "affinity_cpu_count_observed": affinity_counts,
                        "omp_parallel_affinity_observed": parallel_affinity_observed,
                        "omp_parallel_thread_affinity_masks_observed": sorted(thread_masks),
                        "omp_parallel_thread_affinity_cpu_counts_observed": thread_mask_counts,
                        "smt_label": "yes" if threads > lscpu.physical_cores else "no",
                        "energy_bits": cell[threads]["energy_bits"],
                        "force_hash": cell[threads]["force_hash"],
                    }
                )

            base_energy_bits = cell[thread_counts[0]]["energy_bits"]
            base_force_hash = cell[thread_counts[0]]["force_hash"]
            for threads in thread_counts[1:]:
                if (
                    cell[threads]["energy_bits"] != base_energy_bits
                    or cell[threads]["force_hash"] != base_force_hash
                ):
                    bit_identical = False
                    with tempfile.TemporaryDirectory() as tmpdir:
                        base_dump = Path(tmpdir) / "base.bin"
                        other_dump = Path(tmpdir) / "other.bin"
                        run_worker_once(
                            develop_ctx,
                            potential=potential,
                            atoms=atoms,
                            threads=thread_counts[0],
                            sample_seconds=args.sample_seconds,
                            cpus=cpus_by_threads[thread_counts[0]],
                            dump_force=base_dump,
                        )
                        run_worker_once(
                            develop_ctx,
                            potential=potential,
                            atoms=atoms,
                            threads=threads,
                            sample_seconds=args.sample_seconds,
                            cpus=cpus_by_threads[threads],
                            dump_force=other_dump,
                        )
                        _e0, f0 = parse_force_dump(base_dump)
                        _e1, f1 = parse_force_dump(other_dump)
                        max_abs_delta = max(max_abs_delta, max_abs_force_delta(f0, f1))

            determinism_rows.append(
                {
                    "potential": potential,
                    "atoms": atoms,
                    "bit_identical": bit_identical,
                    "max_abs_force_deviation": max_abs_delta,
                }
            )

        weak_threads = thread_counts
        weak_atoms = {threads: args.weak_atoms_per_thread * threads for threads in weak_threads}
        weak_cell = sample_weak_cell(
            develop_ctx,
            potential=potential,
            atoms_by_threads=weak_atoms,
            thread_counts=weak_threads,
            rounds=args.samples,
            sample_seconds=args.sample_seconds,
            cpus_by_threads=cpus_by_threads,
        )
        weak_base = statistics.median(weak_cell[weak_threads[0]]["total_ns"])
        weak_force_base = statistics.median(weak_cell[weak_threads[0]]["force_ns"])
        for threads in weak_threads:
            atoms = weak_atoms[threads]
            samples = weak_cell[threads]["total_ns"]
            summary = summarize_cell(samples, atoms)
            force_median = statistics.median(weak_cell[threads]["force_ns"])
            weak_rows.append(
                {
                    "potential": potential,
                    "threads": threads,
                    "atoms": atoms,
                    "summary": summary,
                    "weak_efficiency": weak_base / summary["median_ns_per_eval"]
                    if summary["median_ns_per_eval"] > 0.0
                    else 0.0,
                    "force_eval_per_s_median": median_force_eval_per_s(
                        weak_cell[threads]["force_ns"]
                    ),
                    "force_weak_efficiency": safe_speedup(weak_force_base, force_median),
                    "smt_label": "yes" if threads > lscpu.physical_cores else "no",
                }
            )

    morse_rows: list[dict[str, Any]] = []
    for atoms in strong_sizes:
        for threads in thread_counts:
            morse_rows.append(
                compare_morse_cell(
                    develop_ctx=develop_ctx,
                    ref_ctx=ref_ctx,
                    atoms=atoms,
                    threads=threads,
                    rounds=args.samples,
                    sample_seconds=args.sample_seconds,
                    cpus=cpus_by_threads[threads],
                )
            )

    cliffs: list[dict[str, Any]] = []
    for row in strong_rows:
        if row["threads"] == thread_counts[0]:
            continue
        if row["efficiency"] < 0.6:
            cliffs.append(
                {
                    "potential": row["potential"],
                    "atoms": row["atoms"],
                    "threads": row["threads"],
                    "efficiency": row["efficiency"],
                    "likely_cause": infer_cliff_cause(
                        atoms=row["atoms"],
                        threads=row["threads"],
                        build_median_ns=row["build_median_ns"],
                        force_median_ns=row["force_median_ns"],
                    ),
                }
            )

    notable_force_cliffs: list[dict[str, Any]] = []
    sc_row = find_row(strong_rows, potential="sutton-chen", atoms=55296, threads=2)
    if sc_row is not None:
        notable_force_cliffs.append(
            {
                "potential": "sutton-chen",
                "atoms": 55296,
                "threads": 2,
                "force_speedup": sc_row["force_speedup"],
                "cause": (
                    "2-thread force stage is slower than 1-thread "
                    f"({format_float(sc_row['force_speedup'], 4)}x) while build still speeds up; "
                    "this points to force-kernel overhead/cache pressure, not neighbour-list build."
                ),
            }
        )
    ter4 = find_row(strong_rows, potential="tersoff", atoms=55296, threads=4)
    ter8 = find_row(strong_rows, potential="tersoff", atoms=55296, threads=8)
    if ter4 is not None and ter8 is not None:
        increment = ter8["force_speedup"] / ter4["force_speedup"] if ter4["force_speedup"] > 0 else 0.0
        notable_force_cliffs.append(
            {
                "potential": "tersoff",
                "atoms": 55296,
                "threads": 8,
                "force_speedup": ter8["force_speedup"],
                "cause": (
                    "8-thread force scaling plateaus versus 4 threads "
                    f"(4->8 gain {format_float(increment, 4)}x); "
                    "build continues to scale, so the plateau is in the force kernel."
                ),
            }
        )

    payload = {
        "metadata": metadata,
        "strong_scaling": strong_rows,
        "weak_scaling": weak_rows,
        "morse_ref_vs_develop": morse_rows,
        "determinism": determinism_rows,
        "scaling_cliffs": cliffs,
        "notable_force_cliffs": notable_force_cliffs,
        "skipped": skipped,
        "emt_reproducer": emt_reproducer,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(
        path=out_md,
        metadata=metadata,
        strong=strong_rows,
        weak=weak_rows,
        morse=morse_rows,
        determinism=determinism_rows,
        cliffs=cliffs,
        skipped=skipped,
        emt_reproducer=emt_reproducer,
        notable_force_cliffs=notable_force_cliffs,
    )


if __name__ == "__main__":
    main()
