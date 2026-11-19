# Copyright (c) anneal developers
"""Head-to-head of anneal against the standalone minima-hopping package.

The package is the Goedecker-group distribution ``minimahopping`` (Gubler
et al.), not ``ase.optimize.minimahopping``. Both arms call the same
Lennard-Jones, Morse, or Gupta potential: anneal through the existing
``h2h_gmin_proof``, ``h2h_morse``, and ``h2h_metal`` examples, and the
package through ``mhpkg_eval`` behind one ASE calculator. One
``value_and_gradient`` is one call.

See ``bench/ase-mhpkg/README.md`` for the unit rule, the parameter
values, and the reason rigid TIP4P is not run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"
HIT_TOL = 1.0e-3
CAP_S = 7200.0
OUTER_S = 7800.0
CANONICAL_SEEDS = list(range(2000, 2012))
PIN_PATH = ROOT / "scripts" / "ase_panel" / "mhpkg_pin.json"
TIP4P_CSV = ROOT / "data" / "tip4p_ccd.csv"
ROWS_H2H = ROOT / "bench" / "ase-mhpkg" / "rows.csv"
ROWS_AA = ROOT / "bench" / "ase-mhpkg" / "aa_rows.csv"
WORK = ROOT / "bench" / "ase-mhpkg" / "work"

THREADS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMEXPR_MAX_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "OMP_PROC_BIND": "TRUE",
    "OMP_PLACES": "cores",
    "OMP_DYNAMIC": "FALSE",
    "LOKY_MAX_CPU_COUNT": "1",
}

COLUMNS = [
    "cell",
    "panel",
    "arm",
    "seed",
    "slot",
    "budget",
    "hit",
    "calls",
    "calls_to_hit",
    "best",
    "wall_s",
    "inside_s",
    "outside_s",
    "outside_per_call_s",
    "core",
    "affinity",
    "machine_id",
    "stop",
    "cap_s",
    "code_revision",
    "package_version",
    "package_revision",
    "ase_version",
    "session",
    "command",
]

# Physics parameters are the cluster example's overrides plus the
# constructor defaults at the pinned revision. Output switches are
# separate and are listed in the README.
RAW_PARAMETERS = {
    "T0": 500.0,
    "dt0": 0.1,
    "Ediff0": 0.1,
    "beta_decrease": 1.0 / 1.05,
    "beta_increase": 1.05,
    "alpha_accept": 1.0 / 1.05,
    "alpha_reject": 1.05,
    "n_soft": 20,
    "soften_positions": 1.0e-3,
    "soften_lattice": 1.0e-3,
    "n_S_orbitals": 1,
    "n_P_orbitals": 0,
    "width_cutoff": 4.0,
    "dt_min": 1.0e-4,
    "mdmin": 2,
    "md_max_steps": 10000,
    "fmax": 0.005,
    "energy_threshold": 0.001,
    "fingerprint_threshold": 5.0e-2,
    "enhanced_feedback": False,
    "margin": 0.3,
    "opt_max_steps": 10000,
    "use_MPI": False,
    "verbose_output": False,
    "write_graph_output": False,
    "collect_md_data": False,
    "new_start": True,
    "logLevel": 30,
    "fixed_cell_simulation": False,
}

CELLS = {
    "lj38": ("lj", 38, 400_000, -173.928427, "h2h_gmin_proof"),
    "lj75": ("lj", 75, 4_000_000, -397.492331, "h2h_gmin_proof"),
    "lj98": ("lj", 98, 4_000_000, -543.665361, "h2h_gmin_proof"),
    "lj104": ("lj", 104, 4_000_000, -582.086642, "h2h_gmin_proof"),
    "m80r6": ("morse", 80, 4_000_000, -378.333471, "h2h_morse"),
    "m80r14": ("morse", 80, 4_000_000, -340.811371, "h2h_morse"),
    "au38": ("gupta", 38, 400_000, -130.54001708, "h2h_metal"),
}

TIP4P_BUDGET = 2_000_000
TIP4P_REASON = (
    "The minimahopping package moves Cartesian ASE atoms. Its documented "
    "constraint support is ASE FixAtoms, with an explicit warning that other "
    "ASE constraints are untested. biomode is imported by the softening step "
    "and never called, and it does not integrate a rigid body. Rigid TIP4P "
    "in this repository is a 6-coordinate angle-axis chart per molecule "
    "(GminTip4p). There is no supported rigid-molecule interface, so the cell "
    "is not run."
)

ENERGY_KEYS = ("Ediff0", "energy_threshold")
LENGTH_KEYS = ("soften_positions", "soften_lattice", "width_cutoff")
FORCE_KEYS = ("fmax",)
TIME_KEYS = ("dt0", "dt_min")
TEMP_KEYS = ("T0",)


def die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def status_field(name: str) -> str:
    text = Path("/proc/self/status").read_text()
    for line in text.splitlines():
        if line.startswith(name + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def processor() -> str:
    stat = Path("/proc/self/stat").read_text()
    rest = stat.rsplit(")", 1)[1].split()
    return rest[36]


def machine_hash() -> str:
    raw = Path("/etc/machine-id").read_text().strip().encode()
    return hashlib.sha256(raw).hexdigest()


def ensure_seat() -> None:
    changed = False
    if os.environ.get("LJOS_SEAT") != "benchbot":
        os.environ["LJOS_SEAT"] = "benchbot"
        changed = True
    for key, value in THREADS.items():
        if os.environ.get(key) != value:
            os.environ[key] = value
            changed = True
    affinity = status_field("Cpus_allowed_list")
    if affinity != "0":
        os.execvp("taskset", ["taskset", "-c", "0", sys.executable, *sys.argv])
    if changed:
        os.execv(sys.executable, [sys.executable, *sys.argv])
    if status_field("Cpus_allowed_list") != "0":
        die("process is not confined to cpu 0")
    if processor() != "0":
        die("process is not running on cpu 0")


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_")):
            del env[key]
    env.update(THREADS)
    env["LJOS_SEAT"] = "benchbot"
    if extra:
        env.update(extra)
    return env


def pin() -> dict:
    return json.loads(PIN_PATH.read_text())


def type7(values: list[float], p: float) -> float:
    """Hyndman-Fan type 7."""
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("type 7 needs a sample")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def iqr(values: list[float]) -> tuple[float, float, float, float]:
    q1 = type7(values, 0.25)
    mid = type7(values, 0.5)
    q3 = type7(values, 0.75)
    return mid, q1, q3, q3 - q1


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided exact p, summing the tables no more likely than the observed."""
    hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    total = sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7))
    return min(1.0, total)


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    """U of ``left`` and the two-sided p, normal approximation with the tie
    and continuity corrections. Every value tied gives p = 1."""
    n1, n2 = len(left), len(right)
    if n1 < 1 or n2 < 1:
        raise ValueError("mann-whitney needs two non-empty samples")
    pooled = sorted(left + right)
    n = n1 + n2
    rank: dict[float, float] = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    u = sum(rank[v] for v in left) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return u, 1.0
    dev = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    return u, min(1.0, math.erfc(dev / math.sqrt(var) / math.sqrt(2.0)))


def hit_tol() -> float:
    text = (ROOT / "scripts" / "target_tol.py").read_text()
    match = re.search(r"^DEFAULT_HIT_TOL\s*(?::[^=]+)?=\s*([0-9eE.+-]+)", text, re.M)
    if match is None:
        die("DEFAULT_HIT_TOL is missing from scripts/target_tol.py")
    value = float(match.group(1))
    if abs(value - HIT_TOL) > 1e-15:
        die(f"DEFAULT_HIT_TOL is {value}, expected {HIT_TOL}")
    return value


def tip4p_table() -> list[tuple[int, float]]:
    rows = []
    with TIP4P_CSV.open() as handle:
        for record in csv.DictReader(handle):
            n = int(record["n"])
            if 8 <= n <= 21:
                rows.append((n, float(record["energy"])))
    if [n for n, _energy in rows] != list(range(8, 22)):
        die("data/tip4p_ccd.csv does not list n=8..21")
    return rows


def scales(kind: str) -> dict[str, float]:
    s_e = 1.0
    s_l = 1.0
    if kind == "tip4p":
        from ase import units

        s_e = float(units.eV / (units.kJ / units.mol))
    s_f = s_e / s_l
    return {
        "s_e": s_e,
        "s_l": s_l,
        "s_f": s_f,
        "s_t": s_e,
        "s_dt": 1.0 / math.sqrt(s_f),
    }


def parameters(kind: str) -> dict:
    factor = scales(kind)
    scaled = dict(RAW_PARAMETERS)
    for key in ENERGY_KEYS:
        scaled[key] = RAW_PARAMETERS[key] * factor["s_e"]
    for key in LENGTH_KEYS:
        scaled[key] = RAW_PARAMETERS[key] * factor["s_l"]
    for key in FORCE_KEYS:
        scaled[key] = RAW_PARAMETERS[key] * factor["s_f"]
    for key in TIME_KEYS:
        scaled[key] = RAW_PARAMETERS[key] * factor["s_dt"]
    for key in TEMP_KEYS:
        scaled[key] = RAW_PARAMETERS[key] * factor["s_t"]
    return scaled


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            lo, hi = int(left), int(right)
            if hi < lo:
                die(f"seed range {part} is empty")
            seeds.extend(range(lo, hi + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        die("no seeds")
    if len(seeds) != len(set(seeds)):
        die("duplicate seeds")
    return seeds


def seed_index(seed: int, seeds: list[int]) -> int:
    if seed in CANONICAL_SEEDS:
        return seed - CANONICAL_SEEDS[0]
    return seeds.index(seed)


def arm_order(index: int, arms: tuple[str, str]) -> list[tuple[int, str]]:
    if index % 2 == 0:
        return [(0, arms[0]), (1, arms[1])]
    return [(0, arms[1]), (1, arms[0])]


def panel_arms(panel: str) -> tuple[str, str]:
    if panel == "h2h":
        return ("anneal", "mhpkg")
    if panel == "aa":
        return ("aa0", "aa1")
    die(f"unknown panel {panel}")
    raise AssertionError


def git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def cat_file(rev: str) -> None:
    subprocess.run(["git", "cat-file", "-e", rev], cwd=ROOT, check=True)


def example_bin(name: str) -> Path:
    proc = subprocess.run(
        ["cargo", "+1.88.0", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    target = Path(json.loads(proc.stdout)["target_directory"])
    for path in (target / "release" / "examples" / name, target / "release" / name):
        if path.is_file():
            return path
    die(f"missing release example {name}")
    raise AssertionError


def run_checked(cmd: list[str], env: dict[str, str], timeout: float) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        die(f"{' '.join(cmd)} failed ({proc.returncode})\n{proc.stderr}")
    return proc


def assert_budget(binary: Path, cell: str, budget: int, reference: float) -> None:
    proc = run_checked([str(binary), "budget", cell], child_env(), 60)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    got_budget = int(lines[0])
    got_reference = float(lines[1])
    if got_budget != budget or abs(got_reference - reference) > 1e-8:
        die(
            f"{cell} budget command returned {got_budget} {got_reference}, "
            f"table has {budget} {reference}"
        )


def reproduce_command(cell: str, panel: str, arm: str, seed: int, revision: str) -> str:
    thread = " ".join(f"{key}={value}" for key, value in THREADS.items())
    return (
        f"git checkout {revision} && "
        "cargo +1.88.0 build --release --locked --example h2h_gmin_proof "
        "--example h2h_morse --example h2h_metal --example mhpkg_eval && "
        "python3 -m venv .mhpkg-venv && "
        ".mhpkg-venv/bin/pip install -r scripts/ase_panel/mhpkg-requirements.txt && "
        f"export {thread} LJOS_SEAT=benchbot && "
        "taskset -c 0 .mhpkg-venv/bin/python scripts/ase_panel/mhpkg.py run "
        f"--cells {cell} --seeds {seed} --panel {panel} --arms {arm}"
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def have_row(path: Path, cell: str, panel: str, arm: str, seed: int, revision: str) -> bool:
    for row in read_rows(path):
        if (
            row["cell"] == cell
            and row["panel"] == panel
            and row["arm"] == arm
            and int(row["seed"]) == seed
            and row["code_revision"] == revision
        ):
            return True
    return False


def session_of(work: Path, fresh: bool) -> tuple[str, str]:
    path = work / "session"
    revision = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    if path.is_file() and not fresh:
        fields = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        if fields.get("revision") != revision:
            die("work session revision differs from HEAD; pass --fresh-session")
        return fields["id"], revision
    work.mkdir(parents=True, exist_ok=True)
    ident = hashlib.sha256(
        f"{machine_hash()}:{os.getpid()}:{time.time_ns()}".encode()
    ).hexdigest()[:16]
    path.write_text(f"id={ident}\nrevision={revision}\ntree={tree}\n")
    return ident, revision


def place(kind: str) -> str:
    if kind == "gupta":
        return "Au"
    from ase import units
    from minimahopping.mh.periodictable import get_rcov_dict

    distance = 2.0 ** (1.0 / 6.0) if kind == "lj" else 1.0
    best: tuple[float, str] | None = None
    for symbol, rcov in get_rcov_dict().items():
        bond = 2.0 * float(rcov) * float(units.Bohr)
        if bond + 1e-12 >= distance and (best is None or bond < best[0]):
            best = (bond, symbol)
    if best is None or best[1] != "Ne":
        die(f"element rule for {kind} returned {best}")
    return "Ne"


def install_velocity_shim() -> None:
    import ase.md.velocitydistribution as velocity
    from ase.parallel import DummyMPI

    original = velocity.MaxwellBoltzmannDistribution

    def maxwell(atoms, *, temperature_K, communicator=None, comm=None, force_temp=False, rng=None):
        if communicator is not None and comm is None:
            comm = DummyMPI() if communicator == "serial" else communicator
        if comm is None:
            comm = velocity.world
        return original(
            atoms,
            temperature_K=temperature_K,
            comm=comm,
            force_temp=force_temp,
            rng=rng,
        )

    velocity.MaxwellBoltzmannDistribution = maxwell


def load_science() -> tuple[str, str, str]:
    record = pin()
    install_velocity_shim()
    import ase
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() != 1 or MPI.COMM_WORLD.Get_rank() != 0:
        die("minimahopping must run as one MPI rank")
    if ase.__version__ != record["ase"]:
        die(f"ASE {ase.__version__} does not match pin {record['ase']}")
    import importlib.metadata as metadata

    dist = metadata.distribution("minimahopping")
    version = dist.version
    if version != record["version"]:
        die(f"minimahopping {version} does not match pin {record['version']}")
    direct = dist.read_text("direct_url.json")
    if direct is None or record["revision"] not in direct:
        die("installed minimahopping revision does not match the pin")
    sq = metadata.distribution("sqnm")
    sq_url = sq.read_text("direct_url.json") or ""
    if record["sqnm_revision"] not in sq_url:
        die("installed sqnm revision does not match the pin")
    import minimahopping.minhop as minhop

    minhop.MaxwellBoltzmannDistribution = __import__(
        "ase.md.velocitydistribution", fromlist=["MaxwellBoltzmannDistribution"]
    ).MaxwellBoltzmannDistribution
    return version, record["revision"], ase.__version__


class StopSearch(BaseException):
    def __init__(self, reason: str) -> None:
        self.reason = reason


class EvalServer:
    def __init__(self, binary: Path, cell: str, log_path: Path) -> None:
        self.log = log_path.open("w")
        self.proc = subprocess.Popen(
            [str(binary), "serve", cell],
            cwd=ROOT,
            env=child_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )

    def eval(self, coords) -> tuple[float, float, list[float]]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        payload = " ".join(format(float(value), ".17g") for value in coords)
        self.proc.stdin.write(payload + "\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline()
        if not reply:
            die("potential server closed its output")
        parts = reply.split()
        energy = float(parts[0])
        seconds = float(parts[1])
        gradient = [float(value) for value in parts[2:]]
        return energy, seconds, gradient

    def close(self) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
                self.proc.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def preflight(binary: Path, server_bin: Path, cell: str, coords: Path, work: Path) -> None:
    proc = run_checked([str(binary), "energy", cell, str(coords)], child_env(), 120)
    expected = float(proc.stdout.splitlines()[-1])
    log_path = work / f"preflight-{cell}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    server = EvalServer(server_bin, cell, log_path)
    try:
        numbers = [float(word) for word in coords.read_text().split()]
        got, _seconds, gradient = server.eval(numbers)
    finally:
        server.close()
    if abs(got - expected) > 1e-8:
        die(f"{cell} server energy {got} disagrees with {binary.name} energy {expected}")
    if len(gradient) != len(numbers):
        die(f"{cell} server gradient length {len(gradient)}")


def parse_anneal(stdout: str) -> tuple[dict[str, str], str]:
    header: list[str] | None = None
    data: dict[str, str] | None = None
    stop = ""
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("arm "):
            continue
        if line.startswith("h2h_stop="):
            stop = line.split("=", 1)[1]
            continue
        if line.startswith("system,"):
            header = line.split(",")
            continue
        if header is not None and data is None and len(line.split(",")) == len(header):
            data = dict(zip(header, line.split(",")))
    if data is None or not stop:
        die("anneal output is missing the result line or h2h_stop")
    return data, stop


def ratio(outside: float, calls: int) -> str:
    if calls < 1:
        return ""
    return f"{outside / calls:.9f}"


def timing(wall: float, inside: float, calls: int) -> tuple[str, str, str, str]:
    outside = max(0.0, wall - inside)
    return f"{wall:.6f}", f"{inside:.6f}", f"{outside:.6f}", ratio(outside, calls)


def run_anneal(
    binary: Path,
    cell: str,
    seed: int,
    coords: Path,
    log_path: Path,
) -> dict[str, str]:
    env = child_env({"H2H_WALL_S": str(int(CAP_S)), "H2H_COORDS": str(coords.relative_to(ROOT))})
    proc = run_checked([str(binary), "run", cell, str(seed)], env, OUTER_S)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout)
    data, stop = parse_anneal(proc.stdout)
    calls = int(data["charged"])
    hit = data["hit"].lower() == "true"
    wall = float(data["wall_s"])
    inside = float(data["pot_s"])
    wall_s, inside_s, outside_s, per_call = timing(wall, inside, calls)
    calls_to_hit = data["evals_to_hit"] if hit else ""
    if hit and int(calls_to_hit) > calls:
        die(f"{cell} seed {seed} calls-to-hit exceed charged calls")
    return {
        "budget": data["budget"],
        "hit": "true" if hit else "false",
        "calls": str(calls),
        "calls_to_hit": calls_to_hit,
        "best": f"{float(data['best']):.12f}",
        "wall_s": wall_s,
        "inside_s": inside_s,
        "outside_s": outside_s,
        "outside_per_call_s": per_call,
        "stop": stop,
    }


def run_mhpkg(
    server_bin: Path,
    cell: str,
    kind: str,
    n: int,
    seed: int,
    budget: int,
    reference: float,
    coords: Path,
    run_dir: Path,
) -> dict[str, str]:
    install_velocity_shim()
    import numpy as np
    from ase import Atoms
    import minimahopping.minhop as minhop

    minhop.MaxwellBoltzmannDistribution = __import__(
        "ase.md.velocitydistribution", fromlist=["MaxwellBoltzmannDistribution"]
    ).MaxwellBoltzmannDistribution
    Minimahopping = minhop.Minimahopping

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    numbers = np.array([float(word) for word in coords.read_text().split()], dtype=float)
    if numbers.size != 3 * n:
        die(f"{cell} coords have {numbers.size} numbers")
    params = parameters(kind)
    meter = {
        "calls": 0,
        "inside": 0.0,
        "best": None,
        "lowest": None,
        "hit": False,
        "calls_to_hit": "",
        "stop": "",
        "t0": 0.0,
    }
    server = EvalServer(server_bin, cell, run_dir / "server.log")

    from ase.calculators.calculator import Calculator

    class AnnealCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms, properties, system_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            if meter["stop"]:
                raise StopSearch(meter["stop"])
            if time.perf_counter() - meter["t0"] >= CAP_S:
                meter["stop"] = "wall"
                raise StopSearch("wall")
            positions = atoms.get_positions().reshape(-1)
            energy, seconds, gradient = server.eval(positions)
            meter["calls"] += 1
            meter["inside"] += seconds
            grad = np.asarray(gradient, dtype=float)
            self.results["energy"] = float(energy)
            self.results["forces"] = (-grad).reshape((-1, 3))
            if meter["lowest"] is None or energy < meter["lowest"]:
                meter["lowest"] = float(energy)
            if float(np.max(np.abs(grad))) <= params["fmax"]:
                if meter["best"] is None or energy < meter["best"]:
                    meter["best"] = float(energy)
                if (not meter["hit"]) and energy <= reference + HIT_TOL:
                    meter["hit"] = True
                    meter["calls_to_hit"] = str(meter["calls"])
                    meter["stop"] = "hit"
                    raise StopSearch("hit")
            if meter["calls"] >= budget:
                meter["stop"] = "budget"
                raise StopSearch("budget")
            if time.perf_counter() - meter["t0"] >= CAP_S:
                meter["stop"] = "wall"
                raise StopSearch("wall")

    atoms = Atoms(
        symbols=[place(kind)] * n,
        positions=numbers.reshape((-1, 3)),
        pbc=False,
    )
    atoms.calc = AnnealCalculator()
    np.random.seed(seed)
    previous = os.getcwd()
    os.chdir(run_dir)
    started = time.perf_counter()
    meter["t0"] = started
    try:
        with Minimahopping(atoms, **params) as hopper:
            try:
                hopper(totalsteps=10**12)
            except StopSearch as stop:
                meter["stop"] = stop.reason
            if not meter["stop"]:
                meter["stop"] = "return"
    finally:
        wall = time.perf_counter() - started
        os.chdir(previous)
        server.close()
    best = meter["best"] if meter["best"] is not None else meter["lowest"]
    if best is None:
        die(f"{cell} seed {seed} made no potential call")
    calls = int(meter["calls"])
    wall_s, inside_s, outside_s, per_call = timing(wall, meter["inside"], calls)
    return {
        "budget": str(budget),
        "hit": "true" if meter["hit"] else "false",
        "calls": str(calls),
        "calls_to_hit": meter["calls_to_hit"],
        "best": f"{float(best):.12f}",
        "wall_s": wall_s,
        "inside_s": inside_s,
        "outside_s": outside_s,
        "outside_per_call_s": per_call,
        "stop": meter["stop"],
    }


def measurement_row(
    cell: str,
    panel: str,
    arm: str,
    seed: int,
    slot: int,
    result: dict[str, str],
    revision: str,
    versions: tuple[str, str, str],
    session: str,
) -> dict[str, str]:
    package_version, package_revision, ase_version = versions
    affinity = status_field("Cpus_allowed_list")
    if affinity != "0" or processor() != "0":
        die("row would not be confined to cpu 0")
    return {
        "cell": cell,
        "panel": panel,
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "budget": result["budget"],
        "hit": result["hit"],
        "calls": result["calls"],
        "calls_to_hit": result["calls_to_hit"],
        "best": result["best"],
        "wall_s": result["wall_s"],
        "inside_s": result["inside_s"],
        "outside_s": result["outside_s"],
        "outside_per_call_s": result["outside_per_call_s"],
        "core": processor(),
        "affinity": affinity,
        "machine_id": machine_hash(),
        "stop": result["stop"],
        "cap_s": str(int(CAP_S)),
        "code_revision": revision,
        "package_version": package_version,
        "package_revision": package_revision,
        "ase_version": ase_version,
        "session": session,
        "command": reproduce_command(cell, panel, arm, seed, revision),
    }


def log_line(work: Path, row: dict[str, str]) -> None:
    path = work / "logs" / "campaign.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(
            f"{row['cell']} {row['panel']} {row['arm']} seed={row['seed']} "
            f"slot={row['slot']} hit={row['hit']} calls={row['calls']} "
            f"best={row['best']} wall={row['wall_s']} stop={row['stop']}\n"
        )


def campaign(args: argparse.Namespace) -> None:
    hit_tol()
    panel = args.panel
    arms = panel_arms(panel)
    selected = tuple(part for part in args.arms.split(",") if part) if args.arms else arms
    for arm in selected:
        if arm not in arms:
            die(f"arm {arm} is not in panel {panel}")
    cells = [part.strip() for part in args.cells.split(",") if part.strip()]
    seeds = parse_seeds(args.seeds)
    for name in cells:
        if name.startswith("tip4p"):
            die(f"{name} is unsupported: {TIP4P_REASON}")
        if name not in CELLS:
            die(f"unknown cell {name}")
    work = Path(args.work) if args.work else WORK
    if not work.is_absolute():
        work = ROOT / work
    dest = Path(args.rows) if args.rows else (ROWS_AA if panel == "aa" else ROWS_H2H)
    if not dest.is_absolute():
        dest = ROOT / dest
    os.environ["NUMBA_CACHE_DIR"] = str(work / "numba-cache")
    (work / "numba-cache").mkdir(parents=True, exist_ok=True)
    versions = load_science()
    session, revision = session_of(work, args.fresh_session)
    server_bin = example_bin("mhpkg_eval")
    seen_preflight: set[str] = set()
    for name in cells:
        kind, n, budget, reference, binary_name = CELLS[name]
        binary = example_bin(binary_name)
        assert_budget(binary, name, budget, reference)
        spec = run_checked([str(server_bin), "spec", name], child_env(), 60)
        if f"budget={budget}" not in spec.stdout or f"n={n}" not in spec.stdout:
            die(f"mhpkg_eval spec disagrees with the {name} table: {spec.stdout.strip()}")
        for seed in seeds:
            index = seed_index(seed, seeds)
            coords = work / "coords" / name / f"{seed}.xyz"
            if not coords.is_file():
                coords.parent.mkdir(parents=True, exist_ok=True)
                run_checked(
                    [str(binary), "coords", name, str(seed), str(coords.relative_to(ROOT))],
                    child_env(),
                    60,
                )
            if name not in seen_preflight:
                preflight(binary, server_bin, name, coords, work / "logs")
                seen_preflight.add(name)
            for slot, arm in arm_order(index, arms):
                if arm not in selected:
                    continue
                if not args.rerun and have_row(dest, name, panel, arm, seed, revision):
                    continue
                if arm == "mhpkg":
                    result = run_mhpkg(
                        server_bin,
                        name,
                        kind,
                        n,
                        seed,
                        budget,
                        reference,
                        coords,
                        work / "runs" / panel / name / arm / str(seed),
                    )
                else:
                    result = run_anneal(
                        binary,
                        name,
                        seed,
                        coords,
                        work / "logs" / f"{panel}-{name}-{arm}-{seed}.log",
                    )
                if int(result["budget"]) != budget:
                    die(f"{name} reported budget {result['budget']}")
                row = measurement_row(
                    name, panel, arm, seed, slot, result, revision, versions, session
                )
                append_row(dest, row)
                log_line(work, row)
                print(
                    f"{name} {arm} seed {seed} slot {slot} hit {row['hit']} "
                    f"calls {row['calls']} wall {row['wall_s']} stop {row['stop']}",
                    file=sys.stderr,
                )


def verify(args: argparse.Namespace) -> None:
    panel = args.panel
    dest = Path(args.rows) if args.rows else (ROWS_AA if panel == "aa" else ROWS_H2H)
    if not dest.is_absolute():
        dest = ROOT / dest
    matches = [
        row
        for row in read_rows(dest)
        if row["cell"] == args.cell
        and row["arm"] == args.arm
        and int(row["seed"]) == args.seed
        and row["panel"] == panel
    ]
    if len(matches) != 1:
        die(f"expected one stored row, found {len(matches)}")
    stored = matches[0]
    diff = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            stored["code_revision"],
            "--",
            "examples/h2h_gmin_proof.rs",
            "examples/h2h_morse.rs",
            "examples/h2h_metal.rs",
            "examples/mhpkg_eval.rs",
            "scripts/ase_panel/mhpkg.py",
            "scripts/ase_panel/mhpkg_pin.json",
        ],
        cwd=ROOT,
        check=False,
    )
    if diff.returncode != 0:
        die("the worktree of the harness differs from the row revision")
    scratch = ROOT / "bench" / "ase-mhpkg" / "work" / "verify"
    if scratch.exists():
        shutil.rmtree(scratch)
    rerun = argparse.Namespace(
        cells=args.cell,
        seeds=str(args.seed),
        panel=panel,
        arms=args.arm,
        rows=str(scratch / "rows.csv"),
        work=str(scratch),
        fresh_session=True,
        rerun=True,
    )
    campaign(rerun)
    got = read_rows(scratch / "rows.csv")
    if len(got) != 1:
        die(f"verify rerun wrote {len(got)} rows")
    fresh = got[0]
    for key in ("hit", "calls", "calls_to_hit", "stop"):
        if fresh[key] != stored[key]:
            die(f"verify {key} {fresh[key]} != stored {stored[key]}")
    if abs(float(fresh["best"]) - float(stored["best"])) > 1e-8:
        die(f"verify best {fresh['best']} != stored {stored['best']}")
    print(f"verify {args.cell} {args.arm} seed {args.seed} matches")


def lock_packages() -> list[tuple[str, str, str]]:
    want = {"anneal-core", "rgmin", "rgpot-core", "rgpot-pair", "readcon-core"}
    found: list[tuple[str, str, str]] = []
    name = version = source = ""
    for line in (ROOT / "Cargo.lock").read_text().splitlines():
        if line.startswith("name = "):
            if name in want:
                found.append((name, version, source))
            name = line.split("=", 1)[1].strip().strip('"')
            version = ""
            source = ""
        elif line.startswith("version = "):
            version = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("source = "):
            source = line.split("=", 1)[1].strip().strip('"')
    if name in want:
        found.append((name, version, source))
    return found


def grouped(rows: list[dict[str, str]], panel: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["panel"] != panel:
            continue
        groups.setdefault((row["cell"], row["arm"]), []).append(row)
    return groups


def complete(groups: dict[tuple[str, str], list[dict[str, str]]], cell: str, arms: tuple[str, str]) -> bool:
    for arm in arms:
        rows = groups.get((cell, arm), [])
        seeds = sorted(int(row["seed"]) for row in rows)
        if seeds != CANONICAL_SEEDS:
            return False
    return True


def fmt(value: float) -> str:
    return f"{value:.6g}"


def write_report(args: argparse.Namespace) -> None:
    h2h = read_rows(Path(args.rows) if args.rows else ROWS_H2H)
    aa = read_rows(Path(args.aa_rows) if args.aa_rows else ROWS_AA)
    record = pin()
    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    cat_file(BASE)
    cat_file(f"{BASE}^{{tree}}")
    cat_file(head)
    cat_file(f"{head}^{{tree}}")
    revisions = sorted({row["code_revision"] for row in h2h + aa})
    for revision in revisions:
        cat_file(revision)
        cat_file(f"{revision}^{{tree}}")
    h2h_groups = grouped(h2h, "h2h")
    aa_groups = grouped(aa, "aa")
    cells = sorted({cell for cell, _arm in h2h_groups})
    done = [cell for cell in cells if complete(h2h_groups, cell, ("anneal", "mhpkg"))]
    m = len(done)
    lines = [
        "# Standalone minima hopping against anneal",
        "",
        f"Package `minimahopping` {record['version']} revision `{record['revision']}`.",
        f"ASE {record['ase']}. sqnm revision `{record['sqnm_revision']}`.",
        f"GitLab archive sha256 `{record['gitlab_archive_sha256']}`.",
        f"sdist sha256 `{record['sdist_sha256']}`.",
        f"wheel sha256 `{record['wheel_sha256']}`.",
        "",
        record["pypi_note"],
        "",
        record["ase_note"],
        "",
        f"Base `{BASE}` tree `{git('rev-parse', BASE + '^{tree}')}`.",
        f"Rows revision `{head}` tree `{tree}`.",
    ]
    for revision in revisions:
        lines.append(
            f"Harness revision `{revision}` tree `{git('rev-parse', revision + '^{tree}')}`."
        )
    lines.extend(["", "Locked stack:", ""])
    for name, version, source in lock_packages():
        lines.append(f"- {name} {version} {source}")
    lines.extend(
        [
            "",
            "A hit is a quenched energy at or below the Cambridge minimum plus 0.001.",
            "Medians and quartiles are Hyndman-Fan type 7.",
            "Calls and wall used below are the stored values, including a capped seed at the calls and wall it reached.",
            "Fisher p is the two-sided exact test on hit counts.",
            "Mann-Whitney p is the two-sided normal approximation with the tie and continuity corrections, anneal as the left sample.",
            f"Bonferroni uses m={m}, the number of complete head-to-head cells, separately for each test.",
            "The same-binary Lennard-Jones 38 pair is the noise band and is not in that family.",
            "",
            "## Head to head",
            "",
        ]
    )
    summary_rows = []
    for cell in cells:
        lines.append(f"### {cell}")
        lines.append("")
        for arm in ("anneal", "mhpkg"):
            rows = h2h_groups.get((cell, arm), [])
            if not rows:
                lines.append(f"{arm}: no rows.")
                continue
            calls = [float(row["calls"]) for row in rows]
            wall = [float(row["wall_s"]) for row in rows]
            inside = [float(row["inside_s"]) for row in rows]
            outside = [float(row["outside_s"]) for row in rows]
            per = [float(row["outside_per_call_s"]) for row in rows if row["outside_per_call_s"]]
            hits = sum(row["hit"] == "true" for row in rows)
            mid_c, q1_c, q3_c, iqr_c = iqr(calls)
            mid_w, q1_w, q3_w, iqr_w = iqr(wall)
            lines.append(
                f"{arm}: n={len(rows)} hits={hits} "
                f"calls median {fmt(mid_c)} IQR {fmt(iqr_c)} "
                f"wall median {fmt(mid_w)} IQR {fmt(iqr_w)} "
                f"inside median {fmt(iqr(inside)[0])} "
                f"outside median {fmt(iqr(outside)[0])} "
                f"outside per call median {fmt(iqr(per)[0]) if per else 'na'}."
            )
            summary_rows.append(
                {
                    "cell": cell,
                    "panel": "h2h",
                    "arm": arm,
                    "n": str(len(rows)),
                    "hits": str(hits),
                    "calls_median": f"{mid_c:.10g}",
                    "calls_q1": f"{q1_c:.10g}",
                    "calls_q3": f"{q3_c:.10g}",
                    "calls_iqr": f"{iqr_c:.10g}",
                    "wall_median": f"{mid_w:.10g}",
                    "wall_q1": f"{q1_w:.10g}",
                    "wall_q3": f"{q3_w:.10g}",
                    "wall_iqr": f"{iqr_w:.10g}",
                }
            )
        if cell in done:
            left = h2h_groups[(cell, "anneal")]
            right = h2h_groups[(cell, "mhpkg")]
            hits_a = sum(row["hit"] == "true" for row in left)
            hits_b = sum(row["hit"] == "true" for row in right)
            fp = fisher(hits_a, len(left), hits_b, len(right))
            u_calls, p_calls = mann_whitney(
                [float(row["calls"]) for row in left],
                [float(row["calls"]) for row in right],
            )
            u_wall, p_wall = mann_whitney(
                [float(row["wall_s"]) for row in left],
                [float(row["wall_s"]) for row in right],
            )
            lines.append(
                f"Fisher p {fp:.6g} adjusted {min(1.0, m * fp):.6g}. "
                f"Mann-Whitney calls U {u_calls:.6g} p {p_calls:.6g} "
                f"adjusted {min(1.0, m * p_calls):.6g}. "
                f"Mann-Whitney wall U {u_wall:.6g} p {p_wall:.6g} "
                f"adjusted {min(1.0, m * p_wall):.6g}."
            )
            counts: dict[tuple[str, str], int] = {}
            for row in left + right:
                counts[(row["slot"], row["arm"])] = counts.get((row["slot"], row["arm"]), 0) + 1
            rendered = ", ".join(
                f"slot {slot} {arm} {counts.get((slot, arm), 0)}"
                for slot in ("0", "1")
                for arm in ("anneal", "mhpkg")
            )
            lines.append(f"Arm order counts: {rendered}.")
            if any(counts.get((slot, arm), 0) != 6 for slot in ("0", "1") for arm in ("anneal", "mhpkg")):
                lines.append("Arm order is not balanced.")
        else:
            lines.append("Incomplete, omitted from Bonferroni.")
        lines.append("")
    lines.extend(["## Same binary, Lennard-Jones 38", ""])
    aa_cells = sorted({cell for cell, _arm in aa_groups})
    for cell in aa_cells:
        for arm in ("aa0", "aa1"):
            rows = aa_groups.get((cell, arm), [])
            if not rows:
                continue
            calls = [float(row["calls"]) for row in rows]
            wall = [float(row["wall_s"]) for row in rows]
            mid_c, _q1, _q3, iqr_c = iqr(calls)
            mid_w, _q1w, _q3w, iqr_w = iqr(wall)
            hits = sum(row["hit"] == "true" for row in rows)
            lines.append(
                f"{cell} {arm}: n={len(rows)} hits={hits} "
                f"calls median {fmt(mid_c)} IQR {fmt(iqr_c)} "
                f"wall median {fmt(mid_w)} IQR {fmt(iqr_w)}."
            )
        if complete(aa_groups, cell, ("aa0", "aa1")):
            by_seed_0 = {int(row["seed"]): row for row in aa_groups[(cell, "aa0")]}
            by_seed_1 = {int(row["seed"]): row for row in aa_groups[(cell, "aa1")]}
            d_calls = [
                float(by_seed_1[seed]["calls"]) - float(by_seed_0[seed]["calls"])
                for seed in CANONICAL_SEEDS
            ]
            d_wall = [
                float(by_seed_1[seed]["wall_s"]) - float(by_seed_0[seed]["wall_s"])
                for seed in CANONICAL_SEEDS
            ]
            lines.append(
                f"Paired aa1 minus aa0 calls median {fmt(iqr(d_calls)[0])} "
                f"IQR {fmt(iqr(d_calls)[3])} max abs {fmt(max(abs(v) for v in d_calls))}."
            )
            lines.append(
                f"Paired aa1 minus aa0 wall median {fmt(iqr(d_wall)[0])} "
                f"IQR {fmt(iqr(d_wall)[3])} max abs {fmt(max(abs(v) for v in d_wall))}."
            )
        lines.append("")
    lines.extend(["## Rigid TIP4P", "", TIP4P_REASON, ""])
    out = ROOT / "bench" / "ase-mhpkg" / "REPORT.md"
    summary = ROOT / "bench" / "ase-mhpkg" / "summary.csv"
    if args.write:
        out.write_text("\n".join(lines) + "\n")
        with summary.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "cell",
                    "panel",
                    "arm",
                    "n",
                    "hits",
                    "calls_median",
                    "calls_q1",
                    "calls_q3",
                    "calls_iqr",
                    "wall_median",
                    "wall_q1",
                    "wall_q3",
                    "wall_iqr",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"wrote {out.relative_to(ROOT)} and {summary.relative_to(ROOT)}")
    else:
        print("\n".join(lines))


def write_unsupported(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["cell", "n", "budget", "reference", "reason"])
        for n, energy in tip4p_table():
            writer.writerow([f"tip4p{n}", n, TIP4P_BUDGET, f"{energy:.10f}", TIP4P_REASON])


def selfcheck() -> None:
    hit_tol()
    if parse_seeds("2000-2011") != CANONICAL_SEEDS:
        die("seed parser")
    if parse_seeds("2000,2002") != [2000, 2002]:
        die("seed list")
    counts = {(slot, arm): 0 for slot in (0, 1) for arm in ("anneal", "mhpkg")}
    for seed in CANONICAL_SEEDS:
        for slot, arm in arm_order(seed_index(seed, CANONICAL_SEEDS), ("anneal", "mhpkg")):
            counts[(slot, arm)] += 1
    if any(value != 6 for value in counts.values()):
        die(f"ABBA counts {counts}")
    if seed_index(2001, [2001]) != 1:
        die("canonical seed index")
    sample = [1.0, 2.0, 3.0, 4.0]
    if abs(type7(sample, 0.5) - 2.5) > 1e-12 or abs(type7(sample, 0.25) - 1.75) > 1e-12:
        die("type 7")
    if abs(fisher(6, 12, 6, 12) - 1.0) > 1e-12:
        die("fisher")
    _u, tied = mann_whitney([1.0, 1.0], [1.0, 1.0])
    if tied != 1.0:
        die("mann-whitney ties")
    if abs(iqr(sample)[3] - (type7(sample, 0.75) - type7(sample, 0.25))) > 1e-12:
        die("iqr")
    record = pin()
    for key in (
        "version",
        "revision",
        "sdist_sha256",
        "wheel_sha256",
        "gitlab_archive_sha256",
        "ase",
        "sqnm_revision",
    ):
        if not record.get(key):
            die(f"pin missing {key}")
    if len(record["revision"]) != 40:
        die("package revision length")
    factor = scales("lj")
    if any(abs(factor[key] - 1.0) > 1e-15 for key in factor):
        die("LJ scales are not identity")
    params = parameters("lj")
    if params["T0"] != 500.0 or params["dt0"] != 0.1 or params["fmax"] != 0.005:
        die("LJ parameters changed under the unit rule")
    if params["use_MPI"] is not False:
        die("MPI flag")
    tip4p_table()
    if "FixAtoms" not in TIP4P_REASON or "GminTip4p" not in TIP4P_REASON:
        die("unsupported reason")
    print("selfcheck ok")


def prep() -> None:
    version, revision, ase_version = load_science()
    print(f"minimahopping {version} {revision}")
    print(f"ase {ase_version}")
    for kind in ("lj", "morse", "gupta"):
        print(f"element {kind} {place(kind)}")
        values = parameters(kind)
        print(
            f"params {kind} T0={values['T0']} dt0={values['dt0']} "
            f"Ediff0={values['Ediff0']} fmax={values['fmax']} "
            f"width_cutoff={values['width_cutoff']}"
        )
    factor = scales("tip4p")
    print(
        "tip4p_scale "
        f"s_e={factor['s_e']:.12g} s_l={factor['s_l']} "
        f"s_f={factor['s_f']:.12g} s_t={factor['s_t']:.12g} "
        f"s_dt={factor['s_dt']:.12g}"
    )
    print(TIP4P_REASON)


def main() -> None:
    ensure_seat()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--cells", required=True)
    run.add_argument("--seeds", required=True)
    run.add_argument("--panel", default="h2h", choices=("h2h", "aa"))
    run.add_argument("--arms", default="")
    run.add_argument("--rows", default="")
    run.add_argument("--work", default="")
    run.add_argument("--fresh-session", action="store_true")
    run.add_argument("--rerun", action="store_true")
    check = sub.add_parser("verify")
    check.add_argument("--cell", required=True)
    check.add_argument("--arm", required=True)
    check.add_argument("--seed", type=int, required=True)
    check.add_argument("--panel", default="h2h", choices=("h2h", "aa"))
    check.add_argument("--rows", default="")
    report = sub.add_parser("report")
    report.add_argument("--rows", default="")
    report.add_argument("--aa-rows", default="")
    report.add_argument("--write", action="store_true")
    sub.add_parser("selfcheck")
    sub.add_parser("prep")
    unsupported = sub.add_parser("unsupported")
    unsupported.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.cmd == "run":
        campaign(args)
    elif args.cmd == "verify":
        verify(args)
    elif args.cmd == "report":
        write_report(args)
    elif args.cmd == "selfcheck":
        selfcheck()
    elif args.cmd == "prep":
        prep()
    elif args.cmd == "unsupported":
        path = ROOT / "bench" / "ase-mhpkg" / "unsupported.csv"
        if args.write:
            write_unsupported(path)
            print(path.relative_to(ROOT))
        else:
            print(TIP4P_REASON)


if __name__ == "__main__":
    main()
