# Copyright (c) anneal developers
"""Per-cell GMIN inputs on the paper panel.

The procedure is the one in ``bench/gmin-tuned/README.md``. It is the
same at every size. Anneal is not tuned. ``--self-test`` checks the grid,
the scoring rule and the ABBA balance without running a search.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

for _candidate in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (_candidate / "target_tol.py").is_file() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break
from run_panel import (  # noqa: E402
    GOLD_LENGTH_PER_LJ,
    GOLD_ROW,
    LJ_AVOID,
    LJ_STEP,
    fisher,
    gmin_gupta_radius,
    mann_whitney,
    type7,
)
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

HIT_TOL = DEFAULT_HIT_TOL
EVAL_SEEDS = tuple(range(2000, 2012))
TUNING_SEEDS = tuple(range(3000, 3008))
SCALES = (0.5, 1.0, 2.0)
ACCEPTS = (0.25, 0.5, 0.75)
TUNING_DIVISOR = 20
WALL_CAP_S = 7200
PANEL_SVNREV = "aac1548a8aec13b53ad12a8d6904625081685d2e"
BUILT_SVNREV = "9cbec1a446352bc9ef064bd2b1a5767194655919"
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "GOTO_NUM_THREADS",
    "MKL_DOMAIN_NUM_THREADS",
)
README = Path("bench/gmin-tuned/README.md")
RESULTS = ("<!-- gmin-tuned results: start -->", "<!-- gmin-tuned results: end -->")

TUNING_FIELDS = [
    "system", "seed", "index", "step_scale", "ang_scale", "temp_scale", "accept",
    "step", "astep", "ostep", "temperature", "budget", "hit", "calls", "calls_to_hit",
    "best", "wall_s", "inside_s", "outside_s", "outside_per_call_s", "core", "affinity",
    "machine_id", "wall_cap_s", "wall_capped", "gmin_svnrev", "binary_sha256",
    "input_sha256", "session", "utc", "command",
]
CHOICE_FIELDS = [
    "system", "index", "step_scale", "ang_scale", "temp_scale", "accept",
    "step", "astep", "ostep", "temperature", "hits", "n", "median_calls_to_hit",
    "median_best", "tuning_calls", "tuning_runs", "paper_budget", "tuning_budget",
    "input_sha256", "input_blob",
]
EVAL_FIELDS = [
    "system", "code", "seed", "pair", "slot", "block", "budget", "hit", "calls",
    "calls_to_hit", "best", "wall_s", "inside_s", "outside_s", "outside_per_call_s",
    "core", "affinity", "machine_id", "wall_cap_s", "wall_capped", "code_revision",
    "harness_commit", "data_blob", "binary_sha256", "session", "utc", "command",
]
SUMMARY_FIELDS = [
    "system", "budget", "n", "anneal_hits", "gmin_hits", "fisher_p", "fisher_p_bonferroni",
    "anneal_calls_median", "anneal_calls_iqr", "gmin_calls_median", "gmin_calls_iqr",
    "anneal_to_hit_median", "anneal_to_hit_iqr", "gmin_to_hit_median", "gmin_to_hit_iqr",
    "mannwhitney_u_calls", "mannwhitney_p_calls", "mannwhitney_p_calls_bonferroni",
    "anneal_wall_median_s", "anneal_wall_iqr_s", "gmin_wall_median_s", "gmin_wall_iqr_s",
    "mannwhitney_u_wall", "mannwhitney_p_wall", "mannwhitney_p_wall_bonferroni",
    "anneal_inside_median_s", "gmin_inside_median_s", "anneal_outside_median_s",
    "gmin_outside_median_s", "anneal_outside_per_call_median_s", "gmin_outside_per_call_median_s",
    "anneal_best_median", "gmin_best_median", "m_cells",
]


@dataclass(frozen=True)
class Cell:
    name: str
    kind: str
    n: int
    reference: float
    rho: float
    budget: int
    binary: str


@dataclass(frozen=True)
class Family:
    step: float
    astep: float
    ostep: float | None
    temperature: float
    adjust: bool


@dataclass(frozen=True)
class Point:
    index: int
    step_scale: float
    ang_scale: float | None
    temp_scale: float
    accept: float | None


@dataclass(frozen=True)
class Job:
    seed: int
    arm: str
    pair: int
    slot: int
    block: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def tip4p_references(root: Path) -> dict[int, float]:
    refs: dict[int, float] = {}
    with (root / "data" / "tip4p_ccd.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            refs[int(row["n"])] = float(row["energy"])
    return refs


def cells_of(root: Path) -> list[Cell]:
    refs = tip4p_references(root)
    rows = [
        Cell("lj38", "lj", 38, -173.928427, 0.0, 400_000, "h2h_gmin_proof"),
        Cell("lj75", "lj", 75, -397.492331, 0.0, 4_000_000, "h2h_gmin_proof"),
        Cell("lj98", "lj", 98, -543.665361, 0.0, 4_000_000, "h2h_gmin_proof"),
        Cell("lj104", "lj", 104, -582.086642, 0.0, 4_000_000, "h2h_gmin_proof"),
        Cell("m80r6", "morse", 80, -378.333471, 6.0, 4_000_000, "h2h_morse"),
        Cell("m80r14", "morse", 80, -340.811371, 14.0, 4_000_000, "h2h_morse"),
        Cell("au38", "gupta", 38, -130.54001708, 0.0, 400_000, "h2h_metal"),
    ]
    for n in range(8, 22):
        if n not in refs:
            raise SystemExit(f"TIP4P n={n} has no Cambridge energy")
        rows.append(Cell(f"tip4p{n}", "tip4p", n, refs[n], 0.0, 2_000_000, "h2h_seed"))
    return rows


def cell_map(root: Path) -> dict[str, Cell]:
    return {cell.name: cell for cell in cells_of(root)}


def family_of(cell: Cell) -> Family:
    if cell.kind in ("lj", "morse"):
        return Family(LJ_STEP, 0.51, None, 1.0, True)
    if cell.kind == "gupta":
        return Family(LJ_STEP * GOLD_LENGTH_PER_LJ, 0.51, None, 0.1, True)
    if cell.kind == "tip4p":
        return Family(0.40, 0.51, 0.30, 2.0, False)
    raise SystemExit(f"no family for {cell.name}")


def grid(cell: Cell) -> list[Point]:
    fam = family_of(cell)
    third = [(None, acc) for acc in ACCEPTS] if fam.adjust else [(ang, None) for ang in SCALES]
    points: list[Point] = []
    for step_scale in SCALES:
        for ang_scale, accept in third:
            for temp_scale in SCALES:
                points.append(Point(len(points), step_scale, ang_scale, temp_scale, accept))
    return points


def fmt_plain(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if "." in text else text + ".0"


def values_of(cell: Cell, point: Point) -> dict[str, float | None]:
    fam = family_of(cell)
    ostep = None if fam.ostep is None or point.ang_scale is None else fam.ostep * point.ang_scale
    return {
        "step": fam.step * point.step_scale,
        "astep": fam.astep,
        "ostep": ostep,
        "temperature": fam.temperature * point.temp_scale,
        "accept": point.accept,
    }


def render(cell: Cell, point: Point) -> str:
    fam = family_of(cell)
    got = values_of(cell, point)
    step = float(got["step"])
    astep = float(got["astep"])
    temperature = float(got["temperature"])
    lines: list[str] = []
    if cell.kind == "gupta":
        lines.extend(
            [
                f"BGUPTAT {cell.n} {GOLD_ROW}",
                f"BGUPTATAB {GOLD_ROW}",
                f"BGUPTATBB {GOLD_ROW}",
            ]
        )
    if cell.kind == "tip4p":
        lines.append("NTIP 4")
    if cell.kind == "tip4p":
        lines.extend(["SLOPPYCONV 1.0D-3", "TIGHTCONV 1.0D-4", "EDIFF 0.01", "MAXIT 800 500", "SAVE 2"])
    else:
        lines.extend(
            ["SLOPPYCONV 5.0D-3", "TIGHTCONV 1.0D-3", "EDIFF 0.003", "MAXIT 600 500", "SAVE 2", "CENTRE", "SORT"]
        )
    if cell.kind == "gupta":
        lines.append(f"RADIUS {gmin_gupta_radius(cell.n):.6f}")
    lines.append(f"TEMPERATURE {fmt_plain(temperature)}")
    if cell.kind == "tip4p":
        lines.append("FIXTEMP")
    if fam.adjust:
        lines.append(f"ACCEPTRATIO {fmt_plain(float(got['accept']))}")
    lines.append("STEPS 90000000 1.0")
    if got["ostep"] is None:
        lines.append(f"STEP {step:.6f} {astep:.6f}")
    else:
        lines.append(f"STEP {step:.6f} {astep:.6f} {float(got['ostep']):.6f}")
    if cell.kind == "tip4p":
        lines.append("FIXSTEP")
    else:
        lines.append("NEWRESTART 40")
        avoid = LJ_AVOID * GOLD_LENGTH_PER_LJ if cell.kind == "gupta" else LJ_AVOID
        lines.append(f"AVOID {avoid:.6f} 20 F" if cell.kind == "gupta" else f"AVOID {avoid:.1f} 20 F")
    if cell.kind == "morse":
        lines.append(f"MORSE {cell.rho:.1f}")
    return "\n".join(lines) + "\n"


def tuning_budget(paper: int) -> int:
    return max(1, paper // TUNING_DIVISOR)


def thread_prefix() -> str:
    parts = [f"{name}=1" for name in THREAD_VARS]
    parts.extend(["OMP_DYNAMIC=FALSE", "MKL_DYNAMIC=FALSE", "OPENBLAS_DYNAMIC=FALSE"])
    return " ".join(parts)


def pinned(binary: str, extra: str) -> str:
    return (
        f"taskset -c 0 timeout --signal=TERM --kill-after=30s {WALL_CAP_S} "
        f"env {thread_prefix()} {extra} {binary}"
    )


def assert_clean(command: str) -> None:
    for token in command.replace("'", " ").replace('"', " ").split():
        if token.startswith("/"):
            raise RuntimeError("command contains an absolute path")


def coords_shell(cell: Cell, seed: int) -> str:
    if cell.kind == "tip4p":
        return (
            f'"$BIN/h2h_seed" tip4p {cell.n} 0 1 {seed} recommended - "$WORK/start" >/dev/null'
            f' && cp "$WORK/start.start.gmin" "$WORK/coords"'
        )
    return f'"$BIN/{cell.binary}" coords {cell.name} {seed} "$WORK/coords"'


def gmin_command(cell: Cell, seed: int, budget: int, data_shell: str, system: str | None = None) -> str:
    name = system or cell.name
    extra = " ".join(
        [
            f"H2H_BUDGET={budget}",
            f"H2H_SYSTEM={name}",
            f"H2H_SEED={seed}",
            f"H2H_TARGET={cell.reference:.12f}",
            f"H2H_TOL={HIT_TOL}",
            'H2H_ROW="$WORK/row.txt"',
        ]
    )
    command = (
        'cd "$WORK" && rm -f coords data row.txt start.start.gmin && '
        f"{coords_shell(cell, seed)} && {data_shell} && {pinned('\"$BIN/GMIN\"', extra)}"
    )
    assert_clean(command)
    return command


def data_shell_file(name: str, seed: int) -> str:
    return f'cp "$ROOT/bench/gmin-tuned/inputs/{name}.data" data && printf \'RANSEED %s\\n\' {seed} >> data'


def data_shell_text(text: str, seed: int) -> str:
    return f"printf %s {shlex.quote(text)} > data && printf 'RANSEED %s\\n' {seed} >> data"


def anneal_command(cell: Cell, seed: int, budget: int) -> str:
    if cell.kind == "tip4p":
        stop = f"{cell.reference + HIT_TOL:.12f}"
        inner = f'"$BIN/h2h_seed" tip4p {cell.n} 0 {budget} {seed} recommended {stop} "$WORK/anneal"'
        extra = ""
    else:
        inner = f'"$BIN/{cell.binary}" run {cell.name} {seed}'
        extra = f"H2H_BUDGET={budget}"
    command = f'cd "$ANNEAL" && {pinned(inner, extra)}'
    assert_clean(command)
    return command


def abba(seeds: tuple[int, ...] | list[int], arm_a: str, arm_b: str) -> list[Job]:
    rows = list(seeds)
    if len(rows) % 2:
        raise SystemExit("ABBA needs an even number of seeds")
    jobs: list[Job] = []
    for pair, index in enumerate(range(0, len(rows), 2)):
        left, right = rows[index], rows[index + 1]
        if pair % 2 == 0:
            block = "ABBA"
            spec = [(left, arm_a, 0), (left, arm_b, 1), (right, arm_b, 2), (right, arm_a, 3)]
        else:
            block = "BAAB"
            spec = [(left, arm_b, 0), (left, arm_a, 1), (right, arm_a, 2), (right, arm_b, 3)]
        for seed, arm, slot in spec:
            jobs.append(Job(seed, arm, pair, slot, block))
    return jobs


def balance(jobs: list[Job]) -> dict[tuple[int, str], int]:
    counts: dict[tuple[int, str], int] = {}
    for job in jobs:
        counts[(job.slot, job.arm)] = counts.get((job.slot, job.arm), 0) + 1
    return counts


def machine_id() -> str:
    for path in (Path("/etc/machine-id"), Path("/proc/sys/kernel/random/boot_id")):
        if path.is_file():
            return hashlib.sha256(path.read_text().strip().encode()).hexdigest()
    raise SystemExit("no machine id source")


def affinity() -> str:
    return ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git_blob(path: Path) -> str:
    return subprocess.run(
        ["git", "hash-object", "-w", str(path)], check=True, capture_output=True, text=True
    ).stdout.strip()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def is_hit(text: str) -> bool:
    return text.strip().lower() in {"1", "true"}


def parse_h2h(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith("h2h_row ")]
    if len(lines) != 1:
        raise RuntimeError(f"expected one h2h_row\n{text[-1500:]}")
    return dict(part.split("=", 1) for part in lines[0].split()[1:] if "=" in part)


def parse_csv_row(text: str) -> dict[str, str]:
    header = ""
    data = ""
    for line in text.splitlines():
        if line.startswith("system,code,seed"):
            header = line
            data = ""
        elif header and line.count(",") == header.count(","):
            data = line
    if not header or not data:
        raise RuntimeError(f"expected one csv result\n{text[-1500:]}")
    return dict(zip(header.split(","), data.split(","), strict=True))


def measured_from(fields: dict[str, str], reference: float, budget: int, label: str) -> dict[str, str]:
    best = float(fields["best"])
    hit = is_hit(fields["hit"])
    evals = int(float(fields["evals_to_hit"]))
    charged = int(float(fields["charged"]))
    wall = float(fields["wall_s"])
    inside = float(fields["pot_s"])
    if hit != energy_hits(best, reference, HIT_TOL):
        raise RuntimeError(f"{label}: hit {int(hit)} best {best} reference {reference}")
    if hit and not (0 < evals <= charged <= budget):
        raise RuntimeError(f"{label}: hit calls {evals} charged {charged} budget {budget}")
    if (not hit) and (evals != budget or charged != budget):
        raise RuntimeError(f"{label}: miss calls {evals} charged {charged} budget {budget}")
    if wall < 0.0 or inside < 0.0 or inside > wall + 2e-4:
        raise RuntimeError(f"{label}: wall {wall} inside {inside}")
    outside = max(wall - inside, 0.0)
    per_call = f"{outside / charged:.6e}" if charged else ""
    return {
        "hit": "1" if hit else "0",
        "calls": str(charged),
        "calls_to_hit": str(evals),
        "best": f"{best:.10f}",
        "wall_s": f"{wall:.6f}",
        "inside_s": f"{inside:.6f}",
        "outside_s": f"{outside:.6f}",
        "outside_per_call_s": per_call,
        "wall_capped": "0",
    }


def run_shell(command: str, log: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("H2H_") or key.startswith("ANNEAL_"):
            del env[key]
    proc = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(command + f"\nexit={proc.returncode}\n" + proc.stdout[-20000:] + proc.stderr[-8000:])
    return proc


def finish_gmin(proc: subprocess.CompletedProcess[str], work: Path, reference: float, budget: int, label: str) -> dict[str, str]:
    row_file = work / "row.txt"
    text = row_file.read_text() if row_file.is_file() else proc.stdout
    if proc.returncode == 124:
        raise RuntimeError(f"{label} hit the wall cap\n{text[-1500:]}")
    if "h2h_row " not in text:
        raise RuntimeError(f"{label} exited {proc.returncode}\n{proc.stderr[-1500:]}\n{proc.stdout[-1500:]}")
    return measured_from(parse_h2h(text), reference, budget, label)


def finish_anneal(proc: subprocess.CompletedProcess[str], cell: Cell, seed: int, budget: int) -> dict[str, str]:
    label = f"{cell.name} anneal {seed}"
    if proc.returncode == 124:
        raise RuntimeError(f"{label} hit the wall cap")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} exited {proc.returncode}\n{proc.stderr[-1500:]}\n{proc.stdout[-1500:]}")
    if cell.kind == "tip4p":
        fields = parse_h2h(proc.stdout)
    else:
        fields = parse_csv_row(proc.stdout)
    if fields.get("system") not in (cell.name, f"tip4p{cell.n}"):
        raise RuntimeError(f"{label}: system {fields.get('system')}")
    return measured_from(fields, cell.reference, budget, label)


def blank_scales(point: Point) -> dict[str, str]:
    return {
        "index": str(point.index),
        "step_scale": fmt_plain(point.step_scale),
        "ang_scale": "" if point.ang_scale is None else fmt_plain(point.ang_scale),
        "temp_scale": fmt_plain(point.temp_scale),
        "accept": "" if point.accept is None else fmt_plain(point.accept),
    }


def value_strings(cell: Cell, point: Point) -> dict[str, str]:
    got = values_of(cell, point)
    return {
        "step": f"{float(got['step']):.6f}",
        "astep": f"{float(got['astep']):.6f}",
        "ostep": "" if got["ostep"] is None else f"{float(got['ostep']):.6f}",
        "temperature": fmt_plain(float(got["temperature"])),
    }


def rank_key(hits: int, median_calls: float | None, median_best: float, index: int) -> tuple:
    calls = math.inf if median_calls is None else median_calls
    return (-hits, calls, median_best, index)


def choose(cell: Cell, rows: list[dict[str, str]]) -> dict[str, str] | None:
    grouped: dict[int, list[dict[str, str]]] = {point.index: [] for point in grid(cell)}
    for row in rows:
        if row["system"] != cell.name:
            continue
        grouped[int(row["index"])].append(row)
    if any(len(group) != len(TUNING_SEEDS) for group in grouped.values()):
        return None
    best_key: tuple | None = None
    winner: dict[str, str] | None = None
    for point in grid(cell):
        group = grouped[point.index]
        seeds = sorted(int(row["seed"]) for row in group)
        if seeds != list(TUNING_SEEDS):
            raise RuntimeError(f"{cell.name} point {point.index} seeds {seeds}")
        hits = [row for row in group if row["hit"] == "1"]
        median_calls = type7([float(row["calls_to_hit"]) for row in hits], 0.5) if hits else None
        median_best = type7([float(row["best"]) for row in group], 0.5)
        key = rank_key(len(hits), median_calls, median_best, point.index)
        if best_key is None or key < best_key:
            best_key = key
            sample = group[0]
            winner = {
                "system": cell.name,
                "index": str(point.index),
                "step_scale": sample["step_scale"],
                "ang_scale": sample["ang_scale"],
                "temp_scale": sample["temp_scale"],
                "accept": sample["accept"],
                "step": sample["step"],
                "astep": sample["astep"],
                "ostep": sample["ostep"],
                "temperature": sample["temperature"],
                "hits": str(len(hits)),
                "n": str(len(TUNING_SEEDS)),
                "median_calls_to_hit": "" if median_calls is None else f"{median_calls:.2f}",
                "median_best": f"{median_best:.10f}",
                "tuning_calls": str(sum(int(row["calls"]) for row in rows if row["system"] == cell.name)),
                "tuning_runs": str(len(TUNING_SEEDS) * len(grid(cell))),
                "paper_budget": str(cell.budget),
                "tuning_budget": str(tuning_budget(cell.budget)),
                "input_sha256": sample["input_sha256"],
                "input_blob": "",
            }
    return winner


def write_inputs(root: Path, table: dict[str, Cell], choices: list[dict[str, str]]) -> list[dict[str, str]]:
    out = root / "bench" / "gmin-tuned" / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for choice in choices:
        cell = table[choice["system"]]
        point = grid(cell)[int(choice["index"])]
        text = render(cell, point)
        digest = sha256_text(text)
        if digest != choice["input_sha256"]:
            raise RuntimeError(f"{cell.name} rendered input does not match the tuning rows")
        path = out / f"{cell.name}.data"
        path.write_text(text)
        choice = dict(choice)
        choice["input_blob"] = git_blob(path)
        written.append(choice)
    return written


def load_build(path: Path) -> dict[str, str]:
    rows = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            rows[key] = value
    if rows.get("gmin_svnrev") != BUILT_SVNREV:
        raise SystemExit(f"build record svnrev {rows.get('gmin_svnrev')} is not {BUILT_SVNREV}")
    if rows.get("panel_svnrev") != PANEL_SVNREV:
        raise SystemExit("build record lost the panel revision")
    return rows


def check_binary(path: Path, digest: str) -> None:
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != digest:
        raise SystemExit(f"{path.name} sha256 {got} is not {digest}")


def require_pin() -> None:
    if socket.gethostname() != "bench":
        raise SystemExit(f"uts name is {socket.gethostname()}")
    if os.sched_getaffinity(0) != {0}:
        raise SystemExit(f"affinity is {sorted(os.sched_getaffinity(0))}")


def stamp(build: dict[str, str], binary: str) -> dict[str, str]:
    return {
        "core": "0",
        "affinity": affinity(),
        "machine_id": machine_id(),
        "wall_cap_s": str(WALL_CAP_S),
        "binary_sha256": binary,
        "session": "",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def tune(args: argparse.Namespace, table: dict[str, Cell], build: dict[str, str]) -> None:
    rows_path = args.root / "bench" / "gmin-tuned" / "tuning-rows.csv"
    stored = read_rows(rows_path)
    done = {(row["system"], row["index"], row["seed"]) for row in stored}
    session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    names = [name for name in table if not args.system or name in args.system]
    for name in names:
        cell = table[name]
        budget = tuning_budget(cell.budget)
        for point in grid(cell):
            text = render(cell, point)
            digest = sha256_text(text)
            for seed in TUNING_SEEDS:
                key = (cell.name, str(point.index), str(seed))
                if key in done:
                    print(f"skip tune {cell.name} {point.index} {seed}", flush=True)
                    continue
                command = gmin_command(cell, seed, budget, data_shell_text(text, seed))
                print(f"tune {cell.name} point {point.index} seed {seed}", flush=True)
                proc = run_shell(command, args.logs / "tune" / cell.name / f"{point.index}-{seed}.log")
                result = finish_gmin(proc, args.work, cell.reference, budget, f"tune {cell.name} {point.index} {seed}")
                row = {
                    "system": cell.name,
                    "seed": str(seed),
                    **blank_scales(point),
                    **value_strings(cell, point),
                    "budget": str(budget),
                    **result,
                    **stamp(build, build["gmin_binary_sha256"]),
                    "gmin_svnrev": build["gmin_svnrev"],
                    "input_sha256": digest,
                    "session": session,
                    "command": command,
                }
                append_row(rows_path, TUNING_FIELDS, row)
                stored.append(row)
                done.add(key)
                print(
                    f"done tune {cell.name} {point.index} {seed} hit={row['hit']} "
                    f"calls={row['calls_to_hit']} best={row['best']}",
                    flush=True,
                )
    choice_path = args.root / "bench" / "gmin-tuned" / "tuning-choice.csv"
    chosen = {row["system"]: row for row in read_rows(choice_path)}
    for name in names:
        choice = choose(table[name], stored)
        if choice is None:
            raise SystemExit(f"{name} tuning grid is incomplete")
        chosen[name] = choice
    ordered = [chosen[name] for name in table if name in chosen]
    ordered = write_inputs(args.root, table, ordered)
    write_csv(choice_path, CHOICE_FIELDS, ordered)
    for choice in ordered:
        print(
            f"choice {choice['system']} index {choice['index']} hits {choice['hits']}/{choice['n']} "
            f"step {choice['step']} temperature {choice['temperature']} accept {choice['accept'] or '-'} "
            f"ostep {choice['ostep'] or '-'} calls {choice['tuning_calls']}",
            flush=True,
        )


def data_blob(root: Path, name: str) -> str:
    path = root / "bench" / "gmin-tuned" / "inputs" / f"{name}.data"
    if not path.is_file():
        raise SystemExit(f"missing tuned input {path.name}")
    return subprocess.run(
        ["git", "rev-parse", f"HEAD:bench/gmin-tuned/inputs/{name}.data"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


def rows_file(args: argparse.Namespace) -> Path:
    if args.rows is not None:
        return args.rows
    return args.root / "bench" / "gmin-tuned" / "rows.csv"


def aa_cell_name(args: argparse.Namespace) -> str:
    # The same-binary block scores the first named cell. With no name it
    # stays on LJ38, which is the block the first shard recorded.
    if args.system:
        return args.system[0]
    return "lj38"


def evaluate(args: argparse.Namespace, table: dict[str, Cell], build: dict[str, str]) -> None:
    rows_path = rows_file(args)
    stored = read_rows(rows_path)
    done = {(row["system"], row["code"], row["seed"]) for row in stored}
    session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.root, check=True, capture_output=True, text=True
    ).stdout.strip()
    names = [name for name in table if not args.system or name in args.system] if args.eval else []
    plan: list[tuple[str, Cell, Job, str]] = []
    for name in names:
        cell = table[name]
        for job in abba(EVAL_SEEDS, "anneal", "gmin"):
            plan.append((name, cell, job, cell.name))
    aa_system = None
    if args.aa:
        chosen = aa_cell_name(args)
        cell = table[chosen]
        aa_system = f"{chosen}aa"
        for job in abba(EVAL_SEEDS, "gmin_a", "gmin_b"):
            plan.append((aa_system, cell, job, chosen))
    counts = balance([job for name, _cell, job, _data in plan if name != aa_system])
    aa_counts = balance([job for name, _cell, job, _data in plan if name == aa_system])
    per_slot = len(EVAL_SEEDS) // 4
    if names and (len(counts) != 8 or set(counts.values()) != {len(names) * per_slot}):
        raise SystemExit(f"ABBA counts {counts}")
    if args.aa and (len(aa_counts) != 8 or set(aa_counts.values()) != {per_slot}):
        raise SystemExit(f"A/A counts {aa_counts}")
    for name, cell, job, data_name in plan:
        code = job.arm
        if (name, code, str(job.seed)) in done:
            print(f"skip {name} {code} {job.seed}", flush=True)
            continue
        budget = cell.budget
        if code == "anneal":
            command = anneal_command(cell, job.seed, budget)
            binary = build[f"example_{cell.binary}"]
            revision = build["anneal_commit"]
        else:
            command = gmin_command(cell, job.seed, budget, data_shell_file(data_name, job.seed), system=name)
            binary = build["gmin_binary_sha256"]
            revision = build["gmin_svnrev"]
        print(f"start {name} {code} {job.seed} slot {job.slot} {job.block}", flush=True)
        proc = run_shell(command, args.logs / "eval" / name / f"{code}-{job.seed}.log")
        if code == "anneal":
            result = finish_anneal(proc, cell, job.seed, budget)
        else:
            result = finish_gmin(proc, args.work, cell.reference, budget, f"{name} {code} {job.seed}")
        row = {
            "system": name,
            "code": code,
            "seed": str(job.seed),
            "pair": str(job.pair),
            "slot": str(job.slot),
            "block": job.block,
            "budget": str(budget),
            **result,
            **stamp(build, binary),
            "code_revision": revision,
            "harness_commit": head,
            "data_blob": data_blob(args.root, data_name),
            "session": session,
            "command": command,
        }
        append_row(rows_path, EVAL_FIELDS, row)
        done.add((name, code, str(job.seed)))
        print(
            f"done {name} {code} {job.seed} hit={row['hit']} calls={row['calls']} "
            f"to_hit={row['calls_to_hit']} wall_s={row['wall_s']} best={row['best']}",
            flush=True,
        )


def latest(rows: list[dict[str, str]], system: str, code: str) -> list[dict[str, str]]:
    by_seed: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["system"] == system and row["code"] == code:
            by_seed[row["seed"]] = row
    return [by_seed[str(seed)] for seed in EVAL_SEEDS if str(seed) in by_seed]


def iqr(values: list[float]) -> float:
    return type7(values, 0.75) - type7(values, 0.25)


def ranked_calls(group: list[dict[str, str]], budget: int) -> list[float] | None:
    if any(row["hit"] == "0" and int(row["calls"]) < budget for row in group):
        return None
    return [float(row["calls_to_hit"]) if row["hit"] == "1" else math.inf for row in group]


def wall_capped(row: dict[str, str]) -> bool:
    return row.get("wall_capped") == "1" or (row["hit"] == "0" and int(row["calls"]) < int(row["budget"]))


def ranked_wall(left: list[dict[str, str]], right: list[dict[str, str]]) -> tuple[list[float], list[float]] | None:
    if any(wall_capped(row) for row in left + right):
        return None

    def series(group: list[dict[str, str]]) -> list[float]:
        return [float(row["wall_s"]) if row["hit"] == "1" else math.inf for row in group]

    return series(left), series(right)


def arm_stats(group: list[dict[str, str]], prefix: str) -> dict[str, str]:
    calls = [float(row["calls"]) for row in group]
    to_hit = [float(row["calls_to_hit"]) for row in group]
    wall = [float(row["wall_s"]) for row in group]
    inside = [float(row["inside_s"]) for row in group]
    outside = [float(row["outside_s"]) for row in group]
    per_call = [float(row["outside_per_call_s"]) for row in group]
    best = [float(row["best"]) for row in group]
    return {
        f"{prefix}_calls_median": f"{type7(calls, 0.5):.2f}",
        f"{prefix}_calls_iqr": f"{iqr(calls):.2f}",
        f"{prefix}_to_hit_median": f"{type7(to_hit, 0.5):.2f}",
        f"{prefix}_to_hit_iqr": f"{iqr(to_hit):.2f}",
        f"{prefix}_wall_median_s": f"{type7(wall, 0.5):.6f}",
        f"{prefix}_wall_iqr_s": f"{iqr(wall):.6f}",
        f"{prefix}_inside_median_s": f"{type7(inside, 0.5):.6f}",
        f"{prefix}_outside_median_s": f"{type7(outside, 0.5):.6f}",
        f"{prefix}_outside_per_call_median_s": f"{type7(per_call, 0.5):.6e}",
        f"{prefix}_best_median": f"{type7(best, 0.5):.10f}",
    }


def check_balance(group_a: list[dict[str, str]], group_b: list[dict[str, str]], label: str) -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in group_a + group_b:
        counts[(row["slot"], row["code"])] = counts.get((row["slot"], row["code"]), 0) + 1
    if not counts or len(set(counts.values())) != 1:
        raise RuntimeError(f"{label} slot/arm counts {counts}")


def summarise(rows: list[dict[str, str]], names: list[str]) -> list[dict[str, str]]:
    ready = []
    for name in names:
        anneal = latest(rows, name, "anneal")
        gmin = latest(rows, name, "gmin")
        if len(anneal) == len(EVAL_SEEDS) and len(gmin) == len(EVAL_SEEDS):
            ready.append((name, anneal, gmin))
    m_cells = len(ready)
    out = []
    for name, anneal, gmin in ready:
        check_balance(anneal, gmin, name)
        budget = int(anneal[0]["budget"])
        ah = sum(row["hit"] == "1" for row in anneal)
        gh = sum(row["hit"] == "1" for row in gmin)
        fisher_p = fisher(ah, len(anneal), gh, len(gmin))
        a_calls = ranked_calls(anneal, budget)
        g_calls = ranked_calls(gmin, budget)
        calls_mw = mann_whitney(a_calls, g_calls) if a_calls is not None and g_calls is not None else None
        wall_pair = ranked_wall(anneal, gmin)
        wall_mw = mann_whitney(*wall_pair) if wall_pair is not None else None

        def adjusted(p: float | None) -> str:
            return "" if p is None else f"{min(1.0, p * m_cells):.6g}"

        out.append(
            {
                "system": name,
                "budget": str(budget),
                "n": str(len(EVAL_SEEDS)),
                "anneal_hits": str(ah),
                "gmin_hits": str(gh),
                "fisher_p": f"{fisher_p:.6g}",
                "fisher_p_bonferroni": adjusted(fisher_p),
                **arm_stats(anneal, "anneal"),
                **arm_stats(gmin, "gmin"),
                "mannwhitney_u_calls": "" if calls_mw is None else f"{calls_mw[0]:.1f}",
                "mannwhitney_p_calls": "" if calls_mw is None else f"{calls_mw[1]:.6g}",
                "mannwhitney_p_calls_bonferroni": adjusted(None if calls_mw is None else calls_mw[1]),
                "mannwhitney_u_wall": "" if wall_mw is None else f"{wall_mw[0]:.1f}",
                "mannwhitney_p_wall": "" if wall_mw is None else f"{wall_mw[1]:.6g}",
                "mannwhitney_p_wall_bonferroni": adjusted(None if wall_mw is None else wall_mw[1]),
                "m_cells": str(m_cells),
            }
        )
    return out


def aa_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    systems: list[str] = []
    for row in rows:
        if row["system"].endswith("aa") and row["system"] not in systems:
            systems.append(row["system"])
    out = []
    for system in systems:
        left = latest(rows, system, "gmin_a")
        right = latest(rows, system, "gmin_b")
        if len(left) != len(EVAL_SEEDS) or len(right) != len(EVAL_SEEDS):
            continue
        check_balance(left, right, system)
        by_seed = {row["seed"]: row for row in right}
        deltas = []
        for row in left:
            other = by_seed[row["seed"]]
            deltas.append(
                {
                    "calls": abs(float(row["calls"]) - float(other["calls"])),
                    "calls_to_hit": abs(float(row["calls_to_hit"]) - float(other["calls_to_hit"])),
                    "best": abs(float(row["best"]) - float(other["best"])),
                    "wall_s": abs(float(row["wall_s"]) - float(other["wall_s"])),
                    "outside_per_call_s": abs(float(row["outside_per_call_s"]) - float(other["outside_per_call_s"])),
                }
            )

        def pack(key: str) -> str:
            values = [item[key] for item in deltas]
            return f"{type7(values, 0.5):.6g} {iqr(values):.6g} {max(values):.6g}"

        out.append(
            {
                "system": system,
                "n": str(len(EVAL_SEEDS)),
                "median_iqr_max_calls": pack("calls"),
                "median_iqr_max_calls_to_hit": pack("calls_to_hit"),
                "median_iqr_max_best": pack("best"),
                "median_iqr_max_wall_s": pack("wall_s"),
                "median_iqr_max_outside_per_call_s": pack("outside_per_call_s"),
            }
        )
    return out


def show(value: str, digits: int = 4) -> str:
    if not value:
        return "-"
    try:
        number = float(value)
    except ValueError:
        return value
    if number >= 100:
        return f"{number:.1f}"
    if number >= 1:
        return f"{number:.3f}"
    return f"{number:.{digits}g}"


def results_block(summary: list[dict[str, str]], choices: list[dict[str, str]], aa_rows: list[dict[str, str]]) -> str:
    lines = ["## Tuned values", ""]
    lines.append("| Cell | Step | Angular | Temperature | Accept | Tuning hits | Median calls-to-hit | Median best | Tuning calls |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for choice in choices:
        lines.append(
            "| {system} | {step} | {angular} | {temperature} | {accept} | {hits}/{n} | {calls} | {best} | {spent} |".format(
                system=choice["system"],
                step=choice["step"],
                angular=choice["ostep"] or "-",
                temperature=choice["temperature"],
                accept=choice["accept"] or "-",
                hits=choice["hits"],
                n=choice["n"],
                calls=choice["median_calls_to_hit"] or "-",
                best=choice["median_best"],
                spent=choice["tuning_calls"],
            )
        )
    lines.extend(["", "## Evaluation", ""])
    if not summary:
        lines.append("No evaluation cell has both arms on seeds 2000-2011 yet.")
    else:
        header = [
            "Cell", "Arm", "Hits", "Calls", "IQR", "To hit", "IQR", "Wall", "IQR",
            "Inside", "Outside", "Out/call", "Fisher", "Bonferroni", "MW calls", "Bonferroni",
            "MW wall", "Bonferroni",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for row in summary:
            for arm, label in (("anneal", "anneal"), ("gmin", "GMIN")):
                first = arm == "anneal"
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            row["system"] if first else "",
                            label,
                            f"{row[f'{arm}_hits']}/{row['n']}",
                            show(row[f"{arm}_calls_median"], 6),
                            show(row[f"{arm}_calls_iqr"], 6),
                            show(row[f"{arm}_to_hit_median"], 6),
                            show(row[f"{arm}_to_hit_iqr"], 6),
                            show(row[f"{arm}_wall_median_s"]),
                            show(row[f"{arm}_wall_iqr_s"]),
                            show(row[f"{arm}_inside_median_s"]),
                            show(row[f"{arm}_outside_median_s"]),
                            show(row[f"{arm}_outside_per_call_median_s"]),
                            row["fisher_p"] if first else "",
                            row["fisher_p_bonferroni"] if first else "",
                            (row["mannwhitney_p_calls"] or "-") if first else "",
                            (row["mannwhitney_p_calls_bonferroni"] or "-") if first else "",
                            (row["mannwhitney_p_wall"] or "-") if first else "",
                            (row["mannwhitney_p_wall_bonferroni"] or "-") if first else "",
                        ]
                    )
                    + " |"
                )
        lines.append("")
        lines.append(f"Bonferroni uses m = {summary[0]['m_cells']} cells in this table.")
    for band in aa_rows:
        label = band["system"][:-2].upper() if band["system"].endswith("aa") else band["system"].upper()
        lines.extend(
            [
                "",
                f"## {label} same-binary noise",
                "",
                "Absolute difference of the two GMIN arms, median, IQR and maximum.",
                "",
                f"- Calls: {band['median_iqr_max_calls']}",
                f"- Calls-to-hit: {band['median_iqr_max_calls_to_hit']}",
                f"- Best energy: {band['median_iqr_max_best']}",
                f"- Wall seconds: {band['median_iqr_max_wall_s']}",
                f"- Outside seconds per call: {band['median_iqr_max_outside_per_call_s']}",
            ]
        )
    return "\n".join(lines)


def splice(path: Path, block: str) -> None:
    text = path.read_text()
    start, end = RESULTS
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"{path} needs the results markers")
    head, rest = text.split(start, 1)
    _old, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n{block}\n{end}{tail}")


def report(args: argparse.Namespace, table: dict[str, Cell]) -> None:
    root = args.root
    choices = read_rows(root / "bench" / "gmin-tuned" / "tuning-choice.csv")
    rows = read_rows(rows_file(args))
    names = [cell.name for cell in cells_of(root)]
    summary = summarise(rows, names)
    aa_rows = aa_summary(rows)
    summary_path = args.summary if args.summary is not None else root / "bench" / "gmin-tuned" / "summary.csv"
    write_csv(summary_path, SUMMARY_FIELDS, summary)
    if aa_rows:
        aa_path = args.aa_csv if args.aa_csv is not None else root / "bench" / "gmin-tuned" / "aa.csv"
        write_csv(aa_path, list(aa_rows[0].keys()), aa_rows)
    splice(root / README, results_block(summary, choices, aa_rows))
    print(f"summary cells {len(summary)} choices {len(choices)} aa {len(aa_rows)}", flush=True)


def verify(args: argparse.Namespace, table: dict[str, Cell]) -> None:
    if not args.system or len(args.system) != 1 or args.code is None or args.seed is None:
        raise SystemExit("verify needs one --system, --code and --seed")
    system = args.system[0]
    rows = read_rows(args.rows)
    matches = [
        row for row in rows
        if row["system"] == system and row["code"] == args.code and row["seed"] == str(args.seed)
    ]
    if not matches:
        raise SystemExit(f"no stored row {system} {args.code} {args.seed}")
    row = matches[-1]
    command = row["command"]
    assert_clean(command)
    proc = run_shell(command, args.logs / "verify" / f"{args.system}-{args.code}-{args.seed}.log")
    if row["code"] == "anneal":
        cell = table[row["system"]]
        fresh = finish_anneal(proc, cell, int(row["seed"]), int(row["budget"]))
    else:
        cell_name = row["system"][:-2] if row["system"].endswith("aa") else row["system"]
        cell = table[cell_name]
        fresh = finish_gmin(proc, args.work, cell.reference, int(row["budget"]), f"verify {row['system']}")
    for key in ("hit", "calls", "calls_to_hit"):
        if fresh[key] != row[key]:
            raise SystemExit(f"verify {key} {fresh[key]} != {row[key]}")
    if abs(float(fresh["best"]) - float(row["best"])) > 1e-8:
        raise SystemExit(f"verify best {fresh['best']} != {row['best']}")
    print(f"verify {row['system']} {row['code']} {row['seed']} ok", flush=True)


def check_readme(root: Path) -> None:
    text = (root / README).read_text()
    for phrase in ("3000-3007", "2000-2011", "1/20", "0.5, 1 and 2", "0.25, 0.5 and 0.75", "7200", "ABBA"):
        if phrase not in text:
            raise SystemExit(f"readme missing {phrase}")


def self_test() -> None:
    root = repo_root()
    check_readme(root)
    table = cell_map(root)
    if set(table) != {
        "lj38", "lj75", "lj98", "lj104", "m80r6", "m80r14", "au38",
        *[f"tip4p{n}" for n in range(8, 22)],
    }:
        raise SystemExit("cell list")
    if set(TUNING_SEEDS) & set(EVAL_SEEDS):
        raise SystemExit("seed overlap")
    if len(TUNING_SEEDS) != 8 or len(EVAL_SEEDS) != 12:
        raise SystemExit("seed count")
    for name in table:
        points = grid(table[name])
        if len(points) != 27 or [point.index for point in points] != list(range(27)):
            raise SystemExit(f"{name} grid")
        if tuning_budget(table[name].budget) != table[name].budget // 20:
            raise SystemExit(f"{name} fraction")
    lj = grid(table["lj38"])[13]
    if (lj.step_scale, lj.ang_scale, lj.temp_scale, lj.accept) != (1.0, None, 1.0, 0.5):
        raise SystemExit(f"default lj point {lj}")
    text = render(table["lj38"], lj)
    if "TEMPERATURE 1.0\n" not in text or "STEP 0.400000 0.510000\n" not in text or "ACCEPTRATIO 0.5\n" not in text:
        raise SystemExit(f"lj render\n{text}")
    if "FIXSTEP" in text or "MORSE" in text:
        raise SystemExit("lj picked up another potential")
    gold = render(table["au38"], grid(table["au38"])[13])
    if "STEP 1.027741 0.510000\n" not in gold or "TEMPERATURE 0.1\n" not in gold:
        raise SystemExit(f"gold render\n{gold}")
    if f"RADIUS {gmin_gupta_radius(38):.6f}\n" not in gold:
        raise SystemExit("gold radius")
    morse = render(table["m80r14"], grid(table["m80r14"])[13])
    if "MORSE 14.0\n" not in morse:
        raise SystemExit("morse rho")
    water = grid(table["tip4p8"])[13]
    if water.accept is not None or water.ang_scale != 1.0:
        raise SystemExit("tip4p default axis")
    tip = render(table["tip4p21"], water)
    if "FIXSTEP\n" not in tip or "FIXTEMP\n" not in tip or "STEP 0.400000 0.510000 0.300000\n" not in tip:
        raise SystemExit(f"tip4p render\n{tip}")
    if "ACCEPTRATIO" in tip or "CENTRE" in tip:
        raise SystemExit("tip4p adjusted a fixed input")
    # Hit count, then calls-to-hit, then energy, then earlier index.
    keys = [
        rank_key(1, 100.0, -1.0, 3),
        rank_key(2, 500.0, -1.0, 9),
        rank_key(2, 100.0, -1.0, 4),
        rank_key(2, 100.0, -2.0, 8),
        rank_key(2, 100.0, -2.0, 1),
    ]
    if min(keys) != rank_key(2, 100.0, -2.0, 1):
        raise SystemExit(f"rank {min(keys)}")
    if not rank_key(0, None, -4.0, 9) < rank_key(0, None, -3.0, 2):
        raise SystemExit("energy separates misses")
    jobs = abba(EVAL_SEEDS, "anneal", "gmin")
    counts = balance(jobs)
    if set(counts.values()) != {3} or len(counts) != 8:
        raise SystemExit(f"balance {counts}")
    if [job.block for job in jobs[:4]] != ["ABBA"] * 4 or [job.block for job in jobs[4:8]] != ["BAAB"] * 4:
        raise SystemExit("block order")
    command = gmin_command(table["lj38"], 2000, 400000, data_shell_file("lj38", 2000))
    anneal = anneal_command(table["lj38"], 2000, 400000)
    for sample in (command, anneal):
        assert_clean(sample)
        if "$BIN" not in sample or "taskset -c 0" not in sample or f" {WALL_CAP_S} " not in sample:
            raise SystemExit(sample)
    if abs(fisher(12, 12, 0, 12) - 2.0 / math.comb(24, 12)) > 1e-15:
        raise SystemExit("fisher")
    if abs(type7([1, 2, 3, 4, 5, 6], 0.25) - 2.25) > 1e-12:
        raise SystemExit("type7")
    fake_a = [{"hit": "1", "calls": "10", "calls_to_hit": "10", "wall_s": "1", "slot": "0", "code": "anneal", "seed": str(s)} for s in EVAL_SEEDS]
    fake_b = [{"hit": "0", "calls": "100", "calls_to_hit": "100", "wall_s": "2", "slot": str(i % 4), "code": "gmin", "seed": str(s)} for i, s in enumerate(EVAL_SEEDS)]
    # The fake slots are not balanced; the real check is the schedule above.
    ranked = ranked_calls(fake_b, 100)
    if ranked != [math.inf] * 12:
        raise SystemExit("ranked misses")
    if ranked_calls([{"hit": "0", "calls": "5"}], 100) is not None:
        raise SystemExit("censor")
    both_miss = ranked_wall(
        [{"hit": "0", "calls": "10", "budget": "10", "wall_s": "2", "wall_capped": "0"}],
        [{"hit": "1", "calls": "4", "budget": "10", "wall_s": "1", "wall_capped": "0"}],
    )
    if both_miss != ([math.inf], [1.0]):
        raise SystemExit(f"wall rank {both_miss}")
    misses = [{"hit": "0", "calls": "10", "budget": "10", "wall_s": "9", "wall_capped": "0"}]
    if ranked_wall(misses, misses) != ([math.inf], [math.inf]):
        raise SystemExit("two misses still have a wall rank")
    capped = [{"hit": "0", "calls": "3", "budget": "10", "wall_s": "7200", "wall_capped": "1"}]
    if ranked_wall(capped, capped) is not None:
        raise SystemExit("wall cap should censor")
    sample = "\n".join(
        [
            "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s",
            "m80r6,anneal,2000,4000000,true,37038,37038,1.347350,-378.33346188,0.972210,0.375140",
            "arm single draws 27 accepts 13 best -367.445388",
            "arm dls-kernel draws 134 accepts 100 best -376.674472",
        ]
    )
    parsed = parse_csv_row(sample)
    if parsed["system"] != "m80r6" or parsed["charged"] != "37038":
        raise SystemExit(f"csv parse {parsed}")
    print("ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--bin", type=Path)
    parser.add_argument("--anneal", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--aa-csv", type=Path)
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--code")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--aa", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv[1:])
    table = cell_map(args.root)
    if args.system:
        unknown = sorted({name for name in args.system if name not in table})
        if unknown:
            raise SystemExit(f"unknown system {', '.join(unknown)}")
    if args.report:
        report(args, table)
        return 0
    if args.verify or args.tune or args.eval or args.aa:
        require_pin()
        if args.build is None or args.bin is None or args.work is None or args.logs is None:
            parser.error("--build, --bin, --work and --logs are required")
        build = load_build(args.build)
        check_binary(args.bin / "GMIN", build["gmin_binary_sha256"])
        for example in ("h2h_gmin_proof", "h2h_morse", "h2h_metal", "h2h_seed"):
            check_binary(args.bin / example, build[f"example_{example}"])
        os.environ["BIN"] = str(args.bin)
        os.environ["WORK"] = str(args.work)
        os.environ["ROOT"] = str(args.root)
        if args.anneal is not None:
            os.environ["ANNEAL"] = str(args.anneal)
        if args.verify:
            if args.rows is None:
                args.rows = args.root / "bench" / "gmin-tuned" / "rows.csv"
            verify(args, table)
            return 0
        if args.tune:
            tune(args, table, build)
        if args.eval or args.aa:
            evaluate(args, table, build)
        return 0
    parser.error("choose --self-test, --tune, --eval, --aa, --report or --verify")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
