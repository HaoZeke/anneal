#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial GFN2 (H2O)20 panel: anneal against CREST, one host, one core.

Both arms evaluate rgpot GFN2 plus the one-sided O-H wall. Anneal does
that in process. CREST 3.0.2 reaches it through xtb ``--driver`` into
``examples/gfn2_water_crest_driver``. Each seed starts from one random
placement of rigid waters, shared by both arms. The placement draws
centres uniformly in a ball and orientations from random quaternions,
and refuses an intermolecular contact inside 2 angstrom. It is not a
motif, a lattice, or a shell built from a group index.

``PAPER_BUDGETS`` is the charged-call cap, the 28000 calls already used
for GFN2 (H2O)20. A pair already in the rows file at that budget is not
run again. Rows are appended. The summary is rebuilt from them.

The Wales--Hodges (H2O)20 minimum in this repository is a TIP4P energy.
This panel is GFN2-xTB, so the reference is the lowest rescored energy
of the two arms. A hit is a vtight rescore at or below that energy by
1e-6 Eh. The rescore is ``xtb --gfn 2 --opt vtight --acc 0.01
--iterations 500``. This process refuses a search unless the UTS
hostname is bench.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from target_tol import energy_hits  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "bench" / "h2o20-crest" / "seeds.csv"
SUMMARY = ROOT / "bench" / "h2o20-crest" / "summary.csv"
PAPER_BUDGETS = {"h2o20": 28000}
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
ARMS = ("anneal", "crest")
MOLECULES = 20
SYSTEM = "h2o20"
BUDGET = PAPER_BUDGETS[SYSTEM]
HIT_TOL_EH = 1.0e-6
# Far below any GFN2 cluster energy, so neither arm stops on a target.
UNREACHABLE_EV = -1.0e6
SCHEDULE_SEED = 20261117
DRIVER_PATH_MAX = 64
BALL_RADIUS_A = 9.0
CONTACT_A = 2.0
WATER = ((0.0, 0.0, 0.0), (0.7572, 0.5865, 0.0), (-0.7572, 0.5865, 0.0))

ROW_FIELDS = [
    "system",
    "arm",
    "seed",
    "budget",
    "rescored_Eh",
    "objective_eV",
    "wall_s",
    "pot_s",
    "outside_s",
    "code_revision",
    "host",
]
SUMMARY_FIELDS = [
    "system",
    "n",
    "reference_Eh",
    "reference_source",
    "anneal_hits",
    "crest_hits",
    "fisher_p",
    "mannwhitney_p",
    "W_plus",
    "W_minus",
    "wilcoxon_p",
    "rank_biserial",
    "median_gap_Eh",
    "anneal_potential_median_s",
    "anneal_outside_median_s",
    "crest_potential_median_s",
    "crest_outside_median_s",
    "code_revision",
    "host",
]


def read_fields(path: Path) -> dict[str, str]:
    fields = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def last_energy(path: Path) -> str:
    found = ""
    for line in path.read_text(errors="replace").splitlines():
        if "TOTAL ENERGY" not in line:
            continue
        parts = line.replace("|", " ").split()
        if "Eh" in parts:
            found = parts[parts.index("Eh") - 1]
    if not found:
        raise RuntimeError(f"no TOTAL ENERGY in {path}")
    return found


def rescore(xtb: Path, xyz: Path, out: Path, core: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    (out / "in.xyz").write_bytes(xyz.read_bytes())
    subprocess.run(
        [
            "taskset",
            "-c",
            core,
            str(xtb),
            "in.xyz",
            "--gfn",
            "2",
            "--opt",
            "vtight",
            "--acc",
            "0.01",
            "--iterations",
            "500",
        ],
        cwd=out,
        stdout=open(out / "opt.out", "w"),
        stderr=open(out / "opt.err", "w"),
        check=True,
    )
    return last_energy(out / "opt.out")


def random_start(seed: int, path: Path) -> None:
    """Rigid waters at random centres and orientations. Identical for both arms."""
    rng = random.Random(seed)
    placed: list[tuple[str, float, float, float]] = []

    def quat() -> tuple[float, float, float, float]:
        u, v, w = rng.random(), rng.random(), rng.random()
        return (
            math.sqrt(1 - u) * math.sin(2 * math.pi * v),
            math.sqrt(1 - u) * math.cos(2 * math.pi * v),
            math.sqrt(u) * math.sin(2 * math.pi * w),
            math.sqrt(u) * math.cos(2 * math.pi * w),
        )

    def rotate(q, point):
        x, y, z, w = q
        vx, vy, vz = point
        tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
        return (
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        )

    radius = BALL_RADIUS_A
    for _molecule in range(MOLECULES):
        for _attempt in range(20000):
            while True:
                centre = [rng.uniform(-1.0, 1.0) for _ in range(3)]
                if sum(value * value for value in centre) <= 1.0:
                    break
            centre = [radius * value for value in centre]
            turn = quat()
            atoms = []
            for index, site in enumerate(WATER):
                rx, ry, rz = rotate(turn, site)
                atoms.append(("O" if index == 0 else "H", centre[0] + rx, centre[1] + ry, centre[2] + rz))
            clear = True
            for _symbol, x, y, z in atoms:
                for _other, ox, oy, oz in placed:
                    dx, dy, dz = x - ox, y - oy, z - oz
                    if dx * dx + dy * dy + dz * dz < CONTACT_A * CONTACT_A:
                        clear = False
                        break
                if not clear:
                    break
            if clear:
                placed.extend(atoms)
                break
        else:
            raise SystemExit(f"seed {seed} did not place molecule {_molecule}")
    lines = [str(len(placed)), f"(H2O){MOLECULES} seed {seed} random rigid waters"]
    lines.extend(f"{symbol} {x:.8f} {y:.8f} {z:.8f}" for symbol, x, y, z in placed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    if not new:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle))
        if header != ROW_FIELDS:
            raise SystemExit(f"{path} has another header; rows are appended, never rewritten")
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    out = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and abs(values[order[end]]) == abs(values[order[start]]):
            end += 1
        rank = (start + 1 + end) / 2
        for pos in range(start, end):
            out[order[pos]] = rank
        start = end
    return out


def wilcoxon(gaps: list[float]) -> tuple[float, float, float, float]:
    kept = [gap for gap in gaps if gap != 0.0]
    n = len(kept)
    if n == 0:
        return 1.0, 0.0, 0.0, 0.0
    weight = ranks(kept)
    observed = sum(rank for gap, rank in zip(kept, weight) if gap > 0.0)
    total = n * (n + 1) / 2
    low = high = 0
    for mask in range(1 << n):
        score = sum(rank for bit, rank in enumerate(weight) if mask & (1 << bit))
        if score <= observed:
            low += 1
        if score >= observed:
            high += 1
    p_value = min(1.0, 2.0 * min(low, high) / (1 << n))
    negative = total - observed
    effect = 0.0 if observed + negative == 0 else (observed - negative) / (observed + negative)
    return p_value, observed, negative, effect


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    hits = hits_a + hits_b
    total = n_a + n_b
    ways = math.comb(total, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def mann_whitney(left: list[float], right: list[float]) -> float:
    """Two-sided Mann-Whitney, tie correction, continuity correction."""
    n1, n2 = len(left), len(right)
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
        width = j - i + 1
        ties += width**3 - width
        i = j + 1
    u = sum(rank[v] for v in left) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return 1.0
    dev = abs(u - n1 * n2 / 2.0) - 0.5
    return min(1.0, math.erfc(max(dev, 0.0) / math.sqrt(var) / math.sqrt(2.0)))


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def plain(value: float) -> str:
    return f"{value:.10g}"


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    chosen = [
        row
        for row in rows
        if row["system"] == SYSTEM and int(row["budget"]) == BUDGET and int(row["seed"]) in SEEDS
    ]
    by_seed: dict[int, dict[str, dict[str, str]]] = {}
    for row in chosen:
        by_seed.setdefault(int(row["seed"]), {})[row["arm"]] = row
    paired = [by_seed[seed] for seed in SEEDS if set(by_seed.get(seed, {})) >= set(ARMS)]
    if len(paired) != len(SEEDS):
        path.write_text(",".join(SUMMARY_FIELDS) + "\n")
        return
    anneal = [float(pair["anneal"]["rescored_Eh"]) for pair in paired]
    crest = [float(pair["crest"]["rescored_Eh"]) for pair in paired]
    reference = min(anneal + crest)
    anneal_hits = sum(energy_hits(value, reference, HIT_TOL_EH) for value in anneal)
    crest_hits = sum(energy_hits(value, reference, HIT_TOL_EH) for value in crest)
    gaps = [a - c for a, c in zip(anneal, crest)]
    p_value, positive, negative, effect = wilcoxon(gaps)
    text = ",".join(SUMMARY_FIELDS) + "\n"
    text += (
        f"{SYSTEM},{len(paired)},{reference:.12f},best_of_arms,"
        f"{anneal_hits},{crest_hits},{plain(fisher(anneal_hits, len(paired), crest_hits, len(paired)))},"
        f"{plain(mann_whitney(anneal, crest))},{positive:.1f},{negative:.1f},{plain(p_value)},"
        f"{plain(effect)},{plain(median(gaps))},"
        f"{plain(median([float(pair['anneal']['pot_s']) for pair in paired]))},"
        f"{plain(median([float(pair['anneal']['outside_s']) for pair in paired]))},"
        f"{plain(median([float(pair['crest']['pot_s']) for pair in paired]))},"
        f"{plain(median([float(pair['crest']['outside_s']) for pair in paired]))},"
        f"{paired[0]['anneal']['code_revision']},{paired[0]['anneal']['host']}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run_anneal(binary: Path, seed: int, start: Path, out: Path, core: str) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, H2H_START_XYZ=str(start))
    subprocess.run(
        ["taskset", "-c", core, str(binary), str(MOLECULES), str(BUDGET), str(seed), str(out)],
        env=env,
        stdout=open(out / "stdout.txt", "w"),
        stderr=open(out / "stderr.txt", "w"),
        check=True,
    )
    fields = read_fields(out / "run.txt")
    energy = rescore(Path(os.environ["XTB_BIN"]), out / "best.xyz", out / "score", core)
    return {
        "rescored_Eh": energy,
        "objective_eV": fields["objective_best_eV"],
        "wall_s": fields["wall_s"],
        "pot_s": fields["potential_s"],
        "outside_s": fields["outside_s"],
    }


def run_crest(
    crest: Path, xtb: Path, driver: Path, start: Path, out: Path, core: str
) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    served = out / "driver"
    served.mkdir(exist_ok=True)
    short = Path(tempfile.mkdtemp(prefix="h2o20-"))
    sock = short / "sock"
    call = short / "call"
    wrapper = short / "xtb"
    if len(str(call)) > DRIVER_PATH_MAX:
        raise SystemExit(f"{call} is longer than {DRIVER_PATH_MAX} characters")
    call.write_text(f'#!/bin/sh\nexec "{driver}" call "{sock}"\n')
    wrapper.write_text(f'#!/bin/sh\nexec "{xtb}" "$@" --driver "{call}"\n')
    call.chmod(0o755)
    wrapper.chmod(0o755)
    with open(out / "driver.out", "w") as log:
        server = subprocess.Popen(
            [
                "taskset",
                "-c",
                core,
                str(driver),
                "serve",
                str(sock),
                str(served),
                str(MOLECULES),
                str(BUDGET),
                f"{UNREACHABLE_EV:.1f}",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    while not (served / "ready").exists():
        if server.poll() is not None:
            raise RuntimeError(f"driver exited {server.returncode}")
        time.sleep(0.05)
    preopt = out / "preopt"
    search = out / "search"
    preopt.mkdir(exist_ok=True)
    search.mkdir(exist_ok=True)
    (preopt / "start.xyz").write_bytes(start.read_bytes())
    began = time.monotonic()
    subprocess.run(
        ["taskset", "-c", core, str(wrapper), "start.xyz", "--gfn", "2", "--opt"],
        cwd=preopt,
        stdout=open(preopt / "opt.out", "w"),
        stderr=subprocess.STDOUT,
        check=False,
    )
    code = "preopt-failed"
    stop = served / "stop"
    if (preopt / "xtbopt.xyz").is_file() and not stop.exists():
        (search / "start.xyz").write_bytes((preopt / "xtbopt.xyz").read_bytes())
        proc = subprocess.Popen(
            [
                "taskset",
                "-c",
                core,
                str(crest),
                "start.xyz",
                "--gfn2",
                "-T",
                "1",
                "--legacy",
                "-notopo",
                "-shake",
                "1",
                "-xnam",
                str(wrapper),
            ],
            cwd=search,
            stdout=open(search / "crest.out", "w"),
            stderr=open(search / "crest.err", "w"),
            start_new_session=True,
        )
        while proc.poll() is None and not stop.exists():
            time.sleep(0.05)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = "stopped"
        else:
            code = f"exit {proc.returncode}"
    wall = time.monotonic() - began
    subprocess.run([str(driver), "stop", str(sock)], check=False)
    server.wait(timeout=60)
    shutil.rmtree(short, ignore_errors=True)
    (out / "exit.txt").write_text(code + "\n")
    fields = read_fields(served / "driver.txt")
    if int(fields["calls"]) == 0:
        raise RuntimeError(f"no call reached the driver ({code})")
    energy = rescore(xtb, served / "best.xyz", out / "score", core)
    inside = float(fields["inside_s"])
    return {
        "rescored_Eh": energy,
        "objective_eV": fields["best_eV"],
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{inside:.6f}",
        "outside_s": f"{max(wall - inside, 0.0):.6f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--anneal", type=Path)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--crest", type=Path)
    parser.add_argument("--xtb", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--rows", type=Path, default=ROWS)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--head-file", type=Path)
    parser.add_argument("--core", default="0")
    args = parser.parse_args()
    if args.summarize_only:
        write_summary(args.summary, load_rows(args.rows))
        return
    if socket.gethostname() != "bench":
        raise SystemExit("hostname is not bench")
    for name in ("anneal", "driver", "crest", "xtb", "engine", "out", "head_file"):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name.replace('_', '-')} is required")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ.pop("LD_LIBRARY_PATH", None)
    os.environ["RGPOT_XTB_ENGINE"] = str(args.engine)
    os.environ["XTB_BIN"] = str(args.xtb)
    revision = args.head_file.read_text().strip()
    host = socket.gethostname()
    starts = args.out / "starts"
    for seed in SEEDS:
        path = starts / f"{seed}.xyz"
        if not path.is_file():
            random_start(seed, path)
    finished = {
        (int(row["seed"]), row["arm"], int(row["budget"]))
        for row in load_rows(args.rows)
        if row["system"] == SYSTEM
    }
    pairs = [(seed, arm) for seed in SEEDS for arm in ARMS]
    random.Random(SCHEDULE_SEED).shuffle(pairs)
    for seed, arm in pairs:
        if (seed, arm, BUDGET) in finished:
            print(f"skip {arm} {seed}", flush=True)
            continue
        print(f"start {arm} {seed} {datetime.datetime.now().isoformat(timespec='seconds')}", flush=True)
        start = starts / f"{seed}.xyz"
        where = args.out / arm / str(seed)
        if arm == "anneal":
            measured = run_anneal(args.anneal, seed, start, where, args.core)
        else:
            measured = run_crest(args.crest, args.xtb, args.driver, start, where, args.core)
        row = {
            "system": SYSTEM,
            "arm": arm,
            "seed": seed,
            "budget": BUDGET,
            "code_revision": revision,
            "host": host,
            **measured,
        }
        append_row(args.rows, row)
        print(",".join(str(row[key]) for key in ROW_FIELDS), flush=True)
        write_summary(args.summary, load_rows(args.rows))


if __name__ == "__main__":
    main()
