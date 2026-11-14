# Copyright (c) anneal developers
"""ASE and GOFEE on the potentials anneal already runs.

Au55, Cu38 and Cu12Au26 are Gupta. Si10 is Stillinger-Weber and C10 is
Tersoff, both through the rgpot shim. (H2O)8 is rigid TIP4P. The calculator
is ``AnnealCalculator`` on ``asg_eval serve``, so every force call is that
potential. Budgets and references are the recorded paper rows. Six seeds,
serial, two-hour wall cap. The Lennard-Jones journal is not touched.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from campaign import (
    AGOX_VERSION,
    ASE_VERSION,
    FMAX,
    HIT_TOL,
    ORDER_SEED,
    SEEDS,
    _finish,
    format_results,
    git_head,
    install_constraint_shim,
    load_journal,
    median_iqr,
    num,
    require_versions,
)
from potential import AnnealCalculator, AnnealServer, BudgetReached, CapReached, HitReached

WALL_CAP_S = 7200.0
SYSTEMS = ("au55", "cu38", "cu12au26", "sw10", "tc10", "tip4p8")
ARMS = ("ase_bh", "ase_mh", "agox_gofee")
# Native energy, atom count, dimension, budget. spec must agree.
EXPECTED = {
    "au55": (55, 165, 1_000_000, -191.03085280),
    "cu38": (38, 114, 400_000, -108.96759055),
    "cu12au26": (38, 114, 400_000, -127.21076866),
    "sw10": (10, 30, 100_000, -13.79658026),
    "tc10": (10, 30, 100_000, -55.08979464),
    "tip4p8": (16, 48, 2_000_000, -305.5183221676),
}

# Published anneal and GMIN rows. Calls equal to the budget are the recorded
# median, including panels that filled a hit with the budget.
RECORDED = (
    ("au55", "anneal", 6, 0, 1_000_000, "", "", 73.372408, 32.254, 41.054),
    ("au55", "gmin", 6, 0, 1_000_000, "", "", 59.565400, 56.656, 2.910),
    ("cu38", "anneal", 6, 6, 400_000, 400_000, -108.96759055, 10.515665, 5.980, 4.536),
    ("cu38", "gmin", 6, 0, 400_000, "", "", 11.464700, 10.752, 0.710),
    ("cu12au26", "anneal", 12, 0, 400_000, "", -127.01334390, "", "", ""),
    ("cu12au26", "gmin", 12, 0, 400_000, "", -126.48767642, 12.217900, 11.312760, 0.908837),
    ("sw10", "anneal", 12, 12, 100_000, 597, -13.796580, 0.011, 0.010, 0.001),
    ("sw10", "gmin", 12, 12, 100_000, 986, -13.796580, 0.006, 0.005, 0.001),
    ("tc10", "anneal", 12, 12, 100_000, 27496, -55.089795, 0.425, 0.405, 0.020),
    ("tc10", "gmin", 12, 5, 100_000, 100_000, -55.089795, 1.700, 1.583, 0.117),
    ("tip4p8", "anneal", 12, 12, 2_000_000, 6526, -305.517695, 0.020022, 0.012387, 0.007633),
    ("tip4p8", "gmin", 12, 8, 2_000_000, 240163, -305.518927, 1.697360, 1.354615, 0.342745),
)


def schedule() -> list[tuple[str, int, str]]:
    jobs = [(system, seed, arm) for system in SYSTEMS for seed in SEEDS for arm in ARMS]
    random.Random(ORDER_SEED).shuffle(jobs)
    return jobs


def read_spec(binary: Path, system: str) -> dict:
    proc = subprocess.run(
        [str(binary), "spec", system],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    fields = lines[0].split()
    symbols = lines[1].split()
    spec = {
        "system": fields[0],
        "n": int(fields[1]),
        "dim": int(fields[2]),
        "budget": int(fields[3]),
        "reference": float(fields[4]),
        "length": float(fields[5]),
        "symbols": symbols,
    }
    expected = EXPECTED[system]
    if (spec["n"], spec["dim"], spec["budget"]) != expected[:3]:
        raise RuntimeError(f"{system} spec {spec} is not {expected}")
    if abs(spec["reference"] - expected[3]) > 1e-9:
        raise RuntimeError(f"{system} reference drifted")
    if len(symbols) != spec["n"]:
        raise RuntimeError(f"{system} symbols")
    return spec


def read_start(binary: Path, system: str, seed: int, path: Path, n: int) -> np.ndarray:
    if path.is_file():
        rows = []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if len(rows) == n:
            return np.asarray(rows, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(binary), "start", system, str(seed)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    path.write_text(proc.stdout)
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if len(rows) != n:
        raise RuntimeError(f"{system} start has {len(rows)} rows")
    return np.asarray(rows, dtype=np.float64)


def check_fixture(binary: Path) -> None:
    path = Path(__file__).resolve().parents[2] / "tests/fixtures/ccd/cuau38/Cu38Au0.txt"
    coords = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] in {"0", "1"}:
            coords.extend(float(part) for part in parts[1:])
    server = AnnealServer(str(binary), "cu38")
    try:
        energy, _inside, grad = server.eval(np.asarray(coords))
    finally:
        server.close()
    if abs(energy - (-108.96759055)) > 1e-6 or grad.size != 114:
        raise RuntimeError(f"Cu38 archive energy {energy}")


def check_gradient(binary: Path, spec: dict, runs: Path) -> None:
    server = AnnealServer(str(binary), spec["system"])
    try:
        start = read_start(
            binary,
            spec["system"],
            2000,
            runs / "starts" / spec["system"] / "seed2000.txt",
            spec["n"],
        )
        x = start.ravel().copy()
        energy, _inside, grad = server.eval(x)
        step = 1.0e-6
        index = min(3, x.size - 1)
        x[index] += step
        shifted, _inside, _grad = server.eval(x)
    finally:
        server.close()
    if not np.isfinite(energy):
        return
    slope = (shifted - energy) / step
    if abs(slope - grad[index]) > 1e-3 * max(1.0, abs(slope)):
        raise RuntimeError(
            f"{spec['system']} gradient {grad[index]} disagrees with {slope}"
        )


def gofee_scale(system: str, length: float) -> float:
    if system == "tip4p8":
        return 1.0
    return 3.4 / length


def make_calc(binary, spec, arm, deadline):
    scale = gofee_scale(spec["system"], spec["length"]) if arm == "agox_gofee" else 1.0
    server = AnnealServer(str(binary), spec["system"])
    calc = AnnealCalculator(
        server,
        spec["budget"],
        spec["reference"],
        HIT_TOL,
        FMAX[arm],
        scale=scale,
        deadline=deadline,
    )
    return calc


def run_ase_bh(binary, spec, seed, work, start) -> dict:
    from ase import Atoms
    from ase.optimize.basin import BasinHopping

    os.chdir(work)
    np.random.seed(seed)
    started = time.perf_counter()
    calc = make_calc(binary, spec, "ase_bh", started + WALL_CAP_S)
    atoms = Atoms(spec["symbols"], positions=start, pbc=False)
    atoms.calc = calc
    capped = False
    try:
        dyn = BasinHopping(
            atoms,
            logfile=str(work / "basin.log"),
            optimizer_logfile="/dev/null",
            trajectory=str(work / "lowest.traj"),
            local_minima_trajectory=str(work / "local_minima.traj"),
        )
        dyn.run(10**9)
    except (HitReached, BudgetReached):
        pass
    except CapReached:
        capped = True
    finally:
        calc.server.close()
    return _finish(calc, time.perf_counter() - started, git_head(), capped)


def run_ase_mh(binary, spec, seed, work, start) -> dict:
    from ase import Atoms
    from ase.optimize.minimahopping import MinimaHopping

    os.chdir(work)
    np.random.seed(seed)
    started = time.perf_counter()
    calc = make_calc(binary, spec, "ase_mh", started + WALL_CAP_S)
    atoms = Atoms(spec["symbols"], positions=start, pbc=False)
    atoms.calc = calc
    capped = False
    try:
        MinimaHopping(atoms)(totalsteps=10**9)
    except (HitReached, BudgetReached):
        pass
    except CapReached:
        capped = True
    finally:
        calc.server.close()
    return _finish(calc, time.perf_counter() - started, git_head(), capped)


def run_gofee(binary, spec, seed, work) -> dict:
    started = time.perf_counter()
    import contextlib

    import matplotlib

    matplotlib.use("Agg")
    from ase import Atoms

    install_constraint_shim()
    from agox import AGOX
    from agox.acquisitors import LowerConfidenceBoundAcquisitor
    from agox.collectors import StandardCollector
    from agox.databases import Database
    from agox.environments import Environment
    from agox.evaluators import LocalOptimizationEvaluator
    from agox.generators import RandomGenerator, RattleGenerator
    from agox.models.GPR import GPR
    from agox.models.GPR.kernels import RBF, Noise
    from agox.models.GPR.kernels import Constant as C
    from agox.models.GPR.priors import Repulsive
    from agox.postprocessors import RelaxPostprocess
    from agox.samplers import KMeansSampler
    from agox.utils.filters.ABC_filter import FilterBaseClass

    class BoundedTrainingFilter(FilterBaseClass):
        name = "BoundedTraining"

        def __init__(self, m_points=512):
            super().__init__()
            self.m_points = m_points

        def _filter(self, atoms):
            count = len(atoms)
            if count <= self.m_points:
                return np.arange(count)
            energies = np.array([structure.get_potential_energy() for structure in atoms])
            chosen = {int(np.argmin(energies)), count - 1}
            pool = np.array([index for index in range(count) if index not in chosen])
            draw = np.random.choice(pool, size=self.m_points - len(chosen), replace=False)
            return np.sort(np.concatenate([np.fromiter(chosen, dtype=int), draw]))

    n = spec["n"]
    os.chdir(work)
    scale_n = max(n, 38) if spec["system"] == "tip4p8" else n
    box = (scale_n / 38.0) ** (1.0 / 3.0)
    cell = 24.0 * box
    conf = 16.0 * box
    corner = (cell - conf) / 2.0
    bond = {"c1": 1.6, "c2": 2.2}
    calc = make_calc(binary, spec, "agox_gofee", started + WALL_CAP_S)
    template = Atoms("", cell=np.eye(3) * cell, pbc=False)
    environment = Environment(
        template=template,
        symbols=_formula(spec["symbols"]),
        confinement_cell=np.eye(3) * conf,
        confinement_corner=np.full(3, corner),
        print_report=False,
    )
    db_path = work / "gofee.db"
    if db_path.exists():
        db_path.unlink()
    database = Database(filename=str(db_path), order=5, initialize=True)
    fingerprint = __import__(
        "agox.models.descriptors.fingerprint", fromlist=["Fingerprint"]
    ).Fingerprint
    descriptor = fingerprint(environment=environment)
    beta = 0.01
    k0 = C(beta, (beta, beta)) * RBF()
    k1 = C(1 - beta, (1 - beta, 1 - beta)) * RBF()
    kernel = C(5000, (1, 1e5)) * (k0 + k1) + Noise(0.01, (0.01, 0.01))
    model = GPR(
        descriptor=descriptor,
        kernel=kernel,
        database=database,
        prior=Repulsive(),
        filter=BoundedTrainingFilter(512),
        n_optimize=0,
        use_ray=False,
    )
    sampler = KMeansSampler(descriptor=descriptor, database=database, sample_size=10)
    confinement = environment.get_confinement()
    generators = [
        RandomGenerator(**confinement, **bond),
        RattleGenerator(**confinement, **bond),
    ]
    collector = StandardCollector(
        generators=generators,
        sampler=sampler,
        environment=environment,
        num_candidates={0: [10, 0], 5: [3, 7]},
        order=1,
    )
    acquisitor = LowerConfidenceBoundAcquisitor(model=model, kappa=2, order=3)
    relaxer = RelaxPostprocess(
        model=acquisitor.get_acquisition_calculator(),
        constraints=environment.get_constraints(),
        optimizer_run_kwargs={"steps": 5},
        start_relax=8,
        order=2,
    )
    evaluator = LocalOptimizationEvaluator(
        calc,
        gets={"get_key": "prioritized_candidates"},
        optimizer_kwargs={"logfile": None},
        optimizer_run_kwargs={"fmax": 0.05, "steps": 1000},
        constraints=environment.get_constraints(),
        store_trajectory=False,
        order=4,
    )
    search = AGOX(collector, relaxer, acquisitor, evaluator, database, seed=seed)
    log = (work / "gofee.stdio").open("w")
    capped = False
    try:
        with contextlib.redirect_stdout(log):
            try:
                search.run(N_iterations=spec["budget"], verbose=False, hide_log=True)
            finally:
                import atexit

                atexit.unregister(search.on_failure)
    except (HitReached, BudgetReached):
        pass
    except CapReached:
        capped = True
    finally:
        log.close()
        calc.server.close()
    return _finish(calc, time.perf_counter() - started, git_head(), capped)


def _formula(symbols: list[str]) -> str:
    parts = []
    current = None
    count = 0
    for symbol in symbols:
        if symbol == current:
            count += 1
            continue
        if current is not None:
            parts.append(f"{current}{count}")
        current = symbol
        count = 1
    if current is not None:
        parts.append(f"{current}{count}")
    return "".join(parts)


def run_job(binary, runs, specs, system, seed, arm, head) -> dict:
    spec = specs[system]
    work = runs / "asg2" / system / arm / f"seed{seed}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    start = read_start(binary, system, seed, runs / "starts" / system / f"seed{seed}.txt", spec["n"])
    if arm == "ase_bh":
        row = run_ase_bh(binary, spec, seed, work, start)
    elif arm == "ase_mh":
        row = run_ase_mh(binary, spec, seed, work, start)
    elif arm == "agox_gofee":
        row = run_gofee(binary, spec, seed, work)
    else:
        raise RuntimeError(arm)
    row.update(
        {
            "system": system,
            "arm": arm,
            "seed": seed,
            "budget": spec["budget"],
            "ase": ASE_VERSION,
            "agox": AGOX_VERSION,
        }
    )
    if not row.get("head"):
        row["head"] = head
    return row


def summarise(rows: list[dict]) -> str:
    header = (
        "system,arm,n,hits,capped,budget,calls_to_hit_median,calls_to_hit_iqr,"
        "best_median,wall_median_s,inside_median_s,outside_median_s,source"
    )
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["system"], row["arm"]), []).append(row)
    lines = [header]
    for system in SYSTEMS:
        for arm in ARMS:
            group = grouped.get((system, arm), [])
            if not group:
                continue
            hits = [float(item["calls_to_hit"]) for item in group if item["hit"]]
            walls = [float(item["wall_s"]) for item in group if item["wall_s"] != ""]
            insides = [float(item["inside_s"]) for item in group if item["inside_s"] != ""]
            outsides = [float(item["outside_s"]) for item in group if item["outside_s"] != ""]
            bests = [float(item["best"]) for item in group if item["best"] != ""]
            if hits:
                call_median, call_iqr = median_iqr(hits)
                call_text = f"{call_median:.6f},{call_iqr:.6f}"
            else:
                call_text = ","
            capped = sum(1 for item in group if item.get("stop") == "capped")
            lines.append(
                ",".join(
                    [
                        system,
                        arm,
                        str(len(group)),
                        str(sum(int(item["hit"]) for item in group)),
                        str(capped),
                        str(group[0]["budget"]),
                        call_text,
                        "" if not bests else f"{median_iqr(bests)[0]:.8f}",
                        "" if not walls else f"{median_iqr(walls)[0]:.6f}",
                        "" if not insides else f"{median_iqr(insides)[0]:.6f}",
                        "" if not outsides else f"{median_iqr(outsides)[0]:.6f}",
                        "measured",
                    ]
                )
            )
    for item in RECORDED:
        system, arm, n, hits, budget, calls, best, wall, inside, outside = item
        call_text = "" if calls == "" else f"{float(calls):.6f}"
        lines.append(
            ",".join(
                [
                    system,
                    arm,
                    str(n),
                    str(hits),
                    "0",
                    str(budget),
                    call_text,
                    "",
                    num(best, 8),
                    num(wall, 6),
                    num(inside, 6),
                    num(outside, 6),
                    "recorded",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def write_summary(rows: list[dict], repo: Path) -> None:
    out = repo / "bench" / "ase-gofee-h2h"
    out.mkdir(parents=True, exist_ok=True)
    (out / "asg2-summary.csv").write_text(summarise(rows))


def losses(rows: list[dict]) -> list[str]:
    """A measured arm beats recorded anneal on hits, or on calls when both hit."""
    found = []
    recorded = {
        item[0]: item
        for item in RECORDED
        if item[1] == "anneal"
    }
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["system"], row["arm"]), []).append(row)
    for system in SYSTEMS:
        anneal = recorded[system]
        anneal_hits = int(anneal[3])
        anneal_n = int(anneal[2])
        anneal_calls = anneal[5]
        anneal_budget = int(anneal[4])
        for arm in ARMS:
            group = grouped.get((system, arm), [])
            if len(group) < len(SEEDS):
                continue
            hits = sum(int(item["hit"]) for item in group)
            if hits / len(group) > anneal_hits / anneal_n:
                found.append(f"{system} {arm} hits {hits}/{len(group)} anneal {anneal_hits}/{anneal_n}")
                continue
            if anneal_calls == "" or float(anneal_calls) >= anneal_budget:
                continue
            calls = [float(item["calls_to_hit"]) for item in group if item["hit"]]
            if len(calls) < len(group):
                continue
            if median_iqr(calls)[0] < float(anneal_calls):
                found.append(
                    f"{system} {arm} calls {median_iqr(calls)[0]:.1f} anneal {float(anneal_calls):.1f}"
                )
    return found


def main(args) -> int:
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=__import__("sys").stderr)
        return 2
    require_versions()
    if not args.binary.is_file():
        print(f"missing binary {args.binary}", file=__import__("sys").stderr)
        return 2
    specs = {system: read_spec(args.binary, system) for system in SYSTEMS}
    check_fixture(args.binary)
    check_gradient(args.binary, specs["cu38"], args.runs)
    head = git_head()
    if args.system or (args.seed is not None and args.arm):
        if not (args.system and args.seed is not None and args.arm):
            print("a single run needs --system, --seed and --arm", file=__import__("sys").stderr)
            return 2
        if args.system not in specs or args.arm not in ARMS:
            print("unknown system or arm", file=__import__("sys").stderr)
            return 2
        row = run_job(args.binary, args.runs, specs, args.system, args.seed, args.arm, head)
        row["order"] = 0
        print(format_results([row]), end="")
        return 0

    jobs = schedule()
    journal_path = args.runs / "journal-asg2.csv"
    rows = load_journal(journal_path)
    done = {(row["system"], row["seed"], row["arm"]) for row in rows}
    print(f"asg2 jobs {len(jobs)} already {len(done)} cap {WALL_CAP_S:.0f}", flush=True)
    for index, (system, seed, arm) in enumerate(jobs, start=1):
        if (system, seed, arm) in done:
            continue
        print(f"start {index}/{len(jobs)} {system} {arm} seed={seed}", flush=True)
        try:
            row = run_job(args.binary, args.runs, specs, system, seed, arm, head)
        except Exception:
            traceback.print_exc()
            return 1
        row["order"] = index
        rows.append(row)
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.write_text(format_results(rows))
        write_summary(rows, args.repo)
        print(
            f"done {system} {arm} seed={seed} hit={row['hit']} stop={row['stop']} "
            f"calls={row['charged']} best={row['best']:.6f} wall={row['wall_s']:.1f}",
            flush=True,
        )
    write_summary(rows, args.repo)
    for line in losses(rows):
        print(f"LOSS {line}", flush=True)
    print("ASG2_FINISHED", flush=True)
    return 0
