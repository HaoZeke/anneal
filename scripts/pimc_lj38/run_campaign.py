#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial RPQA against the paper preset on LJ38.

The RPQA arm is Fiorentino and Marzari's published LJ input
(``examples/features/quantum_annealing/RPQA/LJ_cluster`` in afiorent/i-pi
``4e86f5fd``): 16 beads, 50 K, a 1 fs Langevin step, ``lambdaqkin`` from 19
to 0.01 on the square-root ramp, and an LBFGS relax of every bead every
5000 steps after a 5000-step delocalisation. The cluster has 38 atoms. The
start is the geometry ``h2h_gmin_proof coords`` writes for the same seed,
scaled by their sigma. The run stops when the quenched energy is within
0.001 of the Cambridge minimum, or when the force-call count, including
every local-optimisation evaluation, reaches the LJ38 paper budget.

Six seeds, one session. Inside the size the (seed, arm) pairs are shuffled.
A miss is stored as the full budget. Medians are Hyndman-Fan type 7. The
wall ratio is the RPQA median wall divided by the anneal median wall.
Inside microseconds are the potential time divided by the charged
evaluations. Outside microseconds are the rest of the wall on the same
denominator. This process refuses to run unless the UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from lj_potential import EPSILON, SIGMA_ANGSTROM, reduced_pair_energy
from meter import Meter

PAPER_BUDGETS = {38: 400_000}
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
ARMS = ("anneal", "rpqa")
SCHEDULE_SEED = 20261113
HIT_TOL = 0.001
REFERENCE = -173.928427
NBEADS = 16
TOTAL_STEPS = 30000
PIN_INTERVAL = 5000
START_STEP = 5000
RAMP_STEPS = 25000
# Fiorentino and Marzari, arXiv:2606.03405, the LJ38 comparison with
# basin-hopping: "32 replicas and only 40 local optimization rounds,
# performed once every 10^4 steps." Also paper/related-work.md.
# Forty rounds, one every 10_000 steps, the first after the opening
# interval: steps 10000, 20000, ..., 400000. i-PI's loop is
# range(total_steps), so the input length is the last step plus one.
PAPER_NBEADS = 32
PAPER_ROUNDS = 40
PAPER_PIN_INTERVAL = 10_000
PAPER_START = 10_000
PAPER_LAST_PIN = PAPER_START + (PAPER_ROUNDS - 1) * PAPER_PIN_INTERVAL
PAPER_TOTAL_STEPS = PAPER_LAST_PIN + 1
PAPER_RAMP_STEPS = PAPER_LAST_PIN
PAPER_ARM = "rpqa_paper"
PAPER_SCHEDULE_SEED = 20261114

SUMMARY_FIELDS = [
    "system",
    "arm",
    "n",
    "planned",
    "hits",
    "hits_over_n",
    "budget",
    "evals_median",
    "evals_iqr",
    "wall_median_s",
    "wall_iqr_s",
    "wall_ratio",
    "inside_us",
    "outside_us",
    "seeds",
    "schedule_seed",
    "rpqa_commit",
    "rpqa_source",
    "anneal_head",
    "code_revision",
]

SEED_FIELDS = [
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "pot_s",
    "inside_us",
    "outside_us",
    "best",
    "stop",
    "reached",
    "code_revision",
]


def type7(sorted_values: list[float], p: float) -> float:
    """Hyndman-Fan type 7. ``sorted_values`` is ascending."""
    n = len(sorted_values)
    if n < 1:
        raise ValueError("type 7 needs a sample")
    if n == 1:
        return sorted_values[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return sorted_values[index]
    return sorted_values[index] * (1.0 - frac) + sorted_values[index + 1] * frac


def median_iqr(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    q1 = type7(ordered, 0.25)
    q3 = type7(ordered, 0.75)
    return type7(ordered, 0.5), q3 - q1


def schedule(
    seeds: tuple[int, ...],
    arms: tuple[str, ...],
    rng_seed: int,
) -> list[tuple[int, str]]:
    rng = random.Random(rng_seed)
    jobs = [(seed, arm) for seed in seeds for arm in arms]
    rng.shuffle(jobs)
    return jobs


def load_source(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    for key in ("rpqa_commit", "rpqa_source"):
        if not fields.get(key):
            raise SystemExit(f"{path} has no {key}")
    return fields


def binary_budget(binary: Path) -> tuple[int, float]:
    proc = subprocess.run(
        [str(binary), "budget", "lj38"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise SystemExit(f"budget for LJ38 returned {proc.stdout!r}")
    return int(lines[0]), float(lines[1])


def _env(budget: int) -> dict[str, str]:
    env = os.environ.copy()
    env["H2H_BUDGET"] = str(budget)
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMBA_NUM_THREADS"] = "1"
    return env


def per_eval(wall_s: float, pot_s: float, charged: int) -> tuple[float, float]:
    if charged < 1:
        raise RuntimeError("per-evaluation split needs a charged call")
    inside = pot_s / charged * 1e6
    outside = max(wall_s - pot_s, 0.0) / charged * 1e6
    return inside, outside


def parse_anneal(text: str) -> dict[str, str]:
    data = [line for line in text.splitlines() if ",anneal," in line]
    if len(data) != 1:
        raise ValueError(f"expected one anneal row, saw {len(data)}")
    header = (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,"
        "cpu,pins,pot_s"
    )
    row = next(csv.DictReader([header, data[0]]))
    hit = "1" if row["hit"] in ("1", "true") else "0"
    if row["hit"] not in ("0", "1", "true", "false"):
        raise ValueError(f"anneal hit {row['hit']}")
    stop = "hit" if hit == "1" else "budget"
    return {
        "system": row["system"],
        "arm": "anneal",
        "seed": row["seed"],
        "budget": row["budget"],
        "hit": hit,
        "evals_to_hit": row["evals_to_hit"],
        "charged": row["charged"],
        "wall_s": row["wall_s"],
        "pot_s": row["pot_s"],
        "best": row["best"],
        "stop": stop,
        "reached": "38",
    }


def with_rates(row: dict[str, str]) -> dict[str, str]:
    inside, outside = per_eval(float(row["wall_s"]), float(row["pot_s"]), int(row["charged"]))
    out = dict(row)
    out["inside_us"] = f"{inside:.4f}"
    out["outside_us"] = f"{outside:.4f}"
    return out


def write_seed(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_FIELDS)
        writer.writeheader()
        writer.writerow({key: row[key] for key in SEED_FIELDS})


def load_seed(path: Path) -> dict[str, str] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"{path} has {len(rows)} rows")
    return rows[0]


def run_anneal(binary: Path, seed: int, budget: int, log: Path) -> dict[str, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        proc = subprocess.run(
            ["taskset", "-c", "0", str(binary), "run", "lj38", str(seed)],
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=_env(budget),
        )
    text = log.read_text()
    if proc.returncode != 0:
        raise RuntimeError(f"anneal LJ38 seed {seed} exited {proc.returncode}\n{text[-500:]}")
    row = parse_anneal(text)
    if int(row["budget"]) != budget:
        raise RuntimeError(f"anneal budget {row['budget']} != {budget}")
    return with_rates(row)


def read_reduced(path: Path) -> list[list[float]]:
    atoms = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        atoms.append([float(part) for part in parts])
    if len(atoms) != 38:
        raise RuntimeError(f"{path} has {len(atoms)} atoms")
    return atoms


def write_init(binary: Path, seed: int, path: Path) -> None:
    reduced_path = path.with_suffix(".reduced")
    subprocess.run(
        [str(binary), "coords", "lj38", str(seed), str(reduced_path)],
        check=True,
    )
    atoms = read_reduced(reduced_path)
    lines = [str(len(atoms)), "# angstrom, anneal start times the RPQA sigma"]
    for x, y, z in atoms:
        lines.append(
            f"H {x * SIGMA_ANGSTROM:.10f} {y * SIGMA_ANGSTROM:.10f} {z * SIGMA_ANGSTROM:.10f}"
        )
    path.write_text("\n".join(lines) + "\n")


def write_input(
    path: Path,
    seed: int,
    address: str,
    nbeads: int = NBEADS,
    total_steps: int = TOTAL_STEPS,
    pin_interval: int = PIN_INTERVAL,
    start_step: int = START_STEP,
    ramp_steps: int = RAMP_STEPS,
    checkpoint: int = 30000,
) -> None:
    path.write_text(
        f"""<simulation mode='static' verbosity='low'>
  <output prefix='rpqa'>
    <properties stride='5000' filename='out'> [ step, time{{picosecond}}, potential{{electronvolt}}, lambdaqkin ] </properties>
    <checkpoint stride='{checkpoint}'/>
  </output>
  <prng>
    <seed>{seed}</seed>
  </prng>
  <total_steps>{total_steps}</total_steps>
  <ffsocket mode='unix' name='lj_driver' pbc='False'>
    <address>{address}</address>
    <latency>0.0001</latency>
  </ffsocket>
  <system>
    <initialize nbeads='{nbeads}'>
      <file mode='xyz' units='angstrom'> init.xyz </file>
      <velocities mode='thermal' units='kelvin'> 50 </velocities>
      <cell mode='abc'> [300.0, 300.0, 300.0] </cell>
    </initialize>
    <forces>
      <force forcefield='lj_driver'> </force>
    </forces>
    <ensemble>
      <temperature units='kelvin'> 50.0 </temperature>
      <lambdaqkin> 19.0 </lambdaqkin>
    </ensemble>
    <normal_modes propagator='bab'/>
    <motion mode='multi'>
      <motion mode='dynamics'>
        <fixcom>False</fixcom>
        <dynamics mode='nvt'>
          <thermostat mode='langevin'>
            <tau units='femtosecond'> 20 </tau>
          </thermostat>
          <timestep units='femtosecond'> 1.0 </timestep>
        </dynamics>
      </motion>
      <motion mode='qkin_ramp'>
        <qkin_ramp>
          <lambda_start> 19.0 </lambda_start>
          <lambda_end> 0.01 </lambda_end>
          <total_steps> {ramp_steps} </total_steps>
          <logscale> False </logscale>
          <sqrtscale> True </sqrtscale>
        </qkin_ramp>
      </motion>
    </motion>
  </system>
  <smotion mode='rpqa'>
    <rpqa>
      <pinning_interval> {pin_interval} </pinning_interval>
      <start_step> {start_step} </start_step>
      <max_relax_steps> 5000 </max_relax_steps>
      <optimizer mode='lbfgs'>
        <tolerances>
          <energy> 1e-5 </energy>
          <force> 1e-3 </force>
          <position> 5e-3 </position>
        </tolerances>
      </optimizer>
    </rpqa>
  </smotion>
</simulation>
"""
    )


def _terminate(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run_rpqa(
    binary: Path,
    seed: int,
    budget: int,
    ipi_root: Path,
    script_dir: Path,
    work: Path,
    log: Path,
    arm: str = "rpqa",
    nbeads: int = NBEADS,
    total_steps: int = TOTAL_STEPS,
    pin_interval: int = PIN_INTERVAL,
    start_step: int = START_STEP,
    ramp_steps: int = RAMP_STEPS,
    expect_pins: int | None = None,
    code_revision: str = "",
) -> dict[str, str]:
    work.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    address = f"{arm}_lj38_{seed}"
    socket_path = Path(f"/tmp/ipi_{address}")
    if socket_path.exists():
        socket_path.unlink()
    write_init(binary, seed, work / "init.xyz")
    write_input(
        work / "input.xml",
        seed,
        address,
        nbeads=nbeads,
        total_steps=total_steps,
        pin_interval=pin_interval,
        start_step=start_step,
        ramp_steps=ramp_steps,
        checkpoint=100000 if arm == PAPER_ARM else 30000,
    )
    meter_path = work / "meter.bin"
    if meter_path.exists():
        meter_path.unlink()
    Meter(str(meter_path), create=True).close()
    env = _env(budget)
    env["IPI_ROOT"] = str(ipi_root)
    env["RPQA_METER"] = str(meter_path)
    env["RPQA_BUDGET"] = str(budget)
    env["RPQA_LEVEL_HA"] = f"{(REFERENCE + HIT_TOL) * EPSILON:.16e}"
    env["RPQA_ADDRESS"] = address
    env["RPQA_SOCKET_PREFIX"] = "/tmp/ipi_"
    env["PYTHONPATH"] = str(script_dir)
    env["PYTHONUNBUFFERED"] = "1"
    with log.open("w") as handle:
        started = time.perf_counter()
        ipi = subprocess.Popen(
            [sys.executable, str(script_dir / "rpqa_ipi.py"), "input.xml"],
            cwd=work,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 120.0
        while not socket_path.exists():
            if ipi.poll() is not None:
                raise RuntimeError(f"i-PI exited {ipi.returncode} before the socket\n{log.read_text()[-800:]}")
            if time.monotonic() > deadline:
                _terminate(ipi)
                raise RuntimeError(f"i-PI did not open {socket_path}\n{log.read_text()[-800:]}")
            time.sleep(0.05)
        driver = subprocess.Popen(
            [sys.executable, str(script_dir / "rpqa_driver.py")],
            cwd=work,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        watch = Meter(str(meter_path))
        while True:
            capped = budget > 0 and watch.calls() >= budget
            if capped or watch.hit_calls() > 0:
                time.sleep(0.05)
                _terminate(ipi)
                _terminate(driver)
                break
            if ipi.poll() is not None:
                _terminate(driver)
                break
            if driver.poll() is not None:
                _terminate(ipi)
                break
            time.sleep(0.02)
        watch.close()
        wall = time.perf_counter() - started
    meter = Meter(str(meter_path))
    charged = meter.calls()
    pot_s = meter.pot_ns() / 1e9
    n_pins = meter.n_pins()
    hit_calls = meter.hit_calls()
    best_ha = meter.best()
    meter.close()
    if socket_path.exists():
        socket_path.unlink()
    if charged < 1 or (budget > 0 and charged > budget):
        raise RuntimeError(f"RPQA seed {seed} charged {charged} outside 1..{budget}\n{log.read_text()[-800:]}")
    hit = hit_calls > 0
    if hit and (hit_calls > charged or best_ha > (REFERENCE + HIT_TOL) * EPSILON):
        raise RuntimeError(f"RPQA seed {seed} hit record is inconsistent")
    if budget > 0 and not hit and charged != budget:
        raise RuntimeError(
            f"RPQA seed {seed} stopped at {charged} calls without a hit\n{log.read_text()[-800:]}"
        )
    if budget == 0 and not hit and (expect_pins is None or n_pins != expect_pins):
        raise RuntimeError(
            f"RPQA seed {seed} finished {n_pins} pinning rounds, expected {expect_pins}\n{log.read_text()[-800:]}"
        )
    best = ""
    if n_pins:
        best = f"{best_ha / EPSILON:.8f}"
    if hit:
        stop = "hit"
        evals = hit_calls
    elif budget > 0:
        stop = "budget"
        evals = budget
    else:
        stop = "schedule"
        evals = charged
    row = {
        "system": "lj38",
        "arm": arm,
        "seed": str(seed),
        "budget": str(budget if budget > 0 else 0),
        "hit": "1" if hit else "0",
        "evals_to_hit": str(evals),
        "charged": str(charged),
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot_s:.6f}",
        "best": best,
        "stop": stop,
        "reached": "38",
        "code_revision": code_revision,
        "_pins": str(n_pins),
    }
    rated = with_rates(row)
    rated["_pins"] = row["_pins"]
    return rated


def check_potential(binary: Path, work: Path) -> None:
    from lj_potential import self_test

    self_test()
    work.mkdir(parents=True, exist_ok=True)
    reduced = work / "check.reduced"
    subprocess.run([str(binary), "coords", "lj38", "2000", str(reduced)], check=True)
    atoms = read_reduced(reduced)
    ours = reduced_pair_energy(atoms)
    proc = subprocess.run(
        [str(binary), "energy", "lj38", str(reduced)],
        check=True,
        capture_output=True,
        text=True,
    )
    theirs = float(proc.stdout.strip().split()[0])
    if abs(ours - theirs) > 1e-8:
        raise SystemExit(f"LJ pair sum {ours} != anneal {theirs}")


def summarise(rows: list[dict[str, str]], seeds: tuple[int, ...], source: dict[str, str], head: str):
    by_arm: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    seed_text = " ".join(str(seed) for seed in seeds)
    walls = {
        arm: median_iqr([float(item["wall_s"]) for item in group])[0]
        for arm, group in by_arm.items()
        if group
    }
    ratio = ""
    if "anneal" in walls and "rpqa" in walls and walls["anneal"] > 0.0:
        ratio = f"{walls['rpqa'] / walls['anneal']:.6f}"
    out = []
    for arm in ARMS:
        group = by_arm.get(arm, [])
        if not group:
            continue
        evals_median, evals_iqr = median_iqr([float(item["evals_to_hit"]) for item in group])
        wall_median, wall_iqr = median_iqr([float(item["wall_s"]) for item in group])
        inside_median, _inside_iqr = median_iqr([float(item["inside_us"]) for item in group])
        outside_median, _outside_iqr = median_iqr([float(item["outside_us"]) for item in group])
        hits = sum(item["hit"] == "1" for item in group)
        out.append(
            {
                "system": "lj38",
                "arm": arm,
                "n": str(len(group)),
                "planned": str(len(seeds)),
                "hits": str(hits),
                "hits_over_n": f"{hits}/{len(group)}",
                "budget": group[0]["budget"],
                "evals_median": f"{evals_median:.6f}",
                "evals_iqr": f"{evals_iqr:.6f}",
                "wall_median_s": f"{wall_median:.6f}",
                "wall_iqr_s": f"{wall_iqr:.6f}",
                "wall_ratio": ratio,
                "inside_us": f"{inside_median:.4f}",
                "outside_us": f"{outside_median:.4f}",
                "seeds": seed_text,
                "schedule_seed": str(SCHEDULE_SEED),
                "rpqa_commit": source["rpqa_commit"],
                "rpqa_source": source["rpqa_source"],
                "anneal_head": head,
                "code_revision": source["rpqa_commit"],
            }
        )
    return out


def write_table(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def note(progress: Path, line: str) -> None:
    print(line, flush=True)
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a") as handle:
        handle.write(line + "\n")


def paper_pin_steps() -> list[int]:
    """Step indices at which the published LJ38 schedule pins a replica."""
    return [
        step
        for step in range(PAPER_TOTAL_STEPS)
        if step >= PAPER_START and (step - PAPER_START) % PAPER_PIN_INTERVAL == 0
    ]


def load_table(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_missing(
    path: Path,
    fields: list[str],
    updates: list[dict[str, str]],
    key_fields: tuple[str, ...],
) -> None:
    """Append paper rows that are not already in ``path``. Existing lines stay byte for byte."""
    existing = load_table(path)
    have = {tuple(row[key] for key in key_fields) for row in existing}
    fresh = []
    for row in updates:
        if row["arm"] != PAPER_ARM:
            raise RuntimeError(f"refusing to write arm {row['arm']}")
        stored = {key: row[key] for key in fields}
        key = tuple(stored[name] for name in key_fields)
        if key in have:
            continue
        fresh.append(stored)
        have.add(key)
    if not fresh:
        return
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not existing:
            writer.writeheader()
        writer.writerows(fresh)


def upsert_summary(path: Path, fields: list[str], row: dict[str, str]) -> None:
    """Replace only the paper-arm summary line. Other lines are copied unchanged."""
    raw = path.read_text() if path.is_file() else ""
    kept = [line for line in raw.splitlines(keepends=True) if not line.startswith(f"lj38,{PAPER_ARM},")]
    if kept and not kept[-1].endswith("\n"):
        kept[-1] += "\n"
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    if not raw:
        writer.writeheader()
    writer.writerow({key: row[key] for key in fields})
    path.write_text("".join(kept) + buffer.getvalue())


def anneal_reference(summary_path: Path) -> tuple[float, str]:
    """Wall median and anneal commit of the LJ38 anneal row the ratio divides by."""
    for row in load_table(summary_path):
        if row.get("arm") == "anneal" and row.get("system") == "lj38":
            return float(row["wall_median_s"]), row["anneal_head"]
    raise RuntimeError(f"{summary_path} has no LJ38 anneal row")


def summarise_paper(
    rows: list[dict[str, str]],
    planned: int,
    anneal_wall: float,
    source: dict[str, str],
    anneal_head: str,
) -> list[dict[str, str]]:
    if not rows:
        return []
    evals_median, evals_iqr = median_iqr([float(item["evals_to_hit"]) for item in rows])
    wall_median, wall_iqr = median_iqr([float(item["wall_s"]) for item in rows])
    inside_median, _inside_iqr = median_iqr([float(item["inside_us"]) for item in rows])
    outside_median, _outside_iqr = median_iqr([float(item["outside_us"]) for item in rows])
    hits = sum(item["hit"] == "1" for item in rows)
    ratio = f"{wall_median / anneal_wall:.6f}" if anneal_wall > 0.0 else ""
    return [
        {
            "system": "lj38",
            "arm": PAPER_ARM,
            "n": str(len(rows)),
            "planned": str(planned),
            "hits": str(hits),
            "hits_over_n": f"{hits}/{len(rows)}",
            "budget": "0",
            "evals_median": f"{evals_median:.6f}",
            "evals_iqr": f"{evals_iqr:.6f}",
            "wall_median_s": f"{wall_median:.6f}",
            "wall_iqr_s": f"{wall_iqr:.6f}",
            "wall_ratio": ratio,
            "inside_us": f"{inside_median:.4f}",
            "outside_us": f"{outside_median:.4f}",
            "seeds": " ".join(sorted((item["seed"] for item in rows), key=int)),
            "schedule_seed": str(PAPER_SCHEDULE_SEED),
            "rpqa_commit": source["rpqa_commit"],
            "rpqa_source": source["rpqa_source"],
            "anneal_head": anneal_head,
            "code_revision": source["rpqa_commit"],
        }
    ]


def self_test() -> None:
    from lj_potential import self_test as lj_self_test

    lj_self_test()
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.25) - 2.25) > 1e-12:
        raise SystemExit("type 7 lower quartile")
    if abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7 median")
    median, iqr = median_iqr(sample)
    if abs(median - 3.5) > 1e-12 or abs(iqr - 2.5) > 1e-12:
        raise SystemExit("median iqr")
    if PAPER_BUDGETS != {38: 400_000}:
        raise SystemExit("paper budget")
    if abs(HIT_TOL - 0.001) > 1e-15 or abs(REFERENCE + 173.928427) > 1e-12:
        raise SystemExit("cambridge level")
    inside, outside = per_eval(1.0, 0.25, 1000)
    if abs(inside - 250.0) > 1e-9 or abs(outside - 750.0) > 1e-9:
        raise SystemExit("per-evaluation split")
    order = schedule(SEEDS, ARMS, SCHEDULE_SEED)
    jobs = sorted(order)
    expected = sorted((seed, arm) for seed in SEEDS for arm in ARMS)
    if jobs != expected:
        raise SystemExit("seed list")
    arms = [arm for _seed, arm in order]
    if arms == ["anneal"] * len(SEEDS) + ["rpqa"] * len(SEEDS):
        raise SystemExit("not interleaved")
    if arms == ["rpqa"] * len(SEEDS) + ["anneal"] * len(SEEDS):
        raise SystemExit("not interleaved")
    level = (REFERENCE + HIT_TOL) * EPSILON
    if not (REFERENCE * EPSILON < level < 0.0):
        raise SystemExit("hit level is on the wrong side of the minimum")
    pins = paper_pin_steps()
    if (len(pins), pins[0], pins[-1], PAPER_TOTAL_STEPS) != (40, 10_000, 400_000, 400_001):
        raise SystemExit(f"paper schedule {len(pins)} {pins[:1]} {pins[-1:]} {PAPER_TOTAL_STEPS}")
    if PAPER_NBEADS != 32 or PAPER_LAST_PIN != 400_000:
        raise SystemExit("paper replicas or last pin")
    paper_jobs = list(SEEDS)
    random.Random(PAPER_SCHEDULE_SEED).shuffle(paper_jobs)
    if sorted(paper_jobs) != list(SEEDS) or paper_jobs == list(SEEDS):
        raise SystemExit("paper seed order")
    timed = [
        {"seed": str(seed), "evals_to_hit": "1", "wall_s": "2.0", "inside_us": "1", "outside_us": "1", "hit": "0"}
        for seed in paper_jobs
    ]
    paper_row = summarise_paper(timed, len(SEEDS), 0.5, {"rpqa_commit": "r", "rpqa_source": "s"}, "a")[0]
    if paper_row["seeds"] != " ".join(str(seed) for seed in SEEDS):
        raise SystemExit("paper summary seed order")
    if (paper_row["anneal_head"], paper_row["code_revision"], paper_row["wall_ratio"]) != ("a", "r", "4.000000"):
        raise SystemExit("paper summary provenance")
    print("ok")


def seed_path(rows: Path, seed: int, arm: str) -> Path:
    return rows / "lj38" / f"{arm}-{seed}.row"


def run_paper(args: argparse.Namespace, seeds: tuple[int, ...]) -> int:
    """One serial session of the published LJ38 step schedule.

    Existing anneal and capped-RPQA rows stay on disk. This writes only
    ``rpqa_paper`` rows and appends that arm to the summary tables.
    """
    for seed in seeds:
        if seed not in SEEDS:
            raise SystemExit(f"seed {seed} is outside 2000-2005")
    source = load_source(args.source)
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    calls, reference = binary_budget(args.binary)
    budget = PAPER_BUDGETS[38]
    if calls != budget or abs(reference - REFERENCE) > 1e-12:
        raise SystemExit(f"binary budget {calls} {reference} != paper {budget} {REFERENCE}")
    check_potential(args.binary, args.runs / "check")
    if args.summary is None:
        raise SystemExit("the paper schedule needs --summary so the old rows can be kept")
    anneal_wall, anneal_head = anneal_reference(args.summary)
    order = list(seeds)
    random.Random(PAPER_SCHEDULE_SEED).shuffle(order)
    if seeds != SEEDS:
        full = list(SEEDS)
        random.Random(PAPER_SCHEDULE_SEED).shuffle(full)
        order = [seed for seed in full if seed in seeds]
    progress = args.runs.parent / "progress-paper.txt"
    note(
        progress,
        "paper schedule "
        f"beads={PAPER_NBEADS} rounds={PAPER_ROUNDS} every={PAPER_PIN_INTERVAL} "
        f"last_pin={PAPER_LAST_PIN} total_steps={PAPER_TOTAL_STEPS} "
        + " ".join(f"lj38:{PAPER_ARM}:{seed}" for seed in order),
    )
    rows: list[dict[str, str]] = []
    for seed in order:
        saved = seed_path(args.rows, seed, PAPER_ARM)
        log = args.runs / "lj38" / f"{PAPER_ARM}-{seed}.log"
        existing = load_seed(saved)
        if existing is not None:
            rows.append(existing)
            note(progress, f"skip arm={PAPER_ARM} seed={seed} hit={existing['hit']} evals={existing['evals_to_hit']}")
            continue
        note(progress, f"start arm={PAPER_ARM} seed={seed} steps={PAPER_TOTAL_STEPS}")
        row = run_rpqa(
            args.binary,
            seed,
            0,
            args.ipi_root,
            here,
            args.runs / "lj38" / f"{PAPER_ARM}-{seed}",
            log,
            arm=PAPER_ARM,
            nbeads=PAPER_NBEADS,
            total_steps=PAPER_TOTAL_STEPS,
            pin_interval=PAPER_PIN_INTERVAL,
            start_step=PAPER_START,
            ramp_steps=PAPER_RAMP_STEPS,
            expect_pins=PAPER_ROUNDS,
            code_revision=source["rpqa_commit"],
        )
        pins = row.pop("_pins")
        write_seed(saved, row)
        rows.append(row)
        note(
            progress,
            f"done arm={PAPER_ARM} seed={seed} hit={row['hit']} "
            f"evals={row['evals_to_hit']} charged={row['charged']} "
            f"wall_s={row['wall_s']} best={row['best']} stop={row['stop']} pins={pins}",
        )
        if args.seeds_out:
            append_missing(args.seeds_out, SEED_FIELDS, rows, ("arm", "seed"))
        if args.paper_seeds:
            write_table(args.paper_seeds, SEED_FIELDS, rows)
        summary = summarise_paper(rows, len(SEEDS), anneal_wall, source, anneal_head)
        if args.summary and summary:
            upsert_summary(args.summary, SUMMARY_FIELDS, summary[0])
        if args.paper_summary:
            write_table(args.paper_summary, SUMMARY_FIELDS, summary)
    summary = summarise_paper(rows, len(SEEDS), anneal_wall, source, anneal_head)
    writer = csv.DictWriter(sys.stdout, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(summary)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ipi-root", type=Path, required=True)
    parser.add_argument("--head-file", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--seeds-out", type=Path)
    parser.add_argument("--size", type=int, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--arm", choices=("anneal", "rpqa", "both", PAPER_ARM), default="both")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--paper-summary", type=Path)
    parser.add_argument("--paper-seeds", type=Path)
    args = parser.parse_args(argv[1:])
    if args.size is not None and args.size != [38]:
        raise SystemExit(f"this bench is LJ38, not {args.size}")
    seeds = tuple(args.seed) if args.seed else SEEDS
    if args.arm == PAPER_ARM:
        if args.budget is not None:
            raise SystemExit("the LJ38 paper schedule is not call-capped")
        return run_paper(args, seeds)
    arms = ARMS if args.arm == "both" else (args.arm,)
    budget = PAPER_BUDGETS[38]
    if args.budget is not None and args.budget != budget:
        raise SystemExit(f"LJ38 budget {args.budget} is not the paper budget {budget}")
    for seed in seeds:
        if seed not in SEEDS:
            raise SystemExit(f"seed {seed} is outside 2000-2005")
    source = load_source(args.source)
    head = args.head_file.read_text().strip()
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    calls, reference = binary_budget(args.binary)
    if calls != budget or abs(reference - REFERENCE) > 1e-12:
        raise SystemExit(f"binary budget {calls} {reference} != paper {budget} {REFERENCE}")
    check_potential(args.binary, args.runs / "check")
    progress = args.runs.parent / "progress.txt"
    order = [
        (seed, arm)
        for seed, arm in schedule(SEEDS, ARMS, SCHEDULE_SEED)
        if seed in seeds and arm in arms
    ]
    note(progress, "schedule " + " ".join(f"lj38:{arm}:{seed}" for seed, arm in order))
    rows: list[dict[str, str]] = []
    produced = False
    for seed, arm in order:
        saved = seed_path(args.rows, seed, arm)
        log = args.runs / "lj38" / f"{arm}-{seed}.log"
        existing = load_seed(saved)
        if existing is not None:
            rows.append(existing)
            note(progress, f"skip arm={arm} seed={seed} hit={existing['hit']} evals={existing['evals_to_hit']}")
            continue
        note(progress, f"start arm={arm} seed={seed} budget={budget}")
        if arm == "anneal":
            row = run_anneal(args.binary, seed, budget, log)
        else:
            row = run_rpqa(
                args.binary,
                seed,
                budget,
                args.ipi_root,
                here,
                args.runs / "lj38" / f"rpqa-{seed}",
                log,
                code_revision=source["rpqa_commit"],
            )
        write_seed(saved, row)
        produced = True
        rows.append(row)
        note(
            progress,
            f"done arm={arm} seed={seed} hit={row['hit']} evals={row['evals_to_hit']} "
            f"charged={row['charged']} wall_s={row['wall_s']} pot_s={row['pot_s']} "
            f"best={row['best']} stop={row['stop']} inside_us={row['inside_us']} "
            f"outside_us={row['outside_us']}",
        )
        if args.seeds_out:
            write_table(args.seeds_out, SEED_FIELDS, rows)
        if args.summary:
            write_table(args.summary, SUMMARY_FIELDS, summarise(rows, seeds, source, head))
    text_rows = summarise(rows, seeds, source, head)
    if args.seeds_out:
        write_table(args.seeds_out, SEED_FIELDS, rows)
    if args.summary and (produced or not args.summary.exists()):
        write_table(args.summary, SUMMARY_FIELDS, text_rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerows(text_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
