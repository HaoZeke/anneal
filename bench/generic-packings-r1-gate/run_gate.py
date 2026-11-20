# Copyright (c) anneal developers
"""Four-arm same-session gate on the paper panel.

The panel is every cell of the paper head-to-head tables that the anneal
drivers run: the GMIN panel of ``scripts/gmin_panel/run_panel.py`` (LJ38 to
LJ150, Au55, M55 and M80 at rho 6 and 14), the Morse cells of
``scripts/morse_h2h/run_campaign.py`` not already in it (M38 at rho 14 and
6), and the twelve-seed Sutton-Chen cells of ``scripts/metal_h2h/run_sc.py``.
Budgets and references are those tables' ``PAPER_BUDGETS`` and reference
energies; each binary's own ``budget <system>`` must agree before the run.

Every arm is a release build of its own revision with its own examples:
``h2h_gmin_proof`` for Lennard-Jones, ``h2h_metal`` for Au55, ``h2h_morse``
for Morse and ``h2h_sc`` for Sutton-Chen. The command is ``run <system>
<seed>``; no ``H2H_*`` or ``ANNEAL_*`` variable reaches a run, so the budget
is the driver's paper budget and the only stop besides it is the target.

Seeds 2000-2011. For each cell and seed every arm runs once, in the arm
order given on even seed offsets and reversed on odd ones (ABCD DCBA). One
process, ``taskset -c 0``, every thread count 1. Rows are appended and
fsynced after every run. A row carries the session, a sha256 of the machine
id, the cpu model, the revision and tree of the arm's source, and the
sha256 of the binary.

``--summarize`` reads the rows and writes the per-cell table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


panel = load("gate_gmin_panel", ROOT / "scripts" / "gmin_panel" / "run_panel.py")
morse = load("gate_morse_h2h", ROOT / "scripts" / "morse_h2h" / "run_campaign.py")
sc = load("gate_metal_sc", ROOT / "scripts" / "metal_h2h" / "run_sc.py")

SEEDS = tuple(range(2000, 2012))
HIT_TOL = 1.0e-3

# (system, driver, budget, reference)
CELLS: list[tuple[str, str, int, float]] = []
for _name, _kind, _n, _ref, _rho, _driver, _binary in panel.SYSTEMS:
    CELLS.append((_name, _driver.split("+")[0], panel.PAPER_BUDGETS[_name], _ref))
for _name, _n, _rho, _budget, _ref in morse.SYSTEMS:
    if _name not in panel.PAPER_BUDGETS:
        CELLS.append((_name, "h2h_morse", _budget, _ref))
for _name, _budget in sc.PAPER_BUDGETS.items():
    CELLS.append((_name, "h2h_sc", _budget, sc.REFERENCES[_name]))
CELL = {c[0]: c for c in CELLS}

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "TBB_NUM_THREADS",
)

ROW_FIELDS = [
    "session",
    "system",
    "driver",
    "arm",
    "seed",
    "slot",
    "budget",
    "reference",
    "hit",
    "driver_hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "best",
    "pot_s",
    "outside_s",
    "outside_per_call_s",
    "code_revision",
    "code_tree",
    "binary_sha256",
    "harness_sha256",
    "machine",
    "cpu",
    "utc",
]


def hits(best: float, reference: float) -> bool:
    return math.isfinite(best) and best <= reference + HIT_TOL


def run_env(sc_lib: str | None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("H2H_", "ANNEAL_"))}
    for key in THREAD_VARS:
        env[key] = "1"
    if sc_lib:
        env["RGPOT_SC_LIB"] = sc_lib
    return env


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def machine_hash() -> str:
    for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if path.is_file() and path.read_text().strip():
            return hashlib.sha256(path.read_text().strip().encode()).hexdigest()
    raise SystemExit("no machine id")


def git(tree: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=tree, capture_output=True, text=True, check=True).stdout.strip()


def arm_order(arms: list[str], seed: int) -> list[str]:
    return list(arms) if (seed - SEEDS[0]) % 2 == 0 else list(reversed(arms))


def check_budget(binary: Path, system: str, budget: int, reference: float, env: dict[str, str]) -> None:
    proc = subprocess.run([str(binary), "budget", system], env=env, capture_output=True, text=True, check=False)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not lines:
        raise SystemExit(f"{binary} budget {system} exited {proc.returncode}: {proc.stderr[-500:]}")
    if int(float(lines[0])) != budget:
        raise SystemExit(f"{binary} {system}: budget {lines[0]} is not {budget}")
    if len(lines) > 1 and abs(float(lines[1]) - reference) > 1e-6:
        raise SystemExit(f"{binary} {system}: reference {lines[1]} is not {reference}")


def parse_row(text: str, system: str, seed: int, budget: int) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith("system,code,seed,")]
    data = [line for line in text.splitlines() if line.startswith(f"{system},anneal,{seed},")]
    if len(lines) < 1 or len(data) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one result row\n{text[-2000:]}")
    header = lines[-1].split(",")
    row = dict(zip(header, data[0].split(",")))
    if int(float(row["budget"])) != budget:
        raise RuntimeError(f"{system} seed {seed}: budget {row['budget']}")
    return row


def run_one(binary: Path, cell: tuple[str, str, int, float], seed: int, env: dict[str, str], log: Path) -> dict[str, str]:
    system, _driver, budget, reference = cell
    command = ["taskset", "-c", "0", str(binary), "run", system, str(seed)]
    proc = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(command) + "\n" + proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{binary} {system} {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    row = parse_row(proc.stdout + proc.stderr, system, seed, budget)
    best = float(row["best"])
    hit = hits(best, reference)
    driver_hit = row["hit"].strip().lower() in ("1", "true")
    evals = int(float(row["evals_to_hit"]))
    charged = int(float(row["charged"]))
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    outside = float(row["outside_s"]) if "outside_s" in row else max(wall - pot, 0.0)
    if hit != driver_hit:
        raise RuntimeError(f"{system} {seed}: driver hit {driver_hit} but best {best} against {reference}")
    if hit and not 0 < evals <= charged <= budget:
        raise RuntimeError(f"{system} {seed}: hit with evals {evals} charged {charged}")
    if not hit and (evals != budget or charged != budget):
        raise RuntimeError(f"{system} {seed}: miss stopped at evals {evals} charged {charged} of {budget}")
    return {
        "hit": "1" if hit else "0",
        "driver_hit": "1" if driver_hit else "0",
        "evals_to_hit": str(evals),
        "charged": str(charged),
        "wall_s": f"{wall:.6f}",
        "best": row["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
        "outside_per_call_s": f"{outside / max(charged, 1):.6e}",
    }


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run(args) -> int:
    if os.sched_getaffinity(0) != {0}:
        print(f"pinned to {sorted(os.sched_getaffinity(0))}, expected core 0", file=sys.stderr)
        return 2
    arms = [spec.split("=", 1) for spec in args.arm]
    names = [name for name, _ in arms]
    if len(set(names)) != len(names):
        raise SystemExit("arm names repeat")
    sources = {name: Path(src) for name, src in arms}
    bins = {name: Path(args.bin_root) / f"{name}-bin" for name in names}
    env = run_env(args.sc_lib)
    systems = args.system or [c[0] for c in CELLS]
    provenance = {}
    for name in names:
        tree = sources[name]
        if git(tree, "status", "--porcelain"):
            raise SystemExit(f"arm {name} source is not clean")
        provenance[name] = (git(tree, "rev-parse", "HEAD"), git(tree, "rev-parse", "HEAD^{tree}"))
        for system in systems:
            _s, driver, budget, reference = CELL[system]
            check_budget(bins[name] / driver, system, budget, reference, env)
    machine = machine_hash()
    cpu = panel.cpu_model()
    harness = sha256_file(Path(__file__))
    session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    done = {(r["system"], r["arm"], r["seed"]) for r in read_rows(args.rows)}
    print(f"session {session} arms {' '.join(names)} systems {' '.join(systems)}", flush=True)
    for name in names:
        print(f"arm {name} revision {provenance[name][0]} tree {provenance[name][1]}", flush=True)
    for system in systems:
        cell = CELL[system]
        for seed in SEEDS:
            for slot, name in enumerate(arm_order(names, seed)):
                if (system, name, str(seed)) in done:
                    print(f"skip {system} {name} {seed}", flush=True)
                    continue
                binary = bins[name] / cell[1]
                print(f"start {system} {name} {seed} slot={slot}", flush=True)
                result = run_one(binary, cell, seed, env, Path(args.logs) / system / f"{name}-{seed}.log")
                row = {
                    "session": session,
                    "system": system,
                    "driver": cell[1],
                    "arm": name,
                    "seed": str(seed),
                    "slot": str(slot),
                    "budget": str(cell[2]),
                    "reference": f"{cell[3]:.8f}",
                    **result,
                    "code_revision": provenance[name][0],
                    "code_tree": provenance[name][1],
                    "binary_sha256": sha256_file(binary),
                    "harness_sha256": harness,
                    "machine": machine,
                    "cpu": cpu,
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                append_row(Path(args.rows), row)
                print(
                    f"done {system} {name} {seed} slot={slot} hit={row['hit']} calls={row['evals_to_hit']} "
                    f"wall_s={row['wall_s']} outside_s={row['outside_s']} best={row['best']}",
                    flush=True,
                )
    print("session done", flush=True)
    return 0


def ranked_calls(group: list[dict[str, str]]) -> list[float]:
    return [float(r["evals_to_hit"]) if r["hit"] == "1" else math.inf for r in group]


def show(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


def summarize(args) -> int:
    rows = read_rows(Path(args.rows))
    cand, base, aa, ref = args.candidate, args.baseline, args.aa, args.reference
    by: dict[tuple[str, str], dict[int, dict[str, str]]] = {}
    for r in rows:
        by.setdefault((r["system"], r["arm"]), {})[int(r["seed"])] = r
    fields = [
        "system", "budget", "hits_G", "hits_D", "hits_Dp", "hits_P", "fisher_p_G_D", "hit_loss",
        "calls_median_G", "calls_median_D", "mw_p_calls_G_D", "calls_loss",
        "outside_per_call_median_G", "outside_per_call_median_D", "outside_per_call_median_Dp",
        "outside_ratio_G_D", "aa_ratio_Dp_D", "outside_ok",
        "p_hits_recovered_by_G", "cell_pass",
    ]
    out = []
    totals = {"G": 0, "D": 0, "P": 0, "recovered": 0, "p_only": 0}
    for system, _driver, budget, _reference in CELLS:
        groups = {}
        for label, name in (("G", cand), ("D", base), ("Dp", aa), ("P", ref)):
            group = by.get((system, name), {})
            groups[label] = [group[s] for s in SEEDS if s in group]
        if any(len(g) != len(SEEDS) for g in groups.values()):
            out.append({"system": system, "budget": str(budget), "cell_pass": "incomplete"})
            continue
        h = {k: sum(r["hit"] == "1" for r in g) for k, g in groups.items()}
        fisher_p = panel.fisher(h["G"], len(SEEDS), h["D"], len(SEEDS))
        calls_g, calls_d = ranked_calls(groups["G"]), ranked_calls(groups["D"])
        _u, mw_p = panel.mann_whitney(calls_g, calls_d)
        med_g = panel.type7([float(r["evals_to_hit"]) for r in groups["G"]], 0.5)
        med_d = panel.type7([float(r["evals_to_hit"]) for r in groups["D"]], 0.5)
        calls_loss = mw_p < 0.05 and sum(rank_side(calls_g, calls_d)) > 0
        opc = {k: panel.type7([float(r["outside_per_call_s"]) for r in g], 0.5) for k, g in groups.items()}
        ratio = opc["G"] / opc["D"]
        aa_ratio = opc["Dp"] / opc["D"]
        band = max(aa_ratio, 1.0 / aa_ratio)
        outside_ok = ratio <= 1.10 or ratio <= band
        p_seeds = {r["seed"] for r in groups["P"] if r["hit"] == "1"}
        g_seeds = {r["seed"] for r in groups["G"] if r["hit"] == "1"}
        recovered = len(p_seeds & g_seeds)
        totals["G"] += h["G"]
        totals["D"] += h["D"]
        totals["P"] += h["P"]
        totals["recovered"] += recovered
        hit_loss = h["G"] < h["D"]
        out.append(
            {
                "system": system,
                "budget": str(budget),
                "hits_G": str(h["G"]),
                "hits_D": str(h["D"]),
                "hits_Dp": str(h["Dp"]),
                "hits_P": str(h["P"]),
                "fisher_p_G_D": f"{fisher_p:.6g}",
                "hit_loss": "1" if hit_loss else "0",
                "calls_median_G": f"{med_g:.1f}",
                "calls_median_D": f"{med_d:.1f}",
                "mw_p_calls_G_D": f"{mw_p:.6g}",
                "calls_loss": "1" if calls_loss else "0",
                "outside_per_call_median_G": f"{opc['G']:.6e}",
                "outside_per_call_median_D": f"{opc['D']:.6e}",
                "outside_per_call_median_Dp": f"{opc['Dp']:.6e}",
                "outside_ratio_G_D": f"{ratio:.4f}",
                "aa_ratio_Dp_D": f"{aa_ratio:.4f}",
                "outside_ok": "1" if outside_ok else "0",
                "p_hits_recovered_by_G": f"{recovered}/{len(p_seeds)}",
                "cell_pass": "1" if not hit_loss and not calls_loss and outside_ok else "0",
            }
        )
    with Path(args.summary).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    complete = all(r["cell_pass"] in ("0", "1") for r in out)
    verdict = "PASS" if complete and all(r["cell_pass"] == "1" for r in out) else "FAIL"
    print(f"verdict {verdict} complete={int(complete)} totals {totals}")
    return 0


def rank_side(left: list[float], right: list[float]) -> list[int]:
    """+1 per cross pair where left is worse, -1 where better."""
    return [(a > b) - (a < b) for a in left for b in right]


def self_test() -> None:
    assert hits(-1.0, -1.0005) and not hits(-1.0, -1.002) and not hits(float("nan"), 0.0)
    assert arm_order(["A", "B", "C", "D"], 2000) == ["A", "B", "C", "D"]
    assert arm_order(["A", "B", "C", "D"], 2001) == ["D", "C", "B", "A"]
    slots = {}
    for seed in SEEDS:
        for slot, name in enumerate(arm_order(["A", "B", "C", "D"], seed)):
            slots.setdefault(name, []).append(slot)
    assert all(sum(v) == sum(slots["A"]) for v in slots.values())
    assert len(CELLS) == len(CELL)
    assert CELL["m38r14"][2] == 400_000 and CELL["sc108-38"][2] == 400_000
    text = "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s\n"
    text += "m38r14,anneal,2000,400000,1,10,12,0.5,-144.32105392,0.25,0.2\n"
    row = parse_row(text, "m38r14", 2000, 400_000)
    assert row["outside_s"] == "0.2"
    print("ok")


def main(argv: list[str]) -> int:
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--arm", action="append", default=[], help="name=source tree; run order on even seeds")
    parser.add_argument("--bin-root", help="directory holding <name>-bin/")
    parser.add_argument("--logs")
    parser.add_argument("--sc-lib")
    parser.add_argument("--system", action="append")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--summary")
    parser.add_argument("--candidate", default="G")
    parser.add_argument("--baseline", default="D")
    parser.add_argument("--aa", default="Dp")
    parser.add_argument("--reference", default="P")
    args = parser.parse_args(argv[1:])
    if args.summarize:
        return summarize(args)
    if not (args.arm and args.bin_root and args.logs):
        parser.error("--arm, --bin-root and --logs are needed to run")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
