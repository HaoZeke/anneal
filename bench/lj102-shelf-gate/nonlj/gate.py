#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Non-Lennard-Jones panel of the leave-clock gate, build A against build B.

Two release builds of the same source, differing only by the one change
under test, run on the same seeds in one session. Every process is pinned
to one core and runs one search at a time. Independent seeds run on
different cores at once. The two builds of one seed run back to back on the
same core, A first on even seeds and B first on odd seeds.

Rule, declared before any run. A system is a LOSS for B when either

  * B hits on fewer seeds than A and Fisher's exact test (two-sided) on
    the two hit counts has p < 0.05, or
  * over the seeds each build hit, B's median evaluations-to-hit is
    higher than A's and the two-sided Mann-Whitney test on those
    evaluations has p < 0.05.

The gate is PASS only when no system run is a LOSS. A hit-count drop with
p >= 0.05 is not a loss but is listed, as is an evaluations-to-hit increase
with p >= 0.05. A system that cannot run is listed as not run, with the
reason, and is outside the verdict.

Panel. Every row is a system whose search budget, hit reference and seed
set the repository already declares for the paper panels, run through the
example driver that panel runs, with seeds 2000 to 2011. Budgets and
references come from the PAPER_BUDGETS, SYSTEMS and budget tables of
scripts/gmin_panel, scripts/morse_gmin, scripts/morse_panel,
scripts/morse_h2h, scripts/metal_h2h, scripts/tip4p_h2h and the drivers.
M75 at rho 14 has no panel script. It takes the Cambridge Morse table
energy that PairPotential::ccd_morse_minimum holds, the 4e6 budget the
Lennard-Jones paper map gives every size from 75, and the h2h_seed driver
that the paper_0120 panel already uses for a size its main driver does not
list. BLJ55 and the other binary Lennard-Jones rows take the budgets and
references of the binary panel's PAPER_BUDGETS and SYSTEMS, through a
driver that is the panel's h2h_blj reduced to its run command.

A hit is a quenched energy at or below the reference plus 0.001, in the
potential's own unit. A miss is entered at the budget.

Subcommands:

  gate.py systems
  gate.py run --a-bin DIR --b-bin DIR --rows FILE --cores LIST [--system NAME ...]
  gate.py report --a-bin DIR --b-bin DIR --rows FILE --out-json FILE --out-md FILE
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

SEEDS = tuple(range(2000, 2012))
HIT_TOL = 0.001
ALPHA = 0.05

ROW_FIELDS = [
    "system",
    "arm",
    "seed",
    "order",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "core",
]


@dataclass(frozen=True)
class System:
    name: str
    driver: str
    budget: int
    reference: float
    source: str
    kind: str = ""
    n: int = 0
    rho: float = 0.0


def morse(name, budget, ref, source):
    return System(name, "h2h_morse", budget, ref, source)


def metal(name, budget, ref, source):
    return System(name, "h2h_metal", budget, ref, source)


def blj(name, budget, ref):
    return System(name, "h2h_blj", budget, ref, "bench blj panel PAPER_BUDGETS and SYSTEMS")


def water(n, ref):
    return System(
        f"tip4p{n}",
        "h2h_seed",
        2_000_000,
        ref,
        "scripts/tip4p_h2h run_h2h.py BUDGET, run_water_gmin.py CAMBRIDGE",
        kind="tip4p",
        n=n,
        rho=0.0,
    )


SYSTEMS = (
    morse("m30r6", 400_000, -118.432844, "scripts/morse_gmin SYSTEMS"),
    morse("m30r14", 400_000, -106.835790, "scripts/morse_gmin SYSTEMS"),
    morse("m38r6", 400_000, -157.477108, "scripts/morse_gmin SYSTEMS"),
    morse("m38r14", 400_000, -144.321054, "scripts/morse_gmin SYSTEMS"),
    morse("m55r6", 1_000_000, -250.286609, "scripts/gmin_panel PAPER_BUDGETS"),
    morse("m55r14", 1_000_000, -220.646208, "scripts/gmin_panel PAPER_BUDGETS"),
    morse("m70r3", 1_000_000, -577.739782, "scripts/morse_panel SYSTEMS"),
    morse("m70r6", 1_000_000, -325.887749, "scripts/morse_panel SYSTEMS"),
    morse("m70r10", 1_000_000, -296.412149, "scripts/morse_panel SYSTEMS"),
    morse("m70r14", 1_000_000, -292.462856, "scripts/morse_panel SYSTEMS"),
    System(
        "m75r14",
        "h2h_seed",
        4_000_000,
        -318.407330,
        "PairPotential::ccd_morse_minimum, 4e6 from the paper size map",
        kind="morse",
        n=75,
        rho=14.0,
    ),
    morse("m80r3", 4_000_000, -690.577890, "scripts/morse_panel SYSTEMS"),
    morse("m80r6", 4_000_000, -378.333471, "scripts/gmin_panel PAPER_BUDGETS"),
    morse("m80r10", 4_000_000, -344.911381, "scripts/morse_panel SYSTEMS"),
    morse("m80r14", 4_000_000, -340.811371, "scripts/gmin_panel PAPER_BUDGETS"),
    metal("au38", 400_000, -130.54001708, "scripts/metal_h2h run_darby.py PAPER_BUDGETS"),
    metal("cu6au32", 400_000, -130.79994836, "scripts/metal_h2h run_darby.py PAPER_BUDGETS"),
    metal("au55", 1_000_000, -191.03085280, "scripts/gmin_panel PAPER_BUDGETS"),
    metal("cu27au28", 1_000_000, -182.93136675, "scripts/metal_h2h run_darby.py PAPER_BUDGETS"),
    metal("cu19au19", 80_000, -122.99148391, "h2h_metal budget of the metal_h2h rule"),
    metal("zn38", 50_000, -50.499983, "h2h_metal budget of the metal_h2h rule"),
    metal("zn55", 80_000, -73.310032, "h2h_metal budget of the metal_h2h rule"),
    metal("zn75", 100_000, -100.203580, "h2h_metal budget of the metal_h2h rule"),
    metal("zn80", 100_000, -106.927940, "h2h_metal budget of the metal_h2h rule"),
    blj("blj105n38", 400_000, -177.2090846748),
    blj("blj115n38", 400_000, -187.3676867453),
    blj("blj130n38", 400_000, -191.1840192602),
    blj("blj105n55", 1_000_000, -283.1545313810),
    blj("blj115n55", 1_000_000, -289.2931853058),
    blj("blj130n55", 1_000_000, -306.7337643721),
    blj("blj130n98", 4_000_000, -590.7874132760),
    water(8, -305.5183221676),
    water(12, -492.9082357122),
    water(15, -628.3722525077),
    water(16, -681.1925945898),
    water(20, -872.9887544952),
    water(21, -916.7074302940),
)

NOT_RUN = (
    ("sc126-38 to sc96-80 (nine Sutton-Chen rows)", "needs the rgpot Sutton-Chen shared library through libloading; no such library is installed on this host, and h2h_sc exits at start"),
    ("sw10, sw13, sw20, tc10, tc20 (Stillinger-Weber and Tersoff)", "needs the rgpot covalent shared library; not installed on this host"),
    ("pt13, cu13, pt7cu6, pt12cu11, pt11cu12, pt6cu17, pt38, pt19cu19, cu38 (Pt-Cu Gupta)", "the repository holds no reference energy for these rows; the panel script takes the reference from a GMIN run, so no hit rule exists to score against"),
    ("cu19au19 at the 4e5 Darby budget", "h2h_metal changes its budget only through an environment variable, which this gate does not set; cu19au19 runs at the 8e4 budget of the metal_h2h rule"),
    ("tip4p25", "no Cambridge Cluster Database TIP4P minimum exists at 25 molecules, so no hit rule exists"),
    ("(H2O)6 hexamer on GFN2-xTB, PBE and a machine-learned potential; water16, water20 and ammonia clusters on GFN2-xTB", "needs xtb, NWChem, a machine-learned potential or CREST; none is installed on this host"),
    ("Si and organic covalent rows (covalent_h2h, openconf_h2h, tric_quench)", "need xtb, openconf or geomeTRIC; none is installed on this host"),
    ("Pt(111) heptamer (OptBench)", "needs the OptBench data set and its engine; neither is available on this host"),
)


def system_by_name() -> dict[str, System]:
    return {s.name: s for s in SYSTEMS}


def cpu_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key.strip().endswith(" name"):
            return value.strip()
    return "unknown"


def provenance(a_bin: Path, b_bin: Path) -> dict:
    def read(path: Path, name: str) -> str:
        return (path / "release" / name).read_text().strip()

    return {
        "a_head": read(a_bin, "SOURCE_HEAD"),
        "a_tree": read(a_bin, "SOURCE_TREE"),
        "b_head": read(b_bin, "SOURCE_HEAD"),
        "b_tree": read(b_bin, "SOURCE_TREE"),
        "compiler_a": read(a_bin, "RUSTC_VERSION"),
        "compiler_b": read(b_bin, "RUSTC_VERSION"),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
    }


def refuse_stray_environment() -> None:
    stray = sorted(k for k in os.environ if k.startswith(("H2H_", "PRIOR_", "RGPOT_")))
    if stray:
        raise SystemExit(f"refusing to run with {', '.join(stray)} set")


def example_path(bin_dir: Path, driver: str) -> Path:
    return bin_dir / "release" / "examples" / driver


def command(system: System, bin_dir: Path, seed: int, scratch: str) -> list[str]:
    exe = str(example_path(bin_dir, system.driver))
    if system.driver == "h2h_seed":
        stop = repr(system.reference + HIT_TOL)
        return [
            exe,
            system.kind,
            str(system.n),
            repr(system.rho),
            str(system.budget),
            str(seed),
            "recommended",
            stop,
            os.path.join(scratch, "run"),
        ]
    return [exe, "run", system.name, str(seed)]


def parse(system: System, text: str) -> dict:
    if system.driver == "h2h_seed":
        line = next(x for x in text.splitlines() if x.startswith("h2h_row "))
        fields = dict(part.split("=", 1) for part in line.split()[1:])
        return {
            "budget": int(fields["budget"]),
            "hit": fields["hit"] == "1",
            "evals_to_hit": int(fields["evals_to_hit"]),
            "charged": int(fields["charged"]),
            "best": float(fields["best"]),
        }
    data = [x for x in text.splitlines() if ",anneal," in x]
    if len(data) != 1:
        raise RuntimeError(f"expected one anneal row, saw {len(data)}")
    header = text.splitlines()[0].split(",")
    row = dict(zip(header, data[0].split(",")))
    return {
        "budget": int(row["budget"]),
        "hit": row["hit"].lower() in {"true", "1"},
        "evals_to_hit": int(row["evals_to_hit"]),
        "charged": int(row["charged"]),
        "best": float(row["best"]),
    }


def run_once(system: System, arm: str, bin_dir: Path, seed: int, core: int, order: str) -> dict:
    with tempfile.TemporaryDirectory() as scratch:
        argv = ["taskset", "-c", str(core), *command(system, bin_dir, seed, scratch)]
        started = time.monotonic()
        proc = subprocess.run(argv, capture_output=True, text=True, cwd=scratch)
        wall = time.monotonic() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"{system.name} {arm} seed {seed} exited {proc.returncode}: {proc.stderr[-400:]}"
        )
    row = parse(system, proc.stdout)
    if row["budget"] != system.budget:
        raise RuntimeError(
            f"{system.name} {arm}: driver budget {row['budget']} is not {system.budget}"
        )
    return {
        "system": system.name,
        "arm": arm,
        "seed": seed,
        "order": order,
        "budget": row["budget"],
        "hit": int(row["hit"]),
        "evals_to_hit": row["evals_to_hit"],
        "charged": row["charged"],
        "best": f"{row['best']:.8f}",
        "wall_s": f"{wall:.3f}",
        "core": core,
    }


def verify_drivers(a_bin: Path, b_bin: Path, systems: list[System]) -> None:
    for system in systems:
        for bin_dir in (a_bin, b_bin):
            exe = example_path(bin_dir, system.driver)
            if not exe.is_file():
                raise SystemExit(f"{exe} is missing")
        if system.driver in {"h2h_morse", "h2h_metal"}:
            for bin_dir in (a_bin, b_bin):
                out = subprocess.run(
                    [str(example_path(bin_dir, system.driver)), "budget", system.name],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.split()
                if int(out[0]) != system.budget or abs(float(out[1]) - system.reference) > 1e-6:
                    raise SystemExit(
                        f"{system.name}: driver says budget {out[0]} reference {out[1]}, "
                        f"the gate declares {system.budget} and {system.reference}"
                    )


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def cmd_systems(_args) -> int:
    print(f"seeds {SEEDS[0]} to {SEEDS[-1]} ({len(SEEDS)})")
    for s in SYSTEMS:
        print(f"{s.name}\t{s.driver}\tbudget {s.budget}\treference {s.reference}\t{s.source}")
    print("not run:")
    for name, reason in NOT_RUN:
        print(f"{name}\t{reason}")
    return 0


def cmd_run(args) -> int:
    refuse_stray_environment()
    a_bin, b_bin, rows_path = Path(args.a_bin), Path(args.b_bin), Path(args.rows)
    cores = [int(c) for c in args.cores.split(",")]
    systems = [s for s in SYSTEMS if not args.system or s.name in args.system]
    verify_drivers(a_bin, b_bin, systems)
    done = {(r["system"], int(r["seed"]), r["arm"]) for r in read_rows(rows_path)}
    units = [
        (s, seed)
        for s in systems
        for seed in SEEDS
        if (s.name, seed, "A") not in done or (s.name, seed, "B") not in done
    ]
    units.sort(key=lambda u: (-u[0].budget, u[0].name, u[1]))
    work: queue.Queue = queue.Queue()
    for unit in units:
        work.put(unit)
    new_file = not rows_path.is_file()
    lock = threading.Lock()
    failures: list[str] = []

    def append(rows: list[dict]) -> None:
        nonlocal new_file
        with lock:
            with rows_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
                if new_file:
                    writer.writeheader()
                    new_file = False
                writer.writerows(rows)

    def worker(core: int) -> None:
        while True:
            try:
                system, seed = work.get_nowait()
            except queue.Empty:
                return
            order = "AB" if seed % 2 == 0 else "BA"
            pair = []
            try:
                for arm in order:
                    bin_dir = a_bin if arm == "A" else b_bin
                    pair.append(run_once(system, arm, bin_dir, seed, core, order))
            except Exception as err:  # noqa: BLE001
                failures.append(str(err))
                continue
            append(pair)
            print(
                f"{system.name} seed {seed} core {core} {order} "
                + " ".join(f"{p['arm']}:{p['hit']}/{p['evals_to_hit']}" for p in pair),
                flush=True,
            )

    threads = [threading.Thread(target=worker, args=(core,)) for core in cores]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        for f in failures:
            print(f"failed: {f}", file=sys.stderr)
        return 1
    return 0


def type7_quantile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), 100.0 * p, method="linear"))


def summarize_hits(evals: list[int]) -> dict:
    if not evals:
        return {"n": 0, "median": None, "q1": None, "q3": None}
    return {
        "n": len(evals),
        "median": type7_quantile(evals, 0.5),
        "q1": type7_quantile(evals, 0.25),
        "q3": type7_quantile(evals, 0.75),
    }


def mann_whitney(x: list[float], y: list[float]) -> float | None:
    if not x or not y:
        return None
    if len(set(x) | set(y)) == 1:
        return 1.0
    return float(mannwhitneyu(x, y, alternative="two-sided", method="auto").pvalue)


def analyse(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        by[(r["system"], int(r["seed"]), r["arm"])] = r
    systems = []
    for s in SYSTEMS:
        seeds = [seed for seed in SEEDS if (s.name, seed, "A") in by and (s.name, seed, "B") in by]
        if not seeds:
            continue
        per_seed = []
        for seed in seeds:
            a, b = by[(s.name, seed, "A")], by[(s.name, seed, "B")]
            a_hit, b_hit = int(a["hit"]), int(b["hit"])
            a_ev, b_ev = int(a["evals_to_hit"]), int(b["evals_to_hit"])
            same = (a_hit, a_ev, a["best"]) == (b_hit, b_ev, b["best"])
            per_seed.append(
                {
                    "seed": seed,
                    "order": a["order"],
                    "a_hit": a_hit,
                    "b_hit": b_hit,
                    "a_evals": a_ev,
                    "b_evals": b_ev,
                    "a_best": float(a["best"]),
                    "b_best": float(b["best"]),
                    "identical": same,
                    "b_loses_hit": bool(a_hit and not b_hit),
                    "b_needs_more": bool(a_hit and b_hit and b_ev > a_ev),
                    "b_gains_hit": bool(b_hit and not a_hit),
                    "b_needs_fewer": bool(a_hit and b_hit and b_ev < a_ev),
                }
            )
        n = len(per_seed)
        a_hits = sum(p["a_hit"] for p in per_seed)
        b_hits = sum(p["b_hit"] for p in per_seed)
        fisher = float(
            fisher_exact([[a_hits, n - a_hits], [b_hits, n - b_hits]], alternative="two-sided")[1]
        )
        a_ev = [p["a_evals"] for p in per_seed if p["a_hit"]]
        b_ev = [p["b_evals"] for p in per_seed if p["b_hit"]]
        a_sum, b_sum = summarize_hits(a_ev), summarize_hits(b_ev)
        mw = mann_whitney(a_ev, b_ev)
        all_a = [p["a_evals"] for p in per_seed]
        all_b = [p["b_evals"] for p in per_seed]
        mw_all = mann_whitney(all_a, all_b)
        hit_drop = b_hits < a_hits
        hit_loss = hit_drop and fisher < ALPHA
        ev_up = (
            a_sum["median"] is not None
            and b_sum["median"] is not None
            and b_sum["median"] > a_sum["median"]
        )
        ev_loss = bool(ev_up and mw is not None and mw < ALPHA)
        systems.append(
            {
                "system": s.name,
                "driver": s.driver,
                "budget": s.budget,
                "reference": s.reference,
                "source": s.source,
                "seeds": n,
                "a_hits": a_hits,
                "b_hits": b_hits,
                "fisher_p": fisher,
                "a_evals_to_hit": a_sum,
                "b_evals_to_hit": b_sum,
                "mannwhitney_p": mw,
                "mannwhitney_p_all_seeds_censored": mw_all,
                "hit_drop": hit_drop,
                "hit_drop_significant": hit_loss,
                "evals_increase": bool(ev_up),
                "evals_increase_significant": ev_loss,
                "loss": bool(hit_loss or ev_loss),
                "identical_seeds": sum(p["identical"] for p in per_seed),
                "per_seed": per_seed,
            }
        )
    return {"systems": systems}


def fmt_p(p) -> str:
    if p is None:
        return "n/a"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def fmt_ev(summary: dict) -> str:
    if summary["n"] == 0:
        return "n/a"
    return f"{summary['median']:.0f} [{summary['q1']:.0f}, {summary['q3']:.0f}]"


def render_md(result: dict) -> str:
    sysrows = result["systems"]
    prov = result["provenance"]
    out = []
    out.append("# Non-LJ panel of the leave-clock gate")
    out.append("")
    verdict = result["verdict"]
    out.append(f"Verdict: **{verdict['text']}**")
    out.append("")
    out.append("## Rule, declared before any run")
    out.append("")
    out.append(
        "A system is a loss for B when B hits on fewer seeds than A with Fisher exact "
        "p < 0.05, or when, over the seeds each build hit, B's median evaluations-to-hit "
        "is higher than A's with a two-sided Mann-Whitney p < 0.05. The gate passes only "
        "when no system run is a loss. Non-significant hit-count drops and "
        "non-significant evaluations-to-hit increases are listed."
    )
    out.append("")
    out.append("## Session")
    out.append("")
    out.append(f"- A: `{prov['a_head']}`, tree `{prov['a_tree']}`")
    out.append(f"- B: `{prov['b_head']}`, tree `{prov['b_tree']}`")
    out.append(f"- compiler A: {prov['compiler_a']}")
    out.append(f"- compiler B: {prov['compiler_b']}")
    out.append(f"- CPU: {prov['cpu']}, {prov['logical_cpus']} logical CPUs")
    out.append(
        f"- seeds {SEEDS[0]} to {SEEDS[-1]} ({len(SEEDS)}) on every system; one process per "
        "pinned core; A and B of a seed back to back on one core, A first on even seeds and "
        "B first on odd seeds"
    )
    out.append("")
    out.append("## Systems")
    out.append("")
    out.append(
        "| system | budget | A hits | B hits | Fisher p | A evals to hit, median [IQR] "
        "| B evals to hit, median [IQR] | MW p | identical seeds | loss |"
    )
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in sysrows:
        out.append(
            f"| {s['system']} | {s['budget']:,} | {s['a_hits']}/{s['seeds']} "
            f"| {s['b_hits']}/{s['seeds']} | {fmt_p(s['fisher_p'])} "
            f"| {fmt_ev(s['a_evals_to_hit'])} (n={s['a_evals_to_hit']['n']}) "
            f"| {fmt_ev(s['b_evals_to_hit'])} (n={s['b_evals_to_hit']['n']}) "
            f"| {fmt_p(s['mannwhitney_p'])} | {s['identical_seeds']}/{s['seeds']} "
            f"| {'LOSS' if s['loss'] else 'no'} |"
        )
    out.append("")
    out.append("## Losses")
    out.append("")
    losses = [s for s in sysrows if s["loss"]]
    if not losses:
        out.append("None.")
    for s in losses:
        out.append(
            f"- {s['system']}: hits {s['a_hits']} to {s['b_hits']} (Fisher p {fmt_p(s['fisher_p'])}), "
            f"median evals {fmt_ev(s['a_evals_to_hit'])} to {fmt_ev(s['b_evals_to_hit'])} "
            f"(MW p {fmt_p(s['mannwhitney_p'])})"
        )
    out.append("")
    out.append("## Non-significant hit-count drops")
    out.append("")
    drops = [s for s in sysrows if s["hit_drop"] and not s["hit_drop_significant"]]
    if not drops:
        out.append("None.")
    for s in drops:
        out.append(
            f"- {s['system']}: {s['a_hits']}/{s['seeds']} to {s['b_hits']}/{s['seeds']}, "
            f"Fisher p {fmt_p(s['fisher_p'])}"
        )
    out.append("")
    out.append("## Non-significant evaluations-to-hit increases")
    out.append("")
    ups = [s for s in sysrows if s["evals_increase"] and not s["evals_increase_significant"]]
    if not ups:
        out.append("None.")
    for s in ups:
        out.append(
            f"- {s['system']}: median {fmt_ev(s['a_evals_to_hit'])} to {fmt_ev(s['b_evals_to_hit'])}, "
            f"MW p {fmt_p(s['mannwhitney_p'])}"
        )
    out.append("")
    out.append("## Seeds where B loses a hit A had or needs more evaluations")
    out.append("")
    any_seed = False
    for s in sysrows:
        for p in s["per_seed"]:
            if p["b_loses_hit"]:
                any_seed = True
                out.append(f"- {s['system']} seed {p['seed']}: A hit at {p['a_evals']}, B missed (best {p['b_best']:.8f})")
            elif p["b_needs_more"]:
                any_seed = True
                out.append(f"- {s['system']} seed {p['seed']}: A hit at {p['a_evals']}, B hit at {p['b_evals']}")
    if not any_seed:
        out.append("None.")
    out.append("")
    out.append("## Not run")
    out.append("")
    for name, reason in result["not_run"]:
        out.append(f"- {name}: {reason}.")
    out.append("")
    out.append("## Supplementary, not part of the rule")
    out.append("")
    out.append(
        "Mann-Whitney p on evaluations with every seed counted and a miss entered at the budget."
    )
    out.append("")
    out.append("| system | MW p, all seeds | seeds B gains a hit | seeds B needs fewer evaluations |")
    out.append("| --- | --- | --- | --- |")
    for s in sysrows:
        gains = sum(p["b_gains_hit"] for p in s["per_seed"])
        fewer = sum(p["b_needs_fewer"] for p in s["per_seed"])
        out.append(f"| {s['system']} | {fmt_p(s['mannwhitney_p_all_seeds_censored'])} | {gains} | {fewer} |")
    out.append("")
    out.append("## Per-seed results")
    out.append("")
    out.append(
        "Evaluations are the call index of the hit, or the budget for a miss. Best is the "
        "lowest quenched energy. Flag L is a hit lost by B, M is more evaluations for B, "
        "G is a hit gained by B, F is fewer evaluations for B, = is an identical run."
    )
    for s in sysrows:
        out.append("")
        out.append(f"### {s['system']}")
        out.append("")
        out.append("| seed | A hit | A evals | A best | B hit | B evals | B best | flag |")
        out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for p in s["per_seed"]:
            flag = (
                "L" if p["b_loses_hit"] else "M" if p["b_needs_more"] else
                "G" if p["b_gains_hit"] else "F" if p["b_needs_fewer"] else
                "=" if p["identical"] else ""
            )
            out.append(
                f"| {p['seed']} | {p['a_hit']} | {p['a_evals']} | {p['a_best']:.8f} "
                f"| {p['b_hit']} | {p['b_evals']} | {p['b_best']:.8f} | {flag} |"
            )
    out.append("")
    return "\n".join(out)


def cmd_report(args) -> int:
    a_bin, b_bin = Path(args.a_bin), Path(args.b_bin)
    rows = read_rows(Path(args.rows))
    result = analyse(rows)
    result["provenance"] = provenance(a_bin, b_bin)
    result["rule"] = {
        "alpha": ALPHA,
        "hit_tolerance": HIT_TOL,
        "seeds": list(SEEDS),
    }
    expected = {s.name for s in SYSTEMS}
    ran = {s["system"] for s in result["systems"]}
    incomplete = sorted(
        s["system"] for s in result["systems"] if s["seeds"] != len(SEEDS)
    )
    missing = sorted(expected - ran)
    if missing or incomplete:
        raise SystemExit(f"missing systems {missing}, incomplete {incomplete}")
    losses = [s["system"] for s in result["systems"] if s["loss"]]
    n = len(result["systems"])
    if losses:
        text = f"FAIL, B loses on {', '.join(losses)} ({n} systems run)"
    else:
        text = f"PASS, no system loses ({n} systems run)"
    result["verdict"] = {"pass": not losses, "losses": losses, "text": text}
    result["not_run"] = [list(x) for x in NOT_RUN]
    Path(args.out_json).write_text(json.dumps(result, indent=1) + "\n")
    Path(args.out_md).write_text(render_md(result))
    print(text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("systems").set_defaults(fn=cmd_systems)
    run = sub.add_parser("run")
    run.add_argument("--a-bin", required=True)
    run.add_argument("--b-bin", required=True)
    run.add_argument("--rows", required=True)
    run.add_argument("--cores", required=True)
    run.add_argument("--system", action="append")
    run.set_defaults(fn=cmd_run)
    report = sub.add_parser("report")
    report.add_argument("--a-bin", required=True)
    report.add_argument("--b-bin", required=True)
    report.add_argument("--rows", required=True)
    report.add_argument("--out-json", required=True)
    report.add_argument("--out-md", required=True)
    report.set_defaults(fn=cmd_report)
    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
