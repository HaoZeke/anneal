#!/usr/bin/env python3
"""Run the preset that polishes its records, and score it by the graph rule.

The preset now ends its search a reserve of R calls early and relaxes its
records with a model-Hessian quasi-Newton optimizer under ANCOPT's tight
stop (``Config::record_polish``). Scoring is ``graph_rerun.py``'s, unchanged:
a best with another covalent graph than its start scores ``+inf``, a hit is
within 0.1 kcal/mol of the system's reference, and the reference is the
lowest valid energy over the oracle, CREST, every stored source of
``graph_refs.json``, the invariant run of graph/ and this run. Every result
is scored as stored.

``smoke`` polishes a displaced, xtb-optimized methanol with 48 calls. ``run``
runs one campaign job. ``table`` applies the decision rule of
PROTOCOL-hessopt.md: no seed the rescored 0008 preset hits is lost, no
ethanol-dimer best is reacted, and the LJ digest is byte-identical to the
harness handover's. It also runs the self-checks and reports the polish,
the comparison with the invariant run and the water board.

Usage::

    python experiments/hessopt_rerun.py --campaign C --out OUT smoke
    python experiments/hessopt_rerun.py --campaign C --out OUT run SYSTEM SEED
    python experiments/hessopt_rerun.py --campaign C --before H --robust R --gap G --graph GR \\
        --out OUT --board B --graph-board IB --gap-board PB --lj-harness L --lj LJ table
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_rerun import cell, compared, kcal, minus, summarise  # noqa: E402
from graph_rerun import (  # noqa: E402
    STORED,
    as_json,
    board,
    board_hits,
    graph_valid,
    reacted,
    read_xyz,
    references,
    score,
    stored_runs,
)
from graph_rerun import load as graph_load  # noqa: E402
from graph_rerun import run_job as graph_run_job  # noqa: E402
from organic_low_budget import BENCH, HARTREE_EV, KCAL_PER_EV, XTB  # noqa: E402
from robustness_rerun import HIT_EV, SYSTEMS, write_csv  # noqa: E402

SOURCES = STORED + ("graph", "hessopt")
# R = 3N - 6 for each system, as registered.
RESERVE = {"w4": 30, "w6": 48, "w8": 66, "hexane": 54, "etoh2": 48, "trofinetide": 123}
RECORD_GRADIENT = 5.0e-2
POLISH = re.compile(
    r"^POLISH reserve (\d+) search_charged (\d+) calls (\d+) records (\d+) unspent (\d+)$", re.M
)
RECORD = re.compile(
    r"^POLISH_RECORD (\d+) from (\S+) to (\S+) calls (\d+) converged (true|false) "
    r"gnorm (\S+) recorded (true|false)$",
    re.M,
)
SMOKE = re.compile(
    r"^POLISH_SMOKE from (\S+) to (\S+) calls (\d+) converged (true|false) gnorm (\S+)$", re.M
)
SMOKE_SEED = 20261030
SMOKE_BUDGET = 48
SMOKE_SHIFT = 0.05
METHANOL = """6
methanol
C 0.0000 0.0000 0.0000
O 1.4200 0.0000 0.0000
H -0.3600 1.0300 0.0000
H -0.3600 -0.5150 0.8920
H -0.3600 -0.5150 -0.8920
H 1.7400 0.9000 0.0000
"""


def parse_polish(log: str) -> dict | None:
    found = POLISH.search(log)
    if found is None:
        return None
    reserve, search, calls, records, unspent = (int(v) for v in found.groups())
    rows = [
        {
            "from": float(m.group(2)),
            "to": float(m.group(3)),
            "calls": int(m.group(4)),
            "converged": m.group(5) == "true",
            "gnorm": float(m.group(6)),
            "recorded": m.group(7) == "true",
        }
        for m in RECORD.finditer(log)
    ]
    return {
        "reserve": reserve,
        "search_charged": search,
        "calls": calls,
        "records": rows,
        "unspent": unspent,
    }


def run_job(campaign: Path, out: Path, system: str, seed: int) -> None:
    graph_run_job(campaign, out, system, seed)
    path = out / f"{system}_s{seed}" / "preset.json"
    entry = json.loads(path.read_text())
    if "record_polish" in entry:
        return
    log = (out / f"{system}_s{seed}" / "preset" / "anneal.out").read_text()
    entry["record_polish"] = re.search(r"^record_polish (\S+)$", log, re.M).group(1) == "true"
    entry["polish"] = parse_polish(log)
    found = re.search(r"^BEST_GMAX (\S+)$", log, re.M)
    entry["best_gmax"] = float(found.group(1)) if found else None
    path.write_text(json.dumps(entry, indent=1) + "\n")


def write_xyz(path: Path, symbols: list[str], x: np.ndarray, comment: str) -> None:
    body = [str(len(symbols)), comment]
    body += [f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}" for s, p in zip(symbols, x)]
    path.write_text("\n".join(body) + "\n")


def bench_env() -> dict:
    env = os.environ.copy()
    env.update(ASE_ENGINE="xtb-cli", XTB_ACC="0.01", XTB_EXE=XTB, OMP_NUM_THREADS="1")
    return env


def smoke(out: Path) -> None:
    """The pre-run smoke test: units and signs, on a molecule outside the campaign."""
    work = out / "smoke"
    (work / "opt").mkdir(parents=True, exist_ok=True)
    (work / "opt" / "guess.xyz").write_text(METHANOL)
    opt = subprocess.run(
        [XTB, "guess.xyz", "--gfn", "2", "--acc", "0.01", "--opt", "vtight"],
        cwd=work / "opt",
        env=bench_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    (work / "opt" / "xtb.out").write_text(opt.stdout)
    optimized = work / "opt" / "xtbopt.xyz"
    symbols, x = read_xyz(optimized)
    comment = optimized.read_text().splitlines()[1]
    e_opt = float(re.search(r"energy:\s*(\S+)", comment).group(1)) * HARTREE_EV
    rng = np.random.default_rng(SMOKE_SEED)
    shifted = x + rng.uniform(-SMOKE_SHIFT, SMOKE_SHIFT, size=x.shape)
    write_xyz(work / "start.xyz", symbols, shifted, "methanol, xtbopt displaced")
    run = subprocess.run(
        [BENCH, "start.xyz", str(SMOKE_BUDGET), "0", "polish"],
        cwd=work,
        env=bench_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (work / "anneal.out").write_text(run.stdout)
    found = SMOKE.search(run.stdout)
    best = re.search(r"^BEST (\S+)$", run.stdout, re.M)
    gmax = re.search(r"^BEST_GMAX (\S+)$", run.stdout, re.M)
    charged = re.search(r"^CHARGED (\d+)$", run.stdout, re.M)
    calls = re.search(r"^ENGINE_CALLS (\d+)$", run.stdout, re.M)
    result = {
        "exit": run.returncode,
        "seed": SMOKE_SEED,
        "budget": SMOKE_BUDGET,
        "largest_shift_A": float(np.abs(shifted - x).max()),
        "xtb_vtight_eV": e_opt,
        "from_eV": float(found.group(1)) if found else None,
        "to_eV": float(found.group(2)) if found else None,
        "calls": int(found.group(3)) if found else None,
        "converged": found is not None and found.group(4) == "true",
        "gnorm_eV_per_A": float(found.group(5)) if found else None,
        "best_gmax_eV_per_A": float(gmax.group(1)) if gmax else None,
        "charged": int(charged.group(1)) if charged else None,
        "engine_calls": int(calls.group(1)) if calls else None,
    }
    if best is not None:
        result["best_minus_xtb_kcal"] = (float(best.group(1)) - e_opt) * KCAL_PER_EV
    result["passed"] = bool(
        result["converged"] and result["charged"] == result["engine_calls"] <= SMOKE_BUDGET
    )
    (work / "smoke.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


def load(args: argparse.Namespace) -> tuple[dict, dict]:
    oracles, seeds = graph_load(args, with_runs=False)
    for (system, seed), entry in seeds.items():
        for source, directory in (("graph", args.graph), ("hessopt", args.out)):
            for arm, run in stored_runs(directory / f"{system}_s{seed}", seed).items():
                entry["runs"][(source, arm)] = run
    return oracles, seeds


def search_part(run: dict, cut: int) -> list:
    return [tuple(i) for i in run["improvements"] if i[0] <= cut]


def polish_row(run: dict, budget: int) -> dict:
    polish = run.get("polish") or {}
    records = polish.get("records", [])
    cut = budget - polish.get("reserve", 0)
    searched = [e for c, e in search_part(run, cut)]
    lowest_search = min(searched) if searched else math.inf
    ends = [r["to"] for r in records]
    lowest_end = int(np.argmin(ends)) if ends and math.isfinite(min(ends)) else None
    return {
        "reserve": polish.get("reserve"),
        "search_charged": polish.get("search_charged"),
        "polish_calls": polish.get("calls"),
        "records": len(records),
        "records_converged": sum(r["converged"] for r in records),
        "records_recorded": sum(r["recorded"] for r in records),
        "unspent": polish.get("unspent"),
        "search_best_eV": lowest_search,
        "polish_lowered_kcal": minus(lowest_search, run["energy"]),
        "lowest_end_record": "none" if lowest_end is None else lowest_end,
        "best_gmax": run.get("best_gmax"),
    }


def table(args: argparse.Namespace) -> None:
    out = args.out
    frozen = json.loads((args.graph / "graph_refs.json").read_text())
    oracles, seeds = load(args)
    if as_json(references(oracles, seeds, "graph", STORED)) != frozen["references"]:
        sys.exit("the stored sources no longer give the frozen graph_refs.json references")
    refs = references(oracles, seeds, "graph", SOURCES)
    without = references(oracles, seeds, "graph", STORED + ("graph",))
    paired = sorted(k for k, e in seeds.items() if e["paired"])
    every = sorted(seeds)
    stats: dict = {
        "references": as_json(refs),
        "references_without_this_run": as_json(without),
        "frozen_references": frozen["references"],
    }
    lines: list[str] = []

    def rows_for(keys: list, against: dict, base_source: str) -> list[dict]:
        rows = []
        for system, seed in keys:
            entry = seeds[(system, seed)]
            run = entry["runs"].get(("hessopt", "preset"))
            base = entry["runs"].get((base_source, "preset"))
            if run is None or base is None:
                continue
            start = entry["start"]
            crest, reference = score("graph", start, entry["crest"]), against[system][0]
            r = {"system": system, "seed": seed, "budget": entry["budget"]}
            r["reference_eV"] = reference
            r["crest_eV"] = crest
            r["crest_hit"] = int(crest <= reference + HIT_EV)
            r.update(
                compared(
                    {"best_eV": score("graph", start, base)},
                    {"best_eV": score("graph", start, run)},
                    crest,
                    reference,
                )
            )
            r["arm_raw_eV"] = run["energy"]
            r["base_reacted"] = reacted("graph", start, base)
            r["arm_reacted"] = reacted("graph", start, run)
            r.update(polish_row(run, entry["budget"]))
            r["accounting_ok"] = int(
                run["engine_calls"] == run["charged"] <= entry["budget"]
            )
            rows.append(r)
        return rows

    rows = rows_for(paired, refs, "gap")
    write_csv(out / "hessopt_vs_preset.csv", rows)
    against_invariant = rows_for(paired, refs, "graph")
    write_csv(out / "hessopt_vs_invariant.csv", against_invariant)
    systems = [summarise(s, [r for r in rows if r["system"] == s]) for s in SYSTEMS]
    overall = summarise("all", rows)
    inv_systems = [
        summarise(s, [r for r in against_invariant if r["system"] == s]) for s in SYSTEMS
    ]
    inv_overall = summarise("all", against_invariant)

    lost = [f"{r['system']}_s{r['seed']}" for r in rows if r["base_hit"] and not r["arm_hit"]]
    d1 = len(rows) == len(paired) and not lost
    etoh2 = [(k, seeds[k]["runs"].get(("hessopt", "preset"))) for k in every if k[0] == "etoh2"]
    etoh2_state = [reacted("graph", seeds[k]["start"], run) for k, run in etoh2]
    etoh2_reacted = sum(s is True for s in etoh2_state)
    etoh2_no_record = sum(s is None for s in etoh2_state)
    d2 = len(etoh2) == 16 and all(run is not None for _, run in etoh2) and etoh2_reacted == 0
    harness = args.lj_harness.read_bytes() if args.lj_harness else None
    lj = {str(p): p.exists() and p.read_bytes() == harness for p in args.lj or []}
    d3 = harness is not None and len(lj) == 1 and all(lj.values())
    accepted = d1 and d2 and d3
    stats["decision"] = {
        "D1_no_hit_lost": d1,
        "D1_seeds_lost": lost,
        "D1_rows": len(rows),
        "D2_no_reacted_etoh2": d2,
        "D2_etoh2_reacted": etoh2_reacted,
        "D2_etoh2_no_record": etoh2_no_record,
        "D3_lj_identical": d3,
        "D3_files": lj,
        "accepted": accepted,
    }
    base_hits = sum(r["base_hit"] for r in rows)
    lines += [
        "## Decision",
        "",
        f"D1 no hit lost: {d1} ({len(lost)} of {base_hits} rescored 0008 hits lost: "
        f"{' '.join(lost) or 'none'}; {len(rows)} of {len(paired)} paired seeds)",
        f"D2 no reacted ethanol dimer: {d2} ({etoh2_reacted} of {len(etoh2_state)} etoh2 bests "
        f"reacted; {etoh2_no_record} runs with no record)",
        f"D3 LJ digest identical: {d3} (" + ", ".join(f"{k} {v}" for k, v in lj.items()) + ")",
        f"-> {'accepted' if accepted else 'rejected'}",
        "",
        "## Against CREST on the paired seeds, graph rule, final references",
        "",
        "| run | hits | CREST hits | Fisher vs CREST | ties or beats CREST "
        "| median minus CREST (kcal/mol) |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    medians = {
        "0008 preset (rescored)": float(
            np.median([minus(r["base_eV"], r["crest_eV"]) for r in rows])
        ),
        "invariant": float(np.median([minus(r["base_eV"], r["crest_eV"]) for r in against_invariant])),
        "hessopt": overall["median_arm_minus_crest_kcal"],
    }
    lines.append(
        f"| 0008 preset (rescored) | {cell(overall['base_hits'], overall['n'])} "
        f"| {cell(overall['crest_hits'], overall['n'])} | - | {overall['base_ties_crest']}"
        f"/{overall['n']} | {kcal(medians['0008 preset (rescored)'])} |"
    )
    lines.append(
        f"| invariant | {cell(inv_overall['base_hits'], inv_overall['n'])} "
        f"| {cell(inv_overall['crest_hits'], inv_overall['n'])} | - "
        f"| {inv_overall['base_ties_crest']}/{inv_overall['n']} | {kcal(medians['invariant'])} |"
    )
    lines.append(
        f"| hessopt | {cell(overall['arm_hits'], overall['n'])} "
        f"| {cell(overall['crest_hits'], overall['n'])} | {overall['fisher_arm_vs_crest']:.3g} "
        f"| {overall['arm_ties_crest']}/{overall['n']} | {kcal(medians['hessopt'])} |"
    )
    stats["medians_minus_crest_kcal"] = medians

    def comparison(title: str, base: str, summaries: list[dict]) -> None:
        nonlocal lines
        lines += [
            "",
            f"## hessopt against {title} (reported, not decided)",
            "",
            f"| system | seeds | CREST hits | {base} hits | hessopt hits | only {base} "
            f"| only hessopt | McNemar hits | ties {base} | ties hessopt | McNemar ties | better "
            "| worse | equal | sign test |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
            "| ---: | ---: | ---: |",
        ]
        for e in summaries:
            lines.append(
                f"| {e['system']} | {e['n']} | {e['crest_hits']} | {e['base_hits']} "
                f"| {e['arm_hits']} | {e['only_base']} | {e['only_arm']} "
                f"| {e['mcnemar_hits']:.3g} | {e['base_ties_crest']} | {e['arm_ties_crest']} "
                f"| {e['mcnemar_ties']:.3g} | {e['better']} | {e['worse']} | {e['equal']} "
                f"| {e['sign_p']:.3g} |"
            )

    comparison("the rescored 0008 preset", "0008", systems + [overall])
    comparison("the invariant run of graph/", "invariant", inv_systems + [inv_overall])
    inv_lost = [
        f"{r['system']}_s{r['seed']}"
        for r in against_invariant
        if r["base_hit"] and not r["arm_hit"]
    ]
    inv_gained = [
        f"{r['system']}_s{r['seed']}"
        for r in against_invariant
        if r["arm_hit"] and not r["base_hit"]
    ]
    gained = [f"{r['system']}_s{r['seed']}" for r in rows if r["arm_hit"] and not r["base_hit"]]
    lines += [
        "",
        f"Against the rescored 0008 preset: lost {' '.join(lost) or 'none'}, gained "
        f"{' '.join(gained) or 'none'}.",
        f"Against the invariant run: lost {' '.join(inv_lost) or 'none'}, gained "
        f"{' '.join(inv_gained) or 'none'}.",
    ]
    stats["against_preset"] = {"systems": systems, "overall": overall, "gained": gained}
    stats["against_invariant"] = {
        "systems": inv_systems,
        "overall": inv_overall,
        "lost": inv_lost,
        "gained": inv_gained,
    }
    frozen_refs = {s: (v["eV"], v["from"]) for s, v in frozen["references"].items()}
    against_frozen = summarise("all", rows_for(paired, frozen_refs, "gap"))
    without_rows = rows_for(paired, without, "gap")
    without_lost = [
        f"{r['system']}_s{r['seed']}" for r in without_rows if r["base_hit"] and not r["arm_hit"]
    ]
    stats["against_frozen_references"] = against_frozen
    stats["against_references_without_this_run"] = {
        "summary": summarise("all", without_rows),
        "lost": without_lost,
    }
    lines += [
        "",
        f"Against the frozen graph_refs.json references (not in any decision): 0008 hits "
        f"{against_frozen['base_hits']}, hessopt hits {against_frozen['arm_hits']}, CREST hits "
        f"{against_frozen['crest_hits']}, of {against_frozen['n']}.",
        f"Against the references without this run (not in any decision): 0008 seeds lost "
        f"{' '.join(without_lost) or 'none'}.",
        "",
        "## References (graph rule)",
        "",
        "| system | frozen (stored sources) | with the invariant | final |",
        "| --- | --- | --- | --- |",
    ]
    for system in SYSTEMS:
        a, b, c = frozen_refs[system], without[system], refs[system]
        lines.append(
            f"| {system} | {a[0]:.8f} ({a[1]}) | {b[0]:.8f} ({b[1]}) | {c[0]:.8f} ({c[1]}) |"
        )

    # The polish.
    every_rows = []
    for system, seed in every:
        entry = seeds[(system, seed)]
        run = entry["runs"].get(("hessopt", "preset"))
        if run is None:
            continue
        every_rows.append({"system": system, "seed": seed, **polish_row(run, entry["budget"])})
    write_csv(out / "polish.csv", every_rows)
    records = [
        (system, seed, k, rec)
        for system, seed in every
        if (run := seeds[(system, seed)]["runs"].get(("hessopt", "preset")))
        for k, rec in enumerate((run.get("polish") or {}).get("records", []))
    ]
    write_csv(
        out / "polish_records.csv",
        [
            {
                "system": s,
                "seed": seed,
                "record": k,
                "from_eV": rec["from"],
                "to_eV": rec["to"],
                "lowered_kcal": minus(rec["from"], rec["to"]),
                "calls": rec["calls"],
                "converged": int(rec["converged"]),
                "gnorm_eV_per_A": rec["gnorm"],
                "recorded": int(rec["recorded"]),
            }
            for s, seed, k, rec in records
        ],
    )
    lines += [
        "",
        "## The polish (every campaign run)",
        "",
        "| system | runs | reserve | polish calls | records polished | converged | recorded "
        "| runs lowered | median lowered (kcal/mol) | largest lowered | lowest end not record 0 "
        "| runs with calls unspent | calls unspent |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: |",
    ]
    polish_stats = []
    for system in SYSTEMS + ("all",):
        mine = [r for r in every_rows if system == "all" or r["system"] == system]
        lowered = [
            r["polish_lowered_kcal"] for r in mine if math.isfinite(r["polish_lowered_kcal"])
        ]
        row = {
            "system": system,
            "runs": len(mine),
            "reserve": sorted({r["reserve"] for r in mine}, key=str),
            "polish_calls": sum(r["polish_calls"] or 0 for r in mine),
            "records": sum(r["records"] for r in mine),
            "converged": sum(r["records_converged"] for r in mine),
            "recorded": sum(r["records_recorded"] for r in mine),
            "runs_lowered": sum(v > 0 for v in lowered),
            "median_lowered_kcal": float(np.median(lowered)) if lowered else math.nan,
            "largest_lowered_kcal": max(lowered) if lowered else math.nan,
            "lowest_end_not_first": sum(r["lowest_end_record"] not in (0, "none") for r in mine),
            "runs_unspent": sum((r["unspent"] or 0) > 0 for r in mine),
            "unspent": sum(r["unspent"] or 0 for r in mine),
        }
        polish_stats.append(row)
        lines.append(
            f"| {system} | {row['runs']} | {'/'.join(map(str, row['reserve']))} "
            f"| {row['polish_calls']} | {row['records']} | {row['converged']} | {row['recorded']} "
            f"| {row['runs_lowered']} | {row['median_lowered_kcal']:.4f} "
            f"| {row['largest_lowered_kcal']:.4f} | {row['lowest_end_not_first']} "
            f"| {row['runs_unspent']} | {row['unspent']} |"
        )
    stats["polish"] = polish_stats
    per_record = [minus(rec["from"], rec["to"]) for *_, rec in records if math.isfinite(rec["to"])]
    lines += [
        "",
        f"Per polished record: {len(records)} polished, {sum(r['converged'] for *_, r in records)} "
        f"converged, {sum(not math.isfinite(r['to']) for *_, r in records)} with no recordable "
        f"structure; energy lowered median "
        f"{float(np.median(per_record)) if per_record else math.nan:.4f} kcal/mol, mean "
        f"{float(np.mean(per_record)) if per_record else math.nan:.4f}.",
    ]
    stats["per_record"] = {
        "polished": len(records),
        "converged": sum(r["converged"] for *_, r in records),
        "no_recordable": sum(not math.isfinite(r["to"]) for *_, r in records),
        "median_lowered_kcal": float(np.median(per_record)) if per_record else None,
    }
    apart = [k for k in every if not seeds[k]["paired"]]
    for system, seed in apart:
        entry = seeds[(system, seed)]
        run = entry["runs"].get(("hessopt", "preset"))
        crest = score("graph", entry["start"], entry["crest"])
        if run is not None:
            lines.append(
                f"{system} s{seed} (outside the paired set): hessopt minus CREST "
                f"{kcal(minus(score('graph', entry['start'], run), crest))} kcal/mol"
            )

    # Self-checks.
    checks: dict = {}
    s1, s1_reserve, s2, s2_extra, s4_above, s4_gradient = [], [], [], [], [], []
    for system, seed in every:
        entry = seeds[(system, seed)]
        run = entry["runs"].get(("hessopt", "preset"))
        invariant = entry["runs"].get(("graph", "preset"))
        name = f"{system}_s{seed}"
        if run is None:
            s1.append(f"{name} (no run)")
            continue
        polish = run.get("polish")
        budget = entry["budget"]
        if not run["engine_calls"] == run["charged"] <= budget:
            s1.append(name)
        if (
            polish is None
            or polish["reserve"] != RESERVE[system]
            or polish["search_charged"] != budget - polish["reserve"]
        ):
            s1_reserve.append(name)
            continue
        cut = budget - polish["reserve"]
        if invariant is None or search_part(run, cut) != search_part(invariant, cut):
            s2.append(name)
        if len(run["improvements"]) - len(search_part(run, cut)) != sum(
            r["recorded"] for r in polish["records"]
        ):
            s2_extra.append(name)
        searched = [e for c, e in search_part(run, cut)]
        if searched and not run["energy"] <= min(searched):
            s4_above.append(name)
        from_polish = any(r["recorded"] for r in polish["records"])
        if from_polish and not (run.get("best_gmax") or math.inf) < RECORD_GRADIENT:
            s4_gradient.append(f"{name} ({run.get('best_gmax')})")
    checks["S1_accounting_failures"] = s1
    checks["S1_search_part_not_B_minus_R"] = s1_reserve
    checks["S2_search_part_differs_from_invariant"] = s2
    checks["S2_improvements_beyond_B_minus_R_not_polish_records"] = s2_extra
    checks["S3_hessopt_bests_with_another_graph"] = [
        f"{s}_s{k}"
        for (s, k), e in seeds.items()
        if reacted("graph", e["start"], e["runs"].get(("hessopt", "preset"))) is True
    ]
    checks["S4_best_above_search_part"] = s4_above
    checks["S4_polished_best_gradient_not_below_record_gradient"] = s4_gradient

    # Water board.
    if args.board is not None and args.board.exists():
        new = board(args.board)
        invariant_board = board(args.graph_board) if args.graph_board else {}
        preset_board = board(args.gap_board) if args.gap_board else {}
        geometry = args.board.parent / "geometry"
        board_rows = []
        for key in sorted(new, key=int):
            row = new[key]
            start, best = geometry / f"start_s{key}.xyz", geometry / f"best_s{key}.xyz"
            kept = (graph_valid(start, best) if best.exists() else None) if start.exists() else None
            board_rows.append(
                {
                    "seed": int(key),
                    "isomer": row.get("isomer"),
                    "hit": int(row.get("isomer_reason") == "inside gate"),
                    "invariant_hit": int(
                        invariant_board.get(key, {}).get("isomer_reason") == "inside gate"
                    ),
                    "preset_hit": int(
                        preset_board.get(key, {}).get("isomer_reason") == "inside gate"
                    ),
                    "best_energy": row.get("best_energy"),
                    "invariant_best_energy": invariant_board.get(key, {}).get("best_energy"),
                    "start_graph_rejected": row.get("start_graph_rejected"),
                    "best_keeps_graph": "none" if kept is None else int(kept),
                    "force_calls": row.get("force_calls"),
                    "polish_reserve": row.get("polish_reserve"),
                    "polish_search_force_calls": row.get("polish_search_force_calls"),
                    "polish_force_calls": row.get("polish_force_calls"),
                    "polish_records": row.get("polish_records"),
                    "polish_converged": row.get("polish_converged"),
                    "polish_recorded": row.get("polish_recorded"),
                }
            )
        write_csv(out / "board_hessopt.csv", board_rows)
        hits = board_hits(new)
        b = {
            "hits": len(hits),
            "seeds": len(new),
            "invariant_hits": len(board_hits(invariant_board)),
            "preset_hits": len(board_hits(preset_board)),
            "lost_vs_invariant": sorted(board_hits(invariant_board) - hits, key=int),
            "gained_vs_invariant": sorted(hits - board_hits(invariant_board), key=int),
            "lost_vs_preset": sorted(board_hits(preset_board) - hits, key=int),
            "gained_vs_preset": sorted(hits - board_hits(preset_board), key=int),
            "start_rejected": sum(r["start_graph_rejected"] == "true" for r in board_rows),
            "bests_reacted": sum(r["best_keeps_graph"] == 0 for r in board_rows),
            "no_best": sum(r["best_keeps_graph"] == "none" for r in board_rows),
            "polish_force_calls": sum(int(r["polish_force_calls"] or 0) for r in board_rows),
            "seeds_recorded_by_polish": sum(
                int(r["polish_recorded"] or 0) > 0 for r in board_rows
            ),
            "over_budget": [r["seed"] for r in board_rows if int(r["force_calls"] or 0) > 4000],
        }
        stats["board"] = b
        lines += [
            "",
            "## Water board (apart, reported, not decided)",
            "",
            f"hits {b['hits']}/{b['seeds']}; invariant board {b['invariant_hits']}, lost "
            f"{' '.join(map(str, b['lost_vs_invariant'])) or 'none'}, gained "
            f"{' '.join(map(str, b['gained_vs_invariant'])) or 'none'}; 0008 preset board "
            f"{b['preset_hits']}, lost {' '.join(map(str, b['lost_vs_preset'])) or 'none'}, "
            f"gained {' '.join(map(str, b['gained_vs_preset'])) or 'none'}",
            f"start rejected {b['start_rejected']}, bests with another graph {b['bests_reacted']}, "
            f"no best {b['no_best']}; polish force calls {b['polish_force_calls']}, seeds whose "
            f"best came from the polish {b['seeds_recorded_by_polish']}; over budget "
            f"{' '.join(map(str, b['over_budget'])) or 'none'}",
        ]
    stats["self_checks"] = checks
    lines += ["", "## Self-checks", ""]
    for key, value in checks.items():
        shown = (" ".join(map(str, value)) or "none") if isinstance(value, list) else value
        lines.append(f"{key} {shown}")
    (out / "stats.json").write_text(json.dumps(stats, indent=1, default=str) + "\n")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--before", type=Path, help="0007 harness rerun")
    parser.add_argument("--robust", type=Path, help="robustness rerun")
    parser.add_argument("--gap", type=Path, help="0008 gap runs")
    parser.add_argument("--graph", type=Path, help="invariant runs, with graph_refs.json")
    parser.add_argument("--out", type=Path, default=Path("hessopt-rerun"))
    parser.add_argument("--board", type=Path, help="this arm's water board.txt")
    parser.add_argument("--graph-board", type=Path, help="invariant water board.txt")
    parser.add_argument("--gap-board", type=Path, help="0008 preset water board.txt")
    parser.add_argument("--lj-harness", type=Path, help="lj_digest output of the harness")
    parser.add_argument("--lj", type=Path, action="append", help="lj_digest output to compare")
    parser.add_argument("command", choices=["smoke", "run", "table"])
    parser.add_argument("job", nargs="*")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "smoke":
        smoke(args.out)
        return
    if args.command == "run":
        system, seed = args.job
        run_job(args.campaign, args.out, system, int(seed))
        return
    if None in (args.before, args.robust, args.gap, args.graph):
        parser.error("table needs --before, --robust, --gap and --graph")
    table(args)


if __name__ == "__main__":
    main()
