#!/usr/bin/env python3
"""Rerun the GFN2 preset after the harness change and score it against the robustness rerun.

The jobs, start structures, budgets and scoring are
``robustness_rerun.py``'s. "Before" is a robustness rerun directory, whose
harness answered a failed xtb call with ``(1e6, zero gradient)``. "After"
is this rerun, whose harness answers ``(+inf, NaN gradient)``. CREST is
read from the campaign unchanged.

A run's accounting holds when ``ENGINE_CALLS == CHARGED == budget``. A run
with no failed call must reproduce its before record, because both
harnesses give identical answers until the first failure.

Usage::

    python experiments/harness_rerun.py --campaign CAMPAIGN jobs > jobs.txt
    xargs -P 7 -L 1 python experiments/harness_rerun.py --campaign CAMPAIGN --out OUT run < jobs.txt
    python experiments/harness_rerun.py --campaign CAMPAIGN --before BEFORE --out OUT table
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hit_stats import fisher, mcnemar, wilson  # noqa: E402
from organic_low_budget import KCAL_PER_EV, anneal  # noqa: E402
from robustness_rerun import (  # noqa: E402
    CAMPAIGN_ARMS,
    HIT_EV,
    SEEDS,
    SYSTEMS,
    campaign_record,
    counter,
    same_record,
    scored,
    wall,
    write_csv,
)

COUNTERS = (
    "REFUSED",
    "REFUSED_STARTS",
    "ENGINE_CALLS",
    "FAILED_CALLS",
    "NONFINITE_REQUESTS",
    "SOAP_FALLBACKS",
)


def run_seed(campaign: Path, out: Path, system: str, seed: int) -> None:
    root = out / f"{system}_s{seed}"
    result = root / "preset.json"
    if result.exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    record = campaign_record(campaign, system, seed)
    start = campaign / f"{system}_s{seed}" / "start.xyz"
    (root / "start.xyz").write_bytes(start.read_bytes())
    run = anneal(root / "start.xyz", root / "preset", record["budget"], seed, "preset")
    entry = {
        "system": system,
        "seed": seed,
        "arm": "preset",
        "budget": record["budget"],
        "best_eV": run["best"],
        "charged": run["charged"],
        "improvements": run["improvements"],
        "wall_s": wall(run["log"]),
    }
    for key in COUNTERS:
        entry[key.lower()] = counter(run["log"], key)
    result.write_text(json.dumps(entry, indent=1) + "\n")


def arms_in(root: Path, start: Path, seed: int) -> dict:
    arms = {}
    for path in sorted(root.glob("*.json")):
        run = json.loads(path.read_text())
        arm = run["arm"]
        best = root / arm / f"best_{arm}_s{seed}.xyz"
        arms[arm] = dict(run, best_eV=scored(start, best, run["best_eV"]))
    return arms


def load(campaign: Path, before: Path, out: Path) -> tuple[dict, dict]:
    oracles = {}
    for system in SYSTEMS:
        root = campaign / f"{system}_oracle"
        record = json.loads((root / "result.json").read_text())
        oracles[system] = scored(
            root / "start.xyz", root / "crest" / "crest_best.xyz", record["energy_eV"]
        )
    seeds = {}
    for system in SYSTEMS:
        for seed in SEEDS:
            root = campaign / f"{system}_s{seed}"
            if not root.exists():
                continue
            start = root / "start.xyz"
            record = campaign_record(campaign, system, seed)
            entry = {
                "budget": record["budget"],
                "crest_eV": scored(start, root / "crest" / "crest_best.xyz", record["crest_eV"]),
                "campaign": {},
                "before": arms_in(before / f"{system}_s{seed}", start, seed),
                "after": arms_in(out / f"{system}_s{seed}", start, seed),
            }
            for arm in CAMPAIGN_ARMS:
                if arm in record:
                    best = root / arm / f"best_{arm}_s{seed}.xyz"
                    entry["campaign"][arm] = dict(
                        record[arm], best_eV=scored(start, best, record[arm]["best_eV"])
                    )
            seeds[(system, seed)] = entry
    return oracles, seeds


def references(oracles: dict, seeds: dict, sources: tuple[str, ...]) -> dict:
    refs = {}
    for system in SYSTEMS:
        candidates = [(oracles[system], "oracle")]
        for (name, seed), entry in seeds.items():
            if name != system:
                continue
            candidates.append((entry["crest_eV"], f"crest s{seed}"))
            for source in sources:
                for arm, run in entry[source].items():
                    candidates.append((run["best_eV"], f"{source} {arm} s{seed}"))
        refs[system] = min(candidates)
    return refs


def minus(a: float, b: float) -> float:
    """``a - b`` in kcal/mol, with two equal values, ``inf`` included, at 0."""
    return 0.0 if a == b else (a - b) * KCAL_PER_EV


def row(system: str, seed: int, entry: dict, reference: float) -> dict:
    target = reference + HIT_EV
    crest = entry["crest_eV"]
    old = entry["before"]["preset"]
    new = entry["after"]["preset"]
    e_old, e_new = old["best_eV"], new["best_eV"]
    change = "better" if e_new < e_old else "worse" if e_new > e_old else "equal"
    return {
        "system": system,
        "seed": seed,
        "budget": entry["budget"],
        "reference_eV": reference,
        "crest_eV": crest,
        "crest_hit": int(crest <= target),
        "before_eV": e_old,
        "before_hit": int(e_old <= target),
        "after_eV": e_new,
        "after_hit": int(e_new <= target),
        "after_minus_before_kcal": minus(e_new, e_old),
        "before_minus_crest_kcal": minus(e_old, crest),
        "after_minus_crest_kcal": minus(e_new, crest),
        "before_ties_crest": int(e_old <= crest + HIT_EV),
        "after_ties_crest": int(e_new <= crest + HIT_EV),
        "change": change,
        "same_record": int(same_record(old, new)),
        "charged": new["charged"],
        "engine_calls": new["engine_calls"],
        "accounting_ok": int(new["engine_calls"] == new["charged"] == entry["budget"]),
        "failed_calls": new["failed_calls"],
        "refused": new["refused"],
        "refused_starts": new["refused_starts"],
        "outside_relaxations": new["failed_calls"] - new["refused"],
        "nonfinite_requests": new["nonfinite_requests"],
        "soap_fallbacks_before": old["soap_fallbacks"],
        "soap_fallbacks_after": new["soap_fallbacks"],
        "before_wall_s": old["wall_s"],
        "after_wall_s": new["wall_s"],
    }


def summarise(name: str, rows: list[dict]) -> dict:
    n = len(rows)

    def total(key: str) -> int:
        return sum(r[key] for r in rows)

    crest, old, new = total("crest_hit"), total("before_hit"), total("after_hit")
    only_old = sum(r["before_hit"] and not r["after_hit"] for r in rows)
    only_new = sum(r["after_hit"] and not r["before_hit"] for r in rows)
    ties_old, ties_new = total("before_ties_crest"), total("after_ties_crest")
    only_ties_old = sum(r["before_ties_crest"] and not r["after_ties_crest"] for r in rows)
    only_ties_new = sum(r["after_ties_crest"] and not r["before_ties_crest"] for r in rows)
    better = sum(r["change"] == "better" for r in rows)
    worse = sum(r["change"] == "worse" for r in rows)
    failed = [r["failed_calls"] for r in rows]
    return {
        "system": name,
        "n": n,
        "crest_hits": crest,
        "before_hits": old,
        "after_hits": new,
        "only_before": only_old,
        "only_after": only_new,
        "mcnemar_hits": mcnemar(only_old, only_new),
        "fisher_before_vs_crest": fisher(old, n - old, crest, n - crest),
        "fisher_after_vs_crest": fisher(new, n - new, crest, n - crest),
        "hit_gap_before": crest - old,
        "hit_gap_after": crest - new,
        "median_gap_before_kcal": float(np.median([r["before_minus_crest_kcal"] for r in rows])),
        "median_gap_after_kcal": float(np.median([r["after_minus_crest_kcal"] for r in rows])),
        "ties_crest_before": ties_old,
        "ties_crest_after": ties_new,
        "only_ties_before": only_ties_old,
        "only_ties_after": only_ties_new,
        "mcnemar_ties": mcnemar(only_ties_old, only_ties_new),
        "better": better,
        "worse": worse,
        "equal": n - better - worse,
        "sign_p": mcnemar(better, worse),
        "same_record": total("same_record"),
        "failed_calls": sum(failed),
        "runs_with_failures": sum(f > 0 for f in failed),
        "max_failed_in_a_run": max(failed),
        "line_search_refusals": total("refused") - total("refused_starts"),
        "refused_starts": total("refused_starts"),
        "outside_relaxations": total("outside_relaxations"),
        "nonfinite_requests": total("nonfinite_requests"),
        "soap_fallbacks_before": total("soap_fallbacks_before"),
        "soap_fallbacks_after": total("soap_fallbacks_after"),
        "accounting_ok": total("accounting_ok"),
    }


def cell(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({lo:.2f}-{hi:.2f})"


def kcal(v: float) -> str:
    return f"{v:+.3f}" if math.isfinite(v) else ("inf" if v > 0 else "-inf")


def table(campaign: Path, before: Path, out: Path) -> None:
    oracles, seeds = load(campaign, before, out)
    refs = references(oracles, seeds, ("campaign", "before", "after"))
    old_refs = references(oracles, seeds, ("campaign", "before"))
    paired, apart = [], []
    for system in SYSTEMS:
        reference = refs[system][0]
        for seed in SEEDS:
            entry = seeds.get((system, seed))
            if entry is None or "preset" not in entry["after"]:
                continue
            r = row(system, seed, entry, reference)
            (paired if "preset" in entry["campaign"] else apart).append(r)
    summary = [summarise(s, [r for r in paired if r["system"] == s]) for s in SYSTEMS]
    overall = summarise("all", paired)
    for e in summary:
        reference, holder = refs[e["system"]]
        e["reference_eV"] = reference
        e["reference_from"] = holder
        e["reference_moved"] = int(reference != old_refs[e["system"]][0])
    write_csv(out / "paired.csv", paired)
    if apart:
        write_csv(out / "unpaired.csv", apart)
    mismatched = [r for r in paired + apart if r["failed_calls"] == 0 and not r["same_record"]]
    (out / "stats.json").write_text(
        json.dumps(
            {
                "systems": summary,
                "overall": overall,
                "apart": apart,
                "self_check_failures": [(r["system"], r["seed"]) for r in mismatched],
            },
            indent=1,
        )
        + "\n"
    )
    every = summary + [dict(overall, reference_from="")]
    lines = [
        "| system | seeds | CREST hits | before hits | after hits | only before | only after "
        "| McNemar | Fisher after vs CREST | CREST hit gap before | after | reference from |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for e in every:
        k = e["n"]
        lines.append(
            f"| {e['system']} | {k} | {cell(e['crest_hits'], k)} | {cell(e['before_hits'], k)} "
            f"| {cell(e['after_hits'], k)} | {e['only_before']} | {e['only_after']} "
            f"| {e['mcnemar_hits']:.3g} | {e['fisher_after_vs_crest']:.3g} "
            f"| {e['hit_gap_before']} | {e['hit_gap_after']} | {e['reference_from']} |"
        )
    lines += [
        "",
        "| system | median minus CREST before (kcal/mol) | after | ties or beats CREST before "
        "| after | McNemar | better | worse | equal | sign test | same record |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for e in every:
        lines.append(
            f"| {e['system']} | {kcal(e['median_gap_before_kcal'])} "
            f"| {kcal(e['median_gap_after_kcal'])} | {e['ties_crest_before']} "
            f"| {e['ties_crest_after']} | {e['mcnemar_ties']:.3g} | {e['better']} "
            f"| {e['worse']} | {e['equal']} | {e['sign_p']:.3g} | {e['same_record']}/{e['n']} |"
        )
    lines += [
        "",
        "| system | failed calls | runs with one or more | most in a run | line-search refusals "
        "| refused starts | outside relaxations | non-finite requests | SOAP fallbacks before "
        "| after | accounting holds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for e in every:
        lines.append(
            f"| {e['system']} | {e['failed_calls']} | {e['runs_with_failures']}/{e['n']} "
            f"| {e['max_failed_in_a_run']} | {e['line_search_refusals']} | {e['refused_starts']} "
            f"| {e['outside_relaxations']} | {e['nonfinite_requests']} "
            f"| {e['soap_fallbacks_before']} | {e['soap_fallbacks_after']} "
            f"| {e['accounting_ok']}/{e['n']} |"
        )
    if apart:
        lines += [
            "",
            "| system | seed | budget | before eV | after eV | after minus before (kcal/mol) "
            "| after minus CREST (kcal/mol) | hit | failed calls | refused starts "
            "| accounting holds |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in apart:
            lines.append(
                f"| {r['system']} | {r['seed']} | {r['budget']} | {r['before_eV']:.6f} "
                f"| {r['after_eV']:.6f} | {kcal(r['after_minus_before_kcal'])} "
                f"| {kcal(r['after_minus_crest_kcal'])} | {r['after_hit']} | {r['failed_calls']} "
                f"| {r['refused_starts']} | {r['accounting_ok']} |"
            )
    changed = [r for r in paired if r["before_hit"] != r["after_hit"]]
    lines += ["", f"hit_changes {len(changed)}"]
    for r in changed:
        lines.append(
            f"  {r['system']} s{r['seed']} before {r['before_hit']} after {r['after_hit']} "
            f"delta {kcal(r['after_minus_before_kcal'])} kcal/mol"
        )
    moved = [e["system"] for e in summary if e["reference_moved"]]
    lines.append(f"references_moved {' '.join(moved) if moved else 'none'}")
    lines.append(
        "self_check_failures "
        + (" ".join(f"{r['system']}_s{r['seed']}" for r in mismatched) or "none")
    )
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--out", type=Path, default=Path("harness-rerun"))
    parser.add_argument("command", choices=["jobs", "run", "table"])
    parser.add_argument("job", nargs="*")
    args = parser.parse_args()
    if args.command == "jobs":
        for seed in SEEDS:
            for system in SYSTEMS:
                if (args.campaign / f"{system}_s{seed}").exists():
                    print(f"{system} {seed}")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        run_seed(args.campaign, args.out, args.job[0], int(args.job[1]))
        return
    if args.before is None:
        parser.error("table needs --before")
    table(args.campaign, args.before, args.out)


if __name__ == "__main__":
    main()
