#!/usr/bin/env python3
"""Score completed r-next8 rows. The rules are PROTOCOL.md."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

BOOT_SEED = 20261123
BOOT_DRAWS = 10_000
EQUAL_ABS = 1.0e-3
NOISE_ABS = 1.0e-6
ARMS = ("A", "B", "B2", "A2")


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(text: str) -> float | None:
    if text in {"", "nan", "inf", "-inf"}:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not np.isfinite(value):
        return None
    return value


def complete(row: dict[str, str]) -> bool:
    return (
        row["kind"] != "slab"
        and row["phase"] == ""
        and row["censored"] == "false"
        and row["calls"] != ""
        and row["hit"] in {"true", "false"}
    )


SEARCH_CELLS = (
    "sw10",
    "sw13",
    "sw20",
    "tc10",
    "tc20",
    "gfn2-si10",
    "gfn2-si13",
    "gfn2-c10",
)


def index_search(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["kind"] == "slab" or row["phase"] != "":
            continue
        grouped[row["cell"]][f"{row['arm']}:{row['seed']}"] = row
    return grouped


def seeds_of(cell_rows: dict[str, dict[str, str]], arm: str) -> dict[str, dict[str, str]]:
    prefix = arm + ":"
    return {key.split(":", 1)[1]: row for key, row in cell_rows.items() if key.startswith(prefix)}


def agree(left: float, right: float) -> bool:
    delta = abs(left - right)
    return delta <= EQUAL_ABS or delta <= NOISE_ABS


def bootstrap(ratios: np.ndarray) -> dict[str, float]:
    logs = np.log(ratios)
    rng = np.random.default_rng(BOOT_SEED)
    index = rng.integers(0, ratios.size, size=(BOOT_DRAWS, ratios.size))
    samples = np.exp(logs[index].mean(axis=1))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "n": int(ratios.size),
        "gm": float(np.exp(logs.mean())),
        "low": float(low),
        "high": float(high),
    }


def paired_ratio(
    left: dict[str, dict[str, str]],
    right: dict[str, dict[str, str]],
    field: str,
) -> dict[str, float] | None:
    ratios = []
    for seed, row in sorted(left.items()):
        other = right.get(seed)
        if other is None:
            continue
        a = as_float(row[field])
        b = as_float(other[field])
        if a is None or b is None or a <= 0.0 or b <= 0.0:
            continue
        ratios.append(b / a)
    if not ratios:
        return None
    return bootstrap(np.asarray(ratios, dtype=float))


def calls_to_score(row: dict[str, str]) -> float:
    if row["hit"] == "true":
        return float(row["calls_to_hit"])
    return float(row["budget"]) + 1.0


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}g}"


def search_cell(name: str, rows: dict[str, dict[str, str]], censored: int) -> dict[str, object]:
    finished = {key: row for key, row in rows.items() if complete(row)}
    arms = {arm: seeds_of(finished, arm) for arm in ARMS}
    recorded = {arm: seeds_of(rows, arm) for arm in ARMS}
    common = set(arms["A"]) & set(arms["B"])
    a_hits = sum(arms["A"][seed]["hit"] == "true" for seed in common)
    b_hits = sum(arms["B"][seed]["hit"] == "true" for seed in common)
    a_miss = len(common) - a_hits
    b_miss = len(common) - b_hits
    fisher_p = None
    if common:
        _, fisher_p = fisher_exact([[a_hits, a_miss], [b_hits, b_miss]], alternative="two-sided")
    mw_u = None
    mw_p = None
    if len(common) >= 2:
        a_scores = [calls_to_score(arms["A"][seed]) for seed in sorted(common)]
        b_scores = [calls_to_score(arms["B"][seed]) for seed in sorted(common)]
        mw = mannwhitneyu(b_scores, a_scores, alternative="two-sided", method="auto")
        mw_u = float(mw.statistic)
        mw_p = float(mw.pvalue)
    losses = []
    expected = []
    nonphysical = []
    agree_n = 0
    compared = 0
    for seed in sorted(set(recorded["A"]) & set(recorded["B"])):
        a = recorded["A"][seed]
        b = recorded["B"][seed]
        a_done = a["censored"] == "false"
        b_done = b["censored"] == "false"
        if a_done and b_done and a["physical_hit"] == "true" and b["physical_hit"] != "true":
            losses.append(seed)
        if a_done and a["hit"] == "true" and a["physical_hit"] != "true" and b["hit"] != "true":
            expected.append(seed)
        if b_done and b["hit"] == "true" and b["physical_hit"] != "true":
            nonphysical.append(seed)
        if not (a_done and b_done):
            continue
        a_best = as_float(a["best"])
        b_best = as_float(b["best"])
        if a_best is not None and b_best is not None:
            compared += 1
            if agree(a_best, b_best):
                agree_n += 1
    outside = paired_ratio(arms["A"], arms["B"], "outside_s")
    outside_noise = paired_ratio(arms["A"], arms["A2"], "outside_s")
    memory = paired_ratio(arms["A"], arms["B"], "rss_anon_peak_kb")
    memory_noise = paired_ratio(arms["A"], arms["A2"], "rss_anon_peak_kb")
    noise_high = None if outside_noise is None else outside_noise["high"]
    memory_noise_high = None if memory_noise is None else memory_noise["high"]
    time_n = 0 if outside is None else int(outside["n"])
    memory_n = 0 if memory is None else int(memory["n"])
    time_bar = None if noise_high is None else max(1.10, noise_high)
    memory_bar = None if memory_noise_high is None else max(1.10, memory_noise_high)
    exceeds_110 = bool(outside and outside["low"] > 1.10)
    exceeds_noise = bool(outside and time_bar is not None and outside["low"] > time_bar)
    real_time = bool(
        time_n >= 6 and outside and noise_high is not None and outside["low"] > 1.0 and outside["low"] > noise_high
    )
    memory_fail = bool(memory_n >= 6 and memory and memory_bar is not None and memory["low"] > memory_bar)
    b2_mismatch = []
    b2_wall_calls = []
    for seed, row in sorted(recorded["B"].items()):
        other = recorded["B2"].get(seed)
        if other is None:
            continue
        same_hit = row["hit"] == other["hit"]
        left = as_float(row["best"])
        right = as_float(other["best"])
        same_energy = left is not None and right is not None and agree(left, right)
        if not same_hit or (left is not None and right is not None and not same_energy):
            b2_mismatch.append(seed)
        elif row["calls"] != other["calls"] and same_energy:
            b2_wall_calls.append(seed)
    if losses or nonphysical:
        verdict = "FIX"
    elif real_time or memory_fail:
        verdict = "FAIL"
    elif censored or len(common) < 12:
        verdict = "PARTIAL"
    else:
        verdict = "PASS"
    return {
        "cell": name,
        "n_complete_pairs": len(common),
        "n_censored_rows": censored,
        "hits_a": a_hits,
        "hits_b": b_hits,
        "hits_a2": sum(
            arms["A2"][seed]["hit"] == "true" for seed in common if seed in arms["A2"]
        ),
        "hits_b2": sum(
            arms["B2"][seed]["hit"] == "true" for seed in common if seed in arms["B2"]
        ),
        "physical_a": sum(arms["A"][seed]["physical_hit"] == "true" for seed in common),
        "physical_b": sum(arms["B"][seed]["physical_hit"] == "true" for seed in common),
        "fisher_p": fisher_p,
        "mannwhitney_u": mw_u,
        "mannwhitney_p": mw_p,
        "hit_losses": losses,
        "expected_rejections": expected,
        "b_nonphysical": nonphysical,
        "energy_agree": agree_n,
        "energy_compared": compared,
        "outside_b_a": outside,
        "outside_a2_a": outside_noise,
        "rss_b_a": memory,
        "rss_a2_a": memory_noise,
        "time_exceeds_1_10": exceeds_110,
        "time_exceeds_noise_bar": exceeds_noise,
        "time_real_regression": real_time,
        "memory_failure": memory_fail,
        "b2_call_or_hit_mismatch": b2_mismatch,
        "b2_same_energy_call_difference": b2_wall_calls,
        "verdict": verdict,
    }


def ratio_text(block: dict[str, float] | None) -> str:
    if not block:
        return ""
    return f"{block['gm']:.4f} [{block['low']:.4f}, {block['high']:.4f}] n={block['n']}"


def slab_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    picked = {
        (row["phase"], row["arm"]): row
        for row in rows
        if row["cell"] == "slab" and row["engine_calls"] != ""
    }
    combined = {arm: picked.get(("combined", arm)) for arm in ARMS}
    withgrad = {arm: picked.get(("withgrad", arm)) for arm in ARMS}
    grad_only = {}
    for arm in ARMS:
        left = combined[arm]
        right = withgrad[arm]
        if left and right and left["engine_calls"] and right["engine_calls"]:
            grad_only[arm] = int(right["engine_calls"]) - int(left["engine_calls"])
        else:
            grad_only[arm] = None
    ledger_ok = all(
        (combined[arm] is not None and combined[arm]["ledger_calls"] == "1")
        and (withgrad[arm] is not None and withgrad[arm]["ledger_calls"] == "2")
        and combined[arm]["censored"] == "false"
        and withgrad[arm]["censored"] == "false"
        for arm in ARMS
    )
    counts = [grad_only[arm] for arm in ARMS]
    combined_counts = [
        None if combined[arm] is None else int(combined[arm]["engine_calls"]) for arm in ARMS
    ]
    if not ledger_ok or any(value is None for value in counts):
        verdict = "PARTIAL"
    elif counts[0] == counts[3] and counts[1] == counts[2] and counts[1] > counts[0]:
        verdict = "FIX"
    elif len(set(counts)) == 1 and len(set(combined_counts)) == 1:
        verdict = "PASS"
    else:
        verdict = "PARTIAL"
    return {
        "cell": "slab",
        "combined_engine": {arm: None if combined[arm] is None else int(combined[arm]["engine_calls"]) for arm in ARMS},
        "withgrad_engine": {arm: None if withgrad[arm] is None else int(withgrad[arm]["engine_calls"]) for arm in ARMS},
        "grad_only_engine": {arm: grad_only[arm] for arm in ARMS},
        "ledger_ok": ledger_ok,
        "verdict": verdict,
    }


GFN2_FIELDS = (
    "cell",
    "arm",
    "seed",
    "hit",
    "physical_hit",
    "best",
    "calls",
    "calls_to_hit",
    "censored",
    "detached",
    "connected",
    "nn_min_A",
    "nn_max_A",
    "scc_converged",
    "max_force_eV_A",
    "force_ok",
    "sp_energy_eV",
    "scc_rejected",
    "geometry_rejected",
    "wall_s",
)


def gfn2_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    wanted = [
        row
        for row in rows
        if row["cell"].startswith("gfn2-") and row["phase"] == ""
    ]
    return [{key: row.get(key, "") for key in GFN2_FIELDS} for row in wanted]


def markdown(payload: dict[str, object]) -> str:
    lines = [
        "# PR 211 r-next8 gate",
        "",
        "Rules and arm definitions are in `PROTOCOL.md`. GMIN was not run.",
        "",
        "## Per-cell results",
        "",
        "| Cell | n | Hits A/B/A2/B2 | Physical A/B | Fisher p | Mann-Whitney p | Outside B/A | Outside A2/A | RSS B/A | RSS A2/A | Time >1.10 | Time > noise bar | Real time regression | Memory failure | Verdict |",
        "| --- | ---: | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in payload["cells"]:
        lines.append(
            "| {cell} | {n} | {ha}/{hb}/{ha2}/{hb2} | {pa}/{pb} | {fisher} | {mw} | {out} | {noise} | {rss} | {rssn} | {t110} | {tn} | {real} | {mem} | {verdict} |".format(
                cell=cell["cell"],
                n=cell["n_complete_pairs"],
                ha=cell["hits_a"],
                hb=cell["hits_b"],
                ha2=cell["hits_a2"],
                hb2=cell["hits_b2"],
                pa=cell["physical_a"],
                pb=cell["physical_b"],
                fisher=fmt(cell["fisher_p"]),
                mw=fmt(cell["mannwhitney_p"]),
                out=ratio_text(cell["outside_b_a"]),
                noise=ratio_text(cell["outside_a2_a"]),
                rss=ratio_text(cell["rss_b_a"]),
                rssn=ratio_text(cell["rss_a2_a"]),
                t110=cell["time_exceeds_1_10"],
                tn=cell["time_exceeds_noise_bar"],
                real=cell["time_real_regression"],
                mem=cell["memory_failure"],
                verdict=cell["verdict"],
            )
        )
    lines.extend(["", "Hit losses, expected non-physical rejections, and B hits that are not physical:", ""])
    for cell in payload["cells"]:
        lines.append(
            f"- {cell['cell']}: censored rows {cell['n_censored_rows']}; "
            f"losses {cell['hit_losses'] or 'none'}; "
            f"expected rejections {cell['expected_rejections'] or 'none'}; "
            f"B non-physical {cell['b_nonphysical'] or 'none'}; "
            f"energy agree {cell['energy_agree']}/{cell['energy_compared']}; "
            f"B2 hit or energy mismatches {cell['b2_call_or_hit_mismatch'] or 'none'}; "
            f"B2 same-energy call differences {cell['b2_same_energy_call_difference'] or 'none'}"
        )
        if cell["time_exceeds_1_10"] and not cell["time_real_regression"]:
            lines.append(
                f"  {cell['cell']} outside-time lower bound is above 1.10. "
                "It is not a time failure: that bound does not also exceed the A2/A upper bound on at least six pairs."
            )
    missing = [
        cell["cell"]
        for cell in payload["cells"]
        if cell["n_complete_pairs"] == 0 and cell["n_censored_rows"] == 0
    ]
    if missing:
        lines.append(
            "No rows were recorded for " + ", ".join(missing) + ". Those cells were not started."
        )
    slab = payload["slab"]
    lines.extend(
        [
            "",
            "## Slab call accounting",
            "",
            f"Verdict: {slab['verdict']}. Ledger pattern 1 then 2: {slab['ledger_ok']}.",
            "Grad-only engine calls are the budget-2 total minus the combined total.",
            "The calls column is the engine count.",
            (
                "Grad-only counts are A {A}, B {B}, B2 {B2}, A2 {A2}.".format(
                    **slab["grad_only_engine"]
                )
            ),
            "The FIX rule requires B's grad-only count to be strictly greater than A's.",
            "",
            "| Arm | Combined engine calls | Budget-2 engine calls | Grad-only engine calls |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for arm in ARMS:
        lines.append(
            f"| {arm} | {slab['combined_engine'][arm]} | {slab['withgrad_engine'][arm]} | {slab['grad_only_engine'][arm]} |"
        )
    lines.extend(["", "## GFN2 best geometries", ""])
    detail = payload["gfn2_rows"]
    seed_2000 = [row for row in detail if row["cell"] == "gfn2-si10" and row["seed"] == "2000"]
    by_arm = {row["arm"]: row for row in seed_2000}
    if "A" in by_arm and "B" in by_arm:
        lines.append(
            "Si10 seed 2000, 300 K. Arm A best {abest} eV, detached {adet}, calls {acalls}. "
            "Arm B best {bbest} eV, detached {bdet}, calls {bcalls}, fresh SCC {bscc}.".format(
                abest=by_arm["A"]["best"],
                adet=by_arm["A"]["detached"],
                acalls=by_arm["A"]["calls"],
                bbest=by_arm["B"]["best"],
                bdet=by_arm["B"]["detached"],
                bcalls=by_arm["B"]["calls"],
                bscc=by_arm["B"]["scc_converged"],
            )
        )
        if by_arm["A"]["best"] != by_arm["B"]["best"]:
            lines.append("The arms recorded different best energies, so B did not keep A's geometry.")

        def matches_prior(row: dict[str, str]) -> bool:
            energy = as_float(row["best"])
            separation = as_float(row["nn_max_A"])
            energy_hit = energy is not None and abs(energy - (-475.186)) <= 0.01
            separation_hit = separation is not None and abs(separation - 4.43) <= 0.05
            return energy_hit or separation_hit

        lines.append(
            "The earlier false hit to compare is -475.186 eV with a 4.43 A separation. "
            "Arm A matches that point: {a}. Arm B matches that point: {b}.".format(
                a=matches_prior(by_arm["A"]),
                b=matches_prior(by_arm["B"]),
            )
        )
        lines.append("")
    if not detail:
        lines.append("No GFN2 row has been recorded.")
    else:
        lines.append(
            "| Cell | Seed | Arm | Hit | Physical | Best | Calls | Censored | Detached | NN min | NN max | SCC | Force | Fresh eV |"
        )
        lines.append(
            "| --- | ---: | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | ---: | ---: |"
        )
        for row in detail:
            lines.append(
                "| {cell} | {seed} | {arm} | {hit} | {physical_hit} | {best} | {calls} | {censored} | {detached} | {nn_min_A} | {nn_max_A} | {scc_converged} | {max_force_eV_A} | {sp_energy_eV} |".format(
                    **row
                )
            )
    lines.extend(
        [
            "",
            "A physical hit is an energy hit that is connected and, for GFN2,",
            "whose fresh 300 K single point converged with maximum force below 1e-3 eV/A.",
            "B rejecting a non-physical A hit is listed above and is not a hit loss.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.rows)
    censored: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["kind"] != "slab" and row["phase"] == "" and row["censored"] == "true":
            censored[row["cell"]] += 1
    grouped = index_search(rows)
    cells = [
        search_cell(name, grouped.get(name, {}), censored[name]) for name in SEARCH_CELLS
    ]
    payload = {
        "cells": cells,
        "slab": slab_summary(rows),
        "gfn2_rows": gfn2_rows(rows),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.report.write_text(markdown(payload))
    print(args.report.read_text())


if __name__ == "__main__":
    main()
