#!/usr/bin/env python3
"""Run the opt-in search arms against the GFN2 preset and score them against CREST.

The jobs, start structures, budgets and validity rule are
``harness_rerun.py``'s: the campaign's 95 paired seeds plus trofinetide
seed 4, each at CREST's gradient budget. Every arm runs from one
``organic_bench`` binary, so its ``preset`` arm is the preset of the tree
that binary was built from.

A system's reference is the lowest valid energy over the oracle, every
CREST run, every campaign arm, the robustness rerun, the harness rerun and
every arm run here. A hit is within 0.1 kcal/mol of it, and an arm ties or
beats CREST at no more than 0.1 kcal/mol above that seed's CREST energy.
The paired sign test counts a seed better or worse only beyond 0.1 kcal/mol.

An arm is accepted when all four hold: a pooled exact McNemar p < 0.05 in
its favour on hits or on ties-or-beats-CREST; no system with fewer hits and
no system whose sign test is against it at p < 0.05; the LJ digest total of
the final tree unchanged; and an apart water board with as many hits as the
preset's.

Outside any decision the table also gives the hits against the harness
handover's references, which no run here can move, the calls per hop by
phase, and the runs whose best structure changed the start's constitution.

Usage::

    python experiments/gap_rerun.py --campaign CAMPAIGN jobs > jobs.txt
    xargs -P 7 -L 1 python experiments/gap_rerun.py --campaign CAMPAIGN --out OUT run ARM < jobs.txt
    python experiments/gap_rerun.py --campaign CAMPAIGN --before HARNESS --robust ROBUST \\
        --out OUT --boards BOARDS --before-board BOARD --lj LJ_DIGEST table
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
    scored,
    wall,
    write_csv,
)
from water_board import seeds as board_seeds  # noqa: E402

ARMS = ("preset", "loose", "chains", "tkick", "combo", "chains4")
DECIDED = ("loose", "chains", "tkick", "combo")
WATERS = ("w4", "w6", "w8")
LJ_TOTAL = "b593a05516799aeea1b208813c7a330dbd0d9cc363bfa0e444d3123be65071dc"
COUNTERS = (
    "HOPS",
    "REFUSED",
    "REFUSED_STARTS",
    "ENGINE_CALLS",
    "FAILED_CALLS",
    "NONFINITE_REQUESTS",
    "SOAP_FALLBACKS",
    "REJECTED_TRIALS",
    "ROTORS",
)
# Lines a self-check does not compare: the arm's name, timing, and the
# counters only this harness prints.
UNCOMPARED = re.compile(r"^(arm |WALL |WROTE |REJECTED_TRIALS |REFINES |ROTORS |CHAINS )")
# Lines the 0007 harness did not print.
NEW_SINCE_0007 = re.compile(r"^(WALL |REJECTED_TRIALS |REFINES |ROTORS |CHAINS )")


def run_seed(campaign: Path, out: Path, arm: str, system: str, seed: int) -> None:
    root = out / f"{system}_s{seed}"
    result = root / f"{arm}.json"
    if result.exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    record = campaign_record(campaign, system, seed)
    start = campaign / f"{system}_s{seed}" / "start.xyz"
    (root / "start.xyz").write_bytes(start.read_bytes())
    run = anneal(root / "start.xyz", root / arm, record["budget"], seed, arm)
    log = run["log"]
    entry = {
        "system": system,
        "seed": seed,
        "arm": arm,
        "budget": record["budget"],
        "best_eV": run["best"],
        "charged": run["charged"],
        "improvements": run["improvements"],
        "wall_s": wall(log),
    }
    for key in COUNTERS:
        entry[key.lower()] = counter(log, key)
    refines = re.search(r"^REFINES (\d+) charged (\d+)$", log, re.M)
    entry["refines"], entry["refine_charged"] = (int(v) for v in refines.groups())
    chains = re.search(r"^CHAINS (\d+) swaps (\d+) accepted (\d+)$", log, re.M)
    entry["chains"], entry["swaps"], entry["swaps_accepted"] = (int(v) for v in chains.groups())
    entry["arms"] = {
        m.group(1): int(m.group(2)) for m in re.finditer(r"^ARM (\S+) draws (\d+)", log, re.M)
    }
    result.write_text(json.dumps(entry, indent=1) + "\n")


def arms_in(root: Path, start: Path, seed: int) -> dict:
    arms = {}
    for path in sorted(root.glob("*.json")):
        run = json.loads(path.read_text())
        arm = run["arm"]
        best = root / arm / f"best_{arm}_s{seed}.xyz"
        arms[arm] = dict(run, best_eV=scored(start, best, run["best_eV"]))
    return arms


def load(campaign: Path, before: Path, robust: Path, out: Path) -> tuple[dict, dict]:
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
                "robust": arms_in(robust / f"{system}_s{seed}", start, seed),
                "before": arms_in(before / f"{system}_s{seed}", start, seed),
                "gap": arms_in(out / f"{system}_s{seed}", start, seed),
            }
            for arm in CAMPAIGN_ARMS:
                if arm in record:
                    best = root / arm / f"best_{arm}_s{seed}.xyz"
                    entry["campaign"][arm] = dict(
                        record[arm], best_eV=scored(start, best, record[arm]["best_eV"])
                    )
            seeds[(system, seed)] = entry
    return oracles, seeds


def references(
    oracles: dict, seeds: dict, sources: tuple = ("campaign", "robust", "before", "gap")
) -> dict:
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


def compared(base: dict, arm: dict, crest: float, reference: float) -> dict:
    target = reference + HIT_EV
    e_base, e_arm = base["best_eV"], arm["best_eV"]
    delta = minus(e_arm, e_base)
    change = "better" if delta < -0.1 else "worse" if delta > 0.1 else "equal"
    return {
        "base_eV": e_base,
        "arm_eV": e_arm,
        "base_hit": int(e_base <= target),
        "arm_hit": int(e_arm <= target),
        "base_ties_crest": int(e_base <= crest + HIT_EV),
        "arm_ties_crest": int(e_arm <= crest + HIT_EV),
        "arm_minus_base_kcal": delta,
        "arm_minus_crest_kcal": minus(e_arm, crest),
        "change": change,
    }


def summarise(name: str, rows: list[dict]) -> dict:
    n = len(rows)

    def total(key: str) -> int:
        return sum(r[key] for r in rows)

    def only(left: str, right: str) -> int:
        return sum(bool(r[left]) and not r[right] for r in rows)

    crest, base, arm = total("crest_hit"), total("base_hit"), total("arm_hit")
    better = sum(r["change"] == "better" for r in rows)
    worse = sum(r["change"] == "worse" for r in rows)
    return {
        "system": name,
        "n": n,
        "crest_hits": crest,
        "base_hits": base,
        "arm_hits": arm,
        "only_base": only("base_hit", "arm_hit"),
        "only_arm": only("arm_hit", "base_hit"),
        "mcnemar_hits": mcnemar(only("base_hit", "arm_hit"), only("arm_hit", "base_hit")),
        "base_ties_crest": total("base_ties_crest"),
        "arm_ties_crest": total("arm_ties_crest"),
        "only_base_ties": only("base_ties_crest", "arm_ties_crest"),
        "only_arm_ties": only("arm_ties_crest", "base_ties_crest"),
        "mcnemar_ties": mcnemar(
            only("base_ties_crest", "arm_ties_crest"), only("arm_ties_crest", "base_ties_crest")
        ),
        "better": better,
        "worse": worse,
        "equal": n - better - worse,
        "sign_p": mcnemar(better, worse),
        "fisher_arm_vs_crest": fisher(arm, n - arm, crest, n - crest),
        "median_arm_minus_crest_kcal": float(np.median([r["arm_minus_crest_kcal"] for r in rows])),
    }


def pooled_win(e: dict) -> bool:
    hits = e["mcnemar_hits"] < 0.05 and e["only_arm"] > e["only_base"]
    ties = e["mcnemar_ties"] < 0.05 and e["only_arm_ties"] > e["only_base_ties"]
    return hits or ties


def no_system_worse(systems: list[dict]) -> bool:
    return all(
        e["arm_hits"] >= e["base_hits"] and not (e["sign_p"] < 0.05 and e["worse"] > e["better"])
        for e in systems
    )


def body(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if not UNCOMPARED.match(line)]


def quench_split(out: Path, system: str, seed: int, arm: str) -> tuple[int, int, int]:
    """The run's ``QUENCH screen S full F check C`` line."""
    log = (out / f"{system}_s{seed}" / arm / "anneal.out").read_text()
    found = re.search(r"^QUENCH screen (\d+) full (\d+) check (\d+)$", log, re.M)
    return tuple(int(v) for v in found.groups())


def same_output(out: Path, system: str, seed: int, left: str, right: str) -> bool:
    root = out / f"{system}_s{seed}"
    return body(root / left / "anneal.out") == body(root / right / "anneal.out")


def lj_total(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("total "):
            return line.split()[1]
    return None


def board_hits(path: Path) -> dict:
    rows = {row["seed"]: row for row in board_seeds(path)}
    hits = {seed for seed, row in rows.items() if row["isomer_reason"] == "inside gate"}
    tally: dict[str, int] = {}
    for seed in sorted(hits, key=int):
        tally[rows[seed]["isomer"]] = tally.get(rows[seed]["isomer"], 0) + 1
    return {"rows": rows, "hits": hits, "tally": tally}


BOARD_FIELDS = ("isomer", "isomer_rmsd_A", "isomer_reason", "force_calls", "best_energy")


def boards(directory: Path | None, before: Path | None) -> dict:
    if directory is None:
        return {}
    found = {}
    for arm in ("preset",) + DECIDED:
        path = directory / arm / "board.txt"
        if path.exists():
            found[arm] = board_hits(path)
    if before is not None and before.exists():
        found["0007"] = board_hits(before)
    return found


def same_board(left: dict, right: dict) -> bool:
    keys = set(left["rows"]) | set(right["rows"])
    return all(
        key in left["rows"]
        and key in right["rows"]
        and all(left["rows"][key].get(f) == right["rows"][key].get(f) for f in BOARD_FIELDS)
        for key in keys
    )


def cell(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({lo:.2f}-{hi:.2f})"


def kcal(v: float) -> str:
    return f"{v:+.3f}" if math.isfinite(v) else ("inf" if v > 0 else "-inf")


def table(args: argparse.Namespace) -> None:
    out = args.out
    oracles, seeds = load(args.campaign, args.before, args.robust, out)
    refs = references(oracles, seeds)
    paired = [k for k, e in seeds.items() if "preset" in e["campaign"]]
    apart = [k for k, e in seeds.items() if "preset" not in e["campaign"]]
    ran = {arm for e in seeds.values() for arm in e["gap"]}
    lines: list[str] = []
    stats: dict = {"references": {s: {"eV": r[0], "from": r[1]} for s, r in refs.items()}}

    def rows_for(
        arm: str, base: str, keys: list, source: str = "gap", against: dict = refs
    ) -> list[dict]:
        rows = []
        for system, seed in keys:
            entry = seeds[(system, seed)]
            if arm not in entry["gap"] or base not in entry[source]:
                continue
            crest, reference = entry["crest_eV"], against[system][0]
            r = {"system": system, "seed": seed, "budget": entry["budget"]}
            r["reference_eV"] = reference
            r["crest_eV"] = crest
            r["crest_hit"] = int(crest <= reference + HIT_EV)
            r.update(compared(entry[source][base], entry["gap"][arm], crest, reference))
            run = entry["gap"][arm]
            r["hops"] = run["hops"]
            r["hops_per_budget"] = run["hops"] / entry["budget"]
            r["refines"] = run["refines"]
            r["refine_charged"] = run["refine_charged"]
            r["rejected_trials"] = run["rejected_trials"]
            r["refused_starts"] = run["refused_starts"]
            r["rotors"] = run["rotors"]
            r["chains"] = run["chains"]
            r["accounting_ok"] = int(run["engine_calls"] == run["charged"] == entry["budget"])
            rows.append(r)
        return rows

    def block(rows: list[dict]) -> tuple[list[dict], dict]:
        systems = [summarise(s, [r for r in rows if r["system"] == s]) for s in SYSTEMS]
        return systems, summarise("all", rows)

    # Hits against CREST, every arm, on the paired seeds.
    lines += [
        "## Hits against CREST on the paired seeds",
        "",
        "| arm | hits | CREST hits | Fisher vs CREST | ties or beats CREST "
        "| median minus CREST (kcal/mol) |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        if arm not in ran:
            continue
        _, e = block(rows_for(arm, "preset", paired))
        lines.append(
            f"| {arm} | {cell(e['arm_hits'], e['n'])} | {cell(e['crest_hits'], e['n'])} "
            f"| {e['fisher_arm_vs_crest']:.3g} | {e['arm_ties_crest']}/{e['n']} "
            f"| {kcal(e['median_arm_minus_crest_kcal'])} |"
        )
    # The same hits against the references the harness handover scored with,
    # which no run here can move. Not part of any decision.
    frozen = references(oracles, seeds, sources=("campaign", "robust", "before"))
    stats["handover_references"] = {s: {"eV": r[0], "from": r[1]} for s, r in frozen.items()}
    stats["hits_against_handover_references"] = {}
    lines += [
        "",
        "## Hits against the harness handover's references (not in any decision)",
        "",
        "| arm | hits | CREST hits | Fisher vs CREST |",
        "| --- | --- | --- | ---: |",
    ]
    for arm in ARMS:
        if arm not in ran:
            continue
        _, e = block(rows_for(arm, "preset", paired, against=frozen))
        stats["hits_against_handover_references"][arm] = {
            "arm_hits": e["arm_hits"],
            "crest_hits": e["crest_hits"],
            "n": e["n"],
            "fisher_arm_vs_crest": e["fisher_arm_vs_crest"],
        }
        lines.append(
            f"| {arm} | {cell(e['arm_hits'], e['n'])} | {cell(e['crest_hits'], e['n'])} "
            f"| {e['fisher_arm_vs_crest']:.3g} |"
        )
    stats["arms"] = {}
    board = boards(args.boards, args.before_board)
    lj = lj_total(args.lj)
    for arm in [a for a in ARMS[1:] if a in ran]:
        rows = rows_for(arm, "preset", paired)
        systems, overall = block(rows)
        write_csv(out / f"{arm}_vs_preset.csv", rows)
        c1 = pooled_win(overall)
        c2 = no_system_worse(systems)
        c3 = lj == LJ_TOTAL if lj is not None else None
        if arm in board and "preset" in board:
            lost = sorted(board["preset"]["hits"] - board[arm]["hits"], key=int)
            gained = sorted(board[arm]["hits"] - board["preset"]["hits"], key=int)
            c4 = len(board[arm]["hits"]) >= len(board["preset"]["hits"])
        else:
            lost, gained, c4 = [], [], None
        decided = arm in DECIDED
        accepted = decided and all(c is True for c in (c1, c2, c3, c4))
        stats["arms"][arm] = {
            "systems": systems,
            "overall": overall,
            "pooled_win": c1,
            "no_system_worse": c2,
            "lj_unchanged": c3,
            "board_no_loss": c4,
            "board_seeds_lost": lost,
            "board_seeds_gained": gained,
            "accepted": accepted if decided else "diagnostic",
        }
        lines += [
            "",
            f"## {arm} against the preset"
            + ("" if decided else " (diagnostic, not in any decision)"),
            "",
            "| system | seeds | CREST hits | preset hits | arm hits | only preset | only arm "
            "| McNemar hits | ties preset | ties arm | McNemar ties | better | worse | equal "
            "| sign test |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
            "| ---: | ---: | ---: |",
        ]
        for e in systems + [overall]:
            lines.append(
                f"| {e['system']} | {e['n']} | {e['crest_hits']} | {e['base_hits']} "
                f"| {e['arm_hits']} | {e['only_base']} | {e['only_arm']} "
                f"| {e['mcnemar_hits']:.3g} | {e['base_ties_crest']} | {e['arm_ties_crest']} "
                f"| {e['mcnemar_ties']:.3g} | {e['better']} | {e['worse']} | {e['equal']} "
                f"| {e['sign_p']:.3g} |"
            )
        if decided:
            lines += [
                "",
                f"decision {arm}: pooled_win {c1} no_system_worse {c2} lj_unchanged {c3} "
                f"board_no_loss {c4} -> {'accepted' if accepted else 'rejected'}",
            ]
            if arm in board:
                lines.append(
                    f"board {arm}: hits {len(board[arm]['hits'])}/32 against preset "
                    f"{len(board['preset']['hits'])}/32, seeds lost "
                    f"{' '.join(lost) or 'none'}, gained {' '.join(gained) or 'none'}"
                )

    # Cost per hop.
    lines += [
        "",
        "## Hops per budget (median HOPS / budget over the paired seeds and trofinetide s4)",
        "",
        "| system | " + " | ".join(a for a in ARMS if a in ran) + " | loose refines (median) "
        "| loose refine share of calls |",
        "| --- | " + " | ".join("---:" for a in ARMS if a in ran) + " | ---: | ---: |",
    ]
    hop_rows = []
    for system in SYSTEMS + ("all",):
        keys = [k for k in paired + apart if system == "all" or k[0] == system]
        medians = {}
        for arm in ARMS:
            if arm not in ran:
                continue
            values = [
                seeds[k]["gap"][arm]["hops"] / seeds[k]["budget"]
                for k in keys
                if arm in seeds[k]["gap"]
            ]
            medians[arm] = float(np.median(values)) if values else math.nan
        loose = [seeds[k]["gap"]["loose"] for k in keys if "loose" in seeds[k]["gap"]]
        refines = float(np.median([r["refines"] for r in loose])) if loose else math.nan
        share = (
            sum(r["refine_charged"] for r in loose) / sum(r["charged"] for r in loose)
            if loose
            else math.nan
        )
        hop_rows.append(dict({"system": system}, **medians, refines=refines, share=share))
        lines.append(
            f"| {system} | "
            + " | ".join(f"{medians[a]:.4f}" for a in ARMS if a in ran)
            + f" | {refines:.1f} | {share:.3f} |"
        )
    write_csv(out / "hops_per_budget.csv", hop_rows)
    stats["hops_per_budget"] = hop_rows

    # Where the calls go, pooled over every run of a system.
    lines += [
        "",
        "## Calls per hop by phase (pooled over the paired seeds and trofinetide s4)",
        "",
        "Refine is the part of full that continues a loose quench; other is what "
        "CHARGED holds beyond screen, full and check.",
        "",
        "| system | arm | hops | hops per budget | calls per hop | screen | full | refine "
        "| check | other | refines per hop |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    phase_rows = []
    for system in SYSTEMS + ("all",):
        keys = [k for k in paired + apart if system == "all" or k[0] == system]
        for arm in ARMS:
            runs = [(k, seeds[k]["gap"][arm]) for k in keys if arm in seeds[k]["gap"]]
            if not runs:
                continue
            split = [quench_split(out, s, k, arm) for (s, k), _ in runs]
            hops = sum(r["hops"] for _, r in runs)
            calls = sum(r["charged"] for _, r in runs)
            screen, full, check = (sum(q[i] for q in split) for i in range(3))
            refine = sum(r["refine_charged"] for _, r in runs)
            row = {
                "system": system,
                "arm": arm,
                "hops": hops,
                "budget": sum(seeds[k]["budget"] for k, _ in runs),
                "calls": calls,
                "screen": screen,
                "full": full,
                "refine": refine,
                "check": check,
                "other": calls - screen - full - check,
                "refines": sum(r["refines"] for _, r in runs),
            }
            phase_rows.append(row)
            per = (lambda v: v / hops) if hops else (lambda v: math.nan)
            lines.append(
                f"| {system} | {arm} | {hops} | {hops / row['budget']:.4f} | {per(calls):.1f} "
                f"| {per(screen):.1f} | {per(full):.1f} | {per(refine):.1f} | {per(check):.1f} "
                f"| {per(row['other']):.1f} | {per(row['refines']):.2f} |"
            )
    write_csv(out / "calls_per_hop.csv", phase_rows)
    stats["calls_per_hop"] = phase_rows

    # Runs whose best structure changed the start's covalent graph, and so
    # score +inf whatever their energy.
    shown = [a for a in ARMS if a in ran]
    lines += [
        "",
        "## Runs whose best changed constitution (paired seeds and trofinetide s4)",
        "",
        "| system | " + " | ".join(shown) + " |",
        "| --- | " + " | ".join("---:" for _ in shown) + " |",
    ]
    invalid_rows = []
    for system in SYSTEMS + ("all",):
        keys = [k for k in paired + apart if system == "all" or k[0] == system]
        counts = {
            a: sum(
                1 for k in keys if a in seeds[k]["gap"] and seeds[k]["gap"][a]["best_eV"] == math.inf
            )
            for a in shown
        }
        invalid_rows.append(dict({"system": system, "runs": len(keys)}, **counts))
        lines.append(f"| {system} | " + " | ".join(str(counts[a]) for a in shown) + " |")
    stats["changed_constitution"] = invalid_rows

    # Trofinetide seed 4, outside the paired set.
    if apart:
        shown = [a for a in ARMS if a in ran]
        lines += [
            "",
            "## Outside the paired set: minus CREST (kcal/mol)",
            "",
            "| run | " + " | ".join(shown) + " |",
            "| --- | " + " | ".join("---:" for _ in shown) + " |",
        ]
        for system, seed in apart:
            entry = seeds[(system, seed)]
            cells = [
                kcal(minus(entry["gap"][a]["best_eV"], entry["crest_eV"]))
                if a in entry["gap"]
                else "-"
                for a in shown
            ]
            lines.append(f"| {system} s{seed} | " + " | ".join(cells) + " |")

    # 0008 against 0007.
    rows = rows_for("preset", "preset", paired, source="before")
    systems, overall = block(rows)
    write_csv(out / "preset_0008_vs_0007.csv", rows)
    stats["preset_0008_vs_0007"] = {"systems": systems, "overall": overall}
    lines += [
        "",
        "## 0008 preset against the 0007 preset (harness rerun)",
        "",
        f"hits 0007 {overall['base_hits']}/{overall['n']} 0008 {overall['arm_hits']}/{overall['n']} "
        f"only 0007 {overall['only_base']} only 0008 {overall['only_arm']} "
        f"McNemar {overall['mcnemar_hits']:.3g}; ties-CREST 0007 {overall['base_ties_crest']} "
        f"0008 {overall['arm_ties_crest']}; better {overall['better']} worse {overall['worse']} "
        f"equal {overall['equal']}",
    ]

    # Self-checks.
    checks: dict = {}
    accounting = [
        (arm, s, k)
        for (s, k), e in seeds.items()
        for arm, run in e["gap"].items()
        if not run["engine_calls"] == run["charged"] == e["budget"]
    ]
    checks["accounting_failures"] = accounting
    every = paired + apart
    pairs = [("chains", "preset", SYSTEMS), ("tkick", "preset", WATERS), ("combo", "loose", WATERS)]
    for left, right, systems_in in pairs:
        if left in ran and right in ran:
            checks[f"{left}_equals_{right}"] = [
                f"{s}_s{k}"
                for s, k in every
                if s in systems_in and not same_output(out, s, k, left, right)
            ]
    untouched, differing = 0, []
    for system, seed in every:
        entry = seeds[(system, seed)]
        run = entry["gap"].get("preset")
        if run is None or "preset" not in entry["before"]:
            continue
        if run["rejected_trials"] or run["refused_starts"]:
            continue
        untouched += 1
        new = out / f"{system}_s{seed}" / "preset" / "anneal.out"
        old = args.before / f"{system}_s{seed}" / "preset" / "anneal.out"
        keep = [ln for ln in new.read_text().splitlines() if not NEW_SINCE_0007.match(ln)]
        then = [ln for ln in old.read_text().splitlines() if not NEW_SINCE_0007.match(ln)]
        if keep != then:
            differing.append(f"{system}_s{seed}")
    checks["0008_equals_0007_untouched_runs"] = untouched
    checks["0008_differs_from_0007_untouched"] = differing
    if board:
        for left, right in (("chains", "preset"), ("tkick", "preset"), ("combo", "loose")):
            if left in board and right in board:
                checks[f"board_{left}_equals_{right}"] = same_board(board[left], board[right])
        if "preset" in board and "0007" in board:
            checks["board_0008_equals_0007"] = same_board(board["preset"], board["0007"])
            checks["board_0008_hits"] = len(board["preset"]["hits"])
            checks["board_0007_hits"] = len(board["0007"]["hits"])
    checks["lj_total"] = lj
    stats["self_checks"] = checks
    stats["boards"] = {
        arm: {"hits": len(b["hits"]), "tally": b["tally"]} for arm, b in board.items()
    }
    lines += ["", "## Water boards (apart)", ""]
    for arm, b in board.items():
        tally = " ".join(f"{k} {v}" for k, v in sorted(b["tally"].items()))
        lines.append(f"board {arm}: {len(b['hits'])}/32 hits ({tally})")
    lines += ["", "## Self-checks", ""]
    for key, value in checks.items():
        shown = (" ".join(map(str, value)) or "none") if isinstance(value, list) else value
        lines.append(f"{key} {shown}")
    (out / "stats.json").write_text(json.dumps(stats, indent=1, default=str) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--before", type=Path, help="0007 harness rerun")
    parser.add_argument("--robust", type=Path, help="robustness rerun")
    parser.add_argument("--out", type=Path, default=Path("gap-rerun"))
    parser.add_argument("--boards", type=Path, help="directory of ARM/board.txt")
    parser.add_argument("--before-board", type=Path, help="0007 apart board.txt")
    parser.add_argument("--lj", type=Path, help="lj_digest output of the final tree")
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
        arm, system, seed = args.job
        if arm not in ARMS:
            parser.error(f"unknown arm {arm}")
        run_seed(args.campaign, args.out, arm, system, int(seed))
        return
    if args.before is None or args.robust is None:
        parser.error("table needs --before and --robust")
    table(args)


if __name__ == "__main__":
    main()
