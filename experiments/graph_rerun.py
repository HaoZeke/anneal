#!/usr/bin/env python3
"""Score the GFN2 conformer results by covalent-graph identity, and run the preset that keeps it.

The rule is ``Config::keep_bond_graph``'s. Atoms i and j bond when their
distance is below 1.25 times the sum of their Cordero covalent radii,
strictly. A structure's graph is its set of bonded index pairs, and a result
is valid when its best has the graph of the job's ``start.xyz``. That is
stricter than the canonical SMILES of ``robustness_rerun.py``: a hydrogen
that trades molecules changes the graph even when every molecule is intact.
An invalid result scores ``+inf`` and is never a hit, for anneal, for CREST
and for the oracle alike.

``rescore`` scores every stored source by both rules from the stored
structures, with no run, and freezes the graph-rule references in
``graph_refs.json``. ``run`` runs one job of the preset, which now keeps the
graph. ``table`` scores those runs against the rescored 0008 preset and
CREST, against references that include them, and applies the decision rule:
no seed the rescored preset hits is lost, no ethanol-dimer best is reacted,
and the LJ digests are byte-identical to the harness handover's.

Usage::

    python experiments/graph_rerun.py --campaign C --before H --robust R --gap G --out OUT rescore
    python experiments/graph_rerun.py --campaign C --out OUT run SYSTEM SEED
    python experiments/graph_rerun.py --campaign C --before H --robust R --gap G --out OUT \\
        --board B --gap-board GB --before-board BB --lj-harness L --lj FIX --lj FINAL table
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_rerun import cell, compared, kcal, minus, summarise  # noqa: E402
from gap_rerun import run_seed as gap_run_seed  # noqa: E402
from robustness_rerun import (  # noqa: E402
    CAMPAIGN_ARMS,
    COVALENT,
    HIT_EV,
    SEEDS,
    SYSTEMS,
    campaign_record,
    write_csv,
)
from robustness_rerun import valid as smiles_valid  # noqa: E402

TOLERANCE = 1.25
STORED = ("campaign", "robust", "before", "gap")
SIDES = {"crest": None, "0008 preset": ("gap", "preset"), "0007 preset": ("before", "preset")}
BOARD_FIELDS = ("isomer", "isomer_rmsd_A", "isomer_reason", "force_calls", "best_energy")
# Lines the graph-keeping run prints that the 0008 run did not, and the
# wall-clock time, which no two runs share.
NEW_LINES = re.compile(r"^(keep_bond_graph |GRAPH_REJECTIONS |WALL )")


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    n = int(lines[0])
    symbols, points = [], []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        symbols.append(parts[0].capitalize())
        points.append([float(v) for v in parts[1:4]])
    return symbols, np.array(points)


@functools.lru_cache(maxsize=None)
def bond_graph(path: Path) -> tuple[tuple[str, ...], frozenset[tuple[int, int]]]:
    """The species in order and the bonded index pairs, by the rule above."""
    symbols, x = read_xyz(path)
    pairs = set()
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            cut = TOLERANCE * (COVALENT[symbols[i]] + COVALENT[symbols[j]])
            d = x[i] - x[j]
            if float(d @ d) < cut * cut:
                pairs.add((i, j))
    return tuple(symbols), frozenset(pairs)


def graph_valid(start: Path, best: Path) -> bool:
    return best.exists() and bond_graph(start) == bond_graph(best)


RULES = {"smiles": smiles_valid, "graph": graph_valid}


def score(rule: str, start: Path, result: dict | None) -> float:
    if result is None or result.get("energy") is None:
        return math.inf
    return result["energy"] if RULES[rule](start, result["best"]) else math.inf


def reacted(rule: str, start: Path, result: dict | None) -> bool | None:
    """Whether a stored best changed the start's structure; ``None`` with no best."""
    if result is None or not result["best"].exists():
        return None
    return not RULES[rule](start, result["best"])


def stored_runs(root: Path, seed: int) -> dict:
    runs = {}
    for path in sorted(root.glob("*.json")):
        run = json.loads(path.read_text())
        arm = run["arm"]
        runs[arm] = dict(run, energy=run["best_eV"], best=root / arm / f"best_{arm}_s{seed}.xyz")
    return runs


def load(args: argparse.Namespace, with_runs: bool) -> tuple[dict, dict]:
    oracles = {}
    for system in SYSTEMS:
        root = args.campaign / f"{system}_oracle"
        record = json.loads((root / "result.json").read_text())
        oracles[system] = {
            "start": root / "start.xyz",
            "result": {"energy": record["energy_eV"], "best": root / "crest" / "crest_best.xyz"},
        }
    directories = {"robust": args.robust, "before": args.before, "gap": args.gap}
    if with_runs:
        directories["graph"] = args.out
    seeds = {}
    for system in SYSTEMS:
        for seed in SEEDS:
            root = args.campaign / f"{system}_s{seed}"
            if not root.exists():
                continue
            record = campaign_record(args.campaign, system, seed)
            entry = {
                "budget": record["budget"],
                "start": root / "start.xyz",
                "paired": "preset" in record,
                "crest": {"energy": record["crest_eV"], "best": root / "crest" / "crest_best.xyz"},
                "runs": {},
            }
            for arm in CAMPAIGN_ARMS:
                if arm in record:
                    entry["runs"][("campaign", arm)] = dict(
                        record[arm],
                        energy=record[arm]["best_eV"],
                        best=root / arm / f"best_{arm}_s{seed}.xyz",
                    )
            for source, directory in directories.items():
                for arm, run in stored_runs(directory / f"{system}_s{seed}", seed).items():
                    entry["runs"][(source, arm)] = run
            seeds[(system, seed)] = entry
    return oracles, seeds


def references(oracles: dict, seeds: dict, rule: str, sources: tuple) -> dict:
    refs = {}
    for system in SYSTEMS:
        oracle = oracles[system]
        candidates = [(score(rule, oracle["start"], oracle["result"]), "oracle")]
        for (name, seed), entry in seeds.items():
            if name != system:
                continue
            candidates.append((score(rule, entry["start"], entry["crest"]), f"crest s{seed}"))
            for (source, arm), run in entry["runs"].items():
                if source in sources:
                    candidates.append(
                        (score(rule, entry["start"], run), f"{source} {arm} s{seed}")
                    )
        refs[system] = min(candidates)
    return refs


def side(entry: dict, name: str) -> dict | None:
    return entry["crest"] if SIDES[name] is None else entry["runs"].get(SIDES[name])


def as_json(refs: dict) -> dict:
    return {s: {"eV": r[0], "from": r[1]} for s, r in refs.items()}


def rescore(args: argparse.Namespace) -> None:
    out = args.out
    oracles, seeds = load(args, with_runs=False)
    refs = {rule: references(oracles, seeds, rule, STORED) for rule in RULES}
    frozen = {
        "rule": "bonded when d < 1.25 (r_i + r_j), strictly; Cordero et al. 2008 radii "
        "(doi:10.1039/B801115J) H 0.31 C 0.76 N 0.71 O 0.66; identity of the set of "
        "bonded index pairs and of the species in order",
        "sources": ["oracle", "crest", *STORED],
        "references": as_json(refs["graph"]),
        "smiles_references": as_json(refs["smiles"]),
    }
    path = out / "graph_refs.json"
    if path.exists():
        kept = json.loads(path.read_text())
        if {k: kept.get(k) for k in frozen} != frozen:
            sys.exit(f"{path} is frozen and these references differ from it")
    else:
        frozen["written"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        path.write_text(json.dumps(frozen, indent=1) + "\n")

    rows = []
    for (system, seed), entry in seeds.items():
        row = {"system": system, "seed": seed, "paired": int(entry["paired"])}
        crest = {rule: score(rule, entry["start"], entry["crest"]) for rule in RULES}
        for name in SIDES:
            result = side(entry, name)
            key = name.split()[0]
            for rule in RULES:
                e = score(rule, entry["start"], result)
                row[f"{key}_{rule}_eV"] = e
                row[f"{key}_{rule}_hit"] = int(e <= refs[rule][system][0] + HIT_EV)
                if name != "crest":
                    row[f"{key}_{rule}_ties_crest"] = int(e <= crest[rule] + HIT_EV)
                state = reacted(rule, entry["start"], result)
                row[f"{key}_{rule}_reacted"] = "none" if state is None else int(state)
        rows.append(row)
    write_csv(out / "rescore.csv", rows)

    paired = [r for r in rows if r["paired"]]
    lines = [
        "## Rescore of the stored GFN2 board (95 paired seeds, no run)",
        "",
        "| side | rule | hits | ties or beats CREST | reacted bests | no best stored |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    summary: dict = {"references": {r: as_json(v) for r, v in refs.items()}, "sides": {}}
    for name in SIDES:
        key = name.split()[0]
        summary["sides"][name] = {}
        for rule in RULES:
            hits = sum(r[f"{key}_{rule}_hit"] for r in paired)
            ties = sum(r[f"{key}_{rule}_ties_crest"] for r in paired) if name != "crest" else None
            gone = [r for r in paired if r[f"{key}_{rule}_reacted"] == 1]
            missing = sum(r[f"{key}_{rule}_reacted"] == "none" for r in paired)
            summary["sides"][name][rule] = {
                "hits": hits,
                "ties_crest": ties,
                "reacted": len(gone),
                "reacted_seeds": [f"{r['system']}_s{r['seed']}" for r in gone],
                "no_best": missing,
            }
            lines.append(
                f"| {name} | {rule} | {hits}/{len(paired)} | "
                f"{'-' if ties is None else f'{ties}/{len(paired)}'} | {len(gone)} | {missing} |"
            )
        lost = [r for r in paired if r[f"{key}_smiles_hit"] and not r[f"{key}_graph_hit"]]
        gained = [r for r in paired if r[f"{key}_graph_hit"] and not r[f"{key}_smiles_hit"]]
        summary["sides"][name]["hits_lost"] = [f"{r['system']}_s{r['seed']}" for r in lost]
        summary["sides"][name]["hits_gained"] = [f"{r['system']}_s{r['seed']}" for r in gained]
    lines += ["", "## Hits that change from the SMILES rule to the graph rule", ""]
    for name in SIDES:
        s = summary["sides"][name]
        lines.append(
            f"{name}: lost {len(s['hits_lost'])} ({' '.join(s['hits_lost']) or 'none'}), "
            f"gained {len(s['hits_gained'])} ({' '.join(s['hits_gained']) or 'none'})"
        )
    lines += ["", "## References", "", "| system | SMILES rule | graph rule |", "| --- | --- | --- |"]
    for system in SYSTEMS:
        a, b = refs["smiles"][system], refs["graph"][system]
        lines.append(f"| {system} | {a[0]:.8f} ({a[1]}) | {b[0]:.8f} ({b[1]}) |")
    apart = [r for r in rows if not r["paired"]]
    if apart:
        lines += ["", "## Outside the paired set", ""]
        for r in apart:
            lines.append(
                f"{r['system']} s{r['seed']}: CREST {r['crest_graph_hit']} hit by the graph rule; "
                f"0008 preset reacted {r['0008_graph_reacted']}, 0007 preset reacted "
                f"{r['0007_graph_reacted']}"
            )
    boards = {}
    for label, board_path in (("0008 preset", args.gap_board), ("0007", args.before_board)):
        if board_path is None or not board_path.exists():
            continue
        records = board(board_path)
        boards[label] = {
            "seeds": len(records),
            "hits": len(board_hits(records)),
            "start_monomers_broken": sum(
                r.get("start_monomers_intact") != "true" for r in records.values()
            ),
            "best_monomers_broken": sum(
                r.get("best_monomers_intact") != "true" for r in records.values()
            ),
            "geometry_stored": False,
        }
    if boards:
        lines += [
            "",
            "## Water boards",
            "",
            "Neither stored board keeps a geometry. Each seed records only whether every "
            "oxygen of its start and of its best has exactly two hydrogens within 1.2 A, "
            "from any molecule, so the graph rule cannot be applied to them. The counts "
            "below are what is stored.",
            "",
            "| board | seeds | hits | start monomers broken | best monomers broken |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for label, b in boards.items():
            lines.append(
                f"| {label} | {b['seeds']} | {b['hits']} | {b['start_monomers_broken']} "
                f"| {b['best_monomers_broken']} |"
            )
    summary["boards"] = boards
    (out / "rescore.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    (out / "rescore.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def run_job(campaign: Path, out: Path, system: str, seed: int) -> None:
    gap_run_seed(campaign, out, "preset", system, seed)
    path = out / f"{system}_s{seed}" / "preset.json"
    entry = json.loads(path.read_text())
    if "graph_rejections" in entry:
        return
    log = (out / f"{system}_s{seed}" / "preset" / "anneal.out").read_text()
    found = re.search(r"^GRAPH_REJECTIONS (\d+) start (true|false)$", log, re.M)
    entry["graph_rejections"] = int(found.group(1))
    entry["start_graph_rejected"] = found.group(2) == "true"
    entry["keep_bond_graph"] = re.search(r"^keep_bond_graph (\S+)$", log, re.M).group(1) == "true"
    path.write_text(json.dumps(entry, indent=1) + "\n")


def board(path: Path) -> dict[str, dict[str, str]]:
    """Every ``key: value`` of each seed record of a water board report."""
    rows: dict[str, dict[str, str]] = {}
    current = None
    for line in path.read_text().splitlines():
        key, _, value = line.partition(": ")
        if key == "seed":
            current = rows.setdefault(value, {"seed": value})
        elif current is not None:
            current[key] = value
    return rows


def board_hits(rows: dict) -> set[str]:
    return {seed for seed, row in rows.items() if row.get("isomer_reason") == "inside gate"}


def lines_without_new(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if not NEW_LINES.match(ln)]


def table(args: argparse.Namespace) -> None:
    out = args.out
    frozen = json.loads((out / "graph_refs.json").read_text())
    oracles, seeds = load(args, with_runs=True)
    refs = references(oracles, seeds, "graph", STORED + ("graph",))
    stored_refs = references(oracles, seeds, "graph", STORED)
    if as_json(stored_refs) != frozen["references"]:
        sys.exit("the stored sources no longer give the frozen graph_refs.json references")
    frozen_refs = {s: (v["eV"], v["from"]) for s, v in frozen["references"].items()}
    paired = sorted(k for k, e in seeds.items() if e["paired"])
    every = sorted(seeds)
    lines: list[str] = []
    stats: dict = {"references": as_json(refs), "frozen_references": frozen["references"]}

    def rows_for(keys: list, against: dict) -> list[dict]:
        rows = []
        for system, seed in keys:
            entry = seeds[(system, seed)]
            run = entry["runs"].get(("graph", "preset"))
            base = entry["runs"].get(("gap", "preset"))
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
            r["graph_rejections"] = run["graph_rejections"]
            r["start_graph_rejected"] = int(run["start_graph_rejected"])
            r["hops"] = run["hops"]
            r["rejected_trials"] = run["rejected_trials"]
            r["accounting_ok"] = int(run["engine_calls"] == run["charged"] == entry["budget"])
            rows.append(r)
        return rows

    rows = rows_for(paired, refs)
    write_csv(out / "invariant_vs_preset.csv", rows)
    systems = [summarise(s, [r for r in rows if r["system"] == s]) for s in SYSTEMS]
    overall = summarise("all", rows)

    # D1, per seed.
    lost = [f"{r['system']}_s{r['seed']}" for r in rows if r["base_hit"] and not r["arm_hit"]]
    d1 = len(rows) == len(paired) and not lost
    # D2.
    etoh2 = [(k, seeds[k]["runs"].get(("graph", "preset"))) for k in every if k[0] == "etoh2"]
    etoh2_state = [reacted("graph", seeds[k]["start"], run) for k, run in etoh2]
    etoh2_reacted = sum(s is True for s in etoh2_state)
    etoh2_no_record = sum(s is None for s in etoh2_state)
    d2 = len(etoh2) == 16 and all(run is not None for _, run in etoh2) and etoh2_reacted == 0
    # D3.
    harness = args.lj_harness.read_bytes() if args.lj_harness else None
    lj = {str(p): p.exists() and p.read_bytes() == harness for p in args.lj or []}
    d3 = harness is not None and len(lj) == 2 and all(lj.values())
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
    lines += [
        "## Decision",
        "",
        f"D1 no hit lost: {d1} ({len(lost)} of {sum(r['base_hit'] for r in rows)} rescored "
        f"0008 hits lost: {' '.join(lost) or 'none'}; {len(rows)} of {len(paired)} paired seeds)",
        f"D2 no reacted ethanol dimer: {d2} ({etoh2_reacted} of {len(etoh2_state)} etoh2 bests "
        f"reacted; {etoh2_no_record} runs with no record)",
        f"D3 LJ digests identical: {d3} ("
        + ", ".join(f"{k} {v}" for k, v in lj.items())
        + ")",
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
        "invariant": overall["median_arm_minus_crest_kcal"],
    }
    lines.append(
        f"| 0008 preset (rescored) | {cell(overall['base_hits'], overall['n'])} "
        f"| {cell(overall['crest_hits'], overall['n'])} | - | {overall['base_ties_crest']}"
        f"/{overall['n']} | {kcal(medians['0008 preset (rescored)'])} |"
    )
    lines.append(
        f"| invariant | {cell(overall['arm_hits'], overall['n'])} "
        f"| {cell(overall['crest_hits'], overall['n'])} | {overall['fisher_arm_vs_crest']:.3g} "
        f"| {overall['arm_ties_crest']}/{overall['n']} | {kcal(medians['invariant'])} |"
    )
    stats["medians_minus_crest_kcal"] = medians
    lines += [
        "",
        "## Invariant against the rescored 0008 preset (reported, not decided)",
        "",
        "| system | seeds | CREST hits | 0008 hits | invariant hits | only 0008 | only invariant "
        "| McNemar hits | ties 0008 | ties invariant | McNemar ties | better | worse | equal "
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
    stats["systems"], stats["overall"] = systems, overall
    against_frozen = summarise("all", rows_for(paired, frozen_refs))
    stats["against_frozen_references"] = against_frozen
    lines += [
        "",
        f"Against the frozen graph_refs.json references (not in any decision): 0008 hits "
        f"{against_frozen['base_hits']}, invariant hits {against_frozen['arm_hits']}, CREST hits "
        f"{against_frozen['crest_hits']}, of {against_frozen['n']}.",
    ]

    # Graph rejections.
    lines += [
        "",
        "## Graph rejections of the invariant run (paired seeds and trofinetide s4)",
        "",
        "| system | runs | runs with a rejection | rejections | start rejected | bests reacted "
        "| no best |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rejection_rows = []
    for system in SYSTEMS + ("all",):
        keys = [k for k in every if system == "all" or k[0] == system]
        runs = [(k, seeds[k]["runs"].get(("graph", "preset"))) for k in keys]
        runs = [(k, r) for k, r in runs if r is not None]
        states = [reacted("graph", seeds[k]["start"], r) for k, r in runs]
        row = {
            "system": system,
            "runs": len(runs),
            "runs_with_rejection": sum(r["graph_rejections"] > 0 for _, r in runs),
            "rejections": sum(r["graph_rejections"] for _, r in runs),
            "start_rejected": sum(r["start_graph_rejected"] for _, r in runs),
            "bests_reacted": sum(s is True for s in states),
            "no_best": sum(s is None for s in states),
        }
        rejection_rows.append(row)
        lines.append("| " + " | ".join(str(v) for v in row.values()) + " |")
    write_csv(out / "graph_rejections.csv", rejection_rows)
    stats["graph_rejections"] = rejection_rows
    lines += [
        "",
        "A start rejection costs one zero-step relaxation: one evaluation and the "
        "relaxation's check. The calls spent in the other rejected quenches are not "
        "separated in the log.",
    ]
    apart = [k for k in every if not seeds[k]["paired"]]
    for system, seed in apart:
        entry = seeds[(system, seed)]
        run = entry["runs"].get(("graph", "preset"))
        crest = score("graph", entry["start"], entry["crest"])
        if run is not None:
            lines.append(
                f"{system} s{seed} (outside the paired set): invariant minus CREST "
                f"{kcal(minus(score('graph', entry['start'], run), crest))} kcal/mol, "
                f"graph rejections {run['graph_rejections']}"
            )

    # Self-checks.
    checks: dict = {}
    checks["S1_accounting_failures"] = [
        f"{s}_s{k}"
        for (s, k), e in seeds.items()
        if ("graph", "preset") in e["runs"]
        and not e["runs"][("graph", "preset")]["engine_calls"]
        == e["runs"][("graph", "preset")]["charged"]
        == e["budget"]
    ]
    untouched, differing = 0, []
    for system, seed in every:
        run = seeds[(system, seed)]["runs"].get(("graph", "preset"))
        if run is None or run["graph_rejections"]:
            continue
        untouched += 1
        new = out / f"{system}_s{seed}" / "preset" / "anneal.out"
        old = args.gap / f"{system}_s{seed}" / "preset" / "anneal.out"
        if lines_without_new(new) != lines_without_new(old):
            differing.append(f"{system}_s{seed}")
    checks["S2_untouched_runs"] = untouched
    checks["S2_untouched_runs_differing"] = differing
    checks["S3_invariant_bests_with_another_graph"] = [
        f"{s}_s{k}"
        for (s, k), e in seeds.items()
        if reacted("graph", e["start"], e["runs"].get(("graph", "preset"))) is True
    ]

    # Water boards.
    board_rows = []
    if args.board is not None and args.board.exists():
        new = board(args.board)
        old = board(args.gap_board) if args.gap_board is not None else {}
        geometry = args.board.parent / "geometry"
        for key in sorted(new, key=int):
            row = new[key]
            start, best = geometry / f"start_s{key}.xyz", geometry / f"best_s{key}.xyz"
            kept = (graph_valid(start, best) if best.exists() else None) if start.exists() else None
            board_rows.append(
                {
                    "seed": int(key),
                    "isomer": row.get("isomer"),
                    "hit": int(row.get("isomer_reason") == "inside gate"),
                    "gap_hit": int(old.get(key, {}).get("isomer_reason") == "inside gate"),
                    "graph_rejections": row.get("graph_rejections"),
                    "start_graph_rejected": row.get("start_graph_rejected"),
                    "best_keeps_graph": "none" if kept is None else int(kept),
                    "force_calls": row.get("force_calls"),
                }
            )
        write_csv(out / "board_graph.csv", board_rows)
        untouched_board = [k for k in new if new[k].get("graph_rejections") == "0"]
        checks["S4_untouched_board_seeds"] = len(untouched_board)
        checks["S4_untouched_board_seeds_differing"] = sorted(
            (
                k
                for k in untouched_board
                if k not in old
                or any(new[k].get(f) != old[k].get(f) for f in BOARD_FIELDS)
            ),
            key=int,
        )
        lost_board = sorted(board_hits(old) - board_hits(new), key=int)
        gained_board = sorted(board_hits(new) - board_hits(old), key=int)
        stats["board"] = {
            "hits": len(board_hits(new)),
            "gap_hits": len(board_hits(old)),
            "seeds_lost": lost_board,
            "seeds_gained": gained_board,
            "seeds_with_rejection": sum(r["graph_rejections"] not in ("0", None) for r in board_rows),
            "start_rejected": sum(r["start_graph_rejected"] == "true" for r in board_rows),
            "bests_reacted": sum(r["best_keeps_graph"] == 0 for r in board_rows),
            "no_best": sum(r["best_keeps_graph"] == "none" for r in board_rows),
        }
        b = stats["board"]
        lines += [
            "",
            "## Water board with the invariant (apart, reported, not decided)",
            "",
            f"hits {b['hits']}/{len(new)} against the 0008 preset board's {b['gap_hits']}/"
            f"{len(old)}; seeds lost {' '.join(lost_board) or 'none'}, gained "
            f"{' '.join(gained_board) or 'none'}; seeds with a graph rejection "
            f"{b['seeds_with_rejection']}, start rejected {b['start_rejected']}, bests with "
            f"another graph {b['bests_reacted']}, no best {b['no_best']}",
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
    parser.add_argument("--out", type=Path, default=Path("graph-rerun"))
    parser.add_argument("--board", type=Path, help="invariant water board.txt")
    parser.add_argument("--gap-board", type=Path, help="0008 preset water board.txt")
    parser.add_argument("--before-board", type=Path, help="0007 apart water board.txt")
    parser.add_argument("--lj-harness", type=Path, help="lj_digest output of the harness")
    parser.add_argument("--lj", type=Path, action="append", help="lj_digest output to compare")
    parser.add_argument("command", choices=["rescore", "run", "table"])
    parser.add_argument("job", nargs="*")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        system, seed = args.job
        run_job(args.campaign, args.out, system, int(seed))
        return
    if args.before is None or args.robust is None or args.gap is None:
        parser.error(f"{args.command} needs --before, --robust and --gap")
    if args.command == "rescore":
        rescore(args)
    else:
        table(args)


if __name__ == "__main__":
    main()
