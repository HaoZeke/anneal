#!/usr/bin/env python3
"""Rerun the GFN2 conformer campaign's preset arm and score it against the campaign.

Each (system, seed) reuses the campaign's ``start.xyz`` and budget: CREST
``--gfn2 -mquick -mdlen x0.05 -T 1``'s energy-and-gradient count plus the
xtb rescore. A seed the campaign left without ``result.json`` takes both
from its ``crest`` directory. The rerun arm is ``preset``,
``Config::recommended_molecular``; ``run`` takes another arm name for a
binary that has one.

Scoring is the campaign's. A best structure whose covalent constitution
differs from the start scores ``+inf``. A system's reference is the lowest
valid energy of its oracle, every CREST run, every campaign arm and every
rerun arm; a hit is within 0.1 kcal/mol of it. The paired comparison is
campaign preset against rerun preset over seeds both finished.

Usage::

    python experiments/robustness_rerun.py --campaign CAMPAIGN jobs > jobs.txt
    xargs -P 7 -L 1 python experiments/robustness_rerun.py --campaign CAMPAIGN --out OUT run < jobs.txt
    python experiments/robustness_rerun.py --campaign CAMPAIGN --out OUT table
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
from organic_low_budget import HARTREE_EV, KCAL_PER_EV, anneal, crest_curve  # noqa: E402

SYSTEMS = ("w4", "w6", "hexane", "w8", "etoh2", "trofinetide")
SEEDS = range(16)
CAMPAIGN_ARMS = ("preset", "container")
HIT_KCAL = 0.1
HIT_EV = HIT_KCAL / KCAL_PER_EV
COVALENT = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}


def constitution(path: Path) -> str:
    """Canonical SMILES of the covalent bond graph, single bonds, explicit atoms."""
    from rdkit import Chem

    lines = path.read_text().splitlines()
    n = int(lines[0])
    symbols = []
    points = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        symbols.append(parts[0].capitalize())
        points.append([float(v) for v in parts[1:4]])
    x = np.array(points)
    mol = Chem.RWMol()
    for symbol in symbols:
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(x[i] - x[j]) < 1.25 * (COVALENT[symbols[i]] + COVALENT[symbols[j]]):
                mol.AddBond(i, j, Chem.BondType.SINGLE)
    mol.UpdatePropertyCache(strict=False)
    return Chem.MolToSmiles(mol, canonical=True, allHsExplicit=True)


def valid(start: Path, best: Path) -> bool:
    return best.exists() and constitution(start) == constitution(best)


def campaign_record(campaign: Path, system: str, seed: int) -> dict:
    root = campaign / f"{system}_s{seed}"
    result = root / "result.json"
    if result.exists():
        return json.loads(result.read_text())
    text = (root / "crest" / "crest.out").read_text()
    calls = int(re.search(r"Total number of energy\+grad calls:\s+(\d+)", text).group(1))
    log = (root / "crest" / "rescore" / "xtb.log").read_text()
    energy = float(re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", log).group(1)) * HARTREE_EV
    return {
        "system": system,
        "seed": seed,
        "budget": calls + 1,
        "crest_eV": energy,
        "crest_curve": crest_curve(root / "crest", calls),
    }


def counter(log: str, key: str) -> int | None:
    for line in log.splitlines():
        if line.startswith(key + " "):
            return int(line.split()[1])
    return None


def wall(log: str) -> float | None:
    for line in log.splitlines():
        if line.startswith("WALL "):
            return float(line.split()[1])
    return None


def run_seed(campaign: Path, out: Path, system: str, seed: int, arm: str) -> None:
    root = out / f"{system}_s{seed}"
    result = root / f"{arm}.json"
    if result.exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    record = campaign_record(campaign, system, seed)
    start = campaign / f"{system}_s{seed}" / "start.xyz"
    (root / "start.xyz").write_bytes(start.read_bytes())
    run = anneal(root / "start.xyz", root / arm, record["budget"], seed, arm)
    entry = {
        "system": system,
        "seed": seed,
        "arm": arm,
        "budget": record["budget"],
        "best_eV": run["best"],
        "charged": run["charged"],
        "improvements": run["improvements"],
        "wall_s": wall(run["log"]),
        "refused": counter(run["log"], "REFUSED"),
        "soap_fallbacks": counter(run["log"], "SOAP_FALLBACKS"),
    }
    result.write_text(json.dumps(entry, indent=1) + "\n")


def scored(start: Path, best: Path, energy: float | None) -> float:
    if energy is None or not valid(start, best):
        return math.inf
    return energy


def load(campaign: Path, out: Path) -> tuple[dict, dict]:
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
                "system": system,
                "seed": seed,
                "budget": record["budget"],
                "crest_eV": scored(start, root / "crest" / "crest_best.xyz", record["crest_eV"]),
                "campaign": {},
                "rerun": {},
            }
            for arm in CAMPAIGN_ARMS:
                if arm in record:
                    best = root / arm / f"best_{arm}_s{seed}.xyz"
                    entry["campaign"][arm] = dict(
                        record[arm], best_eV=scored(start, best, record[arm]["best_eV"])
                    )
            rerun_root = out / f"{system}_s{seed}"
            for path in sorted(rerun_root.glob("*.json")):
                run = json.loads(path.read_text())
                arm = run["arm"]
                best = rerun_root / arm / f"best_{arm}_s{seed}.xyz"
                entry["rerun"][arm] = dict(run, best_eV=scored(start, best, run["best_eV"]))
            seeds[(system, seed)] = entry
    return oracles, seeds


def references(oracles: dict, seeds: dict) -> dict:
    refs = {}
    for system in SYSTEMS:
        candidates = [(oracles[system], "oracle")]
        for (name, seed), entry in seeds.items():
            if name != system:
                continue
            candidates.append((entry["crest_eV"], f"crest s{seed}"))
            for arm, run in entry["campaign"].items():
                candidates.append((run["best_eV"], f"campaign {arm} s{seed}"))
            for arm, run in entry["rerun"].items():
                candidates.append((run["best_eV"], f"rerun {arm} s{seed}"))
        refs[system] = min(candidates)
    return refs


def same_record(old: dict, new: dict) -> bool:
    """Equal charged calls, best energy and every improvement (call, energy)."""
    return (
        old["charged"] == new["charged"]
        and old["best_eV"] == new["best_eV"]
        and [tuple(p) for p in old["improvements"]] == [tuple(p) for p in new["improvements"]]
    )


def cell(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({lo:.2f}-{hi:.2f})"


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0].keys())
    with path.open("w") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(
                ",".join(
                    f"{row[k]:.8f}" if isinstance(row[k], float) else str(row[k]) for k in keys
                )
                + "\n"
            )


def table(campaign: Path, out: Path) -> None:
    oracles, seeds = load(campaign, out)
    refs = references(oracles, seeds)
    rows = []
    extra = []
    summary = []
    pooled = {"n": 0, "crest": 0, "old": 0, "new": 0, "only_old": 0, "only_new": 0, "same": 0}
    for system in SYSTEMS:
        reference, holder = refs[system]
        target = reference + HIT_EV
        tally = {"n": 0, "crest": 0, "old": 0, "new": 0, "only_old": 0, "only_new": 0, "same": 0}
        for seed in SEEDS:
            entry = seeds.get((system, seed))
            if entry is None:
                continue
            crest = entry["crest_eV"]
            base = {
                "system": system,
                "seed": seed,
                "budget": entry["budget"],
                "reference_eV": reference,
                "crest_eV": crest,
                "crest_gap_kcal": (crest - reference) * KCAL_PER_EV,
                "crest_hit": int(crest <= target),
            }
            old = entry["campaign"].get("preset")
            new = entry["rerun"].get("preset")
            if old is None or new is None:
                arms = dict(entry["rerun"])
                if not arms:
                    continue
                for arm, run in sorted(arms.items()):
                    extra.append(
                        dict(
                            base,
                            arm=arm,
                            best_eV=run["best_eV"],
                            gap_kcal=(run["best_eV"] - reference) * KCAL_PER_EV,
                            hit=int(run["best_eV"] <= target),
                            minus_crest_kcal=(run["best_eV"] - crest) * KCAL_PER_EV,
                            charged=run["charged"],
                            refused=run["refused"],
                            soap_fallbacks=run["soap_fallbacks"],
                            wall_s=run["wall_s"],
                        )
                    )
                continue
            old_hit = old["best_eV"] <= target
            new_hit = new["best_eV"] <= target
            same = same_record(old, new)
            rows.append(
                dict(
                    base,
                    old_preset_eV=old["best_eV"],
                    old_preset_hit=int(old_hit),
                    new_preset_eV=new["best_eV"],
                    new_preset_hit=int(new_hit),
                    new_minus_old_kcal=(new["best_eV"] - old["best_eV"]) * KCAL_PER_EV,
                    new_minus_crest_kcal=(new["best_eV"] - crest) * KCAL_PER_EV,
                    same_record=int(same),
                    old_charged=old["charged"],
                    new_charged=new["charged"],
                    refused=new["refused"],
                    soap_fallbacks=new["soap_fallbacks"],
                    old_wall_s=old["wall_s"],
                    new_wall_s=new["wall_s"],
                )
            )
            tally["n"] += 1
            tally["crest"] += crest <= target
            tally["old"] += old_hit
            tally["new"] += new_hit
            tally["only_old"] += old_hit and not new_hit
            tally["only_new"] += new_hit and not old_hit
            tally["same"] += same
        for key in pooled:
            pooled[key] += tally[key]
        n = tally["n"]
        summary.append(
            {
                "system": system,
                "reference_eV": reference,
                "reference_from": holder,
                **tally,
                "fisher_new_vs_old": fisher(
                    tally["new"], n - tally["new"], tally["old"], n - tally["old"]
                ),
                "mcnemar_new_vs_old": mcnemar(tally["only_new"], tally["only_old"]),
                "fisher_new_vs_crest": fisher(
                    tally["new"], n - tally["new"], tally["crest"], n - tally["crest"]
                ),
            }
        )
    n = pooled["n"]
    overall = {
        **pooled,
        "fisher_new_vs_old": fisher(
            pooled["new"], n - pooled["new"], pooled["old"], n - pooled["old"]
        ),
        "mcnemar_new_vs_old": mcnemar(pooled["only_new"], pooled["only_old"]),
        "fisher_new_vs_crest": fisher(
            pooled["new"], n - pooled["new"], pooled["crest"], n - pooled["crest"]
        ),
    }
    write_csv(out / "paired.csv", rows)
    if extra:
        write_csv(out / "unpaired.csv", extra)
    (out / "stats.json").write_text(
        json.dumps({"systems": summary, "overall": overall}, indent=1) + "\n"
    )
    lines = [
        "| system | paired seeds | CREST hits | campaign preset hits | rerun preset hits "
        "| only campaign | only rerun | same record | McNemar rerun vs campaign "
        "| Fisher rerun vs CREST | reference from |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for e in summary + [dict(overall, system="all", reference_from="")]:
        k = e["n"]
        lines.append(
            f"| {e['system']} | {k} | {cell(e['crest'], k)} | {cell(e['old'], k)} "
            f"| {cell(e['new'], k)} | {e['only_old']} | {e['only_new']} | {e['same']}/{k} "
            f"| {e['mcnemar_new_vs_old']:.3g} | {e['fisher_new_vs_crest']:.3g} "
            f"| {e['reference_from']} |"
        )
    if extra:
        lines += [
            "",
            "| system | seed | budget | arm | best eV | gap kcal/mol | hit | minus CREST kcal/mol "
            "| CREST gap kcal/mol | charged | refused | SOAP fallbacks | wall s |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for e in extra:
            lines.append(
                f"| {e['system']} | {e['seed']} | {e['budget']} | {e['arm']} | {e['best_eV']:.6f} "
                f"| {e['gap_kcal']:.3f} | {e['hit']} | {e['minus_crest_kcal']:.3f} "
                f"| {e['crest_gap_kcal']:.3f} | {e['charged']} | {e['refused']} "
                f"| {e['soap_fallbacks']} | {e['wall_s']} |"
            )
    changed = [r for r in rows if r["old_preset_hit"] != r["new_preset_hit"]]
    lines += ["", f"hit_changes {len(changed)}"]
    for r in changed:
        lines.append(
            f"  {r['system']} s{r['seed']} campaign {r['old_preset_hit']} rerun {r['new_preset_hit']} "
            f"delta {r['new_minus_old_kcal']:.3f} kcal/mol"
        )
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("robustness-rerun"))
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
        arm = args.job[2] if len(args.job) > 2 else "preset"
        run_seed(args.campaign, args.out, args.job[0], int(args.job[1]), arm)
        return
    table(args.campaign, args.out)


if __name__ == "__main__":
    main()
