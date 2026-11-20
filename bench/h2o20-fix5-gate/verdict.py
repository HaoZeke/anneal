#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write README.md for the h2o20-fix5 gate from the committed rows.

    verdict.py          write README.md
    verdict.py --check  fail unless README.md is what the rows give
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import panel  # noqa: E402
import water  # noqa: E402

ROOT = common.ROOT
README = HERE / "README.md"
OLD_ROWS = ROOT / "bench" / "h2o20-crest" / "seeds.csv"
OLD_REVISION = "131b411a31a2ef87669cba37b692e400819a595f"
DEVELOP = "61fa67cf380e46876b569da45e7cc091647a775b"
TIP = "bace7cfc229a1f9297c74b738af2b7ebfe059295"
LOOSE_TOL_EH = 1.0e-3


def onehost():
    spec = importlib.util.spec_from_file_location("run_onehost", water.ONEHOST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def g(value: float, digits: int = 4) -> str:
    return common.fmt(value, digits)


def med(rows: list[dict[str, str]], key: str) -> float:
    return common.median7([float(row[key]) for row in rows])


def per_call(rows: list[dict[str, str]]) -> list[float]:
    return [float(row["outside_s"]) / float(row["charged"]) for row in rows]


def table(header: list[str], body: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(line) + " |" for line in body]
    return out


def water_section(rows: list[dict[str, str]]) -> tuple[list[str], str]:
    harness = onehost()
    seeds = harness.SEEDS
    by = {(row["arm"], int(row["seed"])): row for row in rows}
    arms = ("anneal", "crest")
    complete = all((arm, seed) in by for arm in arms for seed in seeds)
    lines = ["## Part 1: GFN2 (H2O)20 at TIP, anneal against CREST", ""]
    lines += [
        f"Rows: `bench/h2o20-crest/rows-{water.TIP_TREE}.csv`, {len(rows)} of "
        f"{len(arms) * len(seeds)}. Seeds {seeds[0]}-{seeds[-1]}, budget "
        f"{harness.BUDGET} charged calls, the harness's `run_anneal` (default config: "
        "TIP4P surrogate stage, then the flexible GFN2 search, no capped Lindh) and "
        "`run_crest`, its vtight rescore. Even seeds ran anneal then CREST, odd seeds "
        "CREST then anneal, in one session.",
        "",
        f"A hit is a finite rescore at or below {water.REFERENCE_EH:.12f} Eh plus "
        f"{harness.HIT_TOL_EH:g} Eh, the reference and rule of the 131b411a panel. The "
        f"looser {LOOSE_TOL_EH:g} Eh rule is shown beside it. Calls to a hit rank "
        "misses worst. There is no A/A arm in Part 1, so the A/A ratio is n/a.",
        "",
    ]
    if not complete:
        lines += ["Part 1 is incomplete.", ""]
        return lines, "incomplete"
    cohort = {arm: [by[(arm, seed)] for seed in seeds] for arm in arms}
    hits = {arm: sum(row["hit"] == "1" for row in cohort[arm]) for arm in arms}
    loose = {
        arm: sum(math.isfinite(float(r["best"])) and float(r["best"]) <= water.REFERENCE_EH + LOOSE_TOL_EH for r in cohort[arm])
        for arm in arms
    }
    n = len(seeds)
    fisher = common.fisher(hits["anneal"], n, hits["crest"], n)
    fisher_loose = common.fisher(loose["anneal"], n, loose["crest"], n)
    _, calls_p = common.mann_whitney(
        common.ranked_calls(cohort["anneal"]), common.ranked_calls(cohort["crest"])
    )
    pc = {arm: common.median7(per_call(cohort[arm])) for arm in arms}
    body = []
    for arm in arms:
        body.append(
            [
                arm,
                str(n),
                f"{hits[arm]}/{n}",
                f"{loose[arm]}/{n}",
                g(fisher) if arm == "anneal" else "",
                g(calls_p) if arm == "anneal" else "",
                f"{med(cohort[arm], 'wall'):.3f}",
                f"{med(cohort[arm], 'inside_s'):.3f}",
                f"{med(cohort[arm], 'outside_s'):.3f}",
                f"{pc[arm]:.6e}",
                g(common.ratio(pc["anneal"], pc["crest"])) if arm == "anneal" else "",
                "n/a" if arm == "anneal" else "",
            ]
        )
    lines += table(
        [
            "Arm", "n", "Hits 1e-6", "Hits 1e-3", "Fisher p", "Calls MW p",
            "Wall median s", "Inside median s", "Outside median s",
            "Outside/charged median s", "anneal/CREST", "A/A",
        ],
        body,
    )
    lines += ["", f"Fisher p under the 1e-3 rule: {g(fisher_loose)}.", ""]
    energies = {arm: [float(r["best"]) for r in cohort[arm]] for arm in arms}
    _, energy_p = common.mann_whitney(energies["anneal"], energies["crest"])
    gaps = [a - c for a, c in zip(energies["anneal"], energies["crest"])]
    w_p, w_plus, w_minus, effect = harness.wilcoxon(gaps)
    lines += [
        "On the rescored energies, the harness's statistics give Mann-Whitney p "
        f"{g(energy_p, 10)}, paired Wilcoxon W+ {w_plus:.1f}, W- {w_minus:.1f}, exact p "
        f"{g(w_p, 10)}, rank-biserial {g(effect, 10)}, and median paired gap, anneal minus "
        f"CREST, {g(common.median7(gaps), 10)} Eh.",
        "",
        "Per row:",
        "",
    ]
    body = []
    for seed in seeds:
        for arm in sorted(arms, key=lambda a: int(by[(a, seed)]["slot"])):
            row = by[(arm, seed)]
            body.append(
                [
                    str(seed), row["slot"], arm, row["best"], row["hit"], row["calls"],
                    row["charged"], row["wall"], row["inside_s"], row["outside_s"],
                    f"{float(row['outside_s']) / float(row['charged']):.6e}", row["status"],
                ]
            )
    lines += table(
        ["Seed", "Slot", "Arm", "Rescored Eh", "Hit", "Calls", "Charged", "Wall s",
         "Inside s", "Outside s", "Outside/charged s", "Status"],
        body,
    )
    lines += [""] + old_diff(by, seeds, harness)
    verdict = (
        f"anneal {hits['anneal']}/{n} against CREST {hits['crest']}/{n} hits, Fisher p "
        f"{g(fisher)}, calls Mann-Whitney p {g(calls_p)}"
    )
    return lines, verdict


def old_diff(by, seeds, harness) -> list[str]:
    old = [
        row for row in common.read_rows(OLD_ROWS)
        if row["code_revision"] == OLD_REVISION and int(row["budget"]) == harness.BUDGET
    ]
    prior = {(row["arm"], int(row["seed"])): row for row in old}
    lines = [
        "### Against the 131b411a rows",
        "",
        "The 131b411a rows in `bench/h2o20-crest/seeds.csv` are unchanged. Their "
        "`anneal` arm was the example with the capped Lindh quench always on and auto "
        "TRIC chosen by default; TIP's default arm has both off. Those rows record no "
        "charged count, so charged identity is not available against them. Best "
        "identity is the rescored string.",
        "",
    ]
    body = []
    for seed in seeds:
        line = [str(seed)]
        for arm in ("anneal", "crest"):
            then = prior.get((arm, seed))
            now = by[(arm, seed)]
            if then is None:
                line += ["missing", now["best"], "n/a", "n/a"]
                continue
            delta = float(now["best"]) - float(then["rescored_Eh"])
            line += [
                then["rescored_Eh"], now["best"], f"{delta:+.12f}",
                "yes" if then["rescored_Eh"] == now["best"] else "no",
            ]
        capped = prior.get(("anneal-capped-lindh", seed))
        line.append(capped["rescored_Eh"] if capped else "missing")
        body.append(line)
    lines += table(
        ["Seed", "anneal 131b411a", "anneal TIP", "Delta Eh", "Same", "CREST 131b411a",
         "CREST now", "Delta Eh", "Same", "capped Lindh 131b411a"],
        body,
    )
    lines.append("")
    summary = []
    for arm in ("anneal", "crest"):
        then = [prior[(arm, s)] for s in seeds if (arm, s) in prior]
        now = [by[(arm, s)] for s in seeds]
        then_hits = sum(
            harness.energy_hits(float(r["rescored_Eh"]), water.REFERENCE_EH, harness.HIT_TOL_EH)
            for r in then
        )
        now_hits = sum(r["hit"] == "1" for r in now)
        summary.append(
            [
                arm,
                f"{then_hits}/{len(then)}",
                f"{now_hits}/{len(now)}",
                g(common.median7([float(r["rescored_Eh"]) for r in then]), 12) if then else "n/a",
                g(common.median7([float(r["best"]) for r in now]), 12),
                f"{med(then, 'wall_s'):.3f}" if then else "n/a",
                f"{med(now, 'wall'):.3f}",
                f"{med(then, 'pot_s'):.3f}" if then else "n/a",
                f"{med(now, 'inside_s'):.3f}",
                f"{med(then, 'outside_s'):.3f}" if then else "n/a",
                f"{med(now, 'outside_s'):.3f}",
            ]
        )
    lines += table(
        ["Arm", "Hits 131b411a", "Hits now", "Median Eh 131b411a", "Median Eh now",
         "Wall 131b411a", "Wall now", "Inside 131b411a", "Inside now",
         "Outside 131b411a", "Outside now"],
        summary,
    )
    return lines + [""]


def identical(a: dict[str, str], b: dict[str, str]) -> list[str]:
    return [key for key in ("best_bits", "calls", "charged") if a[key] != b[key]]


def panel_section(rows: list[dict[str, str]]) -> tuple[list[str], str]:
    by = {(r["system"], int(r["seed"]), r["arm"]): r for r in rows}
    lines = ["## Part 2: non-water bit identity, DEVELOP against TIP", ""]
    lines += [
        f"Rows: `bench/h2o20-fix5-gate/panel-rows.csv`, {len(rows)} of "
        f"{len(panel.CELLS) * len(panel.SEEDS) * 3}. A is DEVELOP `{DEVELOP}`, B is TIP "
        f"`{TIP}`, A2 a second fresh release build of DEVELOP. Seeds "
        f"{panel.SEEDS[0]}-{panel.SEEDS[-1]}, the drivers' paper budgets, no H2H_* "
        "variable. Even seeds ran A, B, A2 and odd seeds A2, B, A. A hit is a finite best "
        "at or below the reference plus 1e-3. Fisher and Mann-Whitney compare A with B. "
        "The cost columns are medians of outside_s/charged; B/A sits next to A2/A. "
        "Identity compares best_bits, calls and charged per seed.",
        "",
        "MW is read as the Gupta metal cells of `h2h_metal`; the panel is non-water and "
        "no driver in either tree has an mW water cell.",
        "",
    ]
    body = []
    broken = []
    complete = 0
    for family, system, _, _ in panel.CELLS:
        cell = {arm: [by.get((system, s, arm)) for s in panel.SEEDS] for arm in panel.TREES}
        if any(r is None for rs in cell.values() for r in rs):
            body.append([family, system] + ["incomplete"] * 10)
            continue
        complete += 1
        a, b, a2 = cell["develop"], cell["tip"], cell["develop-aa"]
        n = len(panel.SEEDS)
        hits = [sum(r["hit"] == "1" for r in rs) for rs in (a, b, a2)]
        _, calls_p = common.mann_whitney(common.ranked_calls(a), common.ranked_calls(b))
        pa, pb, pa2 = (common.median7(per_call(rs)) for rs in (a, b, a2))
        same_ab = []
        same_aa = []
        for seed, ra, rb, ra2 in zip(panel.SEEDS, a, b, a2):
            diff = identical(ra, rb)
            same_ab.append(f"{seed} {'yes' if not diff else 'no'}")
            if diff:
                broken.append((system, seed, "A vs B", diff))
            diff2 = identical(ra, ra2)
            same_aa.append(f"{seed} {'yes' if not diff2 else 'no'}")
            if diff2:
                broken.append((system, seed, "A vs A2", diff2))
        body.append(
            [
                family, system, a[0]["budget"], f"{hits[0]}/{n}", f"{hits[1]}/{n}",
                f"{hits[2]}/{n}", g(common.fisher(hits[0], n, hits[1], n)), g(calls_p),
                f"{pa:.4e} / {pb:.4e}", g(common.ratio(pb, pa)), g(common.ratio(pa2, pa)),
                ", ".join(same_ab) + "; A2: " + ", ".join(same_aa),
            ]
        )
    lines += table(
        ["Family", "Cell", "Budget", "Hits A", "Hits B", "Hits A2", "Fisher p", "Calls MW p",
         "Outside/charged A / B s", "B/A", "A2/A", "Identical per seed (A vs B; A vs A2)"],
        body,
    )
    lines.append("")
    if broken:
        lines.append("Non-identical cells:")
        lines.append("")
        for system, seed, pair, fields in broken:
            lines.append(f"- {system} seed {seed}, {pair}: {', '.join(fields)} differ")
    else:
        lines.append("Every complete cell is identical on every seed in best_bits, calls and charged.")
    lines.append("")
    ab_broken = sorted({s for s, _, pair, _ in broken if pair == "A vs B"})
    if complete < len(panel.CELLS):
        verdict = f"incomplete, {complete} of {len(panel.CELLS)} cells"
    elif ab_broken:
        verdict = "not bit-identical in " + ", ".join(ab_broken)
    else:
        verdict = f"bit-identical in all {complete} cells on every seed"
    return lines, verdict


def provenance() -> list[str]:
    lines = ["## Provenance", ""]
    pins = common.read_rows(HERE / "pins.csv")
    lines += table(
        ["Component", "Version", "Source", "sha256"],
        [[p["component"], p["version"], p["source"], f"`{p['sha256']}`"] for p in pins],
    )
    starts = common.read_rows(HERE / "starts.csv")
    same = sum(row["matches_c2de652d"] == "1" for row in starts)
    lines += [
        "",
        f"Starts: {len(starts)} regenerated by TIP's `gfn2_water_h2h` with "
        f"`H2H_START_ONLY=1`, the switch its header documents. {same} of {len(starts)} "
        "are byte-identical to the same switch at c2de652d, the original panel's "
        "revision. Hashes are in `starts.csv`.",
        "",
        "Arms were checked by tree only: DEVELOP "
        f"`{panel.TREES['develop']}`, TIP `{panel.TREES['tip']}`. Every row carries its "
        "library tree, the runner blob, the driver blob and the hashed machine id.",
        "",
    ]
    return lines


def render() -> str:
    water_lines, water_verdict = water_section(common.read_rows(water.ROWS))
    panel_lines, panel_verdict = panel_section(common.read_rows(panel.ROWS))
    machines = sorted(
        {r["machine_id"] for r in common.read_rows(water.ROWS) + common.read_rows(panel.ROWS)}
    )
    head = [
        "# h2o20-fix5 gate",
        "",
        "Written by `verdict.py` from the rows; do not edit by hand.",
        "",
        f"- Part 1: {water_verdict}.",
        f"- Part 2: {panel_verdict}.",
        f"- Hashed machine id: {', '.join(f'`{m}`' for m in machines) or 'none'}.",
        "",
        "Protocol: release builds, `taskset -c 0`, every thread variable at 1, one run "
        "at a time, ABBA arm order by seed, no warm-up and no shortened run. Medians are "
        "Hyndman-Fan type 7. Fisher is exact two-sided. Mann-Whitney is two-sided with "
        "tie and continuity correction.",
        "",
    ]
    return "\n".join(head + water_lines + panel_lines + provenance()).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        if README.read_text() != text:
            raise SystemExit("README.md is not what the rows give")
        return
    README.write_text(text)


if __name__ == "__main__":
    main()
