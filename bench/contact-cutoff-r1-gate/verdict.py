#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the contact-cutoff-r1 gate verdict from ``rows.csv``.

Part 1 passes when DEVELOP and CC agree on best_bits, calls, charged and
(for ``h2h_seed``) the trace digest on every LJ seed.

Part 2, per cell, CC against DEVELOP, rules declared before the runs:

- hits: fail if CC has fewer hits than DEVELOP (any directional loss);
  the exact two-sided Fisher p is reported;
- calls: fail if the exact two-sided Mann-Whitney p is below 0.05 and CC
  ranks worse; misses rank above every hit and tie with each other;
- cost: fail if the ratio CC/DEVELOP of type-7 medians of
  ``outside_s / charged`` is above 1.10 and above the A/A spread, the
  larger of DEVELOP2/DEVELOP and its reciprocal.

Gains are the mirror images. Overall PASS needs Part 1 identical, no
failing Part 2 cell, and no per-system or per-cell branch in the diff.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows.csv"
SESSION = HERE / "session.json"
README = HERE / "README.md"
SUMMARY = HERE / "summary.csv"

COST_LIMIT = 1.10
ALPHA = 0.05

SPECIAL_CASING = (
    "The patch changes one line of library code, `src/soap.rs:86` in the CC tree:\n\n"
    "```\n"
    "    CONTACT_CUTOFF * (spec.rcut_nn / SoapSpec::default().rcut_nn)\n"
    "```\n\n"
    "It replaces `let _ = spec; CONTACT_CUTOFF` (develop `src/soap.rs:86-87`). The other two "
    "files are a news fragment and `tests/contact_cutoff_length.rs`. No added line names a "
    "system, a size, a potential or a cell, and there is no new branch."
)

REACH = (
    "On develop, `soap::contact_cutoff` is read by `target_421`, `atom_w555`, `class_masses` "
    "and `ih_dominated` (`src/soap.rs:1306`, `:1413`, `:1434`, `:1454`). Within the "
    "search those are reached two ways. `fivefold_openings` calls `ih_dominated(x, "
    "SoapSpec::default())` (`src/soap.rs:2522`), which is length one in both arms. "
    "`ClusterMove::Soap` calls `soap::step_away` "
    "(`src/methods/cluster_hopping/moves.rs:2638`) only when `class` is set, and "
    "`class` is `cfg.soap_class_residual && packing` (`moves.rs:1172`). Every preset "
    "sets `soap_class_residual` to false (`src/methods/cluster_hopping/config.rs:1835`, "
    "`:1943`, `:2228`, `:2428`, `:2548`), and no campaign driver sets it to true. "
    "Under the canonical stack, then, no search on any of these cells passes a "
    "non-default spec to `contact_cutoff`."
)

CELLS_NOTE = (
    "Part 2 covers every campaign cell whose search does not run on Lennard-Jones "
    "lengths, as far as this machine can run them. The spec length is "
    "`SOAP_CUTOFF * cfg.length_scale` (`moves.rs:1171`). "
    "The Sutton-Chen panel (`h2h_sc`, nine cells in Cambridge units, nearest "
    "neighbour a/sqrt(2)), Au55 on `h2h_metal`, and the four Morse cells of the GMIN "
    "panel on `h2h_morse` all run `Config::recommended(n)`, so their `length_scale` "
    "is 1 even though their units are not Lennard-Jones units. The ASE Morse panel "
    "(`ase_nonlj`, `PairKind::Morse.length_scale()` = 2^(-1/6)) and ASE Au55 "
    "(`ase_panel`, Gupta dimer scales) run at a length other than 1, under both "
    "presets those runners call.\n\n"
    "Not run: `gfn2_covalent_h2h`, which always uses measured scales, because no xtb "
    "engine is on this machine. `h2h_covalent`'s `general` arm needs `H2H_ARM`, which "
    "the protocol forbids; its default arm is `Config::recommended` at length 1. "
    "`sc_h2h`, `sc_arm3` and `morse_table` use scaled configs, but no campaign rule runs them."
)


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = 1.0 + (len(ordered) - 1) * p
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def fisher_two_sided(a: int, n_a: int, b: int, n_b: int) -> float:
    total = a + b
    n = n_a + n_b

    def prob(x: int) -> float:
        return math.comb(n_a, x) * math.comb(n_b, total - x) / math.comb(n, total)

    observed = prob(a)
    lo, hi = max(0, total - n_b), min(n_a, total)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1.0 + 1e-7)))


def mann_whitney_exact(first: list[float], second: list[float]) -> tuple[float, float]:
    """Two-sided exact p on midranks, and the mean-rank difference second - first."""
    pooled = sorted(first + second)
    doubled: dict[float, int] = {}
    index = 0
    while index < len(pooled):
        end = index
        while end + 1 < len(pooled) and pooled[end + 1] == pooled[index]:
            end += 1
        doubled[pooled[index]] = index + end + 2
        index = end + 1
    ranks = [doubled[value] for value in first + second]
    k = len(first)
    counts: list[dict[int, int]] = [defaultdict(int) for _ in range(k + 1)]
    counts[0][0] = 1
    for rank in ranks:
        for taken in range(k, 0, -1):
            for total, ways in list(counts[taken - 1].items()):
                counts[taken][total + rank] += ways
    observed = sum(doubled[value] for value in first)
    mean = k * sum(ranks) / len(ranks)
    distance = abs(observed - mean)
    all_ways = math.comb(len(ranks), k)
    extreme = sum(ways for total, ways in counts[k].items() if abs(total - mean) >= distance - 1e-9)
    shift = sum(doubled[value] for value in second) / (2 * len(second)) - observed / (2 * k)
    return min(1.0, extreme / all_ways), shift


def calls_value(row: dict[str, str]) -> float:
    return float(row["calls"]) if row["hit"] == "1" else math.inf


def per_call(row: dict[str, str]) -> float:
    return float(row["outside_s"]) / max(int(row["charged"]), 1)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}" if math.isfinite(value) else "inf"


def part1(rows: list[dict[str, str]]) -> tuple[list[str], bool]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    order: list[tuple[str, str]] = []
    for row in rows:
        key = (row["cell"], row["seed"])
        if key not in grouped:
            order.append(key)
        grouped[key][row["arm"]] = row
    lines = [
        "| Cell | Seed | best DEVELOP | best CC | best_bits | calls | charged | trace digest | Identical |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    passed = True
    for key in order:
        pair = grouped[key]
        if set(pair) != {"DEVELOP", "CC"}:
            lines.append(f"| {key[0]} | {key[1]} | incomplete | | | | | | no |")
            passed = False
            continue
        left, right = pair["DEVELOP"], pair["CC"]
        checks = [left[field] == right[field] for field in ("best_bits", "calls", "charged", "trace_sha")]
        same = all(checks)
        passed &= same
        trace = "n/a" if not left["trace_sha"] else ("same" if checks[3] else "differs")
        lines.append(
            f"| {key[0]} | {key[1]} | {left['best']} | {right['best']} | "
            f"{'same' if checks[0] else 'differs'} | {left['calls']} / {right['calls']} | "
            f"{left['charged']} / {right['charged']} | {trace} | {'yes' if same else 'NO'} |"
        )
    return lines, passed


def part2(rows: list[dict[str, str]]) -> tuple[list[str], list[str], list[dict[str, str]], list[str], list[str]]:
    by_cell: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    order: list[str] = []
    for row in rows:
        if row["cell"] not in by_cell:
            order.append(row["cell"])
        by_cell[row["cell"]][row["arm"]].append(row)
    table = [
        "| Cell | Length | Hits D | Hits CC | Hits D' | Fisher p | MW p (calls) | CC calls rank | "
        "D out/call (s) | CC out/call (s) | CC/D | D'/D (A/A) | A/A spread | Seeds same CC=D | Seeds same D'=D | Fail | Gain |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    seeds_table = [
        "| Cell | Seed | best DEVELOP | best CC | best DEVELOP' | best_bits CC=D | charged CC=D | best_bits D'=D | charged D'=D |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    summary: list[dict[str, str]] = []
    fails: list[str] = []
    gains: list[str] = []
    for cell in order:
        arms = by_cell[cell]
        d, c, a = (sorted(arms.get(name, []), key=lambda r: int(r["seed"])) for name in ("DEVELOP", "CC", "DEVELOP2"))
        if not (len(d) == len(c) == len(a) and d):
            table.append(f"| {cell} | incomplete | | | | | | | | | | | | | | incomplete | |")
            fails.append(f"{cell}: incomplete")
            continue
        n = len(d)
        hits = [sum(r["hit"] == "1" for r in group) for group in (d, c, a)]
        fisher = fisher_two_sided(hits[1], n, hits[0], n)
        mw_p, shift = mann_whitney_exact([calls_value(r) for r in d], [calls_value(r) for r in c])
        rank = "worse" if shift > 0 else ("better" if shift < 0 else "tied")
        med = [type7([per_call(r) for r in group]) for group in (d, c, a)]
        ratio = med[1] / med[0]
        aa = med[2] / med[0]
        spread = max(aa, 1.0 / aa)
        same_cc = sum(x["best_bits"] == y["best_bits"] and x["charged"] == y["charged"] for x, y in zip(d, c))
        same_aa = sum(x["best_bits"] == y["best_bits"] and x["charged"] == y["charged"] for x, y in zip(d, a))
        why = []
        if hits[1] < hits[0]:
            why.append("hits")
        if mw_p < ALPHA and shift > 0:
            why.append("calls")
        if ratio > COST_LIMIT and ratio > spread:
            why.append("cost")
        good = []
        if hits[1] > hits[0]:
            good.append("hits")
        if mw_p < ALPHA and shift < 0:
            good.append("calls")
        if ratio < 1.0 / COST_LIMIT and ratio < 1.0 / spread:
            good.append("cost")
        if why:
            fails.append(f"{cell}: {', '.join(why)}")
        if good:
            gains.append(f"{cell}: {', '.join(good)}")
        table.append(
            f"| {cell} | {d[0]['length']} | {hits[0]}/{n} | {hits[1]}/{n} | {hits[2]}/{n} | {fisher:.4g} | {mw_p:.4g} | {rank} | "
            f"{med[0]:.6e} | {med[1]:.6e} | {ratio:.4f} | {aa:.4f} | {spread:.4f} | {same_cc}/{n} | {same_aa}/{n} | "
            f"{', '.join(why) or 'no'} | {', '.join(good) or 'no'} |"
        )
        for x, y, z in zip(d, c, a):
            seeds_table.append(
                f"| {cell} | {x['seed']} | {x['best']} | {y['best']} | {z['best']} | "
                f"{'yes' if x['best_bits'] == y['best_bits'] else 'NO'} | {'yes' if x['charged'] == y['charged'] else 'NO'} | "
                f"{'yes' if x['best_bits'] == z['best_bits'] else 'NO'} | {'yes' if x['charged'] == z['charged'] else 'NO'} |"
            )
        summary.append(
            {
                "cell": cell,
                "length": d[0]["length"],
                "budget": d[0]["budget"],
                "n": str(n),
                "hits_develop": str(hits[0]),
                "hits_cc": str(hits[1]),
                "hits_develop2": str(hits[2]),
                "fisher_p": repr(fisher),
                "mannwhitney_p_calls": repr(mw_p),
                "cc_calls_rank": rank,
                "calls_median_develop": repr(type7([calls_value(r) for r in d])),
                "calls_median_cc": repr(type7([calls_value(r) for r in c])),
                "outside_per_call_develop": repr(med[0]),
                "outside_per_call_cc": repr(med[1]),
                "outside_per_call_develop2": repr(med[2]),
                "ratio_cc_develop": repr(ratio),
                "ratio_aa": repr(aa),
                "aa_spread": repr(spread),
                "seeds_identical_cc": str(same_cc),
                "seeds_identical_aa": str(same_aa),
                "fail": ";".join(why),
                "gain": ";".join(good),
            }
        )
    return table, seeds_table, summary, fails, gains


def main() -> int:
    with ROWS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    sessions = json.loads(SESSION.read_text())
    p1_lines, p1_pass = part1([r for r in rows if r["part"] == "1"])
    table, seeds_table, summary, fails, gains = part2([r for r in rows if r["part"] == "2"])
    disagree = [
        f"{r['cell']} {r['arm']} {r['seed']}" for r in rows if r["driver_hit"] and r["driver_hit"] != r["hit"]
    ]
    overall = p1_pass and not fails
    trees = {arm: tree for s in sessions for arm, tree in s["trees"].items()}
    machines = sorted({r["machine_id"] for r in rows})
    blobs = sorted({r["harness_blob"] for r in rows})
    out = [
        "# contact-cutoff-r1 gate",
        "",
        f"Verdict: {'PASS' if overall else 'FAIL'}.",
        "",
        f"- Part 1, bit identity at length one: {'PASS' if p1_pass else 'FAIL'}.",
        f"- Part 2, non-unit-length cells: {'PASS' if not fails else 'FAIL'}.",
        "- Special casing: none.",
        "",
        "## Arms",
        "",
        "Arms were checked by tree, never by commit id, before the first run and after the last.",
        "",
        f"- DEVELOP: develop 61fa67cf, tree `{trees.get('DEVELOP', '')}`.",
        f"- CC: develop plus `git am -3` of the contact-cutoff-r1 mbox, tree `{trees.get('CC', '')}`.",
        f"- DEVELOP': a second, separate release build of develop, tree `{trees.get('DEVELOP2', '')}`.",
        "",
        f"Machine id (sha256): `{', '.join(machines)}`. Harness blob: `{', '.join(blobs)}`.",
        "",
        "## Protocol",
        "",
        "Release builds. Every run is `taskset -c 0`, with every thread variable at 1, and no "
        "`H2H_*` or `ANNEAL_*` variable set. Budgets are the paper budgets. Runs are serial. "
        "Within a seed the arms run in order on even seed positions and reversed on odd ones (ABBA). "
        "No warm-up and no shortened run. A hit is a finite best at or below the reference plus "
        "1e-3. `calls` is calls to the hit, a miss ranked above every hit. `best` is the driver's "
        "printed value (`h2h_seed` prints 10 decimals, the other drivers 8), and `best_bits` is the "
        "IEEE-754 bits of that printed value. Outside per call is `outside_s / charged` per row. "
        "Medians are Hyndman-Fan type 7. Fisher and Mann-Whitney p are exact and two-sided.",
        "",
        "Rules, declared before the runs: a Part 2 cell fails if CC has fewer hits than DEVELOP, "
        "if the calls Mann-Whitney p is below 0.05 with CC ranked worse, or if CC/DEVELOP "
        "outside per call is above 1.10 and outside the A/A spread (the larger of DEVELOP'/DEVELOP "
        "and its reciprocal). Gains are the mirror images.",
        "",
        "## Part 1: bit identity at length one",
        "",
        "LJ38, LJ55 and LJ75 at 400000, 1000000 and 4000000 calls, seeds 2000-2002, through the "
        "panel driver `h2h_gmin_proof` and the seed driver `h2h_seed`. The seed driver stops at the "
        "reference plus 1e-3 and writes the trace. `h2h_gmin_proof` writes no trace.",
        "",
        *p1_lines,
        "",
        "## Part 2: cells whose units are not Lennard-Jones units",
        "",
        CELLS_NOTE,
        "",
        "Seeds 2000-2011, n=12 per arm. D is DEVELOP, D' is DEVELOP'. Hits are CC against D. "
        "The Mann-Whitney test is on calls, CC against D.",
        "",
        *table,
        "",
        f"Failing cells: {'; '.join(fails) if fails else 'none'}.",
        "",
        f"Gains: {'; '.join(gains) if gains else 'none'}.",
        "",
        f"Rows where the driver's hit flag differs from the harness score: {', '.join(disagree) if disagree else 'none'}.",
        "",
        "### Per seed",
        "",
        *seeds_table,
        "",
        "## Special casing",
        "",
        SPECIAL_CASING,
        "",
        "## Reach",
        "",
        REACH,
        "",
        "## Files",
        "",
        "- `gate.py` runs the arms and appends `rows.csv`.",
        "- `session.json` holds the trees, binary and library digests, and the hashed machine id.",
        "- `verdict.py` writes this file and `summary.csv` from the rows.",
        "",
    ]
    README.write_text("\n".join(out))
    with SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]) if summary else ["cell"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(f"{'PASS' if overall else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
