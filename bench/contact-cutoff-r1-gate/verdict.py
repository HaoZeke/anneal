#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the gate verdict README from the committed rows.

    verdict.py <rows.csv> <provenance.csv> <README.md>

Every number in the README comes from the rows. The rules are fixed
here and were declared before the first run.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

BASE = "DEVELOP"
CANDIDATE = "CC"
REPEAT = "DEVELOP2"
COST_LIMIT = 1.10
P_LIMIT = 0.05

SPEC_LENGTH = """\
Spec length in the code. `contact_cutoff(spec)` now returns
`CONTACT_CUTOFF * (spec.rcut_nn / SoapSpec::default().rcut_nn)`
(CC `src/soap.rs:86`). The search builds its SOAP spec with
`rcut_nn = SOAP_CUTOFF * cfg.length_scale`
(`src/methods/cluster_hopping/moves.rs:361`, `:1171`), so the spec
length of a cell is its `Config::length_scale`. Every campaign driver
builds its config with `Config::recommended(n)`:
`examples/h2h_gmin_proof.rs:68`, `examples/h2h_sc.rs:210`,
`examples/h2h_metal.rs:253` (au55 is `Kind::CopperGold { n_copper: 0 }`,
`:79-84`) and `examples/h2h_morse.rs:159`. `Config::recommended` passes
`LennardJonesPreset::REDUCED_SCALE`
(`src/methods/cluster_hopping/config.rs:1786-1791`), which is 1.0
(`src/methods/cluster_hopping/preset.rs:7`). The Morse driver uses its
pair length only when `H2H_PAIR_SCALE=1` (`examples/h2h_morse.rs:163-167`),
and this gate sets no `H2H_*` variable. So, as the code is written, every
campaign cell has spec length 1, including the Sutton-Chen cells.

The cells below are the panel cells whose geometry is not at the
Lennard-Jones length. These are the cells the gate was asked to test:

- Sutton-Chen in Cambridge units, nearest neighbour `a/sqrt(2)`
  (`scripts/metal_h2h/run_sc.py:71`): sc126-38, sc126-55, sc126-80,
  sc108-38, sc108-55, sc108-80, sc96-38, sc96-55, sc96-80
  (`scripts/metal_h2h/run_sc.py:38-46`, `examples/h2h_sc.rs:93-101`).
- Gupta gold, `r0 = 2.884` angstrom (`scripts/gmin_panel/run_panel.py:109`):
  au55 (`scripts/gmin_panel/run_panel.py:84`, `:102`).
- Morse, pair minimum at 1, `PairKind::length_scale` is
  `r_min / 2^(1/6)` (`src/potentials.rs:166-167`): m55r6, m55r14, m80r6,
  m80r14 (`scripts/gmin_panel/run_panel.py:85-88`, `:103-106`).

The remaining panel cells are Lennard-Jones (`scripts/gmin_panel/run_panel.py:76-83`).

Reach of the change. `contact_cutoff(spec)` is called from `target_421`,
`atom_w555`, `class_masses` and `ih_dominated` (`src/soap.rs:1306`,
`:1413`, `:1434`, `:1454`). In a search, these are reached through
`soap::step_away` (`src/methods/cluster_hopping/moves.rs:2638`) only when
`class` is set, and `class` is `cfg.soap_class_residual && packing`
(`moves.rs:1172`). Every preset sets `soap_class_residual` to false
(`config.rs:1835`, `:1943`, `:2228`, `:2428`, `:2548`).
`fivefold_openings` calls `ih_dominated` with `SoapSpec::default()`
(`src/soap.rs:2522`), which is unchanged."""

SPECIAL_CASING = """\
None. The diff changes `src/soap.rs:86` (the return value of
`contact_cutoff`), rewrites `tests/contact_cutoff_length.rs`, and adds
`docs/newsfragments/+contact-cutoff.fixed.rst`. None of the added
lines names a system, a size, a cell, a potential or an element, and
the diff adds no branch."""


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def type7(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = 1.0 + (len(ordered) - 1) * p
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def fisher_lower(hits_x: int, n_x: int, hits_y: int, n_y: int) -> float:
    """One-sided exact p that x has this few hits or fewer, given the margins."""
    hits = hits_x + hits_y
    ways = math.comb(n_x + n_y, hits)
    lo = max(0, hits - n_y)
    return sum(math.comb(n_x, k) * math.comb(n_y, hits - k) for k in range(lo, hits_x + 1)) / ways


def mann_whitney_greater(x: list[float], y: list[float]) -> float:
    """One-sided p that x ranks above y: normal approximation with the tie
    and continuity corrections. Every value tied gives p = 1."""
    n1, n2 = len(x), len(y)
    pooled = sorted(x + y)
    n = n1 + n2
    rank: dict[float, float] = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    u = sum(rank[v] for v in x) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return 1.0
    z = (u - n1 * n2 / 2.0 - 0.5) / math.sqrt(var)
    return min(1.0, 0.5 * math.erfc(z / math.sqrt(2.0)))


def ranked_calls(rows: list[dict[str, str]]) -> list[float]:
    return [float(r["calls"]) if r["hit"] == "1" else math.inf for r in rows]


def outside_per_call(rows: list[dict[str, str]]) -> float:
    return type7([float(r["outside_s"]) / int(r["charged"]) for r in rows], 0.5)


def by_seed(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    out = {int(r["seed"]): r for r in rows}
    if len(out) != len(rows):
        raise SystemExit("a seed appears twice in one arm")
    return out


def same(a: dict[int, dict[str, str]], b: dict[int, dict[str, str]], field: str) -> int:
    return sum(a[s][field] == b[s][field] for s in a)


def part1(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    groups: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r["system"]][r["arm"]].append(r)
    lines = [
        "| Cell | Seed | best_bits DEVELOP | best_bits CC | calls | charged | trace | Identical |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    failures = []
    for system in groups:
        base, cand = by_seed(groups[system][BASE]), by_seed(groups[system][CANDIDATE])
        if set(base) != set(cand):
            raise SystemExit(f"{system}: part 1 seeds differ")
        for seed in sorted(base):
            a, b = base[seed], cand[seed]
            checks = [a[f] == b[f] for f in ("best_bits", "calls", "charged", "trace")]
            ok = all(checks)
            if not ok:
                failures.append(f"part 1 {system} seed {seed}")
            mark = ["same" if c else "differs" for c in checks]
            lines.append(
                f"| {system} | {seed} | `{a['best_bits']}` | `{b['best_bits']}` | {mark[1]} "
                f"({a['calls']}) | {mark[2]} ({a['charged']}) | {mark[3]} | {'yes' if ok else 'no'} |"
            )
    return lines, failures


def part2(rows: list[dict[str, str]]):
    groups: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r["system"]][r["arm"]].append(r)
    cells = []
    for system, arms in groups.items():
        base, cand, rep = arms[BASE], arms[CANDIDATE], arms[REPEAT]
        n = len(base)
        if not (n == len(cand) == len(rep)):
            raise SystemExit(f"{system}: arms have different counts")
        hb, hc, hr = (sum(r["hit"] == "1" for r in g) for g in (base, cand, rep))
        cost_b, cost_c, cost_r = outside_per_call(base), outside_per_call(cand), outside_per_call(rep)
        ratio = cost_c / cost_b
        aa = cost_r / cost_b
        spread = max(aa, 1.0 / aa)
        sb, sc, sr = by_seed(base), by_seed(cand), by_seed(rep)
        cell = {
            "system": system,
            "n": n,
            "hits": (hb, hc, hr),
            "fisher_loss": fisher_lower(hc, n, hb, n),
            "fisher_gain": fisher_lower(hb, n, hc, n),
            "mw_loss": mann_whitney_greater(ranked_calls(cand), ranked_calls(base)),
            "mw_gain": mann_whitney_greater(ranked_calls(base), ranked_calls(cand)),
            "calls_median": (
                type7([float(r["calls"]) for r in base], 0.5),
                type7([float(r["calls"]) for r in cand], 0.5),
            ),
            "cost": (cost_b, cost_c, cost_r),
            "ratio": ratio,
            "aa": aa,
            "spread": spread,
            "bits_same": same(sb, sc, "best_bits"),
            "charged_same": same(sb, sc, "charged"),
            "trace_same": same(sb, sc, "trace"),
            "aa_bits_same": same(sb, sr, "best_bits"),
            "aa_charged_same": same(sb, sr, "charged"),
            "per_seed": [
                (s, sb[s]["best_bits"] == sc[s]["best_bits"], sb[s]["charged"] == sc[s]["charged"])
                for s in sorted(sb)
            ],
        }
        fails = []
        if hc < hb:
            fails.append("hits")
        if cell["mw_loss"] < P_LIMIT:
            fails.append("calls")
        if ratio > COST_LIMIT and ratio > spread:
            fails.append("cost")
        cell["fails"] = fails
        gains = []
        if hc > hb:
            gains.append("hits")
        if cell["mw_gain"] < P_LIMIT:
            gains.append("calls")
        if ratio < 1.0 / COST_LIMIT and 1.0 / ratio > spread:
            gains.append("cost")
        cell["gains"] = gains
        cells.append(cell)
    return cells


def main(argv: list[str]) -> int:
    rows_path, provenance_path, out = (Path(a) for a in argv)
    rows = read(rows_path)
    provenance = read(provenance_path)
    machines = sorted({r["machine"] for r in rows})
    harness = sorted({r["harness_blob"] for r in rows})
    trees = {arm: sorted({r["library_tree"] for r in rows if r["arm"] == arm}) for arm in (BASE, CANDIDATE, REPEAT)}
    budgets = {r["system"]: r["budget"] for r in rows}
    p1_lines, p1_fail = part1([r for r in rows if r["part"] == "1"])
    cells = part2([r for r in rows if r["part"] == "2"])
    p2_fail = [f"{c['system']} ({', '.join(c['fails'])})" for c in cells if c["fails"]]
    p2_gain = [f"{c['system']} ({', '.join(c['gains'])})" for c in cells if c["gains"]]
    verdict = "PASS" if not p1_fail and not p2_fail else "FAIL"

    text = [
        "# contact-cutoff-r1 gate",
        "",
        f"Verdict: {verdict}.",
        "",
        "## Arms",
        "",
        f"DEVELOP is develop `61fa67cf`, tree `{", ".join(trees[BASE])}`. CC is develop with "
        "`anneal-contact-cutoff-r1.mbox` applied by `git am -3` (one commit), "
        f"tree `{", ".join(trees[CANDIDATE])}`. DEVELOP2 is a second, separate release build of a "
        f"second checkout of develop, tree `{", ".join(trees[REPEAT])}`, for the A/A band. The runner "
        "compared checkout and build trees against the expected values before the first "
        "run and after the last. Commit ids were not compared.",
        "",
        "## Protocol",
        "",
        "Each arm is a fresh release build (`build_arm.sh`, Rust 1.88.0) of `harness/`, which "
        "compiles the arm's own campaign drivers (`h2h_gmin_proof`, `h2h_sc`, `h2h_metal`, "
        "`h2h_morse`) from that arm's `examples/`. One wrapper is added around each driver's "
        "single `search(...)` call. It prints the best energy's bits and a SHA-256 of the "
        "outcome (`trace`). The runner checks that the driver's own row agrees with it on "
        "hit, charged count and the 8-decimal best. The harness lock is the arm's "
        "`Cargo.lock`, and every resolved package is pinned there. Sutton-Chen calls rgpot "
        "3.7.0's `rgpot_sutton_chen_energy_forces`, the same library for all arms.",
        "",
        "Runs are `run <system> <seed>` under `taskset -c 0`, one at a time, with every "
        "`*_THREADS` variable at 1 and no `H2H_*` or `ANNEAL_*` variable. The budget is the "
        "driver's own and was checked against the campaign `PAPER_BUDGETS` "
        "(`scripts/gmin_panel/run_panel.py`, `scripts/metal_h2h/run_sc.py`). There was no "
        "warm-up and no shortened run. Arm order reverses by seed: even seeds run DEVELOP, CC "
        "(part 1) or DEVELOP, CC, DEVELOP2 (part 2), and odd seeds run the reverse. A hit is a "
        "finite best at or below the reference plus 0.001. A miss is entered at the budget. "
        "Outside per call is `outside_s / charged` per row. Medians are Hyndman-Fan type 7.",
        "",
        f"Machine id (sha256) on every row: `{machines[0]}`" + ("" if len(machines) == 1 else f" and {machines[1:]}") + ".",
        f"Harness blob (`harness/build.rs`) on every row: `{harness[0]}`.",
        "",
        "## Rules, declared before the first run",
        "",
        "Part 1 passes when best_bits, calls, charged and trace digest are identical for "
        "DEVELOP and CC on every cell and seed.",
        "",
        "A part 2 cell fails if any of these holds:",
        "",
        "- CC has fewer hits than DEVELOP. Any directional loss fails; exact Fisher one-sided p is reported.",
        f"- CC ranks worse on calls at one-sided Mann-Whitney p < {P_LIMIT}. Misses are ranked worst.",
        f"- The CC/DEVELOP ratio of median outside per call is above {COST_LIMIT} and above the "
        "A/A spread. The spread is the larger of DEVELOP2/DEVELOP and its reciprocal.",
        "",
        "Overall PASS needs part 1 to pass and no part 2 cell to fail.",
        "",
        "## Cells",
        "",
        SPEC_LENGTH,
        "",
        "## Part 1: bit identity at length 1",
        "",
        f"LJ38, LJ55 and LJ75 at budgets {budgets.get('lj38')}, {budgets.get('lj55')} and "
        f"{budgets.get('lj75')}, seeds 2000-2002.",
        "",
        *p1_lines,
        "",
        f"Part 1: {'PASS' if not p1_fail else 'FAIL'}.",
        "",
        "## Part 2: non-unit-length cells, n=12",
        "",
        "| Cell | Budget | Hits D | Hits CC | Hits D2 | Fisher p (loss) | MW p calls (loss) | MW p calls (gain) | Median calls D | Median calls CC | Fail |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in cells:
        hb, hc, hr = c["hits"]
        text.append(
            f"| {c['system']} | {budgets[c['system']]} | {hb}/{c['n']} | {hc}/{c['n']} | {hr}/{c['n']} | "
            f"{c['fisher_loss']:.4f} | {c['mw_loss']:.4f} | {c['mw_gain']:.4f} | "
            f"{c['calls_median'][0]:.1f} | {c['calls_median'][1]:.1f} | {', '.join(c['fails']) or 'no'} |"
        )
    text += [
        "",
        "Outside time per charged call (s), median of the per-row values:",
        "",
        "| Cell | D | CC | D2 | CC/D | D2/D (A/A) | A/A spread | Cost fail |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in cells:
        b, k, r = c["cost"]
        text.append(
            f"| {c['system']} | {b:.6e} | {k:.6e} | {r:.6e} | {c['ratio']:.4f} | {c['aa']:.4f} | "
            f"{c['spread']:.4f} | {'yes' if 'cost' in c['fails'] else 'no'} |"
        )
    text += [
        "",
        "Per seed, CC against DEVELOP (and DEVELOP2 against DEVELOP), counted over 12 seeds:",
        "",
        "| Cell | best_bits same | charged same | trace same | A/A best_bits same | A/A charged same | Seeds that differ |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in cells:
        differ = [str(s) for s, bits, charged in c["per_seed"] if not (bits and charged)]
        text.append(
            f"| {c['system']} | {c['bits_same']}/{c['n']} | {c['charged_same']}/{c['n']} | "
            f"{c['trace_same']}/{c['n']} | {c['aa_bits_same']}/{c['n']} | {c['aa_charged_same']}/{c['n']} | "
            f"{' '.join(differ) or 'none'} |"
        )
    text += [
        "",
        "Gains (CC more hits, calls better at one-sided p < 0.05, or cost below 1/1.10 and "
        f"outside the A/A spread): {', '.join(p2_gain) or 'none'}.",
        "",
        f"Part 2: {'PASS' if not p2_fail else 'FAIL'}.",
        "",
        "## Special casing",
        "",
        SPECIAL_CASING,
        "",
        "## Result",
        "",
        f"{verdict}."
        + (" Failing cases: " + "; ".join(p1_fail + p2_fail) + "." if verdict == "FAIL" else " No failing case."),
        "",
        "## Files",
        "",
        "- `rows.csv`: one row per run, in run order (`slot`).",
        "- `provenance.csv`: binary hashes per arm, the rgpot library and the toolchain.",
        "- `verdict.py`: writes this file from `rows.csv` and `provenance.csv`.",
        "",
        "Binaries:",
        "",
        "| Arm | File | sha256 |",
        "| --- | --- | --- |",
    ]
    for p in provenance:
        text.append(f"| {p['arm']} | {p['file']} | `{p['sha256']}` |")
    out.write_text("\n".join(text) + "\n")
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
