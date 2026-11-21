#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Scan gate on the anneal search panel.

The cells, seeds and call budgets are the paper panel in
``scripts/gmin_panel/run_panel.py`` plus the Doye-Meyer binary
Lennard-Jones structures at 38 and 55 atoms listed in
``tests/fixtures/blj-ccd/sources.csv``. Each search is the driver's
``run`` command: ``Config::recommended``, the driver's own start for
that seed, and the driver's paper budget. No budget override is set.

The four release builds waited on a shared cache lock. Searches then
ran one at a time, pinned to core 0, with every thread-count variable
at 1. Arms A, B and C are three trees. Arm A2 is a second fresh build
of A's tree. For seed index k in the panel seed list, k even runs
A, B, C, A2 and k odd runs A2, C, B, A. That is the trio A B C then
C B A, with the second A build at the other end of the round.

A hit is the driver's hit. A miss is stored at the budget. The value
compared for time is ``outside_s / charged``. Medians are Hyndman-Fan
type 7. The A/A ratio is the median of A2 over the median of A. The
A/A spread is the closed interval from ``min(r, 1/r)`` to
``max(r, 1/r)``. The high end of that interval is ``max(r, 1/r)``.
B fails a cell when its B/A median ratio is above 1.1 and above that
interval. C is clearly faster than B on a cell when the C/B median
ratio is at most 0.95 and the two-sided Mann-Whitney p on per-seed
outside time per charged call, C against B, is below 0.05. C is
clearly faster overall when that holds on at least half the cells
and the C/B ratio is never above 1.05. B passes when it fails no cell.

Best energy, calls to the hit and the charged count of A, B and C are
compared as printed. A cell whose rows differ is also scored with
Fisher's exact test on hits and the Mann-Whitney test on calls to the
hit, misses entered at the budget.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

ROW_FIELDS = [
    "cell",
    "arm",
    "seed",
    "hit",
    "best",
    "evals_to_hit",
    "charged",
    "wall_s",
    "pot_s",
    "outside_s",
    "tree",
    "machine",
    "session",
]

SUMMARY_FIELDS = [
    "cell",
    "n",
    "med_opc_A",
    "med_opc_B",
    "med_opc_C",
    "med_opc_A2",
    "med_wall_A",
    "med_wall_B",
    "med_wall_C",
    "med_wall_A2",
    "ratio_ba",
    "ratio_ca",
    "ratio_cb",
    "ratio_aa",
    "aa_spread",
    "mw_p_ba",
    "mw_p_cb",
    "b_fail",
    "c_clear",
    "identical",
    "fisher_p_ba",
    "fisher_p_ca",
    "mw_p_calls_ba",
    "mw_p_calls_ca",
]

ARMS = ("A", "B", "C", "A2")
FAIL_ABOVE = 1.1
CLEAR_AT_MOST = 0.95
CLEAR_CAP = 1.05
ALPHA = 0.05
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_DOMAIN_NUM_THREADS",
    "JULIA_NUM_THREADS",
    "TBB_NUM_THREADS",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def machine_id() -> str:
    text = Path("/etc/machine-id").read_text().strip()
    return hashlib.sha256(text.encode()).hexdigest()


def thread_env(panel) -> dict[str, str]:
    env = panel.single_thread_env()
    for key in THREAD_KEYS:
        env[key] = "1"
    for key in list(env):
        if "NUM_THREADS" in key or key.endswith("_NUM_THREADS"):
            env[key] = "1"
    return env


def panel_cells(repo: Path, panel) -> list[tuple[str, str, str]]:
    """Cell, kind, binary. Kinds are the paper panel, then binary LJ at 38 and 55."""
    cells: list[tuple[str, str, str]] = []
    for name, kind, _n, _ref, _rho, _driver, binary in panel.SYSTEMS:
        cells.append((name, kind, binary))
    sources = repo / "tests/fixtures/blj-ccd/sources.csv"
    with sources.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] != "reference":
                continue
            if int(row["n"]) not in (38, 55):
                continue
            cells.append((row["file"], "blj", "h2h_blj"))
    return cells


def arm_order(index: int) -> list[str]:
    if index % 2 == 0:
        return ["A", "B", "C", "A2"]
    return ["A2", "C", "B", "A"]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({key: row[key] for key in ROW_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def budget_of(binary: Path, cell: str, env: dict[str, str]) -> tuple[int, float]:
    proc = subprocess.run(
        [str(binary), "budget", cell],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{binary} budget {cell} exited {proc.returncode}\n{proc.stderr[-1000:]}")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"{binary} budget {cell} returned {proc.stdout!r}")
    return int(lines[0]), float(lines[1])


def parse_result(text: str, cell: str, seed: int, budget: int, reference: float, panel) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith("system,") or f",anneal,{seed}," in line]
    headers = [line for line in lines if line.startswith("system,")]
    data = [line for line in lines if not line.startswith("system,")]
    if len(headers) != 1 or len(data) != 1:
        raise RuntimeError(f"{cell} seed {seed}: expected one header and one row\n{text[-2000:]}")
    row = dict(zip(headers[0].split(","), data[0].split(",")))
    if row["system"] != cell or int(row["seed"]) != seed or int(row["budget"]) != budget:
        raise RuntimeError(f"{cell} seed {seed}: row {data[0]}")
    best = float(row["best"])
    hit = panel.is_hit(row["hit"])
    evals = int(row["evals_to_hit"])
    charged = int(row["charged"])
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    panel.check_run_counts(f"{cell} seed {seed}", hit, best, reference, evals, charged, budget, wall, pot)
    outside = float(row["outside_s"]) if "outside_s" in row else max(wall - pot, 0.0)
    if outside < 0.0 or abs(max(wall - pot, 0.0) - outside) > 2e-6:
        raise RuntimeError(f"{cell} seed {seed}: outside_s {outside} against wall {wall} pot {pot}")
    if charged < 1:
        raise RuntimeError(f"{cell} seed {seed}: charged {charged}")
    return {
        "hit": "1" if hit else "0",
        "evals_to_hit": str(evals),
        "charged": str(charged),
        "wall_s": f"{wall:.6f}",
        "best": row["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
    }


def run_one(
    binary: Path,
    cell: str,
    seed: int,
    budget: int,
    reference: float,
    log: Path,
    env: dict[str, str],
    panel,
) -> dict[str, str]:
    command = ["taskset", "-c", "0", str(binary), "run", cell, str(seed)]
    proc = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{cell} seed {seed} exited {proc.returncode}")
    return parse_result(proc.stdout, cell, seed, budget, reference, panel)


def opc(row: dict[str, str]) -> float:
    return float(row["outside_s"]) / int(row["charged"])


def fmt(value: float) -> str:
    return f"{value:.12g}"


def fails_b(ratio_ba: float, ratio_aa: float) -> bool:
    spread_hi = max(ratio_aa, 1.0 / ratio_aa)
    return ratio_ba > FAIL_ABOVE and ratio_ba > spread_hi


def clear_cell(ratio_cb: float, p_value: float) -> bool:
    return ratio_cb <= CLEAR_AT_MOST and p_value < ALPHA


def overall_c(flags: list[bool], ratios: list[float]) -> bool:
    if any(ratio > CLEAR_CAP for ratio in ratios):
        return False
    return sum(flags) * 2 >= len(flags)


def groups_of(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["cell"], row["arm"]), []).append(row)
    for group in grouped.values():
        group.sort(key=lambda item: int(item["seed"]))
    return grouped


def summarise(rows: list[dict[str, str]], cells: list[str], seeds: tuple[int, ...], panel) -> tuple[list[dict[str, str]], dict[str, str]]:
    grouped = groups_of(rows)
    summary: list[dict[str, str]] = []
    differ: list[str] = []
    for cell in cells:
        columns: dict[str, list[dict[str, str]]] = {}
        for arm in ARMS:
            group = grouped.get((cell, arm), [])
            got = [int(row["seed"]) for row in group]
            if got != list(seeds):
                raise RuntimeError(f"{cell} {arm}: seeds {got}")
            columns[arm] = group
        med_opc = {arm: panel.type7([opc(row) for row in columns[arm]], 0.5) for arm in ARMS}
        med_wall = {arm: panel.type7([float(row["wall_s"]) for row in columns[arm]], 0.5) for arm in ARMS}
        if min(med_opc.values()) <= 0.0:
            raise RuntimeError(f"{cell}: non-positive outside-per-call median")
        ratio_ba = med_opc["B"] / med_opc["A"]
        ratio_ca = med_opc["C"] / med_opc["A"]
        ratio_cb = med_opc["C"] / med_opc["B"]
        ratio_aa = med_opc["A2"] / med_opc["A"]
        spread = max(ratio_aa, 1.0 / ratio_aa)
        _u_ba, p_ba = panel.mann_whitney([opc(row) for row in columns["B"]], [opc(row) for row in columns["A"]])
        _u_cb, p_cb = panel.mann_whitney([opc(row) for row in columns["C"]], [opc(row) for row in columns["B"]])
        same = True
        for seed in seeds:
            by_arm = {arm: next(row for row in columns[arm] if int(row["seed"]) == seed) for arm in ("A", "B", "C")}
            keys = ("best", "evals_to_hit", "charged")
            if any(by_arm["A"][key] != by_arm["B"][key] or by_arm["A"][key] != by_arm["C"][key] for key in keys):
                same = False
                differ.append(
                    f"{cell} {seed} best {by_arm['A']['best']} {by_arm['B']['best']} {by_arm['C']['best']} "
                    f"evals {by_arm['A']['evals_to_hit']} {by_arm['B']['evals_to_hit']} {by_arm['C']['evals_to_hit']} "
                    f"charged {by_arm['A']['charged']} {by_arm['B']['charged']} {by_arm['C']['charged']}"
                )
        fisher_ba = fisher_ca = mw_calls_ba = mw_calls_ca = ""
        if not same:
            def hits(arm: str) -> int:
                return sum(row["hit"] == "1" for row in columns[arm])

            def calls(arm: str) -> list[float]:
                return [float(row["evals_to_hit"]) for row in columns[arm]]

            n = len(seeds)
            fisher_ba = fmt(panel.fisher(hits("B"), n, hits("A"), n))
            fisher_ca = fmt(panel.fisher(hits("C"), n, hits("A"), n))
            mw_calls_ba = fmt(panel.mann_whitney(calls("B"), calls("A"))[1])
            mw_calls_ca = fmt(panel.mann_whitney(calls("C"), calls("A"))[1])
        summary.append(
            {
                "cell": cell,
                "n": str(len(seeds)),
                "med_opc_A": fmt(med_opc["A"]),
                "med_opc_B": fmt(med_opc["B"]),
                "med_opc_C": fmt(med_opc["C"]),
                "med_opc_A2": fmt(med_opc["A2"]),
                "med_wall_A": fmt(med_wall["A"]),
                "med_wall_B": fmt(med_wall["B"]),
                "med_wall_C": fmt(med_wall["C"]),
                "med_wall_A2": fmt(med_wall["A2"]),
                "ratio_ba": fmt(ratio_ba),
                "ratio_ca": fmt(ratio_ca),
                "ratio_cb": fmt(ratio_cb),
                "ratio_aa": fmt(ratio_aa),
                "aa_spread": fmt(spread),
                "mw_p_ba": fmt(p_ba),
                "mw_p_cb": fmt(p_cb),
                "b_fail": "1" if fails_b(ratio_ba, ratio_aa) else "0",
                "c_clear": "1" if clear_cell(ratio_cb, p_cb) else "0",
                "identical": "1" if same else "0",
                "fisher_p_ba": fisher_ba,
                "fisher_p_ca": fisher_ca,
                "mw_p_calls_ba": mw_calls_ba,
                "mw_p_calls_ca": mw_calls_ca,
            }
        )
    flags = [row["c_clear"] == "1" for row in summary]
    ratios = [float(row["ratio_cb"]) for row in summary]
    failed = [row["cell"] for row in summary if row["b_fail"] == "1"]
    verdict = {
        "b": "FIX" if failed else "PASS",
        "c": "yes" if overall_c(flags, ratios) else "no",
        "failed": " ".join(failed),
        "clear_cells": str(sum(flags)),
        "n_cells": str(len(summary)),
        "differ": str(len(differ)),
    }
    return summary, {"verdict": verdict, "differ_lines": differ}


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_readme(
    path: Path,
    summary: list[dict[str, str]],
    meta: dict,
    trees: dict[str, str],
    session: str,
    machine: str,
    seeds: tuple[int, ...],
) -> None:
    verdict = meta["verdict"]
    lines = [
        "# Scan gate",
        "",
        f"Session {session}. Machine {machine}.",
        f"Arm A tree `{trees['A']}`. Arm B tree `{trees['B']}`. Arm C tree `{trees['C']}`.",
        "Arm A2 is a second release build of arm A's tree.",
        f"Seeds {seeds[0]}-{seeds[-1]}, n={len(seeds)} on every cell and arm.",
        "Even seed indexes run A, B, C, A2. Odd seed indexes run A2, C, B, A.",
        "",
        "## Verdict",
        "",
        f"B: {verdict['b']}",
        f"C clearly faster than B: {verdict['c']}",
        "",
        "B fails a cell when the ratio of median outside seconds per charged call, B over A,",
        "is above 1.1 and outside the A/A spread. The A/A spread is the closed interval",
        "between the A2/A median ratio and its reciprocal. The aa_hi column is the upper",
        "end of that interval. C is clearly faster than B when C/B is at most 0.95 and the",
        "Mann-Whitney test on per-seed outside time per charged call, C against B, has",
        "p below 0.05 on at least half the cells, and C/B is never above 1.05.",
        "",
    ]
    if verdict["failed"]:
        lines.append(f"Cells B fails: {verdict['failed']}")
    else:
        lines.append("B fails no cell.")
    lines.append(
        f"Cells with C clearly faster than B: {verdict['clear_cells']} of {verdict['n_cells']}."
    )
    lines.append("")
    lines.append("## Cells")
    lines.append("")
    lines.append(
        "cell med_opc_A med_opc_B med_opc_C med_opc_A2 med_wall_A med_wall_B med_wall_C med_wall_A2 "
        "B/A C/A C/B A/A aa_hi mw_p_BA mw_p_CB b_fail c_clear identical"
    )
    for row in summary:
        lines.append(
            " ".join(
                [
                    row["cell"],
                    row["med_opc_A"],
                    row["med_opc_B"],
                    row["med_opc_C"],
                    row["med_opc_A2"],
                    row["med_wall_A"],
                    row["med_wall_B"],
                    row["med_wall_C"],
                    row["med_wall_A2"],
                    row["ratio_ba"],
                    row["ratio_ca"],
                    row["ratio_cb"],
                    row["ratio_aa"],
                    row["aa_spread"],
                    row["mw_p_ba"],
                    row["mw_p_cb"],
                    row["b_fail"],
                    row["c_clear"],
                    row["identical"],
                ]
            )
        )
    lines.append("")
    lines.append("## Identity")
    lines.append("")
    if meta["differ_lines"]:
        lines.append("Best energy, calls to the hit and charged count of A, B and C differ on:")
        lines.append("")
        lines.extend(meta["differ_lines"])
        lines.append("")
        lines.append("Fisher p is the two-sided exact test on hits, B against A and C against A.")
        lines.append("Mann-Whitney p is the two-sided test on calls to the hit, misses at the budget.")
        lines.append("")
        lines.append("cell fisher_p_BA fisher_p_CA mw_p_calls_BA mw_p_calls_CA")
        for row in summary:
            if row["identical"] == "1":
                continue
            lines.append(
                f"{row['cell']} {row['fisher_p_ba']} {row['fisher_p_ca']} {row['mw_p_calls_ba']} {row['mw_p_calls_ca']}"
            )
    else:
        lines.append("A, B and C report the same best energy, calls to the hit and charged count on every seed.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "examples/h2h_blj.rs (patch 1) was overlaid on every arm tree to build the",
            "h2h_blj driver for the binary Lennard-Jones cells (same file for all arms).",
            "",
            "Where a driver logs head and tree, those fields come from the working",
            "directory, not from the arm.",
            "",
            "The four release builds waited on a shared cache lock.",
            "",
            "The run has 17 idle gaps of about 39 minutes while the machine was paused.",
            "Dropping the jobs that straddle those gaps leaves every verdict unchanged.",
            "",
            "The harness was restarted once. lj75 A2 seed 2003 was killed and rerun on",
            "resume. No row was duplicated. Its log was overwritten.",
            "",
            "This gate covers PR #179's first commit 838390775c91cfa717c5bc848e3ee6dddd9f3837,",
            "not its later head.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def expected_budget(cell: str, kind: str, atoms: dict[str, int], panel) -> int:
    if kind == "blj":
        n = atoms[cell]
        if n == 38:
            return panel.PAPER_BUDGETS["lj38"]
        if n == 55:
            return panel.PAPER_BUDGETS["lj55"]
        raise RuntimeError(f"{cell}: size {n} is not a paper size on this panel")
    budget = panel.PAPER_BUDGETS[cell]
    return budget


def preflight(bins: dict[str, Path], cells: list[tuple[str, str, str]], atoms: dict[str, int], env: dict[str, str], panel) -> dict[str, tuple[int, float]]:
    checked: dict[str, tuple[int, float]] = {}
    for cell, kind, binary_name in cells:
        want = expected_budget(cell, kind, atoms, panel)
        found: list[tuple[int, float]] = []
        for arm in ARMS:
            binary = bins[arm] / resolve_binary(panel, bins[arm], cell, kind, binary_name)
            got = budget_of(binary, cell, env)
            found.append(got)
        if any(item != found[0] for item in found):
            raise RuntimeError(f"{cell}: arms disagree on budget {found}")
        if found[0][0] != want:
            raise RuntimeError(f"{cell}: budget {found[0][0]} is not {want}")
        if kind != "blj":
            reference = panel.spec(cell)[3]
            if abs(found[0][1] - reference) > 1e-9:
                raise RuntimeError(f"{cell}: driver reference {found[0][1]} against {reference}")
        checked[cell] = found[0]
    return checked


def resolve_binary(panel, bin_dir: Path, cell: str, kind: str, binary_name: str) -> str:
    if kind == "blj":
        return binary_name
    _driver, binary = panel.resolve_driver(bin_dir, bin_dir, cell)
    return binary


def command_run(args: argparse.Namespace, panel) -> int:
    repo = args.repo.resolve()
    cells = panel_cells(repo, panel)
    seeds = tuple(panel.SEEDS)
    if len(seeds) != 12:
        raise SystemExit(f"panel seeds are {seeds}, n=12 is required")
    atoms = {name: n for name, _kind, n, *_rest in panel.SYSTEMS}
    sources = repo / "tests/fixtures/blj-ccd/sources.csv"
    with sources.open(newline="") as handle:
        for row in csv.DictReader(handle):
            atoms[row["file"]] = int(row["n"])
    bins = {"A": args.bin_a, "B": args.bin_b, "C": args.bin_c, "A2": args.bin_a2}
    trees = {"A": args.tree_a, "B": args.tree_b, "C": args.tree_c, "A2": args.tree_a}
    env = thread_env(panel)
    checked = preflight(bins, cells, atoms, env, panel)
    machine = machine_id()
    session = args.session
    done = {(row["cell"], row["arm"], row["seed"], row["session"]) for row in read_rows(args.rows)}
    names = [cell for cell, _kind, _binary in cells]
    print(
        f"session {session} machine {machine} cells {len(names)} seeds {len(seeds)}",
        flush=True,
    )
    for cell, kind, binary_name in cells:
        budget, reference = checked[cell]
        for index, seed in enumerate(seeds):
            for arm in arm_order(index):
                key = (cell, arm, str(seed), session)
                if key in done:
                    print(f"skip {cell} {arm} {seed}", flush=True)
                    continue
                binary = bins[arm] / resolve_binary(panel, bins[arm], cell, kind, binary_name)
                print(f"start {cell} {arm} {seed}", flush=True)
                started = time.monotonic()
                result = run_one(
                    binary,
                    cell,
                    seed,
                    budget,
                    reference,
                    args.logs / cell / f"{arm}-{seed}.log",
                    env,
                    panel,
                )
                row = {
                    "cell": cell,
                    "arm": arm,
                    "seed": str(seed),
                    **result,
                    "tree": trees[arm],
                    "machine": machine,
                    "session": session,
                }
                append_row(args.rows, row)
                done.add(key)
                elapsed = time.monotonic() - started
                print(
                    f"done {cell} {arm} {seed} hit={row['hit']} evals={row['evals_to_hit']} "
                    f"charged={row['charged']} wall_s={row['wall_s']} outside_s={row['outside_s']} "
                    f"elapsed={elapsed:.1f}",
                    flush=True,
                )
    rows = read_rows(args.rows)
    rows = [row for row in rows if row["session"] == session]
    summary, meta = summarise(rows, names, seeds, panel)
    write_csv(args.summary, SUMMARY_FIELDS, summary)
    write_readme(args.readme, summary, meta, trees, session, machine, seeds)
    print(f"B {meta['verdict']['b']}", flush=True)
    print(f"C clearly-faster {meta['verdict']['c']}", flush=True)
    return 0


def command_summarize(args: argparse.Namespace, panel) -> int:
    repo = args.repo.resolve()
    cells = [cell for cell, _kind, _binary in panel_cells(repo, panel)]
    seeds = tuple(panel.SEEDS)
    rows = [row for row in read_rows(args.rows) if row["session"] == args.session]
    summary, meta = summarise(rows, cells, seeds, panel)
    trees = {"A": args.tree_a, "B": args.tree_b, "C": args.tree_c}
    machine = rows[0]["machine"] if rows else ""
    write_csv(args.summary, SUMMARY_FIELDS, summary)
    write_readme(args.readme, summary, meta, trees, args.session, machine, seeds)
    print(f"B {meta['verdict']['b']}", flush=True)
    print(f"C clearly-faster {meta['verdict']['c']}", flush=True)
    return 0


def self_test(panel) -> None:
    if arm_order(0) != ["A", "B", "C", "A2"] or arm_order(1) != ["A2", "C", "B", "A"]:
        raise SystemExit("arm order")
    if not fails_b(1.2, 1.0):
        raise SystemExit("1.2 against a flat A/A band should fail")
    if fails_b(1.12, 1.15):
        raise SystemExit("1.12 sits inside an A/A ratio of 1.15")
    if fails_b(1.1, 1.0):
        raise SystemExit("the 1.1 boundary is not above 1.1")
    # ratio_aa 0.8 -> spread high end 1.25, so 1.15 is inside and must not fail
    if fails_b(1.15, 0.8):
        raise SystemExit("1.15 is inside the reciprocal of 0.8")
    if not fails_b(1.3, 0.8):
        raise SystemExit("1.3 is outside the reciprocal of 0.8")
    if not clear_cell(0.95, 0.049):
        raise SystemExit("0.95 at p 0.049 is a clear cell")
    if clear_cell(0.95, 0.05):
        raise SystemExit("p 0.05 is not below 0.05")
    if clear_cell(0.9500001, 0.01):
        raise SystemExit("a ratio above 0.95 is not clear")
    if not overall_c([True, True, False, False], [0.9, 0.94, 1.0, 1.05]):
        raise SystemExit("half the cells and a cap of 1.05 should pass")
    if overall_c([True, True, False, False], [0.9, 0.94, 1.0, 1.0500001]):
        raise SystemExit("a cell above 1.05 blocks a clear win")
    if overall_c([True, False, False, False], [0.9, 1.0, 1.0, 1.0]):
        raise SystemExit("one cell of four is not half")
    sample = [float(v) for v in range(1, 13)]
    if abs(panel.type7(sample, 0.5) - 6.5) > 1e-12:
        raise SystemExit("type 7 median")
    _u, tied = panel.mann_whitney([1.0] * 12, [1.0] * 12)
    if tied != 1.0:
        raise SystemExit(f"tied mann-whitney {tied}")
    print("self-test ok")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--bin-a", type=Path, required=True)
    run.add_argument("--bin-b", type=Path, required=True)
    run.add_argument("--bin-c", type=Path, required=True)
    run.add_argument("--bin-a2", type=Path, required=True)
    run.add_argument("--tree-a", required=True)
    run.add_argument("--tree-b", required=True)
    run.add_argument("--tree-c", required=True)
    run.add_argument("--rows", type=Path, required=True)
    run.add_argument("--logs", type=Path, required=True)
    run.add_argument("--summary", type=Path, required=True)
    run.add_argument("--readme", type=Path, required=True)
    run.add_argument("--session", required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--rows", type=Path, required=True)
    summary.add_argument("--summary", type=Path, required=True)
    summary.add_argument("--readme", type=Path, required=True)
    summary.add_argument("--session", required=True)
    summary.add_argument("--tree-a", required=True)
    summary.add_argument("--tree-b", required=True)
    summary.add_argument("--tree-c", required=True)
    test = sub.add_parser("self-test")
    args = parser.parse_args(argv)
    panel = load_module("gmin_panel", args.repo / "scripts/gmin_panel/run_panel.py")
    if args.cmd == "self-test":
        self_test(panel)
        return 0
    if args.cmd == "run":
        return command_run(args, panel)
    return command_summarize(args, panel)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
