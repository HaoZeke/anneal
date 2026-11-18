#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Release A/B gate for the directed-leave stall clock on Lennard-Jones clusters.

A system loses when its hit count drops and the two-sided Fisher exact
p-value is below 0.05, or when the median evaluations to a hit, taken
over hitting seeds only, rises and the two-sided exact Mann-Whitney
p-value is below 0.05. The gate passes only when no system loses.

Systems are LJ38 and every size from LJ75 through LJ104. A size that
has an entry in ``PAPER_BUDGETS`` uses that budget and the paper
12-seed set. Every other size in the span uses the single budget
``PAPER_BUDGETS`` assigns to every named Lennard-Jones size from 75
through 104, and the six declared seeds 2000 through 2005.

Each seed runs arm A and arm B back to back on one pinned core. Even
seeds run A then B. Odd seeds run B then A. Distinct seeds may occupy
distinct cores at the same time. One process is one run.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

HIT_LEVEL = "reference + 0.001"
DECLARED_SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
# Cambridge Cluster Database pair-well global minima. Named paper sizes
# are checked against examples/h2h_gmin_proof.rs before any run.
CCD = {
    38: -173.928427,
    75: -397.492331,
    76: -402.894866,
    77: -409.083517,
    78: -414.794401,
    79: -421.810897,
    80: -428.083564,
    81: -434.343643,
    82: -440.550425,
    83: -446.924094,
    84: -452.657214,
    85: -459.055799,
    86: -465.384493,
    87: -472.098165,
    88: -479.032630,
    89: -486.053911,
    90: -492.433908,
    91: -498.811060,
    92: -505.185309,
    93: -510.877688,
    94: -517.264131,
    95: -523.640211,
    96: -529.879146,
    97: -536.681383,
    98: -543.665361,
    99: -550.666526,
    100: -557.039820,
    101: -563.411308,
    102: -569.363652,
    103: -575.766131,
    104: -582.086642,
}
PANEL = (38, *range(75, 105))
LOSS_P = 0.05
SHELF_SEEDS = (2004, 2010)


def extract_braced(text: str, marker: str) -> str:
    start = text.index(marker)
    brace = text.index("{", start)
    depth = 0
    for index, char in enumerate(text[brace:], brace):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace : index + 1]
    raise RuntimeError(f"unclosed block at {marker}")


def paper_budgets(path: Path) -> dict[str, int]:
    block = extract_braced(path.read_text(), "PAPER_BUDGETS =")
    parsed = ast.literal_eval(block)
    if not isinstance(parsed, dict) or not parsed:
        raise RuntimeError(f"empty PAPER_BUDGETS in {path}")
    out = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, int):
            raise RuntimeError(f"bad PAPER_BUDGETS entry {key!r}")
        out[key] = value
    return out


def paper_seeds(path: Path) -> tuple[int, ...]:
    match = re.search(r"SEEDS = tuple\(range\((\d+), (\d+)\)\)", path.read_text())
    if match is None:
        raise RuntimeError(f"paper seed set missing in {path}")
    start, stop = int(match.group(1)), int(match.group(2))
    seeds = tuple(range(start, stop))
    if len(seeds) != 12:
        raise RuntimeError(f"paper seed set has {len(seeds)} seeds")
    return seeds


def proof_references(path: Path) -> dict[int, float]:
    text = path.read_text()
    found = {}
    for name, count, energy in re.findall(
        r'"(lj\d+)" => \((\d+), (-?\d+\.\d+)\)', text
    ):
        size = int(name[2:])
        if size != int(count):
            raise RuntimeError(f"{name} does not match atom count {count}")
        found[size] = float(energy)
    if not found:
        raise RuntimeError(f"no Lennard-Jones references in {path}")
    match = re.search(r"const HIT_TOL: f64 = ([0-9.]+);", text)
    if match is None or abs(float(match.group(1)) - 0.001) > 1e-15:
        raise RuntimeError("hit tolerance is not 0.001")
    return found


def load_protocol(repo: Path) -> dict:
    panel = repo / "scripts" / "gmin_panel" / "run_panel.py"
    other = repo / "scripts" / "paper_0120" / "run_panel.py"
    proof = repo / "examples" / "h2h_gmin_proof.rs"
    budgets = paper_budgets(panel)
    other_budgets = paper_budgets(other)
    for key, value in budgets.items():
        if key.startswith("lj") and key in other_budgets and other_budgets[key] != value:
            raise RuntimeError(f"{key} budget disagrees between panel scripts")
    seeds = paper_seeds(panel)
    if paper_seeds(other) != seeds:
        raise RuntimeError("paper seed sets disagree")
    references = proof_references(proof)
    for size, energy in references.items():
        if size in CCD and abs(CCD[size] - energy) > 1e-12:
            raise RuntimeError(f"lj{size} reference disagrees with the proof example")
    named_large = [
        budgets[f"lj{size}"]
        for size in range(75, 105)
        if f"lj{size}" in budgets
    ]
    if len(set(named_large)) != 1:
        raise RuntimeError("PAPER_BUDGETS has more than one value on LJ75 through LJ104")
    if "lj38" not in budgets:
        raise RuntimeError("PAPER_BUDGETS has no lj38 entry")
    systems = []
    for size in PANEL:
        key = f"lj{size}"
        if size not in CCD:
            raise RuntimeError(f"missing reference for {key}")
        if key in budgets:
            budget = budgets[key]
            source = "paper"
            use_seeds = seeds
        else:
            budget = named_large[0]
            source = "paper-range"
            use_seeds = DECLARED_SEEDS
        systems.append(
            {
                "system": key,
                "n": size,
                "budget": budget,
                "budget_source": source,
                "reference": CCD[size],
                "seeds": list(use_seeds),
            }
        )
    return {"systems": systems, "paper_seeds": list(seeds), "declared_seeds": list(DECLARED_SEEDS)}


def type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("empty sample")
    if count == 1:
        return float(ordered[0])
    position = 1.0 + (count - 1) * probability
    lower = math.floor(position)
    frac = position - lower
    index = int(lower) - 1
    if frac == 0.0:
        return float(ordered[index])
    return float(ordered[index] * (1.0 - frac) + ordered[index + 1] * frac)


def median_iqr(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    return type7(values, 0.5), type7(values, 0.75) - type7(values, 0.25)


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    column = hits_a + hits_b
    total = n_a + n_b
    ways = math.comb(total, column)

    def probability(hits: int) -> float:
        return math.comb(n_a, hits) * math.comb(n_b, column - hits) / ways

    observed = probability(hits_a)
    low = max(0, column - n_b)
    high = min(n_a, column)
    tail = sum(
        probability(hits)
        for hits in range(low, high + 1)
        if probability(hits) <= observed * (1.0 + 1e-12)
    )
    return min(1.0, tail)


def midranks(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    scaled = [0] * len(values)
    cursor = 0
    count = len(values)
    while cursor < count:
        end = cursor
        while end + 1 < count and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        # 1-based midrank, doubled so ties stay on integers.
        rank = cursor + end + 2
        for index in range(cursor, end + 1):
            scaled[order[index]] = rank
        cursor = end + 1
    return scaled


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float] | None:
    """Exact two-sided Mann-Whitney on the pooled midranks.

    The returned U belongs to ``left``. A larger U means ``left`` tends
    to contain the larger values. The p-value counts subsets whose
    rank sum is at least as far from the null mean as the observed sum.
    """
    if not left or not right:
        return None
    ranks = midranks(list(left) + list(right))
    n_left = len(left)
    n_right = len(right)
    observed = sum(ranks[:n_left])
    total_n = n_left + n_right
    mean = n_left * (total_n + 1)
    deviation = abs(observed - mean)
    by_count = [{0: 1}]
    for rank in ranks:
        nxt = [dict(layer) for layer in by_count]
        nxt.append({})
        for taken, layer in enumerate(by_count):
            if taken >= n_left:
                break
            dest = nxt[taken + 1]
            for size, count in layer.items():
                dest[size + rank] = dest.get(size + rank, 0) + count
        by_count = nxt[: n_left + 1]
    distribution = by_count[n_left]
    total = sum(distribution.values())
    extreme = sum(
        count for size, count in distribution.items() if abs(size - mean) + 1e-9 >= deviation
    )
    u_stat = observed / 2.0 - n_left * (n_left + 1) / 2.0
    return u_stat, extreme / total


def self_check() -> None:
    if abs(fisher(9, 12, 6, 12) - 0.40032305828509895) > 1e-12:
        raise RuntimeError("Fisher check failed")
    if abs(fisher(6, 6, 6, 6) - 1.0) > 1e-12:
        raise RuntimeError("Fisher tie check failed")
    tied = mann_whitney([1, 2, 3], [1, 2, 3])
    if tied is None or abs(tied[1] - 1.0) > 1e-12:
        raise RuntimeError("Mann-Whitney tie check failed")
    shift = mann_whitney([1, 2, 3], [4, 5, 6])
    if shift is None or abs(shift[1] - 0.1) > 1e-12:
        raise RuntimeError("Mann-Whitney shift check failed")
    if abs(type7([1, 2, 3, 4], 0.5) - 2.5) > 1e-12:
        raise RuntimeError("type 7 check failed")
    if set(CCD) != set(PANEL):
        raise RuntimeError("reference table does not cover the panel")
    if len(DECLARED_SEEDS) < 6 or len(set(DECLARED_SEEDS)) != len(DECLARED_SEEDS):
        raise RuntimeError("declared seeds")


def git_sha(repo: Path, rev: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", rev],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def build_arm(args: argparse.Namespace) -> None:
    src = Path(args.src)
    runner = Path(args.runner)
    target = Path(args.target_dir)
    out_bin = Path(args.out_bin)
    example = src / "examples" / "lj_shelf_gate.rs"
    if example.exists():
        raise RuntimeError(f"{example.name} already exists in the source tree")
    example.write_bytes(runner.read_bytes())
    try:
        subprocess.run(
            [
                args.cargo,
                f"+{args.toolchain}",
                "build",
                "--release",
                "--locked",
                "--example",
                "lj_shelf_gate",
                "-j",
                str(args.jobs),
                "--target-dir",
                str(target),
            ],
            cwd=src,
            check=True,
        )
    finally:
        example.unlink(missing_ok=True)
    built = target / "release" / "examples" / "lj_shelf_gate"
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(built, out_bin)
    out_bin.chmod(0o755)


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_row(path: Path, row: dict, lock: threading.Lock) -> None:
    text = json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())


def parse_runner(stdout: str) -> dict:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 2 or not lines[0].startswith("n,seed,budget,hit,"):
        raise RuntimeError(f"unexpected runner output: {stdout!r}")
    fields = lines[1].split(",")
    if len(fields) != 9:
        raise RuntimeError(f"unexpected runner row: {lines[1]!r}")
    hit = fields[3]
    if hit not in {"0", "1"}:
        raise RuntimeError(f"bad hit flag {hit}")
    return {
        "hit": int(hit),
        "evals_to_hit": int(fields[4]),
        "charged": int(fields[5]),
        "best": fields[6],
        "wall_s": fields[7],
        "pot_s": fields[8],
    }


def run_one(
    binary: Path,
    system: dict,
    seed: int,
    arm: str,
    order: str,
    core: int,
    revision: str,
    tree: str,
) -> dict:
    command = [
        "taskset",
        "-c",
        str(core),
        str(binary),
        str(system["n"]),
        str(seed),
        str(system["budget"]),
        f"{system['reference']:.12f}",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{system['system']} seed {seed} arm {arm} exited {proc.returncode}\n"
            f"{proc.stderr[-2000:]}"
        )
    parsed = parse_runner(proc.stdout)
    if parsed["evals_to_hit"] > system["budget"] or parsed["charged"] > system["budget"]:
        raise RuntimeError(f"{system['system']} seed {seed} arm {arm} exceeded the budget")
    return {
        "system": system["system"],
        "n": system["n"],
        "seed": seed,
        "arm": arm,
        "order": order,
        "core": core,
        "budget": system["budget"],
        "budget_source": system["budget_source"],
        "reference": f"{system['reference']:.6f}",
        "code_revision": revision,
        "tree": tree,
        **parsed,
    }


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}g}"


def fmt_num(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}"


def summarise(rows: list[dict], protocol: dict, meta: dict) -> dict:
    by_system: dict[str, list[dict]] = {}
    for row in rows:
        by_system.setdefault(row["system"], []).append(row)
    systems_out = []
    losses = []
    nonsignificant = []
    for spec in protocol["systems"]:
        name = spec["system"]
        group = by_system.get(name, [])
        arms = {"A": {}, "B": {}}
        for row in group:
            arms[row["arm"]][int(row["seed"])] = row
        seeds = list(spec["seeds"])
        if sorted(arms["A"]) != seeds or sorted(arms["B"]) != seeds:
            raise RuntimeError(f"{name} is incomplete")
        hits_a = sum(arms["A"][seed]["hit"] == 1 for seed in seeds)
        hits_b = sum(arms["B"][seed]["hit"] == 1 for seed in seeds)
        evals_a = [arms["A"][seed]["evals_to_hit"] for seed in seeds if arms["A"][seed]["hit"] == 1]
        evals_b = [arms["B"][seed]["evals_to_hit"] for seed in seeds if arms["B"][seed]["hit"] == 1]
        summary_a = median_iqr([float(value) for value in evals_a])
        summary_b = median_iqr([float(value) for value in evals_b])
        fisher_p = fisher(hits_a, len(seeds), hits_b, len(seeds))
        mw = mann_whitney([float(value) for value in evals_b], [float(value) for value in evals_a])
        median_a = None if summary_a is None else summary_a[0]
        median_b = None if summary_b is None else summary_b[0]
        iqr_a = None if summary_a is None else summary_a[1]
        iqr_b = None if summary_b is None else summary_b[1]
        hit_drop = hits_b < hits_a
        eval_increase = (
            median_a is not None and median_b is not None and median_b > median_a
        )
        mw_p = None if mw is None else mw[1]
        mw_u = None if mw is None else mw[0]
        sig_hit = hit_drop and fisher_p < LOSS_P
        sig_eval = eval_increase and mw_p is not None and mw_p < LOSS_P
        reasons = []
        if sig_hit:
            reasons.append("hit-count drop")
        if sig_eval:
            reasons.append("evals-to-hit increase")
        ns = []
        if hit_drop and not sig_hit:
            ns.append("hit-count drop")
        if eval_increase and not sig_eval:
            ns.append("evals-to-hit increase")
        paired = []
        worse = []
        for seed in seeds:
            left = arms["A"][seed]
            right = arms["B"][seed]
            lost_hit = left["hit"] == 1 and right["hit"] == 0
            more_evals = right["evals_to_hit"] > left["evals_to_hit"]
            record = {
                "seed": seed,
                "core": left["core"],
                "order": left["order"],
                "a": {
                    "hit": left["hit"],
                    "evals_to_hit": left["evals_to_hit"],
                    "charged": left["charged"],
                    "best": left["best"],
                    "wall_s": left["wall_s"],
                },
                "b": {
                    "hit": right["hit"],
                    "evals_to_hit": right["evals_to_hit"],
                    "charged": right["charged"],
                    "best": right["best"],
                    "wall_s": right["wall_s"],
                },
                "lost_hit": lost_hit,
                "more_evals": more_evals,
            }
            if left["core"] != right["core"] or left["order"] != right["order"]:
                raise RuntimeError(f"{name} seed {seed} split across cores or orders")
            paired.append(record)
            if lost_hit or more_evals:
                worse.append(record)
        entry = {
            "system": name,
            "n": spec["n"],
            "budget": spec["budget"],
            "budget_source": spec["budget_source"],
            "reference": f"{spec['reference']:.6f}",
            "seeds": seeds,
            "hits_a": hits_a,
            "hits_b": hits_b,
            "n_seeds": len(seeds),
            "fisher_p": fisher_p,
            "evals_hit_a": {"n": len(evals_a), "median": median_a, "iqr": iqr_a},
            "evals_hit_b": {"n": len(evals_b), "median": median_b, "iqr": iqr_b},
            "mannwhitney_u_b": mw_u,
            "mannwhitney_p": mw_p,
            "loss": bool(reasons),
            "loss_reasons": reasons,
            "nonsignificant": ns,
            "seeds_where_b_is_worse": [
                {
                    "seed": item["seed"],
                    "lost_hit": item["lost_hit"],
                    "more_evals": item["more_evals"],
                    "evals_a": item["a"]["evals_to_hit"],
                    "evals_b": item["b"]["evals_to_hit"],
                    "best_a": item["a"]["best"],
                    "best_b": item["b"]["best"],
                }
                for item in worse
            ],
            "per_seed": paired,
        }
        systems_out.append(entry)
        if reasons:
            losses.append({"system": name, "reasons": reasons, "fisher_p": fisher_p, "mannwhitney_p": mw_p})
        if ns:
            nonsignificant.append(
                {"system": name, "drops": ns, "fisher_p": fisher_p, "mannwhitney_p": mw_p}
            )
    lj102 = next(item for item in systems_out if item["system"] == "lj102")
    focus = {}
    for seed in SHELF_SEEDS:
        row = next(item for item in lj102["per_seed"] if item["seed"] == seed)
        focus[str(seed)] = row
    verdict = "PASS" if not losses else "FAIL"
    return {
        "rule": (
            "A loss is a hit-count drop with two-sided Fisher exact p < 0.05, "
            "or an increase in the median evaluations to a hit over hitting seeds "
            "with two-sided exact Mann-Whitney p < 0.05. "
            "PASS requires every system to avoid both."
        ),
        "verdict": verdict,
        "losses": losses,
        "nonsignificant_drops": nonsignificant,
        "compiler": meta["compiler"],
        "cpu": meta["cpu"],
        "hit_tolerance": 0.001,
        "hit_rule": HIT_LEVEL,
        "arms": meta["arms"],
        "paper_seeds": protocol["paper_seeds"],
        "declared_seeds": protocol["declared_seeds"],
        "order": "even seeds run A then B; odd seeds run B then A; both arms of a seed share one core",
        "lj102_focus_seeds": focus,
        "systems": systems_out,
    }


def markdown(report: dict) -> str:
    lines = [
        "# LJ shelf gate",
        "",
        report["rule"],
        "",
        f"Verdict: **{report['verdict']}**.",
        "",
        "## Record",
        "",
        f"- Compiler: {report['compiler']}",
        f"- CPU: {report['cpu']}",
        f"- Arm A commit `{report['arms']['A']['commit']}` tree `{report['arms']['A']['tree']}`",
        f"- Arm B commit `{report['arms']['B']['commit']}` tree `{report['arms']['B']['tree']}`",
        f"- Hit rule: quenched energy at or below {report['hit_rule']}. A miss is entered at the budget. Medians and IQRs use Hyndman-Fan type 7 on hitting seeds only.",
        f"- Paper seeds, where `PAPER_BUDGETS` names the size: {' '.join(str(seed) for seed in report['paper_seeds'])}.",
        f"- Declared seeds, for every other size from 75 through 104: {' '.join(str(seed) for seed in report['declared_seeds'])}.",
        f"- Schedule: {report['order']}.",
        "",
        "## Systems",
        "",
        "| system | budget | source | seeds | hits A | hits B | Fisher p | median evals A | IQR A | median evals B | IQR B | Mann-Whitney p | loss |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for system in report["systems"]:
        lines.append(
            "| {system} | {budget} | {source} | {n} | {ha}/{n} | {hb}/{n} | {fp} | {ma} | {ia} | {mb} | {ib} | {mp} | {loss} |".format(
                system=system["system"],
                budget=system["budget"],
                source=system["budget_source"],
                n=system["n_seeds"],
                ha=system["hits_a"],
                hb=system["hits_b"],
                fp=fmt(system["fisher_p"]),
                ma=fmt_num(system["evals_hit_a"]["median"]),
                ia=fmt_num(system["evals_hit_a"]["iqr"]),
                mb=fmt_num(system["evals_hit_b"]["median"]),
                ib=fmt_num(system["evals_hit_b"]["iqr"]),
                mp=fmt(system["mannwhitney_p"]),
                loss=", ".join(system["loss_reasons"]) if system["loss_reasons"] else "",
            )
        )
    lines.extend(["", "## Non-significant drops", ""])
    if not report["nonsignificant_drops"]:
        lines.append("No system has a non-significant hit-count drop or a non-significant rise in median evaluations to a hit.")
    else:
        lines.append("| system | drop | Fisher p | Mann-Whitney p |")
        lines.append("| --- | --- | ---: | ---: |")
        for item in report["nonsignificant_drops"]:
            lines.append(
                f"| {item['system']} | {', '.join(item['drops'])} | {fmt(item['fisher_p'])} | {fmt(item['mannwhitney_p'])} |"
            )
    lines.extend(["", "## LJ102 seeds 2004 and 2010", ""])
    lines.append("Arm A is expected to finish on the icosahedral shelf at -569.27772309 for these two seeds when the leave clock is consumed on a no-op. The global minimum used as the hit level is -569.363652.")
    lines.append("")
    lines.append("| seed | A hit | A evals | A best | B hit | B evals | B best |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for seed in SHELF_SEEDS:
        row = report["lj102_focus_seeds"][str(seed)]
        lines.append(
            f"| {seed} | {row['a']['hit']} | {row['a']['evals_to_hit']} | {row['a']['best']} | {row['b']['hit']} | {row['b']['evals_to_hit']} | {row['b']['best']} |"
        )
    lines.extend(["", "## Seeds where B loses a hit or uses more evaluations", ""])
    lines.append("| system | seed | lost hit | more evals | A evals | B evals | A best | B best |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |")
    any_worse = False
    for system in report["systems"]:
        for item in system["seeds_where_b_is_worse"]:
            any_worse = True
            lines.append(
                "| {system} | {seed} | {lost} | {more} | {ea} | {eb} | {ba} | {bb} |".format(
                    system=system["system"],
                    seed=item["seed"],
                    lost="yes" if item["lost_hit"] else "",
                    more="yes" if item["more_evals"] else "",
                    ea=item["evals_a"],
                    eb=item["evals_b"],
                    ba=item["best_a"],
                    bb=item["best_b"],
                )
            )
    if not any_worse:
        lines.append("| | | | | | | | |")
    lines.extend(["", "## Best energy per seed", ""])
    for system in report["systems"]:
        lines.append(f"### {system['system']}")
        lines.append("")
        lines.append("| seed | order | core | A hit | A evals | A best | B hit | B evals | B best |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in system["per_seed"]:
            lines.append(
                f"| {row['seed']} | {row['order']} | {row['core']} | {row['a']['hit']} | {row['a']['evals_to_hit']} | {row['a']['best']} | {row['b']['hit']} | {row['b']['evals_to_hit']} | {row['b']['best']} |"
            )
        lines.append("")
    lines.append(f"Verdict: **{report['verdict']}**.")
    lines.append("")
    return "\n".join(lines)


def print_plan(protocol: dict) -> None:
    print("Declared seeds for sizes with no paper 12-seed entry: " + " ".join(str(seed) for seed in protocol["declared_seeds"]))
    print("Paper 12-seed set: " + " ".join(str(seed) for seed in protocol["paper_seeds"]))
    print("system budget source seeds")
    for system in protocol["systems"]:
        print(
            f"{system['system']} {system['budget']} {system['budget_source']} "
            + " ".join(str(seed) for seed in system["seeds"])
        )


def cmd_build(args: argparse.Namespace) -> int:
    build_arm(args)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    self_check()
    repo = Path(args.repo)
    protocol = load_protocol(repo)
    print_plan(protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    done = {(row["system"], int(row["seed"]), row["arm"]) for row in read_rows(rows_path)}
    binaries = {"A": Path(args.binary_a), "B": Path(args.binary_b)}
    revisions = {
        "A": {"commit": git_sha(Path(args.src_a), "HEAD"), "tree": git_sha(Path(args.src_a), "HEAD^{tree}")},
        "B": {"commit": git_sha(Path(args.src_b), "HEAD"), "tree": git_sha(Path(args.src_b), "HEAD^{tree}")},
    }
    if not revisions["A"]["tree"].startswith("db5ad0cf") or not revisions["B"]["tree"].startswith("7fd673e4"):
        raise RuntimeError(f"unexpected trees {revisions}")
    cores = [int(piece) for piece in args.cores.split(",") if piece != ""]
    if not cores:
        raise RuntimeError("cores are empty")
    jobs = []
    for system in protocol["systems"]:
        for seed in system["seeds"]:
            jobs.append((system, seed))
    jobs.sort(key=lambda item: (item[0]["n"], item[1]))
    lock = threading.Lock()
    errors: list[str] = []
    queue: list = jobs[:]
    queue_lock = threading.Lock()

    def worker(core: int) -> None:
        while True:
            with queue_lock:
                if not queue:
                    return
                system, seed = queue.pop()
            order = "AB" if seed % 2 == 0 else "BA"
            sequence = ("A", "B") if order == "AB" else ("B", "A")
            for arm in sequence:
                key = (system["system"], seed, arm)
                if key in done:
                    continue
                try:
                    row = run_one(
                        binaries[arm],
                        system,
                        seed,
                        arm,
                        order,
                        core,
                        revisions[arm]["commit"],
                        revisions[arm]["tree"],
                    )
                except Exception as exc:
                    with lock:
                        errors.append(str(exc))
                    print(f"FAIL {system['system']} {seed} {arm} core {core}: {exc}", flush=True)
                    break
                append_row(rows_path, row, lock)
                print(
                    f"done {system['system']} {seed} {arm} core {core} hit {row['hit']} "
                    f"evals {row['evals_to_hit']} best {row['best']} wall {row['wall_s']}",
                    flush=True,
                )

    threads = [threading.Thread(target=worker, args=(core,)) for core in cores]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError("\n".join(errors[:8]))
    rows = read_rows(rows_path)
    report = summarise(
        rows,
        protocol,
        {"compiler": args.compiler, "cpu": args.cpu, "arms": revisions},
    )
    results_json = Path(args.results_json)
    results_md = Path(args.results_md)
    results_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_md.write_text(markdown(report), encoding="utf-8")
    print(f"verdict {report['verdict']}", flush=True)
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    self_check()
    protocol = load_protocol(Path(args.repo))
    rows = read_rows(Path(args.rows))
    revisions = {}
    for arm in ("A", "B"):
        sample = next(row for row in rows if row["arm"] == arm)
        revisions[arm] = {"commit": sample["code_revision"], "tree": sample["tree"]}
    report = summarise(
        rows,
        protocol,
        {"compiler": args.compiler, "cpu": args.cpu, "arms": revisions},
    )
    Path(args.results_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.results_md).write_text(markdown(report), encoding="utf-8")
    print(f"verdict {report['verdict']}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--src", required=True)
    build.add_argument("--runner", required=True)
    build.add_argument("--target-dir", required=True)
    build.add_argument("--out-bin", required=True)
    build.add_argument("--cargo", required=True)
    build.add_argument("--toolchain", required=True)
    build.add_argument("--jobs", required=True, type=int)
    build.set_defaults(func=cmd_build)

    run = sub.add_parser("run")
    run.add_argument("--repo", required=True)
    run.add_argument("--src-a", required=True)
    run.add_argument("--src-b", required=True)
    run.add_argument("--binary-a", required=True)
    run.add_argument("--binary-b", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--results-json", required=True)
    run.add_argument("--results-md", required=True)
    run.add_argument("--cores", required=True)
    run.add_argument("--compiler", required=True)
    run.add_argument("--cpu", required=True)
    run.set_defaults(func=cmd_run)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--repo", required=True)
    summarize.add_argument("--rows", required=True)
    summarize.add_argument("--results-json", required=True)
    summarize.add_argument("--results-md", required=True)
    summarize.add_argument("--compiler", required=True)
    summarize.add_argument("--cpu", required=True)
    summarize.set_defaults(func=cmd_summarize)
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"gate: {exc}", file=sys.stderr)
        sys.exit(1)
