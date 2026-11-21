#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Turn gate rows into summary.csv and REPORT.md.

The tests are the ones fixed in README.md before the runs.
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELLS = (
    "lj38",
    "lj55",
    "lj75",
    "lj98",
    "lj102",
    "lj103",
    "lj104",
    "lj150",
    "au55",
    "m80r6",
    "m80r14",
)
REFERENCES = {
    "lj38": -173.928427,
    "lj55": -279.248470,
    "lj75": -397.492331,
    "lj98": -543.665361,
    "lj102": -569.363652,
    "lj103": -575.766131,
    "lj104": -582.086642,
    "lj150": -893.310258,
    "au55": -191.03085280,
    "m80r6": -378.333471,
    "m80r14": -340.811371,
}
SEEDS = tuple(range(2000, 2006))
ARMS = ("A", "B", "A2")
BOOTSTRAP_DRAWS = 10_000
OUTSIDE_LIMIT = 1.10
HIT_TOL = 1.0e-3

SUMMARY_FIELDS = [
    "cell",
    "status",
    "replacements",
    "hits_a",
    "hits_b",
    "fisher_p",
    "mwu_p",
    "hits_verdict",
    "outside_ba",
    "outside_ba_lo",
    "outside_ba_hi",
    "outside_a2a_lo",
    "outside_a2a_hi",
    "time_verdict",
    "bit_seeds",
    "bit_verdict",
    "nearest_index_calls_min_b",
    "nearest_index_calls_max_b",
    "path_verdict",
    "rss_delta_kb",
    "rss_ba",
    "rss_ba_lo",
    "rss_ba_hi",
    "rss_a2a_lo",
    "rss_a2a_hi",
    "rss_verdict",
    "ru_maxrss_ba",
    "vm_hwm_ba",
]


def type7(values: list[float], p: float = 0.5) -> float:
    if not values:
        raise ValueError("type7 requires at least one value")
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lo = int(math.floor(h))
    frac = h - lo
    idx = lo - 1
    if frac == 0.0 or idx + 1 >= n:
        return ordered[idx]
    return ordered[idx] * (1.0 - frac) + ordered[idx + 1] * frac


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    miss_a = n_a - hits_a
    miss_b = n_b - hits_b
    del miss_a, miss_b
    row1 = hits_a + hits_b
    total = n_a + n_b
    lo = max(0, row1 - n_b)
    hi = min(n_a, row1)
    denom = _log_comb(total, row1)

    def prob(k: int) -> float:
        return math.exp(_log_comb(n_a, k) + _log_comb(n_b, row1 - k) - denom)

    p_obs = prob(hits_a)
    cutoff = p_obs * (1.0 + 1.0e-12)
    p_val = sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= cutoff)
    return min(1.0, p_val)


def mann_whitney_two_sided(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n1 = len(xs)
    n2 = len(ys)
    if n1 == 0 or n2 == 0:
        raise ValueError("mann whitney requires non-empty samples")
    combined = [(val, 0) for val in xs] + [(val, 1) for val in ys]
    combined.sort(key=lambda item: item[0])
    total = n1 + n2
    rank_sum_x = 0.0
    tie_sum = 0.0
    i = 0
    while i < total:
        j = i + 1
        while j < total and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        count = j - i
        if count > 1:
            tie_sum += count * (count * count - 1)
        for k in range(i, j):
            if combined[k][1] == 0:
                rank_sum_x += avg_rank
        i = j
    u1 = rank_sum_x - n1 * (n1 + 1) / 2.0
    mean_u = n1 * n2 / 2.0
    var_u = (n1 * n2 / 12.0) * ((total + 1.0) - tie_sum / (total * (total - 1.0)))
    if var_u <= 0.0:
        return u1, 1.0
    z = max(0.0, abs(u1 - mean_u) - 0.5) / math.sqrt(var_u)
    p = math.erfc(z / math.sqrt(2.0))
    return u1, min(1.0, p)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float, digits: int = 6) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}g}"


def bootstrap_ratio(
    numerator: list[float],
    denominator: list[float],
    rng: random.Random,
) -> tuple[float, float, float]:
    point = type7(numerator) / type7(denominator)
    draws = []
    n1 = len(numerator)
    n2 = len(denominator)
    for _ in range(BOOTSTRAP_DRAWS):
        left = [numerator[rng.randrange(n1)] for _ in range(n1)]
        right = [denominator[rng.randrange(n2)] for _ in range(n2)]
        base = type7(right)
        draws.append(type7(left) / base if base != 0.0 else math.inf)
    return point, type7(draws, 0.025), type7(draws, 0.975)


def usable(row: dict[str, str]) -> bool:
    return row.get("exit") == "0" and row.get("parse_ok") == "1"


def hit_of(row: dict[str, str]) -> int:
    best = float(row["best"])
    reference = REFERENCES[row["cell"]]
    computed = int(math.isfinite(best) and best <= reference + HIT_TOL)
    recorded = int(row["hit"])
    if computed != recorded:
        raise ValueError(
            f"hit flag mismatch on {row['cell']} {row['arm']} seed {row['seed']} round {row['round']}"
        )
    return computed


def calls_for_test(row: dict[str, str], hit: int) -> float:
    if hit == 0:
        return math.inf
    return float(row["calls"])


def outside_per_call(row: dict[str, str]) -> float:
    charged = float(row["charged"])
    if charged <= 0.0:
        raise ValueError(f"non-positive charged count on {row['cell']} {row['arm']}")
    return float(row["outside_s"]) / charged


def identity_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["final_bits"], row["calls"], row["trajectory_sha256"])


def journal_notes(path: Path) -> list[str]:
    if not path.is_file():
        return ["The journal is missing."]
    restarts = []
    abandoned = []
    starts = set()
    ends = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        item = {}
        for part in line.split():
            if "=" in part:
                key, value = part.split("=", 1)
                item[key] = value
        event = item.get("event", "")
        if event == "orchestrator_restart":
            restarts.append(item.get("ts", "unknown"))
        elif event == "start" and item.get("run"):
            starts.add(item["run"])
        elif event == "end" and item.get("run"):
            ends.add(item["run"])
    abandoned = sorted(starts - ends)
    lines = []
    if restarts:
        lines.append(
            "The orchestrator was restarted at " + ", ".join(restarts) + "."
        )
    else:
        lines.append("The orchestrator was not restarted.")
    if abandoned:
        shown = ", ".join(abandoned[:12])
        extra = "" if len(abandoned) <= 12 else f" and {len(abandoned) - 12} more"
        lines.append(
            f"Runs with a start line and no end line: {shown}{extra}. "
            "They were not launched again."
        )
    else:
        lines.append("Every started run has an end line.")
    return lines


def rollup(verdicts: list[str]) -> str:
    if any(item == "fail" for item in verdicts):
        return "FAIL"
    if any(item == "inconclusive" for item in verdicts):
        return "INCONCLUSIVE"
    if verdicts and all(item == "pass" for item in verdicts):
        return "PASS"
    return "INCONCLUSIVE"


def analyse_cell(
    cell: str,
    index: int,
    grouped: dict[tuple[str, str], list[dict[str, str]]],
) -> dict[str, str]:
    all_rows = []
    for arm in ARMS:
        for seed in SEEDS:
            all_rows.extend(grouped.get((arm, str(seed)), []))
    slots: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        slots[(row["arm"], row["seed"], row["round"])].append(row)
    present = []
    for rows in slots.values():
        rows.sort(key=lambda row: int(row.get("attempt", "0") or "0"))
        present.append(rows[-1])
    expected = 8 * len(SEEDS)
    # Four A rounds, two B rounds, two A2 rounds, six seeds.
    bad_exit = [row for row in present if row.get("exit") != "0" or row.get("parse_ok") != "1"]
    missing_path = any(
        row.get("arm") == "B"
        and row.get("exit") == "0"
        and not row.get("nearest_index_calls")
        for row in present
    )
    missing = expected - len(slots)
    replacements = sum(
        1 for row in all_rows if int(row.get("attempt", "0") or "0") > 0
    )
    bad_charge = False
    for row in present:
        if row.get("exit") != "0" or row.get("parse_ok") != "1":
            continue
        try:
            if float(row["charged"]) <= 0.0:
                bad_charge = True
        except (TypeError, ValueError):
            bad_charge = True
    if missing or bad_exit or bad_charge or replacements > 2:
        return {
            "cell": cell,
            "status": "inconclusive",
            "replacements": str(replacements),
            "hits_a": "",
            "hits_b": "",
            "fisher_p": "",
            "mwu_p": "",
            "hits_verdict": "inconclusive",
            "outside_ba": "",
            "outside_ba_lo": "",
            "outside_ba_hi": "",
            "outside_a2a_lo": "",
            "outside_a2a_hi": "",
            "time_verdict": "inconclusive",
            "bit_seeds": "",
            "bit_verdict": "inconclusive",
            "nearest_index_calls_min_b": "",
            "nearest_index_calls_max_b": "",
            "path_verdict": "fail" if missing_path else "inconclusive",
            "rss_delta_kb": "",
            "rss_ba": "",
            "rss_ba_lo": "",
            "rss_ba_hi": "",
            "rss_a2a_lo": "",
            "rss_a2a_hi": "",
            "rss_verdict": "inconclusive",
            "ru_maxrss_ba": "",
            "vm_hwm_ba": "",
        }

    by_arm_seed: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in present:
        by_arm_seed[(row["arm"], row["seed"])].append(row)

    hits = {"A": 0, "B": 0}
    call_samples = {"A": [], "B": []}
    hits_verdict = "pass"
    for seed in SEEDS:
        for arm in ("A", "B"):
            rows = by_arm_seed[(arm, str(seed))]
            hits_here = {hit_of(row) for row in rows}
            calls_here = {calls_for_test(row, hit_of(row)) for row in rows}
            if len(hits_here) != 1 or len(calls_here) != 1:
                hits_verdict = "inconclusive"
            elif arm in hits:
                hit = next(iter(hits_here))
                hits[arm] += hit
                call_samples[arm].append(next(iter(calls_here)))
    fisher_p = ""
    mwu_p = ""
    if hits_verdict != "inconclusive":
        fisher_p = fisher_exact_two_sided(hits["A"], len(SEEDS), hits["B"], len(SEEDS))
        u_stat, mwu = mann_whitney_two_sided(call_samples["B"], call_samples["A"])
        mean_u = len(SEEDS) * len(SEEDS) / 2.0
        mwu_p = mwu
        if hits["B"] < hits["A"]:
            hits_verdict = "fail"
        if mwu < 0.05 and u_stat > mean_u:
            hits_verdict = "fail"
        fisher_p = f"{fisher_p:.6g}"
        mwu_p = f"{mwu_p:.6g}"

    def samples(arm: str, fn) -> list[float]:
        out = []
        for seed in SEEDS:
            for row in by_arm_seed[(arm, str(seed))]:
                out.append(fn(row))
        return out

    rng = random.Random(2000 + index)
    outside = {arm: samples(arm, outside_per_call) for arm in ARMS}
    outside_point, outside_lo, outside_hi = bootstrap_ratio(outside["B"], outside["A"], rng)
    _, a2_lo, a2_hi = bootstrap_ratio(outside["A2"], outside["A"], rng)
    time_verdict = "pass"
    if outside_point > OUTSIDE_LIMIT and outside_lo > a2_hi:
        time_verdict = "fail"

    bit_ok = 0
    for seed in SEEDS:
        a_keys = {identity_key(row) for row in by_arm_seed[("A", str(seed))]}
        b_keys = {identity_key(row) for row in by_arm_seed[("B", str(seed))]}
        if len(a_keys) == 1 and a_keys == b_keys:
            bit_ok += 1
    bit_verdict = "pass" if bit_ok == len(SEEDS) else "fail"

    path_counts = samples("B", lambda row: float(row["nearest_index_calls"]))
    path_min = min(path_counts)
    path_max = max(path_counts)
    path_verdict = "pass" if path_min > 0 else "fail"

    rss = {
        arm: samples(arm, lambda row: float(row["rss_anon_peak_kb"]))
        for arm in ARMS
    }
    rss_point, rss_lo, rss_hi = bootstrap_ratio(rss["B"], rss["A"], rng)
    _, rss_a2_lo, rss_a2_hi = bootstrap_ratio(rss["A2"], rss["A"], rng)
    rss_delta = type7(rss["B"]) - type7(rss["A"])
    rss_verdict = "fail" if rss_lo > rss_a2_hi else "pass"

    def ratio_of(column: str) -> float:
        num = samples("B", lambda row, column=column: float(row[column]))
        den = samples("A", lambda row, column=column: float(row[column]))
        return type7(num) / type7(den)

    return {
        "cell": cell,
        "status": "ok",
        "replacements": str(replacements),
        "hits_a": str(hits["A"]) if hits_verdict != "inconclusive" else "",
        "hits_b": str(hits["B"]) if hits_verdict != "inconclusive" else "",
        "fisher_p": fisher_p,
        "mwu_p": mwu_p,
        "hits_verdict": hits_verdict,
        "outside_ba": fmt(outside_point, 4),
        "outside_ba_lo": fmt(outside_lo, 4),
        "outside_ba_hi": fmt(outside_hi, 4),
        "outside_a2a_lo": fmt(a2_lo, 4),
        "outside_a2a_hi": fmt(a2_hi, 4),
        "time_verdict": time_verdict,
        "bit_seeds": f"{bit_ok}/{len(SEEDS)}",
        "bit_verdict": bit_verdict,
        "nearest_index_calls_min_b": fmt(path_min, 8),
        "nearest_index_calls_max_b": fmt(path_max, 8),
        "path_verdict": path_verdict,
        "rss_delta_kb": fmt(rss_delta, 8),
        "rss_ba": fmt(rss_point, 4),
        "rss_ba_lo": fmt(rss_lo, 4),
        "rss_ba_hi": fmt(rss_hi, 4),
        "rss_a2a_lo": fmt(rss_a2_lo, 4),
        "rss_a2a_hi": fmt(rss_a2_hi, 4),
        "rss_verdict": rss_verdict,
        "ru_maxrss_ba": fmt(ratio_of("ru_maxrss_kb"), 4),
        "vm_hwm_ba": fmt(ratio_of("vm_hwm_kb"), 4),
    }


def write_report(rows_path: Path, journal_path: Path) -> None:
    rows = load_rows(rows_path)
    grouped: dict[str, dict[tuple[str, str], list[dict[str, str]]]] = {
        cell: defaultdict(list) for cell in CELLS
    }
    seen = set()
    for row in rows:
        cell = row["cell"]
        if cell not in grouped:
            raise ValueError(f"unexpected cell {cell}")
        key = (row["arm"], row["seed"], row["round"], row.get("attempt", "0"))
        if (cell, key) in seen:
            raise ValueError(f"duplicate row for {cell} {key}")
        seen.add((cell, key))
        grouped[cell][(row["arm"], row["seed"])].append(row)

    summary = [
        analyse_cell(cell, index, grouped[cell]) for index, cell in enumerate(CELLS)
    ]
    summary_path = HERE / "summary.csv"
    report_path = HERE / "REPORT.md"
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
        handle.flush()

    bars = {
        "hits": rollup([row["hits_verdict"] for row in summary]),
        "time": rollup([row["time_verdict"] for row in summary]),
        "bits": rollup([row["bit_verdict"] for row in summary]),
        "path": rollup([row["path_verdict"] for row in summary]),
        "rss": rollup([row["rss_verdict"] for row in summary]),
    }
    overall = rollup(
        [
            row[key]
            for row in summary
            for key in (
                "hits_verdict",
                "time_verdict",
                "path_verdict",
                "rss_verdict",
            )
        ]
    )
    lines = [
        "# scan-safe6 r6 against develop",
        "",
        f"- Overall: {overall}",
        "",
        f"- Hits: {bars['hits']}",
        f"- Time outside the potential per call: {bars['time']}",
        f"- Bit identity (report only): {bars['bits']}",
        f"- Safe path exercise: {bars['path']}",
        f"- Peak RssAnon: {bars['rss']}",
        "",
        "`ru_maxrss` and `VmHWM` ratios are in the table and do not gate.",
        "",
        "| cell | repl | hits A | hits B | Fisher p | MWU p | hits | B/A outside | outside 95% | A2/A 95% | time | bits | B path calls | path | RssAnon B-A KiB | B/A RssAnon | RssAnon 95% | A2/A RssAnon | rss | ru_maxrss B/A | VmHWM B/A |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for row in summary:
        outside_ci = (
            f"{row['outside_ba_lo']} .. {row['outside_ba_hi']}"
            if row["outside_ba_lo"]
            else ""
        )
        a2_ci = (
            f"{row['outside_a2a_lo']} .. {row['outside_a2a_hi']}"
            if row["outside_a2a_lo"]
            else ""
        )
        rss_ci = (
            f"{row['rss_ba_lo']} .. {row['rss_ba_hi']}" if row["rss_ba_lo"] else ""
        )
        rss_a2 = (
            f"{row['rss_a2a_lo']} .. {row['rss_a2a_hi']}" if row["rss_a2a_lo"] else ""
        )
        lines.append(
            "| {cell} | {replacements} | {hits_a} | {hits_b} | {fisher_p} | {mwu_p} | {hits_verdict} | {outside_ba} | {outside_ci} | {a2_ci} | {time_verdict} | {bit_seeds} {bit_verdict} | {nearest_index_calls_min_b} .. {nearest_index_calls_max_b} | {path_verdict} | {rss_delta_kb} | {rss_ba} | {rss_ci} | {rss_a2} | {rss_verdict} | {ru_maxrss_ba} | {vm_hwm_ba} |".format(
                outside_ci=outside_ci,
                a2_ci=a2_ci,
                rss_ci=rss_ci,
                rss_a2=rss_a2,
                **row,
            )
        )
    lines.extend(["", "## Journal", ""])
    notes = journal_notes(journal_path)
    lines.extend(notes)
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    restart_path = HERE / "RESTARTS.md"
    restart_lines = ["# Orchestrator restarts", "", *notes, ""]
    restart_path.write_text(
        "\n".join(restart_lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    write_report(HERE / "rows.csv", HERE / "journal.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
