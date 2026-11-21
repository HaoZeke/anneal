# Copyright (c) anneal developers
"""Statistics for the morse m38 rho-14 round-four gate.

The pass rule lives in README.md. This file applies that rule
to raw rows. It does not drop, relabel, or rerun anything.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cells as gate


def type7(values: list[float], p: float = 0.5) -> float:
    """Hyndman-Fan type 7."""
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("empty sample")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def mcnemar_p(b_count: int, c_count: int) -> float:
    """Exact two-sided McNemar probability.

    b_count is A miss and B hit. c_count is A hit and B miss.
    Under the null each discordant pair is a fair coin. The
    returned value sums the binomial probabilities that are no
    larger than the probability of the smaller count.
    """
    total = b_count + c_count
    if total == 0:
        return 1.0
    observed = math.comb(total, min(b_count, c_count))
    mass = sum(math.comb(total, k) for k in range(total + 1) if math.comb(total, k) <= observed)
    return mass / (2**total)


def ranked_charged(row: dict[str, str], budget: int) -> float:
    """Charged calls with misses ranked worst.

    A completed hit keeps its charged count. A completed miss
    is at least the budget. Any non-zero exit is the sentinel.
    """
    exit_code = int(row.get("exit_code") or "0")
    if exit_code != 0:
        raise ValueError("inconclusive row")
    hit = row.get("hit") == "1"
    charged_text = row.get("charged") or ""
    charged = float(charged_text) if charged_text else 0.0
    if not hit:
        return max(charged, float(budget))
    return charged


def wilcoxon(left: list[float], right: list[float]) -> tuple[float, float]:
    """Two-sided signed-rank on right minus left.

    Zero differences are dropped. Ties in the absolute
    differences take average ranks. Each non-zero rank is given
    an independent sign under the null. The p value is twice
    the smaller tail, capped at one. The returned statistic is
    the sum of the ranks whose difference is positive.
    """
    diffs = [b - a for a, b in zip(left, right)]
    nonzero = [diff for diff in diffs if diff != 0.0]
    count = len(nonzero)
    if count == 0:
        return 0.0, 1.0
    ordered = sorted(abs(diff) for diff in nonzero)
    rank_twice: dict[float, int] = {}
    index = 0
    while index < count:
        end = index
        while end + 1 < count and ordered[end + 1] == ordered[index]:
            end += 1
        rank_twice[ordered[index]] = index + end + 2
        index = end + 1
    positive = 0
    items: list[int] = []
    for diff in nonzero:
        twice = rank_twice[abs(diff)]
        items.append(twice)
        if diff > 0.0:
            positive += twice
    ways_at = {0: 1}
    for item in items:
        nxt: dict[int, int] = {}
        for total, ways in ways_at.items():
            nxt[total] = nxt.get(total, 0) + ways
            nxt[total + item] = nxt.get(total + item, 0) + ways
        ways_at = nxt
    all_ways = sum(ways_at.values())
    left_tail = sum(ways for total, ways in ways_at.items() if total <= positive)
    right_tail = sum(ways for total, ways in ways_at.items() if total >= positive)
    probability = min(1.0, 2.0 * min(left_tail, right_tail) / all_ways)
    return positive / 2.0, probability


def ratio(num: float, den: float) -> float:
    if den == 0.0:
        return math.inf if num > 0.0 else math.nan
    return num / den


def near(value: float, limit: float) -> bool:
    if math.isinf(value) or math.isnan(value):
        return False
    return value <= limit or abs(value - limit) <= 1.0e-12


def bootstrap_ratio(left: list[float], right: list[float], rng: random.Random) -> tuple[float, float]:
    """Type-7 percentiles of the paired median ratio right/left."""
    count = len(left)
    draws: list[float] = []
    for _ in range(gate.BOOT_DRAWS):
        pick = [rng.randrange(count) for _ in range(count)]
        med_left = type7([left[i] for i in pick])
        med_right = type7([right[i] for i in pick])
        draws.append(ratio(med_right, med_left))
    return type7(draws, 0.025), type7(draws, 0.975)


def finite_float(text: str) -> float | None:
    if text is None or text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, int], dict[str, str]]:
    found: dict[tuple[str, str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["cell"], row["driver"], row["arm"], int(row["seed"]))
        found[key] = row
    return found


def hit_bit(row: dict[str, str] | None) -> int:
    if row is None or int(row.get("exit_code") or "0") != 0:
        return 0
    return 1 if row.get("hit") == "1" else 0


def series(found, cell: gate.Cell, arm: str, field: str, ranked: bool) -> list[float] | None:
    values: list[float] = []
    for seed in gate.SEEDS:
        row = found.get((cell.name, cell.driver, arm, seed))
        if row is None:
            return None
        if ranked:
            values.append(ranked_charged(row, cell.budget))
            continue
        if int(row.get("exit_code") or "0") != 0:
            return None
        value = finite_float(row.get(field) or "")
        if value is None:
            return None
        values.append(value)
    return values


def outside_per(found, cell: gate.Cell, arm: str) -> list[float] | None:
    values: list[float] = []
    for seed in gate.SEEDS:
        row = found.get((cell.name, cell.driver, arm, seed))
        if row is None or int(row.get("exit_code") or "0") != 0:
            return None
        outside = finite_float(row.get("outside_s") or "")
        charged = finite_float(row.get("charged") or "")
        if outside is None or charged is None or charged <= 0.0:
            return None
        values.append(outside / charged)
    return values


def fmt(value: float | None, digits: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return format(value, digits)


def evaluate(found) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    rng = random.Random(gate.BOOT_SEED)
    prepared = []
    for cell in gate.CELLS:
        hits = {}
        strings = {}
        complete = True
        for arm in ("A", "A2", "B", "B2"):
            bits = []
            for seed in gate.SEEDS:
                row = found.get((cell.name, cell.driver, arm, seed))
                if row is None or row.get("conclusive") != "1":
                    complete = False
                    bits.append("0")
                else:
                    bits.append(str(hit_bit(row)))
            strings[arm] = "".join(bits)
            hits[arm] = sum(int(bit) for bit in bits) if complete else None
        charged = {
            arm: series(found, cell, arm, "charged", ranked=True)
            for arm in ("A", "A2", "B")
        }
        calls = {
            arm: series(found, cell, arm, "calls", ranked=False)
            for arm in ("A", "A2", "B")
        }
        wall = {
            arm: series(found, cell, arm, "wall_s", ranked=False)
            for arm in ("A", "A2", "B")
        }
        outside = {
            arm: outside_per(found, cell, arm)
            for arm in ("A", "A2", "B")
        }
        prepared.append(
            {
                "cell": cell,
                "active": any(
                    key[0] == cell.name and key[1] == cell.driver
                    for key in found
                ),
                "complete": complete,
                "hits": hits,
                "strings": strings,
                "charged": charged,
                "calls": calls,
                "wall": wall,
                "outside": outside,
            }
        )

    hit_allowance = 0
    charged_terms: list[float] = []
    for item in prepared:
        if not item["complete"]:
            continue
        hit_allowance = max(hit_allowance, abs(item["hits"]["A2"] - item["hits"]["A"]))
        med_a = type7(item["charged"]["A"])
        med_ap = type7(item["charged"]["A2"])
        term = ratio(med_ap, med_a)
        if math.isfinite(term) and term > 0.0:
            charged_terms.append(max(term, 1.0 / term))
    charged_band = max(charged_terms) if charged_terms else math.nan

    summary_rows = []
    band_rows = []
    failed = []
    for item in prepared:
        cell = item["cell"]
        b_count = c_count = None
        probability = None
        worse = ""
        deficit = None
        rule1 = rule2 = rule3 = "0"
        search_equal = charge_equal = repeat_equal = False
        score_differences = 0
        if item["complete"]:
            search_equal = all(
                found[(cell.name, cell.driver, "A", seed)].get("calls")
                == found[(cell.name, cell.driver, "B", seed)].get("calls")
                and found[(cell.name, cell.driver, "A", seed)].get("best_bits")
                == found[(cell.name, cell.driver, "B", seed)].get("best_bits")
                for seed in gate.SEEDS
            )
            charge_equal = all(
                row.get("hit") != "1"
                or row.get("charged") == row.get("calls")
                for row in (
                    found[(cell.name, cell.driver, "B", seed)]
                    for seed in gate.SEEDS
                )
            )
            repeat_equal = all(
                all(
                    found[(cell.name, cell.driver, "B", seed)].get(field)
                    == found[(cell.name, cell.driver, "B2", seed)].get(field)
                    for field in (
                        "hit",
                        "printed_hit",
                        "calls",
                        "charged",
                        "best_bits",
                    )
                )
                for seed in gate.SEEDS
            )
            score_differences = sum(
                found[(cell.name, cell.driver, "B", seed)].get("hit")
                != found[(cell.name, cell.driver, "B", seed)].get("printed_hit")
                for seed in gate.SEEDS
            )
            b_count = c_count = 0
            for seed in gate.SEEDS:
                a_hit = hit_bit(found[(cell.name, cell.driver, "A", seed)])
                b_hit = hit_bit(found[(cell.name, cell.driver, "B", seed)])
                if a_hit == 0 and b_hit == 1:
                    b_count += 1
                elif a_hit == 1 and b_hit == 0:
                    c_count += 1
            probability = mcnemar_p(b_count, c_count)
            is_worse = c_count > b_count and probability < 0.05
            worse = "1" if is_worse else "0"
            deficit = item["hits"]["A"] - item["hits"]["B"]
            rule1 = "1" if (not is_worse and deficit <= hit_allowance) else "0"
        med_charged = {}
        med_calls = {}
        med_wall = {}
        med_out = {}
        for arm in ("A", "A2", "B"):
            med_charged[arm] = type7(item["charged"][arm]) if item["charged"][arm] is not None else None
            med_calls[arm] = type7(item["calls"][arm]) if item["calls"][arm] is not None else None
            med_wall[arm] = type7(item["wall"][arm]) if item["wall"][arm] is not None else None
            med_out[arm] = type7(item["outside"][arm]) if item["outside"][arm] is not None else None
        charged_ba = charged_aa = None
        if med_charged["A"] is not None and med_charged["B"] is not None:
            charged_ba = ratio(med_charged["B"], med_charged["A"])
        if med_charged["A"] is not None and med_charged["A2"] is not None:
            charged_aa = ratio(med_charged["A2"], med_charged["A"])
        improved = bool(item["complete"] and item["hits"]["B"] > item["hits"]["A"])
        if item["complete"] and charged_ba is not None and math.isfinite(charged_band):
            rule2 = "1" if improved or near(charged_ba, charged_band) else "0"
        out_ba = out_aa = None
        if med_out["A"] is not None and med_out["B"] is not None:
            out_ba = ratio(med_out["B"], med_out["A"])
        if med_out["A"] is not None and med_out["A2"] is not None:
            out_aa = ratio(med_out["A2"], med_out["A"])
        if out_ba is not None:
            rule3 = "1" if near(out_ba, gate.OUTSIDE_CAP) else "0"
        calls_ba = calls_aa = wall_ba = wall_aa = None
        if med_calls["A"] not in (None, 0.0) and med_calls["B"] is not None and med_calls["A"] != 0:
            calls_ba = ratio(med_calls["B"], med_calls["A"])
        if med_calls["A"] not in (None, 0.0) and med_calls["A2"] is not None:
            calls_aa = ratio(med_calls["A2"], med_calls["A"])
        if med_wall["A"] not in (None, 0.0) and med_wall["B"] is not None:
            wall_ba = ratio(med_wall["B"], med_wall["A"])
        if med_wall["A"] not in (None, 0.0) and med_wall["A2"] is not None:
            wall_aa = ratio(med_wall["A2"], med_wall["A"])

        def ci(left, right):
            if left is None or right is None:
                return None, None
            return bootstrap_ratio(left, right, rng)

        ch_lo, ch_hi = ci(item["charged"]["A"], item["charged"]["B"])
        cha_lo, cha_hi = ci(item["charged"]["A"], item["charged"]["A2"])
        ca_lo, ca_hi = ci(item["calls"]["A"], item["calls"]["B"])
        w_lo, w_hi = ci(item["wall"]["A"], item["wall"]["B"])
        o_lo, o_hi = ci(item["outside"]["A"], item["outside"]["B"])
        oa_lo, oa_hi = ci(item["outside"]["A"], item["outside"]["A2"])
        w_stat = w_prob = None
        if item["charged"]["A"] is not None and item["charged"]["B"] is not None:
            w_stat, w_prob = wilcoxon(item["charged"]["A"], item["charged"]["B"])
        equal = compared = 0
        if item["complete"]:
            for seed in gate.SEEDS:
                a_row = found[(cell.name, cell.driver, "A", seed)]
                b_row = found[(cell.name, cell.driver, "B", seed)]
                a_bits = a_row.get("best_bits") or ""
                b_bits = b_row.get("best_bits") or ""
                if a_bits and b_bits:
                    compared += 1
                    if a_bits == b_bits:
                        equal += 1
        cell_pass = (
            "1"
            if (
                rule1 == "1"
                and rule2 == "1"
                and rule3 == "1"
                and search_equal
                and charge_equal
                and repeat_equal
            )
            else "0"
        )
        if cell_pass == "0":
            reasons = []
            if rule1 != "1":
                reasons.append("hits")
            if rule2 != "1":
                reasons.append("charged")
            if rule3 != "1":
                reasons.append("outside")
            if not search_equal:
                reasons.append("search")
            if not charge_equal:
                reasons.append("hit charge")
            if not repeat_equal:
                reasons.append("repeat")
            failed.append(f"{cell.name} {cell.driver} ({', '.join(reasons)})")
        summary_rows.append(
            {
                "cell": cell.name,
                "driver": cell.driver,
                "role": cell.role,
                "n": str(len(gate.SEEDS)) if item["complete"] else "0",
                "hits_a": "" if item["hits"]["A"] is None else str(item["hits"]["A"]),
                "hits_ap": "" if item["hits"]["A2"] is None else str(item["hits"]["A2"]),
                "hits_b": "" if item["hits"]["B"] is None else str(item["hits"]["B"]),
                "mcnemar_b": "" if b_count is None else str(b_count),
                "mcnemar_c": "" if c_count is None else str(c_count),
                "mcnemar_p": fmt(probability, ".6g"),
                "b_worse": worse,
                "hit_deficit": "" if deficit is None else str(deficit),
                "hit_allowance": str(hit_allowance),
                "rule1": rule1,
                "med_charged_a": fmt(med_charged["A"], ".3f"),
                "med_charged_ap": fmt(med_charged["A2"], ".3f"),
                "med_charged_b": fmt(med_charged["B"], ".3f"),
                "charged_ratio_ba": fmt(charged_ba, ".6f"),
                "charged_ratio_aa": fmt(charged_aa, ".6f"),
                "charged_band": fmt(charged_band, ".6f"),
                "charged_ratio_ba_lo": fmt(ch_lo, ".6f"),
                "charged_ratio_ba_hi": fmt(ch_hi, ".6f"),
                "charged_ratio_aa_lo": fmt(cha_lo, ".6f"),
                "charged_ratio_aa_hi": fmt(cha_hi, ".6f"),
                "hits_improved": "1" if improved else "0",
                "rule2": rule2,
                "med_calls_a": fmt(med_calls["A"], ".3f"),
                "med_calls_ap": fmt(med_calls["A2"], ".3f"),
                "med_calls_b": fmt(med_calls["B"], ".3f"),
                "calls_ratio_ba": fmt(calls_ba, ".6f"),
                "calls_ratio_aa": fmt(calls_aa, ".6f"),
                "calls_ratio_ba_lo": fmt(ca_lo, ".6f"),
                "calls_ratio_ba_hi": fmt(ca_hi, ".6f"),
                "med_wall_a": fmt(med_wall["A"], ".6f"),
                "med_wall_ap": fmt(med_wall["A2"], ".6f"),
                "med_wall_b": fmt(med_wall["B"], ".6f"),
                "wall_ratio_ba": fmt(wall_ba, ".6f"),
                "wall_ratio_aa": fmt(wall_aa, ".6f"),
                "wall_ratio_ba_lo": fmt(w_lo, ".6f"),
                "wall_ratio_ba_hi": fmt(w_hi, ".6f"),
                "med_outside_per_a": fmt(med_out["A"], ".6e"),
                "med_outside_per_ap": fmt(med_out["A2"], ".6e"),
                "med_outside_per_b": fmt(med_out["B"], ".6e"),
                "outside_ratio_ba": fmt(out_ba, ".6f"),
                "outside_ratio_aa": fmt(out_aa, ".6f"),
                "outside_ratio_ba_lo": fmt(o_lo, ".6f"),
                "outside_ratio_ba_hi": fmt(o_hi, ".6f"),
                "outside_ratio_aa_lo": fmt(oa_lo, ".6f"),
                "outside_ratio_aa_hi": fmt(oa_hi, ".6f"),
                "rule3": rule3,
                "search_equal": "1" if search_equal else "0",
                "hit_charge_equal": "1" if charge_equal else "0",
                "repeat_equal": "1" if repeat_equal else "0",
                "score_differences": str(score_differences),
                "wilcoxon_w": fmt(w_stat, ".3f"),
                "wilcoxon_p": fmt(w_prob, ".6g"),
                "bit_equal": str(equal),
                "bit_compared": str(compared),
                "hitstr_a": item["strings"]["A"],
                "hitstr_b": item["strings"]["B"],
                "cell_pass": cell_pass,
            }
        )
        if item["complete"] and charged_aa is not None:
            band_rows.append(
                {
                    "cell": cell.name,
                    "driver": cell.driver,
                    "hits_a": str(item["hits"]["A"]),
                    "hits_ap": str(item["hits"]["A2"]),
                    "hit_abs": str(abs(item["hits"]["A2"] - item["hits"]["A"])),
                    "med_charged_a": fmt(med_charged["A"], ".3f"),
                    "med_charged_ap": fmt(med_charged["A2"], ".3f"),
                    "ratio_aa": fmt(charged_aa, ".6f"),
                    "ratio_term": fmt(max(charged_aa, 1.0 / charged_aa), ".6f")
                    if math.isfinite(charged_aa) and charged_aa > 0.0
                    else "",
                    "hit_allowance": str(hit_allowance),
                    "charged_band": fmt(charged_band, ".6f"),
                }
            )
    if any(item["active"] and not item["complete"] for item in prepared):
        verdict = "INCONCLUSIVE"
    elif not summary_rows:
        verdict = "INCONCLUSIVE (no rows)"
    elif failed:
        verdict = "FAIL " + "; ".join(failed)
    else:
        verdict = "PASS"
    return summary_rows, band_rows, verdict


SUMMARY_FIELDS = [
    "cell",
    "driver",
    "role",
    "n",
    "hits_a",
    "hits_ap",
    "hits_b",
    "mcnemar_b",
    "mcnemar_c",
    "mcnemar_p",
    "b_worse",
    "hit_deficit",
    "hit_allowance",
    "rule1",
    "med_charged_a",
    "med_charged_ap",
    "med_charged_b",
    "charged_ratio_ba",
    "charged_ratio_aa",
    "charged_band",
    "charged_ratio_ba_lo",
    "charged_ratio_ba_hi",
    "charged_ratio_aa_lo",
    "charged_ratio_aa_hi",
    "hits_improved",
    "rule2",
    "med_calls_a",
    "med_calls_ap",
    "med_calls_b",
    "calls_ratio_ba",
    "calls_ratio_aa",
    "calls_ratio_ba_lo",
    "calls_ratio_ba_hi",
    "med_wall_a",
    "med_wall_ap",
    "med_wall_b",
    "wall_ratio_ba",
    "wall_ratio_aa",
    "wall_ratio_ba_lo",
    "wall_ratio_ba_hi",
    "med_outside_per_a",
    "med_outside_per_ap",
    "med_outside_per_b",
    "outside_ratio_ba",
    "outside_ratio_aa",
    "outside_ratio_ba_lo",
    "outside_ratio_ba_hi",
    "outside_ratio_aa_lo",
    "outside_ratio_aa_hi",
    "rule3",
    "search_equal",
    "hit_charge_equal",
    "repeat_equal",
    "score_differences",
    "wilcoxon_w",
    "wilcoxon_p",
    "bit_equal",
    "bit_compared",
    "hitstr_a",
    "hitstr_b",
    "cell_pass",
]

BAND_FIELDS = [
    "cell",
    "driver",
    "hits_a",
    "hits_ap",
    "hit_abs",
    "med_charged_a",
    "med_charged_ap",
    "ratio_aa",
    "ratio_term",
    "hit_allowance",
    "charged_band",
]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def synthetic_row(cell, driver, arm, seed, hit, charged, outside, best_bits, exit_code="0"):
    calls = charged if hit == "1" else "400000"
    return {
        "cell": cell,
        "driver": driver,
        "arm": arm,
        "seed": str(seed),
        "slot": "1",
        "core": "0",
        "affinity": "0",
        "hit": hit,
        "calls": calls,
        "charged": charged,
        "wall_s": "1.0",
        "pot_s": "0.5",
        "outside_s": outside,
        "best": "-1.0",
        "best_bits": best_bits,
        "budget": "400000",
        "rss_anon_peak": "1",
        "ru_maxrss": "1",
        "vmhwm_peak": "1",
        "printed_hit": hit,
        "conclusive": "1",
        "log_sha256": "aa",
        "binary_sha256": "aa",
        "revision": "rev",
        "machine_id_sha256": "mm",
        "session": "s",
        "sequence": "1",
        "exit_code": exit_code,
    }


def fill(cell_name, driver, spec) -> list[dict[str, str]]:
    """spec maps arm to (hit_string, charged, outside, bits)."""
    rows = []
    spec = dict(spec)
    spec.setdefault("B2", spec["B"])
    budget_cell = next(cell for cell in gate.CELLS if cell.name == cell_name and cell.driver == driver)
    for arm, (hits, charged, outside, bits) in spec.items():
        for offset, seed in enumerate(gate.SEEDS):
            value = charged[offset] if isinstance(charged, list) else charged
            row = synthetic_row(
                cell_name,
                driver,
                arm,
                seed,
                hits[offset],
                str(value),
                str(outside),
                bits,
            )
            row["budget"] = str(budget_cell.budget)
            row["calls"] = str(value if hits[offset] == "1" else budget_cell.budget)
            rows.append(row)
    return rows


def self_test() -> None:
    if abs(type7([1.0, 2.0, 3.0, 4.0]) - 2.5) > 1e-12:
        raise SystemExit("type7")
    if abs(mcnemar_p(0, 0) - 1.0) > 1e-12:
        raise SystemExit("mcnemar ties")
    if not (mcnemar_p(0, 6) < 0.05):
        raise SystemExit("mcnemar 0/6")
    if mcnemar_p(1, 5) < 0.05:
        raise SystemExit("mcnemar 1/5 should not be below 0.05")
    statistic, probability = wilcoxon([0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6])
    if abs(statistic - 21.0) > 1e-9 or abs(probability - 2.0 / 64.0) > 1e-12:
        raise SystemExit(f"wilcoxon {statistic} {probability}")
    zero_stat, zero_p = wilcoxon([1, 2], [1, 2])
    if zero_stat != 0.0 or zero_p != 1.0:
        raise SystemExit("wilcoxon zeros")

    ones = "1" * 24
    quiet = fill(
        "lj98",
        "h2h_gmin_proof",
        {
            "A": (ones, 100, 0.10, "ab"),
            "A2": (ones, 100, 0.10, "ab"),
            "B": (ones, 100, 0.10, "ab"),
        },
    )
    # A second cell widens nothing. B loses six hits and is slower.
    lost = "1" * 18 + "0" * 6
    loud = fill(
        "m30r14",
        "h2h_morse",
        {
            "A": (ones, 100, 0.10, "cd"),
            "A2": (ones, 100, 0.10, "cd"),
            "B": (lost, 400, 0.20, "ef"),
        },
    )
    found = index_rows(quiet + loud)
    summary, _band, verdict = evaluate(found)
    by_name = {(row["cell"], row["driver"]): row for row in summary}
    loud_row = by_name[("m30r14", "h2h_morse")]
    if loud_row["rule1"] != "0" or loud_row["rule2"] != "0" or loud_row["cell_pass"] != "0":
        raise SystemExit(f"loud should fail {loud_row['rule1']} {loud_row['rule2']}")
    if not verdict.startswith("FAIL"):
        raise SystemExit(verdict)
    quiet_row = by_name[("lj98", "h2h_gmin_proof")]
    if quiet_row["cell_pass"] != "1":
        raise SystemExit("quiet cell should pass")

    # Hits improve, so a charged ratio above the A/A2 band is waived.
    # Outside stays under the cap.
    better = "1" * 24
    base_hits = "1" * 20 + "0" * 4
    waived = fill(
        "m38r6",
        "h2h_morse",
        {
            "A": (base_hits, 100, 0.10, "aa"),
            "A2": (base_hits, 100, 0.10, "aa"),
            "B": (better, 500, 0.10, "bb"),
        },
    )
    # Keep one quiet cell so the charged band stays 1.
    found = index_rows(quiet + waived)
    summary, _band, _verdict = evaluate(found)
    by_name = {(row["cell"], row["driver"]): row for row in summary}
    waived_row = by_name[("m38r6", "h2h_morse")]
    if waived_row["rule2"] != "1" or waived_row["hits_improved"] != "1":
        raise SystemExit("improved hits should waive charged")
    if waived_row["rule3"] != "1":
        raise SystemExit("outside cap")

    # Outside ratio 1.2 fails the cap. Charged matches, hits match.
    slow = fill(
        "m38r6",
        "h2h_morse",
        {
            "A": (ones, 100, 0.10, "aa"),
            "A2": (ones, 100, 0.10, "aa"),
            "B": (ones, 100, 0.12, "aa"),
        },
    )
    found = index_rows(quiet + slow)
    summary, _band, _verdict = evaluate(found)
    by_name = {(row["cell"], row["driver"]): row for row in summary}
    if by_name[("m38r6", "h2h_morse")]["rule3"] != "0":
        raise SystemExit("outside 1.2 should fail")

    # A2 differs by 2 hits in one cell. B may lose 2 and not 3.
    ap = "1" * 22 + "0" * 2
    lose2 = "1" * 22 + "0" * 2
    band_cell = fill(
        "lj98",
        "h2h_gmin_proof",
        {
            "A": (ones, 100, 0.10, "aa"),
            "A2": (ap, 110, 0.10, "aa"),
            "B": (lose2, 100, 0.10, "aa"),
        },
    )
    other = fill(
        "m30r14",
        "h2h_morse",
        {
            "A": (ones, 100, 0.10, "aa"),
            "A2": (ones, 100, 0.10, "aa"),
            "B": (ones, 100, 0.10, "aa"),
        },
    )
    found = index_rows(band_cell + other)
    summary, band, verdict = evaluate(found)
    by_name = {(row["cell"], row["driver"]): row for row in summary}
    if by_name[("lj98", "h2h_gmin_proof")]["rule1"] != "1":
        raise SystemExit("allowance of 2 should cover a deficit of 2")
    if by_name[("lj98", "h2h_gmin_proof")]["hit_allowance"] != "2":
        raise SystemExit("allowance")
    # Deficit of 3 against the same allowance fails.
    lose3 = "1" * 21 + "0" * 3
    band_cell = fill(
        "lj98",
        "h2h_gmin_proof",
        {
            "A": (ones, 100, 0.10, "aa"),
            "A2": (ap, 100, 0.10, "aa"),
            "B": (lose3, 100, 0.10, "aa"),
        },
    )
    found = index_rows(band_cell + other)
    summary, _band, _verdict = evaluate(found)
    by_name = {(row["cell"], row["driver"]): row for row in summary}
    if by_name[("lj98", "h2h_gmin_proof")]["rule1"] != "0":
        raise SystemExit("deficit 3 should fail an allowance of 2")
    if not band:
        raise SystemExit("band rows")
    print("summarize ok")


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--band", type=Path, required=True)
    args = parser.parse_args()
    found = index_rows(load_rows(args.rows))
    summary, band, verdict = evaluate(found)
    write_csv(args.summary, SUMMARY_FIELDS, summary)
    write_csv(args.band, BAND_FIELDS, band)
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
