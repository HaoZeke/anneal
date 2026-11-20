# Copyright (c) anneal developers
"""Statistic checks for the part B verdict. No searches."""

from __future__ import annotations

import summarize


def row(system: str, arm: str, seed: int, hit: int, calls: int, budget: int, outside: float, charged: int, slot: int) -> dict[str, str]:
    return {
        "system": system,
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "cohort": "gate",
        "status": "ok",
        "hit": str(hit),
        "calls": str(calls),
        "budget": str(budget),
        "charged": str(charged),
        "wall_s": "1.0",
        "outside_per_call_s": f"{outside / charged:.12g}",
        "core": "0",
    }


def paired(system: str, hits_a, hits_b, calls_a, calls_b, budget: int, out_a: float, out_b: float, charged: int = 1000):
    rows = []
    for index, (ha, hb, ca, cb) in enumerate(zip(hits_a, hits_b, calls_a, calls_b, strict=True)):
        seed = 2000 + index
        slot_a, slot_b = (1, 2) if index % 2 == 0 else (2, 1)
        rows.append(row(system, "A", seed, ha, ca, budget, out_a, charged, slot_a))
        rows.append(row(system, "B", seed, hb, cb, budget, out_b, charged, slot_b))
    return rows


def test_hit_loss_ignores_p() -> None:
    rows = paired(
        "demo",
        [1] * 6 + [0] * 6,
        [1] * 5 + [0] * 7,
        [10] * 12,
        [10] * 12,
        100,
        1.0,
        1.0,
    )
    scored = summarize.score_system("demo", rows)
    assert scored["loss_hits"] == "1"
    assert scored["verdict"] == "FIX"
    assert float(scored["fisher_p"]) > 0.05


def test_calls_loss_needs_significance() -> None:
    tied = summarize.score_system(
        "demo", paired("demo", [1] * 12, [1] * 12, [10] * 12, [10] * 12, 100, 1.0, 1.0)
    )
    assert tied["verdict"] == "PASS"
    assert tied["loss_calls"] == "0"
    slower = summarize.score_system(
        "demo",
        paired("demo", [1] * 12, [1] * 12, [10] * 12, [1000] * 12, 2000, 1.0, 1.0),
    )
    assert slower["loss_calls"] == "1"
    assert slower["verdict"] == "FIX"


def test_outside_ratio_threshold() -> None:
    under = summarize.score_system("demo", paired("demo", [1] * 4, [1] * 4, [5] * 4, [5] * 4, 20, 1.0, 1.1, 1))
    over = summarize.score_system("demo", paired("demo", [1] * 4, [1] * 4, [5] * 4, [5] * 4, 20, 1.0, 1.11, 1))
    assert float(under["outside_per_call_ratio"]) == 1.1
    assert under["loss_outside"] == "0"
    assert over["loss_outside"] == "1"
    assert over["verdict"] == "FIX"


def test_miss_ranks_after_budget_hit() -> None:
    hit = {"hit": "1", "calls": "100", "budget": "100"}
    miss = {"hit": "0", "calls": "100", "budget": "100"}
    assert summarize.call_score(miss) > summarize.call_score(hit)


def test_balanced_slots() -> None:
    rows = paired("demo", [1, 0], [1, 0], [3, 4], [3, 4], 10, 1.0, 1.0)
    counts = summarize.verify_abba(rows)
    assert set(counts.values()) == {1}


def main() -> None:
    test_hit_loss_ignores_p()
    test_calls_loss_needs_significance()
    test_outside_ratio_threshold()
    test_miss_ranks_after_budget_hit()
    test_balanced_slots()
    fisher = summarize.fisher_two_sided(12, 0, 12, 0)
    assert abs(fisher - 1.0) < 1e-12
    print("ok")


if __name__ == "__main__":
    main()
