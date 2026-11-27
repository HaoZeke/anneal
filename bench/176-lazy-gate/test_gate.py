# Copyright (c) anneal developers
"""Bars for the lazy-basin gate, including the no-relabel rule."""

from __future__ import annotations

import unittest

from analyze import identity_table, summarize_cell
from gate_stats import student_t_crit
from panel import CELL_BY_NAME, SESSION


def row(cell: str, arm: str, seed: int, **overrides: str) -> dict[str, str]:
    base = {
        "session_id": SESSION,
        "cell": cell,
        "arm": arm,
        "seed": str(seed),
        "status": "ok",
        "hit": "1",
        "charged": "10",
        "budget": "100",
        "wall_s": "1",
        "outside_per_charged_s": "1",
        "rss_anon_kb": "1000",
        "basis_walks": "1",
        "best_bits": "0x1",
        "struct_sha256": "aa",
        "final_energy_bits": "0x2",
        "final_struct_sha256": "bb",
    }
    base.update(overrides)
    return base


def panel(cell: str, **scale: str) -> dict[str, list[dict[str, str]]]:
    spec = CELL_BY_NAME[cell]
    groups = {arm: [] for arm in ("A", "A2", "B")}
    for seed in spec.seeds:
        for arm in groups:
            groups[arm].append(row(cell, arm, seed))
    for arm, fields in scale.items():
        for item in groups[arm]:
            item.update(fields)
    return groups


class CritTest(unittest.TestCase):
    def test_known_critical_values(self) -> None:
        self.assertAlmostEqual(student_t_crit(23), 2.068657610419, places=9)
        self.assertAlmostEqual(student_t_crit(11), 2.200985160082, places=9)


class BarTest(unittest.TestCase):
    def test_equal_arms_pass(self) -> None:
        spec = CELL_BY_NAME["m38r14"]
        summary = summarize_cell(spec, panel(spec.cell), 0, 40)
        self.assertEqual(summary["verdict"], "PASS")
        self.assertEqual(summary["outside_bar"], "PASS")
        self.assertEqual(summary["rss_bar"], "PASS")

    def test_outside_fail_is_not_relabeled(self) -> None:
        spec = CELL_BY_NAME["m38r14"]
        groups = panel(
            spec.cell,
            B={"outside_per_charged_s": "1.2"},
            A2={"outside_per_charged_s": "1.5"},
        )
        summary = summarize_cell(spec, groups, 1, 40)
        self.assertEqual(summary["outside_bar"], "FAIL")
        self.assertEqual(summary["verdict"], "FAIL")
        self.assertGreater(float(summary["a2_outside_upper"]), 1.2)

    def test_rss_fail_only_when_lower_bound_clears_the_band(self) -> None:
        spec = CELL_BY_NAME["m38r14"]
        high = panel(spec.cell, B={"rss_anon_kb": "1200"})
        failed = summarize_cell(spec, high, 2, 40)
        self.assertEqual(failed["rss_bar"], "FAIL")
        wide = panel(
            spec.cell,
            B={"rss_anon_kb": "1200"},
            A2={"rss_anon_kb": "1300"},
        )
        held = summarize_cell(spec, wide, 3, 40)
        self.assertEqual(held["rss_bar"], "PASS")
        self.assertLessEqual(float(held["rss_lower"]), float(held["rss_limit"]))

    def test_hit_loss_fails(self) -> None:
        spec = CELL_BY_NAME["m38r14"]
        groups = panel(spec.cell, B={"hit": "0", "charged": "100"})
        summary = summarize_cell(spec, groups, 4, 40)
        self.assertEqual(summary["hit_bar"], "FAIL")
        self.assertEqual(summary["median_calls_b"], "400001")
        self.assertEqual(summary["median_charged_b"], "100")

    def test_identity_reports_each_mismatched_seed(self) -> None:
        spec = CELL_BY_NAME["m38r14"]
        rows = []
        for seed in spec.seeds:
            rows.append(row(spec.cell, "B", seed))
            digest = "cc" if seed == 2003 else "aa"
            rows.append(row(spec.cell, "B0", seed, struct_sha256=digest))
        mismatches, summary = identity_table(rows)
        cell_summary = next(item for item in summary if item["cell"] == spec.cell)
        self.assertEqual(cell_summary["verdict"], "FAIL")
        cell_mismatches = [
            (item["seed"], item["field"])
            for item in mismatches
            if item["cell"] == spec.cell and item["field"] != "missing"
        ]
        self.assertEqual(cell_mismatches, [("2003", "struct_sha256")])


if __name__ == "__main__":
    unittest.main()
