"""Placement pads are compared with arm A at the same seed."""

from __future__ import annotations

import unittest

from analyze_gate import summarize_cell
from gate_panel import CELL_BY_NAME, PLACEMENT_PADS, PLACEMENT_SEED, SEEDS, SESSION_R2


def row(cell: str, arm: str, seed: int, rss: float) -> dict[str, str]:
    return {
        "session_id": SESSION_R2,
        "cell": cell,
        "arm": arm,
        "seed": str(seed),
        "status": "ok",
        "superseded": "",
        "attempt": "1",
        "hit": "1",
        "charged": "10",
        "outside_s": "1.0",
        "rss_anon_kb": str(rss),
        "vm_hwm_kb": "1",
        "ru_maxrss_kb": "1",
        "basis_walks": "1",
        "basis_evidence": "1",
        "basis_claims": "1",
        "basis_probes": "1",
        "basis_candidates": "1",
        "basis_spent": "1",
    }


class PlacementSeedTest(unittest.TestCase):
    def test_pad_uses_same_seed_not_the_median(self) -> None:
        cell = "lj38"
        rows: list[dict[str, str]] = []
        for seed in SEEDS:
            # Seed 2000 is far from the median of the other A rows.
            rss_a = 1000.0 if seed == PLACEMENT_SEED else 5000.0
            rows.append(row(cell, "A", seed, rss_a))
            rows.append(row(cell, "A'", seed, 4000.0))
            rows.append(row(cell, "B", seed, 4000.0))
        for arm, _size in PLACEMENT_PADS:
            rows.append(row(cell, arm, PLACEMENT_SEED, 1000.0))
        summary = summarize_cell(CELL_BY_NAME[cell], rows)
        # Each pad matches A at seed 2000, so the placement term is zero.
        # A median comparison would report 4000.
        self.assertEqual(summary["rss_place_kb"], "0")


if __name__ == "__main__":
    unittest.main()
