# Copyright (c) anneal developers
"""Checks for the PR 186 restack gate's non-timing invariants."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "openconf_h2h"))
import h2h_job  # noqa: E402
import paired_score  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "pr186_restack_analyze",
    ROOT / "bench" / "gate-pr186-restack" / "analyze.py",
)
assert SPEC is not None and SPEC.loader is not None
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


def rows(
    *,
    path_enabled: bool = False,
    path_draws: int = 0,
    path_spent: int = 0,
    calls_to_hit: int = 40,
):
    out = []
    for arm in ("A", "B", "Ap"):
        out.append(
            {
                "cell": "rq3",
                "kind": "molecule",
                "seed": "2000",
                "arm": arm,
                "hit": "1",
                "budget": "100",
                "callback_calls": "180",
                "calls_to_hit": str(calls_to_hit),
                "charged": "90",
                "wall_s": "2",
                "inside_s": "1",
                "outside_per_call_s": "0.0055555556",
                "rss_anon_kb": "1000",
                "peak_rss_anon_kb": "1100",
                "branch_on_stall": (
                    str(int(path_enabled)) if arm == "B" else "0"
                ),
                "b_path_draws": str(path_draws if arm == "B" else 0),
                "b_path_spent": str(path_spent if arm == "B" else 0),
                "trajectory": "same",
                "energy_bits": "",
                "error": "",
            }
        )
    return out


def test_disabled_conf_miss_path_requires_zero_counters():
    result = ANALYZE.judge(rows(path_draws=0, path_spent=0), "rq3")
    assert result["verdict"] == "PASS"
    assert result["b_path_matches"] == 1
    result = ANALYZE.judge(rows(path_draws=1, path_spent=10), "rq3")
    assert result["verdict"] == "FIX"
    assert "conf-miss path" in result["failures"]


def test_enabled_conf_miss_path_requires_nonzero_counters():
    result = ANALYZE.judge(
        rows(path_enabled=True, path_draws=1, path_spent=10),
        "rq3",
    )
    assert result["verdict"] == "PASS"
    assert result["b_path_enabled"] == 1
    result = ANALYZE.judge(rows(path_enabled=True), "rq3")
    assert result["verdict"] == "FIX"
    assert "conf-miss path" in result["failures"]


def test_gate_rejects_calls_to_hit_above_the_budget():
    result = ANALYZE.judge(rows(calls_to_hit=101), "rq3")
    assert result["verdict"] == "FIX"
    assert result["failures"] == "call count exceeds budget"


def test_molecular_trace_records_charged_calls_at_first_hit():
    harvest = h2h_job.Harvest(0.1)
    point = np.zeros(3)
    harvest.energy(point, -1.0)
    harvest.gradient(point, np.zeros(3))
    harvest.energy(point, -2.0)
    harvest.gradient(point, np.zeros(3))
    harvest.close()
    assert harvest.calls == 4
    assert harvest.found == [2]


def test_paired_calls_use_one_value_and_gradient_unit():
    counts = {"energy_calls": 7, "grad_calls": 3}
    assert h2h_job.charged_value_gradient_calls(counts) == 3
    assert paired_score.required_charged_calls({"charged_calls": 3}) == 3


def test_paired_calls_require_the_producer_column():
    with np.testing.assert_raises_regex(ValueError, "charged_calls"):
        paired_score.required_charged_calls(
            {"energy_calls": 7, "grad_calls": 3}
        )
    with np.testing.assert_raises_regex(ValueError, "first-hit trace"):
        paired_score.required_first_hit(Path("job"), None, 5)
