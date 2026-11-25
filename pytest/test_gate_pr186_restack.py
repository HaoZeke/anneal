# Copyright (c) anneal developers
"""Checks for the PR 186 restack gate's non-timing invariants."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


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

RUN_SPEC = importlib.util.spec_from_file_location(
    "pr186_restack_run",
    ROOT / "bench" / "gate-pr186-restack" / "run_gate.py",
)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
RUN_GATE = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(RUN_GATE)


def rows(
    *,
    path_enabled: bool = False,
    path_draws: int = 0,
    path_spent: int = 0,
    calls_to_hit: int = 40,
    a_path_absent: bool = False,
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
                    "" if a_path_absent and arm != "B"
                    else str(int(path_enabled)) if arm == "B" else "0"
                ),
                "b_path_draws": (
                    "" if a_path_absent and arm != "B"
                    else str(path_draws if arm == "B" else 0)
                ),
                "b_path_spent": (
                    "" if a_path_absent and arm != "B"
                    else str(path_spent if arm == "B" else 0)
                ),
                "trajectory": "same",
                "energy_bits": "",
                "error": "",
            }
        )
    return out


def test_disabled_conf_miss_path_requires_zero_counters():
    pytest.importorskip("scipy.stats")
    result = ANALYZE.judge(rows(path_draws=0, path_spent=0), "rq3")
    assert result["verdict"] == "PASS"
    assert result["b_path_matches"] == 1
    result = ANALYZE.judge(rows(path_draws=1, path_spent=10), "rq3")
    assert result["verdict"] == "FIX"
    assert "conf-miss path" in result["failures"]


def test_enabled_conf_miss_path_requires_nonzero_counters():
    pytest.importorskip("scipy.stats")
    result = ANALYZE.judge(
        rows(path_enabled=True, path_draws=1, path_spent=10),
        "rq3",
    )
    assert result["verdict"] == "PASS"
    assert result["b_path_enabled"] == 1
    result = ANALYZE.judge(rows(path_enabled=True), "rq3")
    assert result["verdict"] == "FIX"
    assert "conf-miss path" in result["failures"]


def test_absent_path_counters_are_not_synthetic_zeros():
    pytest.importorskip("scipy.stats")
    result = ANALYZE.judge(rows(a_path_absent=True), "rq3")
    assert result["verdict"] == "PASS"
    missing = rows()
    for row in missing:
        if row["arm"] == "B":
            row["branch_on_stall"] = ""
            row["b_path_draws"] = ""
            row["b_path_spent"] = ""
    result = ANALYZE.judge(missing, "rq3")
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["b_path_checkable"] == 0


def test_checkable_path_failure_wins_over_unavailable_row():
    pytest.importorskip("scipy.stats")
    mixed = rows() + [{**row, "seed": "2001"} for row in rows()]
    for row in mixed:
        if row["arm"] == "B" and row["seed"] == "2000":
            row["branch_on_stall"] = "1"
        if row["arm"] == "B" and row["seed"] == "2001":
            row["branch_on_stall"] = ""
            row["b_path_draws"] = ""
            row["b_path_spent"] = ""
    result = ANALYZE.judge(mixed, "rq3")
    assert result["verdict"] == "FIX"
    assert result["b_path_checkable"] == 1
    assert result["b_path_failures"] == 1


def test_unavailable_path_is_inconclusive_when_checkable_rows_pass():
    pytest.importorskip("scipy.stats")
    mixed = rows() + [{**row, "seed": "2001"} for row in rows()]
    for row in mixed:
        if row["arm"] == "B" and row["seed"] == "2001":
            row["branch_on_stall"] = ""
            row["b_path_draws"] = ""
            row["b_path_spent"] = ""
    result = ANALYZE.judge(mixed, "rq3")
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["b_path_checkable"] == 1
    assert result["b_path_failures"] == 0


def test_molecular_path_fields_preserve_absence():
    assert RUN_GATE.molecular_path_fields({}, {}) == (None, None, None)
    assert RUN_GATE.molecular_path_fields(
        {
            "branch_on_stall": False,
            "ledger": {"branch_draws": 0, "branch_spent": 0},
        },
        {},
    ) == (False, 0, 0)


def test_csv_writes_absent_path_fields_as_empty(tmp_path: Path):
    record = {field: "" for field in RUN_GATE.ROW_FIELDS}
    record.update(
        cell="rq3",
        seed=2000,
        arm="A",
        branch_on_stall=None,
        b_path_draws=None,
        b_path_spent=None,
    )
    path = tmp_path / "rows.csv"
    RUN_GATE.write_csv(path, [record])
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["branch_on_stall"] == ""
    assert row["b_path_draws"] == ""
    assert row["b_path_spent"] == ""


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
    assert harvest.found == [4]


def test_paired_calls_charge_every_energy_and_gradient_evaluation():
    counts = {"energy_calls": 7, "grad_calls": 3}
    assert h2h_job.charged_value_gradient_calls(counts) == 10
    assert paired_score.required_charged_calls(counts) == 10
    assert paired_score.required_charged_calls(
        {**counts, "charged_calls": 3}
    ) == 10
    assert paired_score.call_budget({"budget": 5}, 10) == 10
    assert paired_score.call_budget({}, 10) == 10


def test_paired_calls_require_raw_counter_columns():
    with np.testing.assert_raises_regex(ValueError, "energy_calls and grad_calls"):
        paired_score.required_charged_calls({"energy_calls": 7})
    with np.testing.assert_raises_regex(ValueError, "first-hit trace"):
        paired_score.required_first_hit(Path("job"), None, 5)
