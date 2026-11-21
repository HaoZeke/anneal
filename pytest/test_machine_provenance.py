# Copyright (c) anneal developers
"""Shared benchmark machine-provenance helpers."""

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from machine_provenance import (  # noqa: E402
    aggregate_machine_columns,
    csv_fields,
    require_csv_schema,
)

ACTIVE_ROW_WRITERS = (
    "scripts/ase_nonlj/run_one.py",
    "scripts/ase_nonlj/summarize.py",
    "scripts/ase_panel/run_one.py",
    "scripts/ase_panel/summarize.py",
    "scripts/covalent_h2h/gfn2_crest.py",
    "scripts/covalent_h2h/gfn2_floor_gate.py",
    "scripts/covalent_h2h/reference.py",
    "scripts/covalent_h2h/run_campaign.py",
    "scripts/generic_alloy/run_ablation.py",
    "scripts/gmin_h2h/run_campaign.py",
    "scripts/growpal_h2h/run_campaign.py",
    "scripts/growpal_h2h_v2/run_campaign.py",
    "scripts/lj_bonobo_h2h/run_campaign.py",
    "scripts/metal_h2h/run_campaign.py",
    "scripts/metal_h2h/enter_gold.sh",
    "scripts/metal_h2h/run_darby.py",
    "scripts/metal_h2h/run_ptcu.py",
    "scripts/metal_h2h/run_sc.py",
    "scripts/metal_h2h/run_sc96_gmin.py",
    "scripts/metal_h2h/run_sc_n12.py",
    "scripts/metal_h2h/run_sc_rows.py",
    "scripts/metal_h2h/summarise_sc_chain.py",
    "scripts/metal_h2h/summarise_sc_pop.py",
    "scripts/metal_h2h/summarise_sc_portfolio.py",
    "scripts/metal_h2h/summarise_sc_restart.py",
    "scripts/occat_h2h/run_campaign.py",
    "scripts/tip4p_h2h/run_budget.py",
    "scripts/tip4p_h2h/run_h2h.py",
    "scripts/tip4p_h2h/run_water_gmin.py",
)


def test_require_csv_schema_refuses_legacy_header(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("host\nbench\n")

    with pytest.raises(RuntimeError, match="legacy columns"):
        require_csv_schema(path, ("host", "machine_id", "cpu_model"))

    assert path.read_text() == "host\nbench\n"


def test_require_csv_schema_accepts_current_header(tmp_path):
    path = tmp_path / "rows.csv"
    fields = ("host", "machine_id", "cpu_model")
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerow(fields)

    require_csv_schema(path, fields)


def test_aggregate_machine_columns_preserves_legacy_groups():
    assert aggregate_machine_columns([{"host": "bench"}]) == {}


def test_aggregate_machine_columns_rejects_mixed_groups():
    with pytest.raises(RuntimeError, match="cannot mix"):
        aggregate_machine_columns(
            [
                {"host": "bench"},
                {"machine_id": "a" * 64, "cpu_model": "Example CPU"},
            ]
        )


def test_csv_fields_quotes_cpu_models_with_commas():
    assert csv_fields(("a" * 64, "Example CPU, revision 2")) == (
        "a" * 64 + ',"Example CPU, revision 2"'
    )


@pytest.mark.parametrize("relative", ACTIVE_ROW_WRITERS)
def test_active_row_writers_include_shared_machine_provenance(relative):
    source = (ROOT / relative).read_text()

    assert "machine_provenance" in source
    assert any(
        helper in source
        for helper in (
            "machine_columns",
            "machine_provenance(",
            "aggregate_machine_columns",
            "csv_fields",
        )
    )
