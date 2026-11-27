# Copyright (c) anneal developers
"""The Sutton-Chen README prints the GMIN pin on GMIN rows."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sc_h2h_summarize", ROOT / "scripts" / "paper_rerun" / "sc_h2h_summarize.py"
)
summarize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize)

PIN = "9cbec1a446352bc9ef064bd2b1a5767194655919"
CODE = "61fa67cf380e46876b569da45e7cc091647a775b"


def _row(arm: str, pin: str = PIN) -> dict[str, str]:
    return {"arm": arm, "code_revision": CODE, "gmin_pin": pin}


def _summary(arm: str) -> dict[str, str]:
    return {
        "system": "sc126-38",
        "arm": arm,
        "hits_over_n": "1/1",
        "calls_median": "1.0",
        "fisher_p_vs_gmin": "1" if arm == "anneal" else "",
        "mannwhitney_p_calls": "1" if arm == "anneal" else "",
    }


def test_gmin_rows_print_the_pin_and_anneal_rows_print_the_code():
    text = summarize.readme(
        [_row("anneal"), _row("gmin")],
        [_summary("anneal"), _summary("gmin")],
    )
    assert "| sc126-38 | anneal | 61fa67cf |" in text
    assert "| sc126-38 | gmin | 9cbec1a4 |" in text
    assert "GMIN rows are pin `9cbec1a4`." in text


def test_disagreeing_gmin_pins_are_not_printed_as_a_code_revision():
    text = summarize.readme(
        [_row("gmin", PIN), _row("gmin", "0" * 40)],
        [_summary("gmin")],
    )
    assert "| sc126-38 | gmin |  |" in text
    assert "| sc126-38 | gmin | 61fa67cf |" not in text


def test_load_refuses_rows_with_carriage_returns(tmp_path):
    rows = tmp_path / "rows.csv"
    rows.write_bytes(b"system,arm,code_revision,gmin_pin\r\nsc126-38,gmin,abc,def\r\n")
    with pytest.raises(SystemExit, match="contains CR"):
        summarize.load(rows)
    rows.write_bytes(b"system,arm,code_revision,gmin_pin\nsc126-38,gmin,abc,def\n")
    assert summarize.load(rows) == [
        {"system": "sc126-38", "arm": "gmin", "code_revision": "abc", "gmin_pin": "def"}
    ]
