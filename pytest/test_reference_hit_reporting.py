# Copyright (c) anneal developers
"""Benchmark rows cannot claim hits away from their declared reference."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "no_motif_gate", ROOT / "bench" / "no-motif-soap-gate" / "run_gate.py"
)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def row(best: str) -> str:
    return (
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s\n"
        f"au38,anneal,2002,400000,true,3000,3000,1.0,{best},0.8\n"
    )


def test_non_finite_best_is_not_an_ok_hit() -> None:
    with pytest.raises(RuntimeError, match="non-finite best"):
        GATE.parse_row(row("inf"), "au38", 2002, 400_000, -130.54001708)


def test_finite_off_reference_best_is_not_a_hit() -> None:
    with pytest.raises(RuntimeError, match="against reference"):
        GATE.parse_row(row("-130.3201"), "au38", 2002, 400_000, -130.54001708)
