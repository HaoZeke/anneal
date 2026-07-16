"""Bind shipped temperature-law constants to the closed-form design lab."""

from pathlib import Path

from proofs.gpmd_derive import (
    ALPHA_TOL,
    SHIP_ALPHA_TARGET,
    THETA_STAR,
    algorithm_constants,
    read_shipped_rust_constants,
)


def test_lab_emits_closed_form_operating_constants():
    c = algorithm_constants()
    assert c["THETA_STAR"] == THETA_STAR
    assert 0.0 < c["THETA_STAR"] < 2.0
    assert abs(c["ALPHA_TARGET_LAB"] - c["ALPHA_TARGET_SHIP"]) <= c["ALPHA_TOL"]
    assert c["RESIDUAL_RATE"] > 0.85
    assert 1.5 < c["THETA_C"] < 2.5
    assert c["ALPHA_TARGET_SHIP"] == SHIP_ALPHA_TARGET


def test_shipped_rust_matches_lab_design_constants():
    rs = Path(__file__).resolve().parents[2] / "src" / "methods" / "gpmd.rs"
    assert rs.is_file(), f"missing shipped source {rs}"
    shipped = read_shipped_rust_constants(rs)
    lab = algorithm_constants()
    assert shipped["THETA_STAR"] == lab["THETA_STAR"]
    assert shipped["ALPHA_TARGET"] == lab["ALPHA_TARGET_SHIP"]
    assert abs(lab["ALPHA_TARGET_LAB"] - shipped["ALPHA_TARGET"]) <= ALPHA_TOL
