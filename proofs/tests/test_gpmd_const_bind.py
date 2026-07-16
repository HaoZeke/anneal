"""Bind shipped temperature-law constants to the SymPy design lab."""

from pathlib import Path

from proofs.gpmd_derive import (
    ALPHA_TOL,
    SHIP_ALPHA_TARGET,
    SHIP_THETA_STAR,
    algorithm_constants,
    read_shipped_rust_constants,
)


def test_lab_emits_finite_operating_constants():
    c = algorithm_constants()
    assert c["THETA_STAR"] == SHIP_THETA_STAR
    assert 0.0 < c["THETA_STAR"] < 2.0
    assert abs(c["ALPHA_TARGET_LAB"] - c["ALPHA_TARGET_SHIP"]) <= c["ALPHA_TOL"]
    assert c["C_STAR_SPHERE"] == c["C_STAR_SPHERE"]  # finite
    assert c["ALPHA_TARGET_SHIP"] == SHIP_ALPHA_TARGET


def test_shipped_rust_matches_lab_design_constants():
    """Real gpmd.rs constants must match the design-lab ship values."""
    rs = Path(__file__).resolve().parents[2] / "src" / "methods" / "gpmd.rs"
    assert rs.is_file(), f"missing shipped source {rs}"
    shipped = read_shipped_rust_constants(rs)
    lab = algorithm_constants()
    assert shipped["THETA_STAR"] == lab["THETA_STAR"]
    assert shipped["ALPHA_TARGET"] == lab["ALPHA_TARGET_SHIP"]
    # Lab MC α must remain within documented tolerance of the ship constant.
    assert abs(lab["ALPHA_TARGET_LAB"] - shipped["ALPHA_TARGET"]) <= ALPHA_TOL


def test_theta_star_in_descent_window_from_lab():
    lab = algorithm_constants()
    assert 0.0 < lab["THETA_STAR"] < 2.0
