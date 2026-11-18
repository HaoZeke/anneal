# Copyright (c) anneal developers
"""Edge-case tests for the shared one-sided hit predicate ``E <= target + tol``."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import anneal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import target_tol as py_target_tol  # noqa: E402


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "fn,default_tol",
    [
        (py_target_tol.energy_hits, py_target_tol.DEFAULT_HIT_TOL),
        (anneal.energy_hits, anneal.DEFAULT_HIT_TOL),
    ],
)
def test_shared_helpers_handle_boundary_below_target_and_nonfinite(fn, default_tol):
    assert default_tol == 1.0e-3

    target = -397.492331
    tol = 1.0e-3
    boundary = target + tol

    # Exact upper boundary (with explicit tol, default tol, and zero tol).
    assert fn(boundary, target, tol) is True
    assert fn(target + default_tol, target) is True
    assert fn(target, target, 0.0) is True

    # Strictly above target + tol by a single ULP is a miss.
    assert fn(math.nextafter(boundary, math.inf), target, tol) is False
    assert fn(math.nextafter(target, math.inf), target, 0.0) is False

    # Arbitrarily below target is a hit (one-sided, never |E - target| <= tol).
    assert fn(target - 1.0e-9, target, tol) is True
    assert fn(target - 10.0 * tol, target, tol) is True
    assert fn(-1.0e9, target, tol) is True

    # NaN in any argument (Python float or NumPy scalar) is never a hit.
    for nan_val in (float("nan"), float(np.nan)):
        assert fn(nan_val, target, tol) is False
        assert fn(nan_val, target) is False
        assert fn(target, nan_val, tol) is False
        assert fn(target, target, nan_val) is False
        assert fn(nan_val, nan_val, nan_val) is False

    # Infinite inputs and negative tolerances are rejected.
    for inf_val in (math.inf, -math.inf):
        assert fn(inf_val, target, tol) is False
        assert fn(target, inf_val, tol) is False
        assert fn(target, target, inf_val) is False
    assert fn(target - 10.0, target, -1.0e-9) is False


def test_python_and_rust_helpers_agree_bit_for_bit():
    target = -173.928427
    tol = py_target_tol.DEFAULT_HIT_TOL
    boundary = target + tol
    cases = [
        (boundary, target, tol),
        (math.nextafter(boundary, math.inf), target, tol),
        (math.nextafter(boundary, -math.inf), target, tol),
        (target - 50.0, target, tol),
        (target, target, 0.0),
        (math.nextafter(target, math.inf), target, 0.0),
        (float("nan"), target, tol),
        (target, float("nan"), tol),
        (target, target, float("nan")),
        (-math.inf, target, tol),
        (math.inf, target, tol),
        (target, math.inf, tol),
        (target, target, math.inf),
        (target - 1.0, target, -1.0e-6),
    ]
    for e, t, w in cases:
        assert py_target_tol.energy_hits(e, t, w) is anneal.energy_hits(e, t, w)


def test_selector_fisher_one_sided_and_nan():
    mod = _load_module("test_selector_fisher_mod", "paper/jctc/scripts/selector_fisher.py")
    ref = mod.REFERENCE[38]
    boundary = ref + mod.TOL
    assert mod.check_hit(38, "1", repr(boundary)) == 1
    assert mod.check_hit(38, "1", repr(ref - 5.0)) == 1
    assert mod.check_hit(38, "0", repr(math.nextafter(boundary, math.inf))) == 0
    assert mod.check_hit(38, "0", "nan") == 0
    with pytest.raises(SystemExit):
        mod.check_hit(38, "1", "nan")
    with pytest.raises(SystemExit):
        mod.check_hit(38, "0", repr(boundary))


def test_hq_sci_tally_boundary_below_and_nan(tmp_path: Path):
    mod = _load_module("test_hq_sci_tally_mod", "paper/jctc/scripts/hq_sci_tally.py")
    hexamer = tmp_path / "hex"
    copper = tmp_path / "cu"
    hexamer.mkdir()
    copper.mkdir()

    # Hexamer: floor -3.0, exact floor + 0.01 hits, nan is None/False.
    hex_tokens = [
        "-3.000000",
        repr(-3.0 + mod.HIT_ABOVE_EV),
        repr(math.nextafter(-3.0 + mod.HIT_ABOVE_EV, math.inf)),
        "nan",
        "inf",
        "-2.500000",
        "-3.000000",
        "-3.000000",
    ]
    for idx, tok in enumerate(hex_tokens):
        (hexamer / f"h2o6_rec_{idx}.out").write_text(f"seed {idx}: best {tok} eV\n")

    # Copper: exact REFERENCE_EV + REFERENCE_TOL_EV, far below REFERENCE_EV,
    # 1 ULP above boundary, and nan.
    boundary = mod.REFERENCE_EV + mod.REFERENCE_TOL_EV
    cu_tokens = [
        repr(boundary),
        repr(mod.REFERENCE_EV - 2.0),
        repr(math.nextafter(boundary, math.inf)),
        "nan",
        "inf",
        repr(mod.REFERENCE_EV),
        repr(mod.REFERENCE_EV - 0.5),
        repr(boundary + 0.1),
    ]
    for idx, tok in enumerate(cu_tokens):
        (copper / f"cuh2_fcc_rec_{idx}.out").write_text(f"seed {idx}: best {tok} eV\n")

    out = mod.tally(hexamer, copper)
    assert "h2o6_rec_1.out best=-2.99 hit=True" in out
    assert "h2o6_rec_2.out" in out and "hit=False" in out
    assert "h2o6_rec_3.out best=None hit=False" in out
    assert f"cuh2_fcc_rec_0.out best={cu_tokens[0]} hit=True" in out
    assert f"cuh2_fcc_rec_1.out best={cu_tokens[1]} hit=True" in out
    assert f"cuh2_fcc_rec_2.out best={cu_tokens[2]} hit=False" in out
    assert "cuh2_fcc_rec_3.out best=None hit=False" in out
    assert "4/8=4/8" in out


def test_gmin_panel_check_run_counts_boundary_below_and_nan():
    mod = _load_module("test_gmin_panel_mod", "scripts/gmin_panel/run_panel.py")
    ref = -397.492331
    boundary = ref + mod.HIT_TOL
    # Exact boundary and deeper-than-reference hits are valid.
    mod.check_run_counts("exact", True, boundary, ref, 100, 100, 1000, 1.0, 0.5)
    mod.check_run_counts("below", True, ref - 5.0, ref, 100, 100, 1000, 1.0, 0.5)
    # 1 ULP above boundary is a miss, not a hit.
    above = math.nextafter(boundary, math.inf)
    mod.check_run_counts("above-miss", False, above, ref, 1000, 1000, 1000, 1.0, 0.5)
    with pytest.raises(RuntimeError):
        mod.check_run_counts("above-hit", True, above, ref, 100, 100, 1000, 1.0, 0.5)
    # NaN cannot be reported as a hit.
    mod.check_run_counts("nan-miss", False, float("nan"), ref, 1000, 1000, 1000, 1.0, 0.5)
    with pytest.raises(RuntimeError):
        mod.check_run_counts("nan-hit", True, float("nan"), ref, 100, 100, 1000, 1.0, 0.5)


def test_compare_bank_ctrl_first_encounter_edge_cases():
    mod = _load_module("test_compare_bank_ctrl_mod", "scripts/compare_bank_ctrl.py")
    target = -50.0
    tol = mod.DEFAULT_HIT_TOL
    boundary = target + tol

    run_boundary = {
        "best": boundary,
        "charged": 200,
        "improvements": [(1, 40, 1, float("nan")), (2, 85, 2, boundary)],
    }
    assert mod.first_encounter(run_boundary, target, tol) == (True, 85)

    run_below = {"best": target - 12.0, "charged": 200, "improvements": []}
    assert mod.first_encounter(run_below, target, tol) == (True, 200)

    run_above = {
        "best": math.nextafter(boundary, math.inf),
        "charged": 200,
        "improvements": [(1, 50, 1, math.nextafter(boundary, math.inf))],
    }
    assert mod.first_encounter(run_above, target, tol) == (False, 200)

    run_nan = {"best": float("nan"), "charged": 200, "improvements": [(1, 10, 1, float("nan"))]}
    assert mod.first_encounter(run_nan, target, tol) == (False, 200)
    assert mod.first_encounter(run_below, float("nan"), tol) == (False, 200)


def test_lj_board_predicate_edge_cases():
    pytest.importorskip("numba")
    mod = _load_module("test_lj_board_mod", "scripts/lj_board/boardlib.py")
    ref = -173.928427
    boundary = ref + mod.HIT_TOL
    assert mod.predicate(boundary, mod.FORCE_BAR, 100, ref, 1000) is True
    assert mod.predicate(ref - 10.0, mod.FORCE_BAR, 100, ref, 1000) is True
    assert mod.predicate(math.nextafter(boundary, math.inf), mod.FORCE_BAR, 100, ref, 1000) is False
    assert mod.predicate(float("nan"), mod.FORCE_BAR, 100, ref, 1000) is False


def test_terra_summarizers_boundary_and_nan(tmp_path: Path):
    mol = tmp_path / "mol"
    (mol / "h2o6_plain").mkdir(parents=True)
    (mol / "h2o6_k0.7").mkdir(parents=True)
    (mol / "cuh2_plain").mkdir(parents=True)
    # Global best is -10.0; plain has -10.0 and nan, k0.7 has exact -10.0 + 1e-3 and above.
    (mol / "h2o6_plain" / "seed_0.out").write_text(
        "seed 0: best -10.0 eV hops 5 charged 100 basins 2\n"
    )
    (mol / "h2o6_plain" / "seed_1.out").write_text(
        "seed 1: best nan eV hops 0 charged 100 basins 0\n"
    )
    (mol / "h2o6_k0.7" / "seed_0.out").write_text(
        f"seed 0: best {repr(-10.0 + 1e-3)} eV hops 4 charged 80 basins 2\n"
    )
    (mol / "h2o6_k0.7" / "seed_1.out").write_text(
        f"seed 1: best {repr(math.nextafter(-10.0 + 1e-3, math.inf))} eV hops 4 charged 90 basins 2\n"
    )
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "terra_summarize_molslab.py"), str(mol)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "6  plain         1/2" in proc.stdout
    assert "6  k0.7          1/2" in proc.stdout

    slab = tmp_path / "slab"
    (slab / "cuh2_h6_plain").mkdir(parents=True)
    (slab / "cuh2_h6_recommended").mkdir(parents=True)
    (slab / "cuh2_h6_plain" / "seed_0.out").write_text(
        "seed 0: best -20.0 eV hops 3 charged 50 basins 1\n"
    )
    (slab / "cuh2_h6_plain" / "seed_1.out").write_text(
        "seed 1: best nan eV hops 0 charged 50 basins 0\n"
    )
    (slab / "cuh2_h6_recommended" / "seed_0.out").write_text(
        f"seed 0: best {repr(-20.0 + 1e-3)} eV hops 4 charged 60 basins 2\n"
    )
    (slab / "cuh2_h6_recommended" / "seed_1.out").write_text(
        f"seed 1: best {repr(math.nextafter(-20.0 + 1e-3, math.inf))} eV hops 4 charged 70 basins 2\n"
    )
    proc_slab = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "terra_summarize_slab_h6.py"), str(slab)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "plain           1/2" in proc_slab.stdout
    assert "recommended     1/2" in proc_slab.stdout


def test_ase_summarizers_reject_nan_hit_rows(tmp_path: Path, monkeypatch):
    ase_panel = _load_module("test_ase_panel_nan_mod", "scripts/ase_panel/summarize.py")
    lj38_dir = tmp_path / "bench" / "ase-lj38"
    lj38_dir.mkdir(parents=True)
    (lj38_dir / "rows.csv").write_text(
        ",".join(ase_panel.ROW_FIELDS)
        + "\n"
        + "lj38,recommended,2000,400000,1,100,100,nan,1.0,0.5,0.5,"
        + "a" * 40
        + ",bench,hit,\n"
    )
    monkeypatch.setattr(ase_panel, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="hit above threshold"):
        ase_panel.load("lj38")

    ase_nonlj = _load_module("test_ase_nonlj_nan_mod", "scripts/ase_nonlj/summarize.py")
    nonlj_dir = tmp_path / "bench" / "ase-nonlj"
    nonlj_dir.mkdir(parents=True)
    (nonlj_dir / "rows.csv").write_text(
        ",".join(ase_nonlj.ROW_FIELDS)
        + "\n"
        + "m38r6,recommended,2000,400000,1,100,100,nan,1.0,0.5,0.5,"
        + ase_nonlj.REVISION
        + ",bench,hit,\n"
    )
    monkeypatch.setattr(ase_nonlj, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="hit above threshold"):
        ase_nonlj.load_rows()
