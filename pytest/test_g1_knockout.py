# Copyright (c) anneal developers
"""Tests for the G1 single-knockout ablation driver, schedule, and statistics."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (
    REPO_ROOT / "scripts",
    REPO_ROOT / "scripts" / "gmin_panel",
    REPO_ROOT / "scripts" / "g1_knockout",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_g1  # noqa: E402
from run_panel import PAPER_BUDGETS  # noqa: E402
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402


def _g1_binary() -> Path:
    candidates = [
        REPO_ROOT / ".." / "tmp" / "anneal-target" / "release" / "examples" / "g1_knockout",
        REPO_ROOT / ".." / "tmp" / "anneal-target" / "debug" / "examples" / "g1_knockout",
        REPO_ROOT / "target" / "release" / "examples" / "g1_knockout",
        REPO_ROOT / "target" / "debug" / "examples" / "g1_knockout",
    ]
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_file():
            return resolved
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "--example", "g1_knockout"],
        cwd=REPO_ROOT,
        check=True,
    )
    resolved = candidates[0].resolve()
    assert resolved.is_file(), f"missing compiled binary at {resolved}"
    return resolved


@pytest.fixture(scope="module")
def g1_binary() -> Path:
    return _g1_binary()


def test_panel_systems_and_paper_budgets_match_binary(g1_binary: Path) -> None:
    proc = subprocess.run(
        [str(g1_binary), "systems"],
        env=run_g1.child_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    bin_systems = {item["name"]: item for item in json.loads(proc.stdout)}
    specs = run_g1.panel_specs()
    assert tuple(bin_systems.keys()) == run_g1.SYSTEM_ORDER
    for name, spec in specs.items():
        item = bin_systems[name]
        assert item["n"] == spec["n"]
        assert item["budget"] == PAPER_BUDGETS[name] == spec["budget"]
        assert abs(item["reference"] - spec["reference"]) < 1e-12
        if spec["n"] <= 38:
            assert spec["budget"] == 400_000
        elif spec["n"] < 75:
            assert spec["budget"] == 1_000_000
        else:
            assert spec["budget"] == 4_000_000


def test_every_arm_differs_from_recommended_in_exact_declared_fields(g1_binary: Path) -> None:
    specs = run_g1.panel_specs()
    for system, spec in specs.items():
        base = run_g1.fetch_config_json(g1_binary, system, "recommended")
        base_hash = run_g1.fetch_config_hash(g1_binary, system, "recommended")
        for arm in run_g1.ARMS:
            cfg = run_g1.fetch_config_json(g1_binary, system, arm)
            cfg_hash = run_g1.fetch_config_hash(g1_binary, system, arm)
            all_keys = set(base) | set(cfg)
            diff = frozenset(k for k in all_keys if base.get(k) != cfg.get(k))
            expected = run_g1.expected_diff_fields(spec["n"], arm)
            assert diff == expected, (
                f"{system} (n={spec['n']}) arm={arm}: diff={sorted(diff)} != expected={sorted(expected)}"
            )
            if expected:
                assert cfg_hash != base_hash
            else:
                assert cfg_hash == base_hash


def test_expected_knockout_field_values(g1_binary: Path) -> None:
    # Check exact field values for n=55 (has surfaces) and n=98 (no surfaces).
    for system, has_surface in (("lj55", True), ("lj98", False)):
        rec = run_g1.fetch_config_json(g1_binary, system, "recommended")
        assert rec["restart_on_stall"] is False
        assert rec["escape_on_stall"] is True
        assert rec["trail_on_stall"] is True
        assert rec["tabu_on_stall"] is True
        assert rec["orbit_complete_on_new"] is True
        assert rec["continuous_symmetry"] == {"group": "measure", "interval": 10}
        assert rec["symmetrise_on_stall"] is True
        assert rec["return_screen"] is True
        assert rec["bayes_screen"] is True
        assert bool(rec["surfaces"]) is has_surface
        assert rec["energy_lattice"] is True
        assert rec.get("prior_soap", True) is True
        assert "prior_soap" not in rec
        assert rec["bias_height"] > 0.0

        arm_a = run_g1.fetch_config_json(g1_binary, system, "a_restart_on_stall")
        assert arm_a["restart_on_stall"] is True
        assert arm_a["escape_on_stall"] is False
        assert arm_a["trail_on_stall"] is False
        assert arm_a["tabu_on_stall"] is False

        arm_b = run_g1.fetch_config_json(g1_binary, system, "b_no_orbit")
        assert arm_b["orbit_complete_on_new"] is False

        arm_c = run_g1.fetch_config_json(g1_binary, system, "c_no_symmetry")
        assert arm_c["continuous_symmetry"] == {"group": "off"}
        assert arm_c["symmetrise_on_stall"] is False

        arm_d = run_g1.fetch_config_json(g1_binary, system, "d_no_screens")
        assert arm_d["return_screen"] is False
        assert arm_d["bayes_screen"] is False

        arm_e = run_g1.fetch_config_json(g1_binary, system, "e_no_surfaces")
        assert arm_e["surfaces"] == []

        arm_f = run_g1.fetch_config_json(g1_binary, system, "f_no_energy_lattice")
        assert arm_f["energy_lattice"] is False

        arm_g = run_g1.fetch_config_json(g1_binary, system, "g_no_priors")
        assert arm_g["prior_soap"] is False

        arm_h = run_g1.fetch_config_json(g1_binary, system, "h_no_basin_bias")
        assert arm_h["bias_height"] == 0.0


def test_abba_rotation_balance() -> None:
    schedule = run_g1.build_schedule()
    assert len(schedule) == 9 * 12 * 9 == 972
    assert [row["sequence"] for row in schedule] == list(range(1, 973))

    # Every (system, seed) visits all 9 arms once, and odd seeds reverse the preceding even seed.
    by_sys_seed: dict[tuple[str, int], list[str]] = {}
    for item in schedule:
        by_sys_seed.setdefault((item["system"], item["seed"]), []).append(item["arm"])
    for system in run_g1.SYSTEM_ORDER:
        for i in range(0, len(run_g1.SEEDS), 2):
            even_order = by_sys_seed[(system, run_g1.SEEDS[i])]
            odd_order = by_sys_seed[(system, run_g1.SEEDS[i + 1])]
            assert set(even_order) == set(run_g1.ARMS)
            assert odd_order == list(reversed(even_order))

        # Exact mean slot == 4.0 per arm within each system.
        for arm in run_g1.ARMS:
            slots = [
                item["slot"]
                for item in schedule
                if item["system"] == system and item["arm"] == arm
            ]
            assert len(slots) == 12
            assert sum(slots) / len(slots) == 4.0

    # Globally across the 9 systems, every (arm, slot) pair appears exactly 12 times.
    counts = Counter((item["arm"], item["slot"]) for item in schedule)
    assert len(counts) == 81
    assert all(count == 12 for count in counts.values())


def test_child_env_strips_h2h_and_pins_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("H2H_BUDGET", "123")
    monkeypatch.setenv("H2H_PRIORS", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    env = run_g1.child_env()
    assert all(not k.startswith("H2H_") for k in env)
    for key, expected in run_g1.THREAD_ENV_VARS.items():
        assert env[key] == expected


def test_statistics_match_scipy_reference() -> None:
    # Fisher exact two-sided
    for a_hit, b_hit in ((12, 12), (12, 7), (10, 2), (4, 0), (6, 6), (9, 3)):
        ours = run_g1.fisher_exact_two_sided(a_hit, 12, b_hit, 12)
        ref = scipy_stats.fisher_exact(
            [[a_hit, 12 - a_hit], [b_hit, 12 - b_hit]], alternative="two-sided"
        ).pvalue
        assert math.isclose(ours, ref, rel_tol=1e-9, abs_tol=1e-12)

    # Log-rank two-sided against scipy.stats.logrank (Mantel-Haenszel)
    t_a = [100, 200, 350, 500, 1000, 1000]
    e_a = [1, 1, 1, 1, 0, 0]
    t_b = [150, 200, 600, 1000, 1000, 1000]
    e_b = [1, 1, 1, 0, 0, 0]
    chi2, p_lr = run_g1.logrank_two_sided(t_a, e_a, t_b, e_b)
    sc_lr = scipy_stats.logrank(
        scipy_stats.CensoredData.right_censored(t_a, [e == 0 for e in e_a]),
        scipy_stats.CensoredData.right_censored(t_b, [e == 0 for e in e_b]),
    )
    assert math.isclose(p_lr, sc_lr.pvalue, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(chi2, sc_lr.statistic**2, rel_tol=1e-9, abs_tol=1e-12)

    # Kaplan-Meier survival values
    curve = run_g1.kaplan_meier(t_a, e_a)
    assert curve[0]["calls"] == 0 and curve[0]["survival"] == 1.0
    assert math.isclose(curve[-1]["survival"], 2.0 / 6.0, rel_tol=1e-12)
    assert run_g1.km_median(curve) == 350.0

    # Mann-Whitney with misses ranked +inf matches scipy.stats.mannwhitneyu (asymptotic, cc=True)
    x = [100.0, 200.0, 350.0, 500.0, math.inf, math.inf]
    y = [150.0, 200.0, 600.0, math.inf, math.inf, math.inf]
    u_val, p_mw = run_g1.mann_whitney_two_sided(x, y)
    # Replace +inf with a finite sentinel above all hits for scipy compatibility
    x_fin = [v if math.isfinite(v) else 1e9 for v in x]
    y_fin = [v if math.isfinite(v) else 1e9 for v in y]
    sc_mw = scipy_stats.mannwhitneyu(
        x_fin, y_fin, alternative="two-sided", method="asymptotic", use_continuity=True
    )
    assert math.isclose(u_val, sc_mw.statistic, rel_tol=1e-12)
    assert math.isclose(p_mw, sc_mw.pvalue, rel_tol=1e-9, abs_tol=1e-12)


def test_error_and_timeout_rows_stay_in_as_misses(tmp_path: Path) -> None:
    specs = run_g1.panel_specs()
    ref = specs["lj38"]["reference"]
    budget = specs["lj38"]["budget"]
    assert energy_hits(ref, ref, DEFAULT_HIT_TOL)

    # Fake binary that exits 1 on error and sleeps past timeout on timeout
    err_bin = tmp_path / "fake_err"
    err_bin.write_text("#!/bin/sh\necho boom >&2\nexit 1\n", encoding="utf-8")
    err_bin.chmod(0o755)
    row_err = run_g1.run_one(
        binary=err_bin,
        system="lj38",
        arm="recommended",
        seed=2000,
        slot=0,
        sequence=1,
        budget=budget,
        reference=ref,
        core=0,
        machine_id="a" * 64,
        binary_sha256="b" * 64,
        config_hash="c" * 64,
        session="20261120T000000Z",
        log_dir=tmp_path / "logs",
        timeout_s=5.0,
    )
    assert row_err["status"] == "error"
    assert row_err["hit"] == "0"
    assert int(row_err["calls"]) == budget

    to_bin = tmp_path / "fake_to"
    to_bin.write_text("#!/bin/sh\nsleep 2\n", encoding="utf-8")
    to_bin.chmod(0o755)
    row_to = run_g1.run_one(
        binary=to_bin,
        system="lj38",
        arm="b_no_orbit",
        seed=2000,
        slot=1,
        sequence=2,
        budget=budget,
        reference=ref,
        core=0,
        machine_id="a" * 64,
        binary_sha256="b" * 64,
        config_hash="d" * 64,
        session="20261120T000000Z",
        log_dir=tmp_path / "logs",
        timeout_s=0.1,
    )
    assert row_to["status"] == "timeout"
    assert row_to["hit"] == "0"
    assert int(row_to["calls"]) == budget

    surv, summ = run_g1.summarize_rows([row_err, row_to], specs=specs)
    assert len(surv) >= 4
    by_arm = {r["arm"]: r for r in summ}
    assert by_arm["recommended"]["hits"] == "0"
    assert by_arm["recommended"]["errors"] == "1"
    assert by_arm["b_no_orbit"]["hits"] == "0"
    assert by_arm["b_no_orbit"]["timeouts"] == "1"


def test_hashed_machine_id_is_sha256_and_never_raw_hostname() -> None:
    mid = run_g1.hashed_machine_id()
    assert len(mid) == 64
    assert all(c in "0123456789abcdef" for c in mid)
    assert os.uname().nodename not in mid
