# Copyright (c) anneal developers
"""Unit tests for the no-size-constants r5 part B gate runner and summarizer."""

from __future__ import annotations

import csv
import importlib.util
import itertools
import math
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from scipy.stats import fisher_exact, mannwhitneyu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GATE = load("nsc_r5_run_gate", HERE / "run_gate.py")
SUMM = load("nsc_r5_summarize", HERE / "summarize.py")


def test_type7_matches_definition():
    assert SUMM.type7([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.25) == pytest.approx(2.25)
    assert SUMM.type7([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.5) == pytest.approx(3.5)
    assert SUMM.type7([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.75) == pytest.approx(4.75)


def test_fisher_matches_scipy():
    for a, b, c, d in [(12, 0, 0, 12), (9, 3, 2, 10), (6, 6, 6, 6), (4, 8, 9, 3)]:
        want = float(fisher_exact([[a, b], [c, d]], alternative="two-sided")[1])
        assert SUMM.fisher_two_sided(a, b, c, d) == pytest.approx(want, rel=1e-12)


def test_mann_whitney_matches_scipy_with_ties_and_continuity():
    left = [9.0, 4e6, 4e6, 120.0, 55.0, 4e6, 7.0, 3.0, 7.0]
    right = [4e6, 4e6, 4e6, 900.0, 4e6, 4e6, 4e6, 4e6, 55.0]
    want = mannwhitneyu(left, right, alternative="two-sided", method="asymptotic", use_continuity=True)
    u, p, ra, rb = SUMM.mann_whitney(left, right)
    assert u == pytest.approx(float(want.statistic), rel=1e-12)
    assert p == pytest.approx(float(want.pvalue), rel=1e-12)
    assert ra < rb
    assert SUMM.mann_whitney([4e6] * 12, [4e6] * 12)[:2] == (72.0, 1.0)


def test_stratified_fisher_and_van_elteren():
    assert SUMM.stratified_fisher([(9, 3, 2, 10)]) == pytest.approx(
        SUMM.fisher_two_sided(9, 3, 2, 10), rel=1e-12
    )
    strata = [(3, 0, 0, 3), (2, 1, 1, 2)]
    outcomes = [[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]]
    mass: dict[int, float] = {}
    shares = [list(itertools.combinations(range(6), 3)) for _ in strata]
    total_ways = len(shares[0]) * len(shares[1])
    for first in shares[0]:
        for second in shares[1]:
            s = sum(outcomes[0][i] for i in first) + sum(outcomes[1][i] for i in second)
            mass[s] = mass.get(s, 0.0) + 1.0 / total_ways
    obs = 3 + 2
    expected = sum(m for m in mass.values() if m <= mass[obs] * (1.0 + 1e-9))
    assert SUMM.stratified_fisher(strata) == pytest.approx(expected, rel=1e-12)

    ve_strata = [([1.0, 2.0], [3.0, 4.0]), ([1.0, 3.0], [2.0, 4.0])]
    p, z = SUMM.van_elteren(ve_strata)
    assert z == pytest.approx(-0.6 / math.sqrt(2.0 / 15.0), rel=1e-12)
    assert p == pytest.approx(0.10034824646229074, rel=1e-9)
    assert SUMM.van_elteren([([4e6] * 12, [4e6] * 12)]) == (1.0, 0.0)


def make_row(
    panel: str,
    system: str,
    arm: str,
    seed: int,
    slot: int,
    hit: int,
    calls: int,
    wall: float = 2.0,
    pot: float = 1.2,
    status: str = "ok",
    trace_sha: str = "",
) -> dict[str, str]:
    outside = max(wall - pot, 0.0)
    return {
        "session_id": "20261120T000000Z",
        "run_seq": "1",
        "panel": panel,
        "system": system,
        "driver": "nsc-r5-gate-driver",
        "arm": arm,
        "seed": str(seed),
        "slot": str(slot),
        "arm_order": "AB" if seed % 2 == 0 else "BA",
        "budget": "1000",
        "reference": "-100.0",
        "hit": str(hit),
        "evals_to_hit": str(calls),
        "charged": str(calls),
        "best": "-100.0",
        "best_bits_hex": "0xc059000000000000",
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
        "outside_per_call_s": f"{outside / max(calls, 1):.12f}",
        "status": status,
        "library": (
            "commit:61fa67cf380e46876b569da45e7cc091647a775b"
            if arm == "A"
            else "tree:64b12739c8707669e6ae203d8f8b99a160609b55"
        ),
        "library_tree": (
            "cff796c507a8fa1d0bfb28888c3570b61988055e"
            if arm == "A"
            else "64b12739c8707669e6ae203d8f8b99a160609b55"
        ),
        "harness_blobs": "bench/nsc-r5-gate-b/run_gate.py:0000000000000000000000000000000000000000",
        "binary_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "trace_sha256": trace_sha,
        "xyz_sha256": trace_sha,
        "gmin_sha256": trace_sha,
        "start_sha256": trace_sha,
        "machine_id_sha256": "deadbeef",
        "utc": "2026-11-20T00:00:00Z",
    }


def test_abba_balance_verdict_rules_and_hit_flips():
    rows = []
    for idx, seed in enumerate(GATE.SEEDS):
        order = ("A", "B") if idx % 2 == 0 else ("B", "A")
        for slot, arm in enumerate(order, start=1):
            rows.append(make_row("BLJ", "blj105n38", arm, seed, slot, 1, 100 + idx))
    counts = SUMM.verify_abba_balance(rows)
    assert counts == {(1, "A"): 6, (1, "B"): 6, (2, "A"): 6, (2, "B"): 6}
    meta = {"driver": "nsc-r5-gate-driver", "budget": "1000", "reference": "-100.0"}
    sr = SUMM.system_summary("BLJ", "blj105n38", rows, meta)
    assert sr["verdict"] == "PASS"
    assert float(sr["outside_per_call_us_a"]) > 0.0

    # Any hit drop (even 1 seed) -> FIX
    bad_hits = []
    for idx, seed in enumerate(GATE.SEEDS):
        bad_hits.append(make_row("BLJ", "blj105n38", "A", seed, 1, 1, 100))
        bad_hits.append(make_row("BLJ", "blj105n38", "B", seed, 2, int(idx < 11), 100 if idx < 11 else 1000))
    sr_bad_hit = SUMM.system_summary("BLJ", "blj105n38", bad_hits, meta)
    assert sr_bad_hit["loss_hits"] == "1"
    assert sr_bad_hit["verdict"] == "FIX"
    flips = SUMM.build_hit_flips(bad_hits)
    assert len(flips) == 1
    assert flips[0]["direction"] == "A_hit_B_miss"
    assert flips[0]["seed"] == "2011"

    # Loss on calls at p<0.05 -> FIX
    bad_calls = []
    for idx, seed in enumerate(GATE.SEEDS):
        bad_calls.append(make_row("BLJ", "blj105n38", "A", seed, 1, 1, 100 + idx))
        bad_calls.append(make_row("BLJ", "blj105n38", "B", seed, 2, 1, 500 + idx))
    assert SUMM.system_summary("BLJ", "blj105n38", bad_calls, meta)["verdict"] == "FIX"

    # Outside-per-call ratio > 1.10 -> FIX
    bad_outside = []
    for idx, seed in enumerate(GATE.SEEDS):
        bad_outside.append(make_row("BLJ", "blj105n38", "A", seed, 1, 1, 100, wall=2.0, pot=1.0))
        bad_outside.append(make_row("BLJ", "blj105n38", "B", seed, 2, 1, 100, wall=2.2, pot=1.0))
    sr_bad_out = SUMM.system_summary("BLJ", "blj105n38", bad_outside, meta)
    assert sr_bad_out["loss_outside"] == "1"
    assert sr_bad_out["verdict"] == "FIX"

    # Gain on calls (with identical outside-per-call) -> PASS
    good_calls = []
    for idx, seed in enumerate(GATE.SEEDS):
        c_a = 500 + idx
        c_b = 100 + idx
        good_calls.append(make_row("BLJ", "blj105n38", "A", seed, 1, 1, c_a, wall=0.002 * c_a, pot=0.001 * c_a))
        good_calls.append(make_row("BLJ", "blj105n38", "B", seed, 2, 1, c_b, wall=0.002 * c_b, pot=0.001 * c_b))
    assert SUMM.system_summary("BLJ", "blj105n38", good_calls, meta)["verdict"] == "PASS"


def test_cell_set_exclusions_and_coverage():
    refs = GATE.read_tip4p_references(ROOT)
    assert tuple(sorted(refs)) == tuple(range(8, 22))
    panels = GATE.build_panel_list(tuple(sorted(refs)))
    systems = [s for s, _ in panels]
    for sc in ("sc126-38", "sc108-38", "sc96-38", "sc126-55", "sc108-55", "sc96-55", "sc126-80", "sc108-80", "sc96-80"):
        assert sc in systems
    assert set(GATE.BLJ_SYSTEMS) <= set(systems)
    for forbidden in ("lj38", "lj75", "lj98", "m75r14", "m80r6", "m80r14", "au38", "au55", "gfn2"):
        assert forbidden not in systems


def write_stub_driver(path: Path, tag: str) -> None:
    path.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  budget) printf '400000\\n-177.2090846748\\n' ;;\n"
        "  run)\n"
        '    echo "h2h_budget=${H2H_BUDGET:-unset} omp=$OMP_NUM_THREADS tag=' + tag + '"\n'
        "    echo system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s\n"
        '    echo "$2,anneal,$3,400000,true,1234,1234,1.0,-177.2090846748,0.4,0.6" ;;\n'
        "  tip4p)\n"
        '    prefix="$8"\n'
        '    printf "calls,energy\\n1,-10.0\\n" > "${prefix}.trace.csv"\n'
        '    printf "O 0 0 0\\n" > "${prefix}.xyz"\n'
        '    printf "0 0 0 0 0 0\\n" > "${prefix}.gmin"\n'
        '    printf "0 0 0 0 0 0\\n" > "${prefix}.start.gmin"\n'
        '    echo "h2h_row system=tip4p$2 code=anneal seed=$5 budget=$4 hit=1 evals_to_hit=55 '
        'charged=55 best=-305.5180000000 wall_s=0.1 pot_s=0.04 outside_s=0.06 basins=2" ;;\n'
        "esac\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.mark.skipif(
    shutil.which("taskset") is None or not Path("/etc/machine-id").is_file(),
    reason="needs taskset and /etc/machine-id",
)
def test_runner_abba_scrubs_budget_and_summarizes(tmp_path, monkeypatch):
    for tag in ("A", "B"):
        d = tmp_path / f"bin{tag}"
        d.mkdir()
        for name in set(GATE.DRIVERS.values()):
            write_stub_driver(d / name, tag)
    monkeypatch.setenv("H2H_BUDGET", "123")
    rows = tmp_path / "rows.csv"
    panels = tmp_path / "panels.csv"
    sessions = tmp_path / "sessions.csv"
    bin_sha = tmp_path / "bin_sha256.csv"
    summary = tmp_path / "summary.csv"
    panel_summary = tmp_path / "panel_summary.csv"
    hit_flips = tmp_path / "hit_flips.csv"
    tip4p_id = tmp_path / "tip4p_identity.csv"
    readme = tmp_path / "README.md"

    cmd = [
        "taskset", "-c", "0",
        sys.executable, str(HERE / "run_gate.py"), "run",
        "--repo", str(ROOT),
        "--arm-a-bin", str(tmp_path / "binA"),
        "--arm-b-bin", str(tmp_path / "binB"),
        "--arm-a-tree", str(ROOT),
        "--arm-b-tree", str(ROOT),
        "--arm-a-library", "commit:61fa67cf380e46876b569da45e7cc091647a775b",
        "--arm-b-library", "tree:64b12739c8707669e6ae203d8f8b99a160609b55",
        "--arm-a-tree-sha", "cff796c507a8fa1d0bfb28888c3570b61988055e",
        "--arm-b-tree-sha", "64b12739c8707669e6ae203d8f8b99a160609b55",
        "--work", str(tmp_path / "work"),
        "--logs", str(tmp_path / "logs"),
        "--rows", str(rows),
        "--panels", str(panels),
        "--sessions", str(sessions),
        "--bin-sha256", str(bin_sha),
        "--only", "blj105n38", "tip4p8",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    log_text = (tmp_path / "logs" / "blj105n38-A-s2000.log").read_text()
    assert "h2h_budget=unset omp=1 tag=A" in log_text

    recorded = GATE.read_csv(rows)
    assert len(recorded) == 2 * 2 * 12
    assert all(r["binary_sha256"] for r in recorded)
    assert SUMM.verify_abba_balance(recorded) == {(1, "A"): 12, (1, "B"): 12, (2, "A"): 12, (2, "B"): 12}

    quick = [*cmd, "--quick"]
    assert subprocess.run(quick, check=False, capture_output=True).returncode != 0

    subprocess.run(
        [
            sys.executable, str(HERE / "summarize.py"),
            "--rows", str(rows),
            "--panels", str(panels),
            "--summary", str(summary),
            "--panel-summary", str(panel_summary),
            "--hit-flips", str(hit_flips),
            "--tip4p-identity", str(tip4p_id),
            "--readme", str(readme),
            "--sessions", str(sessions),
            "--bin-sha256", str(bin_sha),
        ],
        check=True,
        capture_output=True,
    )
    tip_rows = list(csv.DictReader(tip4p_id.open()))
    assert len(tip_rows) == 12
    assert all(r["all_bit_identical"] == "1" for r in tip_rows)
    assert "Hit Status Changes" in readme.read_text()
    assert "TIP4P bit-identity" in readme.read_text()
