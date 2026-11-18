# Copyright (c) anneal developers
"""The two-arm gate runner scores, orders and resumes runs as documented."""

from __future__ import annotations

import csv
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "no_motif_pins_gate" / "run_gate.py"
SPEC = importlib.util.spec_from_file_location("no_motif_pins_gate", SCRIPT)
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)


def test_median_is_hyndman_fan_type_seven():
    assert GATE.type7([1, 2, 3, 4]) == 2.5
    assert GATE.type7([3, 1, 2]) == 2
    assert GATE.type7([5]) == 5


def test_fisher_matches_reference_values():
    assert GATE.fisher_two_sided(12, 0, 0, 12) == pytest.approx(7.39602e-07, rel=1e-5)
    assert GATE.fisher_two_sided(0, 12, 12, 0) == pytest.approx(7.39602e-07, rel=1e-5)
    assert GATE.fisher_two_sided(9, 3, 2, 10) == pytest.approx(0.012278137799742, rel=1e-9)
    assert GATE.fisher_two_sided(6, 6, 6, 6) == pytest.approx(1.0)


def test_mann_whitney_carries_tie_and_continuity_corrections():
    x = [1, 2, 2, 3, 5, 5, 5, 8]
    y = [2, 3, 3, 5, 9, 9, 10, 10, 10]
    p, rank_x, rank_y = GATE.mann_whitney(x, y)
    assert p == pytest.approx(0.0789196355026, rel=1e-9)
    assert rank_x == pytest.approx(6.6875)
    assert rank_y == pytest.approx(11.0555555555)
    swapped, rank_y2, rank_x2 = GATE.mann_whitney(y, x)
    assert swapped == pytest.approx(p)
    assert (rank_x2, rank_y2) == pytest.approx((rank_x, rank_y))


def test_all_tied_calls_give_p_one():
    assert GATE.mann_whitney([4e6] * 12, [4e6] * 12)[0] == 1.0


def row(system, arm, seed, hit, calls, status="ok"):
    return {
        "system": system,
        "arm": arm,
        "seed": str(seed),
        "position": "1",
        "budget": "1000",
        "hit": str(int(hit)),
        "evals_to_hit": str(calls),
        "charged": str(calls),
        "best": "-1.0",
        "wall_s": "1.0",
        "pot_s": "0.5",
        "outside_s": "0.5",
        "status": status,
    }


def arm_rows(system, arm, hits, hit_calls=100, seeds=GATE.SEEDS):
    rows = []
    for index, seed in enumerate(seeds):
        hit = index < hits
        rows.append(row(system, arm, seed, hit, hit_calls + index if hit else 1000))
    return rows


PANEL = {"budget": "1000", "reference": "-1.0000000000"}


def summary(rows):
    return GATE.system_summary("m80r6", rows, PANEL)


def test_identical_arms_pass():
    rows = arm_rows("m80r6", "A", 8) + arm_rows("m80r6", "B", 8)
    result = summary(rows)
    assert result["verdict"] == "PASS"
    assert (result["loss_hits"], result["loss_calls"], result["loss_failure"]) == ("0", "0", "0")


def test_fewer_hits_at_five_percent_is_a_loss():
    rows = arm_rows("m80r6", "A", 12) + arm_rows("m80r6", "B", 4)
    result = summary(rows)
    assert result["loss_hits"] == "1"
    assert result["verdict"] == "FIX"


def test_a_hit_deficit_that_is_not_significant_is_no_loss():
    rows = arm_rows("m80r6", "A", 9) + arm_rows("m80r6", "B", 7)
    result = summary(rows)
    assert float(result["fisher_p"]) > GATE.ALPHA
    assert result["verdict"] == "PASS"


def test_slower_calls_at_equal_hits_are_a_loss():
    slow = arm_rows("m80r6", "B", 12, hit_calls=600)
    rows = arm_rows("m80r6", "A", 12) + slow
    result = summary(rows)
    assert result["loss_hits"] == "0"
    assert result["loss_calls"] == "1"
    assert result["verdict"] == "FIX"


def test_more_hits_and_fewer_calls_are_gains_and_pass():
    rows = arm_rows("m80r6", "A", 3) + arm_rows("m80r6", "B", 12)
    result = summary(rows)
    assert (result["gain_hits"], result["gain_calls"]) == ("1", "1")
    assert result["verdict"] == "PASS"


def test_a_failure_of_b_alone_is_a_loss_and_is_scored_as_a_miss():
    a = arm_rows("m80r6", "A", 12)
    b = arm_rows("m80r6", "B", 12)
    b[3] = row("m80r6", "B", GATE.SEEDS[3], False, 1000, status="exit101")
    result = summary(a + b)
    assert result["loss_failure"] == "1"
    assert result["hits_b"] == "11"
    assert result["verdict"] == "FIX"


def test_a_failure_of_both_arms_on_one_seed_is_no_loss_of_b():
    a = arm_rows("m80r6", "A", 12)
    b = arm_rows("m80r6", "B", 12)
    a[3] = row("m80r6", "A", GATE.SEEDS[3], False, 1000, status="timeout")
    b[3] = row("m80r6", "B", GATE.SEEDS[3], False, 1000, status="timeout")
    assert summary(a + b)["loss_failure"] == "0"


def test_missing_seeds_leave_the_gate_incomplete():
    seeds = GATE.SEEDS[:5]
    rows = arm_rows("m80r6", "A", 5, seeds=seeds) + arm_rows("m80r6", "B", 5, seeds=seeds)
    assert summary(rows)["verdict"] == "INCOMPLETE"


def fake_driver(path: Path, tag: str) -> None:
    path.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  budget) printf '4000000\\n-1.5\\n' ;;\n"
        "  run)\n"
        '    echo "env budget=${H2H_BUDGET:-unset} omp=$OMP_NUM_THREADS tag=' + tag + '"\n'
        "    echo system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s\n"
        '    if [ "$3" = 2003 ] && [ "' + tag + '" = B ]; then exit 3; fi\n'
        '    echo "$2,anneal,$3,4000000,true,1234,1234,0.5,-1.0,0.2,0.3" ;;\n'
        "  tip4p)\n"
        '    echo "h2h_row system=tip4p$2 code=anneal seed=$5 budget=$4 hit=1 evals_to_hit=77 '
        'charged=77 best=-1.0 wall_s=0.1 pot_s=0.05 outside_s=0.05 basins=3" ;;\n'
        "esac\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.mark.skipif(
    shutil.which("taskset") is None or not Path("/proc/sys/kernel/hostname").is_file(),
    reason="needs taskset and a Linux /proc",
)
def test_runner_flips_the_arm_order_resumes_and_scrubs_the_environment(tmp_path, monkeypatch):
    for tag in ("A", "B"):
        directory = tmp_path / f"bin{tag}"
        directory.mkdir()
        for name in GATE.DRIVERS.values():
            fake_driver(directory / name, tag)
    monkeypatch.setenv("H2H_BUDGET", "5")
    host = Path("/proc/sys/kernel/hostname").read_text().strip()
    rows = tmp_path / "rows.csv"
    command = [
        sys.executable,
        str(SCRIPT),
        "run",
        "--arm-a-bin", str(tmp_path / "binA"),
        "--arm-b-bin", str(tmp_path / "binB"),
        "--arm-a-library", "library-a",
        "--arm-b-library", "library-b",
        "--harness", "harness-id",
        "--repo", str(ROOT),
        "--work", str(tmp_path / "work"),
        "--rows", str(rows),
        "--panels", str(tmp_path / "panels.csv"),
        "--session-host", host,
        "--cpu", str(sorted(os.sched_getaffinity(0))[0]),
        "--only", "m80r6", "tip4p8",
    ]
    subprocess.run(command, check=True, capture_output=True)
    with rows.open(newline="") as handle:
        recorded = list(csv.DictReader(handle))
    assert len(recorded) == 2 * 2 * len(GATE.SEEDS)
    first = {(r["system"], r["seed"]): r["arm"] for r in recorded if r["position"] == "1"}
    for (system, seed), arm in first.items():
        assert arm == ("A" if int(seed) % 2 == 0 else "B"), (system, seed)
    failed = [r for r in recorded if r["status"] != "ok"]
    assert [(r["system"], r["arm"], r["seed"], r["status"], r["hit"]) for r in failed] == [
        ("m80r6", "B", "2003", "exit3", "0")
    ]
    assert {r["library"] for r in recorded} == {"library-a", "library-b"}
    log = (tmp_path / "work" / "logs" / "m80r6-A-s2000.log").read_text()
    assert "budget=unset omp=1 tag=A" in log
    panels = {r["system"]: r for r in csv.DictReader((tmp_path / "panels.csv").open())}
    assert panels["m80r6"]["budget"] == "4000000"
    assert panels["tip4p8"]["budget"] == str(GATE.tip4p_constants(ROOT)[0])

    before = rows.read_text()
    subprocess.run(command, check=True, capture_output=True)
    assert rows.read_text() == before

    at = command.index("--session-host") + 1
    wrong_host = [*command[:at], host + "-other", *command[at + 1 :]]
    assert subprocess.run(wrong_host, check=False, capture_output=True).returncode == 2
