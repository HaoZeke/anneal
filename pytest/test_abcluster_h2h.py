# Copyright (c) anneal developers
"""The ABCluster head-to-head: its statistics, summarizer and runner.

The rank and exact tests are checked against scipy on tables with ties. The
summarizer is run on synthetic rows in a scratch copy of the bench
directory, so the committed rows are never touched. The runner's own
self-test and the shim's build need no ABCluster download.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "abcluster_h2h"


def load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(f"abcluster_h2h_{name}", SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    return module


@pytest.fixture(scope="module")
def rc():
    return load("run_campaign")


@pytest.fixture(scope="module")
def summ():
    return load("summarize")


def test_fisher_matches_scipy(summ):
    stats = pytest.importorskip("scipy.stats")
    rng = random.Random(7)
    for _ in range(200):
        a, b, c, d = (rng.randint(0, 12) for _ in range(4))
        if a + b == 0 or c + d == 0 or a + c == 0 or b + d == 0:
            continue
        want = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided").pvalue
        assert summ.fisher_exact(a, b, c, d) == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_mann_whitney_matches_scipy_with_ties(summ):
    stats = pytest.importorskip("scipy.stats")
    rng = random.Random(11)
    for _ in range(200):
        x = [float(rng.choice([1, 2, 2, 3, 400000, 400000, rng.randint(1, 50)])) for _ in range(12)]
        y = [float(rng.choice([2, 400000, 400000, 400000, rng.randint(1, 80)])) for _ in range(12)]
        if len(set(x + y)) == 1:
            continue
        want = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic", use_continuity=True).pvalue
        assert summ.mann_whitney(x, y) == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_mann_whitney_all_tied_is_one(summ):
    assert summ.mann_whitney([5.0] * 12, [5.0] * 12) == 1.0


def test_type7_median(rc):
    assert rc.median([3.0, 1.0, 2.0, 4.0]) == 2.5
    assert rc.median([7.0]) == 7.0
    assert rc.type7([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.25) == pytest.approx(2.25)


def test_runner_self_test():
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_campaign.py"), "--self-test"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "ok"


def test_append_rows_is_append_only(rc, tmp_path):
    path = tmp_path / "rows.csv"
    fields = ["system", "arm", "seed"]
    rc.append_rows(path, fields, [{"system": "lj38", "arm": "abcluster", "seed": "2000"}])
    before = path.read_text()
    rc.append_rows(path, fields, [{"system": "lj38", "arm": "abcluster", "seed": "2001"}])
    assert path.read_text().startswith(before)
    assert len(path.read_text().splitlines()) == 3
    with pytest.raises(RuntimeError):
        rc.append_rows(path, ["system", "arm"], [{"system": "lj38", "arm": "abcluster"}])


def test_schedule_covers_every_pair_once(rc):
    order = rc.schedule(list(rc.PAPER_BUDGETS), rc.SEEDS, rc.ARMS, rc.SCHEDULE_SEED)
    assert len(order) == len(set(order)) == len(rc.PAPER_BUDGETS) * len(rc.SEEDS) * len(rc.ARMS)
    assert order == rc.schedule(list(rc.PAPER_BUDGETS), rc.SEEDS, rc.ARMS, rc.SCHEDULE_SEED)


def test_hosts_and_budgets_are_pinned(rc):
    assert rc.PARITY_TOL == 1e-8
    assert rc.HIT_TOL == 1e-3
    assert rc.SEEDS == tuple(range(2000, 2012))
    assert not set(rc.PILOT_SEEDS) & set(rc.SEEDS)


def synthetic_rows(rc, host: str = "bench") -> list[dict[str, str]]:
    rng = random.Random(3)
    rows = []
    for system, budget in rc.PAPER_BUDGETS.items():
        for arm in rc.ARMS:
            for seed in rc.SEEDS:
                hit = rng.random() < (0.9 if arm != "abcluster" else 0.3)
                calls = rng.randint(1000, budget) if hit else budget
                wall = calls * 1e-5
                pot = wall * 0.6
                rows.append(
                    {
                        "system": system,
                        "arm": arm,
                        "seed": str(seed),
                        "budget": str(budget),
                        "hit": "1" if hit else "0",
                        "calls_to_hit": str(calls),
                        "time_to_hit_s": f"{wall:.6f}" if hit else "",
                        "charged": str(calls),
                        "wall_s": f"{wall:.6f}",
                        "pot_s": f"{pot:.6f}",
                        "outside_s": f"{wall - pot:.6f}",
                        "best": "-1.0",
                        "stop": "hit" if hit else "budget",
                        "process_s": f"{wall:.6f}",
                        "setting": "x",
                        "code_revision": "rev",
                        "anneal_head": "head",
                        "host": host,
                        "cpu": "cpu",
                    }
                )
    return rows


def write_rows(rc, path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rc.SEED_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def scratch(summ, rc, tmp_path, monkeypatch):
    bench = tmp_path / "abcluster-h2h"
    bench.mkdir()
    shutil.copy(ROOT / "bench" / "abcluster-h2h" / "README.md", bench / "README.md")
    monkeypatch.setattr(summ, "BENCH", bench)
    return bench


def test_summarizer_writes_table_between_markers(summ, rc, scratch):
    rows = synthetic_rows(rc)
    write_rows(rc, scratch / "seeds.csv", rows)
    summ.main()
    summary = list(csv.DictReader((scratch / "summary.csv").open()))
    assert len(summary) == len(rc.PAPER_BUDGETS) * len(rc.ARMS)
    first = summary[0]
    group = [r for r in rows if r["system"] == first["system"] and r["arm"] == first["arm"]]
    assert int(first["hits"]) == sum(r["hit"] == "1" for r in group)
    assert first["fisher_p"] == ""
    assert summary[1]["fisher_p"] != "" and summary[1]["mwu_calls_p"] != "" and summary[1]["mwu_time_p"] != ""
    text = (scratch / "README.md").read_text()
    head, rest = text.split(summ.BEGIN)
    body, tail = rest.split(summ.END)
    assert "| lj38 | 400,000 | ABCluster |" in body
    assert body.count("\n| ") >= len(summary)
    assert tail.lstrip().startswith("`scripts/abcluster_h2h/summarize.py`")
    again = text
    summ.main()
    assert (scratch / "README.md").read_text() == again


@pytest.mark.parametrize("break_row", ["host", "duplicate", "missing", "budget"])
def test_summarizer_refuses_bad_rows(summ, rc, scratch, break_row):
    rows = synthetic_rows(rc)
    if break_row == "host":
        rows[0]["host"] = "laptop"
    elif break_row == "duplicate":
        rows.append(dict(rows[0]))
    elif break_row == "missing":
        rows.pop()
    else:
        rows[0]["budget"] = "12345"
    write_rows(rc, scratch / "seeds.csv", rows)
    with pytest.raises(SystemExit):
        summ.main()


def test_runner_refuses_other_hostnames():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_campaign.py"),
            "--anneal", "x", "--atom", "x", "--shim", "x", "--pins", "x",
            "--head-file", "x", "--runs", "x", "--out", "x",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if os.uname().nodename == "bench":
        pytest.skip("this host is named bench")
    assert proc.returncode == 2


def test_shim_compiles_without_warnings(tmp_path):
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("gcc is not installed")
    out = tmp_path / "abc_shim.so"
    proc = subprocess.run(
        [gcc, "-O2", "-fPIC", "-shared", "-Wall", "-Wextra", "-Werror", "-o", str(out), str(SCRIPTS / "abc_shim.c"), "-ldl"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
