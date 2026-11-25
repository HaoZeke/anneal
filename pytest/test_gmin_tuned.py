# Copyright (c) anneal developers
"""The tuned-GMIN harness runs only commands it builds, and records the input bytes that ran."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "bench" / "gmin-tuned" / "rows-shard2.csv"
INPUT = "bench/gmin-tuned/inputs/lj98.data"


def load_tuned():
    """Import tuned.py as its directory would, beside the GMIN panel's run_panel.py."""
    here = ROOT / "scripts" / "gmin_panel"
    other = sys.modules.get("run_panel")
    if other is not None and Path(getattr(other, "__file__", "")).resolve() != here / "run_panel.py":
        del sys.modules["run_panel"]
    sys.path.insert(0, str(here))
    try:
        spec = importlib.util.spec_from_file_location("gmin_tuned", here / "tuned.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["gmin_tuned"] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(here))
        if other is not None:
            sys.modules["run_panel"] = other
    return module


tuned = load_tuned()


def committed_rows() -> list[dict[str, str]]:
    with ROWS.open(newline="") as handle:
        return list(csv.DictReader(handle))


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=fixture", "-c", "user.email=fixture@example.org",
         "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A repository with the TIP4P references, the tuned inputs and a stand-in script, all committed."""
    root = tmp_path / "repo"
    for relative in ("data/tip4p_ccd.csv", INPUT, "bench/gmin-tuned/inputs/lj104.data"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, root / relative)
    (root / "scripts").mkdir()
    (root / "scripts" / "harness.py").write_text("STEP = 1\n")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-q", "--no-verify", "-m", "fixture")
    return root


@pytest.fixture()
def bench(tmp_path: Path, repo: Path, monkeypatch) -> argparse.Namespace:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    build = {}
    for name in ("GMIN", "h2h_gmin_proof", "h2h_morse", "h2h_metal", "h2h_seed"):
        (bin_dir / name).write_bytes(name.encode())
        key = "gmin_binary_sha256" if name == "GMIN" else f"example_{name}"
        build[key] = hashlib.sha256(name.encode()).hexdigest()
    build["anneal_commit"] = git(repo, "rev-parse", "HEAD")
    build["gmin_svnrev"] = tuned.BUILT_SVNREV
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("WORK", str(work))
    monkeypatch.setenv("BIN", str(bin_dir))
    monkeypatch.setenv("ROOT", str(repo))
    monkeypatch.setattr(tuned, "machine_id", lambda: "0" * 64)
    monkeypatch.setattr(tuned, "affinity", lambda: "0")
    return argparse.Namespace(
        root=repo, bin=bin_dir, anneal=repo, work=work, logs=tmp_path / "logs", rows=tmp_path / "rows.csv",
        system=None, code=None, seed=None, eval=False, aa=False, build=build,
    )


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    tuned.write_csv(path, tuned.EVAL_FIELDS, rows)


def gmin_row() -> dict[str, str]:
    return next(row for row in committed_rows() if row["system"] == "lj98" and row["code"] == "gmin")


def test_every_committed_row_holds_the_command_tuned_builds_from_its_fields():
    table = tuned.cell_map(ROOT)
    rows = committed_rows()
    assert rows
    for row in rows:
        cell, _data, command = tuned.job_command(table, row["system"], row["code"], int(row["seed"]))
        assert row["command"] == command, (row["system"], row["code"], row["seed"])
        assert row["budget"] == str(cell.budget)


@pytest.mark.parametrize(
    "forged",
    [
        'touch "$WORK/pwned"',
        'cd "$WORK" && touch pwned',
        None,
    ],
)
def test_verify_never_runs_the_command_a_row_holds(bench, forged):
    row = dict(gmin_row())
    marker = bench.work / "pwned"
    row["command"] = forged if forged is not None else row["command"] + ' ; touch "$WORK/pwned"'
    tuned.assert_clean(row["command"])
    write_rows(bench.rows, [row])
    bench.system, bench.code, bench.seed = ["lj98"], "gmin", int(row["seed"])
    with pytest.raises(SystemExit, match="is not the one tuned.py builds"):
        tuned.verify(bench, tuned.cell_map(bench.root))
    assert not marker.exists()


def test_verify_takes_a_noise_label_and_rejects_an_unknown_code():
    table = tuned.cell_map(ROOT)
    cell, data_name, command = tuned.job_command(table, "lj98aa", "gmin_b", 2000)
    assert (cell.name, data_name) == ("lj98", "lj98")
    assert "H2H_SYSTEM=lj98aa" in command and "inputs/lj98.data" in command
    for name, code in (("lj98", "gmin_a"), ("lj98aa", "anneal"), ("lj98aa", "gmin"), ("nope", "gmin"), ("lj98", "sh")):
        with pytest.raises(SystemExit, match="no evaluation command"):
            tuned.job_command(table, name, code, 2000)
