# Copyright (c) anneal developers
"""The tuned-GMIN harness runs only commands it builds, and records the input bytes that ran."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import os
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
    for name in ("GMIN", *tuned.EXAMPLES):
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


def fake_gmin(args: argparse.Namespace, seen: list[tuple[str, dict[str, str], bytes]], row: dict[str, str] | None = None):
    """Stand in for bash: record the command and the bytes ``$ROOT`` offers, and write GMIN's row file."""

    def run(command: str, log: Path, overrides: dict[str, str] | None = None):
        overrides = dict(overrides or {})
        data = (Path(overrides["ROOT"]) / INPUT).read_bytes()
        seen.append((command, overrides, data))
        if row is None:
            fields = "best=-543.665361 hit=1 evals_to_hit=10 charged=10 wall_s=1.0 pot_s=0.5"
        else:
            fields = (
                f"best={row['best']} hit={row['hit']} evals_to_hit={row['calls_to_hit']} "
                f"charged={row['calls']} wall_s={row['wall_s']} pot_s={row['inside_s']}"
            )
        (args.work / "row.txt").write_text(f"h2h_row system=lj98 {fields}\n")
        return subprocess.CompletedProcess(["bash", "-c", command], 0, "", "")

    return run


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
        tuned.verify(bench, tuned.cell_map(bench.root), bench.build)
    assert not marker.exists()


def test_verify_runs_the_rebuilt_command_on_the_bytes_it_hashed(bench, monkeypatch):
    row = gmin_row()
    write_rows(bench.rows, [row])
    seen: list = []
    monkeypatch.setattr(tuned, "run_shell", fake_gmin(bench, seen, row))
    bench.system, bench.code, bench.seed = ["lj98"], "gmin", int(row["seed"])
    tuned.verify(bench, tuned.cell_map(bench.root), bench.build)
    [(command, overrides, data)] = seen
    assert command == row["command"]
    assert Path(overrides["ROOT"]) != bench.root
    assert data == (bench.root / INPUT).read_bytes()
    assert tuned.blob_of_bytes(bench.root, data) == row["data_blob"]


def test_verify_refuses_a_row_whose_data_blob_is_not_the_input(bench, monkeypatch):
    row = {**gmin_row(), "data_blob": "0" * 40}
    write_rows(bench.rows, [row])
    seen: list = []
    monkeypatch.setattr(tuned, "run_shell", fake_gmin(bench, seen, row))
    bench.system, bench.code, bench.seed = ["lj98"], "gmin", int(row["seed"])
    with pytest.raises(SystemExit, match="the row recorded"):
        tuned.verify(bench, tuned.cell_map(bench.root), bench.build)
    assert not seen


def test_a_run_refuses_an_input_that_differs_from_head(bench, monkeypatch, tmp_path):
    (bench.root / INPUT).write_text((bench.root / INPUT).read_text() + "STEPS 1 1.0\n")
    with pytest.raises(SystemExit, match="differs from HEAD"):
        tuned.stage_input(bench.root, "lj98", tmp_path / "stage")
    seen: list = []
    monkeypatch.setattr(tuned, "run_shell", fake_gmin(bench, seen))
    monkeypatch.setattr(tuned, "harness_sources", lambda root: [])
    bench.system, bench.aa = ["lj98"], True
    with pytest.raises(SystemExit, match="differs from HEAD"):
        tuned.evaluate(bench, tuned.cell_map(bench.root), bench.build)
    assert not seen
    row = gmin_row()
    write_rows(bench.rows, [row])
    bench.code, bench.seed = "gmin", int(row["seed"])
    with pytest.raises(SystemExit, match="differs from HEAD"):
        tuned.verify(bench, tuned.cell_map(bench.root), bench.build)
    assert not seen


def test_an_evaluation_row_records_the_blob_of_the_bytes_that_ran(bench, monkeypatch):
    seen: list = []
    monkeypatch.setattr(tuned, "run_shell", fake_gmin(bench, seen))
    monkeypatch.setattr(tuned, "harness_sources", lambda root: [root / "scripts" / "harness.py"])
    bench.system, bench.aa = ["lj98"], True
    tuned.evaluate(bench, tuned.cell_map(bench.root), bench.build)
    rows = tuned.read_rows(bench.rows)
    assert len(rows) == len(seen) == 2 * len(tuned.EVAL_SEEDS)
    head = git(bench.root, "rev-parse", "HEAD")
    for row, (command, overrides, data) in zip(rows, seen, strict=True):
        assert row["command"] == command
        assert 'cp "$ROOT/bench/gmin-tuned/inputs/lj98.data" data' in command
        assert row["data_blob"] == tuned.blob_of_bytes(bench.root, data) == git(bench.root, "rev-parse", f"HEAD:{INPUT}")
        assert row["harness_commit"] == head
        assert Path(overrides["ROOT"]) != bench.root and not Path(overrides["ROOT"]).exists()


def test_an_evaluation_refuses_a_harness_source_that_differs_from_head(bench, monkeypatch):
    seen: list = []
    monkeypatch.setattr(tuned, "run_shell", fake_gmin(bench, seen))
    script = bench.root / "scripts" / "harness.py"
    monkeypatch.setattr(tuned, "harness_sources", lambda root: [script])
    script.write_text("STEP = 999\n")
    bench.system, bench.aa = ["lj98"], True
    with pytest.raises(SystemExit, match="scripts/harness.py differs from HEAD"):
        tuned.evaluate(bench, tuned.cell_map(bench.root), bench.build)
    monkeypatch.setattr(tuned, "harness_sources", lambda root: [ROOT / "scripts" / "gmin_panel" / "tuned.py"])
    with pytest.raises(SystemExit, match="outside"):
        tuned.evaluate(bench, tuned.cell_map(bench.root), bench.build)
    assert not seen


def test_the_harness_sources_are_tuned_and_the_modules_it_imports():
    names = {path.relative_to(ROOT).as_posix() for path in tuned.harness_sources(ROOT)}
    assert {"scripts/gmin_panel/tuned.py", "scripts/gmin_panel/run_panel.py", "scripts/target_tol.py"} <= names
    assert all(name.startswith("scripts/") for name in names)


def test_a_binary_swapped_after_the_start_is_refused_before_the_next_run(bench, monkeypatch):
    seen: list = []
    monkeypatch.setattr(tuned, "run_shell", fake_gmin(bench, seen))
    monkeypatch.setattr(tuned, "harness_sources", lambda root: [])
    (bench.bin / "GMIN").write_bytes(b"other")
    bench.system, bench.aa = ["lj98"], True
    with pytest.raises(SystemExit, match="GMIN sha256"):
        tuned.evaluate(bench, tuned.cell_map(bench.root), bench.build)
    assert not seen


def test_the_anneal_arm_needs_the_clean_recorded_checkout(repo):
    commit = git(repo, "rev-parse", "HEAD")
    tuned.check_anneal_tree(repo, commit)
    with pytest.raises(SystemExit, match="is not at"):
        tuned.check_anneal_tree(repo, "0" * 40)
    (repo / INPUT).write_text("changed\n")
    with pytest.raises(SystemExit, match="has changes"):
        tuned.check_anneal_tree(repo, commit)
    with pytest.raises(SystemExit, match="needs --anneal"):
        tuned.check_anneal_tree(None, commit)


def test_the_entry_script_passes_its_paths_to_tuned_as_words(tmp_path):
    text = (ROOT / "scripts" / "gmin_panel" / "enter_tuned.sh").read_text()
    block = text[text.index("user_name=$(id -un)"):text.index('chmod +x "$WORK/inside.sh"')]
    work = tmp_path / "w $(touch pwned) `touch pwned2`; x"
    work.mkdir()
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "taskset").write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$OUT"\n')
    (fake / "taskset").chmod(0o755)
    env = {**os.environ, "WORK": str(work), "ROOT": str(work), "HERE": str(work)}
    subprocess.run(["bash", "-c", "set -eu\n" + block], env=env, cwd=tmp_path, check=True)
    out = tmp_path / "args"
    subprocess.run(
        ["bash", str(work / "inside.sh"), "--report"],
        env={**os.environ, "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}", "OUT": str(out)},
        cwd=tmp_path, check=True,
    )
    assert out.read_text().splitlines() == [
        "-c", "0", "/usr/bin/python3", "-u", f"{work}/tuned.py",
        "--root", str(work), "--bin", f"{work}/bin", "--anneal", f"{work}/anneal-src",
        "--work", f"{work}/runs/current", "--logs", f"{work}/logs", "--build", f"{work}/build-record.txt",
        "--report",
    ]
    assert not list(tmp_path.rglob("pwned*"))


def test_verify_takes_a_noise_label_and_rejects_an_unknown_code():
    table = tuned.cell_map(ROOT)
    cell, data_name, command = tuned.job_command(table, "lj98aa", "gmin_b", 2000)
    assert (cell.name, data_name) == ("lj98", "lj98")
    assert "H2H_SYSTEM=lj98aa" in command and "inputs/lj98.data" in command
    for name, code in (("lj98", "gmin_a"), ("lj98aa", "anneal"), ("lj98aa", "gmin"), ("nope", "gmin"), ("lj98", "sh")):
        with pytest.raises(SystemExit, match="no evaluation command"):
            tuned.job_command(table, name, code, 2000)
