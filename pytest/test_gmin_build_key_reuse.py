# Copyright (c) anneal developers
"""Per-runner tests for GMIN revision + build-key provenance on saved-row reuse."""

from __future__ import annotations

import csv
import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from gmin_pin import REVISION, TARBALL_SHA256  # noqa: E402

EXPECTED_KEY = "b" * 64
WRONG_KEY = "c" * 64
WRONG_REV = "1" * 40
FAKE_HEAD = "a" * 40
FAKE_BASE = "e" * 40
FAKE_MACHINE = {"machine_id": "d" * 64, "cpu_model": "Example Test CPU"}


def _load_module(name: str, rel_path: str) -> ModuleType:
    path = ROOT / rel_path
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_gmin_meta(
    directory: Path,
    *,
    revision: str = REVISION,
    build_key: str = EXPECTED_KEY,
    url: str = "https://example.invalid/gmin.tar.bz2",
    prefix: str = "gmin",
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    svnrev = directory / f"{prefix}-SVNREV"
    svnrev.write_text(f"{revision}\n")
    source = directory / f"{prefix}-source.txt"
    source.write_text(
        f"url={url}\n"
        f"revision={revision}\n"
        f"svnrev={revision}\n"
        f"tarball_sha256={TARBALL_SHA256}\n"
        f"build_key={build_key}\n"
    )
    return svnrev, source


def _patch_bench_env(monkeypatch: pytest.MonkeyPatch, *modules: ModuleType) -> None:
    real_uname = os.uname()
    fake_uname = os.uname_result(
        (
            real_uname.sysname,
            "bench",
            real_uname.release,
            real_uname.version,
            real_uname.machine,
        )
    )
    monkeypatch.setattr(socket, "gethostname", lambda: "bench")
    monkeypatch.setattr(os, "uname", lambda: fake_uname)
    if hasattr(os, "sched_getaffinity"):
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    import machine_provenance as mp

    monkeypatch.setattr(mp, "machine_columns", lambda: dict(FAKE_MACHINE))
    monkeypatch.setattr(
        mp,
        "machine_provenance",
        lambda: (FAKE_MACHINE["machine_id"], FAKE_MACHINE["cpu_model"]),
    )
    for mod in modules:
        if hasattr(mod, "machine_columns"):
            monkeypatch.setattr(mod, "machine_columns", lambda: dict(FAKE_MACHINE))
        if hasattr(mod, "machine_provenance"):
            monkeypatch.setattr(
                mod,
                "machine_provenance",
                lambda: (FAKE_MACHINE["machine_id"], FAKE_MACHINE["cpu_model"]),
            )


def _write_csv_with_fields(
    path: Path,
    fields: list[str] | tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_list = list(fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=field_list,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            merged = {key: "" for key in field_list}
            merged.update({k: v for k, v in row.items() if k in merged})
            writer.writerow(merged)


def _init_fake_git_tree(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / "README").write_text("fixture\n")
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        }
    )
    subprocess.run(["git", "add", "README"], cwd=path, env=env, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=path,
        env=env,
        check=True,
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def test_gmin_panel_run_panel_rejects_mismatched_build_key_and_reuses_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = _load_module("rnext5_gmin_panel", "scripts/gmin_panel/run_panel.py")
    _patch_bench_env(monkeypatch, panel)

    tree = tmp_path / "tree"
    head = _init_fake_git_tree(tree)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    lj_data = tmp_path / "lj.data"
    morse_data = tmp_path / "morse.data"
    lj_data.write_text("STEPS 10 1.0\n")
    morse_data.write_text("STEPS 10 1.0\n")
    svnrev, gmin_source = _write_gmin_meta(tmp_path / "meta", build_key=EXPECTED_KEY)

    monkeypatch.setattr(
        panel,
        "binary_budget",
        lambda _b, _t, system: (panel.PAPER_BUDGETS[system], panel.spec(system)[3]),
    )
    monkeypatch.setattr(
        panel,
        "resolve_driver",
        lambda _b, _t, _system: ("h2h_gmin_proof", bin_dir / "h2h_gmin_proof"),
    )
    monkeypatch.setattr(panel, "write_tables", lambda *_a, **_k: None)

    def _forbid_run(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise AssertionError("saved row should have been reused or rejected before running")

    monkeypatch.setattr(panel, "run_anneal", _forbid_run)
    monkeypatch.setattr(panel, "run_gmin", _forbid_run)

    def _panel_rows(gmin_rev: str, build_key: str) -> list[dict[str, str]]:
        base = {
            "system": "lj38",
            "seed": "2000",
            "budget": str(panel.PAPER_BUDGETS["lj38"]),
            "hit": "1",
            "evals_to_hit": "9000",
            "charged": "9000",
            "wall_s": "0.500000",
            "pot_s": "0.400000",
            "outside_s": "0.100000",
            "best": "-173.928427000",
            "harness_revision": head,
            "host": "bench",
            **FAKE_MACHINE,
            "session": "20261122T000000Z",
            "utc": "2026-11-22T00:00:00Z",
        }
        return [
            {
                **base,
                "code": "anneal",
                "driver": "h2h_gmin_proof",
                "code_revision": head,
                "gmin_build_key": "",
            },
            {
                **base,
                "code": "gmin",
                "driver": "GMIN",
                "code_revision": gmin_rev,
                "gmin_build_key": build_key,
            },
        ]

    rows_csv = tmp_path / "rows.csv"
    argv = [
        "run_panel.py",
        "--root",
        str(ROOT),
        "--rows",
        str(rows_csv),
        "--summary",
        str(tmp_path / "summary.csv"),
        "--revision",
        head,
        "--bin",
        str(bin_dir),
        "--tree",
        str(tree),
        "--work",
        str(tmp_path / "work"),
        "--logs",
        str(tmp_path / "logs"),
        "--gmin-source",
        str(gmin_source),
        "--svnrev",
        str(svnrev),
        "--lj-data",
        str(lj_data),
        "--morse-data",
        str(morse_data),
        "--system",
        "lj38",
        "--seed",
        "2000",
    ]

    _write_csv_with_fields(rows_csv, panel.ROW_FIELDS, _panel_rows(REVISION, WRONG_KEY))
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        panel.main(argv)

    _write_csv_with_fields(rows_csv, panel.ROW_FIELDS, _panel_rows(WRONG_REV, EXPECTED_KEY))
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="code_revision"):
        panel.main(argv)

    _write_csv_with_fields(rows_csv, panel.ROW_FIELDS, _panel_rows(REVISION, EXPECTED_KEY))
    assert panel.main(argv) == 0


def test_gmin_panel_run_losses2_rejects_mismatched_build_key_and_reuses_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    losses2 = _load_module("rnext5_gmin_losses2", "scripts/gmin_panel/run_losses2.py")
    panel = losses2.panel
    _patch_bench_env(monkeypatch, losses2, panel)

    base_tree = tmp_path / "base-tree"
    head_tree = tmp_path / "head-tree"
    base_rev = _init_fake_git_tree(base_tree)
    head_rev = _init_fake_git_tree(head_tree)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    lj_data = tmp_path / "lj.data"
    morse_data = tmp_path / "morse.data"
    lj_data.write_text("STEPS 10 1.0\n")
    morse_data.write_text("STEPS 10 1.0\n")
    svnrev, gmin_source = _write_gmin_meta(tmp_path / "meta", build_key=EXPECTED_KEY)

    monkeypatch.setattr(
        panel,
        "binary_budget",
        lambda _b, _t, system: (panel.PAPER_BUDGETS[system], panel.spec(system)[3]),
    )
    monkeypatch.setattr(
        panel,
        "resolve_driver",
        lambda _b, _t, _system: ("h2h_gmin_proof", bin_dir / "h2h_gmin_proof"),
    )

    def _forbid_run(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise AssertionError("saved row should have been reused or rejected before running")

    monkeypatch.setattr(panel, "run_anneal", _forbid_run)
    monkeypatch.setattr(panel, "run_gmin", _forbid_run)

    def _losses_rows(gmin_rev: str, build_key: str) -> list[dict[str, str]]:
        common = {
            "system": "lj38",
            "seed": "2000",
            "budget": str(panel.PAPER_BUDGETS["lj38"]),
            "hit": "1",
            "evals_to_hit": "9000",
            "charged": "9000",
            "wall_s": "0.500000",
            "pot_s": "0.400000",
            "outside_s": "0.100000",
            "best": "-173.928427000",
            "harness_revision": head_rev,
            "host": "bench",
            **FAKE_MACHINE,
            "session": "20261122T000000Z",
            "utc": "2026-11-22T00:00:00Z",
        }
        return [
            {
                **common,
                "code": "baseline",
                "driver": "h2h_gmin_proof",
                "code_revision": base_rev,
                "gmin_build_key": "",
            },
            {
                **common,
                "code": "fix2",
                "driver": "h2h_gmin_proof",
                "code_revision": head_rev,
                "gmin_build_key": "",
            },
            {
                **common,
                "code": "gmin",
                "driver": "GMIN",
                "code_revision": gmin_rev,
                "gmin_build_key": build_key,
            },
        ]

    rows_csv = tmp_path / "losses2-rows.csv"
    summary_csv = tmp_path / "losses2-summary.csv"
    argv = [
        "run_losses2.py",
        "--rows",
        str(rows_csv),
        "--summary",
        str(summary_csv),
        "--revision",
        head_rev,
        "--base-revision",
        base_rev,
        "--base-bin",
        str(bin_dir),
        "--base-tree",
        str(base_tree),
        "--head-bin",
        str(bin_dir),
        "--head-tree",
        str(head_tree),
        "--work",
        str(tmp_path / "work"),
        "--logs",
        str(tmp_path / "logs"),
        "--gmin-source",
        str(gmin_source),
        "--svnrev",
        str(svnrev),
        "--lj-data",
        str(lj_data),
        "--morse-data",
        str(morse_data),
        "--system",
        "lj38",
        "--seed",
        "2000",
    ]

    _write_csv_with_fields(rows_csv, panel.ROW_FIELDS, _losses_rows(REVISION, WRONG_KEY))
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        losses2.main(argv)

    _write_csv_with_fields(rows_csv, panel.ROW_FIELDS, _losses_rows(WRONG_REV, EXPECTED_KEY))
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="code_revision"):
        losses2.main(argv)

    _write_csv_with_fields(rows_csv, panel.ROW_FIELDS, _losses_rows(REVISION, EXPECTED_KEY))
    assert losses2.main(argv) == 0


def test_morse_h2h_rejects_mismatched_build_key_and_reuses_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    morse_h2h = _load_module("rnext5_morse_h2h", "scripts/morse_h2h/run_campaign.py")
    _patch_bench_env(monkeypatch, morse_h2h)

    svnrev, gmin_source = _write_gmin_meta(tmp_path / "meta", build_key=EXPECTED_KEY)
    head_file = tmp_path / "head"
    head_file.write_text(f"{FAKE_HEAD}\n")
    binary = tmp_path / "h2h_morse"
    binary.write_text("#!/bin/sh\nexit 99\n")
    gmin = tmp_path / "GMIN"
    gmin.write_text("#!/bin/sh\nexit 99\n")
    data = tmp_path / "morse.data"
    data.write_text("STEPS 10 1.0\n")

    monkeypatch.setattr(
        morse_h2h,
        "binary_budget",
        lambda _b, name: (
            morse_h2h.spec(name)[3],
            morse_h2h.spec(name)[4],
            morse_h2h.spec(name)[2],
        ),
    )
    monkeypatch.setattr(
        morse_h2h,
        "write_start",
        lambda _b, _name, _seed, path: (
            path.parent.mkdir(parents=True, exist_ok=True),
            path.write_text("38\n\n"),
        ),
    )

    def _forbid_run(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise AssertionError("saved row should have been reused or rejected before running")

    monkeypatch.setattr(morse_h2h, "run_anneal", _forbid_run)
    monkeypatch.setattr(morse_h2h, "run_gmin", _forbid_run)

    runs = tmp_path / "runs"
    gmin_saved = runs / "m38r14" / "gmin-2000.row"

    def _write_saved_gmin(rev: str, build_key: str) -> None:
        _write_csv_with_fields(
            gmin_saved,
            morse_h2h.PER_SEED_FIELDS,
            [
                {
                    "system": "m38r14",
                    "code": "gmin",
                    "seed": "2000",
                    "budget": "400000",
                    "hit": "1",
                    "evals_to_hit": "20000",
                    "charged": "20000",
                    "wall_s": "0.600000",
                    "best": "-144.321054",
                    "pot_s": "0.500000",
                    "outside_s": "0.100000",
                    "code_revision": rev,
                    "gmin_url": "https://example.invalid/gmin.tar.bz2",
                    "gmin_revision": rev,
                    "gmin_build_key": build_key,
                    "anneal_head": "",
                    "host": "bench",
                    **FAKE_MACHINE,
                }
            ],
        )

    argv = [
        "run_campaign.py",
        "--binary",
        str(binary),
        "--gmin",
        str(gmin),
        "--data",
        str(data),
        "--source",
        str(gmin_source),
        "--svnrev",
        str(svnrev),
        "--head-file",
        str(head_file),
        "--runs",
        str(runs),
        "--starts",
        str(tmp_path / "starts"),
        "--system",
        "m38r14",
        "--seed",
        "2000",
        "--arm",
        "gmin",
    ]

    _write_saved_gmin(REVISION, WRONG_KEY)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        morse_h2h.main(argv)

    _write_saved_gmin(WRONG_REV, EXPECTED_KEY)
    with pytest.raises(
        (ValueError, RuntimeError, SystemExit), match="code_revision|gmin_revision"
    ):
        morse_h2h.main(argv)

    _write_saved_gmin(REVISION, EXPECTED_KEY)
    assert morse_h2h.main(argv) == 0


def test_morse_gmin_rejects_mismatched_build_key_and_reuses_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    morse_gmin = _load_module("rnext5_morse_gmin", "scripts/morse_gmin/run_panel.py")
    _patch_bench_env(monkeypatch, morse_gmin)

    fake_root = tmp_path / "workspace" / "repo"
    (fake_root / "examples").mkdir(parents=True)
    (fake_root / "examples" / "h2h_gmin_morse.data").write_text("STEPS 10 1.0\n")
    binary = fake_root.parent / "tmp" / "anneal-target" / "release" / "examples" / "h2h_morse"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 99\n")
    gmin = tmp_path / "GMIN"
    gmin.write_text("#!/bin/sh\nexit 99\n")
    svnrev, gmin_source = _write_gmin_meta(tmp_path / "meta", build_key=EXPECTED_KEY)
    monkeypatch.setenv("GMIN_SOURCE_FILE", str(gmin_source))
    monkeypatch.setattr(morse_gmin, "git_head", lambda _r: FAKE_HEAD)

    def _forbid_run(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise AssertionError("saved row should have been reused or rejected before running")

    monkeypatch.setattr(morse_gmin, "run_gmin", _forbid_run)

    archive_csv = fake_root / "bench" / "morse-h2h" / "seeds.csv"
    seeds_csv = fake_root / "bench" / "morse-gmin" / "seeds.csv"

    def _archive_row(rev: str, build_key: str) -> dict[str, str]:
        return {
            "system": "m30r6",
            "code": "gmin",
            "seed": "2000",
            "budget": "400000",
            "hit": "1",
            "evals_to_hit": "10000",
            "charged": "10000",
            "wall_s": "0.500000",
            "best": "-106.835790",
            "pot_s": "0.400000",
            "outside_s": "0.100000",
            "code_revision": rev,
            "gmin_build_key": build_key,
            "host": "bench",
            **FAKE_MACHINE,
        }

    argv = [
        "run_panel.py",
        "--root",
        str(fake_root),
        "--work",
        str(tmp_path / "work"),
        "--gmin",
        str(gmin),
        "--svnrev",
        str(svnrev),
        "--system",
        "m30r6",
        "--seed",
        "2000",
        "--arm",
        "gmin",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    _write_csv_with_fields(archive_csv, morse_gmin.FIELDS, [_archive_row(REVISION, WRONG_KEY)])
    if seeds_csv.exists():
        seeds_csv.unlink()
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        morse_gmin.main()

    archive_csv.unlink()
    _write_csv_with_fields(seeds_csv, morse_gmin.FIELDS, [_archive_row(WRONG_REV, EXPECTED_KEY)])
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="code_revision"):
        morse_gmin.main()

    _write_csv_with_fields(seeds_csv, morse_gmin.FIELDS, [_archive_row(REVISION, EXPECTED_KEY)])
    assert morse_gmin.main() == 0


def test_morse_panel_rejects_mismatched_build_key_and_reuses_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    morse_panel = _load_module("rnext5_morse_panel", "scripts/morse_panel/run_panel.py")
    _patch_bench_env(monkeypatch, morse_panel)

    fake_root = tmp_path / "repo"
    (fake_root / "examples").mkdir(parents=True)
    (fake_root / "examples" / "h2h_gmin_morse.data").write_text("STEPS 10 1.0\n")
    gmin = tmp_path / "GMIN"
    gmin.write_text("#!/bin/sh\nexit 99\n")
    svnrev, gmin_source = _write_gmin_meta(tmp_path / "meta", build_key=EXPECTED_KEY)
    monkeypatch.setenv("GMIN_SOURCE_FILE", str(gmin_source))

    real_run = subprocess.run

    def _fake_subprocess_run(cmd: list[str], *args: object, **kwargs: object):
        if "build" in cmd and "h2h_morse" in cmd:
            target = Path(kwargs["env"]["CARGO_TARGET_DIR"]) / "release" / "examples" / "h2h_morse"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("#!/bin/sh\nexit 99\n")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "rev-parse" in cmd:
            return subprocess.CompletedProcess(cmd, 0, f"{FAKE_HEAD}\n", "")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run)

    def _forbid_gmin(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise AssertionError("saved GMIN row should have been reused or rejected before running")

    monkeypatch.setattr(morse_panel, "run_gmin", _forbid_gmin)

    archive_csv = fake_root / "bench" / "morse-h2h" / "seeds.csv"
    work_dir = tmp_path / "work"
    seeds_csv = work_dir / "seeds.csv"

    def _row(rev: str, build_key: str) -> dict[str, str]:
        return {
            "system": "m55r6",
            "code": "gmin",
            "seed": "2000",
            "budget": "1000000",
            "hit": "1",
            "evals_to_hit": "25000",
            "charged": "25000",
            "wall_s": "0.800000",
            "best": "-250.286609",
            "pot_s": "0.700000",
            "outside_s": "0.100000",
            "code_revision": rev,
            "gmin_build_key": build_key,
            "host": "bench",
            **FAKE_MACHINE,
        }

    argv = [
        "run_panel.py",
        "--root",
        str(fake_root),
        "--work",
        str(work_dir),
        "--gmin",
        str(gmin),
        "--svnrev",
        str(svnrev),
        "--system",
        "m55r6",
        "--seed",
        "2000",
        "--arm",
        "gmin",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    _write_csv_with_fields(archive_csv, morse_panel.ROW_FIELDS, [_row(REVISION, WRONG_KEY)])
    if seeds_csv.exists():
        seeds_csv.unlink()
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        morse_panel.main()

    archive_csv.unlink()
    _write_csv_with_fields(seeds_csv, morse_panel.ROW_FIELDS, [_row(WRONG_REV, EXPECTED_KEY)])
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="code_revision"):
        morse_panel.main()

    _write_csv_with_fields(seeds_csv, morse_panel.ROW_FIELDS, [_row(REVISION, EXPECTED_KEY)])
    morse_panel.main()


def test_tip4p_budget_rejects_mismatched_build_key_and_reuses_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget_mod = _load_module("rnext5_tip4p_budget", "scripts/tip4p_h2h/run_budget.py")
    _patch_bench_env(monkeypatch, budget_mod)

    svnrev, gmin_source = _write_gmin_meta(tmp_path / "meta", build_key=EXPECTED_KEY)
    head_file = tmp_path / "head"
    head_file.write_text(f"{FAKE_HEAD}\n")
    binary = tmp_path / "h2h_seed"
    binary.write_text("#!/bin/sh\nexit 99\n")
    gmin = tmp_path / "GMIN"
    gmin.write_text("#!/bin/sh\nexit 99\n")
    data = tmp_path / "tip4p.data"
    data.write_text("STEPS 10 1.0\n")
    start_file = tmp_path / "start.gmin"
    start_file.write_text("0 0 0\n")

    monkeypatch.setattr(Path, "cwd", staticmethod(lambda: Path("/work")))
    monkeypatch.setattr(
        budget_mod,
        "run_anneal",
        lambda *_a, **_k: {
            "hit": "1",
            "evals_to_hit": "12000",
            "charged": "12000",
            "best": "-916.7074302940",
            "wall_s": "0.500000",
            "pot_s": "0.400000",
            "outside_s": "0.100000",
            "basins": "10",
            "start": str(start_file),
        },
    )

    def _forbid_gmin(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise AssertionError("reused GMIN row should have been reused or rejected before running")

    monkeypatch.setattr(budget_mod, "run_gmin", _forbid_gmin)

    gmin_rows_csv = tmp_path / "prior-gmin.csv"
    out_rows_csv = tmp_path / "out-rows.csv"
    summary_csv = tmp_path / "summary.csv"

    prior_fields = [
        "system",
        "code",
        "budget",
        "seed",
        "hit",
        "evals_to_hit",
        "charged",
        "best",
        "wall_s",
        "pot_s",
        "outside_s",
        "basins",
        "code_revision",
        "gmin_build_key",
        "host",
        "machine_id",
        "cpu_model",
    ]

    def _write_prior(rev: str, build_key: str) -> None:
        _write_csv_with_fields(
            gmin_rows_csv,
            prior_fields,
            [
                {
                    "system": "tip4p21",
                    "code": "gmin",
                    "budget": "2000000",
                    "seed": "2000",
                    "hit": "1",
                    "evals_to_hit": "25000",
                    "charged": "25000",
                    "best": "-916.7074302940",
                    "wall_s": "1.200000",
                    "pot_s": "1.000000",
                    "outside_s": "0.200000",
                    "basins": "20",
                    "code_revision": rev,
                    "gmin_build_key": build_key,
                    "host": "bench",
                    **FAKE_MACHINE,
                }
            ],
        )

    argv = [
        "run_budget.py",
        "--binary",
        str(binary),
        "--gmin",
        str(gmin),
        "--data",
        str(data),
        "--work",
        str(tmp_path / "work"),
        "--rows",
        str(out_rows_csv),
        "--summary",
        str(summary_csv),
        "--gmin-rows",
        str(gmin_rows_csv),
        "--budget",
        "2000000",
        "--seed",
        "2000",
        "--head-file",
        str(head_file),
        "--gmin-source",
        str(gmin_source),
        "--svnrev",
        str(svnrev),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    _write_prior(REVISION, WRONG_KEY)
    if out_rows_csv.exists():
        out_rows_csv.unlink()
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        budget_mod.main()

    _write_prior(WRONG_REV, EXPECTED_KEY)
    if out_rows_csv.exists():
        out_rows_csv.unlink()
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="code_revision"):
        budget_mod.main()

    _write_prior(REVISION, EXPECTED_KEY)
    if out_rows_csv.exists():
        out_rows_csv.unlink()
    assert budget_mod.main() == 0


def test_occat_h2h_rejects_mismatched_build_key_and_reuses_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    occat = _load_module("rnext5_occat_h2h", "scripts/occat_h2h/run_campaign.py")
    _patch_bench_env(monkeypatch, occat)

    svnrev, gmin_source = _write_gmin_meta(tmp_path / "meta", build_key=EXPECTED_KEY)
    monkeypatch.setenv("GMIN_SVNREV_FILE", str(svnrev))
    monkeypatch.setenv("GMIN_SOURCE_FILE", str(gmin_source))

    monkeypatch.setattr(occat, "SYSTEMS", {"lj38": occat.SYSTEMS["lj38"]})
    monkeypatch.setattr(occat, "SEEDS", (2000,))
    monkeypatch.setattr(occat, "EXTRA_SEEDS", ())
    monkeypatch.setattr(occat, "wall_allows", lambda *_a, **_k: False)
    monkeypatch.setattr(
        occat,
        "run_anneal",
        lambda system, seed, _w: {
            "system": system,
            "code": "anneal",
            "seed": str(seed),
            "budget": "400000",
            "hit": "1",
            "evals_to_hit": "9000",
            "charged": "9000",
            "wall_s": "0.500000",
            "best": "-173.928427",
            "pot_s": "0.400000",
            "outside_s": "0.100000",
        },
    )
    monkeypatch.setattr(
        occat,
        "run_occat",
        lambda system, seed, _w: {
            "system": system,
            "code": "occat",
            "seed": str(seed),
            "budget": "400000",
            "hit": "1",
            "evals_to_hit": "15000",
            "charged": "15000",
            "wall_s": "0.800000",
            "best": "-173.928427",
            "pot_s": "0.700000",
            "outside_s": "0.100000",
        },
    )

    def _forbid_gmin(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise AssertionError("stored GMIN row should have been reused or rejected before running")

    monkeypatch.setattr(occat, "run_gmin", _forbid_gmin)

    source_csv = tmp_path / "gmin-archive.csv"
    monkeypatch.setattr(occat, "GMIN_SOURCES", [source_csv])

    def _write_archive(rev: str, build_key: str) -> None:
        _write_csv_with_fields(
            source_csv,
            occat.SEED_FIELDS,
            [
                {
                    "system": "lj38",
                    "code": "gmin",
                    "seed": "2000",
                    "budget": "400000",
                    "hit": "1",
                    "evals_to_hit": "12000",
                    "charged": "12000",
                    "wall_s": "0.600000",
                    "best": "-173.928427",
                    "pot_s": "0.500000",
                    "outside_s": "0.100000",
                    "code_revision": rev,
                    "gmin_build_key": build_key,
                    "host": "bench",
                    **FAKE_MACHINE,
                }
            ],
        )

    seeds_path = tmp_path / "out" / "seeds.csv"
    summary_path = tmp_path / "out" / "summary.csv"
    tests_path = tmp_path / "out" / "tests.csv"

    def _seed_rows(rev: str, build_key: str) -> list[dict[str, str]]:
        return [
            {
                "system": "lj38",
                "code": "anneal",
                "seed": "2000",
                "budget": "400000",
                "hit": "1",
                "evals_to_hit": "9000",
                "charged": "9000",
                "wall_s": "0.500000",
                "best": "-173.928427",
                "pot_s": "0.400000",
                "outside_s": "0.100000",
                "code_revision": FAKE_HEAD,
                "gmin_build_key": "",
                "host": "bench",
                **FAKE_MACHINE,
            },
            {
                "system": "lj38",
                "code": "occat",
                "seed": "2000",
                "budget": "400000",
                "hit": "1",
                "evals_to_hit": "15000",
                "charged": "15000",
                "wall_s": "0.800000",
                "best": "-173.928427",
                "pot_s": "0.700000",
                "outside_s": "0.100000",
                "code_revision": FAKE_HEAD,
                "gmin_build_key": "",
                "host": "bench",
                **FAKE_MACHINE,
            },
            {
                "system": "lj38",
                "code": "gmin",
                "seed": "2000",
                "budget": "400000",
                "hit": "1",
                "evals_to_hit": "12000",
                "charged": "12000",
                "wall_s": "0.600000",
                "best": "-173.928427",
                "pot_s": "0.500000",
                "outside_s": "0.100000",
                "code_revision": rev,
                "gmin_build_key": build_key,
                "host": "bench",
                **FAKE_MACHINE,
            },
        ]

    # 1. Stored row in seeds_path with wrong build_key or wrong code_revision must be rejected.
    _write_csv_with_fields(seeds_path, occat.SEED_FIELDS, _seed_rows(REVISION, WRONG_KEY))
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        occat.campaign(seeds_path, summary_path, tests_path, tmp_path / "work")

    _write_csv_with_fields(seeds_path, occat.SEED_FIELDS, _seed_rows(WRONG_REV, EXPECTED_KEY))
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="code_revision"):
        occat.campaign(seeds_path, summary_path, tests_path, tmp_path / "work")

    # 2. Archive GMIN_SOURCES row with wrong build_key or wrong code_revision must be rejected.
    seeds_path.unlink()
    _write_archive(REVISION, WRONG_KEY)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        occat.campaign(seeds_path, summary_path, tests_path, tmp_path / "work")

    if seeds_path.exists():
        seeds_path.unlink()
    _write_archive(WRONG_REV, EXPECTED_KEY)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="code_revision"):
        occat.campaign(seeds_path, summary_path, tests_path, tmp_path / "work")

    # 3. Matching revision and build_key reuses stored GMIN row without running GMIN.
    if seeds_path.exists():
        seeds_path.unlink()
    _write_archive(REVISION, EXPECTED_KEY)
    occat.campaign(seeds_path, summary_path, tests_path, tmp_path / "work")


def test_gad_published_rejects_mismatched_build_key_and_reuses_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gad = _load_module("rnext5_gad", "scripts/gad_published/run_campaign.py")
    _patch_bench_env(monkeypatch, gad)

    svnrev, gmin_source = _write_gmin_meta(tmp_path / "meta", build_key=EXPECTED_KEY)
    head_file = tmp_path / "head"
    head_file.write_text(f"{FAKE_HEAD}\n")
    binary = tmp_path / "gad_published"
    binary.write_text("#!/bin/sh\nexit 99\n")
    gmin = tmp_path / "GMIN"
    gmin.write_text("#!/bin/sh\nexit 99\n")
    data = tmp_path / "lj.data"
    data.write_text("STEPS 10 1.0\n")
    fake_zip = tmp_path / "deposit.zip"
    fake_zip.write_bytes(b"placeholder")

    monkeypatch.setattr(
        gad,
        "gad_rows",
        lambda _z: [
            {
                "system": "lj38",
                "arm": "gad_deposit",
                "provenance": "deposited",
                "seed": "1",
                "run": "1",
                "budget": gad.BUDGET,
                "reached": 1,
                "hit": "",
                "evals_lower": 640,
                "evals_upper": 640_000,
                "best": -173.928427,
                "wall_s": "",
                "inside_us": "",
                "outside_us": "",
                "samples": 64,
                "relax_steps": 1000,
                "step_source": "relaxer cap",
                "hit_generation": 10,
                "generations_charged": 10,
            }
        ],
    )

    def _forbid_gmin(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise AssertionError("saved GMIN row should have been reused or rejected before running")

    monkeypatch.setattr(gad, "run_gmin", _forbid_gmin)

    runs = tmp_path / "runs"
    saved_row = runs / "lj38" / "gmin-2000.row"

    def _write_saved(rev: str, build_key: str) -> None:
        _write_csv_with_fields(
            saved_row,
            gad.PER_SEED_FIELDS,
            [
                {
                    "system": "lj38",
                    "code": "gmin",
                    "seed": "2000",
                    "budget": str(gad.BUDGET),
                    "hit": "1",
                    "evals_to_hit": "12000",
                    "charged": "12000",
                    "wall_s": "0.600000",
                    "best": "-173.928427",
                    "pot_s": "0.500000",
                    "outside_s": "0.100000",
                    "code_revision": rev,
                    "gmin_build_key": build_key,
                    "host": "bench",
                    **FAKE_MACHINE,
                }
            ],
        )

    argv = [
        "run_campaign.py",
        "--binary",
        str(binary),
        "--gmin",
        str(gmin),
        "--data",
        str(data),
        "--zip",
        str(fake_zip),
        "--source",
        str(gmin_source),
        "--svnrev",
        str(svnrev),
        "--head-file",
        str(head_file),
        "--runs",
        str(runs),
        "--size",
        "38",
        "--seed",
        "2000",
        "--arm",
        "gmin",
    ]

    _write_saved(REVISION, WRONG_KEY)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        gad.main(argv)

    _write_saved(WRONG_REV, EXPECTED_KEY)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="code_revision"):
        gad.main(argv)

    _write_saved(REVISION, EXPECTED_KEY)
    assert gad.main(argv) == 0


def test_covalent_h2h_rejects_mismatched_build_key_and_reuses_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    covalent = _load_module("rnext5_covalent", "scripts/covalent_h2h/run_campaign.py")
    _patch_bench_env(monkeypatch, covalent)

    svnrev, sw_source = _write_gmin_meta(
        tmp_path / "meta", build_key=EXPECTED_KEY, prefix="sw"
    )
    _tsvn, tc_source = _write_gmin_meta(
        tmp_path / "meta", build_key="f" * 64, prefix="gmin-tersoff"
    )
    # Place gmin-tersoff-source.txt beside sw_source so default discovery finds it too.
    (sw_source.parent / "gmin-tersoff-source.txt").write_text(tc_source.read_text())

    head_file = tmp_path / "head"
    head_file.write_text(f"{FAKE_HEAD}\n")
    dummy = tmp_path / "dummy"
    dummy.write_text("#!/bin/sh\nexit 99\n")

    monkeypatch.setattr(covalent, "SEEDS", (2000,))

    def _fake_binary_lines(_binary: Path, mode: str, name: str) -> list[str]:
        if mode == "budget":
            return [str(covalent.PAPER_BUDGETS[name]), "-30.0", "10"]
        if mode == "scales":
            return ["1.0", "1.0", "5.0"]
        raise AssertionError(f"unexpected binary_lines {mode}")

    monkeypatch.setattr(covalent, "binary_lines", _fake_binary_lines)

    def _forbid_gmin(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise AssertionError("saved GMIN row should have been reused or rejected before running")

    monkeypatch.setattr(covalent, "run_gmin", _forbid_gmin)

    starts = tmp_path / "starts"
    for sys_name in ("sw10", "tc10"):
        start_xyz = starts / sys_name / "seed-2000.xyz"
        start_xyz.parent.mkdir(parents=True, exist_ok=True)
        start_xyz.write_text("10\n\n")

    runs = tmp_path / "runs"

    def _write_saved(sys_name: str, rev: str, build_key: str) -> None:
        _write_csv_with_fields(
            runs / sys_name / "gmin-2000.row",
            covalent.ROW_FIELDS,
            [
                {
                    "system": sys_name,
                    "code": "gmin",
                    "seed": "2000",
                    "budget": str(covalent.PAPER_BUDGETS[sys_name]),
                    "hit": "1",
                    "evals_to_hit": "5000",
                    "charged": "5000",
                    "wall_s": "0.400000",
                    "best": "-30.000000",
                    "pot_s": "0.300000",
                    "outside_s": "0.100000",
                    "code_revision": rev,
                    "gmin_build_key": build_key,
                    "host": "bench",
                    **FAKE_MACHINE,
                }
            ],
        )

    def _run_for(sys_name: str) -> int:
        rows_csv = tmp_path / f"{sys_name}-rows.csv"
        if rows_csv.exists():
            rows_csv.unlink()
        return covalent.main(
            [
                "run_campaign.py",
                "--binary",
                str(dummy),
                "--library",
                str(dummy),
                "--gmin",
                str(dummy),
                "--data",
                str(dummy),
                "--gmin-tersoff",
                str(dummy),
                "--carbon-data",
                str(dummy),
                "--ase",
                str(dummy),
                "--source",
                str(sw_source),
                "--svnrev",
                str(svnrev),
                "--head-file",
                str(head_file),
                "--runs",
                str(runs),
                "--starts",
                str(starts),
                "--rows",
                str(rows_csv),
                "--summary",
                str(tmp_path / f"{sys_name}-summary.csv"),
                "--system",
                sys_name,
                "--code",
                "gmin",
            ]
        )

    _write_saved("sw10", REVISION, WRONG_KEY)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        _run_for("sw10")

    _write_saved("sw10", WRONG_REV, EXPECTED_KEY)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="code_revision"):
        _run_for("sw10")

    # tc10 uses GMIN-tersoff's build key ("f" * 64), NOT sw10's EXPECTED_KEY ("b" * 64)!
    _write_saved("tc10", REVISION, EXPECTED_KEY)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        _run_for("tc10")

    _write_saved("sw10", REVISION, EXPECTED_KEY)
    assert _run_for("sw10") == 0

    _write_saved("tc10", REVISION, "f" * 64)
    assert _run_for("tc10") == 0


def test_metal_darby_rejects_mismatched_build_key_and_reuses_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    darby = _load_module("rnext5_darby", "scripts/metal_h2h/run_darby.py")
    _patch_bench_env(monkeypatch, darby)

    svnrev, gmin_source = _write_gmin_meta(tmp_path / "meta", build_key=EXPECTED_KEY)
    head_file = tmp_path / "head"
    head_file.write_text(f"{FAKE_HEAD}\n")
    dummy = tmp_path / "dummy"
    dummy.write_text("#!/bin/sh\nexit 99\n")

    monkeypatch.setattr(Path, "cwd", staticmethod(lambda: Path("/work")))
    monkeypatch.setattr(darby, "check_potentials", lambda *_a, **_k: None)

    def _forbid_gmin(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise AssertionError("saved GMIN row should have been reused or rejected before running")

    monkeypatch.setattr(darby, "run_gmin", _forbid_gmin)

    runs = tmp_path / "runs"
    coords = runs / "cu19au19" / "start-2000.xyz"
    coords.parent.mkdir(parents=True, exist_ok=True)
    coords.write_text("38\n\n")
    saved_row = runs / "cu19au19" / "gmin-2000.row"

    darby_fields = getattr(
        darby,
        "PER_SEED_FIELDS",
        (
            "system",
            "code",
            "seed",
            "budget",
            "hit",
            "evals_to_hit",
            "charged",
            "wall_s",
            "best",
            "pot_s",
            "outside_s",
            "code_revision",
            "gmin_build_key",
            "host",
            "machine_id",
            "cpu_model",
        ),
    )

    def _write_saved(rev: str, build_key: str) -> None:
        _write_csv_with_fields(
            saved_row,
            darby_fields,
            [
                {
                    "system": "cu19au19",
                    "code": "gmin",
                    "seed": "2000",
                    "budget": "400000",
                    "hit": "1",
                    "evals_to_hit": "15000",
                    "charged": "15000",
                    "wall_s": "0.500000",
                    "best": "-122.99148391",
                    "pot_s": "0.400000",
                    "outside_s": "0.100000",
                    "code_revision": rev,
                    "gmin_build_key": build_key,
                    "host": "bench",
                    **FAKE_MACHINE,
                }
            ],
        )

    argv = [
        "--binary",
        str(dummy),
        "--gmin",
        str(dummy),
        "--fixtures",
        str(ROOT / "tests" / "fixtures" / "ccd"),
        "--runs",
        str(runs),
        "--logs",
        str(tmp_path / "logs"),
        "--summary",
        str(tmp_path / "summary.csv"),
        "--source",
        str(gmin_source),
        "--svnrev",
        str(svnrev),
        "--head-file",
        str(head_file),
        "--only",
        "cu19au19",
        "2000",
        "gmin",
    ]

    _write_saved(REVISION, WRONG_KEY)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        darby.main(argv)

    _write_saved(WRONG_REV, EXPECTED_KEY)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="code_revision"):
        darby.main(argv)

    _write_saved(REVISION, EXPECTED_KEY)
    assert darby.main(argv) == 0


def test_metal_sc_n12_rejects_mismatched_build_key_and_reuses_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sc_n12 = _load_module("rnext5_sc_n12", "scripts/metal_h2h/run_sc_n12.py")
    _patch_bench_env(monkeypatch, sc_n12)

    svnrev, gmin_source = _write_gmin_meta(tmp_path / "meta", build_key=EXPECTED_KEY)
    head_file = tmp_path / "head"
    head_file.write_text(f"{FAKE_HEAD}\n")
    dummy = tmp_path / "dummy"
    dummy.write_text("#!/bin/sh\nexit 99\n")

    monkeypatch.setattr(Path, "cwd", staticmethod(lambda: Path("/work")))
    monkeypatch.setattr(sc_n12, "jobs", lambda: [("sc126-38", 2000, "gmin")])

    def _forbid_job(*_args: object, **_kwargs: object):
        raise AssertionError("stored GMIN row should have been reused or rejected before running")

    monkeypatch.setattr(sc_n12, "one_job", _forbid_job)

    seeds_out = tmp_path / "seeds.csv"
    summary_out = tmp_path / "summary.csv"

    def _write_seed(rev: str, build_key: str) -> None:
        _write_csv_with_fields(
            seeds_out,
            sc_n12.SEED_FIELDS,
            [
                {
                    "system": "sc126-38",
                    "arm": "gmin",
                    "seed": "2000",
                    "budget": "400000",
                    "hit": "1",
                    "best": "-173.928427",
                    "charged": "10000",
                    "evals_to_hit": "10000",
                    "wall_s": "0.500000",
                    "pot_s": "0.400000",
                    "outside_s": "0.100000",
                    "pot_per_eval_s": "0.000040",
                    "outside_per_eval_s": "0.000010",
                    "code_revision": rev,
                    "gmin_build_key": build_key,
                    "host": "bench",
                    "harness_revision": FAKE_HEAD,
                    **FAKE_MACHINE,
                }
            ],
        )

    argv = [
        "--binary",
        str(dummy),
        "--gmin",
        str(dummy),
        "--runs",
        str(tmp_path / "runs"),
        "--logs",
        str(tmp_path / "logs"),
        "--summary",
        str(summary_out),
        "--seeds-out",
        str(seeds_out),
        "--source",
        str(gmin_source),
        "--svnrev",
        str(svnrev),
        "--head-file",
        str(head_file),
    ]

    _write_seed(REVISION, WRONG_KEY)
    if summary_out.exists():
        summary_out.unlink()
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        sc_n12.main(argv)

    _write_seed(WRONG_REV, EXPECTED_KEY)
    if summary_out.exists():
        summary_out.unlink()
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="code_revision"):
        sc_n12.main(argv)

    _write_seed(REVISION, EXPECTED_KEY)
    if summary_out.exists():
        summary_out.unlink()
    assert sc_n12.main(argv) == 0


def test_metal_sc96_gmin_rejects_mismatched_build_key_and_reuses_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sc96 = _load_module("rnext5_sc96_gmin", "scripts/metal_h2h/run_sc96_gmin.py")
    _patch_bench_env(monkeypatch, sc96, sc96.n12)

    svnrev, gmin_source = _write_gmin_meta(tmp_path / "meta", build_key=EXPECTED_KEY)
    head_file = tmp_path / "head"
    head_file.write_text(f"{FAKE_HEAD}\n")
    ref78 = tmp_path / "ref78.txt"
    ref78.write_text("-500.0000\n")
    dummy = tmp_path / "dummy"
    dummy.write_text("#!/bin/sh\nexit 99\n")

    monkeypatch.setattr(Path, "cwd", staticmethod(lambda: Path("/work")))
    monkeypatch.setattr(sc96.n12, "SEEDS", (2000,))
    monkeypatch.setattr(sc96, "N78_SEEDS", (9532,))

    def _forbid_run(*_args: object, **_kwargs: object):
        raise AssertionError("stored GMIN row should have been reused or rejected before running")

    monkeypatch.setattr(sc96, "run_panel", _forbid_run)
    monkeypatch.setattr(sc96, "run_n78", _forbid_run)

    panel_seeds = tmp_path / "panel-seeds.csv"
    panel_summary = tmp_path / "panel-summary.csv"
    n78_rows = tmp_path / "n78-rows.csv"

    def _write_tables(
        panel_rev: str, panel_key: str, n78_rev: str, n78_key: str
    ) -> None:
        _write_csv_with_fields(
            panel_seeds,
            sc96.n12.SEED_FIELDS,
            [
                {
                    "system": "sc96-38",
                    "arm": "gmin",
                    "seed": "2000",
                    "budget": "400000",
                    "hit": "1",
                    "best": "-150.0",
                    "charged": "10000",
                    "evals_to_hit": "10000",
                    "wall_s": "0.500000",
                    "pot_s": "0.400000",
                    "outside_s": "0.100000",
                    "pot_per_eval_s": "0.000040",
                    "outside_per_eval_s": "0.000010",
                    "code_revision": panel_rev,
                    "gmin_build_key": panel_key,
                    "host": "bench",
                    "harness_revision": FAKE_HEAD,
                    **FAKE_MACHINE,
                }
            ],
        )
        _write_csv_with_fields(
            n78_rows,
            sc96.N78_FIELDS,
            [
                {
                    "arm": "gmin",
                    "seed": "9532",
                    "hit": "true",
                    "hit_calls": "50000",
                    "spent": "50000",
                    "best": "-500.0",
                    "reference": "-500.0000",
                    "code_revision": n78_rev,
                    "gmin_build_key": n78_key,
                    "host": "bench",
                    "harness_revision": FAKE_HEAD,
                    **FAKE_MACHINE,
                    "wall_s": "1.000000",
                    "pot_s": "0.900000",
                }
            ],
        )

    argv = [
        "--binary",
        str(dummy),
        "--gmin",
        str(dummy),
        "--runs",
        str(tmp_path / "runs"),
        "--logs",
        str(tmp_path / "logs"),
        "--panel-seeds",
        str(panel_seeds),
        "--panel-summary",
        str(panel_summary),
        "--n78-rows",
        str(n78_rows),
        "--starts78",
        str(tmp_path / "starts78"),
        "--reference78",
        str(ref78),
        "--source",
        str(gmin_source),
        "--svnrev",
        str(svnrev),
        "--head-file",
        str(head_file),
        "--panel-system",
        "sc96-38",
    ]

    _write_tables(REVISION, WRONG_KEY, REVISION, EXPECTED_KEY)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        sc96.main(argv)

    _write_tables(REVISION, EXPECTED_KEY, REVISION, WRONG_KEY)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        sc96.main(argv)

    _write_tables(WRONG_REV, EXPECTED_KEY, REVISION, EXPECTED_KEY)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="code_revision"):
        sc96.main(argv)

    _write_tables(REVISION, EXPECTED_KEY, REVISION, EXPECTED_KEY)
    assert sc96.main(argv) == 0


def test_tip4p_water_gmin_rejects_mismatched_build_key_and_reuses_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    water = _load_module("rnext5_water_gmin", "scripts/tip4p_h2h/run_water_gmin.py")
    _patch_bench_env(monkeypatch, water)

    svnrev, gmin_source = _write_gmin_meta(tmp_path / "meta", build_key=EXPECTED_KEY)
    monkeypatch.setenv("GMIN_SOURCE_FILE", str(gmin_source))
    monkeypatch.setattr(Path, "cwd", staticmethod(lambda: Path("/work")))
    monkeypatch.setattr(water, "write_gm_blocks", lambda *_a, **_k: None)
    monkeypatch.setattr(water, "verify", lambda *_a, **_k: None)

    def _forbid_run(*_args: object, **_kwargs: object):
        raise AssertionError("stored row should have been reused or rejected before running")

    monkeypatch.setattr(water, "run_anneal", _forbid_run)
    monkeypatch.setattr(water, "run_gmin", _forbid_run)

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    start_file = work_dir / "tip4p8-s2000.start.gmin"
    start_file.write_text("0.0 0.0 0.0\n")
    start_hash = water.sha1(start_file)
    dummy = tmp_path / "dummy"
    dummy.write_text("#!/bin/sh\nexit 99\n")
    rows_csv = tmp_path / "rows.csv"

    def _write_rows(gmin_rev: str, build_key: str) -> None:
        _write_csv_with_fields(
            rows_csv,
            water.FIELDS,
            [
                {
                    "system": "tip4p8",
                    "code": "anneal",
                    "seed": "2000",
                    "budget": "2000000",
                    "hit": "1",
                    "evals_to_hit": "5000",
                    "charged": "5000",
                    "best": "-305.5183221676",
                    "wall_s": "0.400000",
                    "pot_s": "0.300000",
                    "outside_s": "0.100000",
                    "basins": "5",
                    "start_sha1": start_hash,
                    "code_revision": FAKE_HEAD,
                    "gmin_build_key": "",
                    "host": "bench",
                    "cpu": FAKE_MACHINE["cpu_model"],
                    **FAKE_MACHINE,
                },
                {
                    "system": "tip4p8",
                    "code": "gmin",
                    "seed": "2000",
                    "budget": "2000000",
                    "hit": "1",
                    "evals_to_hit": "8000",
                    "charged": "8000",
                    "best": "-305.5183221676",
                    "wall_s": "0.700000",
                    "pot_s": "0.600000",
                    "outside_s": "0.100000",
                    "basins": "8",
                    "start_sha1": start_hash,
                    "code_revision": gmin_rev,
                    "gmin_build_key": build_key,
                    "host": "bench",
                    "cpu": FAKE_MACHINE["cpu_model"],
                    **FAKE_MACHINE,
                },
            ],
        )

    argv = [
        "run_water_gmin.py",
        "--repo",
        str(ROOT),
        "--binary",
        str(dummy),
        "--point",
        str(dummy),
        "--anneal-revision",
        FAKE_HEAD,
        "--gmin",
        str(dummy),
        "--gmin-svnrev",
        str(svnrev),
        "--data",
        str(dummy),
        "--verify-data",
        str(dummy),
        "--gm-dir",
        str(tmp_path / "gm"),
        "--verify-out",
        str(tmp_path / "verify.csv"),
        "--work",
        str(work_dir),
        "--rows",
        str(rows_csv),
        "--summary",
        str(tmp_path / "summary.csv"),
        "--size",
        "8",
        "--seed",
        "2000",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    _write_rows(REVISION, WRONG_KEY)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        water.main()

    _write_rows(WRONG_REV, EXPECTED_KEY)
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="code_revision"):
        water.main()

    _write_rows(REVISION, EXPECTED_KEY)
    assert water.main() == 0


def test_metal_gold_shell_rejects_mismatched_build_key_and_stamps_gmin_provenance(
    tmp_path: Path,
) -> None:
    svnrev, gmin_source = _write_gmin_meta(tmp_path / "meta", build_key=EXPECTED_KEY)
    shims = tmp_path / "shims"
    shims.mkdir()
    for cmd_name, body in (
        ("hostname", "#!/bin/sh\necho bench\n"),
        ("taskset", '#!/bin/sh\nshift 2\nexec "$@"\n'),
    ):
        p = shims / cmd_name
        p.write_text(body)
        p.chmod(0o755)

    fake_bin = tmp_path / "h2h_metal"
    fake_bin.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "coords" ]; then\n'
        '  printf "75\\n\\n" > "$4"\n'
        "  exit 0\n"
        "fi\n"
        'sys="$2"; seed="$3"\n'
        'echo "$sys,anneal,$seed,4000000,false,4000000,4000000,0.5,-300.0,0.4,0.1"\n'
    )
    fake_bin.chmod(0o755)

    fake_gmin = tmp_path / "GMIN"
    fake_gmin.write_text(
        "#!/bin/sh\n"
        'echo "h2h_row system=$H2H_SYSTEM code=gmin seed=$H2H_SEED budget=$H2H_BUDGET '
        'hit=0 evals_to_hit=$H2H_BUDGET charged=$H2H_BUDGET best=-300.0 wall_s=0.6 pot_s=0.5" '
        '> "$H2H_ROW"\n'
    )
    fake_gmin.chmod(0o755)

    stored_csv = tmp_path / "stored-seeds.csv"
    gold_script = (ROOT / "scripts" / "metal_h2h" / "enter_gold.sh").read_text()
    gold_fields = [
        "system",
        "code",
        "seed",
        "budget",
        "hit",
        "evals_to_hit",
        "charged",
        "wall_s",
        "best",
        "pot_s",
        "outside_s",
        "revision",
        "code_revision",
        *(["gmin_build_key"] if "gmin_build_key" in gold_script else []),
        "host",
        "machine_id",
        "cpu_model",
    ]
    _write_csv_with_fields(
        stored_csv,
        gold_fields,
        [
            {
                "system": "au75",
                "code": "gmin",
                "seed": "2000",
                "budget": "4000000",
                "hit": "false",
                "evals_to_hit": "4000000",
                "charged": "4000000",
                "wall_s": "0.600000",
                "best": "-300.0",
                "pot_s": "0.500000",
                "outside_s": "0.100000",
                "revision": "testrev12345",
                "code_revision": REVISION,
                "gmin_build_key": WRONG_KEY,
                "host": "bench",
                **FAKE_MACHINE,
            }
        ],
    )
    _write_csv_with_fields(tmp_path / "work" / "rows.csv", gold_fields, [])

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{shims}{os.pathsep}{env.get('PATH', '')}",
            "BIN": str(fake_bin),
            "GMIN": str(fake_gmin),
            "WORK": str(tmp_path / "work"),
            "GOLD_SEEDS_OUT": str(stored_csv),
            "GOLD_REVISION": "testrev12345",
            "GMIN_SVNREV": str(svnrev),
            "GMIN_SOURCE": str(gmin_source),
        }
    )

    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / "metal_h2h" / "enter_gold.sh"), "--from-large"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "gmin_build_key" in (proc.stderr + proc.stdout)

    # Matching build_key runs cleanly and stamps GMIN_REVISION + GMIN_BUILD_KEY on gmin rows
    seq_shim = shims / "seq"
    seq_shim.write_text("#!/bin/sh\necho 2000\n")
    seq_shim.chmod(0o755)
    stored_csv.unlink()
    (tmp_path / "work" / "rows.csv").unlink()
    ok = subprocess.run(
        ["bash", str(ROOT / "scripts" / "metal_h2h" / "enter_gold.sh"), "--from-large"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr + ok.stdout
    with stored_csv.open() as handle:
        out_rows = list(csv.DictReader(handle))
    gmin_rows = [r for r in out_rows if r["code"] == "gmin"]
    anneal_rows = [r for r in out_rows if r["code"] == "anneal"]
    assert len(gmin_rows) == 2
    assert len(anneal_rows) == 2
    assert all(r["code_revision"] == REVISION for r in gmin_rows)
    assert all(r["gmin_build_key"] == EXPECTED_KEY for r in gmin_rows)
    assert all(r["gmin_build_key"] == "" for r in anneal_rows)


def test_au55_scale_and_morse14_h2h_reject_mismatched_build_key_and_reuse_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    au55 = _load_module("rnext5_au55_scale", "scripts/au55_scale/run_panel.py")
    morse14 = _load_module("rnext5_morse14_h2h", "scripts/morse14_h2h/run_panel.py")
    _patch_bench_env(monkeypatch, au55, morse14)

    svnrev, gmin_source = _write_gmin_meta(tmp_path / "meta", build_key=EXPECTED_KEY)
    monkeypatch.setenv("GMIN_SVNREV_FILE", str(svnrev))
    monkeypatch.setenv("GMIN_SOURCE_FILE", str(gmin_source))

    fake_root = tmp_path / "repo"
    au55_csv = fake_root / "bench" / "au55-gupta" / "seeds.csv"
    au55_fields = list(getattr(au55, "ROW_FIELDS", au55.SEED_FIELDS)) + [
        col
        for col in ("code_revision", "gmin_build_key", "host", "machine_id", "cpu_model")
        if col not in getattr(au55, "ROW_FIELDS", au55.SEED_FIELDS)
    ]
    _write_csv_with_fields(
        au55_csv,
        au55_fields,
        [
            {
                "system": "au55",
                "code": "gmin",
                "seed": "2000",
                "budget": "1000000",
                "hit": "1",
                "evals_to_hit": "20000",
                "charged": "20000",
                "wall_s": "0.600000",
                "best": "-191.03085280",
                "pot_s": "0.500000",
                "outside_s": "0.100000",
                "code_revision": REVISION,
                "gmin_build_key": WRONG_KEY,
                "host": "bench",
                **FAKE_MACHINE,
            }
        ],
    )
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        au55.stored_gmin(fake_root)

    morse14_csv = fake_root / "bench" / "morse-h2h" / "seeds.csv"
    morse14_fields = list(morse14.ROW_FIELDS) + [
        col
        for col in ("gmin_build_key", "machine_id", "cpu_model")
        if col not in morse14.ROW_FIELDS
    ]
    _write_csv_with_fields(
        morse14_csv,
        morse14_fields,
        [
            {
                "system": "m55r14",
                "code": "gmin",
                "seed": "2000",
                "budget": "1000000",
                "hit": "1",
                "evals_to_hit": "20000",
                "charged": "20000",
                "wall_s": "0.600000",
                "best": "-220.646208",
                "pot_s": "0.500000",
                "outside_s": "0.100000",
                "code_revision": REVISION,
                "gmin_build_key": WRONG_KEY,
                "host": "bench",
                **FAKE_MACHINE,
            }
        ],
    )
    with pytest.raises((ValueError, RuntimeError, SystemExit), match="gmin_build_key"):
        morse14.load_gmin(morse14_csv)

    _write_csv_with_fields(
        au55_csv,
        au55_fields,
        [
            {
                "system": "au55",
                "code": "gmin",
                "seed": "2000",
                "budget": "1000000",
                "hit": "1",
                "evals_to_hit": "20000",
                "charged": "20000",
                "wall_s": "0.600000",
                "best": "-191.03085280",
                "pot_s": "0.500000",
                "outside_s": "0.100000",
                "code_revision": REVISION,
                "gmin_build_key": EXPECTED_KEY,
                "host": "bench",
                **FAKE_MACHINE,
            }
        ],
    )
    assert len(au55.stored_gmin(fake_root)) == 1

    _write_csv_with_fields(
        morse14_csv,
        morse14_fields,
        [
            {
                "system": "m55r14",
                "code": "gmin",
                "seed": "2000",
                "budget": "1000000",
                "hit": "1",
                "evals_to_hit": "20000",
                "charged": "20000",
                "wall_s": "0.600000",
                "best": "-220.646208",
                "pot_s": "0.500000",
                "outside_s": "0.100000",
                "code_revision": REVISION,
                "gmin_build_key": EXPECTED_KEY,
                "host": "bench",
                **FAKE_MACHINE,
            }
        ],
    )
    assert ("m55r14", 2000) in morse14.load_gmin(morse14_csv)
