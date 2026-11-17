"""The CPU scaling harness writes three CSV tables with a host header."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.thread_unsafe,
    pytest.mark.iterations(1),
    pytest.mark.skipif(
        "TSAN_OPTIONS" in os.environ,
        reason="ThreadSanitizer cannot finish this timing run inside the pytest timeout",
    ),
]

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cpu_scaling.py"


def _build_dir(root: Path) -> Path:
    info = root / "meson-info"
    info.mkdir(parents=True)
    compilers = {"host": {"cpp": {"id": "gcc", "version": "13.3.0"}}}
    (info / "intro-compilers.json").write_text(json.dumps(compilers))
    options = [{"name": "buildtype", "value": "release"}]
    (info / "intro-buildoptions.json").write_text(json.dumps(options))
    return root


def _harness(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(_SCRIPT), "--build-dir",
           str(_build_dir(tmp_path / "build")), "--out", str(tmp_path / "out")]
    return subprocess.run(cmd + list(extra), capture_output=True, text=True,
                          check=False)


def _read(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    lines = path.read_text(encoding="ascii").splitlines()
    header = dict(line[2:].split(": ", 1) for line in lines if line[0] == "#")
    rows = list(csv.DictReader(line for line in lines if line[0] != "#"))
    return header, rows


def test_tables_match_serial(tmp_path: Path) -> None:
    proc = _harness(tmp_path, "--threads", "2", "--repeats", "6")
    assert proc.returncode == 0, proc.stderr
    cells = {"strong": [16, 16], "onecell": [1, 1], "weak": [8, 16]}
    for name, want in cells.items():
        header, rows = _read(tmp_path / "out" / f"{name}.csv")
        assert {"uname_m", "cpu_model", "cores"} <= header.keys()
        assert header["compilers"] == "cpp=gcc-13.3.0"
        assert header["build_flags"] == "buildtype=release"
        assert [int(r["threads"]) for r in rows] == [1, 2]
        assert [int(r["cells"]) for r in rows] == want
        for row in rows:
            assert row["match"] == "true"
            assert int(row["repeats"]) >= 6
            q1, mid, q3 = (float(row[k]) for k in ("q1_s", "median_s", "q3_s"))
            assert 0 < q1 <= mid <= q3


def test_rejects_fewer_than_six_repeats(tmp_path: Path) -> None:
    proc = _harness(tmp_path, "--repeats", "5")
    assert proc.returncode == 2
    assert not (tmp_path / "out").exists()
