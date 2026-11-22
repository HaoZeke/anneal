# Copyright 2023--present rgpot developers
"""Toolchain and archiver choice of scripts/build_rust_core.py.

A fake rustc on PATH reports the workspace rust-version, 1.88, and a fake
rustup has no 1.89.0 toolchain. rgpot-core on its own must build with the
cargo it is given, and only the bundle asks for 1.89.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import build_rust_core

FAKE_CARGO = """\
import json, os, sys
argv = sys.argv[1:]
target = argv[argv.index("--target-dir") + 1]
out = os.path.join(target, "debug", "librgpot_core.a")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "wb") as handle:
    handle.write(b"!<arch>\\n")
with open(os.environ["FAKE_CARGO_LOG"], "w") as handle:
    json.dump({"argv": argv, "ar": os.environ.get("AR")}, handle)
"""


def _exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def rust_188(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _exe(bin_dir / "rustc", "#!/bin/sh\necho 'release: 1.88.0'\n")
    _exe(bin_dir / "rustup", "#!/bin/sh\nexit 1\n")
    cargo = bin_dir / "cargo"
    _exe(cargo, f"#!{sys.executable}\n{FAKE_CARGO}")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CARGO_LOG", str(tmp_path / "cargo.json"))
    monkeypatch.setenv("AR", "arm64-apple-darwin20.0.0-ar")
    return tmp_path, cargo


def _run(tmp_path: Path, cargo: Path, *extra: str, monkeypatch) -> dict:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_rust_core.py",
            "--cargo",
            str(cargo),
            "--manifest",
            "rgpot-core/Cargo.toml",
            "--target-dir",
            str(tmp_path / "target"),
            "--profile",
            "debug",
            "--artifact",
            "librgpot_core.a",
            *extra,
            str(tmp_path / "librgpot_core.a"),
            str(tmp_path / "rgpot_rust_built.h"),
        ],
    )
    build_rust_core.main()
    return json.loads((tmp_path / "cargo.json").read_text())


def test_core_only_builds_on_188(rust_188, monkeypatch):
    tmp_path, cargo = rust_188
    log = _run(tmp_path, cargo, monkeypatch=monkeypatch)
    assert (tmp_path / "librgpot_core.a").read_bytes() == b"!<arch>\n"
    assert (tmp_path / "rgpot_rust_built.h").is_file()
    assert "--locked" in log["argv"]


def test_fast_toolchain_needs_189(rust_188, monkeypatch):
    tmp_path, cargo = rust_188
    with pytest.raises(SystemExit, match="rustc >= 1.89"):
        _run(tmp_path, cargo, "--fast-toolchain", monkeypatch=monkeypatch)


@pytest.mark.skipif(
    not (os.access("/usr/bin/ar", os.X_OK) and os.access("/usr/bin/ranlib", os.X_OK)),
    reason="needs /usr/bin/ar and /usr/bin/ranlib",
)
def test_core_only_uses_system_ar_on_darwin(rust_188, monkeypatch):
    tmp_path, cargo = rust_188
    monkeypatch.setattr(sys, "platform", "darwin")
    log = _run(tmp_path, cargo, monkeypatch=monkeypatch)
    assert log["ar"] == "/usr/bin/ar"
    assert log["argv"][0] == "build"
