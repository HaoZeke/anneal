from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def test_bootstrap_passes_absolute_install_prefixes_to_meson(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    prefix_log = tmp_path / "prefixes.txt"

    _executable(
        fake_bin / "git",
        """#!/usr/bin/env bash
set -euo pipefail
dst="${@: -1}"
mkdir -p "$dst/.git"
""",
    )
    _executable(
        fake_bin / "ninja",
        """#!/usr/bin/env bash
exit 0
""",
    )
    _executable(
        fake_bin / "meson",
        """#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  setup)
    prefix="${3#--prefix=}"
    case "$prefix" in /*) ;; *) exit 91 ;; esac
    printf '%s\n' "$prefix" >> "$PREFIX_LOG"
    mkdir -p builddir
    touch builddir/build.ninja
    ;;
  compile) ;;
  install)
    case "$PWD" in
      */SIFDecode) mkdir -p install/bin; touch install/bin/sifdecoder ;;
      */CUTEst)
        mkdir -p install/lib
        touch install/lib/libcutest_single.a install/lib/libcutest_double.a
        ;;
    esac
    ;;
esac
""",
    )

    script = Path(__file__).parents[1] / "benchmarks" / "bootstrap_cutest.sh"
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PREFIX_LOG": str(prefix_log),
    }
    subprocess.run(
        ["bash", str(script), "project"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    prefixes = prefix_log.read_text().splitlines()
    assert prefixes == [
        str(project / ".bench" / "SIFDecode" / "install"),
        str(project / ".bench" / "CUTEst" / "install"),
    ]
