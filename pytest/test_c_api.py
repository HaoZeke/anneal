# Copyright (c) anneal developers
"""The C ABI: the generated header, and the smoke program against it.

``tests/ffi_smoke.c`` runs the box ensemble through the header cbindgen
generates, so a setting that the Rust options carry and the header lacks
fails here as a compile error, and a setting that does not reach the
ensemble fails as a check.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FFI_RS = ROOT / "src" / "ffi.rs"
SCRIPT = ROOT / "scripts" / "ffi_smoke.sh"


def _needs_toolchain():
    for tool in ("cargo", "cbindgen", "cc"):
        if shutil.which(tool) is None:
            pytest.skip(f"needs {tool} on PATH")
    if not (ROOT / "Cargo.toml").exists():
        pytest.skip("needs the source checkout")


@pytest.fixture(scope="module")
def header(tmp_path_factory):
    _needs_toolchain()
    out = tmp_path_factory.mktemp("header") / "anneal-core.h"
    subprocess.run(
        [
            "cbindgen",
            "--config",
            str(ROOT / "cbindgen.toml"),
            "--crate",
            "anneal-core",
            "--output",
            str(out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return out.read_text()


def _exported(name):
    return re.findall(rf'pub (?:unsafe )?extern "C" fn ({name}\w*)\(', FFI_RS.read_text())


def test_header_declares_every_exported_function(header):
    names = _exported("anneal_")
    assert "anneal_box_minimize_ex" in names
    missing = [n for n in names if not re.search(rf"\b{n}\(", header)]
    assert not missing, f"exported but not in the header: {missing}"


def test_header_names_the_enums_the_options_hold_as_integers(header):
    for enum in (
        "anneal_box_history_t",
        "anneal_box_membership_t",
        "anneal_box_visit_policy_t",
        "anneal_box_coverage_t",
    ):
        assert re.search(rf"\}} {enum};", header), enum


def test_c_smoke_program_passes():
    _needs_toolchain()
    result = subprocess.run(
        ["bash", str(SCRIPT)], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
