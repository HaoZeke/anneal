"""Unit tests for Meson's Cargo archive wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("meson_cargo_build.py")
SPEC = importlib.util.spec_from_file_location("meson_cargo_build", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reproducible_archive_is_opt_in() -> None:
    root = Path("/work/source")
    target = Path("/work/build/deep")
    disabled = MODULE.reproducible_env({}, root, target)
    assert disabled == {}
    enabled = MODULE.reproducible_env(
        {"ANNEAL_REPRODUCIBLE_ARCHIVE": "1"}, root, target
    )
    assert enabled["CARGO_INCREMENTAL"] == "0"
    assert "--remap-path-prefix /work/source=/source" in enabled["RUSTFLAGS"]
    assert "--remap-path-prefix /work/build/deep=/build" in enabled["RUSTFLAGS"]
    assert "-Ccodegen-units=1" in enabled["RUSTFLAGS"]
