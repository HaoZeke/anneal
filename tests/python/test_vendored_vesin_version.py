"""Keep vendored vesin source and wheel constraints in lockstep."""

# Copyright 2023--present rgpot developers

import re

import pytest

from repo_data import source_file, source_root


_ROOT = source_root()
VESIN = source_file(_ROOT, "third_party/vesin/VERSION").parent
POTCTL_CI = source_file(_ROOT, "potctl/src/ci.rs")
source_file(_ROOT, "third_party/vesin/vesin-single-build.cpp")

pytestmark = pytest.mark.skipif(
    _ROOT is None,
    reason="rgpot source tree is not available",
)


def _read_vendored_version() -> str:
    versions = [
        line.strip()
        for line in (VESIN / "VERSION").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(versions) == 1
    return versions[0]


def _read_ci_spec(name: str) -> str:
    text = POTCTL_CI.read_text()
    match = re.search(rf'pub const {name}: &str = "([^"]+)";', text)
    assert match is not None
    return match.group(1)


def test_vendored_vesin_version_matches_source() -> None:
    declared = _read_vendored_version()
    header = (VESIN / "vesin-single-build.cpp").read_text()
    match = re.search(r"^// vesin version: (\S+)$", header, re.MULTILINE)
    assert match is not None
    assert declared == match.group(1)


def test_potctl_vesin_specs_match_vendored_minor_series() -> None:
    vendored = _read_vendored_version()
    major, minor, *_ = vendored.split(".")
    next_minor = f"{major}.{int(minor) + 1}"
    for name in ("VESIN_SPEC", "VESIN_TORCH_SPEC"):
        spec = _read_ci_spec(name)
        assert f">={major}.{minor}" in spec
        assert f"<{next_minor}" in spec
