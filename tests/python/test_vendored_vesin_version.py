"""Keep vendored vesin source and wheel constraints in lockstep."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VESIN = ROOT / "third_party" / "vesin"
POTCTL_CI = ROOT / "potctl" / "src" / "ci.rs"


def _read_ci_spec(name: str) -> str:
    text = POTCTL_CI.read_text()
    match = re.search(rf'pub const {name}: &str = "([^"]+)";', text)
    assert match is not None
    return match.group(1)


def test_vendored_vesin_version_matches_source() -> None:
    declared = (VESIN / "VERSION").read_text().strip()
    header = (VESIN / "vesin-single-build.cpp").read_text()
    match = re.search(r"^// vesin version: (\S+)$", header, re.MULTILINE)
    assert match is not None
    assert declared == match.group(1)


def test_potctl_vesin_specs_match_vendored_minor_series() -> None:
    vendored = (VESIN / "VERSION").read_text().strip()
    major, minor, *_ = vendored.split(".")
    next_minor = f"{major}.{int(minor) + 1}"
    for name in ("VESIN_SPEC", "VESIN_TORCH_SPEC"):
        spec = _read_ci_spec(name)
        assert f">={major}.{minor}" in spec
        assert f"<{next_minor}" in spec
