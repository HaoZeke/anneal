"""Keep the SBOM-facing vendored vesin version in step with the source."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VESIN = ROOT / "third_party" / "vesin"


def test_vendored_vesin_version_matches_source() -> None:
    declared = (VESIN / "VERSION").read_text().strip()
    header = (VESIN / "vesin-single-build.cpp").read_text()
    match = re.search(r"^// vesin version: (\S+)$", header, re.MULTILINE)
    assert match is not None
    assert declared == match.group(1)
