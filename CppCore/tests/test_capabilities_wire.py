#!/usr/bin/env python3
"""Cross-version wire check for Potentials.capnp Capabilities.

A message written before buildVersion/buildRevision (@18, @19) existed must
decode under the current schema with both fields empty, and a message that
carries them must decode under the older definition with every earlier field
intact. The older definition is the whole pre-build-identity Potentials.capnp
(CppCore/tests/fixtures), and every member it declares must still exist in the
current file with the same ordinal and type (append-only evolution).
Usage: test_capabilities_wire.py <root> [<capnp>]
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

BASE = (
    'backendName = "cpmdc", backendVersion = "1.2", abiVersion = 3, '
    'available = true, operations = [energy, forces], '
    'protocolFamily = "rgpot.potentials", protocolMajor = 1, protocolMinor = 1, '
    'schemaId = "0xbd1f89fa17369103", bridgeAbiMajor = 1, bridgeLayout = 1, '
    "dlpackMajor = 1, bridgeFeatures = 3"
)
BUILD = 'buildVersion = "3.5.0", buildRevision = "abc123def456"'


def run(capnp: str, mode: str, schema: Path, text: str | None = None, data: bytes | None = None) -> bytes:
    cmd = [capnp, mode, "--packed", str(schema), "Capabilities"]
    if mode == "decode":
        cmd.insert(2, "--short")
    out = subprocess.run(
        cmd,
        input=data if data is not None else text.encode(),
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return out


def fields(decoded: str) -> dict[str, str]:
    inner = decoded.strip()[1:-1]
    parts = re.split(r",\s*(?=[a-zA-Z]+ = )", inner)
    return dict(p.split(" = ", 1) for p in parts)


OPENERS = re.compile(r"^\s*(struct|enum|union|interface|group)\b\s*(\w*)")
MEMBER = re.compile(r"^\s*(\w+)\s+@(\d+)\s*(.*?)\s*(?:#.*)?$")


def members(text: str) -> set[tuple[str, str, int, str]]:
    """(scope, name, ordinal, type text) for every ordinal-bearing member."""
    scope: list[str] = []
    depth_at_open: list[int] = []
    depth = 0
    found: set[tuple[str, str, int, str]] = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0]
        opener = OPENERS.match(line)
        if opener and "{" in line:
            scope.append(opener.group(2) or opener.group(1))
            depth_at_open.append(depth)
        elif (m := MEMBER.match(line)) and scope:
            kind = re.sub(r"\s+", " ", m.group(3).rstrip(";").strip())
            kind = re.sub(r"\s*=.*$", "", kind)
            found.add(("/".join(scope), m.group(1), int(m.group(2)), kind))
        depth += line.count("{") - line.count("}")
        while depth_at_open and depth <= depth_at_open[-1]:
            depth_at_open.pop()
            scope.pop()
    return found


def main() -> int:
    root = Path(sys.argv[1])
    capnp = sys.argv[2] if len(sys.argv) > 2 else shutil.which("capnp")
    if capnp is None:
        raise AssertionError("capnp is required")
    new = root / "CppCore" / "rgpot" / "rpc" / "Potentials.capnp"
    old = root / "CppCore" / "tests" / "fixtures" / "Potentials.pre-build-identity.capnp"

    # Append-only: nothing the older file declares may move, rename or retype.
    old_members = members(old.read_text(encoding="utf-8"))
    new_members = members(new.read_text(encoding="utf-8"))
    assert len(old_members) > 500, len(old_members)
    missing = sorted(old_members - new_members)
    if missing:
        raise AssertionError(f"members changed or removed since the older schema: {missing[:5]}")
    added = {m for m in new_members - old_members if m[0] == "Capabilities"}
    assert {(m[1], m[2]) for m in added} >= {("buildVersion", 18), ("buildRevision", 19)}, added

    schema_text = new.read_text(encoding="utf-8")
    for ordinal, name in ((18, "buildVersion"), (19, "buildRevision")):
        if not re.search(rf"{name}\s+@{ordinal}\s*:\s*Text", schema_text):
            raise AssertionError(f"{name} must be Text @{ordinal} (append only)")

    # New fields written, read by the older definition.
    written_new = run(capnp, "encode", new, text=f"({BASE}, {BUILD})")
    seen_by_old = fields(run(capnp, "decode", old, data=written_new).decode())
    expected = fields(f"({BASE})")
    for key in ("backendName", "protocolFamily", "protocolMajor", "bridgeFeatures", "schemaId"):
        assert seen_by_old[key] == expected[key], (key, seen_by_old[key])
    assert "buildVersion" not in seen_by_old

    # Older message, read by the current definition.
    written_old = run(capnp, "encode", old, text=f"({BASE})")
    seen_by_new = fields(run(capnp, "decode", new, data=written_old).decode())
    for key in ("backendName", "protocolFamily", "protocolMajor", "bridgeFeatures", "schemaId"):
        assert seen_by_new[key] == expected[key], (key, seen_by_new[key])
    assert seen_by_new.get("buildVersion", '""') == '""', seen_by_new
    assert seen_by_new.get("buildRevision", '""') == '""', seen_by_new

    # Round trip under the current definition keeps the new fields.
    round_trip = fields(run(capnp, "decode", new, data=written_new).decode())
    assert round_trip["buildVersion"] == '"3.5.0"', round_trip
    assert round_trip["buildRevision"] == '"abc123def456"', round_trip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
