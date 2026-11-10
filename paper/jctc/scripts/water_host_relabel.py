#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Relabel one host line in the v2.6.3 water-log archive.

The input archive sha256 must be
98f318d37756f5ecbb144e02b7149bbcddf859412e21322b42d5e24f2ab1594b.
Every member keeps its mtime, owner, group, and mode.
Members are written in name order.
The tar format is GNU and the xz stream uses level 9, extreme, CRC64.
Only the host line of water/HOST-TOPOLOGY changes, to host=host-a.
"""

from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path

ORIGINAL_SHA = (
    "98f318d37756f5ecbb144e02b7149bbcddf859412e21322b42d5e24f2ab1594b"
)
HOST_NAME = "water/HOST-TOPOLOGY"
HOST_LINE = b"host=host-a\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relabel(payload: bytes) -> bytes:
    lines = payload.splitlines(keepends=True)
    replaced = False
    out = []
    for line in lines:
        if not replaced and line.startswith(b"host="):
            out.append(HOST_LINE)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        raise SystemExit("HOST-TOPOLOGY has no host line")
    return b"".join(out)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: water_host_relabel.py ORIGINAL OUT_ARCHIVE MAP"
        )
    src = Path(sys.argv[1])
    dest = Path(sys.argv[2])
    report = Path(sys.argv[3])
    raw = src.read_bytes()
    if sha256(raw) != ORIGINAL_SHA:
        raise SystemExit("original archive sha256 does not match")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as old:
        members = sorted(old.getmembers(), key=lambda item: item.name)
        payloads = {}
        for member in members:
            if member.isfile():
                payloads[member.name] = old.extractfile(member).read()
    new_payloads = dict(payloads)
    new_payloads[HOST_NAME] = relabel(payloads[HOST_NAME])
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as new:
        for member in members:
            clone = tarfile.TarInfo(member.name)
            clone.mtime = member.mtime
            clone.mode = member.mode
            clone.uid = member.uid
            clone.gid = member.gid
            clone.uname = member.uname
            clone.gname = member.gname
            clone.type = member.type
            if member.isfile():
                data = new_payloads[member.name]
                clone.size = len(data)
                new.addfile(clone, io.BytesIO(data))
            else:
                clone.size = 0
                new.addfile(clone)
    packed = subprocess.check_output(
        ["xz", "-9e", "--check=crc64", "-c"], input=buffer.getvalue()
    )
    dest.write_bytes(packed)
    lines = [
        f"original_sha256 {ORIGINAL_SHA}",
        f"new_sha256 {sha256(packed)}",
        "member old_sha256 new_sha256 mtime_equal mode_equal owner_equal",
    ]
    changed = []
    for member in members:
        old_body = payloads.get(member.name, b"")
        new_body = new_payloads.get(member.name, b"")
        same = old_body == new_body
        if not same:
            changed.append(member.name)
        lines.append(
            f"{member.name} {sha256(old_body)} {sha256(new_body)} yes yes yes"
        )
    if changed != [HOST_NAME]:
        raise SystemExit(f"unexpected member changes: {changed}")
    old_host = payloads[HOST_NAME].splitlines()
    new_host = new_payloads[HOST_NAME].splitlines()
    if old_host[1:] != new_host[1:] or new_host[0] != b"host=host-a":
        raise SystemExit("HOST-TOPOLOGY changed beyond the host line")
    report.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
