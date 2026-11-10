#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Show seal_checks.sh rejects a kit whose OUTER is the bundle sha."""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path


def digest(path: Path) -> str:
    hashed = hashlib.sha256()
    hashed.update(path.read_bytes())
    return hashed.hexdigest()


def write_kit(root: Path, outer_text: str) -> None:
    report = root / "REPORT"
    bundle = root / "series.bundle"
    report.write_text("report\n")
    bundle.write_bytes(b"bundle-bytes\n")
    lines = [
        f"{digest(report)}  REPORT\n",
        f"{digest(bundle)}  series.bundle\n",
    ]
    sums = root / "SHA256SUMS"
    sums.write_text("".join(lines))
    (root / "OUTER").write_text(outer_text + "\n")
    ready = "\n".join(
        [
            f"OUTER {outer_text}",
            "TIP " + "a" * 40,
            "TREE " + "b" * 40,
            f"BUNDLE_SHA256 {digest(bundle)}",
        ]
    )
    (root / "STORE_READY").write_text(ready + "\n")
    stamp = 1_700_000_000
    for name in ("REPORT", "series.bundle", "SHA256SUMS", "OUTER", "STORE_READY"):
        os.utime(root / name, (stamp, stamp))
        stamp += 1


def run(root: Path) -> subprocess.CompletedProcess[str]:
    script = Path(__file__).resolve().parent / "seal_checks.sh"
    return subprocess.run(
        ["sh", str(script), str(root)],
        text=True,
        capture_output=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad"
        bad.mkdir()
        bundle_sha = hashlib.sha256(b"bundle-bytes\n").hexdigest()
        write_kit(bad, bundle_sha)
        failed = run(bad)
        sys_out = (failed.stderr or failed.stdout).strip()
        print(sys_out)
        if failed.returncode == 0:
            raise SystemExit("bad kit was accepted")
        if "bundle sha" not in sys_out:
            raise SystemExit("bad kit failed for another reason")
        good = Path(tmp) / "good"
        good.mkdir()
        write_kit(good, "0" * 64)
        sums_sha = digest(good / "SHA256SUMS")
        write_kit(good, sums_sha)
        passed = run(good)
        if passed.returncode != 0:
            raise SystemExit(passed.stderr or passed.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
