#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write bundle-verify.txt from a full-history bundle of HEAD.

The bundle includes the branch and every refs/archive/prereg ref.
No prerequisite is excluded. Absolute paths are replaced with PATH.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout.strip() or "git failed")
    return proc.stdout


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            root = candidate
            break
    tip = git(root, "rev-parse", "HEAD").strip()
    refs = ["refs/heads/prereg/lj104-sota-v1"]
    listed = git(root, "for-each-ref", "--format=%(refname)", "refs/archive/prereg")
    refs.extend(line for line in listed.splitlines() if line)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = str(Path(tmp) / "verify.bundle")
        empty = str(Path(tmp) / "empty")
        git(root, "bundle", "create", bundle, *refs)
        subprocess.run(["git", "init", "-q", empty], check=True)
        proc = subprocess.run(
            ["git", "-C", empty, "bundle", "verify", bundle],
            check=False,
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout or "verify failed")
    raw = (proc.stdout + proc.stderr).strip()
    text = re.sub(r"(^|\s)(/[^\s]+)", r"\1PATH", raw, flags=re.M)
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and lines[0].endswith("is okay"):
        lines[0] = "The bundle is okay."
    if any("requires" in line.lower() for line in lines):
        raise SystemExit("bundle has a requires line")
    if not any("complete history" in line for line in lines):
        raise SystemExit("bundle is not a complete history")
    body = [
        "Copyright (c) anneal developers",
        "",
        "We record the verify of a full-history bundle.",
        "The verified tip follows.",
        f"{tip}.",
        *lines,
        "VERIFY_EXIT:0",
        "",
    ]
    (here / "bundle-verify.txt").write_text("\n".join(body), encoding="utf-8")
    print(f"refs {len(refs)}")
    print(tip)


if __name__ == "__main__":
    sys.exit(main() or 0)
