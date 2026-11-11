#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write TREE_DIFF_STAT.txt from git numstat against the v1.2 tree.

The note's own path is omitted so a later commit of this file does not
change the counts of the other paths.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from prose_rate import balance

BASE = "4b168122d30bc00afa0e27a26c67863ffc7b312c"
SELF = "docs/prereg/lj104-sota/TREE_DIFF_STAT.txt"


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            root = candidate
            break
    from note_tip import stat_tree

    proc = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--numstat",
            BASE,
            stat_tree(root),
            "--",
            ".",
            f":(exclude){SELF}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or "git diff failed")
    rows = [line for line in proc.stdout.splitlines() if line.strip()]
    if not any(line.endswith("bundle-verify.txt") for line in rows):
        raise SystemExit("bundle-verify.txt is absent from the diff")
    lines = [
        "Copyright (c) anneal developers",
        "",
        "We record this stat against the v1.2 tree.",
        "That tree follows.",
        f"{BASE}.",
        "We leave this note out of its own count.",
        "Insertions and deletions come from git numstat.",
        "",
    ]
    for row in rows:
        ins, dele, path = row.split("\t")
        left = f"Path `{path}`."
        right = f"{ins} insertions and {dele} deletions."
        if len(left) > 72 or len(right) > 72:
            raise SystemExit(f"stat line over 72: {len(left)} {left}")
        lines.append(left)
        lines.append(right)
    if not any("PROSE-RECEIPT.txt" in line for line in lines):
        raise SystemExit("PROSE-RECEIPT.txt is absent from the diff")
    lines = balance(lines)
    lines.append("")
    text = "\n".join(lines)
    for line in text.splitlines():
        if not line.startswith("docs/") and len(line) > 72:
            raise SystemExit(f"line over 72: {line}")
    (here / "TREE_DIFF_STAT.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main() or 0)
