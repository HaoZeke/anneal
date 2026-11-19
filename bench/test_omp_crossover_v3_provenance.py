#!/usr/bin/env python3
"""Require every recorded arm tree to resolve in the repository."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.json.read_text(encoding="utf-8"))
    for arm in payload["metadata"]["arms"]:
        tree = arm.get("tree")
        if not tree:
            raise SystemExit(f"{arm['name']} has no tree")
        resolved_type = subprocess.run(
            ["git", "cat-file", "-t", tree],
            cwd=args.repo,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if resolved_type != "tree":
            raise SystemExit(f"{arm['name']} object is not a tree")


if __name__ == "__main__":
    main()
