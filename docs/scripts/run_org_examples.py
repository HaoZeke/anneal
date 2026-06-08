#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026-present Rohit Goswami
# SPDX-License-Identifier: MIT
"""Placeholder: extract and execute Python code blocks from org-mode files.

Modeled on rgpycrumbs/scripts/run_org_doctests.py .
To be wired into pixi [feature.docs.tasks] and CI once tutorials land.
Extracts #+begin_src python ... #+end_src (skipping :eval no or SKIP-DOCTEST),
runs in shared namespace for basic smoke of stable API examples.
"""
import re
import sys
from pathlib import Path


def extract_python_blocks(org_path: Path) -> list[tuple[int, str]]:
    """Extract Python code blocks from an org-mode file.

    Returns list of (line_number, code_string) tuples.
    """
    text = org_path.read_text()
    blocks = []
    pattern = re.compile(
        r"#\+begin_src python([^\n]*)\n(.*?)#\+end_src",
        re.DOTALL | re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        header_args = match.group(1)
        code = match.group(2)
        line = text[: match.start()].count("\n") + 1
        if ":eval no" in header_args or "SKIP-DOCTEST" in code:
            continue
        blocks.append((line, code))
    return blocks


def run_blocks(org_path: Path, blocks: list[tuple[int, str]]) -> bool:
    """Run extracted code blocks in sequence with shared namespace."""
    namespace: dict = {}
    all_passed = True
    for i, (line, code) in enumerate(blocks, 1):
        try:
            exec(compile(code, str(org_path), "exec"), namespace)  # noqa: S102
            print(f"  Block {i} (line {line}): OK")
        except Exception as e:
            print(f"  Block {i} (line {line}): FAILED")
            print(f"    {type(e).__name__}: {e}")
            all_passed = False
    return all_passed


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.org> [file2.org ...]")
        return 1

    all_passed = True
    for path_str in sys.argv[1:]:
        org_path = Path(path_str)
        if not org_path.exists():
            print(f"File not found: {org_path}")
            all_passed = False
            continue

        blocks = extract_python_blocks(org_path)
        if not blocks:
            print(f"{org_path}: no Python blocks found, skipping")
            continue

        print(f"{org_path}: {len(blocks)} Python block(s)")
        if not run_blocks(org_path, blocks):
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
