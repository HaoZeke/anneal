# Copyright (c) anneal developers
"""Every tracked Python file parses on the oldest supported interpreter.

pyproject.toml supports Python 3.10. PEP 701 lets an f-string reuse its own
quote or hold a backslash inside a replacement field, and only 3.12 and newer
accept that. On 3.10 and 3.11 compile() rejects such a file, so the check is
exact there. On 3.12 and newer the tokenizer sees the replacement fields, and
the second test finds the same spellings.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def tracked_python_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [ROOT / name for name in out.split("\0") if name]


def fstring_spellings_before_312(source: str) -> list[tuple[int, str]]:
    """Replacement-field text that only Python 3.12 and newer accepts."""
    found: list[tuple[int, str]] = []
    outer: list[tuple[str, int]] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.FSTRING_START:
            quote = token.string.lstrip("rRbBfF")
            if outer and outer[-1][1] > 0 and (len(outer[-1][0]) == 1 or len(quote) == 3):
                if quote[0] == outer[-1][0][0]:
                    found.append((token.start[0], "nested f-string reuses the outer quote"))
            outer.append((quote, 0))
        elif token.type == tokenize.FSTRING_END:
            outer.pop()
        elif outer and token.type == tokenize.OP and token.string == "{":
            outer[-1] = (outer[-1][0], outer[-1][1] + 1)
        elif outer and token.type == tokenize.OP and token.string == "}" and outer[-1][1] > 0:
            outer[-1] = (outer[-1][0], outer[-1][1] - 1)
        elif outer and outer[-1][1] > 0 and token.type == tokenize.STRING:
            body = token.string.lstrip("rRbBuU")
            if body[0] == outer[-1][0][0] and len(outer[-1][0]) == 1:
                found.append((token.start[0], "string reuses the outer quote"))
            if "\\" in token.string:
                found.append((token.start[0], "backslash inside a replacement field"))
    return found


def test_every_tracked_python_file_compiles_on_this_interpreter():
    failures = []
    for path in tracked_python_files():
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
        except SyntaxError as error:
            failures.append(f"{path.relative_to(ROOT)}:{error.lineno}: {error.msg}")
    assert failures == []


@pytest.mark.skipif(sys.version_info < (3, 12), reason="older interpreters reject these spellings at compile time")
def test_no_fstring_needs_python_312():
    failures = []
    for path in tracked_python_files():
        for line, why in fstring_spellings_before_312(path.read_text(encoding="utf-8")):
            failures.append(f"{path.relative_to(ROOT)}:{line}: {why}")
    assert failures == []


@pytest.mark.skipif(sys.version_info < (3, 12), reason="the scan reads 3.12 tokens")
def test_the_scan_finds_the_spellings_it_looks_for():
    reused = "row = {'a': 1}\ntext = f'{row['a']}'\n"
    backslash = 'text = f"{pin(\'\\"x\\"\')}"\n'
    clean = "row = {'a': 1}\ntext = f\"{row['a']}\"\n"
    assert [why for _line, why in fstring_spellings_before_312(reused)] == ["string reuses the outer quote"]
    assert any("backslash" in why for _line, why in fstring_spellings_before_312(backslash))
    assert fstring_spellings_before_312(clean) == []
