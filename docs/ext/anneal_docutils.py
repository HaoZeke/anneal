"""Repair generated rustdoc RST before Sphinx reads it.

sphinx-rustdoc-postprocess converts markdown inline code, and its pattern
refuses ``<`` and ``>``. A span such as ``gamma > 1`` is left with a single
backtick. Wrapped spans and LaTeX bars then become docutils warnings.
This module joins wrapped spans and widens that pattern before the
converter runs, then escapes the forms the converter leaves behind.
"""

from __future__ import annotations

import re
from pathlib import Path

_INLINE_CODE = re.compile(r"(?<!`)(`)((?!`)(?:[^`\n])+)\1(?!`)(?!_)")
_MATH = re.compile(r"\\\((.+?)\\\)")
_LATEX_BAR = re.compile(r"\|([^|\n]*[_\\^][^|\n]*)\|")
_LIST_ITEM = re.compile(r"^([ ]+)(?:\d+\.|[-*]) ")
_TABLE_LINE = re.compile(r"^[ ]*(?:\|.*\||\+[-=+]+)\s*$")
_LITERAL_TAIL = re.compile(r"(?<=[A-Za-z0-9_>\])])``(?=[A-Za-z0-9_])")
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_DOUBLE = re.compile(r"``.*?``", re.DOTALL)


def _join_wrapped_code(text: str) -> str:
    """Pull a single-backtick span that rustdoc wrapped back onto one line."""
    parts = []
    cursor = 0
    for fence in _FENCE.finditer(text):
        parts.append(_join_region(text[cursor : fence.start()]))
        parts.append(fence.group(0))
        cursor = fence.end()
    parts.append(_join_region(text[cursor:]))
    return "".join(parts)


def _join_region(text: str) -> str:
    parts = []
    cursor = 0
    index = 0
    while index < len(text):
        if text.startswith("``", index):
            end = text.find("``", index + 2)
            if end < 0:
                break
            index = end + 2
            continue
        if text[index] != "`":
            index += 1
            continue
        end = text.find("`", index + 1)
        if end < 0:
            break
        span = text[index + 1 : end]
        if "\n" not in span:
            index = end + 1
            continue
        parts.append(text[cursor:index])
        parts.append("`" + re.sub(r"\n[ ]*", " ", span) + "`")
        cursor = end + 1
        index = cursor
    parts.append(text[cursor:])
    return "".join(parts)


def _prepare_generated(_app) -> None:
    import sphinx_rustdoc_postprocess as post

    post._INLINE_CODE_RE = _INLINE_CODE
    root = Path(_app.srcdir) / _app.config.rustdoc_postprocess_rst_dir
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.rst")):
        original = path.read_text(encoding="utf-8")
        joined = _join_wrapped_code(original)
        if joined != original:
            path.write_text(joined, encoding="utf-8")


def _repair_generated(_app) -> None:
    root = Path(_app.srcdir) / _app.config.rustdoc_postprocess_rst_dir
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.rst")):
        original = path.read_text(encoding="utf-8")
        repaired = _repair(original)
        if repaired != original:
            path.write_text(repaired, encoding="utf-8")


def _outside_literals(line: str) -> str:
    if _TABLE_LINE.match(line):
        return line
    pieces = _DOUBLE.split(line)
    literals = _DOUBLE.findall(line)
    rebuilt = []
    for index, piece in enumerate(pieces):
        rebuilt.append(_LATEX_BAR.sub(lambda match: rf"\|{match.group(1)}\|", piece))
        if index < len(literals):
            rebuilt.append(literals[index])
    return _LITERAL_TAIL.sub(r"``\\", "".join(rebuilt))


def _math(match: re.Match[str]) -> str:
    body = match.group(1).replace("\\", "\\\\")
    return f":math:`{body}`"


def _repair(text: str) -> str:
    text = re.sub(
        r"\\\((.*?)\\\)",
        lambda match: match.group(0).replace("\n", " "),
        text,
        flags=re.DOTALL,
    )
    lines = []
    in_code = False
    previous_blank = True
    previous_table = False
    for line in text.split("\n"):
        stripped = line.lstrip()
        if (
            stripped.startswith(".. ")
            and lines
            and lines[-1].strip()
            and not lines[-1].lstrip().startswith("..")
            and not lines[-1].lstrip().startswith(":")
        ):
            lines.append("")
            previous_blank = True
        if stripped.startswith(".. code-block") or stripped.startswith("::"):
            in_code = True
            lines.append(line)
            previous_blank = False
            previous_table = False
            continue
        if in_code:
            if stripped and not line.startswith(" ") and not line.startswith("\t"):
                in_code = False
            else:
                lines.append(line)
                previous_blank = not stripped
                previous_table = False
                continue
        table = _TABLE_LINE.match(line) is not None
        if (
            lines
            and not previous_blank
            and (_LIST_ITEM.match(line) or (table and not previous_table))
            and not lines[-1].lstrip().startswith("..")
        ):
            lines.append("")
        line = _MATH.sub(_math, line)
        line = _outside_literals(line)
        if previous_table and not table and line.strip() and not line.lstrip().startswith(".."):
            lines.append("")
        lines.append(line)
        previous_blank = not line.strip()
        previous_table = table
    return "\n".join(lines)


def setup(app):
    app.connect("builder-inited", _prepare_generated, priority=550)
    app.connect("builder-inited", _repair_generated, priority=700)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
