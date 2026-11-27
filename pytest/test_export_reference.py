# Copyright (c) anneal developers
"""Reference pages list the current public exports and no others."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "docs/orgmode/reference/presets-and-advanced.org"
RUST_API = ROOT / "docs/orgmode/reference/rust-api.org"
INIT = ROOT / "python/anneal/__init__.py"
CORE_PYI = ROOT / "python/anneal/_core.pyi"
CHEMFIT = ROOT / "python/anneal/chemfit.py"
LIB_RS = ROOT / "src/lib.rs"

_BLOCK_BEGIN = re.compile(
    r"#\+begin_(?:src|example|export|comment|verbatim)\b", re.IGNORECASE
)
_BLOCK_END = re.compile(
    r"#\+end_(?:src|example|export|comment|verbatim)\b", re.IGNORECASE
)
_MARKUP_NAME = re.compile(r"=([A-Za-z_][A-Za-z0-9_]*)=")
_RUST_KEYWORDS = {"as", "pub", "use", "crate", "self", "super"}


def _assigned_list(path: Path, name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                found = [elt.value for elt in node.value.elts]
    if found is None:
        raise AssertionError(f"{path} has no {name}")
    return found


def _strip_blocks(text: str) -> str:
    out: list[str] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.lstrip()
        if not in_block and _BLOCK_BEGIN.match(stripped):
            in_block = True
            continue
        if in_block:
            if _BLOCK_END.match(stripped):
                in_block = False
            continue
        out.append(line)
    return "\n".join(out)


def _marked(text: str, begin: str, end: str) -> str:
    start = text.index(begin) + len(begin)
    stop = text.index(end, start)
    return text[start:stop]


def _marked_names(path: Path, begin: str, end: str) -> set[str]:
    body = _strip_blocks(_marked(path.read_text(encoding="utf-8"), begin, end))
    return set(_MARKUP_NAME.findall(body))


def _pub_use_names(text: str) -> set[str]:
    stripped = re.sub(r"//.*?$", "", text, flags=re.M)
    names: set[str] = set()
    for match in re.finditer(r"pub use\s+(.*?);", stripped, flags=re.S):
        body = re.sub(r"\b[a-z_][a-z0-9_]*::", "", match.group(1))
        for ident in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", body):
            if ident not in _RUST_KEYWORDS:
                names.add(ident)
    return names


def test_python_reference_lists_current_exports():
    expected = set(_assigned_list(INIT, "__all__"))
    expected.update(_assigned_list(CORE_PYI, "__all__"))
    expected.update(_assigned_list(CHEMFIT, "__all__"))
    listed = _marked_names(PRESETS, "# BEGIN-PY-EXPORTS", "# END-PY-EXPORTS")
    assert listed == expected, (
        f"missing {sorted(expected - listed)}; "
        f"extra {sorted(listed - expected)}"
    )


def test_rust_reference_lists_current_reexports():
    expected = _pub_use_names(LIB_RS.read_text(encoding="utf-8"))
    listed = _marked_names(RUST_API, "# BEGIN-RUST-EXPORTS", "# END-RUST-EXPORTS")
    assert listed == expected, (
        f"missing {sorted(expected - listed)}; "
        f"extra {sorted(listed - expected)}"
    )
