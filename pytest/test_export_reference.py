# Copyright (c) anneal developers
"""Reference pages list the current public exports and no others."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

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


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize_use(body: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(body):
        if body[index].isspace():
            index += 1
            continue
        if body.startswith("::", index):
            tokens.append("::")
            index += 2
            continue
        if body[index] in "{}*,":
            tokens.append(body[index])
            index += 1
            continue
        match = _IDENT.match(body, index)
        if match:
            tokens.append(match.group(0))
            index = match.end()
            continue
        raise AssertionError(f"unparsed pub use syntax near {body[index:index + 20]!r}")
    return tokens


def _exported(name: str) -> bool:
    return name not in _RUST_KEYWORDS and name != "_"


def _parse_use_tree(tokens: list[str], index: int, source: str) -> tuple[set[str], int]:
    if index >= len(tokens) or _IDENT.fullmatch(tokens[index]) is None:
        raise AssertionError(f"pub use tree does not start with a name: {source}")
    last = tokens[index]
    index += 1
    while (
        index + 1 < len(tokens)
        and tokens[index] == "::"
        and _IDENT.fullmatch(tokens[index + 1]) is not None
    ):
        last = tokens[index + 1]
        index += 2
    names: set[str] = set()
    if index < len(tokens) and tokens[index] == "::":
        nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
        if nxt == "*":
            raise AssertionError(f"pub use glob is not expanded: {source}")
        if nxt != "{":
            raise AssertionError(f"unparsed pub use syntax: {source}")
        index += 2
        if index < len(tokens) and tokens[index] == "}":
            return names, index + 1
        while True:
            found, index = _parse_use_tree(tokens, index, source)
            names.update(found)
            if index < len(tokens) and tokens[index] == ",":
                index += 1
                if index < len(tokens) and tokens[index] == "}":
                    return names, index + 1
                continue
            if index < len(tokens) and tokens[index] == "}":
                return names, index + 1
            raise AssertionError(f"unclosed pub use group: {source}")
    if index < len(tokens) and tokens[index] == "as":
        if index + 1 >= len(tokens) or _IDENT.fullmatch(tokens[index + 1]) is None:
            raise AssertionError(f"pub use rename lacks a name: {source}")
        exported = tokens[index + 1]
        if _exported(exported):
            names.add(exported)
        return names, index + 2
    if _exported(last):
        names.add(last)
    return names, index


def _pub_use_names(text: str) -> set[str]:
    stripped = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    stripped = re.sub(r"//.*?$", "", stripped, flags=re.M)
    names: set[str] = set()
    for match in re.finditer(r"pub use\s+(.*?);", stripped, flags=re.S):
        body = match.group(1).strip()
        if not body:
            raise AssertionError("empty pub use")
        tokens = _tokenize_use(body)
        found, index = _parse_use_tree(tokens, 0, body)
        if index != len(tokens):
            raise AssertionError(f"unparsed pub use tail: {body}")
        names.update(found)
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


def test_pub_use_glob_is_rejected():
    with pytest.raises(AssertionError, match="glob"):
        _pub_use_names("pub use crate::methods::*;")
    with pytest.raises(AssertionError, match="glob"):
        _pub_use_names("pub use crate::methods::{bar::*, Baz};")


def test_pub_use_rename_counts_exported_name():
    text = "\n".join(
        [
            "pub use crate::methods::Bar as Baz;",
            "pub use crate::methods::{A as B, C, D as _};",
            "pub use crate::noise_accept::{OsaAccept, OsaResult};",
        ]
    )
    assert _pub_use_names(text) == {"Baz", "B", "C", "OsaAccept", "OsaResult"}


def test_rust_reference_lists_current_reexports():
    expected = _pub_use_names(LIB_RS.read_text(encoding="utf-8"))
    listed = _marked_names(RUST_API, "# BEGIN-RUST-EXPORTS", "# END-RUST-EXPORTS")
    assert listed == expected, (
        f"missing {sorted(expected - listed)}; "
        f"extra {sorted(listed - expected)}"
    )
