# Copyright (c) anneal developers
"""Check that Org docs under docs/orgmode do not put ``=`` inside ``=...=``.

Org closes a ``=...=`` verbatim span at the first ``=`` preceded by
non-whitespace and followed by a post-boundary character (such as a quote,
space, period, or closing parenthesis). Signatures or expressions containing
``=`` (for example ``layout="block"`` or ``low == high``) must use ``~...~``
code markup or a source block instead, and slash-joined tokens must use
spaces around ``/`` rather than ``=A=/=B=`` or ``~A=/=B~``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ORG_ROOT = ROOT / "docs" / "orgmode"

# Org default `org-emphasis-regexp-components` boundaries.
PRE = r"(?:^|(?<=[-\s('\"{]))"
POST = r"(?:$|(?=[-\s.,:!?;\"')}\[]))"
POST_CLOSING = r"(?:$|(?=[-\s.,:!?;\"')}\]]))"

_CONT_LINE = r"\n(?![ \t]*(?:\*+ |- |\+ |\d+[.)] |\||$))[^\n]*?"
_SPAN_BODY = rf"(?:\S|\S[^\n]*?(?:{_CONT_LINE})?\S)"

CODE_RE = re.compile(rf"{PRE}~({_SPAN_BODY})~{POST}")
VERB_RE = re.compile(rf"{PRE}=({_SPAN_BODY})={POST}")
LINK_RE = re.compile(r"\[\[[^\]]+\](?:\[[^\]]*\])?\]")
MATH_RE = re.compile(r"\\\(.*?\\\)|\\\[.*?\\\]", re.DOTALL)

OPEN_EQ_RE = re.compile(rf"{PRE}=(?=[^\s=>])")
CLOSE_EQ_RE = re.compile(rf"(?<=[^\s<>!+\-*/:=])={POST_CLOSING}")
MANGLED_SLASH_RE = re.compile(r"=/\s*=")

_BLOCK_BEGIN_RE = re.compile(
    r"#\+begin_(?:src|example|export|comment|verbatim)\b", re.IGNORECASE
)
_BLOCK_END_RE = re.compile(
    r"#\+end_(?:src|example|export|comment|verbatim)\b", re.IGNORECASE
)


def _strip_blocks(text: str) -> str:
    """Blank out verbatim/src/export blocks, keywords, and comments."""
    out: list[str] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.lstrip()
        if not in_block and _BLOCK_BEGIN_RE.match(stripped):
            in_block = True
            out.append("")
            continue
        if in_block:
            if _BLOCK_END_RE.match(stripped):
                in_block = False
            out.append("")
            continue
        if (
            stripped.startswith("#+")
            or stripped.startswith("# ")
            or stripped == "#"
            or stripped.startswith(":")
        ):
            out.append("")
            continue
        out.append(line)
    return "\n".join(out)


def _mask(match: re.Match[str]) -> str:
    return re.sub(r"[^\n]", " ", match.group(0))


def _line_number(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def find_verbatim_equals(text: str) -> list[tuple[int, str, str]]:
    """Return ``(line_no, kind, excerpt)`` for every broken Org verbatim span."""
    lines = text.splitlines()
    cleaned = _strip_blocks(text)
    findings: list[tuple[int, str, str]] = []

    for match in CODE_RE.finditer(cleaned):
        if MANGLED_SLASH_RE.search(match.group(1)):
            findings.append(
                (
                    _line_number(cleaned, match.start()),
                    "code-contains-slash-eq",
                    match.group(0),
                )
            )

    work = MATH_RE.sub(_mask, LINK_RE.sub(_mask, CODE_RE.sub(_mask, cleaned)))

    for match in VERB_RE.finditer(work):
        if "=" in match.group(1):
            findings.append(
                (
                    _line_number(work, match.start()),
                    "verbatim-contains-eq",
                    match.group(0),
                )
            )

    residual = VERB_RE.sub(_mask, work)
    for match in CLOSE_EQ_RE.finditer(residual):
        ln = _line_number(residual, match.start())
        findings.append((ln, "stray-closing-eq", lines[ln - 1].strip()))
    for match in OPEN_EQ_RE.finditer(residual):
        ln = _line_number(residual, match.start())
        findings.append((ln, "stray-opening-eq", lines[ln - 1].strip()))

    findings.sort(key=lambda item: (item[0], item[1]))
    return findings


def _org_files() -> list[Path]:
    return sorted(
        path
        for path in ORG_ROOT.rglob("*.org")
        if not path.name.startswith(".")
    )


def test_org_docs_have_no_equals_inside_verbatim_spans():
    files = _org_files()
    assert files, f"no .org files found under {ORG_ROOT}"
    failures: list[str] = []
    for path in files:
        rel = path.relative_to(ROOT)
        for line_no, kind, excerpt in find_verbatim_equals(
            path.read_text(encoding="utf-8")
        ):
            failures.append(f"{rel}:{line_no} [{kind}]: {excerpt}")
    assert not failures, (
        "Org verbatim spans (=...=) must not contain '='; use ~...~ or a "
        "#+begin_src block:\n" + "\n".join(failures)
    )


@pytest.mark.parametrize(
    "sample",
    [
        '- =cluster_search(obj, layout="block", target=None)= :: desc',
        '- =global_optimize(obj, policy="auto", x0=None)= :: desc',
        'With =policy="auto"=, no =grad_fn=.',
        "At =cap >= 50000=, =N >= 70=.",
        "A coordinate with =low == high= stays fixed.",
        "Use =rigid=True= or =water_sites=.",
        "Variants =Same=/=Known=/=New=.",
        "Variants ~Same=/=Known=/=New~.",
        "- =polish(obj, grad, max_fevals=200, ...)",
    ],
)
def test_checker_rejects_equals_in_verbatim_spans(sample: str):
    assert find_verbatim_equals(sample)


@pytest.mark.parametrize(
    "sample",
    [
        '- ~cluster_search(obj, layout="block", target=None)~ :: desc',
        '- ~global_optimize(obj, policy="auto", x0=None)~ :: desc',
        'With ~policy="auto"~, no =grad_fn=.',
        "At ~cap >= 50000~, ~N >= 70~.",
        "A coordinate with ~low == high~ stays fixed.",
        "Use ~rigid=True~ or =water_sites=.",
        "Variants =Same= / =Known= / =New=.",
        "Plain prose: L3 downhill=1, T(k+1) <= T(k), alpha=[4,1], p = 0.05.",
        "#+begin_src python\nx = cluster_search(layout='block')\n#+end_src\n",
    ],
)
def test_checker_accepts_code_markup_and_plain_prose(sample: str):
    assert find_verbatim_equals(sample) == []
