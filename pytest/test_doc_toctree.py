"""Every page under docs/orgmode is in a toctree that the index reaches.

``docs/export.el`` turns each Org page under ``docs/orgmode`` into the reST
page of the same name under ``docs/source``, and Sphinx builds all of them.
The site's navigation is its toctrees, starting at ``index``: a page that no
toctree reaches is built but never shown, and the build warns that it is not
included in any toctree. So every page but the index must be an entry of a
toctree on the index or on a page the index reaches, and every entry must name
a page.

A toctree is read where ox-rst passes it through unchanged, in an Org page's
``#+begin_export rst`` blocks; one shown inside a source, example or comment
block is text. A ``.. toctree::`` line opens the directive and its content is
the lines indented under it. Option lines such as ``:maxdepth:``,
``:caption:`` and ``:glob:`` are skipped, and the other lines are entries,
read as Sphinx reads them. ``Title <name>`` names ``name``, a trailing
``.rst`` or ``.md`` (the build's source suffixes) is dropped, and a name is
read from the directory of the page that holds the toctree, or from the top
of ``docs/source`` when it starts with ``/``. ``self`` and a URL name no
page. Under ``:glob:``, an untitled entry
holding ``*`` or ``?`` is a pattern over the other pages, whose ``*`` and
``?`` stay within one directory while ``**`` crosses them.
"""

from __future__ import annotations

import functools
import posixpath
import re
import textwrap
from collections.abc import Collection
from pathlib import Path

import pytest

ORG = Path(__file__).resolve().parents[1] / "docs" / "orgmode"

_TEXT_BLOCK = re.compile(
    r"^[ \t]*#\+begin_(src|example|comment)\b.*?^[ \t]*#\+end_\1\b[^\n]*$",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
_RST_BLOCK = re.compile(
    r"^[ \t]*#\+begin_export[ \t]+rst\b[^\n]*\n(.*?)^[ \t]*#\+end_export\b",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
_TOCTREE = re.compile(r"([ \t]*)\.\.[ \t]+toctree::[ \t]*")
_TITLED = re.compile(r"(.+?)\s*<([^<]*?)>", re.DOTALL)
_URL = re.compile(r".+://")
_GLOB = re.compile(r"[*?]")
_SUFFIX = re.compile(r"\.(?:rst|md)$")


def pages() -> list[str]:
    """Each Org page's name in Sphinx: its path under docs/orgmode less ``.org``."""
    return sorted(
        path.relative_to(ORG).with_suffix("").as_posix() for path in ORG.rglob("*.org")
    )


def toctrees(text: str) -> list[tuple[bool, list[str]]]:
    """Whether each toctree of the Org page ``text`` globs, and its entries."""
    found = []
    for block in _RST_BLOCK.finditer(_TEXT_BLOCK.sub("", text)):
        lines = block.group(1).splitlines()
        for at, line in enumerate(lines):
            directive = _TOCTREE.fullmatch(line)
            if not directive:
                continue
            indent, glob, entries = len(directive.group(1)), False, []
            for line in lines[at + 1 :]:
                entry = line.strip()
                if entry and len(line) - len(line.lstrip()) <= indent:
                    break
                if entry.startswith(":"):
                    glob = glob or entry.startswith(":glob:")
                elif entry:
                    entries.append(entry)
            found.append((glob, entries))
    return found


def _join(page: str, name: str) -> str:
    """``name`` read from the directory of ``page``, or from the top after a ``/``."""
    return posixpath.normpath(posixpath.join("/" + page, "..", name))[1:]


def _pattern(glob: str) -> re.Pattern[str]:
    parts = re.split(r"(\*\*|\*|\?)", glob)
    table = {"**": ".*", "*": "[^/]*", "?": "[^/]"}
    return re.compile("".join(table.get(part, re.escape(part)) for part in parts))


def entries(
    page: str, text: str, names: Collection[str]
) -> list[tuple[str, list[str]]]:
    """Each toctree entry of ``page``, whose source is ``text``, and what it names.

    What it names is the pages among ``names`` it reaches; ``self`` and URLs,
    which name no page by design, are left out.
    """
    named = []
    for glob, lines in toctrees(text):
        for entry in lines:
            titled = _TITLED.fullmatch(entry)
            if _URL.match(entry):
                continue
            if glob and not titled and _GLOB.search(entry):
                pattern = _pattern(_join(page, entry))
                found = [
                    name
                    for name in sorted(names)
                    if name != page and pattern.fullmatch(name)
                ]
                named.append((entry, found))
                continue
            target = titled.group(2) if titled else entry
            if target == "self":
                continue
            target = _join(page, _SUFFIX.sub("", target))
            named.append((entry, [target] if target in names else []))
    return named


def _source(page: str) -> str:
    return (ORG / f"{page}.org").read_text(encoding="utf-8")


@functools.cache
def reached() -> frozenset[str]:
    """The pages the index reaches through toctrees, the index among them."""
    names = frozenset(pages())
    seen, todo = {"index"}, ["index"]
    while todo:
        page = todo.pop()
        for _, found in entries(page, _source(page), names):
            for name in found:
                if name not in seen:
                    seen.add(name)
                    todo.append(name)
    return frozenset(seen)


@pytest.mark.parametrize("page", [page for page in pages() if page != "index"])
def test_every_page_is_in_a_toctree_the_index_reaches(page):
    assert page in reached(), (
        f"docs/orgmode/{page}.org is in no toctree the index reaches; "
        "list it in the toctree of its section in docs/orgmode/index.org"
    )


def test_every_toctree_entry_names_a_page():
    names = frozenset(pages())
    broken = [
        f"docs/orgmode/{page}.org: {entry}"
        for page in sorted(names)
        for entry, found in entries(page, _source(page), names)
        if not found
    ]
    assert not broken, "toctree entries naming no page:\n" + "\n".join(broken)


def test_the_index_and_its_pages_are_read():
    assert {"index", "quickstart"} <= set(pages())
    assert "quickstart" in reached()


def test_toctrees_are_read_as_sphinx_reads_them():
    text = textwrap.dedent(
        """\
        #+begin_export rst
        .. toctree::
           :maxdepth: 1
           :caption: Guides

           a
           Second <b.rst>
           /c/d.md
           ../e
           missing
           self
           Home <self>
           https://x.org
        Prose after the directive.

        .. toctree::
           :glob:

           f/*
           Titled <f/*>
        #+end_export
        #+begin_example
        #+begin_export rst
        .. toctree::

           g
        #+end_export
        #+end_example
        #+begin_src rst
        .. toctree::

           h
        #+end_src
        """
    )
    assert toctrees(text) == [
        (
            False,
            [
                "a",
                "Second <b.rst>",
                "/c/d.md",
                "../e",
                "missing",
                "self",
                "Home <self>",
                "https://x.org",
            ],
        ),
        (True, ["f/*", "Titled <f/*>"]),
    ]
    names = {"x/a", "x/b", "c/d", "e", "x/f/g", "x/f/h/i", "x/y"}
    assert entries("x/y", text, names) == [
        ("a", ["x/a"]),
        ("Second <b.rst>", ["x/b"]),
        ("/c/d.md", ["c/d"]),
        ("../e", ["e"]),
        ("missing", []),
        ("f/*", ["x/f/g"]),
        ("Titled <f/*>", []),
    ]
