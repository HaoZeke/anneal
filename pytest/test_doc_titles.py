"""Every page under docs/orgmode gives Sphinx one title of its own.

``docs/export.el`` turns each Org page under ``docs/orgmode`` into the reST
page of the same name under ``docs/source``. Sphinx reads the sections at a
page's outermost level as its titles: the first names the page in the
navigation, the browser tab and links to it, and each other one becomes an
entry of its own wherever a toctree lists the page. So a page must give Sphinx
exactly one outermost title, its sections under it, and that title must name
the page as its ``#+TITLE`` does: hold the words of the ``#+TITLE``, compared
without case or Org and reST code marks.

ox-rst writes the ``#+TITLE`` as a title over- and underlined with ``=``,
unless ``#+OPTIONS`` sets ``title:nil``, and underlines each headline with the
character for its level counted from the page's outermost headlines: ``-``,
``~``, ``^``, ``:``, ``'``, a space (no title at all) and ``_``. A headline
marked ``COMMENT`` or tagged ``noexport`` is left out with all that lies under
it, and a source, example or comment block is text. An rst export block passes
through unchanged. A title in it is a line at its left margin above a line of
one repeated punctuation mark as long as the title or at least four long, or a
line between two equal lines of one repeated mark; a line of marks above text
but unlike the line below it makes no title. docutils ranks title styles, the
mark and whether it also overlines, in the order they first appear, so the
outermost titles are those in the style of the first. A page that sets
``title:nil`` must therefore give its title itself: in an rst block before its
first headline, as the tutorials do, or as its only outermost headline.
Sections the build adds itself are not read, such as the Rust API toctree that
``docs/source/conf.py`` appends to ``reference/rust-api``.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

ORG = Path(__file__).resolve().parents[1] / "docs" / "orgmode"

_TEXT_BLOCK = re.compile(
    r"^[ \t]*#\+begin_(src|example|comment)\b.*?^[ \t]*#\+end_\1\b[^\n]*$",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
_BEGIN_EXPORT = re.compile(r"[ \t]*#\+begin_export[ \t]+(\S+).*", re.IGNORECASE)
_END_EXPORT = re.compile(r"[ \t]*#\+end_export\b.*", re.IGNORECASE)
_RST = {"rst", "rest", "restructuredtext"}
_HEADLINE = re.compile(r"(\*+)[ \t]+(.*?)(?:[ \t]+(:[\w@#%:]+:))?[ \t]*")
_ADORNMENT = re.compile(r"([!-/:-@\[-`{-~])\1*")
_UNDERLINES = "-~^:' _"
_MARKS = re.compile(r"[=~`]")

Style = tuple[str, bool]


def pages() -> list[str]:
    """Each Org page's name in Sphinx: its path under docs/orgmode less ``.org``."""
    return sorted(
        path.relative_to(ORG).with_suffix("").as_posix() for path in ORG.rglob("*.org")
    )


def page_title(text: str) -> str:
    """The ``#+TITLE`` of the Org page ``text``, its lines joined."""
    body = _TEXT_BLOCK.sub("", text)
    lines = re.findall(r"^[ \t]*#\+title:(.*)$", body, re.MULTILINE | re.IGNORECASE)
    return " ".join(line.strip() for line in lines).strip()


def rst_titles(block: str) -> list[tuple[Style, str]]:
    """The section titles of the reST ``block``, each with its style.

    A style is the adornment's mark and whether it overlines the title too.
    """
    lines = [line.rstrip() for line in block.splitlines()]
    found, at = [], 0
    while at < len(lines):
        line = lines[at]
        text = lines[at + 1] if at + 1 < len(lines) else ""
        under = lines[at + 2] if at + 2 < len(lines) else ""
        if _ADORNMENT.fullmatch(line) and text.strip():
            if under == line:
                found.append(((line[0], True), text.strip()))
            at += 3
        elif (
            line[:1].strip()
            and not _ADORNMENT.fullmatch(line)
            and _ADORNMENT.fullmatch(text)
            and (len(text) >= len(line) or len(text) >= 4)
        ):
            found.append(((text[0], False), line))
            at += 2
        else:
            at += 1
    return found


def titles(text: str) -> list[tuple[Style, str]]:
    """The section titles ox-rst writes for the Org page ``text``, in order."""
    body = _TEXT_BLOCK.sub("", text)
    options = re.findall(r"^[ \t]*#\+options:(.*)$", body, re.MULTILINE | re.IGNORECASE)
    shown = [
        option.removeprefix("title:")
        for option in " ".join(options).split()
        if option.startswith("title:")
    ]
    name = page_title(body)
    found: list[tuple[Style | int, str]] = []
    if name and shown[-1:] != ["nil"]:
        found.append((("=", True), name))
    export, block, hidden = None, [], None
    for line in body.splitlines():
        if export is not None:
            if not _END_EXPORT.fullmatch(line):
                block.append(line)
                continue
            if export in _RST and hidden is None:
                found.extend(rst_titles("\n".join(block)))
            export, block = None, []
            continue
        begin = _BEGIN_EXPORT.fullmatch(line)
        if begin:
            export = begin.group(1).lower()
            continue
        headline = _HEADLINE.fullmatch(line)
        if not headline:
            continue
        stars, words, tags = headline.groups()
        level = len(stars)
        if hidden is not None and level > hidden:
            continue
        hidden = None
        if re.match(r"COMMENT\b", words) or ":noexport:" in (tags or ""):
            hidden = level
            continue
        found.append((level, words))
    top = min((level for level, _ in found if isinstance(level, int)), default=1)
    styled = []
    for style, words in found:
        if isinstance(style, int):
            mark = _UNDERLINES[style - top : style - top + 1]
            if not mark.strip():
                continue
            style = (mark, False)
        styled.append((style, words))
    return styled


def outermost(text: str) -> list[str]:
    """The titles Sphinx reads at the outermost level of the Org page ``text``."""
    found = titles(text)
    return [words for style, words in found if style == found[0][0]]


def _words(text: str) -> str:
    return " ".join(_MARKS.sub("", text).casefold().split())


def _source(page: str) -> str:
    return (ORG / f"{page}.org").read_text(encoding="utf-8")


@pytest.mark.parametrize("page", pages())
def test_every_page_gives_sphinx_one_title_of_its_own(page):
    text = _source(page)
    found = outermost(text)
    assert len(found) == 1, (
        f"docs/orgmode/{page}.org gives Sphinx {len(found)} outermost titles, "
        f"{found}, each an entry of its own wherever a toctree lists the page; "
        "give it one title with its sections under it, as its #+TITLE is when "
        "its #+OPTIONS leave out title:nil"
    )
    name = page_title(text)
    assert name and _words(name) in _words(found[0]), (
        f"docs/orgmode/{page}.org gives Sphinx the title {found[0]!r}, which "
        f"does not name the page as its #+TITLE {name!r} does"
    )


def test_the_pages_are_read():
    assert {"index", "quickstart"} <= set(pages())


def test_titles_are_read_as_ox_rst_writes_them():
    page = textwrap.dedent(
        """\
        #+TITLE: The =page=
        #+OPTIONS: toc:nil
        Lead.
        * One
        ** Under one
        * COMMENT Draft
        ** Under the draft
        * Two :tag:
        ** Hidden :noexport:
        *** Under the hidden
        ** Shown
        #+begin_src rst
        =====
        Shown
        =====
        #+end_src
        #+begin_export html
        Shown
        -----
        #+end_export
        """
    )
    assert page_title(page) == "The =page="
    assert titles(page) == [
        (("=", True), "The =page="),
        (("-", False), "One"),
        (("~", False), "Under one"),
        (("-", False), "Two"),
        (("~", False), "Shown"),
    ]
    assert outermost(page) == ["The =page="]
    hidden = page.replace("toc:nil", "toc:nil title:nil")
    assert outermost(hidden) == ["One", "Two"]

    def lead(rst: str, text: str = hidden) -> list[str]:
        block = f"#+begin_export rst\n{rst}\n#+end_export"
        return outermost(text.replace("Lead.", block))

    assert lead("========\nThe page\n========") == ["The page"]
    assert lead("The page\n--------") == ["The page", "One", "Two"]
    assert lead("=====\nExtra\n=====", page) == ["The =page=", "Extra"]
    assert lead("Extra\n=====", page) == ["The =page="]
    assert outermost("#+OPTIONS: title:nil\n** A\n*** B\n** C\n") == ["A", "C"]


def test_rst_titles_are_read_as_docutils_reads_them():
    block = textwrap.dedent(
        """\
        ==========
          Over
        ==========

        Under
        ~~~~~

        ----

        A longer title
        ^^^^

        ab
        -

         Indented
        ++++++++

        -----
        Mismatch
        =====
        """
    )
    assert rst_titles(block) == [
        (("=", True), "Over"),
        (("~", False), "Under"),
        (("^", False), "A longer title"),
    ]
