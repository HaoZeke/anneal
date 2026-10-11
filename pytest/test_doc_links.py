"""Every relative link of the README, the docs and the notebooks names a file.

The pages are README.md, every Org page under ``docs/orgmode`` and every MyST
notebook under ``examples/notebooks``. A link with a scheme (``https:``,
``mailto:``, ``doi:`` and the like) or to an anchor on its own page is left
alone. Any other target loses its ``#fragment`` and is resolved against the
directory of the page that links it, and the result must be a file or a
directory the repository tracks.

Org links are read as Org reads them. A bracket link ``[[target]]`` or
``[[target][description]]`` names a file when its target starts with
``file:`` or is a path starting with ``./``, ``../``, ``/`` or ``~/``; any
other target without a scheme is a heading, custom id, coderef or target on
the same page. An angle link ``<file:target>`` and a plain ``file:target`` in
the text name a file too. A ``::search`` option is dropped like a fragment,
and an absolute path names nothing in the repository. Source, example, export
and comment blocks, fixed-width ``:`` lines, comment lines and verbatim or
code spans hold no links.

Markdown links, in the README and in a notebook's Markdown cells, are read as
GitHub reads them: an inline link or image ``[text](target)``, a reference
definition ``[label]: target`` and the ``href`` or ``src`` of an HTML tag. A
target may sit in angle brackets, loses a ``?query`` as well, has its percent
escapes decoded and, when it starts with ``/``, is read from the repository
root. Front matter, fenced blocks (which hold a notebook's code cells), code
spans and HTML comments hold no links.

The tracked files are those ``git ls-files`` lists when the tests run from a
checkout, so a file that exists only locally, such as a notebook made from its
MyST source and ignored by git, does not count. Anywhere else, such as an
export of the tree, they are the files on disk.
"""

from __future__ import annotations

import functools
import posixpath
import re
import subprocess
import textwrap
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]

_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
_ORG_FILE = re.compile(r"file(?:\+sys|\+emacs)?:")
_ORG_QUIET = (
    re.compile(
        r"^[ \t]*#\+begin_(src|example|export|comment)\b.*?^[ \t]*#\+end_\1\b[^\n]*$",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    ),
    re.compile(r"^[ \t]*[:#](?:[ \t][^\n]*)?$", re.MULTILINE),
    re.compile(
        r"(?<![^\s\-({'\"])([=~])(?=\S)[^\n]*?(?<=\S)\1(?=[\s\-.,;:!?'\")}\[\\]|$)",
        re.MULTILINE,
    ),
)
_ORG_BRACKET = re.compile(r"\[\[((?:[^\[\]\\\n]|\\.)+)\](?:\[.*?\])?\]", re.DOTALL)
_ORG_ANGLE = re.compile(r"<(file(?:\+sys|\+emacs)?:[^<>\n]+)>")
_ORG_PLAIN = re.compile(r"(?<![\w/])file(?:\+sys|\+emacs)?:[^\s\[\]()<>]+")
_MD_QUIET = (
    re.compile(r"\A---\n.*?^---[ \t]*$", re.MULTILINE | re.DOTALL),
    re.compile(
        r"^[ \t]{0,3}(`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]{0,3}\1[`~]*[ \t]*$|\Z)",
        re.MULTILINE | re.DOTALL,
    ),
    re.compile(r"<!--.*?-->", re.DOTALL),
    re.compile(r"(?<!`)(`+)(?!`)(?:(?!\n[ \t]*\n).)+?(?<!`)\1(?!`)", re.DOTALL),
)
_MD_INLINE = re.compile(
    r"\]\(\s*(<[^<>\n]*>|(?:[^\s()\\]|\\.|\((?:[^\s()\\]|\\.)*\))+)"
)
_MD_DEFINITION = re.compile(
    r"^[ \t]{0,3}\[(?!\^)[^\]\n]+\]:[ \t]*(<[^<>\n]*>|\S+)", re.MULTILINE
)
_MD_TAG = re.compile(r"<[A-Za-z][^<>]*>")
_MD_ATTRIBUTE = re.compile(
    r"\b(?:href|src)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s\"'<>`=]+))", re.IGNORECASE
)
_MYST = re.compile(
    r"\A---\n.*?^[ \t]*format_name:[ \t]*myst\b.*?^---", re.MULTILINE | re.DOTALL
)


def _quiet(text: str, patterns) -> str:
    """``text`` with what ``patterns`` match blanked, lines kept in place."""
    for pattern in patterns:
        text = pattern.sub(lambda match: re.sub(r"[^\n]", " ", match.group(0)), text)
    return text


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def org_links(text: str) -> list[tuple[int, str, str]]:
    """The line, target and path of each link to a file in Org ``text``."""
    text = _quiet(text, _ORG_QUIET)
    found = []
    for match in _ORG_BRACKET.finditer(text):
        target = re.sub(r"\\([\[\]\\])", r"\1", match.group(1))
        if _ORG_FILE.match(target):
            path = target[_ORG_FILE.match(target).end() :]
        elif _SCHEME.match(target) or not target.startswith(("./", "../", "/", "~/")):
            continue
        else:
            path = target
        found.append((match.start(), target, path))
    text = _quiet(text, (_ORG_BRACKET,))
    found += [(m.start(), m.group(1), m.group(1)) for m in _ORG_ANGLE.finditer(text)]
    text = _quiet(text, (_ORG_ANGLE,))
    for match in _ORG_PLAIN.finditer(text):
        target = match.group(0).rstrip(".,;:!?'\"")
        found.append((match.start(), target, target))
    links = []
    for offset, target, path in sorted(found):
        if _ORG_FILE.match(path):
            path = path[_ORG_FILE.match(path).end() :]
        path = path.split("::", 1)[0].split("#", 1)[0]
        if path:
            links.append((_line(text, offset), target, path))
    return links


def markdown_links(text: str) -> list[tuple[int, str, str]]:
    """The line, target and path of each relative link in Markdown ``text``."""
    text = _quiet(text, _MD_QUIET)
    found = [(m.start(1), m.group(1)) for m in _MD_INLINE.finditer(text)]
    found += [(m.start(1), m.group(1)) for m in _MD_DEFINITION.finditer(text)]
    for tag in _MD_TAG.finditer(text):
        for attribute in _MD_ATTRIBUTE.finditer(tag.group(0)):
            value = next(group for group in attribute.groups() if group is not None)
            found.append((tag.start() + attribute.start(), value))
    links = []
    for offset, target in sorted(found):
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        if _SCHEME.match(target):
            continue
        path = unquote(
            re.sub(r"\\(.)", r"\1", target.split("#", 1)[0].split("?", 1)[0])
        )
        if path:
            links.append((_line(text, offset), target, path))
    return links


def resolved(page: str, path: str) -> str | None:
    """The repository path ``path`` names from ``page``, or None if it names none."""
    if page.endswith(".org") and path.startswith(("/", "~")):
        return None
    if path.startswith("/"):
        path = posixpath.normpath(path.lstrip("/") or ".")
    else:
        path = posixpath.normpath(posixpath.join(posixpath.dirname(page), path))
    return None if path == ".." or path.startswith("../") else path


def _git_files() -> list[str] | None:
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        if Path(top.stdout.strip()).resolve() != ROOT:
            return None
        listed = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [
        name
        for name in listed.stdout.decode("utf-8", "surrogateescape").split("\0")
        if name
    ]


@functools.cache
def tracked() -> frozenset[str]:
    """Every file the repository tracks and every directory holding one."""
    files = _git_files()
    if files is None:
        files = [
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file() and ".git" not in path.relative_to(ROOT).parts
        ]
    names = {"."}
    for name in files:
        while name and name not in names:
            names.add(name)
            name = posixpath.dirname(name)
    return frozenset(names)


def pages() -> list[str]:
    """README.md, the Org pages under docs/orgmode and the MyST notebooks."""
    notebooks = [
        path
        for path in sorted((ROOT / "examples" / "notebooks").glob("*.md"))
        if _MYST.match(path.read_text(encoding="utf-8"))
    ]
    found = [
        ROOT / "README.md",
        *sorted((ROOT / "docs" / "orgmode").rglob("*.org")),
        *notebooks,
    ]
    return [path.relative_to(ROOT).as_posix() for path in found]


def links(page: str) -> list[tuple[int, str, str | None]]:
    """The line, target and resolved repository path of each file link of ``page``."""
    text = (ROOT / page).read_text(encoding="utf-8")
    read = org_links if page.endswith(".org") else markdown_links
    return [(line, target, resolved(page, path)) for line, target, path in read(text)]


@pytest.mark.parametrize("page", pages())
def test_every_relative_link_names_a_file_the_repository_tracks(page):
    names = tracked()
    broken = [
        f"{page}:{line}: {target}" + ("" if path is None else f" ({path})")
        for line, target, path in links(page)
        if path not in names
    ]
    assert not broken, "links to nothing the repository tracks:\n" + "\n".join(broken)


def test_the_readme_the_org_pages_and_the_notebooks_are_read():
    found = pages()
    assert found[0] == "README.md" and "examples/notebooks/01_quickstart.md" in found
    assert links("README.md")
    assert any(links(page) for page in found if page.endswith(".org"))


def test_org_links_are_read_as_org_reads_them():
    text = textwrap.dedent(
        """\
        See [[file:a.org][a]], [[file:b.org::*Part]], [[./c.png]] and [[../d/]].
        Also <file:e.org> and file:f.org, but not =file:g.org= or ~file:h~.
        [[https://x.org][x]] [[mailto:a@x.org]] [[#id]] [[*Part]] [[Target]]
        #+begin_src python
        x = "[[file:i.org]]"
        #+end_src
        : [[file:j.org]]
        # [[file:k.org]]
        [[file:l.org#part][l]] [[/m.org]]
        """
    )
    assert [(line, path) for line, _, path in org_links(text)] == [
        (1, "a.org"),
        (1, "b.org"),
        (1, "./c.png"),
        (1, "../d/"),
        (2, "e.org"),
        (2, "f.org"),
        (9, "l.org"),
        (9, "/m.org"),
    ]
    assert resolved("docs/orgmode/howto/p.org", "../q.org") == "docs/orgmode/q.org"
    assert resolved("docs/orgmode/p.org", "/m.org") is None
    assert resolved("docs/orgmode/p.org", "~/m.org") is None
    assert resolved("docs/orgmode/p.org", "../../../r.org") is None


def test_markdown_links_are_read_as_github_reads_them():
    text = textwrap.dedent(
        """\
        ---
        title: "[x](front.md)"
        ---
        A [link](a.md), ![image](img/b.png "b"), [s](<c d.md>) and [e](e%20f.md#part).
        [ref]: g.md
        <img src="./h.png"> <a href='i/'>i</a>
        [w](https://x.org) [m](mailto:a@x.org) [here](#anchor) [root](/j.md?raw=true)
        `[code](k.md)`
        ```python
        print("[fence](l.md)")
        ```
        <!-- [hidden](m.md) -->
        """
    )
    assert [(line, path) for line, _, path in markdown_links(text)] == [
        (4, "a.md"),
        (4, "img/b.png"),
        (4, "c d.md"),
        (4, "e f.md"),
        (5, "g.md"),
        (6, "./h.png"),
        (6, "i/"),
        (7, "/j.md"),
    ]
    assert resolved("examples/notebooks/n.md", "../q.py") == "examples/q.py"
    assert resolved("examples/notebooks/n.md", "/j.md") == "j.md"
