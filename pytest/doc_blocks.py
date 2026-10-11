"""Run the Python blocks of a documentation page in order, in one namespace.

A page is the README, an Org page under ``docs/orgmode`` or a MyST notebook
under ``examples/notebooks``. Its blocks run top to bottom against the
installed module, as a reader pasting them in order would, in a working
directory of their own. A block that imports a module this environment
cannot provide is skipped with a reason naming it, and so is a later block
that needs a name only a skipped block defines. A block that needs a name an
earlier block failed to define fails, naming that block.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import os
import re
import symtable
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

_ORG = re.compile(
    r"^[ \t]*#\+begin_src[ \t]+python\b[^\n]*\n(.*?)^[ \t]*#\+end_src", re.M | re.S | re.I
)
_FENCE = re.compile(r"^```(?:python|\{code-cell\}[^\n]*)[ \t]*\n(.*?)^```[ \t]*$", re.M | re.S)
_CELL_OPTION = re.compile(r"^:[\w-]+:.*$")


def _campaign_dir(workdir: Path) -> dict:
    campaign = workdir / "campaign"
    campaign.mkdir()
    return {"campaign_dir": str(campaign)}


def _supplied_loss(workdir: Path) -> dict:
    return {
        "loss": lambda x: float(np.sum((x - 0.371) ** 2)),
        "low": np.full(5, -2.0),
        "high": np.full(5, 2.0),
        "x0": np.full(5, 1.0),
    }


# Names a page's prose leaves to the reader ("an existing campaign_dir", "a
# supplied scalar loss"), and the stand-ins its blocks run on.
GIVEN = {
    "docs/orgmode/howto/chemfit.org": _campaign_dir,
    "docs/orgmode/explanation/communication.org": _supplied_loss,
}


@dataclass(frozen=True)
class Block:
    page: str
    line: int
    code: str

    def compiled(self):
        # Leading newlines keep tracebacks on the page's own line numbers.
        return compile("\n" * self.line + self.code, str(ROOT / self.page), "exec")

    def imports(self) -> set[str]:
        """The top-level modules the block imports."""
        names = set()
        for node in ast.walk(ast.parse(self.code)):
            if isinstance(node, ast.Import):
                names.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.partition(".")[0])
        return names

    def binds(self) -> set[str]:
        """The names the block binds at module level."""
        table = symtable.symtable(self.code, self.page, "exec")
        return {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}


@functools.cache
def blocks(page: str) -> tuple[Block, ...]:
    """The Python blocks of ``page``, a path relative to the repository root."""
    text = (ROOT / page).read_text(encoding="utf-8")
    found = []
    for match in (_ORG if page.endswith(".org") else _FENCE).finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        rows = textwrap.dedent(match.group(1)).splitlines()
        # MyST cell options open a cell; blanking them keeps the numbering.
        for i, row in enumerate(rows):
            if not _CELL_OPTION.match(row):
                break
            rows[i] = ""
        found.append(Block(page, line, "\n".join(rows) + "\n"))
    return tuple(found)


def pages() -> list[str]:
    """Every page with a Python block."""
    candidates = [
        ROOT / "README.md",
        *sorted((ROOT / "docs" / "orgmode").rglob("*.org")),
        *sorted((ROOT / "examples" / "notebooks").glob("*.md")),
    ]
    found = [path.relative_to(ROOT).as_posix() for path in candidates]
    return [page for page in found if blocks(page)]


def needs_chemfit(page: str) -> bool:
    return any("chemfit" in block.imports() for block in blocks(page))


@functools.cache
def unavailable(module: str) -> str | None:
    """Why ``module`` cannot serve a block here, or None when it can."""
    if importlib.util.find_spec(module) is None:
        return f"needs {module}, which is not installed"
    if module == "cupy":
        try:
            import cupy

            devices = cupy.cuda.runtime.getDeviceCount()
        except Exception:  # noqa: BLE001 - no driver is no device
            devices = 0
        if devices == 0:
            return "needs cupy with a CUDA device, and there is none"
    if module == "chemfit":
        try:
            from chemfit.fitter import Fitter
        except ImportError:
            Fitter = None
        pairs = (("ask", "tell"), ("evaluate", "step"))
        if not any(all(callable(getattr(Fitter, n, None)) for n in pair) for pair in pairs):
            return (
                "needs a ChemFit Fitter with ask and tell, first released in 3.1.1, "
                "or evaluate and step; the installed one has neither pair"
            )
    return None


class Page:
    """One page's blocks, run in order as its tests ask for them."""

    def __init__(self, page: str, workdir: Path):
        self.page = page
        self.workdir = workdir
        self.namespace = {"__name__": "__main__"}
        given = GIVEN.get(page)
        if given is not None:
            self.namespace.update(given(workdir))
        self.outcomes = []
        # Names a block that did not finish would have bound: name -> (skip, why).
        self.lost = {}

    def outcome(self, index: int):
        """``None`` when block ``index`` ran, else ``(skip, why)`` or the exception."""
        page_blocks = blocks(self.page)
        while len(self.outcomes) <= index:
            self.outcomes.append(self._run(page_blocks[len(self.outcomes)]))
        return self.outcomes[index]

    def _lose(self, block: Block, skip: bool, why: str) -> None:
        for name in block.binds() - self.namespace.keys():
            self.lost.setdefault(name, (skip, why))

    def _run(self, block: Block):
        try:
            code = block.compiled()
        except SyntaxError as err:
            return err
        for module in sorted(block.imports()):
            why = unavailable(module)
            if why is not None:
                self._lose(block, True, f"{why} (the block at line {block.line})")
                return True, why
        cwd = os.getcwd()
        os.chdir(self.workdir)
        try:
            exec(code, self.namespace)  # noqa: S102
        except Exception as err:  # noqa: BLE001 - reported by the block's test
            if isinstance(err, NameError) and err.name in self.lost:
                skip, why = self.lost[err.name]
                self._lose(block, skip, why)
                return skip, f"{err.name} is undefined: {why}"
            self._lose(block, False, f"the block at line {block.line} failed")
            return err
        finally:
            os.chdir(cwd)
        return None


_RUNS: dict[str, Page] = {}


def check(page: str, index: int, tmp_path_factory) -> None:
    """Run ``page`` through block ``index`` and report that block's outcome."""
    if page not in _RUNS:
        _RUNS[page] = Page(page, tmp_path_factory.mktemp(Path(page).stem))
    outcome = _RUNS[page].outcome(index)
    if isinstance(outcome, BaseException):
        raise outcome
    if outcome is not None:
        skip, why = outcome
        if skip:
            pytest.skip(why)
        pytest.fail(why, pytrace=False)


def cases(chemfit: bool) -> list:
    """One ``(page, index)`` parameter per block, of the ChemFit pages or the rest."""
    return [
        pytest.param(page, index, id=f"{page}:{block.line}")
        for page in pages()
        if needs_chemfit(page) == chemfit
        for index, block in enumerate(blocks(page))
    ]
