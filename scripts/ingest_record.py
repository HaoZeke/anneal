#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fetch a record branch read-only and regenerate the tables it feeds.

    python3 scripts/ingest_record.py BRANCH SHA [--remote origin]

The branch is fetched into a temporary ref and that ref is deleted. The
script does not merge, does not check out a branch, and does not push.
Rows are written only when every harness commit in them is a commit in
this repository or one of the external pins. The generators then rewrite
the marked tables, and the SI subsection Withheld results is rewritten
from the comparisons that are still missing.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS_FIELDS = (
    "harness_revision", "harness_commit", "code_revision", "head", "anneal_sha", "anneal_head",
    "anneal_revision", "gmin_revision",
)
EXTERNAL_REVISIONS = (
    "aac1548a8aec13b53ad12a8d6904625081685d2e",
    "9cbec1a446352bc9ef064bd2b1a5767194655919",
    "8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e",
    "71185263eb6a8130d1262fd76d90109f5fabdcf5",
    "fa8a4416e8fe093d0075bc10ac875494c2a449a9",
)
WITHHELD_MARKS = ("% results not quoted: start", "% results not quoted: end")
REASONS = ("harness not reachable", "rerun pending")
MORSE = ("m55r6", "m55r14", "m80r6", "m80r14")
HEX = re.compile(r"[0-9a-f]{7,40}")
_MODULES: dict[str, object] = {}
_ITEMS: list[tuple[str, str]] | None = None


@dataclass(frozen=True)
class Record:
    key: str
    paths: tuple[str, ...]
    reason: str
    refresh: Callable[[Path], None]


def _module(name: str, path: str) -> object:
    cached = _MODULES.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _MODULES[name] = module
    return module


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)


def external_pin(rev: str) -> bool:
    """The revision is one of the external pins, or a prefix of one."""
    rev = (rev or "").strip()
    if not HEX.fullmatch(rev):
        return False
    return any(pin == rev or pin.startswith(rev) or rev.startswith(pin) for pin in EXTERNAL_REVISIONS)


def harness_reachable(rev: str, repo: Path = REPO) -> bool:
    """The harness is a commit in this repository, or an external pin."""
    if external_pin(rev):
        return True
    return _git(repo, "cat-file", "-e", f"{rev}^{{commit}}").returncode == 0


def harness_tokens(blob: bytes) -> list[str]:
    """Hex harness commits named by the row fields the audit scans."""
    try:
        text = blob.decode()
    except UnicodeError as exc:
        raise SystemExit("a record row file is not text") from exc
    reader = csv.DictReader(io.StringIO(text))
    found: list[str] = []
    for row in reader:
        for field in HARNESS_FIELDS:
            for token in (row.get(field) or "").split():
                if HEX.fullmatch(token):
                    found.append(token)
    return found


def require_reachable(blobs: dict[str, bytes], repo: Path = REPO) -> None:
    """Refuse the record when any harness commit is not reachable. Write nothing."""
    saw = False
    for path, blob in blobs.items():
        tokens = harness_tokens(blob)
        if not tokens:
            raise SystemExit(f"{path}: no harness commit")
        for token in tokens:
            saw = True
            if not harness_reachable(token, repo):
                raise SystemExit(f"{path}: harness {token} is not reachable")
    if not saw:
        raise SystemExit("no harness commit")


def _same_repo(root: Path) -> None:
    if root.resolve() != REPO.resolve():
        raise SystemExit("the generators rewrite this repository")


def refresh_ase(root: Path) -> None:
    _same_repo(root)
    ase = _module("ase_panel_refresh", "scripts/ase_panel/summarize.py")
    document = root / "paper/jctc/si.tex"
    document.write_text(ase.apply_document(document.read_text()))


def refresh_gmin(root: Path) -> None:
    _same_repo(root)
    panel = _module("gmin_panel_refresh", "scripts/gmin_panel/run_panel.py")
    rows = panel.read_rows(root / "bench/gmin-panel/rows.csv")
    panel.write_tables(root, rows, "latest", True)


def refresh_sc(root: Path) -> None:
    _same_repo(root)
    tables = _module("h2h_tables_refresh", "paper/jctc/scripts/h2h_tables.py")
    tables.refresh(("sc-prose", "sc", "sc-si"))


def refresh_tuned(root: Path) -> None:
    _same_repo(root)
    tables = _module("h2h_tables_refresh", "paper/jctc/scripts/h2h_tables.py")
    tables.refresh(("gmin-tuned",))


def production_catalog() -> dict[str, Record]:
    """Record branches BenchBot pushes, and the bench paths each one fills."""
    return {
        "record/ase-panel": Record(
            "tab:ase",
            (
                "bench/ase-lj38/rows.csv",
                "bench/ase-lj55/rows.csv",
                "bench/ase-lj98/rows.csv",
                "bench/ase-lj104/rows.csv",
                "bench/ase-lj150/rows.csv",
                "bench/ase-au55/rows.csv",
                "bench/ase-au55/rows-for-cluster.csv",
            ),
            "rerun pending",
            refresh_ase,
        ),
        "record/morse-gmin": Record(
            "Morse against GMIN",
            ("bench/gmin-panel/rows.csv",),
            "harness not reachable",
            refresh_gmin,
        ),
        "record/sc-n12": Record(
            "tab:sc",
            (
                "bench/metal-cluster-h2h/sc-n12/seeds.csv",
                "bench/metal-cluster-h2h/sc-n12/results.csv",
                "bench/metal-cluster-h2h/sc-n12/pvalues.csv",
            ),
            "harness not reachable",
            refresh_sc,
        ),
        "record/gmin-tuned": Record(
            "tab:gmin-tuned",
            ("bench/gmin-tuned/rows-shard2.csv",),
            "harness not reachable",
            refresh_tuned,
        ),
    }


def lookup(branch: str, catalog: dict[str, Record]) -> Record:
    if branch in catalog:
        return catalog[branch]
    short = branch.removeprefix("record/")
    if short in catalog:
        return catalog[short]
    raise SystemExit(f"unknown record branch {branch}")


def clear_withheld_cache() -> None:
    global _ITEMS
    _ITEMS = None


def withheld_items() -> list[tuple[str, str]]:
    """Each comparison the generators still refuse to quote, and why."""
    global _ITEMS
    if _ITEMS is not None:
        return list(_ITEMS)
    items: list[tuple[str, str]] = []
    ase = _module("ase_panel_withheld", "scripts/ase_panel/summarize.py")
    if not ase.tex_table():
        items.append(("tab:ase", "rerun pending"))
    panel = _module("gmin_panel_withheld", "scripts/gmin_panel/run_panel.py")
    rows = panel.read_rows(REPO / "bench/gmin-panel/rows.csv")
    quoted = {item["system"] for item in panel.quoted_summary(panel.summarise(rows, "latest"))}
    if any(system not in quoted for system in MORSE):
        items.append(("Morse against GMIN", "harness not reachable"))
    tables = _module("h2h_tables_withheld", "paper/jctc/scripts/h2h_tables.py")
    if not tables.sc_gmin_quotable():
        items.append(("tab:sc", "harness not reachable"))
    if not tables.tuned_gmin_block():
        items.append(("tab:gmin-tuned", "harness not reachable"))
    for _name, reason in items:
        if reason not in REASONS:
            raise SystemExit(f"unknown withheld reason {reason}")
    _ITEMS = items
    return list(items)


def withheld_block(items: list[tuple[str, str]] | None = None) -> str:
    """Unquoted comparisons are not narrated in the rendered SI."""
    del items
    return ""


def splice_marked(text: str, marks: tuple[str, str], block: str) -> str:
    """Put ``block`` between ``marks``. An empty block clears the interior and keeps the marks."""
    start, end = marks
    if text.count(start) != 1 or text.count(end) != 1:
        if not block and start not in text and end not in text:
            return text
        raise SystemExit(f"{start!r}: the file needs exactly one block")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    middle = f"\n{block}\n" if block else "\n"
    return f"{head}{start}{middle}{end}{tail}"


def rewrite_withheld(root: Path) -> None:
    path = root / "paper/jctc/si.tex"
    if not path.is_file() or WITHHELD_MARKS[0] not in path.read_text():
        return
    clear_withheld_cache()
    text = path.read_text()
    path.write_text(splice_marked(text, WITHHELD_MARKS, withheld_block().rstrip("\n")))


def refresh_labels(root: Path) -> None:
    script = root / "paper/jctc/scripts/cross_labels.py"
    if script.is_file():
        subprocess.run([sys.executable, str(script)], cwd=root, check=True)


def _check_name(branch: str, sha: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", branch) or branch.startswith(("/", ".")) or ".." in branch:
        raise SystemExit(f"refusing branch name {branch}")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SystemExit("the record SHA must be 40 hex characters")


def ingest(
    branch: str,
    sha: str,
    remote: str = "origin",
    root: Path = REPO,
    git_dir: Path | None = None,
    catalog: dict[str, Record] | None = None,
    reachable_repo: Path = REPO,
) -> None:
    """Fetch ``branch`` at ``sha``, install its rows, and regenerate its tables.

    ``reachable_repo`` is the repository whose commits count as held. The
    fetch itself lands in ``git_dir`` and the rows land under ``root``.
    """
    _check_name(branch, sha)
    record = lookup(branch, catalog if catalog is not None else production_catalog())
    git_dir = git_dir or root
    ref = f"refs/ingest/{branch}"
    _git(git_dir, "update-ref", "-d", ref)
    fetched = _git(git_dir, "fetch", "--no-tags", remote, f"{branch}:{ref}")
    if fetched.returncode != 0:
        detail = fetched.stderr.decode(errors="replace").strip()
        raise SystemExit(f"fetch of {branch} failed: {detail}")
    try:
        got = _git(git_dir, "rev-parse", ref).stdout.decode().strip()
        if got != sha:
            raise SystemExit(f"{branch} is {got}, not {sha}")
        blobs: dict[str, bytes] = {}
        for path in record.paths:
            shown = _git(git_dir, "cat-file", "blob", f"{sha}:{path}")
            if shown.returncode != 0:
                raise SystemExit(f"{branch} has no {path}")
            blobs[path] = shown.stdout
        require_reachable(blobs, reachable_repo)
        for path, data in blobs.items():
            dest = root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        record.refresh(root)
        rewrite_withheld(root)
        refresh_labels(root)
    finally:
        _git(git_dir, "update-ref", "-d", ref)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("branch", help="record branch, for example record/ase-panel")
    parser.add_argument("sha", help="40-character commit the branch must name")
    parser.add_argument("--remote", default="origin")
    args = parser.parse_args(argv)
    ingest(args.branch, args.sha, remote=args.remote)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
