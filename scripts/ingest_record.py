#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fetch a record branch read-only and regenerate the tables it feeds.

    python3 scripts/ingest_record.py BRANCH SHA [--remote origin]

The branch is fetched into a temporary ref and that ref is deleted. The
script does not merge, does not check out a branch, and does not push.
Rows are written only when every revision field of every row is a
commit and every harness commit in them is an ancestor of HEAD, an
external pin, or a repository URL at a full SHA. A blank, "unknown" or
non-hex revision field refuses the record. A tuned-GMIN row is accepted
when the paper audit accepts it, so its side commit is checked against
the data and tuned.py of HEAD. The fetch lands in a scratch repository
when it would otherwise land in the repository whose HEAD the check
uses. The row files are staged beside their destinations and moved into
place with os.replace, and the generators then rewrite the marked tables.
When a generator fails, every file the ingest may have touched is put
back as it was.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_reach  # noqa: E402
from atomic_write import stage, write_atomic  # noqa: E402
HARNESS_FIELDS = (
    "harness_revision", "harness_commit", "code_revision", "head", "anneal_sha", "anneal_head",
    "anneal_revision", "gmin_revision",
)
EXTERNAL_REVISIONS = tuple(revision_reach.EXTERNAL_REVISIONS)
WITHHELD_MARKS = ("% results not quoted: start", "% results not quoted: end")
REASONS = ("harness not reachable", "rerun pending")
MORSE = ("m55r6", "m55r14", "m80r6", "m80r14")
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


external_pin = revision_reach.external_pin
external_form = revision_reach.external_form


def harness_reachable(rev: str, repo: Path = REPO) -> bool:
    """A full SHA that is an ancestor of HEAD, an external pin, or a repository URL at a full SHA."""
    return revision_reach.revision_reachable(rev, repo)


REVISION_FIELDS = (*HARNESS_FIELDS, "release_revision")
_SEPARATORS = re.compile(r"[;\s]+")


def revision_tokens(value: str | None) -> list[str]:
    """The revisions one field names. A field may list several, apart by a semicolon or a space."""
    return [token for token in _SEPARATORS.split((value or "").strip()) if token]


def malformed_revision(value: str | None) -> str | None:
    """Why a revision field cannot name a commit, or None when it can.

    A field is well formed when it holds one or more full lowercase SHAs of
    forty characters, or a repository URL at a full SHA. Blank, "unknown", an
    abbreviated SHA and anything else is refused, as the paper audit counts a
    blank or "unknown" revision as unrecorded.
    """
    tokens = revision_tokens(value)
    if not tokens:
        return "blank"
    for token in tokens:
        if token.lower() == "unknown":
            return "unknown"
        if revision_reach.is_abbreviated_sha(token):
            return f"abbreviated SHA ({token!r}); a full 40-character SHA is required"
        if not (revision_reach.is_full_sha(token) or external_form(token)):
            return f"not a commit ({token!r})"
    return None


def tuned_row(row: dict[str, str | None]) -> bool:
    """A tuned-GMIN row names its harness by a commit and the data it ran on."""
    return bool((row.get("harness_commit") or "").strip() and (row.get("data_blob") or "").strip())


def tuned_verified(row: dict[str, str | None], repo: Path) -> bool:
    """The check the paper audit makes of a tuned-GMIN row, one implementation, so a record lands exactly when the audit accepts it."""
    return revision_reach.tuned_gmin_harness_verified({k: v or "" for k, v in row.items() if k}, repo)


def tuned_pairs(rows: list[dict[str, str | None]], repo: Path) -> set[tuple[str, str]]:
    """The (harness commit, data blob) pairs of the rows the audit verifies, for those that pass."""
    return {
        ((row.get("harness_commit") or "").strip(), (row.get("data_blob") or "").strip())
        for row in rows
        if tuned_row(row)
        and (row.get("system") or "").strip() in revision_reach.TUNED_SYSTEMS
        and (row.get("code") or "").strip() in revision_reach.TUNED_CODES
        and tuned_verified(row, repo)
    }


def tuned_accepted(row: dict[str, str | None], pairs: set[tuple[str, str]], repo: Path) -> bool:
    """A tuned row passes when the audit verifies it, or, for a control row the audit does not read, when
    a row the audit does verify names the same harness commit and data blob."""
    pair = ((row.get("harness_commit") or "").strip(), (row.get("data_blob") or "").strip())
    if (row.get("system") or "").strip() in revision_reach.TUNED_SYSTEMS and (row.get("code") or "").strip() in revision_reach.TUNED_CODES:
        return tuned_verified(row, repo)
    return pair in pairs and harness_reachable((row.get("code_revision") or "").strip(), repo)


def parse_rows(path: str, blob: bytes) -> tuple[list[str], list[dict[str, str | None]]]:
    try:
        text = blob.decode()
    except UnicodeError as exc:
        raise SystemExit(f"{path}: a record row file is not text") from exc
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    return list(reader.fieldnames or []), rows


def require_reachable(blobs: dict[str, bytes], repo: Path = REPO) -> None:
    """Refuse the record unless every revision field of every row is well formed and reachable. Write nothing.

    Each revision column of each row is judged on its own: a blank, "unknown"
    or non-hex field is refused whatever else the row holds. A tuned-GMIN row
    names its harness by a side commit; it is accepted when the paper audit
    accepts it, which checks the commit against the data and tuned.py of HEAD.
    """
    saw = False
    for path, blob in blobs.items():
        header, rows = parse_rows(path, blob)
        columns = [field for field in REVISION_FIELDS if field in header]
        if not any(field in header for field in HARNESS_FIELDS) or not rows:
            raise SystemExit(f"{path}: no harness commit")
        pairs = tuned_pairs(rows, repo) if any(tuned_row(row) for row in rows) else set()
        for number, row in enumerate(rows, start=1):
            if None in row:
                raise SystemExit(f"{path}: row {number} has more fields than the header")
            tuned = tuned_row(row)
            for field in columns:
                value = row.get(field)
                why = malformed_revision(value)
                if why:
                    raise SystemExit(f"{path}: row {number} {field} is {why}")
                if field == "release_revision" or (tuned and field in {"harness_commit", "code_revision"}):
                    continue
                for token in revision_tokens(value):
                    saw = True
                    if not harness_reachable(token, repo):
                        raise SystemExit(f"{path}: harness {token} is not reachable")
            if tuned:
                saw = True
                if not tuned_accepted(row, pairs, repo):
                    raise SystemExit(
                        f"{path}: row {number} harness {row.get('harness_commit')} is not reachable "
                        "and is not a verified tuned-GMIN side commit"
                    )
    if not saw:
        raise SystemExit("no harness commit")


def _same_repo(root: Path) -> None:
    if root.resolve() != REPO.resolve():
        raise SystemExit("the generators rewrite this repository")


def refresh_ase(root: Path) -> None:
    _same_repo(root)
    ase = _module("ase_panel_refresh", "scripts/ase_panel/summarize.py")
    document = root / "paper/jctc/si.tex"
    write_atomic(document, ase.apply_document(document.read_text()))


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
    """Comparisons that are not quoted stay out of the paper."""
    if items is None:
        items = withheld_items()
    for _name, reason in items:
        if reason not in REASONS:
            raise SystemExit(f"unknown withheld reason {reason}")
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
    write_atomic(path, splice_marked(text, WITHHELD_MARKS, withheld_block().rstrip("\n")))


def refresh_labels(root: Path) -> None:
    script = root / "paper/jctc/scripts/cross_labels.py"
    if script.is_file():
        subprocess.run([sys.executable, str(script)], cwd=root, check=True)


GENERATED_PATHSPECS = ("paper/jctc/*.tex", "paper/jctc/*.md", "bench/gmin-panel/*")


def touched_files(root: Path, record: Record) -> list[Path]:
    """Files an ingest may rewrite: the record's rows and, for this repository, the tables built from them."""
    names = list(record.paths)
    if root.resolve() == REPO.resolve():
        listed = _git(root, "ls-files", "-z", "--", *GENERATED_PATHSPECS)
        if listed.returncode == 0:
            names.extend(name for name in listed.stdout.decode().split("\0") if name)
    return [root / name for name in dict.fromkeys(names)]


def snapshot(paths: list[Path]) -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.is_file() else None for path in paths}


def restore(saved: dict[Path, bytes | None]) -> None:
    """Put back every file as it was: its old bytes, or no file where there was none."""
    for path, data in saved.items():
        if data is None:
            path.unlink(missing_ok=True)
        elif not path.is_file() or path.read_bytes() != data:
            write_atomic(path, data)


def install_rows(root: Path, blobs: dict[str, bytes]) -> None:
    """Stage every row file beside its destination, then move them into place."""
    staged: list[tuple[Path, Path]] = []
    try:
        for name, data in blobs.items():
            dest = root / name
            staged.append((stage(dest, data), dest))
        while staged:
            temporary, dest = staged[0]
            os.replace(temporary, dest)
            staged.pop(0)
    finally:
        for temporary, _dest in staged:
            temporary.unlink(missing_ok=True)


def _check_name(branch: str, sha: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", branch) or branch.startswith(("/", ".")) or ".." in branch:
        raise SystemExit(f"refusing branch name {branch}")
    if not revision_reach.is_full_sha(sha):
        raise SystemExit("the record SHA must be 40 hex characters")


def _resolve_remote(repo: Path, remote: str) -> str:
    """Resolve a named git remote in ``repo`` to its fetch URL or path."""
    shown = _git(repo, "remote", "get-url", remote)
    if shown.returncode != 0:
        return remote
    url = shown.stdout.decode().strip()
    if not url:
        return remote
    if "://" not in url and ":" not in url.split("/", 1)[0]:
        candidate = Path(url)
        if not candidate.is_absolute():
            return str((repo / candidate).resolve())
    return url


def ingest(
    branch: str,
    sha: str,
    remote: str = "origin",
    root: Path | None = None,
    git_dir: Path | None = None,
    catalog: dict[str, Record] | None = None,
    reachable_repo: Path | None = None,
) -> None:
    """Fetch ``branch`` at ``sha``, install its rows, and regenerate its tables.

    ``reachable_repo`` is the repository whose commits count as held. When
    the fetch would land in that repository, it lands in a scratch
    repository instead, and the rows land under ``root``.
    """
    _check_name(branch, sha)
    root = REPO if root is None else root
    reachable_repo = REPO if reachable_repo is None else reachable_repo
    record = lookup(branch, catalog if catalog is not None else production_catalog())
    requested = git_dir or root
    fetch_remote = remote
    scratch: tempfile.TemporaryDirectory[str] | None = None
    if requested.resolve() == reachable_repo.resolve():
        fetch_remote = _resolve_remote(requested, remote)
        scratch = tempfile.TemporaryDirectory(prefix="ingest-")
        fetch_into = Path(scratch.name)
        _git(fetch_into, "init", "-q")
    else:
        fetch_into = requested
    ref = f"refs/ingest/{branch}"
    try:
        _git(fetch_into, "update-ref", "-d", ref)
        fetched = _git(fetch_into, "fetch", "--no-tags", fetch_remote, f"{branch}:{ref}")
        if fetched.returncode != 0:
            detail = fetched.stderr.decode(errors="replace").strip()
            raise SystemExit(f"fetch of {branch} failed: {detail}")
        got = _git(fetch_into, "rev-parse", ref).stdout.decode().strip()
        if got != sha:
            raise SystemExit(f"{branch} is {got}, not {sha}")
        blobs: dict[str, bytes] = {}
        for path in record.paths:
            shown = _git(fetch_into, "cat-file", "blob", f"{sha}:{path}")
            if shown.returncode != 0:
                raise SystemExit(f"{branch} has no {path}")
            blobs[path] = shown.stdout
        require_reachable(blobs, reachable_repo)
        saved = snapshot(touched_files(root, record))
        try:
            install_rows(root, blobs)
            record.refresh(root)
            rewrite_withheld(root)
            refresh_labels(root)
        except BaseException:
            restore(saved)
            clear_withheld_cache()
            raise
    finally:
        _git(fetch_into, "update-ref", "-d", ref)
        if scratch is not None:
            scratch.cleanup()


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
