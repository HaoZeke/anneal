# Copyright (c) anneal developers
"""One definition of a held revision: an ancestor of the audited head.

The paper audit, the ingest of a record, and the generators that decide
whether a table may be quoted all ask this question. A commit object that is
present in the repository but is not an ancestor of HEAD (an unmerged side
commit, a fetched record branch) is not held.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNRECORDED = frozenset({"", "unknown", "latest"})


def commit_held(revision: str, repo: Path = ROOT, head: str = "HEAD") -> bool:
    """The revision names a commit that is an ancestor of ``head`` in ``repo``.

    A blank value, "unknown" and "latest" name no commit. A value that git
    could read as an option is refused before git is run.
    """
    revision = (revision or "").strip()
    if revision in UNRECORDED or revision.startswith("-") or any(ch.isspace() for ch in revision):
        return False
    return subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", revision, head],
        capture_output=True, check=False,
    ).returncode == 0
