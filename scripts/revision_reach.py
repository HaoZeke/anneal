# Copyright (c) anneal developers
"""One definition of a held revision, and of the rows that cite one.

The paper audit, the ingest of a record, the generators that decide whether
a table may be quoted and the tuned-GMIN check all ask the same questions
here, so the gate and the refresh cannot disagree.

A revision a row cites is a full 40-character SHA, a repository URL at a
full SHA, or one of the pins of other projects. It counts as held when it is
an ancestor of HEAD, a pin, or a URL form. A commit object that is present in
the repository but is not an ancestor of HEAD (an unmerged side commit, a
fetched record branch) is not held. An abbreviated SHA is refused wherever a
row cites a revision.
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from gmin_pin import REVISION as GMIN_PIN_REVISION  # noqa: E402

UNRECORDED = frozenset({"", "unknown", "latest"})
FULL_SHA = re.compile(r"[0-9a-f]{40}")
ABBREVIATED_SHA = re.compile(r"[0-9a-f]{7,39}")
FULL_PIN_DIGEST = re.compile(r"[0-9a-f]{64}")
EXTERNAL_FORM = re.compile(r"^https://[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+\.git@[0-9a-f]{40}$")
# Revisions of other codes. The GMIN build scripts refuse a tarball whose SVNREV differs.
EXTERNAL_REVISIONS = {
    "aac1548a8aec13b53ad12a8d6904625081685d2e": "GMIN source/SVNREV",
    GMIN_PIN_REVISION: "GMIN source/SVNREV (2026-11-19)",
    "8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e": "rgpot pin",
    "71185263eb6a8130d1262fd76d90109f5fabdcf5": "Hugging Face dataset revision",
    "fa8a4416e8fe093d0075bc10ac875494c2a449a9": "tblite commit",
    "0d9917600ecda72b2176253d0e838bcb58ec6e38902ef7d0affc7e7e7eba8f39": "GMIN 2026-11-21T03:01:22Z tarball sha256",
}
TUNED_SYSTEMS = frozenset({"lj98", "lj104"})
TUNED_CODES = frozenset({"anneal", "gmin"})
# scripts/gmin_panel/tuned.py of the harness commit the recorded rows name has a
# quoted path inside an f-string replacement field, which Python 3.10 and 3.11
# reject. The file of HEAD builds the path in a variable first. Both give the
# same command text, so the one rewrite is accepted: old blob to new blob, and
# only for that harness commit, whose SHA fixes every file tuned.py imports.
TUNED_SOURCE_REWRITES = {
    "3d71865c78e5ff19866bae2616251929fd08e453": (
        "9b8e4c3a12291fa33804fa35a2c09a6142316233",
        "78b5aae0f4d22fa4f8237a2bb68aded3f5f3d197",
    ),
}
TUNED_SOURCE = "scripts/gmin_panel/tuned.py"


def is_full_sha(value: str | None) -> bool:
    """Forty lowercase hex characters, with nothing around them."""
    return FULL_SHA.fullmatch(value or "") is not None


def is_abbreviated_sha(value: str | None) -> bool:
    """Seven to thirty-nine lowercase hex characters: a prefix of a SHA, which no row may cite."""
    return ABBREVIATED_SHA.fullmatch(value or "") is not None


def commit_id_pattern(shortest: int = 40) -> str:
    """Regex source for a commit id of ``shortest`` to forty lowercase hex characters.

    For text that quotes a commit, which may name a prefix. A row's revision is
    judged by ``is_full_sha``, never by this pattern.
    """
    return "[0-9a-f]{%d,40}" % shortest


def external_form(rev: str | None) -> bool:
    """A repository URL and a full SHA, for a commit this repository does not contain."""
    return EXTERNAL_FORM.fullmatch((rev or "").strip()) is not None


def external_pin(rev: str | None, *, abbreviated: bool = False) -> bool:
    """The revision is one of the external pins.

    A prefix of a pin counts only with ``abbreviated``, for prose that cites
    eight characters. A row never cites a prefix.
    """
    rev = (rev or "").strip()
    if is_full_sha(rev) or FULL_PIN_DIGEST.fullmatch(rev):
        return rev in EXTERNAL_REVISIONS
    if abbreviated and is_abbreviated_sha(rev):
        return any(pin.startswith(rev) for pin in EXTERNAL_REVISIONS)
    return False


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


def revision_reachable(
    revision: str | None, repo: Path = ROOT, held: Callable[[str], bool] | None = None,
) -> bool:
    """A row's revision: a full SHA that is an ancestor of HEAD, a pin, or a repository URL at a full SHA.

    ``held`` replaces the ancestry question, for a caller that caches it.
    """
    revision = (revision or "").strip()
    if external_form(revision) or external_pin(revision):
        return True
    if not is_full_sha(revision):
        return False
    return held(revision) if held is not None else commit_held(revision, repo)


def tuned_gmin_harness_verified(
    row: dict[str, str], repo: Path = ROOT, held: Callable[[str], bool] | None = None,
) -> bool:
    """Verify a tuned-GMIN row: reachable code_revision, exact data_blob and tuned.py in HEAD.

    A harness commit that is an ancestor of HEAD is accepted. Otherwise it must
    be a commit of ``repo`` whose ``bench/gmin-tuned/inputs/<system>.data`` is
    the recorded blob, whose ``scripts/gmin_panel/tuned.py`` is the one of HEAD
    or, for the harness ``TUNED_SOURCE_REWRITES`` names, its listed old blob,
    and that changed no ``src/``, ``examples/`` or Cargo file and no Python
    file under ``scripts/`` but tuned.py since the merge base with HEAD, so the
    modules tuned.py imports are ones HEAD's history holds.
    """

    def run(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(repo), *args], input=stdin, capture_output=True, text=True, check=False)

    system = (row.get("system") or "").strip()
    data_blob = (row.get("data_blob") or "").strip()
    harness = (row.get("harness_commit") or "").strip()
    code_rev = (row.get("code_revision") or "").strip()
    if system not in TUNED_SYSTEMS or not data_blob or not harness:
        return False
    if not revision_reachable(code_rev, repo, held):
        return False
    if revision_reachable(harness, repo, held):
        return True
    if not is_full_sha(harness):
        return False
    head_data = run("rev-parse", f"HEAD:bench/gmin-tuned/inputs/{system}.data")
    head_tuned = run("rev-parse", f"HEAD:{TUNED_SOURCE}")
    if head_data.returncode != 0 or head_data.stdout.strip() != data_blob or head_tuned.returncode != 0:
        return False
    present = run("cat-file", "--batch-check", stdin=f"{harness}^{{commit}}\n").stdout.split()
    if present[1:2] != ["commit"]:
        return False
    ref_data = run("rev-parse", f"{harness}:bench/gmin-tuned/inputs/{system}.data")
    ref_tuned = run("rev-parse", f"{harness}:{TUNED_SOURCE}")
    if ref_data.returncode != 0 or ref_data.stdout.strip() != data_blob:
        return False
    ref_blob, head_blob = ref_tuned.stdout.strip(), head_tuned.stdout.strip()
    if ref_tuned.returncode != 0 or (ref_blob != head_blob and TUNED_SOURCE_REWRITES.get(harness) != (ref_blob, head_blob)):
        return False
    base = run("merge-base", "HEAD", harness)
    if base.returncode != 0 or not base.stdout.strip():
        return False
    diff = run("diff", "--name-only", base.stdout.strip(), harness)
    if diff.returncode != 0:
        return False
    return not any(
        name.startswith(("src/", "examples/")) or name in {"Cargo.toml", "Cargo.lock"}
        or (name.startswith("scripts/") and name.endswith(".py") and name != TUNED_SOURCE)
        for name in diff.stdout.splitlines()
    )


def main(argv: list[str]) -> int:
    """``full-sha VALUE...`` exits 0 when every value is a full SHA, for the shell scripts that check one."""
    if len(argv) < 2 or argv[0] != "full-sha":
        print("usage: revision_reach.py full-sha VALUE...", file=sys.stderr)
        return 2
    return 0 if all(is_full_sha(value) for value in argv[1:]) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
