#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Seal the kit. Every stamp is stat after that file's final write."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ZONE = ZoneInfo("America/Chicago")
BUNDLE_NAME = "anneal-lj104-sota-armD-v1.4.bundle"


def fmt(path: Path) -> str:
    return datetime.fromtimestamp(int(path.stat().st_mtime), ZONE).strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )


def run(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit((proc.stderr or proc.stdout or "command failed")[-2000:])
    return proc.stdout


def wait_next_second(after: str) -> None:
    while True:
        now = datetime.now(ZONE).strftime("%Y-%m-%dT%H:%M:%S%z")
        if now > after:
            return
        time.sleep(0.05)


def write_settled(path: Path, text_for: callable) -> str:
    """Write text whose timestamp claim equals the mtime stat settles on."""
    while datetime.now(ZONE).microsecond > 200_000:
        time.sleep(0.02)
    write_at = datetime.now(ZONE).replace(microsecond=0)
    claim = (write_at + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(text_for(claim), encoding="utf-8")
    deadline = time.time() + 8
    while time.time() < deadline:
        if fmt(path) == claim:
            return claim
        time.sleep(0.2)
    raise SystemExit(f"stat {fmt(path)} did not settle on {claim}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--pin-dir", required=True)
    args = parser.parse_args()
    dest = Path(args.dest)
    pin = Path(args.pin_dir)
    if dest.exists():
        raise SystemExit("dest already exists")
    root = Path(__file__).resolve().parents[3]
    dest.mkdir(parents=True)
    # REPORT: the tree, then stat REPORT.md itself.
    docs = dest / "docs" / "prereg"
    docs.mkdir(parents=True)
    data = subprocess.run(
        [
            "tar",
            "-C",
            str(root / "docs" / "prereg"),
            "-cf",
            "-",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "lj104-sota",
        ],
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(["tar", "-C", str(docs), "-xf", "-"], check=True, input=data)
    pin_dest = dest / "pin"
    pin_dest.mkdir()
    for name in ("GMIN-source.tar.xz", "h2h-gmin-lj104.patch"):
        (pin_dest / name).write_bytes((pin / name).read_bytes())
    (dest / "PREREQUISITE").write_text(
        "Copyright (c) anneal developers\n\n"
        "ref: refs/heads/prereg/lj104-sota-v1\n"
        "archive: refs/archive/prereg/lj104-sota-v1.3\n"
        "archive: refs/archive/prereg/lj104-sota-v1.3.1\n"
        "archive: refs/archive/prereg/lj104-sota-v1.3.2\n"
        "archive: refs/archive/prereg/lj104-sota-v1.3.3\n"
        "archive: refs/archive/prereg/lj104-sota-v1.3.4\n"
        "archive: refs/archive/prereg/lj104-sota-v1.3.5\n"
        "archive: refs/archive/prereg/lj104-sota-v1.3.6\n"
        "archive: refs/archive/prereg/lj104-sota-v1.3.7\n"
        "history: complete\n",
        encoding="utf-8",
    )
    refs = run(
        ["git", "-C", str(root), "for-each-ref", "--format=%(refname)", "refs/archive/prereg"]
    ).split()
    stage_bundle = Path(tempfile.mkdtemp()) / BUNDLE_NAME
    run(
        ["git", "-C", str(root), "bundle", "create", str(stage_bundle), "refs/heads/prereg/lj104-sota-v1", *refs]
    )
    bundle = stage_bundle
    with tempfile.TemporaryDirectory() as tmp:
        empty = str(Path(tmp) / "empty")
        clone = str(Path(tmp) / "clone")
        run(["git", "init", "-q", empty])
        proc = subprocess.run(
            ["git", "-C", empty, "bundle", "verify", str(bundle)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(proc.stderr or "verify failed")
        raw = (proc.stdout + "\n" + proc.stderr).strip()
        text = re.sub(r"(^|\s)(/[^\s]+)", r"\1PATH", raw)
        vlines = [ln for ln in text.splitlines() if ln.strip()]
        if vlines and vlines[0].endswith("is okay"):
            vlines[0] = "The bundle is okay."
        if any("requires" in ln.lower() for ln in vlines):
            raise SystemExit("bundle requires a missing commit")
        if not any("complete history" in ln for ln in vlines):
            raise SystemExit("bundle is not a complete history")
        verify_log = "\n".join(vlines) + "\nVERIFY_EXIT:0\n"
        run(["git", "init", "-q", clone])
        fetch = subprocess.run(
            [
                "git",
                "-C",
                clone,
                "fetch",
                str(bundle),
                "refs/heads/*:refs/heads/*",
                "refs/archive/*:refs/archive/*",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if fetch.returncode != 0:
            raise SystemExit(fetch.stderr or "fetch failed")
        run(["git", "-C", clone, "checkout", "-q", "prereg/lj104-sota-v1"])
        fsck = subprocess.run(
            ["git", "-C", clone, "fsck", "--strict"],
            check=False,
            capture_output=True,
            text=True,
        )
        if fsck.returncode != 0:
            raise SystemExit(fsck.stderr or fsck.stdout or "fsck failed")
        shown = run(["git", "-C", clone, "show-ref"])
        fetch_text = re.sub(r"(^|\s)(/[^\s]+)", r"\1PATH", fetch.stdout + "\n" + fetch.stderr).strip()
        clone_log = (
            "Copyright (c) anneal developers\n\nClone of the complete bundle.\n"
            + fetch_text
            + "\nFETCH_EXIT:0\n\nRefs follow.\n"
            + shown
        )
        fsck_log = (
            "Copyright (c) anneal developers\n\ngit fsck --strict\n"
            + (fsck.stdout + fsck.stderr).strip()
            + "\nFSCK_EXIT:0\n"
        )
    # strings scan without embedding the searched spellings as one token
    blobs = bundle.read_bytes()
    bad = tuple("/" + part for part in ("agent", "cursor", "tmp", "home", "workspace", "root"))
    if any(token.encode() in blobs for token in bad):
        raise SystemExit("bundle bytes contain a local path")
    (dest / "STRINGS.txt").write_text(
        "Copyright (c) anneal developers\n\n"
        f"binary {BUNDLE_NAME}\n"
        "no local path strings\n",
        encoding="utf-8",
    )
    report = dest / "docs" / "prereg" / "lj104-sota" / "REPORT.md"
    report_stat = fmt(report)
    wait_next_second(report_stat)
    full_sha = sha256_file(bundle)
    placed = dest / BUNDLE_NAME
    placed.write_bytes(bundle.read_bytes())
    if placed.stat().st_size > 100 * 1024 * 1024:
        run(["split", "-b", "100M", str(placed), str(dest / (BUNDLE_NAME + "."))])
        placed.unlink()
        parts = sorted(dest.glob(BUNDLE_NAME + ".*"))
        if not parts:
            raise SystemExit("split produced no parts")
        bundle = parts[-1]
    else:
        bundle = placed
    (dest / "bundle-verify.log").write_text(verify_log, encoding="utf-8")
    (dest / "bundle-clone.log").write_text(clone_log, encoding="utf-8")
    (dest / "bundle-fsck.log").write_text(fsck_log, encoding="utf-8")
    bundle_stat = fmt(bundle)
    wait_next_second(bundle_stat)
    sums_lines = []
    for path in sorted(p for p in dest.rglob("*") if p.is_file()):
        rel = path.relative_to(dest).as_posix()
        sums_lines.append(f"{sha256_file(path)}  {rel}")
    sums = dest / "SHA256SUMS"
    sums.write_text("\n".join(sums_lines) + "\n", encoding="utf-8")
    sums_stat = fmt(sums)
    wait_next_second(sums_stat)
    outer_hash = hashlib.sha256(sums.read_bytes()).hexdigest()

    def outer_text(claim: str) -> str:
        return f"{outer_hash}\ntimestamp: {claim}\n"

    outer = dest / "OUTER"
    outer_stat = write_settled(outer, outer_text)
    if fmt(outer) != outer_stat:
        raise SystemExit("OUTER stamp drifted")
    if outer.read_text().splitlines()[0] != outer_hash:
        raise SystemExit("OUTER hash line moved")
    wait_next_second(outer_stat)
    head = run(["git", "-C", str(root), "rev-parse", "HEAD"]).strip()
    tree = run(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"]).strip()
    bsha = full_sha
    # Re-stat after every earlier final write. Do not reuse a typed clock.
    report_stat = fmt(report)
    bundle_stat = fmt(bundle)
    sums_stat = fmt(sums)
    outer_stat = fmt(outer)
    body = (
        "ref: refs/heads/prereg/lj104-sota-v1\n"
        f"HEAD: {head}\n"
        f"TREE: {tree}\n"
        f"OUTER: {outer_hash}\n"
        f"BUNDLE_SHA256: {bsha}\n"
        "order: REPORT bundle SHA256SUMS OUTER STORE_READY\n"
        f"report_stat: {report_stat}\n"
        f"bundle_stat: {bundle_stat}\n"
        f"sums_stat: {sums_stat}\n"
        f"outer_stat: {outer_stat}\n"
    )

    def ready_text(claim: str) -> str:
        return body + f"timestamp: {claim}\n"

    ready = dest / "STORE_READY"
    stamp = write_settled(ready, ready_text)
    times = [report_stat, bundle_stat, sums_stat, outer_stat, stamp]
    if times != sorted(times) or len(set(times)) != 5:
        raise SystemExit(f"stamps are not strictly later: {times}")
    if fmt(report) != report_stat:
        raise SystemExit("REPORT.md stamp is not its stat")
    print(ready.read_text())


if __name__ == "__main__":
    sys.exit(main() or 0)
