#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Scrub selected text logs and create the registered log archive."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

LOCAL_PREFIXES = ("/tmp/", "/home/", "/opt/", "/agent/")
INHERITED_NOTICE = "Inherited environment"
REPLACEMENT = "[scrubbed]"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def secret_values() -> list[str]:
    values = {
        value
        for value in os.environ.values()
        if len(value) > 16 and "\n" not in value and "\0" not in value
    }
    return sorted(values, key=len, reverse=True)


def scrub(path: Path, secrets: list[str]) -> None:
    raw = path.read_bytes()
    if b"\0" in raw:
        raise SystemExit(f"refusing non-text log {path.name}")
    text = raw.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")
    for item in (*LOCAL_PREFIXES, INHERITED_NOTICE, *secrets):
        text = text.replace(item, REPLACEMENT)
    path.write_text(text, encoding="utf-8", newline="\n")


def grep_clean(stage: Path, secrets: list[str]) -> None:
    patterns = [*LOCAL_PREFIXES, INHERITED_NOTICE, *secrets]
    proc = subprocess.run(
        ["grep", "-R", "-I", "-l", "-F", "-f", "-", "."],
        cwd=stage,
        input="\n".join(patterns) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0:
        raise SystemExit(f"pre-archive grep found sensitive text:\n{proc.stdout}")
    if proc.returncode != 1:
        raise SystemExit(f"pre-archive grep failed:\n{proc.stderr}")
    for path in stage.rglob("*"):
        if path.is_file() and b"\r" in path.read_bytes():
            raise SystemExit(f"non-LF text remains in {path.name}")


def move_aside(path: Path, trash: Path) -> None:
    if not path.exists():
        return
    trash.mkdir(parents=True, exist_ok=True)
    target = trash / f"{path.name}.{os.getpid()}"
    if target.exists():
        raise SystemExit(f"trash target already exists: {target.name}")
    path.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trash", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".logs-stage-{os.getpid()}"
    temporary = output.with_name(output.name + ".tmp")
    move_aside(stage, args.trash)
    move_aside(temporary, args.trash)
    stage.mkdir()

    for source in args.source:
        source = source.resolve()
        if not source.is_dir():
            raise SystemExit(f"log source is not a directory: {source.name}")
        target = stage / source.name
        if target.exists():
            raise SystemExit(f"duplicate log source name: {source.name}")
        shutil.copytree(source, target)

    secrets = secret_values()
    for path in stage.rglob("*"):
        if path.is_file():
            scrub(path, secrets)
    grep_clean(stage, secrets)

    subprocess.run(
        [
            "tar",
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "-czf",
            str(temporary),
            "-C",
            str(stage),
            ".",
        ],
        check=True,
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.sync()
    move_aside(output, args.trash)
    temporary.replace(output)
    os.sync()
    digest = sha256(output)
    print(f"{digest}  {output.name}")
    move_aside(stage, args.trash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
