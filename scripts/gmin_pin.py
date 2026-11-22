#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One verified source pin for every GMIN build and benchmark row."""

from __future__ import annotations

import argparse
import hashlib
import re
import shlex
from collections.abc import Sequence
from pathlib import Path

SNAPSHOT = "2026-11-19"
URL = "http://www-wales.ch.cam.ac.uk/svn.tar.bz2"
REVISION = "9cbec1a446352bc9ef064bd2b1a5767194655919"
TARBALL_SHA256 = "ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776"
BUILD_KEY_RE = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tarball(path: Path) -> str:
    """Refuse any tarball other than the pinned snapshot."""
    actual = sha256(path)
    if actual != TARBALL_SHA256:
        raise ValueError(
            f"tarball sha256 mismatch: got {actual}, expected {TARBALL_SHA256}"
        )
    return actual


def read_svnrev(path: Path) -> str:
    """Read and verify the revision embedded in a built source tree."""
    actual = path.read_text().strip()
    if actual != REVISION:
        raise ValueError(f"GMIN SVNREV is {actual}, expected {REVISION}")
    return actual


def compute_build_key(
    revision: str,
    tarball_sha256: str,
    inputs: Sequence[Path],
    options: Sequence[str],
) -> str:
    """Hash source revision, tarball digest, input file contents, and build options."""
    rev = revision.strip()
    tar_digest = tarball_sha256.strip()
    if not rev:
        raise ValueError("missing GMIN revision for build key")
    if not BUILD_KEY_RE.fullmatch(tar_digest):
        raise ValueError(f"invalid tarball sha256 for build key: {tar_digest!r}")
    if not inputs:
        raise ValueError("build key needs at least one input file")
    digest = hashlib.sha256()
    digest.update(f"gmin-build-key-v1\nrevision={rev}\n".encode("utf-8"))
    digest.update(f"tarball_sha256={tar_digest}\n".encode("utf-8"))
    for index, raw in enumerate(inputs):
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"missing GMIN build input: {path}")
        digest.update(f"input[{index}]:{path.name}={sha256(path)}\n".encode("utf-8"))
    for opt in options:
        digest.update(f"option:{opt}\n".encode("utf-8"))
    return digest.hexdigest()


def read_build_key(metadata: Path | None = None, key_file: Path | None = None) -> str:
    """Return the recorded build key when present and consistent, else empty."""
    meta_key = ""
    if metadata is not None and metadata.is_file():
        source = dict(
            line.split("=", 1)
            for line in metadata.read_text().splitlines()
            if "=" in line
        )
        meta_key = source.get("build_key", "").strip()
        if not BUILD_KEY_RE.fullmatch(meta_key):
            return ""
    file_key = ""
    if key_file is not None and key_file.is_file():
        file_key = key_file.read_text().strip()
        if not BUILD_KEY_RE.fullmatch(file_key):
            return ""
    if metadata is not None and metadata.is_file() and key_file is not None and key_file.is_file():
        if meta_key != file_key:
            return ""
    if metadata is not None and metadata.is_file():
        return meta_key
    return file_key


def require_build_key(value: str, label: str = "build_key") -> str:
    """Validate and return a 64-hex GMIN build key, rejecting absent or malformed values."""
    cleaned = str(value).strip()
    if not BUILD_KEY_RE.fullmatch(cleaned):
        raise ValueError(f"missing or invalid {label}: {value!r}")
    return cleaned


def read_built_revision(svnrev: Path, metadata: Path | None = None) -> str:
    """Read a built SVNREV and optionally verify its generated source note."""
    revision = read_svnrev(svnrev)
    if metadata is not None:
        source = dict(
            line.split("=", 1)
            for line in metadata.read_text().splitlines()
            if "=" in line
        )
        if source.get("svnrev") != revision:
            raise ValueError(f"{metadata} does not describe {svnrev}")
        if source.get("tarball_sha256") != TARBALL_SHA256:
            raise ValueError(f"{metadata} has another GMIN tarball")
        require_build_key(source.get("build_key", ""), f"{metadata} build_key")
    return revision


def shell_assignments() -> str:
    """Return shell-safe assignments consumed by build scripts."""
    values = {
        "GMIN_PIN_SNAPSHOT": SNAPSHOT,
        "GMIN_PIN_URL": URL,
        "GMIN_PIN_REVISION": REVISION,
        "GMIN_PIN_TARBALL_SHA256": TARBALL_SHA256,
    }
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("shell")
    tarball = sub.add_parser("verify-tarball")
    tarball.add_argument("path", type=Path)
    svnrev = sub.add_parser("read-svnrev")
    svnrev.add_argument("path", type=Path)
    bkey = sub.add_parser("build-key")
    bkey.add_argument("--revision", default=REVISION)
    bkey_tar = bkey.add_mutually_exclusive_group(required=True)
    bkey_tar.add_argument("--tarball", type=Path)
    bkey_tar.add_argument("--tarball-sha256")
    bkey.add_argument("--input", dest="inputs", type=Path, action="append", default=[])
    bkey.add_argument("--option", dest="options", action="append", default=[])
    rkey = sub.add_parser("read-build-key")
    rkey.add_argument("--metadata", type=Path)
    rkey.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "shell":
            print(shell_assignments())
        elif args.command == "verify-tarball":
            print(verify_tarball(args.path))
        elif args.command == "read-svnrev":
            print(read_svnrev(args.path))
        elif args.command == "build-key":
            digest = (
                verify_tarball(args.tarball)
                if args.tarball is not None
                else args.tarball_sha256
            )
            print(compute_build_key(args.revision, digest, args.inputs, args.options))
        else:
            print(read_build_key(args.metadata, args.key_file))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
