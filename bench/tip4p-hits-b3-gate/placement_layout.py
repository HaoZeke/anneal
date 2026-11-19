#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Layout copies of an installed anneal package.

Each variant is a copy of arm A's package. Only the shared object
``anneal/_core`` changes, and only by a trailing padding section or by
that section's alignment. A pure padding variant keeps the original text
section byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def core_object(package: Path) -> Path:
    found = sorted((package / "anneal").glob("_core*.so"))
    if len(found) != 1:
        raise SystemExit(f"{package} has {len(found)} _core objects")
    return found[0]


def section_bytes(path: Path, section: str) -> bytes:
    out = path.with_suffix(f".{section}.bin")
    proc = subprocess.run(
        ["objcopy", "--dump-section", f"{section}={out}", str(path)],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.decode("utf-8", errors="replace")[-500:])
    data = out.read_bytes()
    out.unlink(missing_ok=True)
    return data


def pad_object(path: Path, nbytes: int, align: int | None) -> None:
    if nbytes < 1:
        raise SystemExit("padding must be at least one byte")
    blob = path.with_suffix(".padbin")
    blob.write_bytes(b"\0" * nbytes)
    cmd = [
        "objcopy",
        "--add-section",
        f".gatepad={blob}",
        "--set-section-flags",
        ".gatepad=contents,readonly",
        str(path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    blob.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr[-500:])
    if align is None:
        return
    # A combined add-section invocation leaves the new section at alignment 1.
    aligned = subprocess.run(
        ["objcopy", "--set-section-alignment", f".gatepad={align}", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if aligned.returncode != 0:
        raise SystemExit(aligned.stderr[-500:])


def make_variant(source: Path, dest: Path, nbytes: int, align: int | None) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    original = core_object(source)
    copied = core_object(dest)
    before = section_bytes(original, ".text")
    pad_object(copied, nbytes, align)
    after = section_bytes(copied, ".text")
    if before != after:
        raise SystemExit(f"{dest.name}: text section changed")
    print(
        f"{dest.name} bytes={copied.stat().st_size} text_sha256={hashlib.sha256(after).hexdigest()} "
        f"file_sha256={sha256_file(copied)}",
        flush=True,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--align", type=int)
    args = parser.parse_args(argv[1:])
    make_variant(args.source, args.dest, args.bytes, args.align)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
