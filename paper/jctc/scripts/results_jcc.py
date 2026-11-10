#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Read the sealed results_jcc campaigns of the anneal_repro port.

data/results_jcc_c47d301/ copies, byte for byte, the folders of the handoff that the
drafts quote: lj_cell and gmin_lj75, the LJ75 boards, and four arms of the water hexamer
against CREST under water_36086e98/. provenance.txt records the digest of every file, and
each folder carries its own SHA256SUMS, PROVENANCE and TERMINAL_OK. load holds every file
to both, requires the TERMINAL_OK line of a sealed folder, and returns for each folder its
PROVENANCE keys and its summary.json, so a script prints what the seal gives and nothing
it computes.

Usage: results_jcc.py [data-dir]   prints each folder, its rule and its binary
"""
from __future__ import annotations

import functools
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "results_jcc_c47d301"
SEALED = "sealed JCC production evidence"
FOLDERS = (
    "lj_cell",
    "gmin_lj75",
    "water_36086e98/default",
    "water_36086e98/budget-fair",
    "water_36086e98/crest",
    "water_36086e98/crest-polished",
)
SEALED_FILES = ("PROVENANCE", "SHA256SUMS", "TERMINAL_OK", "summary.json")


def die(message: str) -> NoReturn:
    raise SystemExit(f"results_jcc: {message}")


@dataclass(frozen=True)
class Folder:
    name: str
    record: dict[str, str]
    summary: dict


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(data: Path) -> tuple[dict[str, str], dict[str, str]]:
    """The keyed lines of provenance.txt, and the digest it records for each file."""
    keys: dict[str, str] = {}
    digests: dict[str, str] = {}
    for line in (data / "provenance.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        first, _, rest = line.partition(" ")
        if len(first) == 64 and all(c in "0123456789abcdef" for c in first):
            digests[rest.strip()] = first
        else:
            keys[first] = rest.strip()
    return keys, digests


def _sums(path: Path) -> dict[str, str]:
    found = {}
    for line in path.read_text().splitlines():
        value, _, name = line.partition("  ")
        found[name.removeprefix("./")] = value
    return found


def _keyed(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if line)


@functools.cache
def load(data: Path = DATA) -> tuple[dict[str, str], dict[str, Folder]]:
    keys, digests = provenance(data)
    listed = {str(path.relative_to(data)) for path in data.rglob("*") if path.is_file()} - {"provenance.txt"}
    if listed != set(digests):
        die(f"provenance.txt does not list exactly the files of {data.name}")
    for name, want in digests.items():
        if digest(data / name) != want:
            die(f"{name} does not match the digest provenance.txt records")
    folders = {}
    for name in FOLDERS:
        root = data / name
        present = sorted(path.name for path in root.iterdir())
        if present != sorted(SEALED_FILES):
            die(f"{name} holds {present}, not the four files of a seal")
        sealed = _sums(root / "SHA256SUMS")
        if sorted(sealed) != sorted(set(SEALED_FILES) - {"SHA256SUMS"}):
            die(f"{name}/SHA256SUMS does not seal its folder")
        for file, want in sealed.items():
            if digest(root / file) != want:
                die(f"{name}/{file} fails sha256sum -c")
        if (root / "TERMINAL_OK").read_text().strip() != SEALED:
            die(f"{name}/TERMINAL_OK is not the line of a sealed folder")
        record = _keyed(root / "PROVENANCE")
        summary = json.loads((root / "summary.json").read_text())
        if record.get("binary_sha256") != summary.get("binary_sha256"):
            die(f"{name}: PROVENANCE and summary.json name different binaries")
        if "anneal_commit" in summary and record.get("anneal_commit", "") != summary["anneal_commit"]:
            die(f"{name}: PROVENANCE and summary.json name different anneal commits")
        folders[name] = Folder(name, record, summary)
    return keys, folders


def main(argv: list[str]) -> None:
    keys, folders = load(Path(argv[0]) if argv else DATA)
    print(f"port {keys['port_commit']}")
    for folder in folders.values():
        print(f"{folder.name} {folder.record['rule']} {folder.record['binary_sha256']}")


if __name__ == "__main__":
    main(sys.argv[1:])
