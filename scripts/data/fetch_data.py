#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fetch the paper dumps from the Hugging Face dataset anneal-paper-data.

The dataset repository is ``HaoZeke/anneal-paper-data`` (repo type
dataset). ``HF_REVISION`` is the pinned dataset commit, which holds every
manifest file.
An environment variable of the same name overrides either value, so
another machine can point at another snapshot without editing this
file.

``data/hf-files.csv`` lists every file the dataset holds: a path
relative to the anneal repository root, the size in bytes, and the
sha256. The dataset layout is those same paths. Groups:

- ``paper/jctc/data/landscapes_7695255/`` minima dumps and their provenance
- ``paper/jctc/data/**/*.min`` other minima dumps
- ``paper/jctc/data/**/*.ld`` landfold landscape dumps
- ``paper/jctc/data/**/*.xyz`` structure dumps
- ``paper/jctc/data/w6gofee4k/*.dat`` per-seed call traces
- ``paper/jctc/data/system_visuals/landfold/*_hist.txt`` bulk histograms
- ``paper/jctc/logs/water/`` raw xtb and CREST runs
- ``paper/jctc/logs/logs-water.tar.xz`` and ``water-tarball-map.txt``
- ``paper/jctc/logs/lj75-source-37535ec1.tar.xz`` and its ``.lst``
- ``paper/jctc/logs/lj*-seed*.log`` raw Lennard-Jones seed logs
- ``paper/jctc/logs/cuh2_fcc_rec_*.out`` raw copper runs
- ``bench/depth-posterior-ablation/logs/`` raw depth-posterior ablation logs
- ``bench/metal-cluster-h2h/logs/`` raw metal head-to-head logs
- ``bench/metal-cluster-h2h/darby-logs/`` raw Darby potential logs

Small summary CSVs, ``tests/fixtures``, rendered figures, and source
stay in git. This script checks every effective manifest row. When the
working tree already matches, it does not contact the hub. A missing
or mismatched file is downloaded directly at ``HF_REVISION`` and then
checked again. The request does not authenticate.

Usage: fetch_data.py [PREFIX ...]

With prefixes, only the manifest rows under them are checked and fetched;
the figure scripts ask for ``paper/jctc/``. Without, every row is. A row
the dataset does not hold at ``HF_REVISION`` is reported by its group
and fails the run.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "hf-files.csv"
MANIFEST_UPDATES = ROOT / "data" / "hf-files-updates.csv"

HF_REPO = os.environ.get("HF_REPO", "HaoZeke/anneal-paper-data")
# Dataset commit that holds every file in data/hf-files.csv.
HF_REVISION_DEFAULT = "71185263eb6a8130d1262fd76d90109f5fabdcf5"
HF_REVISION = os.environ.get(
    "HF_REVISION", HF_REVISION_DEFAULT
)
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

CACHE = Path(
    os.environ.get(
        "ANNEAL_HF_CACHE",
        Path.home() / ".cache" / "anneal" / "anneal-paper-data",
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path | None = None) -> list[dict[str, str]]:
    path = path or MANIFEST
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no files")
    for row in rows:
        if not {"path", "size", "sha256"} <= set(row):
            raise SystemExit(f"{path} must have columns path,size,sha256")
        row["path"] = row["path"].strip()
        row["size"] = row["size"].strip()
        row["sha256"] = row["sha256"].strip()
        parts = Path(row["path"]).parts
        if (
            not row["path"]
            or row["path"].startswith("/")
            or ".." in parts
            or not row["size"].isdigit()
            or len(row["sha256"]) != 64
        ):
            raise SystemExit(f"refusing manifest row {row['path']!r}")
    if path != MANIFEST or not MANIFEST_UPDATES.is_file():
        return rows
    by_path = {row["path"]: row for row in rows}
    with MANIFEST_UPDATES.open(newline="") as handle:
        for update in csv.DictReader(handle):
            name = update["path"].strip()
            if update["status"] == "superseded":
                by_path.pop(name, None)
            elif update["status"] == "active":
                by_path[name] = {
                    key: update[key].strip()
                    for key in ("path", "size", "sha256")
                }
            else:
                raise SystemExit(
                    f"{MANIFEST_UPDATES}: invalid status {update['status']!r}"
                )
    return list(by_path.values())


def matches(path: Path, size: str, digest: str) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size != int(size):
        return False
    return sha256(path) == digest


def resolve_url(path: str) -> str:
    quoted = urllib.parse.quote(path)
    return f"{HF_ENDPOINT}/datasets/{HF_REPO}/resolve/{HF_REVISION}/{quoted}"


def download(path: str, dest: Path) -> bool:
    """Fetch one dataset file to dest; False when the revision does not hold it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(resolve_url(path), timeout=120) as response, part.open("wb") as out:
            shutil.copyfileobj(response, out)
    except urllib.error.HTTPError as exc:
        part.unlink(missing_ok=True)
        if exc.code == 404:
            return False
        raise SystemExit(f"{HF_REPO}@{HF_REVISION}: {path}: HTTP {exc.code}") from exc
    part.replace(dest)
    return True


def group(path: str) -> str:
    return "/".join(Path(path).parts[:2])


def ensure(root: Path = ROOT, prefixes: tuple[str, ...] = ()) -> None:
    """Make every manifest file under prefixes, or every one, present at its repository path."""
    rows = [row for row in read_manifest() if not prefixes or row["path"].startswith(prefixes)]
    if prefixes and not rows:
        raise SystemExit(f"no row of {MANIFEST.name} lies under {' '.join(prefixes)}")
    missing = [
        row
        for row in rows
        if not matches(root / row["path"], row["size"], row["sha256"])
    ]
    if not missing:
        return
    absent = []
    for row in missing:
        src = CACHE / HF_REVISION / row["path"]
        if not matches(src, row["size"], row["sha256"]):
            if not download(row["path"], src):
                absent.append(row["path"])
                continue
            if not matches(src, row["size"], row["sha256"]):
                raise SystemExit(f"downloaded {row['path']} does not match {MANIFEST.name}")
        dest = root / row["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        if not matches(dest, row["size"], row["sha256"]):
            raise SystemExit(f"copied {row['path']} does not match {MANIFEST.name}")
    if absent:
        groups = ", ".join(f"{name} ({count})" for name, count in sorted(Counter(map(group, absent)).items()))
        raise SystemExit(
            f"{len(absent)} file(s) of {MANIFEST.name} are not in {HF_REPO} at {HF_REVISION}: {groups}; "
            f"first {absent[0]}"
        )


def main(argv: list[str]) -> int:
    ensure(prefixes=tuple(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
