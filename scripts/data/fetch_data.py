#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fetch the paper dumps from the Hugging Face dataset anneal-paper-data.

The dataset repository is ``rgoswami/anneal-paper-data`` (repo type
dataset). ``HF_REVISION`` is the pinned dataset commit. Replace
``PINNED_REVISION`` after the upload.
An environment variable of the same name overrides either value, so
another machine can point at the uploaded snapshot without editing
this file.

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

Small summary CSVs, ``tests/fixtures``, rendered figures, and source
stay in git. This script checks every manifest row. When the working
tree already matches, it does not contact the hub. A miss is filled
with ``huggingface_hub.snapshot_download`` at ``HF_REVISION``, then
checked again. The call does not authenticate.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "hf-files.csv"

# Dataset repository rgoswami/anneal-paper-data, repo type dataset.
HF_REPO = os.environ.get("HF_REPO", "rgoswami/anneal-paper-data")
# Pin the dataset git revision after upload.
HF_REVISION = os.environ.get("HF_REVISION", "PINNED_REVISION")

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


def read_manifest(path: Path = MANIFEST) -> list[dict[str, str]]:
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
    return rows


def matches(path: Path, size: str, digest: str) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size != int(size):
        return False
    return sha256(path) == digest


def ensure(root: Path = ROOT) -> None:
    """Make every manifest file present at its repository path."""
    rows = read_manifest()
    missing = [
        row
        for row in rows
        if not matches(root / row["path"], row["size"], row["sha256"])
    ]
    if not missing:
        return
    if HF_REVISION == "PINNED_REVISION":
        sample = missing[0]["path"]
        raise SystemExit(
            f"{len(missing)} file(s) do not match {MANIFEST.name}, "
            f"including {sample}. HF_REPO is {HF_REPO} and HF_REVISION is "
            f"{HF_REVISION}. Set HF_REVISION to the uploaded dataset commit "
            "before downloading."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required to download paper data"
        ) from exc
    CACHE.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        revision=HF_REVISION,
        allow_patterns=[row["path"] for row in rows],
        local_dir=str(CACHE),
        token=False,
    )
    for row in missing:
        src = CACHE / row["path"]
        dest = root / row["path"]
        if not matches(src, row["size"], row["sha256"]):
            raise SystemExit(
                f"downloaded {row['path']} does not match {MANIFEST.name}"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        if not matches(dest, row["size"], row["sha256"]):
            raise SystemExit(f"copied {row['path']} does not match {MANIFEST.name}")
    still = [
        row["path"]
        for row in rows
        if not matches(root / row["path"], row["size"], row["sha256"])
    ]
    if still:
        raise SystemExit(f"still missing {still[0]}")


def main() -> int:
    ensure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
