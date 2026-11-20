#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fetch or regenerate benchmark data that git does not store.

Raw logs, trajectories, frames, and the archives headed for Zenodo
are listed in bench_data_manifest.txt. They live in the pinned
HaoZeke/anneal-paper-data dataset. HF_DATASET and HF_REVISION can
override that snapshot. Each file is downloaded to the same relative path.

The water hexamer session is regenerated from the committed driver
instead of being downloaded:

    python3 paper/jctc/scripts/fetch_bench_data.py --water
"""

import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path(__file__).with_name("bench_data_manifest.txt")
HF_DATASET = os.environ.get(
    "HF_DATASET", "HaoZeke/anneal-paper-data"
).strip() or "HaoZeke/anneal-paper-data"
HF_REVISION_DEFAULT = "71185263eb6a8130d1262fd76d90109f5fabdcf5"
HF_REVISION = os.environ.get(
    "HF_REVISION", HF_REVISION_DEFAULT
).strip() or HF_REVISION_DEFAULT


def manifest_paths():
    paths = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        path = Path(text)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"manifest path is not relative: {text}")
        paths.append(path)
    if not paths:
        raise SystemExit(f"{MANIFEST} lists no files")
    return paths


def fetch(dataset, revision):
    for path in manifest_paths():
        quoted = urllib.parse.quote(path.as_posix())
        url = (
            f"https://huggingface.co/datasets/{dataset}"
            f"/resolve/{revision}/{quoted}"
        )
        dest = ROOT / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(path.as_posix(), flush=True)
        urllib.request.urlretrieve(url, dest)


def regenerate_water():
    script = ROOT / "paper/jctc/scripts/engrad_session.sh"
    subprocess.run(["/bin/sh", str(script)], cwd=ROOT, check=True)


def main(argv):
    if argv == ["--water"]:
        regenerate_water()
        return
    if argv:
        raise SystemExit(
            "usage: fetch_bench_data.py [--water]\n"
            "HF_DATASET and HF_REVISION can override the pinned dataset."
        )
    fetch(HF_DATASET, HF_REVISION)


if __name__ == "__main__":
    main(sys.argv[1:])
