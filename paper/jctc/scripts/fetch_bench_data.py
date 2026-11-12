#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fetch or regenerate benchmark data that git does not store.

Raw logs, trajectories, frames, and the archives headed for Zenodo
are listed in bench_data_manifest.txt. They will live in a Hugging
Face dataset. Set HF_DATASET to that dataset id and run this script.
Each file is downloaded to the same relative path.

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
            "HF_DATASET must name the dataset for a download."
        )
    dataset = os.environ.get("HF_DATASET", "").strip()
    if not dataset:
        raise SystemExit(
            "HF_DATASET is unset. The Hugging Face dataset is not "
            "published yet.\n"
            "Regenerate the water hexamer logs with "
            "paper/jctc/scripts/fetch_bench_data.py --water."
        )
    revision = os.environ.get("HF_REVISION", "main").strip() or "main"
    fetch(dataset, revision)


if __name__ == "__main__":
    main(sys.argv[1:])
