#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Audit log files for hygiene and build deterministic numeric-owner tar.gz."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import re
import sys
import tarfile
from pathlib import Path

FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "Inherited" + " environment",
    "/" + "home" + "/",
    "/" + "cur" + "sor",
    "/" + "opt" + "/",
    "/" + "tmp" + "/",
    "/" + "agent" + "/",
)

IP_HOST_RE = re.compile(r"\bip-\d+(?:-\d+){1,3}\b", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_sensitive_env_values() -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for key, val in os.environ.items():
        if len(val) > 16:
            items.append((key, val))
    return items


def audit_files(files: list[tuple[str, Path]]) -> None:
    long_env = collect_sensitive_env_values()
    violations: list[str] = []
    for arcname, path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "\r" in text:
            violations.append(f"{arcname}: contains CR byte")
        for sub in FORBIDDEN_SUBSTRINGS:
            if sub in text:
                violations.append(f"{arcname}: contains forbidden pattern {sub!r}")
        if IP_HOST_RE.search(text):
            violations.append(f"{arcname}: contains ip-* hostname pattern")
        for env_key, env_val in long_env:
            if env_val in text:
                violations.append(
                    f"{arcname}: leaks >16-char env value from {env_key}"
                )
    if violations:
        for v in violations[:25]:
            print(f"AUDIT_VIOLATION: {v}", file=sys.stderr)
        raise RuntimeError(f"log hygiene audit failed with {len(violations)} hits")


import subprocess

def build_deterministic_targz(
    run_dir: Path, rel_paths: list[str], out_path: Path
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tar_cmd = [
        "tar",
        "--sort=name",
        "--mtime=UTC 2026-01-01",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "--transform=s,^,cap-turn-gate/,",
        "-cf",
        "-",
        *sorted(rel_paths),
    ]
    tar_proc = subprocess.run(
        tar_cmd, cwd=str(run_dir), check=True, stdout=subprocess.PIPE
    )
    gz_buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=gz_buf, mtime=0) as gz:
        gz.write(tar_proc.stdout)
    gz_bytes = gz_buf.getvalue()

    with tmp_path.open("wb") as handle:
        handle.write(gz_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, out_path)
    dir_fd = os.open(str(out_path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    digest = sha256_file(out_path)
    if digest != hashlib.sha256(gz_bytes).hexdigest():
        raise RuntimeError("archive sha256 readback mismatch")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and archive gate logs.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing rows.csv, summary.csv, REPORT.md, journal.jsonl, build_manifest.json, raw/.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output logs.tar.gz path.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    files: list[tuple[str, Path]] = []
    rel_paths: list[str] = []
    for top_name in (
        "rows.csv",
        "summary.csv",
        "REPORT.md",
        "journal.jsonl",
        "build_manifest.json",
    ):
        cand = run_dir / top_name
        if cand.exists():
            files.append((f"cap-turn-gate/{top_name}", cand))
            rel_paths.append(top_name)

    raw_dir = run_dir / "raw"
    if raw_dir.is_dir():
        for log_file in sorted(raw_dir.iterdir()):
            if log_file.is_file() and not log_file.name.endswith(".tmp"):
                files.append((f"cap-turn-gate/raw/{log_file.name}", log_file))
                rel_paths.append(f"raw/{log_file.name}")

    if not files:
        raise RuntimeError("no files found to archive")

    audit_files(files)
    digest = build_deterministic_targz(run_dir, rel_paths, Path(args.out))
    print(f"ARCHIVED files={len(files)} sha256={digest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
