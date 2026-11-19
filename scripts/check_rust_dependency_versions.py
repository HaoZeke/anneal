#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""Reject duplicate versions of shared Rust crates in every crate graph."""

import json
from pathlib import Path
import subprocess
import sys


WATCHED = {"linkcell", "minimage", "readcon-core", "rgmin", "rgpot-core"}


def manifests(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*Cargo.toml"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / name.decode() for name in result.stdout.split(b"\0") if name]


def versions_for(manifest: Path) -> dict[str, set[str]]:
    result = subprocess.run(
        [
            "cargo",
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--manifest-path",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    packages = {package["id"]: package for package in metadata["packages"]}
    resolved = {node["id"] for node in metadata["resolve"]["nodes"]}
    versions = {name: set() for name in WATCHED}
    for package_id in resolved:
        package = packages[package_id]
        if package["name"] in WATCHED:
            versions[package["name"]].add(package["version"])
    return versions


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    failed = False
    crate_manifests = manifests(root)
    for manifest in crate_manifests:
        relative = manifest.relative_to(root)
        for name, versions in sorted(versions_for(manifest).items()):
            if len(versions) > 1:
                print(
                    f"{relative}: {name} resolves to {', '.join(sorted(versions))}",
                    file=sys.stderr,
                )
                failed = True
    if failed:
        return 1
    print(
        f"Checked {len(crate_manifests)} Rust crate manifests for duplicate "
        "shared dependencies"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
