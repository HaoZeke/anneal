#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write ``anneal-core.pc`` for a native library archive.

The release workflow stages ``include/`` and ``lib/`` and archives them.
This script adds ``lib/pkgconfig/anneal-core.pc`` to that staging tree.
Every path in it is relative to the file, so the extracted archive works
wherever it lands: ``pkg-config --cflags --libs anneal-core`` with
``PKG_CONFIG_PATH`` pointing at ``lib/pkgconfig`` is enough.

The version, the package name, the description and the library name come
from the same Cargo.toml tables cargo-c reads, so the archive and
``cargo cinstall`` describe the library alike. ``Libs.private`` is what
rustc reports the static library needs (``--print native-static-libs``),
so ``pkg-config --static`` links it.

usage: native_pkgconfig.py STAGING [--private-libs-file FILE]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anneal_version  # noqa: E402


def pc_text(version: str, name: str, description: str, library: str, private: str) -> str:
    lines = [
        "prefix=${pcfiledir}/../..",
        "libdir=${prefix}/lib",
        "includedir=${prefix}/include",
        "",
        f"Name: {name}",
        f"Description: {description}",
        f"Version: {version}",
        "Cflags: -I${includedir}",
        f"Libs: -L${{libdir}} -l{library}",
    ]
    if private:
        lines.append(f"Libs.private: {private}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("staging", type=Path)
    parser.add_argument("--private-libs-file", type=Path)
    args = parser.parse_args(argv)

    for part in ("include", "lib"):
        if not (args.staging / part).is_dir():
            raise SystemExit(f"{args.staging / part} is missing: stage the archive first")

    manifest = anneal_version.load_toml(anneal_version.repo_root() / "Cargo.toml")
    capi = manifest["package"]["metadata"]["capi"]
    pkg_config = capi["pkg_config"]
    private = ""
    if args.private_libs_file is not None:
        private = " ".join(args.private_libs_file.read_text(encoding="utf-8").split())
        if not private:
            raise SystemExit(f"{args.private_libs_file} is empty")

    text = pc_text(
        anneal_version.package_version(),
        pkg_config["name"],
        pkg_config["description"],
        capi["library"]["name"],
        private,
    )
    out = args.staging / "lib" / "pkgconfig" / f"{pkg_config['filename']}.pc"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
