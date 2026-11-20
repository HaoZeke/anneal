#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Build a Rust staticlib with cargo and copy it to the meson output.

Meson links the C++ library against a file this script writes, so ninja
runs cargo first. The archive is the library that rgpot-core builds on its
own, or ``rgpot-bundle`` when the fast pair list is linked as well: both
are Rust staticlibs that embed a copy of std, and linking the two archives
defines those symbols twice, so the bundle carries both in one.

The bundle needs the same rustc as the pair list (1.89), so
``--fast-toolchain`` reuses the toolchain choice of ``build_fastnlist.py``.
rgpot-core on its own builds with the workspace rust-version (1.88) and
takes the cargo on the command line. Both builds use the Darwin archiver
override of ``build_fastnlist.py``: rustc writes every staticlib with the
``AR`` it finds, and Apple ld rejects one written by the conda archiver.
When the output is a ``.o`` the archive is partial-linked to one
relocatable object, as for the pair list.

The second output is a header with no declarations. A meson dependency
lists it as a source, so each target that uses the library waits for this
script without ar being handed the archive as a source.

The outputs are replaced only when their bytes change. Cargo runs on every
build, and an unchanged archive must not relink everything that uses it.
"""

import argparse
import filecmp
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_fastnlist as fast

STAMP = """/* Copyright 2023--present rgpot developers */
/* Written by build_rust_core.py once the Rust library is built. */
"""


def find_built(target_dir: str, profile: str, name: str) -> str:
    direct = os.path.join(target_dir, profile, name)
    if os.path.isfile(direct):
        return direct
    if os.path.isdir(target_dir):
        for entry in sorted(os.listdir(target_dir)):
            nested = os.path.join(target_dir, entry, profile, name)
            if os.path.isfile(nested):
                return nested
    sys.exit(f"cargo did not produce {name} under {target_dir}/{profile}")


def replace_if_changed(candidate: str, out: str) -> None:
    if os.path.isfile(out) and filecmp.cmp(candidate, out, shallow=False):
        if candidate != out:
            os.remove(candidate)
        return
    os.replace(candidate, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cargo", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--profile", choices=["debug", "release"], required=True)
    ap.add_argument("--artifact", required=True, help="file name cargo writes")
    ap.add_argument("--features", default="")
    ap.add_argument("--fast-toolchain", action="store_true")
    ap.add_argument("out")
    ap.add_argument(
        "stamp", help="header meson uses to order consumers after the build"
    )
    args = ap.parse_args()

    base = fast.cargo_argv(args.cargo) if args.fast_toolchain else [args.cargo]
    cmd = base + [
        "build",
        "--locked",
        "--manifest-path",
        args.manifest,
        "--target-dir",
        args.target_dir,
    ]
    if args.profile == "release":
        cmd.append("--release")
    if args.features:
        cmd += ["--features", args.features]
    env = fast.cargo_env()
    proc = subprocess.run(cmd, check=False, env=env)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

    built = find_built(args.target_dir, args.profile, args.artifact)
    staged = args.out + ".new"
    if args.out.endswith(".o"):
        fast.relocatable_object(built, staged)
    else:
        shutil.copyfile(built, staged)
    replace_if_changed(staged, args.out)

    with open(args.stamp + ".new", "w") as handle:
        handle.write(STAMP)
    replace_if_changed(args.stamp + ".new", args.stamp)


if __name__ == "__main__":
    main()
