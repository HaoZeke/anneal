# Copyright (c) anneal developers
"""Build the anneal-core static archive with its C ABI and localize foreign C symbols."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 9:
        print(
            "usage: meson_cargo_build.py CARGO SRC_ROOT TARGET_DIR LIB_NAME OUT FEATURES OBJCOPY NM",
            file=sys.stderr,
        )
        return 2
    cargo, src_root, target_dir, lib_name, out, features, objcopy, nm = argv[1:]
    # Meson runs the command from the build directory; the target and
    # output paths are relative to it, so they are resolved before cargo
    # runs from the source root.
    root = Path(src_root).resolve()
    target_dir = str(Path(target_dir).resolve())
    out = str(Path(out).resolve())
    cmd = [
        cargo, "rustc", "--lib", "--release",
        "--manifest-path", str(root / "Cargo.toml"),
        "--target-dir", target_dir,
    ]
    # meson lists Cargo.lock as an input: the build is always locked.
    cmd.append("--locked")
    if features.strip():
        cmd.extend(["--features", features.strip()])
    cmd.extend(["--", "--crate-type=staticlib"])
    subprocess.check_call(cmd, cwd=root)
    built = Path(target_dir) / "release" / lib_name
    shutil.copy2(built, out)
    if not objcopy or not nm:
        return 0
    # eindir-core, rgmin and rgsaddle also export C symbols, and a consumer
    # that links this archive beside one of those would see each symbol
    # twice. The copy here is made local. A static archive is several
    # objects: one member defines eindir_core_abi_stamp and another still
    # has an undefined reference to it. Localizing the definition inside
    # the archive leaves that reference unresolved, so the consumer fails
    # to link. Resolve the members into one object first; the reference is
    # then inside that object and stays resolved after the symbol is local.
    ld = shutil.which("ld")
    ar = shutil.which("ar")
    if ld is None or ar is None:
        print("ld and ar are required to pack the C archive", file=sys.stderr)
        return 1
    combined = str(Path(out).with_suffix(".o"))
    subprocess.check_call([ld, "-r", "-o", combined, "--whole-archive", out])
    listing = subprocess.run(
        [nm, "-g", "--defined-only", combined],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    foreign = sorted({
        line.split()[-1] for line in listing.splitlines()
        if line.strip() and not line.split()[-1].startswith("anneal_")
        and line.split()[-1].startswith(("eindir_", "rgmin_", "rgsaddle_"))
    })
    if foreign:
        subprocess.check_call(
            [objcopy] + [f"--localize-symbol={s}" for s in foreign] + [combined]
        )
    Path(out).unlink()
    subprocess.check_call([ar, "crs", out, combined])
    Path(combined).unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
