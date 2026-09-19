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
    if (root / "Cargo.lock").is_file():
        cmd.append("--locked")
    if features.strip():
        cmd.extend(["--features", features.strip()])
    cmd.extend(["--", "--crate-type=staticlib"])
    subprocess.check_call(cmd, cwd=root)
    built = Path(target_dir) / "release" / lib_name
    shutil.copy2(built, out)
    if not objcopy or not nm:
        return 0
    # Every defined global symbol that is not anneal's own C ABI is made
    # local: the archive then adds nothing to the consumer's symbol table
    # that another Rust archive (rgmin carries eindir-core too) also defines.
    listing = subprocess.run([nm, "-g", "--defined-only", out], check=True,
                             capture_output=True, text=True).stdout
    foreign = sorted({
        line.split()[-1] for line in listing.splitlines()
        if line.strip() and not line.split()[-1].startswith("anneal_")
        and line.split()[-1].startswith(("eindir_", "rgmin_", "rgsaddle_"))
    })
    if foreign:
        subprocess.check_call([objcopy] + [f"--localize-symbol={s}" for s in foreign] + [out])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
