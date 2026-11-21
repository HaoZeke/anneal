# Copyright (c) anneal developers
"""Build the anneal-core static archive with its C ABI and localize foreign C symbols.

``ANNEAL_REPRODUCIBLE_ARCHIVE=1`` remaps the source, target, and ``CARGO_HOME``
prefixes and rewrites the archive that actually ships so every member has a
zero mtime, uid, and gid. The default build is unchanged.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

_UNIT_SEP = "\x1f"
_AR_MAGIC = b"!<arch>\n"
_AR_HEADER = 60
_AR_END = b"`\n"
_AR_SUPPORTS_D: dict[str, bool] = {}


def rustflags_from_env(environment: Mapping[str, str]) -> list[str]:
    """Split flags the way Cargo does.

    ``CARGO_ENCODED_RUSTFLAGS`` wins when it is set, including when it is
    empty, and its unit separator keeps a flag that contains a space intact.
    ``RUSTFLAGS`` is split on spaces only as Cargo's fallback.
    """
    if "CARGO_ENCODED_RUSTFLAGS" in environment:
        encoded = environment["CARGO_ENCODED_RUSTFLAGS"]
        if encoded == "":
            return []
        return encoded.split(_UNIT_SEP)
    raw = environment.get("RUSTFLAGS", "")
    if raw == "":
        return []
    return [flag for flag in raw.split(" ") if flag]


def _cargo_home(environment: Mapping[str, str]) -> Path:
    raw = environment.get("CARGO_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cargo"


def reproducible_env(
    environment: Mapping[str, str], root: Path, target_dir: Path
) -> dict[str, str]:
    """Opt in to deterministic Cargo inputs for a reproducible archive.

    Caller flags are merged in front of the opt-in flags. Setting
    ``CARGO_ENCODED_RUSTFLAGS`` makes Cargo ignore ``RUSTFLAGS``, so a
    caller's encoded flags have to be copied into the value this writes
    or they would replace the remap flags.
    """
    env = dict(environment)
    if env.get("ANNEAL_REPRODUCIBLE_ARCHIVE") != "1":
        return env
    env["CARGO_INCREMENTAL"] = "0"
    flags = rustflags_from_env(env)
    prefixes = [
        (str(Path(root).resolve()), "/source"),
        (str(Path(target_dir).resolve()), "/build"),
        (str(_cargo_home(env).resolve()), "/cargo"),
    ]
    # rustc applies the last matching --remap-path-prefix. Shortest first
    # puts a target directory inside the source tree after the source prefix.
    prefixes.sort(key=lambda item: len(item[0]))
    for src, dest in prefixes:
        flags.append("--remap-path-prefix")
        flags.append(f"{src}={dest}")
    flags.append("-Ccodegen-units=1")
    env["CARGO_ENCODED_RUSTFLAGS"] = _UNIT_SEP.join(flags)
    env.pop("RUSTFLAGS", None)
    return env


def resolve_archiver(environment: Mapping[str, str]) -> str | None:
    """Archiver for the opt-in pack.

    ``AR`` is cargo's toolchain override. Otherwise use ``ar`` on ``PATH``
    (GNU ar on Linux, Apple ar on macOS). ``llvm-ar`` is the fallback on a
    Windows Rust toolchain, which has no ``ar``.
    """
    chosen = environment.get("AR")
    if chosen:
        return chosen
    found = shutil.which("ar")
    if found:
        return found
    return shutil.which("llvm-ar")


def archiver_supports_deterministic(ar: str) -> bool:
    """True when ``ar rcD`` can create an archive.

    Apple ar rejects ``D``. The probe is cached per path. A failure here
    selects the header rewrite, it does not fail the build.
    """
    cached = _AR_SUPPORTS_D.get(ar)
    if cached is not None:
        return cached
    supported = False
    try:
        with tempfile.TemporaryDirectory(prefix="anneal-ar-") as tmp:
            member = Path(tmp) / "member"
            member.write_bytes(b"member\n")
            archive = Path(tmp) / "probe.a"
            result = subprocess.run(
                [ar, "rcD", str(archive), str(member)],
                check=False,
                capture_output=True,
            )
            supported = result.returncode == 0 and archive.is_file()
    except OSError:
        supported = False
    _AR_SUPPORTS_D[ar] = supported
    return supported


def _member_mode(name: bytes) -> bytes:
    # GNU ar -D stores the symbol table and the long-name table with mode 0
    # and every other member as 0644. Match that so the fallback is stable.
    label = name.rstrip(b" ")
    if label in (b"/", b"//"):
        return b"0".ljust(8)
    return b"644".ljust(8)


def _normalize_ar_bytes(data: bytes) -> bytes:
    if not data.startswith(_AR_MAGIC):
        raise RuntimeError("not an ar archive")
    out = bytearray(_AR_MAGIC)
    index = len(_AR_MAGIC)
    total = len(data)
    while index < total:
        if total - index < _AR_HEADER:
            raise RuntimeError("truncated ar header")
        header = data[index : index + _AR_HEADER]
        if header[58:60] != _AR_END:
            raise RuntimeError("corrupt ar header")
        try:
            size = int(header[48:58])
        except ValueError as exc:
            raise RuntimeError("non-numeric ar member size") from exc
        if size < 0:
            raise RuntimeError("negative ar member size")
        payload_end = index + _AR_HEADER + size
        if payload_end > total:
            raise RuntimeError("ar member overruns the archive")
        name = header[:16]
        rewritten = (
            name
            + b"0".ljust(12)
            + b"0".ljust(6)
            + b"0".ljust(6)
            + _member_mode(name)
            + header[48:58]
            + _AR_END
        )
        if len(rewritten) != _AR_HEADER:
            raise RuntimeError("internal ar header width")
        out += rewritten
        out += data[index + _AR_HEADER : payload_end]
        index = payload_end
        if size % 2 == 1:
            if index >= total or data[index : index + 1] != b"\n":
                raise RuntimeError("ar member is missing its padding byte")
            out += b"\n"
            index += 1
    return bytes(out)


def normalize_archive(path: Path) -> bool:
    """Zero mtime, uid, and gid. Leave a non-archive untouched.

    Returns False when ``path`` does not start with the ar magic, so a
    platform whose staticlib is not an ar archive still builds. A corrupt
    archive raises instead of being rewritten into something that looks done.
    """
    data = path.read_bytes()
    if not data.startswith(_AR_MAGIC):
        return False
    path.write_bytes(_normalize_ar_bytes(data))
    return True


def pack_deterministic_archive(ar: str, out: str, members: list[str]) -> None:
    """Create ``out`` and force zero mtime, uid, and gid on every member.

    This is the archive that ships, after ``ld -r``. ``D`` is passed only
    when this archiver accepts it. The header rewrite still runs: an
    archiver can ignore ``D`` and write a real timestamp, and an earlier
    ``ar sD`` would be thrown away by this write.
    """
    if archiver_supports_deterministic(ar):
        command = [ar, "crsD", out, *members]
    else:
        command = [ar, "crs", out, *members]
    subprocess.check_call(command)
    if not normalize_archive(Path(out)):
        raise RuntimeError(f"{out} is not an ar archive")


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
    deterministic = os.environ.get("ANNEAL_REPRODUCIBLE_ARCHIVE") == "1"
    cmd = [
        cargo, "rustc", "--lib", "--release",
        "--manifest-path", str(root / "Cargo.toml"),
        "--target-dir", target_dir,
        "--no-default-features",
    ]
    # meson lists Cargo.lock as an input: the build is always locked.
    cmd.append("--locked")
    if features.strip():
        cmd.extend(["--features", features.strip()])
    cmd.extend(["--", "--crate-type=staticlib"])
    if deterministic:
        env = reproducible_env(os.environ, root, Path(target_dir))
        subprocess.check_call(cmd, cwd=root, env=env)
    else:
        subprocess.check_call(cmd, cwd=root)
    built = Path(target_dir) / "release" / lib_name
    shutil.copy2(built, out)
    if not objcopy or not nm:
        # macOS and Windows take this path: meson passes an empty objcopy
        # and nm. Do not call `ar sD` here. Apple ar has no D flag and a
        # Windows toolchain has no `ar`. Rewriting headers needs neither.
        if deterministic:
            normalize_archive(Path(out))
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
    # The default pack stays on PATH `ar`. The opt-in may use $AR or llvm-ar
    # when the host has no `ar`.
    if deterministic:
        ar = resolve_archiver(os.environ)
    else:
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
        and line.split()[-1].startswith(("eindir_", "rgmin_", "rgsaddle_", "rkr_", "rkrdb_"))
    })
    if foreign:
        subprocess.check_call(
            [objcopy] + [f"--localize-symbol={s}" for s in foreign] + [combined]
        )
    Path(out).unlink()
    if deterministic:
        pack_deterministic_archive(ar, out, [combined])
    else:
        subprocess.check_call([ar, "crs", out, combined])
    Path(combined).unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
