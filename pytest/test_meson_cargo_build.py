# Copyright (c) anneal developers
"""Opt-in reproducible archives from scripts/meson_cargo_build.py.

The Linux tests build a tiny staticlib twice and also drive ``main`` through
an ``ar`` shim on ``PATH``. CI collects this file because it lives under
``pytest/``.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "meson_cargo_build.py"
_SPEC = importlib.util.spec_from_file_location("meson_cargo_build", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MODULE)

LIB_NAME = "librepro_fixture.a"
_STAMP_MTIME = b"1700000000"
_STAMP_UID = b"1234"
_STAMP_GID = b"4321"
_SPACED_FLAG = "--remap-path-prefix=/tmp/not a real prefix=/unused"


def _headers(path: Path) -> list[dict[str, bytes | int]]:
    data = path.read_bytes()
    assert data.startswith(b"!<arch>\n"), data[:16]
    rows: list[dict[str, bytes | int]] = []
    index = 8
    while index < len(data):
        header = data[index : index + 60]
        assert len(header) == 60 and header[58:60] == b"`\n", header
        size = int(header[48:58])
        rows.append(
            {
                "name": header[:16],
                "mtime": header[16:28],
                "uid": header[28:34],
                "gid": header[34:40],
                "mode": header[40:48],
                "size": size,
            }
        )
        index += 60 + size
        if size % 2 == 1:
            assert data[index : index + 1] == b"\n"
            index += 1
    assert rows, path
    return rows


def _assert_zero_owners(path: Path) -> None:
    for row in _headers(path):
        assert int(row["mtime"]) == 0, row
        assert int(row["uid"]) == 0, row
        assert int(row["gid"]) == 0, row


def _craft_archive(path: Path) -> None:
    """One odd-sized member with a non-zero mtime and uid."""
    payload = b"xyz"
    name = b"member.txt/".ljust(16)
    header = (
        name
        + _STAMP_MTIME.ljust(12)
        + _STAMP_UID.ljust(6)
        + _STAMP_GID.ljust(6)
        + b"100644".ljust(8)
        + f"{len(payload):<10d}".encode("ascii")
        + b"`\n"
    )
    assert len(header) == 60
    path.write_bytes(b"!<arch>\n" + header + payload + b"\n")


def _write_shim(path: Path, *, reject_d: bool) -> Path:
    path.write_text(
        f"""#!/usr/bin/python3
import os
import subprocess
import sys

REAL = "/usr/bin/ar"
LOG = os.environ.get("ANNEAL_TEST_AR_LOG")
REJECT = {reject_d!r}
argv = sys.argv[1:]
if LOG:
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(repr(sys.argv) + "\\n")
op = argv[0] if argv else ""
if REJECT and op and not op.startswith("-") and "D" in op:
    sys.stderr.write("ar: deterministic mode is not supported\\n")
    sys.exit(1)
result = subprocess.run([REAL, *argv], check=False)
if result.returncode != 0:
    sys.exit(result.returncode)
writes = bool(op) and not op.startswith("-") and any(letter in op for letter in "rqs")
if writes and len(argv) > 1 and os.path.isfile(argv[1]):
    archive = argv[1]
    data = bytearray(open(archive, "rb").read())
    if data.startswith(b"!<arch>\\n"):
        index = 8
        total = len(data)
        while index + 60 <= total and data[index + 58 : index + 60] == b"`\\n":
            size = int(data[index + 48 : index + 58])
            data[index + 16 : index + 28] = {_STAMP_MTIME!r}.ljust(12)
            data[index + 28 : index + 34] = {_STAMP_UID!r}.ljust(6)
            data[index + 34 : index + 40] = {_STAMP_GID!r}.ljust(6)
            index += 60 + size
            if size % 2 == 1:
                index += 1
        open(archive, "wb").write(data)
sys.exit(0)
"""
    )
    path.chmod(0o755)
    return path


def _require_linux_toolchain() -> str:
    if sys.platform != "linux":
        pytest.skip("archive comparison runs on Linux")
    for tool in ("/usr/bin/ar", "/usr/bin/ld", "/usr/bin/nm", "/usr/bin/objcopy"):
        if not Path(tool).is_file():
            pytest.skip(f"needs {tool}")
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("needs cargo")
    return cargo


def _prepare_env(monkeypatch: pytest.MonkeyPatch, **extra: str) -> None:
    monkeypatch.setenv("RUSTUP_TOOLCHAIN", "stable")
    monkeypatch.setenv("CARGO_NET_OFFLINE", "true")
    monkeypatch.setenv("CARGO_INCREMENTAL", "0")
    monkeypatch.delenv("AR", raising=False)
    monkeypatch.delenv("RUSTFLAGS", raising=False)
    monkeypatch.delenv("CARGO_ENCODED_RUSTFLAGS", raising=False)
    monkeypatch.delenv("ANNEAL_TEST_AR_LOG", raising=False)
    monkeypatch.delenv("ANNEAL_TEST_AR_REJECT_D", raising=False)
    for key, value in extra.items():
        monkeypatch.setenv(key, value)


def _write_marker(cargo_home: Path) -> None:
    dep = cargo_home / "marker_dep"
    (dep / "src").mkdir(parents=True)
    (dep / "Cargo.toml").write_text(
        "[package]\nname = \"marker_dep\"\nversion = \"0.0.0\"\nedition = \"2021\"\n",
        encoding="utf-8",
    )
    (dep / "src" / "lib.rs").write_text(
        "pub fn marker_from_cargo_home() -> &'static str {\n    file!()\n}\n",
        encoding="utf-8",
    )


def _write_crate(directory: Path, cargo_home: Path) -> None:
    (directory / "src").mkdir(parents=True)
    dep = (cargo_home / "marker_dep").resolve()
    (directory / "Cargo.toml").write_text(
        "\n".join(
            [
                "[package]",
                "name = \"repro_fixture\"",
                "version = \"0.0.0\"",
                "edition = \"2021\"",
                "",
                "[lib]",
                "crate-type = [\"staticlib\"]",
                "",
                "[dependencies]",
                f"marker_dep = {{ path = \"{dep}\" }}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (directory / "src" / "lib.rs").write_text(
        "\n".join(
            [
                "#[no_mangle]",
                "pub extern \"C\" fn repro_fixture_stamp() -> *const u8 {",
                "    file!().as_ptr()",
                "}",
                "",
                "#[no_mangle]",
                "pub extern \"C\" fn repro_marker() -> *const u8 {",
                "    marker_dep::marker_from_cargo_home().as_ptr()",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _lock(directory: Path, cargo: str) -> None:
    subprocess.check_call(
        [cargo, "generate-lockfile", "--manifest-path", str(directory / "Cargo.toml"), "--offline"],
        cwd=directory,
        env=os.environ.copy(),
    )


def _build(
    monkeypatch: pytest.MonkeyPatch,
    cargo: str,
    root: Path,
    target: Path,
    out: Path,
    *,
    opt_in: bool,
) -> None:
    if opt_in:
        monkeypatch.setenv("ANNEAL_REPRODUCIBLE_ARCHIVE", "1")
    else:
        monkeypatch.delenv("ANNEAL_REPRODUCIBLE_ARCHIVE", raising=False)
    code = MODULE.main(
        [
            "meson_cargo_build.py",
            cargo,
            str(root),
            str(target),
            LIB_NAME,
            str(out),
            "",
            "/usr/bin/objcopy",
            "/usr/bin/nm",
        ]
    )
    assert code == 0
    assert out.is_file()


def test_reproducible_env_is_opt_in() -> None:
    root = Path("/work/source")
    target = Path("/work/build/target")
    disabled = MODULE.reproducible_env({"RUSTFLAGS": "-C opt-level=1"}, root, target)
    assert disabled == {"RUSTFLAGS": "-C opt-level=1"}
    enabled = MODULE.reproducible_env({}, root, target)
    assert enabled == {}


def test_encoded_rustflags_with_spaces_are_merged_not_dropped() -> None:
    root = Path("/work/source")
    target = Path("/work/source/target/deep")
    cargo_home = "/opt/cargo home"
    env = MODULE.reproducible_env(
        {
            "ANNEAL_REPRODUCIBLE_ARCHIVE": "1",
            # Cargo ignores RUSTFLAGS once the encoded form is set. The
            # space-split copy must not replace, or be merged ahead of, the
            # encoded flag that actually contains a space.
            "RUSTFLAGS": "-Cshould-not-be-read",
            "CARGO_ENCODED_RUSTFLAGS": _SPACED_FLAG + "\x1f" + "-Copt-level=z",
            "CARGO_HOME": cargo_home,
        },
        root,
        target,
    )
    flags = env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
    assert _SPACED_FLAG in flags
    assert "-Copt-level=z" in flags
    assert "-Cshould-not-be-read" not in "".join(flags)
    assert "RUSTFLAGS" not in env
    assert env["CARGO_INCREMENTAL"] == "0"
    assert "-Ccodegen-units=1" in flags
    remaps = [
        flags[index + 1]
        for index, flag in enumerate(flags[:-1])
        if flag == "--remap-path-prefix"
    ]
    source = f"{root.resolve()}=/source"
    build = f"{target.resolve()}=/build"
    home = f"{Path(cargo_home).resolve()}=/cargo"
    assert source in remaps
    assert build in remaps
    assert home in remaps
    # The nested target prefix is longer than the source prefix and has to win.
    assert remaps.index(build) < remaps.index(source)

    spaced_only = MODULE.reproducible_env(
        {
            "ANNEAL_REPRODUCIBLE_ARCHIVE": "1",
            "RUSTFLAGS": "-C debuginfo=2  -C link-dead-code",
        },
        root,
        Path("/work/build"),
    )
    parsed = spaced_only["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
    assert parsed[:4] == ["-C", "debuginfo=2", "-C", "link-dead-code"]
    assert "" not in parsed


def test_normalize_archive_zeros_headers_and_is_idempotent(tmp_path: Path) -> None:
    archive = tmp_path / "lib.a"
    _craft_archive(archive)
    assert MODULE.normalize_archive(archive) is True
    _assert_zero_owners(archive)
    regular = _headers(archive)[0]
    assert regular["mode"] == b"644".ljust(8)
    once = archive.read_bytes()
    assert MODULE.normalize_archive(archive) is True
    assert archive.read_bytes() == once

    other = tmp_path / "note.txt"
    other.write_bytes(b"not an archive")
    assert MODULE.normalize_archive(other) is False
    assert other.read_bytes() == b"not an archive"


def test_resolve_archiver_prefers_ar_env_then_path_then_llvm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    ar = bindir / "ar"
    llvm = bindir / "llvm-ar"
    ar.write_text("#!/bin/sh\n", encoding="utf-8")
    llvm.write_text("#!/bin/sh\n", encoding="utf-8")
    ar.chmod(0o755)
    llvm.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.delenv("AR", raising=False)
    assert MODULE.resolve_archiver(dict(os.environ)) == str(ar)
    ar.unlink()
    assert MODULE.resolve_archiver(dict(os.environ)) == str(llvm)
    monkeypatch.setenv("AR", "/toolchain/bin/ar")
    assert MODULE.resolve_archiver(dict(os.environ)) == "/toolchain/bin/ar"
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.delenv("AR", raising=False)
    assert MODULE.resolve_archiver(dict(os.environ)) is None


def test_early_return_does_not_call_ar_and_keeps_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """macOS and Windows pass an empty objcopy. ``ar sD`` must not run."""
    payload = tmp_path / "payload.a"
    _craft_archive(payload)
    src = tmp_path / "src"
    src.mkdir()
    lib_name = "libfixture.a"
    cargo = tmp_path / "cargo"
    cargo.write_text(
        "\n".join(
            [
                "#!/usr/bin/python3",
                "import shutil, sys",
                "from pathlib import Path",
                "args = sys.argv[1:]",
                "target = Path(args[args.index('--target-dir') + 1])",
                f"dest = target / 'release' / '{lib_name}'",
                "dest.parent.mkdir(parents=True, exist_ok=True)",
                f"shutil.copyfile({str(payload)!r}, dest)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cargo.chmod(0o755)
    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.delenv("AR", raising=False)

    def invoke(out: Path, opt_in: bool) -> int:
        if opt_in:
            monkeypatch.setenv("ANNEAL_REPRODUCIBLE_ARCHIVE", "1")
        else:
            monkeypatch.delenv("ANNEAL_REPRODUCIBLE_ARCHIVE", raising=False)
        return MODULE.main(
            [
                "meson_cargo_build.py",
                str(cargo),
                str(src),
                str(tmp_path / out.name / "target"),
                lib_name,
                str(out),
                "",
                "",
                "",
            ]
        )

    default_out = tmp_path / "default" / "lib.a"
    default_out.parent.mkdir()
    assert invoke(default_out, False) == 0
    default_headers = _headers(default_out)
    assert int(default_headers[0]["mtime"]) == int(_STAMP_MTIME)
    assert int(default_headers[0]["uid"]) == int(_STAMP_UID)
    assert int(default_headers[0]["gid"]) == int(_STAMP_GID)

    opt_out = tmp_path / "opt" / "lib.a"
    opt_out.parent.mkdir()
    assert invoke(opt_out, True) == 0
    _assert_zero_owners(opt_out)


def test_stamping_shim_is_cleared_only_on_the_shipped_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ar`` on PATH writes a real mtime and uid. Opt-in zeros the archive that ships.

    ``ar sD`` before ``ld -r`` does not: the shim stamps the final ``ar crs``
    write, and that is the archive a consumer links.
    """
    if not Path("/usr/bin/ar").is_file() or shutil.which("as") is None:
        pytest.skip("needs /usr/bin/ar and as")
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    log = tmp_path / "ar.log"
    _write_shim(shim_dir / "ar", reject_d=False)
    monkeypatch.setenv("PATH", str(shim_dir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("ANNEAL_TEST_AR_LOG", str(log))
    monkeypatch.delenv("AR", raising=False)
    obj = tmp_path / "foo.o"
    subprocess.check_call(
        "echo '.globl foo; foo: ret' | as -o foo.o",
        shell=True,
        cwd=tmp_path,
    )
    out = tmp_path / "lib.a"
    MODULE._AR_SUPPORTS_D.clear()
    MODULE.pack_deterministic_archive(str(shim_dir / "ar"), str(out), [str(obj)])
    _assert_zero_owners(out)
    subprocess.check_call(["/usr/bin/ar", "t", str(out)], stdout=subprocess.DEVNULL)
    invocations = [ast.literal_eval(line) for line in log.read_text(encoding="utf-8").splitlines()]
    final = [args for args in invocations if str(out) in args]
    assert final, invocations
    assert any("D" in args[1] for args in final), final

    log.write_text("", encoding="utf-8")
    reject_dir = tmp_path / "reject"
    reject_dir.mkdir()
    reject = _write_shim(reject_dir / "ar", reject_d=True)
    reject_out = tmp_path / "reject.a"
    MODULE._AR_SUPPORTS_D.clear()
    MODULE.pack_deterministic_archive(str(reject), str(reject_out), [str(obj)])
    _assert_zero_owners(reject_out)
    assert MODULE.archiver_supports_deterministic(str(reject)) is False
    reject_log = tmp_path / "reject.log"
    # The pack above did not set a log. Re-run against a logged shim path.
    logged = tmp_path / "logged-reject"
    logged.mkdir()
    logged_shim = _write_shim(logged / "ar", reject_d=True)
    monkeypatch.setenv("ANNEAL_TEST_AR_LOG", str(reject_log))
    logged_out = tmp_path / "logged-reject.a"
    MODULE._AR_SUPPORTS_D.clear()
    MODULE.pack_deterministic_archive(str(logged_shim), str(logged_out), [str(obj)])
    _assert_zero_owners(logged_out)
    logged_calls = [
        ast.literal_eval(line) for line in reject_log.read_text(encoding="utf-8").splitlines()
    ]
    shipped = [args for args in logged_calls if str(logged_out) in args]
    assert shipped, logged_calls
    assert all("D" not in args[1] for args in shipped), shipped


def test_linux_shim_on_path_and_merged_rustflags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cargo = _require_linux_toolchain()
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    _write_shim(shim_dir / "ar", reject_d=False)
    flag_log = tmp_path / "flags"
    wrapper = tmp_path / "cargo"
    wrapper.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                f"printf '%s' \"${{CARGO_ENCODED_RUSTFLAGS-}}\" > {flag_log}",
                f"printf '%s' \"${{RUSTFLAGS-<unset>}}\" > {flag_log}.rustflags",
                f"exec {cargo} \"$@\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    cargo_home = tmp_path / "cargo-home"
    _write_marker(cargo_home)
    root = tmp_path / "src"
    _write_crate(root, cargo_home)
    _prepare_env(
        monkeypatch,
        PATH=str(shim_dir) + os.pathsep + os.environ.get("PATH", ""),
        CARGO_HOME=str(cargo_home),
        ANNEAL_TEST_AR_LOG=str(tmp_path / "ar.log"),
    )
    _lock(root, cargo)

    default_out = tmp_path / "default" / LIB_NAME
    default_out.parent.mkdir()
    _build(
        monkeypatch,
        str(wrapper),
        root,
        tmp_path / "target-default",
        default_out,
        opt_in=False,
    )
    default_headers = _headers(default_out)
    assert any(int(row["mtime"]) == int(_STAMP_MTIME) for row in default_headers)
    assert any(int(row["uid"]) == int(_STAMP_UID) for row in default_headers)
    assert any(int(row["gid"]) == int(_STAMP_GID) for row in default_headers)
    default_calls = [
        ast.literal_eval(line)
        for line in (tmp_path / "ar.log").read_text(encoding="utf-8").splitlines()
    ]
    default_final = [args for args in default_calls if str(default_out) in args]
    assert default_final, default_calls
    assert all(args[1] == "crs" for args in default_final), default_final

    # debuginfo is what embeds the absolute source path. Release does not.
    monkeypatch.setenv(
        "CARGO_ENCODED_RUSTFLAGS", _SPACED_FLAG + "\x1f" + "-Cdebuginfo=2"
    )
    (tmp_path / "ar.log").write_text("", encoding="utf-8")
    opt_out = tmp_path / "opt" / LIB_NAME
    opt_out.parent.mkdir()
    _build(
        monkeypatch,
        str(wrapper),
        root,
        tmp_path / "target-opt",
        opt_out,
        opt_in=True,
    )
    _assert_zero_owners(opt_out)
    subprocess.check_call(["/usr/bin/nm", "-g", "--defined-only", str(opt_out)], stdout=subprocess.DEVNULL)
    recorded = flag_log.read_text(encoding="utf-8").split("\x1f")
    assert _SPACED_FLAG in recorded
    assert f"{root.resolve()}=/source" in "".join(recorded)
    assert f"{cargo_home.resolve()}=/cargo" in "".join(recorded)
    assert flag_log.with_suffix(".rustflags").read_text(encoding="utf-8") == "<unset>"
    opt_calls = [
        ast.literal_eval(line)
        for line in (tmp_path / "ar.log").read_text(encoding="utf-8").splitlines()
    ]
    opt_final = [args for args in opt_calls if str(opt_out) in args]
    assert opt_final, opt_calls
    assert any(args[1] == "crsD" for args in opt_final), opt_final
    opt_bytes = opt_out.read_bytes()
    assert str(root).encode() not in opt_bytes
    assert str(cargo_home).encode() not in opt_bytes
    assert b"/source" in opt_bytes
    assert b"/cargo/marker_dep" in opt_bytes


def test_archiver_without_d_still_ships_a_zeroed_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cargo = _require_linux_toolchain()
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    _write_shim(shim_dir / "ar", reject_d=True)
    log = tmp_path / "ar.log"
    cargo_home = tmp_path / "cargo-home"
    _write_marker(cargo_home)
    root = tmp_path / "src"
    _write_crate(root, cargo_home)
    _prepare_env(
        monkeypatch,
        PATH=str(shim_dir) + os.pathsep + os.environ.get("PATH", ""),
        CARGO_HOME=str(cargo_home),
        ANNEAL_TEST_AR_LOG=str(log),
        ANNEAL_REPRODUCIBLE_ARCHIVE="1",
    )
    _lock(root, cargo)
    out = tmp_path / "out" / LIB_NAME
    out.parent.mkdir()
    MODULE._AR_SUPPORTS_D.clear()
    _build(monkeypatch, cargo, root, tmp_path / "target", out, opt_in=True)
    _assert_zero_owners(out)
    calls = [ast.literal_eval(line) for line in log.read_text(encoding="utf-8").splitlines()]
    final = [args for args in calls if str(out) in args]
    assert final, calls
    assert all("D" not in args[1] for args in final), final


def test_same_path_is_already_identical_and_opt_in_strips_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cargo = _require_linux_toolchain()
    cargo_home = tmp_path / "cargo-home"
    _write_marker(cargo_home)
    root = tmp_path / "src-a"
    _write_crate(root, cargo_home)
    _prepare_env(monkeypatch, CARGO_HOME=str(cargo_home))
    # debuginfo embeds the real source path. Without it a release staticlib
    # does not mention that path, so the opt-in would have nothing to strip.
    monkeypatch.setenv("RUSTFLAGS", "-Cdebuginfo=2")
    _lock(root, cargo)
    other = tmp_path / "src-b"
    shutil.copytree(root, other)

    def fresh(name: str) -> tuple[Path, Path]:
        target = tmp_path / f"target-{name}"
        out_dir = tmp_path / f"out-{name}"
        out_dir.mkdir()
        return target, out_dir / LIB_NAME

    # GNU ar 2.42 defaults to -D. Two clean builds of the same paths match
    # with the opt-in unset; the script does not rewrite that archive.
    plain_target = tmp_path / "target-plain"
    plain_out = tmp_path / "out-plain" / LIB_NAME
    plain_out.parent.mkdir()
    _build(monkeypatch, cargo, root, plain_target, plain_out, opt_in=False)
    plain_a = plain_out.read_bytes()
    shutil.rmtree(plain_target)
    plain_out.unlink()
    _build(monkeypatch, cargo, root, plain_target, plain_out, opt_in=False)
    plain_b = plain_out.read_bytes()
    assert plain_a == plain_b
    assert str(root).encode() in plain_a
    # GNU ar leaves the long-name member's date blank. The opt-in rewrite
    # stores an explicit zero, so a blank date shows this build was not rewritten.
    string_table = [row for row in _headers(plain_out) if bytes(row["name"]).rstrip() == b"//"]
    assert string_table
    assert bytes(string_table[0]["mtime"]).strip() == b""

    target_c, out_c = fresh("remap-a")
    _build(monkeypatch, cargo, root, target_c, out_c, opt_in=True)
    target_d, out_d = fresh("remap-b")
    _build(monkeypatch, cargo, other, target_d, out_d, opt_in=True)
    remap_a = out_c.read_bytes()
    remap_b = out_d.read_bytes()
    assert remap_a == remap_b
    assert str(root).encode() not in remap_a
    assert str(other).encode() not in remap_a
    assert str(cargo_home).encode() not in remap_a
    assert b"/source" in remap_a
    assert b"/cargo/marker_dep" in remap_a
    _assert_zero_owners(out_c)
