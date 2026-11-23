# Copyright 2023--present rgpot developers
"""A missing Python include directory must not fail configure.

engines/ probes sys.prefix/include for the optional GROMACS unit header.
include_directories() rejects a path that does not exist. A uv isolated
sdist build has no include directory under its prefix
(…/uv/builds-v0/…/include), while a pip venv usually does. The probe has
to skip a missing directory. GROMACS that was requested and is absent
must still stop configure, with no silent fallback.
Usage: test_gromacs_missing_include.py <root> <meson>
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import venv
except ImportError:
    venv = None  # type: ignore[assignment]

OPTIONS = """
        'with_tests=false',
        'with_eigen=false',
        'with_rpc=false',
        'with_fortran_pots=disabled',
        'with_fast_nlist=disabled',
        'with_metatomic=false',
        'with_highway=disabled',
"""


def parent_meson(extra: str = "") -> str:
    return f"""project('rgpot_parent', 'c', 'cpp')
subproject(
    'rgpot',
    default_options: [{OPTIONS}{extra}],
)
"""


def _meson_argv(argv: list[str]) -> list[str]:
    if len(argv) == 1 and Path(argv[0]).name.lower().startswith("python"):
        return [argv[0], "-m", "mesonbuild.mesonmain"]
    return argv


def _drop_include(prefix: Path) -> None:
    include = prefix / "include"
    if include.is_symlink() or include.is_file():
        include.unlink()
    elif include.is_dir():
        shutil.rmtree(include)


def _path_without(names: set[str], path: str, shadow_root: Path) -> str:
    """Return ``path`` with the executables in ``names`` hidden.

    A conda or pixi prefix keeps ``gmx`` in the same ``bin`` directory as the
    compilers, so a directory that holds ``gmx`` cannot simply be dropped. Such
    a directory is replaced by a shadow directory under ``shadow_root`` that
    links every entry except the hidden ones. Directories without any of the
    names stay as they are.
    """
    kept = []
    for index, entry in enumerate(path.split(os.pathsep)):
        if not entry:
            continue
        directory = Path(entry)
        if not any((directory / name).exists() for name in names):
            kept.append(entry)
            continue
        shadow = shadow_root / f"path-{index}"
        shadow.mkdir(parents=True)
        for child in directory.iterdir():
            if child.name not in names:
                (shadow / child.name).symlink_to(child)
        kept.append(str(shadow))
    return os.pathsep.join(kept)


def _setup(meson: list[str], cwd: Path, build: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*meson, "setup", str(build)],
        cwd=cwd,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    if venv is None:
        print(
            "gromacs_missing_include requires the Python standard library venv module "
            "(on Debian/Ubuntu, install python3-venv)",
            file=sys.stderr,
        )
        return 2
    root = Path(sys.argv[1]).resolve()
    meson = _meson_argv(sys.argv[2:])
    with tempfile.TemporaryDirectory(prefix="rgpot-gromacs-miss-") as work:
        parent = Path(work) / "src"
        prefix_root = Path(work) / "py"
        parent.mkdir()
        try:
            venv.create(prefix_root, with_pip=False)
        except (OSError, subprocess.SubprocessError) as exc:
            print(
                f"gromacs_missing_include could not create a Python venv ({exc}); "
                "ensure python3-venv is installed",
                file=sys.stderr,
            )
            return 1
        py_prefix = prefix_root
        _drop_include(py_prefix)
        include = py_prefix / "include"
        if include.exists():
            print(f"failed to remove {include}", file=sys.stderr)
            return 1
        py = py_prefix / "bin" / "python"
        reported = subprocess.check_output(
            [str(py), "-c", "import pathlib, sys; print(pathlib.Path(sys.prefix) / 'include')"],
            text=True,
        ).strip()
        if Path(reported).exists():
            print(f"probe include dir still exists: {reported}", file=sys.stderr)
            return 1
        if reported.startswith(str(parent)):
            print(f"venv prefix {reported} is inside the source tree", file=sys.stderr)
            return 1
        (parent / "subprojects").mkdir()
        (parent / "subprojects" / "rgpot").symlink_to(root, target_is_directory=True)
        (parent / "meson.build").write_text(parent_meson())
        env = os.environ.copy()
        env["PATH"] = str(py_prefix / "bin") + os.pathsep + env.get("PATH", "")
        setup = _setup(meson, parent, parent / "build", env)
        if setup.returncode != 0:
            print(setup.stdout, file=sys.stderr)
            if "does not exist" in setup.stdout and "include" in setup.stdout.lower():
                print(
                    "configure rejected a missing Python include directory",
                    file=sys.stderr,
                )
            elif "Could not detect Ninja" in setup.stdout:
                print(
                    "gromacs_missing_include requires ninja on PATH or in NINJA",
                    file=sys.stderr,
                )
            elif "Unknown compiler(s)" in setup.stdout:
                print(
                    "gromacs_missing_include requires C and C++ compilers on PATH or in CC/CXX",
                    file=sys.stderr,
                )
            else:
                print(
                    "parent project did not configure with a missing include directory",
                    file=sys.stderr,
                )
            return 1
        if "Include dir" in setup.stdout and "does not exist" in setup.stdout:
            print(setup.stdout, file=sys.stderr)
            print(
                "configure rejected a missing Python include directory",
                file=sys.stderr,
            )
            return 1

        (parent / "meson.build").write_text(
            parent_meson("        'with_gromacs=enabled',\n")
        )
        hidden = os.environ.copy()
        hidden["PATH"] = _path_without(
            {"gmx", "gmx_d", "gmx.exe"},
            str(py_prefix / "bin") + os.pathsep + os.environ.get("PATH", ""),
            Path(work) / "hidden-path",
        )
        leaked = shutil.which("gmx", path=hidden["PATH"])
        if leaked is not None:
            print(f"gmx is still on the hidden PATH: {leaked}", file=sys.stderr)
            return 1
        requested = _setup(meson, parent, parent / "build-enabled", hidden)
        if requested.returncode == 0:
            print(requested.stdout, file=sys.stderr)
            print(
                "with_gromacs=enabled configured even though gmx is not on PATH",
                file=sys.stderr,
            )
            return 1
        output = requested.stdout
        clear = (
            "with_gromacs is enabled" in output
            or "Program 'gmx' not found" in output
            or 'Program "gmx" not found' in output
        )
        if not clear:
            print(output, file=sys.stderr)
            print(
                "with_gromacs=enabled failed without naming the missing gmx program",
                file=sys.stderr,
            )
            return 1
        if "does not exist" in output and "include" in output.lower():
            print(output, file=sys.stderr)
            print(
                "with_gromacs=enabled failed on the include directory instead of the missing program",
                file=sys.stderr,
            )
            return 1
    print("missing Python include directory does not fail configure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
