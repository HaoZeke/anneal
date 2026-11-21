# Copyright (c) anneal developers
"""Every GMIN build and row uses one verified source pin and build key."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "gmin_panel" / "build_gmin.sh"
sys.path.insert(0, str(ROOT / "scripts"))
from gmin_pin import (  # noqa: E402
    REVISION,
    TARBALL_SHA256,
    compute_build_key,
    read_build_key,
    read_svnrev,
)
from morse_gmin.run_panel import gmin_line  # noqa: E402


def test_every_gmin_builder_loads_the_shared_pin_and_build_key() -> None:
    builds = tuple((ROOT / "scripts").glob("**/build_gmin*.sh"))
    assert len(builds) == 6
    for build in builds:
        text = build.read_text()
        assert "gmin_pin.py" in text, build
        assert (
            "build-key" in text or 'build_gmin.sh"' in text
        ), f"{build} does not compute or delegate a shared build key"
        assert (
            "verify-tarball" in text
            or "build_gmin.sh" in text
            or "build_gmin_darby.sh" in text
        ), build


def test_h2h_patch_embeds_current_cap_module() -> None:
    cap_lines = (ROOT / "bench/lj104-lattice/v2/h2h_cap.f90").read_text().splitlines()
    patch_text = (ROOT / "bench/lj104-lattice/v2/h2h-gmin.patch").read_text()
    marker = "--- a/source/h2h_cap.f90\n+++ b/source/h2h_cap.f90\n"
    assert marker in patch_text
    after = patch_text.split(marker, 1)[1].splitlines()
    end = next(
        (idx for idx, line in enumerate(after) if line.startswith("--- ")),
        len(after),
    )
    embedded = [line[1:] for line in after[1:end] if line.startswith("+")]
    assert embedded == cap_lines


def test_compute_build_key_changes_with_every_input_and_option(tmp_path: Path) -> None:
    cap = tmp_path / "h2h_cap.f90"
    patch = tmp_path / "h2h-gmin.patch"
    cap.write_text("cap-v1\n")
    patch.write_text("patch-v1\n")
    base = compute_build_key(REVISION, TARBALL_SHA256, [cap, patch], ["FC=gfortran", "-DWITH_MPI=OFF"])

    assert base == compute_build_key(
        REVISION, TARBALL_SHA256, [cap, patch], ["FC=gfortran", "-DWITH_MPI=OFF"]
    )
    assert base != compute_build_key(
        "a" * 40, TARBALL_SHA256, [cap, patch], ["FC=gfortran", "-DWITH_MPI=OFF"]
    )
    assert base != compute_build_key(
        REVISION, "b" * 64, [cap, patch], ["FC=gfortran", "-DWITH_MPI=OFF"]
    )
    assert base != compute_build_key(
        REVISION, TARBALL_SHA256, [cap, patch], ["FC=gfortran", "-DWITH_MPI=ON"]
    )
    cap.write_text("cap-v2\n")
    after_cap = compute_build_key(
        REVISION, TARBALL_SHA256, [cap, patch], ["FC=gfortran", "-DWITH_MPI=OFF"]
    )
    assert after_cap != base
    patch.write_text("patch-v2\n")
    after_patch = compute_build_key(
        REVISION, TARBALL_SHA256, [cap, patch], ["FC=gfortran", "-DWITH_MPI=OFF"]
    )
    assert after_patch not in {base, after_cap}


def test_read_build_key_rejects_missing_or_mismatched_metadata(tmp_path: Path) -> None:
    meta = tmp_path / "gmin-source.txt"
    key_file = tmp_path / "gmin-build-key"
    valid = "c" * 64
    meta.write_text(f"revision={REVISION}\nsvnrev={REVISION}\n")
    key_file.write_text(f"{valid}\n")
    assert read_build_key(meta, key_file) == ""

    meta.write_text(f"revision={REVISION}\nbuild_key={valid}\n")
    assert read_build_key(meta, key_file) == valid

    key_file.write_text("d" * 64 + "\n")
    assert read_build_key(meta, key_file) == ""


def test_fake_built_svnrev_flows_into_code_revision(tmp_path: Path) -> None:
    svnrev = tmp_path / "SVNREV"
    svnrev.write_text(f"{REVISION}\n")
    revision = read_svnrev(svnrev)
    row = gmin_line(
        "h2h_row best=-1 wall_s=1 pot_s=0.5 charged=10 evals_to_hit=10",
        "m30r6",
        2000,
        400_000,
        -1.0,
        revision,
    )
    assert row["code_revision"] == REVISION


@pytest.mark.parametrize("how", ["env", "flag"])
def test_a_local_tarball_with_the_wrong_sha256_is_refused(tmp_path: Path, how: str) -> None:
    bad = tmp_path / "svn.tar.bz2"
    bad.write_bytes(b"not the pinned snapshot\n")
    work = tmp_path / "work"
    env = os.environ.copy()
    env["GMIN_PANEL_WORK"] = str(work)
    env.pop("GMIN_TARBALL", None)
    command = ["bash", str(BUILD)]
    if how == "env":
        env["GMIN_TARBALL"] = str(bad)
    else:
        command += ["--tarball", str(bad)]
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 1
    assert "tarball sha256 mismatch:" in proc.stderr
    assert TARBALL_SHA256 in proc.stderr
    assert not work.exists()


def _make_builder_sandbox(tmp_path: Path) -> tuple[Path, Path, dict[str, str], Path]:
    repo = tmp_path / "repo"
    for rel in (
        "scripts/gmin_pin.py",
        "scripts/gmin_panel/build_gmin.sh",
        "scripts/gmin_panel/enter_bench.sh",
        "scripts/gmin_panel/enter_losses2.sh",
        "scripts/gmin_h2h/build_gmin.sh",
        "scripts/covalent_h2h/build_gmin.sh",
        "scripts/covalent_h2h/build_gmin_tersoff.sh",
        "scripts/covalent_h2h/sw-timer.patch",
        "scripts/covalent_h2h/tersoff_userpot.f90",
        "scripts/morse_h2h/morse-timer.patch",
        "scripts/metal_h2h/build_gmin_darby.sh",
        "scripts/metal_h2h/build_gmin_sc.sh",
        "bench/lj104-lattice/v2/h2h_cap.f90",
        "bench/lj104-lattice/v2/h2h-gmin.patch",
        "bench/metal-cluster-h2h/gmin-bgupta.patch",
        "bench/metal-cluster-h2h/gmin-sc-time.patch",
    ):
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)

    tree = tmp_path / "tar-src"
    for rel, content in (
        ("wales/GMIN/source/SVNREV", f"{REVISION}\n"),
        ("wales/GMIN/source/SW.f90", "SUBROUTINE SWTWO\nEND SUBROUTINE SWTWO\n"),
        ("wales/GMIN/source/dummy_userpot.f90", "SUBROUTINE USERPOT\nEND SUBROUTINE USERPOT\n"),
        ("wales/CMakeModules/placeholder.cmake", "# cmake\n"),
        ("wales/INTERFACES/placeholder.txt", "interfaces\n"),
        ("wales/SCRIPTS/all/placeholder.txt", "scripts\n"),
        ("wales/HiRE/hire_interface.F90", "! hire\n"),
    ):
        target = tree / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    tarball = tmp_path / "svn.tar.bz2"
    with tarfile.open(tarball, "w:bz2") as archive:
        archive.add(tree / "wales", arcname="./wales")
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()

    pin_path = repo / "scripts/gmin_pin.py"
    pin_path.write_text(
        pin_path.read_text().replace(TARBALL_SHA256, digest, 1)
    )

    stubs = tmp_path / "stubs"
    stubs.mkdir()
    counter = tmp_path / "build-count.txt"
    counter.write_text("0\n")

    (stubs / "gfortran").write_text(
        "#!/usr/bin/env bash\n"
        'echo "${FAKE_GFORTRAN_VER:-13.3.0}"\n'
    )
    (stubs / "patch").write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "dest = pathlib.Path(args[args.index('-d') + 1])\n"
        "data = sys.stdin.read()\n"
        "log = dest / 'applied-patches.log'\n"
        "with log.open('a') as handle:\n"
        "    handle.write(data + '\\n===\\n')\n"
        "if 'SW.f90' in data:\n"
        "    sw = dest / 'source/SW.f90'\n"
        "    sw.write_text(sw.read_text() + '! H2H_POT_ENTER\\n')\n"
    )
    (stubs / "cmake").write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "pathlib.Path('cmake-Record.txt').write_text(\n"
        "    f\"FC={os.environ.get('FC','')}\\n\" + '\\n'.join(sys.argv[1:]) + '\\n'\n"
        ")\n"
    )
    (stubs / "make").write_text(
        f"#!/usr/bin/env python3\n"
        f"import hashlib, pathlib\n"
        f"count_path = pathlib.Path({str(counter)!r})\n"
        f"n = int(count_path.read_text().strip()) + 1\n"
        f"count_path.write_text(f'{{n}}\\n')\n"
        f"src = pathlib.Path('../../source')\n"
        f"cap = (src / 'h2h_cap.f90').read_bytes()\n"
        f"patches = (src.parent / 'applied-patches.log').read_bytes() if (src.parent / 'applied-patches.log').is_file() else b''\n"
        f"userpot = (src / 'dummy_userpot.f90').read_bytes()\n"
        f"cmake_rec = pathlib.Path('cmake-Record.txt').read_bytes()\n"
        f"h = hashlib.sha256(cap + b'|' + patches + b'|' + userpot + b'|' + cmake_rec).hexdigest()\n"
        f"gmin = pathlib.Path('GMIN')\n"
        f"gmin.write_text('#!/usr/bin/env bash\\n# build=' + str(n) + ' sig=' + h + '\\n')\n"
        f"gmin.chmod(0o755)\n"
    )
    (stubs / "ldd").write_text(
        "#!/usr/bin/env bash\n"
        "echo '\tlibopenblas.so.0 => /usr/lib/libopenblas.so.0 (0x0)'\n"
    )
    for tool in ("gfortran", "patch", "cmake", "make", "ldd"):
        (stubs / tool).chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{stubs}:{env['PATH']}"
    env["GMIN_TARBALL"] = str(tarball)
    return repo, tarball, env, counter


@pytest.mark.parametrize(
    ("script_rel", "work_var", "input_rel", "binary_name", "needs_lib"),
    [
        ("scripts/gmin_panel/build_gmin.sh", "GMIN_PANEL_WORK", "bench/lj104-lattice/v2/h2h_cap.f90", "GMIN", False),
        ("scripts/gmin_h2h/build_gmin.sh", "GMIN_H2H_WORK", "bench/lj104-lattice/v2/h2h-gmin.patch", "GMIN", False),
        ("scripts/metal_h2h/build_gmin_darby.sh", "GMIN_DARBY_WORK", "bench/lj104-lattice/v2/h2h_cap.f90", "GMIN", False),
        ("scripts/metal_h2h/build_gmin_sc.sh", "GMIN_SC_WORK", "bench/metal-cluster-h2h/gmin-sc-time.patch", "GMIN", False),
        ("scripts/covalent_h2h/build_gmin_tersoff.sh", "GMIN_H2H_WORK", "bench/lj104-lattice/v2/h2h_cap.f90", "GMIN-tersoff", True),
    ],
)
def test_every_gmin_builder_rebuilds_when_inputs_or_options_change_under_same_svnrev(
    tmp_path: Path,
    script_rel: str,
    work_var: str,
    input_rel: str,
    binary_name: str,
    needs_lib: bool,
) -> None:
    repo, _tarball, base_env, counter = _make_builder_sandbox(tmp_path)
    work = tmp_path / "work"
    env = dict(base_env)
    env[work_var] = str(work)
    if script_rel == "scripts/metal_h2h/build_gmin_sc.sh":
        env["GMIN_DARBY_WORK"] = str(tmp_path / "darby-work")
    lib = tmp_path / "librgpot_covalent.so"
    lib.write_bytes(b"fake-rgpot-v1\n")

    cmd = ["bash", str(repo / script_rel)]
    if needs_lib:
        cmd.append(str(lib))

    subprocess.run(cmd, cwd=repo, env=env, check=True)
    binary = work / "bin" / binary_name
    first_text = binary.read_text()
    first_count = int(counter.read_text().strip())
    assert first_count >= 1

    # Identical inputs and options reuse the cached build.
    subprocess.run(cmd, cwd=repo, env=env, check=True)
    assert int(counter.read_text().strip()) == first_count
    assert binary.read_text() == first_text

    # Mutating h2h_cap.f90 / h2h-gmin.patch without bumping SVNREV forces a rebuild.
    touched = repo / input_rel
    touched.write_text(touched.read_text() + "\n! modified under same SVNREV\n")
    subprocess.run(cmd, cwd=repo, env=env, check=True)
    second_text = binary.read_text()
    second_count = int(counter.read_text().strip())
    assert second_count > first_count
    assert second_text != first_text

    # Changing compiler/build options without changing files also forces a rebuild.
    opt_env = dict(env, FAKE_GFORTRAN_VER="14.1.0")
    subprocess.run(cmd, cwd=repo, env=opt_env, check=True)
    third_count = int(counter.read_text().strip())
    assert third_count > second_count
    assert binary.read_text() != second_text


def test_gmin_panel_builder_replaces_stale_pre_build_key_tree(tmp_path: Path) -> None:
    repo, _tarball, base_env, counter = _make_builder_sandbox(tmp_path)
    work = tmp_path / "panel-work"
    stale_src = work / "gmin-src" / "wales" / "GMIN" / "source"
    stale_src.mkdir(parents=True)
    (stale_src / "SVNREV").write_text(f"{REVISION}\n")
    (stale_src / "h2h_cap.f90").write_text("! stale develop cap\n")
    (work / "bin").mkdir(parents=True)
    stale_bin = work / "bin" / "GMIN"
    stale_bin.write_text("#!/usr/bin/env bash\n# stale develop binary\n")
    stale_bin.chmod(0o755)
    (work / "gmin-source.txt").write_text(
        f"revision={REVISION}\nsvnrev={REVISION}\ntarball_sha256=legacy\n"
    )

    env = dict(base_env, GMIN_PANEL_WORK=str(work))
    subprocess.run(["bash", str(repo / "scripts/gmin_panel/build_gmin.sh")], cwd=repo, env=env, check=True)
    assert int(counter.read_text().strip()) == 1
    assert "stale develop binary" not in stale_bin.read_text()
    assert (stale_src / "h2h_cap.f90").read_text() == (
        repo / "bench/lj104-lattice/v2/h2h_cap.f90"
    ).read_text()
    assert re.search(r"^build_key=[0-9a-f]{64}$", (work / "gmin-source.txt").read_text(), re.M)


def test_gmin_panel_entry_scripts_pass_extracted_gmin_src_svnrev(tmp_path: Path) -> None:
    repo, _tarball, base_env, _counter = _make_builder_sandbox(tmp_path)
    work = tmp_path / "panel-work"
    env = dict(base_env, GMIN_PANEL_WORK=str(work))
    subprocess.run(["bash", str(repo / "scripts/gmin_panel/build_gmin.sh")], cwd=repo, env=env, check=True)

    for script_name, var_name in (
        ("enter_bench.sh", "$WORK"),
        ("enter_losses2.sh", "$OUT"),
    ):
        text = (ROOT / "scripts/gmin_panel" / script_name).read_text()
        matches = re.findall(r"--svnrev\s+(\S+)", text)
        assert len(matches) == 1, (script_name, matches)
        raw_arg = matches[0]
        assert raw_arg.startswith(f"{var_name}/"), (script_name, raw_arg)
        resolved = Path(raw_arg.replace(var_name, str(work), 1))
        assert resolved.is_file(), f"{script_name} points at missing {resolved}"
        assert read_svnrev(resolved) == REVISION
