#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Self-test for check_installed_sources.py on staged install trees.

The check runs `meson install --destdir` and fails when a staged file is
an implementation source. A .py file is allowed only inside the installed
rgpot package directory. These cases stage a tree with a stand-in meson
that copies a fixed tree into the destdir, run the script against it, and
check the exit status and the files it names.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "check_installed_sources.py"

PLATLIB = "usr/lib/python3/dist-packages"
PYTHON_DIR = f"/{PLATLIB}/rgpot"

FAKE_MESON = textwrap.dedent(
    """\
    import shutil
    import sys
    from pathlib import Path

    tree = Path(sys.argv[1])
    dest = Path(sys.argv[sys.argv.index("--destdir") + 1])
    for src in tree.rglob("*"):
        target = dest / src.relative_to(tree)
        if src.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    """
)

LEGIT = [
    "usr/lib/librgpot.so.1",
    "usr/include/rgpot/types.h",
    f"{PLATLIB}/rgpot/__init__.py",
    f"{PLATLIB}/rgpot/_core.abi3.so",
    f"{PLATLIB}/rgpot/ase/__init__.py",
    f"{PLATLIB}/rgpot/ase/calculator.py",
]


class StagedInstallCheck(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory(prefix="rgpot-staged-")
        self.addCleanup(holder.cleanup)
        self.work = Path(holder.name)
        self.fake = self.work / "fake_meson.py"
        self.fake.write_text(FAKE_MESON)
        self.build = self.work / "build"
        self.build.mkdir()

    def stage(self, files: list[str]) -> Path:
        tree = self.work / "tree"
        for rel in files:
            path = tree / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x = 1\n")
        tree.mkdir(exist_ok=True)
        return tree

    def run_check(
        self, files: list[str], python_dir: str | None = PYTHON_DIR
    ) -> subprocess.CompletedProcess[str]:
        tree = self.stage(files)
        argv = [sys.executable, str(SCRIPT)]
        if python_dir is not None:
            argv += ["--python-dir", python_dir]
        argv += [str(self.build), sys.executable, str(self.fake), str(tree)]
        return subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

    def assertRejects(self, proc, *leaked: str) -> None:
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("installed implementation sources:", proc.stderr)
        listed = [
            line.strip()
            for line in proc.stderr.splitlines()
            if line.startswith("  ")
        ]
        self.assertEqual(sorted(listed), sorted(leaked), proc.stderr)

    def test_package_modules_are_allowed(self) -> None:
        proc = self.run_check(LEGIT)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("and no implementation sources", proc.stdout)
        self.assertIn(f"{PLATLIB}/rgpot/__init__.py", proc.stdout)

    def test_stray_module_under_share_is_rejected(self) -> None:
        stray = "usr/share/rgpot/stray.py"
        proc = self.run_check([*LEGIT, stray])
        self.assertRejects(proc, stray)

    def test_module_outside_the_package_is_rejected(self) -> None:
        stray = [
            f"{PLATLIB}/other/helper.py",
            f"{PLATLIB}/rgpot_extra/helper.py",
            f"{PLATLIB}/stray.py",
            "usr/lib/rgpot/stray.py",
            "opt/rgpot/rgpot/stray.py",
        ]
        proc = self.run_check([*LEGIT, *stray])
        self.assertRejects(proc, *stray)

    def test_module_under_the_include_tree_is_rejected(self) -> None:
        stray = "usr/include/rgpot/leak.py"
        proc = self.run_check([*LEGIT, stray])
        self.assertRejects(proc, stray)

    def test_module_is_rejected_without_a_package_directory(self) -> None:
        proc = self.run_check(LEGIT, python_dir=None)
        self.assertRejects(
            proc,
            f"{PLATLIB}/rgpot/__init__.py",
            f"{PLATLIB}/rgpot/ase/__init__.py",
            f"{PLATLIB}/rgpot/ase/calculator.py",
        )

    def test_other_sources_stay_rejected_in_the_package(self) -> None:
        stray = f"{PLATLIB}/rgpot/PotHelpers.cc"
        proc = self.run_check([*LEGIT, stray])
        self.assertRejects(proc, stray)


if __name__ == "__main__":
    unittest.main()
