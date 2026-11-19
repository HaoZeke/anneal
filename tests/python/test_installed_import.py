"""Pytest must import an installed rgpot, not the source tree copy."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFTEST = ROOT / "tests" / "python" / "conftest.py"


def test_conftest_imports_the_installed_package(tmp_path: Path) -> None:
    site = tmp_path / "site-packages"
    package = site / "rgpot"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("ORIGIN = 'installed'\n", encoding="utf-8")
    script = tmp_path / "check_import.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import importlib.util
            import sys
            from pathlib import Path

            sys.path.insert(0, {str(site)!r})
            path = Path({str(CONFTEST)!r})
            spec = importlib.util.spec_from_file_location("conftest_under_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            import rgpot
            print(rgpot.ORIGIN)
            print(Path(rgpot.__file__).resolve())
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[0] == "installed"
    loaded = Path(proc.stdout.splitlines()[1])
    assert site in loaded.parents


def test_conftest_does_not_prepend_the_source_tree() -> None:
    text = CONFTEST.read_text(encoding="utf-8")
    assert "sys.path.insert(0, str(_PYTHON))" not in text
    spec = importlib.util.spec_from_file_location("rgpot_python_conftest", CONFTEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = (ROOT / "python").resolve()
    assert all(Path(entry).resolve() != source for entry in sys.path if entry)
