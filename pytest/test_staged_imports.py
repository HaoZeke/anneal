# Copyright (c) anneal developers
"""Every staged campaign script finds the local modules it imports.

The enter_*.sh wrappers copy a script into a scratch directory and run it
there. A helper the script imports by name has to be copied beside it, or the
run dies with ModuleNotFoundError after the build and the sudo step have
already been paid for. The wrappers do not list helpers by hand: after each
``cp`` of Python files they call ``scripts/stage_python.py`` on the same
directory, which copies whatever the staged scripts import. This test checks
both halves: every ``cp`` of Python files is followed by that call, and a
staging of the real sources leaves no local import unresolved.
"""

from __future__ import annotations

import re
import shlex
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage_python  # noqa: E402

SCAN_DIRS = ("scripts", "bench", "paper", "examples")


def logical_lines(text: str) -> list[str]:
    lines: list[str] = []
    current = ""
    for line in text.splitlines():
        if line.rstrip().endswith("\\"):
            current += line.rstrip()[:-1] + " "
            continue
        lines.append(current + line)
        current = ""
    if current:
        lines.append(current)
    return lines


def normal_dir(token: str) -> str:
    token = token.replace("${", "$").replace("}", "")
    return token.rstrip("/")


def substitute(token: str, here: Path, root: Path) -> str:
    for name, value in (("HERE", here), ("ROOT", root)):
        token = token.replace("${%s}" % name, str(value)).replace("$%s" % name, str(value))
    return token


def python_copies(text: str, here: Path, root: Path) -> dict[str, dict[str, Path | None]]:
    """Destination directory -> {staged file name: repo source or None}."""
    groups: dict[str, dict[str, Path | None]] = {}
    for line in logical_lines(text):
        stripped = line.strip()
        if not re.match(r"cp\s", stripped) or ".py" not in stripped:
            continue
        try:
            tokens = [t for t in shlex.split(stripped) if not t.startswith("-")][1:]
        except ValueError:
            continue
        *sources, dest = tokens
        if dest.endswith(".py"):
            sources = sources[:1]
            directory, names = normal_dir(str(Path(dest).parent)), [Path(dest).name]
        else:
            directory, names = normal_dir(dest), [Path(s).name for s in sources]
        group = groups.setdefault(directory, {})
        for source, name in zip(sources, names):
            resolved = Path(substitute(source, here, root))
            group[name] = resolved if resolved.is_absolute() and resolved.is_file() else None
    return groups


def stage_calls(text: str) -> set[str]:
    calls: set[str] = set()
    for line in logical_lines(text):
        if "stage_python.py" not in line:
            continue
        try:
            tokens = shlex.split(line.strip())
        except ValueError:
            continue
        after = tokens[[i for i, t in enumerate(tokens) if t.endswith("stage_python.py")][0] + 1 :]
        if after:
            calls.add(normal_dir(after[0]))
    return calls


def wrappers() -> list[Path]:
    found: list[Path] = []
    for name in SCAN_DIRS:
        for path in sorted((ROOT / name).rglob("enter_*.sh")):
            if "target" in path.relative_to(ROOT).parts:
                continue
            if python_copies(path.read_text(encoding="utf-8"), path.parent, ROOT):
                found.append(path)
    return found


def unresolved_local_imports(directory: Path, search: list[Path]) -> dict[str, set[str]]:
    """Local modules some script in ``directory`` imports but lacks."""
    missing: dict[str, set[str]] = {}
    for script in sorted(directory.glob("*.py")):
        for name in stage_python.imported_modules(script):
            if (directory / f"{name}.py").exists():
                continue
            if stage_python.find_local(name, search) is not None:
                missing.setdefault(name, set()).add(script.name)
    return missing


def test_staging_wrappers_exist() -> None:
    assert len(wrappers()) >= 20


@pytest.mark.parametrize("wrapper", wrappers(), ids=lambda p: str(p.relative_to(ROOT)))
def test_wrapper_completes_every_staged_directory(wrapper: Path) -> None:
    text = wrapper.read_text(encoding="utf-8")
    called = stage_calls(text)
    for directory, files in python_copies(text, wrapper.parent, ROOT).items():
        if not any(path is not None for path in files.values()):
            continue
        assert directory in called, (
            f"{wrapper.relative_to(ROOT)} copies Python files into {directory} "
            "without calling scripts/stage_python.py on it"
        )


@pytest.mark.parametrize("wrapper", wrappers(), ids=lambda p: str(p.relative_to(ROOT)))
def test_staged_scripts_resolve_their_local_imports(wrapper: Path, tmp_path: Path) -> None:
    text = wrapper.read_text(encoding="utf-8")
    search = [wrapper.parent, ROOT / "scripts"]
    for index, (directory, files) in enumerate(python_copies(text, wrapper.parent, ROOT).items()):
        sources = {name: path for name, path in files.items() if path is not None}
        if not sources:
            continue
        staged = tmp_path / f"dest{index}"
        staged.mkdir()
        for name, path in sources.items():
            shutil.copy2(path, staged / name)
        stage_python.complete(staged, search)
        assert unresolved_local_imports(staged, search) == {}, (
            f"{wrapper.relative_to(ROOT)} leaves local imports unresolved in {directory}"
        )


def test_checker_flags_a_directory_missing_a_helper(tmp_path: Path) -> None:
    helper_dir = tmp_path / "helpers"
    helper_dir.mkdir()
    (helper_dir / "shared.py").write_text("VALUE = 1\n", encoding="utf-8")
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "run.py").write_text("import shared\nimport os\n", encoding="utf-8")
    assert unresolved_local_imports(staged, [helper_dir]) == {"shared": {"run.py"}}
    stage_python.complete(staged, [helper_dir])
    assert unresolved_local_imports(staged, [helper_dir]) == {}
    assert not (staged / "os.py").exists()


def test_stage_python_follows_imports_inside_helpers(tmp_path: Path) -> None:
    helper_dir = tmp_path / "helpers"
    helper_dir.mkdir()
    (helper_dir / "outer.py").write_text("def f():\n    import inner\n", encoding="utf-8")
    (helper_dir / "inner.py").write_text("X = 1\n", encoding="utf-8")
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "run.py").write_text("from outer import f\n", encoding="utf-8")
    copied = [path.name for path in stage_python.complete(staged, [helper_dir])]
    assert sorted(copied) == ["inner.py", "outer.py"]


def test_wrapper_without_the_call_is_flagged() -> None:
    text = 'cp -a "$HERE/run.py" "$WORK/bin/"\n'
    assert stage_calls(text) == set()
    assert stage_calls(text + 'python3 "$ROOT/scripts/stage_python.py" "$WORK/bin/"\n') == {"$WORK/bin"}
