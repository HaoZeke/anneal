# Copyright (c) anneal developers
"""The Python blocks of the Python API page run against the built module."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "orgmode" / "reference" / "python-api.org"
RUNNER = ROOT / "docs" / "scripts" / "run_org_examples.py"


def _blocks():
    spec = importlib.util.spec_from_file_location("run_org_examples", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_python_blocks(PAGE)


def test_the_page_has_examples():
    assert _blocks()


@pytest.mark.parametrize("line, code", _blocks() or [(0, "")])
def test_example_runs(line, code):
    exec(compile(code, f"{PAGE.name}:{line}", "exec"), {})
