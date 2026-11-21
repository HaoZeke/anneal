# Copyright (c) anneal developers
"""PEP 561 py.typed and .pyi stub parity against the runtime package."""

from __future__ import annotations

import ast
import importlib
import inspect
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import anneal
import anneal._core as core

PKG_DIR = Path(anneal.__file__).resolve().parent
HEADER = "# Copyright (c) anneal developers"

STUB_MODULES = (
    ("anneal", PKG_DIR / "__init__.pyi"),
    ("anneal._core", PKG_DIR / "_core.pyi"),
    ("anneal.ase", PKG_DIR / "ase.pyi"),
    ("anneal.device", PKG_DIR / "device.pyi"),
    ("anneal.chemfit", PKG_DIR / "chemfit.pyi"),
    ("anneal.tvm_ffi", PKG_DIR / "tvm_ffi.pyi"),
    ("anneal._landfold", PKG_DIR / "_landfold.pyi"),
)


def _stub_top_level(path: Path) -> tuple[set[str], dict[str, ast.FunctionDef]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    functions: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
            functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.asname is not None:
                    names.add(alias.asname)
    return names, functions


def _ast_param_spec(fn: ast.FunctionDef) -> list[tuple[str, str, bool]]:
    spec: list[tuple[str, str, bool]] = []
    pos_args = list(fn.args.posonlyargs) + list(fn.args.args)
    n_pos_defaults = len(fn.args.defaults)
    first_default = len(pos_args) - n_pos_defaults
    for idx, arg in enumerate(pos_args):
        kind = (
            "POSITIONAL_ONLY"
            if idx < len(fn.args.posonlyargs)
            else "POSITIONAL_OR_KEYWORD"
        )
        spec.append((arg.arg, kind, idx >= first_default))
    if fn.args.vararg is not None:
        spec.append((fn.args.vararg.arg, "VAR_POSITIONAL", False))
    for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        spec.append((arg.arg, "KEYWORD_ONLY", default is not None))
    if fn.args.kwarg is not None:
        spec.append((fn.args.kwarg.arg, "VAR_KEYWORD", False))
    return spec


def _runtime_param_spec(callable_obj) -> list[tuple[str, str, bool]]:
    sig = inspect.signature(callable_obj)
    return [
        (p.name, p.kind.name, p.default is not inspect.Parameter.empty)
        for p in sig.parameters.values()
    ]


def test_py_typed_marker_and_stub_headers_exist():
    assert (PKG_DIR / "py.typed").is_file()
    for _, stub_path in STUB_MODULES:
        assert stub_path.is_file(), f"missing stub {stub_path}"
        first_line = stub_path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line == HEADER, f"missing copyright header in {stub_path}"


@pytest.mark.parametrize(("mod_name", "stub_path"), STUB_MODULES)
def test_stub_covers_exported_names(mod_name: str, stub_path: Path):
    mod = importlib.import_module(mod_name)
    names, _ = _stub_top_level(stub_path)
    if hasattr(mod, "__all__"):
        expected = set(mod.__all__)
    else:
        expected = {
            n for n in dir(mod) if not n.startswith("__") or n == "__version__"
        }
    missing = sorted(expected - names)
    assert not missing, f"{stub_path.name} missing exports: {missing}"


@pytest.mark.parametrize(
    ("mod", "stub_path"),
    [
        (anneal, PKG_DIR / "__init__.pyi"),
        (core, PKG_DIR / "_core.pyi"),
    ],
)
def test_stub_function_signatures_match_runtime(mod, stub_path: Path):
    _, functions = _stub_top_level(stub_path)
    for name, fn_node in sorted(functions.items()):
        runtime_fn = getattr(mod, name)
        assert _ast_param_spec(fn_node) == _runtime_param_spec(runtime_fn), (
            f"signature mismatch for {mod.__name__}.{name}"
        )


def test_mypy_validates_stubs_and_typed_consumer(tmp_path: Path):
    if shutil.which("mypy") is None:
        pytest.skip("mypy is not installed")
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import numpy as np\n"
        "import anneal\n"
        "from anneal.ase import Anneal\n"
        "\n"
        "cfg: anneal.Config = anneal.Config.recommended(6)\n"
        "led: anneal.Ledger = anneal.Ledger(100)\n"
        "preset: anneal.Boltzmann = anneal.Boltzmann(t_init=1.0, sigma=0.5)\n"
        "low = np.zeros(2, dtype=np.float64)\n"
        "high = np.ones(2, dtype=np.float64)\n"
        "hist: anneal.History = anneal.run(lambda x: float(x.sum()), low, high, preset)\n"
        "hit: bool = anneal.energy_hits(-1.0, -1.0)\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", str(consumer)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
