# Copyright (c) anneal developers
"""PEP 561 py.typed and .pyi stub parity against the runtime package."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

import anneal
import anneal._core as core

PKG_DIR = Path(anneal.__file__).resolve().parent
ALLOWLIST_PATH = Path(__file__).resolve().parent / "stubtest_allowlist.txt"
ASE_ALLOWLIST_PATH = Path(__file__).resolve().parent / "stubtest_allowlist_ase.txt"
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


def _stub_top_level(
    path: Path,
) -> tuple[set[str], dict[str, ast.FunctionDef], dict[str, ast.ClassDef]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    functions: dict[str, ast.FunctionDef] = {}
    classes: dict[str, ast.ClassDef] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
            functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
            classes[node.name] = node
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
    return names, functions, classes


def _ast_param_spec(
    fn: ast.FunctionDef, *, drop_self: bool = False
) -> list[tuple[str, str, bool]]:
    spec: list[tuple[str, str, bool]] = []
    pos_args = list(fn.args.posonlyargs) + list(fn.args.args)
    n_pos_defaults = len(fn.args.defaults)
    first_default = len(pos_args) - n_pos_defaults
    start_idx = 1 if drop_self else 0
    for idx in range(start_idx, len(pos_args)):
        arg = pos_args[idx]
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
    names, _, _ = _stub_top_level(stub_path)
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
    _, functions, _ = _stub_top_level(stub_path)
    for name, fn_node in sorted(functions.items()):
        runtime_fn = getattr(mod, name)
        assert _ast_param_spec(fn_node) == _runtime_param_spec(runtime_fn), (
            f"signature mismatch for {mod.__name__}.{name}"
        )


def test_core_pyo3_classes_marked_final_and_constructor_signatures_match():
    _, _, core_classes = _stub_top_level(PKG_DIR / "_core.pyi")
    assert len(core_classes) == 17
    for cls_name, cls_node in sorted(core_classes.items()):
        decorator_names = {
            dec.id if isinstance(dec, ast.Name) else getattr(dec, "attr", "")
            for dec in cls_node.decorator_list
        }
        assert "final" in decorator_names, f"{cls_name} must be marked @final"
        runtime_cls = getattr(core, cls_name)
        with pytest.raises(TypeError):
            type(f"_Sub{cls_name}", (runtime_cls,), {})

    checked_inits = 0
    for mod_name, stub_path in STUB_MODULES:
        mod = importlib.import_module(mod_name)
        _, _, classes = _stub_top_level(stub_path)
        for cls_name, cls_node in sorted(classes.items()):
            init_nodes = [
                n
                for n in cls_node.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            ]
            if not init_nodes:
                continue
            runtime_cls = getattr(mod, cls_name)
            assert _ast_param_spec(init_nodes[0], drop_self=True) == _runtime_param_spec(
                runtime_cls
            ), f"constructor signature mismatch for {mod_name}.{cls_name}"
            checked_inits += 1
    assert checked_inits == 18


def _parse_allowlist_entries(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines and lines[0] == HEADER, f"missing copyright header in {path}"
    entries: list[str] = []
    saw_reason = False
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            saw_reason = False
            continue
        if stripped.startswith("#"):
            if "Reason:" in stripped or saw_reason:
                saw_reason = True
            continue
        assert saw_reason, (
            f"allowlist entry {stripped!r} in {path.name} is missing a `# Reason:` comment"
        )
        entries.append(stripped)
        saw_reason = False
    return entries


def test_stubtest_allowlist_entries_all_have_reasons():
    assert len(_parse_allowlist_entries(ALLOWLIST_PATH)) == 2
    assert len(_parse_allowlist_entries(ASE_ALLOWLIST_PATH)) == 1


def _stubtest_args(*, with_ase: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "mypy.stubtest",
        "--allowlist",
        str(ALLOWLIST_PATH),
    ]
    if with_ase:
        cmd.extend(["--allowlist", str(ASE_ALLOWLIST_PATH)])
    cmd.append("anneal")
    return cmd


def test_mypy_stubtest_passes_across_package():
    import mypy  # noqa: F401

    has_ase = importlib.util.find_spec("ase") is not None
    proc = subprocess.run(
        _stubtest_args(with_ase=has_ase),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_mypy_stubtest_passes_without_ase(tmp_path: Path):
    import mypy  # noqa: F401

    (tmp_path / "sitecustomize.py").write_text(
        "import sys\n"
        "for _mod in ('ase', 'ase.calculators', 'ase.calculators.calculator'):\n"
        "    sys.modules[_mod] = None\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{tmp_path}{os.pathsep}{existing}" if existing else str(tmp_path)
    )

    check_no_ase = subprocess.run(
        [
            sys.executable,
            "-c",
            "from anneal.ase import PotentialCalculator; assert PotentialCalculator.__mro__ == (PotentialCalculator, object)",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert check_no_ase.returncode == 0, check_no_ase.stdout + check_no_ase.stderr

    proc = subprocess.run(
        _stubtest_args(with_ase=False),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_mypy_validates_stubs_and_typed_consumer(tmp_path: Path):
    import mypy  # noqa: F401

    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import numpy as np\n"
        "import anneal\n"
        "from anneal.ase import Anneal\n"
        "\n"
        "cfg: anneal.Config = anneal.Config.recommended(6)\n"
        "bonded_cfg: anneal.Config = cfg.with_rigid_bodies(\n"
        "    [[0.0, 0.0, 0.0], [0.0, 0.75, 0.58], [0.0, -0.75, 0.58]],\n"
        "    [8, 1, 1],\n"
        "    bonds=[[0, 1], [0, 2]],\n"
        ")\n"
        "led: anneal.Ledger = anneal.Ledger(100)\n"
        "preset: anneal.Boltzmann = anneal.Boltzmann(t_init=1.0, sigma=0.5)\n"
        "low = np.zeros(2, dtype=np.float64)\n"
        "high = np.ones(2, dtype=np.float64)\n"
        "hist: anneal.History = anneal.run(lambda x: float(x.sum()), low, high, preset)\n"
        "hit: bool = anneal.energy_hits(-1.0, -1.0)\n"
        "gupta: anneal.Gupta = anneal.Gupta.zinc(4)\n"
        "tip: anneal.GminTip4p = anneal.GminTip4p(2)\n"
        "cs: dict[str, object] = anneal.cluster_search(\n"
        "    tip,\n"
        "    None,\n"
        "    2,\n"
        "    20,\n"
        "    start=np.zeros(12, dtype=np.float64),\n"
        "    water_sites=[[0.0, 0.0, -0.0656], [0.0, 0.757, 0.5205], [0.0, -0.757, 0.5205]],\n"
        "    bonds=[[0, 1], [0, 2]],\n"
        ")\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", str(consumer)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    bad_subclass = tmp_path / "bad_subclass.py"
    bad_subclass.write_text(
        "import anneal\n\nclass BadConfig(anneal.Config):\n    pass\n",
        encoding="utf-8",
    )
    bad_proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", str(bad_subclass)],
        capture_output=True,
        text=True,
    )
    assert bad_proc.returncode != 0
    assert "Cannot inherit from final class" in bad_proc.stdout

    bad_direct = tmp_path / "bad_direct.py"
    bad_direct.write_text(
        "import anneal\n\n"
        "g = anneal.Gupta()\n"
        "c = anneal.Config()\n"
        "g2 = anneal._core.Gupta()\n"
        "c2 = anneal._core.Config()\n",
        encoding="utf-8",
    )
    direct_proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", str(bad_direct)],
        capture_output=True,
        text=True,
    )
    assert direct_proc.returncode != 0
    assert direct_proc.stdout.count("[call-arg]") == 4


def test_factory_only_classes_raise_type_error_on_direct_construction():
    for cls in (
        anneal.Gupta,
        anneal.Config,
        anneal._core.Gupta,
        anneal._core.Config,
    ):
        with pytest.raises(TypeError):
            cls()

    assert anneal.Gupta.zinc(4).n_points == 4
    assert anneal.Config.recommended(4).n_points == 4
