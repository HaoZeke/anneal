# Copyright (c) anneal developers
"""Bidirectional parity tests for ``crates/anneal-python`` type stubs and docstrings."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import anneal
import anneal._core as core

ROOT = Path(__file__).resolve().parent.parent
STUB_PATH = ROOT / "python" / "anneal" / "_core.pyi"
PY_TYPED_PATH = ROOT / "python" / "anneal" / "py.typed"

_KIND_NAMES = {
    inspect.Parameter.POSITIONAL_ONLY: "POSITIONAL_ONLY",
    inspect.Parameter.POSITIONAL_OR_KEYWORD: "POSITIONAL_OR_KEYWORD",
    inspect.Parameter.VAR_POSITIONAL: "VAR_POSITIONAL",
    inspect.Parameter.KEYWORD_ONLY: "KEYWORD_ONLY",
    inspect.Parameter.VAR_KEYWORD: "VAR_KEYWORD",
}


def _stub_tree() -> ast.Module:
    source = STUB_PATH.read_text(encoding="utf-8")
    compile(source, str(STUB_PATH), "exec")
    return ast.parse(source, filename=str(STUB_PATH))


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            names.add(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.add(dec.attr)
    return names


def _stub_params(
    fn: ast.FunctionDef,
    *,
    drop_self: bool,
) -> list[tuple[str, str, bool, Any]]:
    args = fn.args
    posonly = list(args.posonlyargs)
    regular = list(args.args)
    kwonly = list(args.kwonlyargs)

    pos_all = posonly + regular
    pos_defaults: list[ast.expr | None] = [None] * (
        len(pos_all) - len(args.defaults)
    ) + list(args.defaults)

    params: list[tuple[str, str, bool, Any]] = []
    for idx, arg in enumerate(pos_all):
        if idx == 0 and drop_self:
            assert arg.arg == "self", f"{fn.name} must accept self first"
            assert arg.annotation is None, f"{fn.name} self must not be annotated"
            continue
        assert arg.annotation is not None, f"{fn.name}.{arg.arg} missing type annotation"
        kind = "POSITIONAL_ONLY" if idx < len(posonly) else "POSITIONAL_OR_KEYWORD"
        default_node = pos_defaults[idx]
        if default_node is None:
            params.append((arg.arg, kind, False, None))
        else:
            default_val = ast.literal_eval(default_node)
            assert default_val is not Ellipsis, f"{fn.name}.{arg.arg} stub default is Ellipsis"
            params.append((arg.arg, kind, True, default_val))

    if args.vararg is not None:
        assert args.vararg.annotation is not None, f"{fn.name}.*{args.vararg.arg} missing annotation"
        params.append((args.vararg.arg, "VAR_POSITIONAL", False, None))

    for arg, default_node in zip(kwonly, args.kw_defaults):
        assert arg.annotation is not None, f"{fn.name}.{arg.arg} missing type annotation"
        if default_node is None:
            params.append((arg.arg, "KEYWORD_ONLY", False, None))
        else:
            default_val = ast.literal_eval(default_node)
            assert default_val is not Ellipsis, f"{fn.name}.{arg.arg} stub default is Ellipsis"
            params.append((arg.arg, "KEYWORD_ONLY", True, default_val))

    if args.kwarg is not None:
        assert args.kwarg.annotation is not None, f"{fn.name}.**{args.kwarg.arg} missing annotation"
        params.append((args.kwarg.arg, "VAR_KEYWORD", False, None))

    assert fn.returns is not None, f"{fn.name} missing return type annotation"
    return params


def _runtime_params(
    obj: Any,
    *,
    drop_self: bool,
) -> list[tuple[str, str, bool, Any]]:
    sig = inspect.signature(obj)
    params: list[tuple[str, str, bool, Any]] = []
    for idx, param in enumerate(sig.parameters.values()):
        if idx == 0 and drop_self:
            assert param.name == "self", f"{obj} first parameter is {param.name!r}, not 'self'"
            continue
        has_default = param.default is not inspect.Parameter.empty
        if has_default:
            assert param.default is not Ellipsis, f"{obj} parameter {param.name} has Ellipsis default"
        params.append(
            (
                param.name,
                _KIND_NAMES[param.kind],
                has_default,
                param.default if has_default else None,
            )
        )
    return params


def _assert_sig_equal(
    label: str,
    runtime_obj: Any,
    stub_fn: ast.FunctionDef,
    *,
    drop_runtime_self: bool,
    drop_stub_self: bool,
) -> None:
    rt = _runtime_params(runtime_obj, drop_self=drop_runtime_self)
    st = _stub_params(stub_fn, drop_self=drop_stub_self)
    assert st == rt, f"signature mismatch for {label}:\n  stub:    {st}\n  runtime: {rt}"


def _parse_stub() -> tuple[dict[str, ast.AnnAssign], dict[str, ast.ClassDef], dict[str, ast.FunctionDef]]:
    tree = _stub_tree()
    consts: dict[str, ast.AnnAssign] = {}
    classes: dict[str, ast.ClassDef] = {}
    funcs: dict[str, ast.FunctionDef] = {}
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            assert stmt.target.id not in consts, f"duplicate stub constant {stmt.target.id}"
            consts[stmt.target.id] = stmt
        elif isinstance(stmt, ast.ClassDef):
            assert stmt.name not in classes, f"duplicate stub class {stmt.name}"
            classes[stmt.name] = stmt
        elif isinstance(stmt, ast.FunctionDef):
            assert stmt.name not in funcs, f"duplicate stub function {stmt.name}"
            funcs[stmt.name] = stmt
        else:
            pytest.fail(f"unexpected top-level stub statement: {ast.dump(stmt)}")
    return consts, classes, funcs


def _runtime_exports() -> tuple[dict[str, Any], dict[str, type], dict[str, Any]]:
    consts: dict[str, Any] = {}
    classes: dict[str, type] = {}
    funcs: dict[str, Any] = {}
    for name in dir(core):
        if name.startswith("__") and name != "__version__":
            continue
        val = getattr(core, name)
        if inspect.isclass(val):
            classes[name] = val
        elif callable(val):
            funcs[name] = val
        else:
            consts[name] = val
    return consts, classes, funcs


def test_py_typed_marker_and_stub_exist() -> None:
    assert STUB_PATH.is_file(), f"missing type stub {STUB_PATH}"
    assert PY_TYPED_PATH.is_file(), f"missing PEP 561 marker {PY_TYPED_PATH}"
    _stub_tree()


def test_core_exported_names_bidirectional_parity() -> None:
    stub_consts, stub_classes, stub_funcs = _parse_stub()
    rt_consts, rt_classes, rt_funcs = _runtime_exports()

    assert set(stub_consts) == set(rt_consts)
    assert set(stub_classes) == set(rt_classes)
    assert set(stub_funcs) == set(rt_funcs)

    for name, value in rt_consts.items():
        ann = ast.unparse(stub_consts[name].annotation)
        assert ann == type(value).__name__, f"constant {name}: stub {ann!r} != runtime {type(value).__name__!r}"


def test_core_function_signatures_and_defaults_bidirectional_parity() -> None:
    _, _, stub_funcs = _parse_stub()
    _, _, rt_funcs = _runtime_exports()

    assert set(stub_funcs) == set(rt_funcs)
    for name, rt_fn in sorted(rt_funcs.items()):
        assert not _decorator_names(stub_funcs[name]), f"unexpected decorator on {name}"
        _assert_sig_equal(
            name,
            rt_fn,
            stub_funcs[name],
            drop_runtime_self=False,
            drop_stub_self=False,
        )


def test_core_class_members_and_signatures_bidirectional_parity() -> None:
    _, stub_classes, _ = _parse_stub()
    _, rt_classes, _ = _runtime_exports()

    assert set(stub_classes) == set(rt_classes)
    for cls_name, rt_cls in sorted(rt_classes.items()):
        stub_cls = stub_classes[cls_name]
        stub_methods: dict[str, ast.FunctionDef] = {}
        stub_props: dict[str, ast.FunctionDef] = {}
        stub_attrs: dict[str, ast.AnnAssign] = {}

        for stmt in stub_cls.body:
            if isinstance(stmt, ast.FunctionDef):
                decs = _decorator_names(stmt)
                if "property" in decs:
                    assert stmt.name not in stub_props, f"duplicate property {cls_name}.{stmt.name}"
                    assert stmt.returns is not None, f"{cls_name}.{stmt.name} property missing return type"
                    assert len(stmt.args.args) == 1 and stmt.args.args[0].arg == "self"
                    stub_props[stmt.name] = stmt
                else:
                    assert stmt.name not in stub_methods, f"duplicate method {cls_name}.{stmt.name}"
                    stub_methods[stmt.name] = stmt
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                assert stmt.target.id not in stub_attrs, f"duplicate attr {cls_name}.{stmt.target.id}"
                stub_attrs[stmt.target.id] = stmt
            else:
                pytest.fail(f"unexpected class body node in {cls_name}: {ast.dump(stmt)}")

        # Constructor parity.
        has_ctor = getattr(rt_cls, "__text_signature__", None) is not None
        if has_ctor:
            assert "__init__" in stub_methods, f"{cls_name} has a constructor at runtime but no __init__ in stub"
            init_fn = stub_methods["__init__"]
            assert ast.unparse(init_fn.returns) == "None"
            _assert_sig_equal(
                f"{cls_name}.__init__",
                rt_cls,
                init_fn,
                drop_runtime_self=False,
                drop_stub_self=True,
            )
        else:
            assert "__init__" not in stub_methods, f"{cls_name} is not constructible at runtime"
            assert "__new__" not in stub_methods, f"{cls_name} is not constructible at runtime"
            with pytest.raises(TypeError):
                rt_cls()

        rt_members = {
            k: v
            for k, v in rt_cls.__dict__.items()
            if not k.startswith("__") or k == "__repr__"
        }
        stub_members = (set(stub_methods) - {"__init__"}) | set(stub_props) | set(stub_attrs)
        assert stub_members == set(rt_members), (
            f"{cls_name} members mismatch: stub-only={stub_members - set(rt_members)}, "
            f"runtime-only={set(rt_members) - stub_members}"
        )

        for member_name, rt_member in sorted(rt_members.items()):
            raw_kind = type(rt_member).__name__
            if raw_kind == "staticmethod":
                fn = stub_methods[member_name]
                assert "staticmethod" in _decorator_names(fn), f"{cls_name}.{member_name} must be @staticmethod"
                _assert_sig_equal(
                    f"{cls_name}.{member_name}",
                    getattr(rt_cls, member_name),
                    fn,
                    drop_runtime_self=False,
                    drop_stub_self=False,
                )
            elif raw_kind in {"method_descriptor", "wrapper_descriptor"}:
                fn = stub_methods[member_name]
                assert not _decorator_names(fn), f"{cls_name}.{member_name} has unexpected decorator"
                _assert_sig_equal(
                    f"{cls_name}.{member_name}",
                    rt_member,
                    fn,
                    drop_runtime_self=True,
                    drop_stub_self=True,
                )
            elif raw_kind == "getset_descriptor":
                assert member_name in stub_props or member_name in stub_attrs
            else:
                pytest.fail(f"unhandled descriptor type {raw_kind!r} on {cls_name}.{member_name}")


def test_core_docstrings_present_and_accurate() -> None:
    _, rt_classes, rt_funcs = _runtime_exports()

    for name, fn in sorted(rt_funcs.items()):
        doc = inspect.getdoc(fn)
        assert doc and doc.strip(), f"anneal._core.{name} is missing a docstring"

    for name, cls in sorted(rt_classes.items()):
        doc = inspect.getdoc(cls)
        assert doc and doc.strip(), f"anneal._core.{name} is missing a docstring"
        if cls.__module__ == "anneal._core":
            for member_name in cls.__dict__:
                if member_name.startswith("__") and member_name != "__repr__":
                    continue
                member_doc = inspect.getdoc(getattr(cls, member_name))
                assert member_doc and member_doc.strip(), (
                    f"anneal._core.{name}.{member_name} is missing a docstring"
                )


def _assert_matches_stub_type(value: Any, ann: str, label: str) -> None:
    if ann == "None":
        assert value is None, f"{label}: expected None, got {value!r}"
    elif ann == "bool":
        assert isinstance(value, bool), f"{label}: expected bool, got {type(value).__name__}"
    elif ann == "int":
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"{label}: expected int, got {type(value).__name__}"
        )
    elif ann == "float":
        assert isinstance(value, float), f"{label}: expected float, got {type(value).__name__}"
    elif ann == "str":
        assert isinstance(value, str), f"{label}: expected str, got {type(value).__name__}"
    elif ann == "tuple[float, float]":
        assert isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, float) for v in value), (
            f"{label}: expected tuple[float, float], got {value!r}"
        )
    elif ann == "list[list[float]]":
        assert isinstance(value, list) and all(
            isinstance(r, list) and all(isinstance(v, float) for v in r) for r in value
        ), f"{label}: expected list[list[float]], got {value!r}"
    elif ann == "list[float]":
        assert isinstance(value, list) and all(isinstance(v, float) for v in value), (
            f"{label}: expected list[float], got {value!r}"
        )
    elif ann == "list[bool]":
        assert isinstance(value, list) and all(isinstance(v, bool) for v in value), (
            f"{label}: expected list[bool], got {value!r}"
        )
    elif ann == "list[EpochLine]":
        assert isinstance(value, list) and all(isinstance(e, core.EpochLine) for e in value), (
            f"{label}: expected list[EpochLine], got {value!r}"
        )
    elif ann == "dict[str, Any]":
        assert isinstance(value, dict) and all(isinstance(k, str) for k in value), (
            f"{label}: expected dict[str, Any], got {type(value).__name__}"
        )
    elif ann == "npt.NDArray[np.float64]":
        assert isinstance(value, np.ndarray) and value.dtype == np.float64, (
            f"{label}: expected float64 ndarray, got {type(value).__name__}"
        )
    elif ann == "DeviceHistory":
        assert isinstance(value, anneal.DeviceHistory), f"{label}: expected DeviceHistory, got {type(value).__name__}"
    elif ann == "EnsembleHistory":
        assert isinstance(value, anneal.EnsembleHistory), (
            f"{label}: expected EnsembleHistory, got {type(value).__name__}"
        )
    elif hasattr(core, ann):
        expected_cls = getattr(core, ann)
        assert isinstance(value, expected_cls), f"{label}: expected {ann}, got {type(value).__name__}"
    else:
        pytest.fail(f"unhandled stub return annotation {ann!r} for {label}")


def _stub_return_ann(func_name: str, cls_name: str | None = None) -> str:
    _, stub_classes, stub_funcs = _parse_stub()
    if cls_name is None:
        return ast.unparse(stub_funcs[func_name].returns)
    for stmt in stub_classes[cls_name].body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == func_name:
            return ast.unparse(stmt.returns)
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == func_name:
            return ast.unparse(stmt.annotation)
    raise AssertionError(f"missing {cls_name}.{func_name} in stub")


def test_core_runtime_return_types_and_documented_keys(tmp_path: Path) -> None:
    low = np.array([-2.0, -2.0], dtype=np.float64)
    high = np.array([2.0, 2.0], dtype=np.float64)
    x0 = np.array([0.5, -0.5], dtype=np.float64)

    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x * x))

    def sphere_grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    def sphere_fg(x: np.ndarray) -> tuple[float, np.ndarray]:
        return sphere(x), sphere_grad(x)

    assert isinstance(core.__version__, str)
    assert isinstance(core.DEFAULT_HIT_TOL, float)
    assert isinstance(core.SEARCH_HIT_TOL, float)

    _assert_matches_stub_type(core.energy_hits(-1.0, -1.0), _stub_return_ann("energy_hits"), "energy_hits")
    _assert_matches_stub_type(
        core.osa_acceptance_rate(0.1, 1.0, 0.2, trials=20, seed=1),
        _stub_return_ann("osa_acceptance_rate"),
        "osa_acceptance_rate",
    )
    _assert_matches_stub_type(
        core.low_discrepancy_points(low, high, 4),
        _stub_return_ann("low_discrepancy_points"),
        "low_discrepancy_points",
    )
    _assert_matches_stub_type(
        core.pilot_draws_qmc(4, seed=1),
        _stub_return_ann("pilot_draws_qmc"),
        "pilot_draws_qmc",
    )
    _assert_matches_stub_type(
        core.estimate_gle_omega0(sphere, sphere_grad, low, high),
        _stub_return_ann("estimate_gle_omega0"),
        "estimate_gle_omega0",
    )

    boltzmann = core.Boltzmann()
    boltzmann.t_init = 1.5
    boltzmann.sigma = 0.25
    fast = core.Fast()
    fast.t_init = 1.2
    fast.gamma = 0.4
    gsa = core.Gsa()
    gsa.t_init = 1.1
    gsa.q_v = 2.5
    gsa.q_a = 1.5
    for cls_name, inst, attrs in (
        ("Boltzmann", boltzmann, ("t_init", "sigma")),
        ("Fast", fast, ("t_init", "gamma")),
        ("Gsa", gsa, ("t_init", "q_v", "q_a")),
    ):
        _assert_matches_stub_type(repr(inst), _stub_return_ann("__repr__", cls_name), f"{cls_name}.__repr__")
        for attr in attrs:
            _assert_matches_stub_type(getattr(inst, attr), _stub_return_ann(attr, cls_name), f"{cls_name}.{attr}")

    ledger = core.Ledger(10)
    with pytest.raises(AttributeError):
        ledger.budget = 5  # type: ignore[misc]
    _assert_matches_stub_type(repr(ledger), _stub_return_ann("__repr__", "Ledger"), "Ledger.__repr__")
    for prop in ("budget", "spent", "remaining", "best"):
        _assert_matches_stub_type(getattr(ledger, prop), _stub_return_ann(prop, "Ledger"), f"Ledger.{prop}")

    hist = core.run(sphere, low, high, boltzmann, n_epochs=2, steps_per_epoch=3, seed=1, x0=x0)
    _assert_matches_stub_type(hist, _stub_return_ann("run"), "run")
    for prop in ("epochs", "best_pos", "best_val", "stationarity_flags", "total_accepted", "total_rejected"):
        _assert_matches_stub_type(getattr(hist, prop), _stub_return_ann(prop, "History"), f"History.{prop}")
    ep = hist.epochs[0]
    for prop in ("epoch", "temp", "accepted", "rejected", "best_val"):
        _assert_matches_stub_type(getattr(ep, prop), _stub_return_ann(prop, "EpochLine"), f"EpochLine.{prop}")

    _assert_matches_stub_type(
        core.run_hmc(sphere, sphere_grad, low, high, n_epochs=2, steps_per_epoch=2, seed=1, x0=x0),
        _stub_return_ann("run_hmc"),
        "run_hmc",
    )
    _assert_matches_stub_type(
        core.run_qmc(sphere, low, high, fast, n_starts=2, n_epochs=2, steps_per_epoch=2, seed=1),
        _stub_return_ann("run_qmc"),
        "run_qmc",
    )
    _assert_matches_stub_type(
        core._run_device(lambda x: x[0] * x[0] + x[1] * x[1], low, high, gsa, n_epochs=2, steps_per_epoch=2, seed=1),
        _stub_return_ann("_run_device"),
        "_run_device",
    )
    _assert_matches_stub_type(
        core._run_device_ensemble(
            lambda x: x[:, 0] ** 2 + x[:, 1] ** 2,
            low,
            high,
            boltzmann,
            n_chains=2,
            n_epochs=2,
            steps_per_epoch=2,
            seed=1,
        ),
        _stub_return_ann("_run_device_ensemble"),
        "_run_device_ensemble",
    )

    bounds = core.Bounds(low, high)
    obj = core.PyObjective(sphere, bounds, sphere_grad)
    _assert_matches_stub_type(bounds.dims, _stub_return_ann("dims", "Bounds"), "Bounds.dims")
    _assert_matches_stub_type(bounds.slack, _stub_return_ann("slack", "Bounds"), "Bounds.slack")
    _assert_matches_stub_type(bounds.contains(x0), _stub_return_ann("contains", "Bounds"), "Bounds.contains")
    _assert_matches_stub_type(bounds.clip(x0), _stub_return_ann("clip", "Bounds"), "Bounds.clip")
    _assert_matches_stub_type(obj.bounds, _stub_return_ann("bounds", "PyObjective"), "PyObjective.bounds")
    _assert_matches_stub_type(obj.dim, _stub_return_ann("dim", "PyObjective"), "PyObjective.dim")
    _assert_matches_stub_type(obj.eval(x0), _stub_return_ann("eval", "PyObjective"), "PyObjective.eval")
    _assert_matches_stub_type(obj.grad(x0), _stub_return_ann("grad", "PyObjective"), "PyObjective.grad")

    bias = core.BasinBias(2)
    pts6 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    _assert_matches_stub_type(bias.potential(pts6), _stub_return_ann("potential", "BasinBias"), "BasinBias.potential")
    _assert_matches_stub_type(bias.deposit(pts6, 1.0), _stub_return_ann("deposit", "BasinBias"), "BasinBias.deposit")
    _assert_matches_stub_type(repr(bias), _stub_return_ann("__repr__", "BasinBias"), "BasinBias.__repr__")
    for prop in ("n_basins", "deepest", "missing_mass"):
        _assert_matches_stub_type(getattr(bias, prop), _stub_return_ann(prop, "BasinBias"), f"BasinBias.{prop}")

    for ctor_name in (
        "recommended",
        "communicating",
        "derived",
        "for_cluster",
        "communicating_tip4p",
        "recommended_tip4p",
    ):
        cfg = getattr(core.Config, ctor_name)(3)
        _assert_matches_stub_type(cfg, _stub_return_ann(ctor_name, "Config"), f"Config.{ctor_name}")
    _assert_matches_stub_type(
        core.Config.validate_json('{"prior_soap": true}'),
        _stub_return_ann("validate_json", "Config"),
        "Config.validate_json",
    )
    _assert_matches_stub_type(
        core.Config.validate_toml("prior_soap = true\n"),
        _stub_return_ann("validate_toml", "Config"),
        "Config.validate_toml",
    )
    cfg = core.Config.recommended(3)
    _assert_matches_stub_type(repr(cfg), _stub_return_ann("__repr__", "Config"), "Config.__repr__")
    for prop in (
        "n_points",
        "rigid_body",
        "screen_margin",
        "calibrated_screen",
        "finish_improving",
        "burst_moves",
        "allocate_moves",
        "depth_reward",
        "orbit_complete_on_new",
        "shared_visit_policy",
        "tabu_on_stall",
        "bayes_threshold",
        "budget_window",
    ):
        _assert_matches_stub_type(getattr(cfg, prop), _stub_return_ann(prop, "Config"), f"Config.{prop}")

    def doc_return_keys(fn: Any) -> set[str]:
        m = re.search(r"Returns\s+`\{([^}]+)\}`", inspect.getdoc(fn) or "", re.DOTALL)
        assert m, f"missing Returns {{...}} in {fn.__name__} docstring"
        return {part.strip() for part in m.group(1).split(",")}

    quench_out = core.quench(sphere_fg, x0, 1e-4, max_iter=20)
    _assert_matches_stub_type(quench_out, _stub_return_ann("quench"), "quench")
    assert set(quench_out) == doc_return_keys(core.quench)

    add_out = core.additive_independence(sphere, low, high, 300, seed=1, n_epochs=2)
    _assert_matches_stub_type(add_out, _stub_return_ann("additive_independence"), "additive_independence")
    assert set(add_out) == doc_return_keys(core.additive_independence)

    gle_out = core.gle_langevin(sphere, sphere_grad, low, high, 20, seed=1, n_epochs=2, x0=x0)
    _assert_matches_stub_type(gle_out, _stub_return_ann("gle_langevin"), "gle_langevin")
    assert set(gle_out) == doc_return_keys(core.gle_langevin)

    def lj2(x: np.ndarray) -> float:
        d = x[3:] - x[:3]
        r2 = float(np.dot(d, d))
        inv6 = (1.0 / max(r2, 1e-6)) ** 3
        return 4.0 * (inv6 * inv6 - inv6)

    def lj2_grad(x: np.ndarray) -> np.ndarray:
        d = x[3:] - x[:3]
        r2 = max(float(np.dot(d, d)), 1e-6)
        inv2 = 1.0 / r2
        inv6 = inv2**3
        factor = 24.0 * (inv6 - 2.0 * inv6 * inv6) * inv2
        g = np.zeros(6, dtype=np.float64)
        g[:3] = -factor * d
        g[3:] = factor * d
        return g

    cs_out = core.cluster_search(lj2, lj2_grad, 2, 20, seed=1)
    _assert_matches_stub_type(cs_out, _stub_return_ann("cluster_search"), "cluster_search")
    assert set(cs_out) == doc_return_keys(core.cluster_search)

    for fn_name, out in (
        ("polish", core.polish(sphere, sphere_grad, low, high, x0, max_fevals=10)),
        ("qmc_polish", core.qmc_polish(sphere, sphere_grad, low, high, 2, 5, seed=1)),
        ("qmc_polish_objective", core.qmc_polish_objective(obj, 2, 5, seed=1)),
        ("qmc_best1bin_scout", core.qmc_best1bin_scout(sphere, low, high, 12, seed=1, population_size=4)),
        ("qmc_best1bin_scout_objective", core.qmc_best1bin_scout_objective(obj, 12, seed=1, population_size=4)),
        ("qmc_gsa_global_search", core.qmc_gsa_global_search(sphere, low, high, 12, seed=1, n_chains=2)),
        ("qmc_gsa_global_search_objective", core.qmc_gsa_global_search_objective(obj, 12, seed=1, n_chains=2)),
        ("qmc_trust_region_poll", core.qmc_trust_region_poll(sphere, low, high, x0, 12, seed=1)),
        ("qmc_trust_region_poll_objective", core.qmc_trust_region_poll_objective(obj, x0, 12, seed=1)),
        ("shifted_qmc_polish", core.shifted_qmc_polish(sphere, sphere_grad, low, high, 2, 5, seed=1)),
        ("gle_langevin_objective", core.gle_langevin_objective(obj, 20, seed=1, n_epochs=2, x0=x0)),
        ("gle_langevin_preconditioned", core.gle_langevin_preconditioned(sphere, sphere_grad, low, high, 20, seed=1, n_epochs=2, x0=x0)),
        ("gle_langevin_preconditioned_objective", core.gle_langevin_preconditioned_objective(obj, 20, seed=1, n_epochs=2, x0=x0)),
        ("dmc_population_optimize", core.dmc_population_optimize(sphere, low, high, 20, seed=1, target_n=4)),
        ("gpmd_optimize", core.gpmd_optimize(sphere, low, high, 20, seed=1)),
        ("amsa_optimize", core.amsa_optimize(sphere, low, high, 20, seed=1)),
        ("box_ensemble_optimize", core.box_ensemble_optimize(sphere, low, high, 20, seed=1, grad_fn=sphere_grad, replicas=2)),
        ("ensemble_optimize", core.ensemble_optimize(sphere, low, high, 20, seed=1, replicas=2)),
        ("bfwt_optimize", core.bfwt_optimize(sphere, low, high, 20, seed=1)),
        ("global_optimize", core.global_optimize(sphere, low, high, 20, seed=1)),
        ("global_optimize_objective", core.global_optimize_objective(obj, 20, seed=1, use_gradient=False)),
    ):
        _assert_matches_stub_type(out, _stub_return_ann(fn_name), fn_name)
        assert "best_pos" in out and "best_val" in out

    rows = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    lf_out = core.landfold(rows, landmarks=4, grid=5, iterations=5, seed=1)
    _assert_matches_stub_type(lf_out, _stub_return_ann("landfold"), "landfold")
    lf_map = lf_out["map"]
    _assert_matches_stub_type(lf_map, "LandfoldMap", "landfold['map']")
    _assert_matches_stub_type(lf_map.landmark_xy, _stub_return_ann("landmark_xy", "LandfoldMap"), "LandfoldMap.landmark_xy")
    _assert_matches_stub_type(lf_map.weights, _stub_return_ann("weights", "LandfoldMap"), "LandfoldMap.weights")
    _assert_matches_stub_type(lf_map.project(rows), _stub_return_ann("project", "LandfoldMap"), "LandfoldMap.project")

    frames = []
    for k in range(3):
        atoms = "\n".join(
            f"Ar {i * 1.1 + 0.05 * k:.3f} {(i % 3) * 1.1:.3f} {(i // 3) * 1.1:.3f}"
            for i in range(13)
        )
        frames.append(f"13\nenergy = {-10.0 + k * 0.5}\n{atoms}\n")
    xyz = tmp_path / "minima.xyz"
    xyz.write_text("".join(frames), encoding="utf-8")
    book_out = core.landfold_book([xyz], landmarks=3, grid=5, iterations=5, seed=1)
    _assert_matches_stub_type(book_out, _stub_return_ann("landfold_book"), "landfold_book")
    assert {"energy", "rows", "family", "community"} <= set(book_out)


def test_public_core_symbols_are_reexported_by_anneal() -> None:
    public_core = {
        name for name in dir(core) if not name.startswith("_") or name == "__version__"
    }
    assert public_core <= set(anneal.__all__)
    for name in public_core:
        assert hasattr(anneal, name)

    # Check wrapper parity for quench and device helpers.
    for fn_name, core_name in (
        ("quench", "quench"),
        ("run_device", "_run_device"),
        ("run_ensemble", "_run_device_ensemble"),
    ):
        assert _runtime_params(getattr(anneal, fn_name), drop_self=False) == _runtime_params(
            getattr(core, core_name), drop_self=False
        )


def test_python_api_reference_docs_list_every_public_name() -> None:
    import importlib
    import pkgutil
    import re

    ref_doc = ROOT / "docs" / "orgmode" / "reference" / "presets-and-advanced.org"
    text = ref_doc.read_text(encoding="utf-8")
    documented = set(
        re.findall(r"^- =([A-Za-z_][A-Za-z0-9_]*)(?:[=(])", text, re.MULTILINE)
    )

    modules = ["anneal", "anneal._core"] + [
        m.name for m in pkgutil.walk_packages(anneal.__path__, prefix="anneal.")
    ]
    missing: dict[str, list[str]] = {}
    for modname in modules:
        mod = importlib.import_module(modname)
        if hasattr(mod, "__all__"):
            raw_names = list(mod.__all__)
        else:
            raw_names = dir(mod)
        names = [n for n in raw_names if not n.startswith("_") or n == "__version__"]
        mod_missing = [n for n in names if n not in documented]
        if mod_missing:
            missing[modname] = mod_missing
    assert not missing, f"Public Python exports missing from {ref_doc}: {missing}"
