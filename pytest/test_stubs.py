# Copyright (c) anneal developers
"""The stubs of ``anneal._core`` describe the compiled module and nothing else.

``mypy.stubtest`` is the reference check and runs when mypy is installed. The
introspection check below needs only the standard library, so it runs
everywhere and also covers what the stubs leave to stubtest's defaults: the
argument kinds, the defaults and the way a class is built.
"""

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import anneal
import anneal._core as core

PACKAGE = Path(anneal.__file__).parent
STUB = PACKAGE / "_core.pyi"
MODULE_DUNDERS = {
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "__path__",
    "__all__",
}


def _arguments(node, drop_self):
    """Names, kinds and default presence of a stub function's arguments."""
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    out = [(a.arg, "pos", d is not None) for a, d in zip(positional, defaults)]
    if args.vararg:
        out.append((f"*{args.vararg.arg}", "var", False))
    out += [(a.arg, "kw", d is not None) for a, d in zip(args.kwonlyargs, args.kw_defaults)]
    if args.kwarg:
        out.append((f"**{args.kwarg.arg}", "var", False))
    if drop_self and out and out[0][0] in ("self", "cls"):
        out = out[1:]
    return out


def _runtime_arguments(callable_, drop_self):
    out = []
    for p in inspect.signature(callable_).parameters.values():
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            out.append((f"*{p.name}", "var", False))
        elif p.kind is inspect.Parameter.VAR_KEYWORD:
            out.append((f"**{p.name}", "var", False))
        else:
            kind = "kw" if p.kind is inspect.Parameter.KEYWORD_ONLY else "pos"
            out.append((p.name, kind, p.default is not inspect.Parameter.empty))
    if drop_self and out and out[0][0] == "self":
        out = out[1:]
    return out


def _stub_all(text):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["__all__"]:
            return {element.value for element in node.value.elts}
    return None


def _stub_surface(text):
    """Functions, classes and constants a stub declares at module level."""
    functions, classes, constants = {}, {}, set()
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            constants.add(node.target.id)
    return functions, classes, constants


def _decorators(node):
    return {d.id for d in node.decorator_list if isinstance(d, ast.Name)}


def _class_members(node):
    members = {}
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            decorators = _decorators(item)
            if "property" in decorators:
                members[item.name] = ("property", item)
            elif "staticmethod" in decorators:
                members[item.name] = ("staticmethod", item)
            else:
                members[item.name] = ("method", item)
    return members


def _runtime_kind(cls, name):
    attribute = inspect.getattr_static(cls, name)
    if isinstance(attribute, staticmethod):
        return "staticmethod"
    if isinstance(attribute, property) or type(attribute).__name__ == "getset_descriptor":
        return "property"
    return "method"


def stub_problems(text, module):
    """Every way the stub text disagrees with the module, as sentences."""
    problems = []
    functions, classes, constants = _stub_surface(text)
    declared = set(functions) | set(classes) | constants
    present = {
        name
        for name in dir(module)
        if not (name.startswith("__") and name.endswith("__")) or name == "__version__"
    } - MODULE_DUNDERS
    for name in sorted(present - declared):
        problems.append(f"{name} is in the module but not in the stub")
    for name in sorted(declared - present):
        problems.append(f"{name} is in the stub but not in the module")
    listed = _stub_all(text)
    if listed is None:
        problems.append("the stub has no __all__")
    elif listed != set(getattr(module, "__all__", ())):
        problems.append(f"__all__ differs: {sorted(listed ^ set(module.__all__))}")

    for name, node in functions.items():
        if name not in present:
            continue
        stub_args = _arguments(node, drop_self=False)
        run_args = _runtime_arguments(getattr(module, name), drop_self=False)
        if stub_args != run_args:
            problems.append(f"{name}: stub arguments {stub_args} != module arguments {run_args}")

    for name, node in classes.items():
        if name not in present:
            continue
        cls = getattr(module, name)
        if issubclass(cls, BaseException):
            stub_bases = [ast.unparse(base) for base in node.bases]
            run_bases = [base.__name__ for base in cls.__bases__]
            if stub_bases != run_bases:
                problems.append(f"{name}: stub bases {stub_bases} != module bases {run_bases}")
            continue
        members = _class_members(node)
        init = members.pop("__new__", None)
        runtime_members = {
            member
            for member in dir(cls)
            if not (member.startswith("__") and member.endswith("__"))
        }
        for member in sorted(runtime_members - set(members)):
            problems.append(f"{name}.{member} is in the module but not in the stub")
        for member in sorted(set(members) - runtime_members):
            problems.append(f"{name}.{member} is in the stub but not in the module")
        for member, (kind, member_node) in members.items():
            if member not in runtime_members:
                continue
            actual = _runtime_kind(cls, member)
            if actual != kind:
                problems.append(f"{name}.{member}: stub says {kind}, module has {actual}")
                continue
            if kind == "property":
                continue
            drop = kind == "method"
            stub_args = _arguments(member_node, drop_self=drop)
            if kind == "staticmethod":
                run_args = _runtime_arguments(getattr(cls, member), drop_self=False)
            else:
                run_args = _runtime_arguments(getattr(cls, member), drop_self=True)
            if stub_args != run_args:
                problems.append(f"{name}.{member}: stub arguments {stub_args} != module arguments {run_args}")
        if init is not None:
            stub_args = _arguments(init[1], drop_self=True)
            run_args = _runtime_arguments(cls, drop_self=False)
            if stub_args != run_args:
                problems.append(f"{name}(): stub arguments {stub_args} != module arguments {run_args}")
        elif cls.__text_signature__ is not None:
            problems.append(f"{name}() is constructible but the stub has no __new__")
        else:
            try:
                cls()
            except TypeError:
                pass
            else:
                problems.append(f"{name}() builds an instance but the stub has no __new__")
    return problems


def test_package_ships_the_stub_and_the_typed_marker():
    assert STUB.is_file()
    assert (PACKAGE / "py.typed").is_file()


def test_stub_describes_the_compiled_module():
    assert stub_problems(STUB.read_text(), core) == []


def test_stub_covers_every_name_the_package_re_exports():
    exported = {
        getattr(obj, "__name__", name)
        for name, obj in vars(anneal).items()
        if getattr(obj, "__module__", None) == "anneal._core"
    }
    functions, classes, _ = _stub_surface(STUB.read_text())
    assert exported <= set(functions) | set(classes)


@pytest.mark.parametrize(
    "edit, fragment",
    [
        (("def run_hmc(", "def run_hmc_gone("), "run_hmc is in the module but not in the stub"),
        (("def energy_hits(", "def energy_hits_extra(\n    x: float,\n) -> bool: ...\ndef energy_hits("), "energy_hits_extra is in the stub"),
        (("    seed: int = 42,\n    x0: _Vector | None = None,\n    max_evals: int | None = None,\n) -> History: ...\ndef run_qmc",
          "    seed: int = 42,\n    x0: _Vector | None = None,\n    max_evals: int | None = None,\n    extra: int = 0,\n) -> History: ...\ndef run_qmc"),
         "stub arguments"),
        (("    local_step: str | None = None,\n    ring_moves", "    local_step: str | None,\n    ring_moves"), "cluster_search: stub arguments"),
        (("    *,\n    ras: bool = False,", "    ras: bool = False,"), "cluster_search: stub arguments"),
        (("    @property\n    def sigma(self) -> float: ...", "    def sigma(self) -> float: ..."), "Boltzmann.sigma: stub says method"),
        (("    @staticmethod\n    def derived(n: int) -> Config: ...", "    def derived(n: int) -> Config: ..."), "Config.derived: stub says method"),
        (("class Ledger:\n    def __new__(cls, budget: int) -> Self: ...", "class Ledger:"), "Ledger() is constructible"),
        (("class EpochLine:", "class EpochLine:\n    def __new__(cls, epoch: int) -> Self: ..."), "EpochLine(): stub arguments"),
        (('    "Gsa",\n', ""), "__all__ differs"),
        (("    def replace(self, **fields: Any) -> Config: ...", "    def replace(self, fields: Any) -> Config: ..."), "Config.replace: stub arguments"),
        (("class BoundsError(InvalidArgumentError)", "class BoundsError(AnnealError)"), "BoundsError: stub bases"),
        (("    @property\n    def remaining(self) -> int: ...\n", ""), "Ledger.remaining is in the module"),
    ],
)
def test_the_comparison_catches_drift(edit, fragment):
    old, new = edit
    text = STUB.read_text()
    assert old in text, old
    problems = stub_problems(text.replace(old, new, 1), core)
    assert any(fragment in problem for problem in problems), problems


def test_stubtest_agrees():
    pytest.importorskip("mypy.stubtest")
    result = subprocess.run(
        [sys.executable, "-m", "mypy.stubtest", "anneal._core"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
