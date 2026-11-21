# Copyright (c) anneal developers
"""``anneal.__all__``, the stubs and the documented errors agree with the runtime.

``test_stubs.py`` compares what the stubs declare with the compiled module.
This file compares what they say: the exception classes the package exports,
the docstrings that name an error, and the calls that the stubs say raise
``MemoryError`` or ``InvalidArgumentError``, each of which is run in a child
process (a call that aborts takes the child and not the test run).
"""

import ast
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import anneal
import anneal._core as core

STUB = Path(anneal.__file__).parent / "_core.pyi"
ERROR_NAME = re.compile(r"\b[A-Z]\w*Error\b")
USIZE_MAX = 2**64 - 1


def _exceptions(module):
    return {
        name: obj
        for name, obj in vars(module).items()
        if isinstance(obj, type) and issubclass(obj, core.AnnealError)
    }


def stub_docstrings(text):
    """The docstring of every function, class and method in the stub, by dotted name."""
    found = {}
    for node in ast.parse(text).body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue
        if doc := ast.get_docstring(node):
            found[node.name] = doc
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and (doc := ast.get_docstring(item)):
                    found[f"{node.name}.{item.name}"] = doc
    return found


def _runtime_doc(dotted):
    obj = core
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return inspect.cleandoc(obj.__doc__ or "")


def errors_named(doc):
    return set(ERROR_NAME.findall(doc))


def _squash(doc):
    return " ".join(doc.split())


def doc_problems(text):
    """Every way a stub docstring disagrees with the runtime's, as sentences."""
    problems = []
    exceptions = _exceptions(core)
    for dotted, doc in stub_docstrings(text).items():
        runtime = _runtime_doc(dotted)
        if not runtime:
            problems.append(f"{dotted} has a docstring in the stub and none in the module")
        elif dotted in exceptions:
            if _squash(doc) != _squash(runtime):
                problems.append(f"{dotted}: the stub says {doc!r}, the module {runtime!r}")
        elif errors_named(doc) != errors_named(runtime):
            problems.append(
                f"{dotted}: the stub names {sorted(errors_named(doc))}, "
                f"the module {sorted(errors_named(runtime))}"
            )
    for name in sorted(exceptions):
        if name not in stub_docstrings(text):
            problems.append(f"{name} is an exception of the module with no docstring in the stub")
    return problems


def _config_presets(text):
    """Static methods of ``Config`` in the stub that take ``n`` and nothing else."""
    names = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    args = [a.arg for a in item.args.args]
                    if args == ["n"]:
                        names.append(item.name)
    return names


PROBE = """
import json
import sys

import anneal
import anneal._core as core

names = json.loads(sys.argv[1])
out = {}
for name in names:
    seen = []
    for n in (10**13, 2**64 - 1):
        try:
            if name == "cluster_search":
                core.cluster_search(lambda x: 0.0, lambda x: x, n, 5)
            else:
                getattr(core.Config, name)(n)
            seen.append("returned")
        except BaseException as err:
            seen.append(type(err).__name__)
    out[name] = seen
print(json.dumps(out))
"""


def raises_problems(text):
    """Calls whose documented errors differ from the ones they raise."""
    docs = stub_docstrings(text)
    probes = ["cluster_search"] + [f for f in _config_presets(text)]
    done = subprocess.run(
        [sys.executable, "-c", PROBE, json.dumps(probes)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if done.returncode != 0:
        return [f"the probe died with {done.returncode}: {done.stderr[-300:]}"]
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    problems = []
    for name, (huge, overflow) in seen.items():
        doc = docs.get(name) or docs.get(f"Config.{name}", "")
        for error, got in (("MemoryError", huge), ("InvalidArgumentError", overflow)):
            says = error in errors_named(doc)
            raises = got == error
            if says and not raises:
                problems.append(f"{name}: the stub says {error}, the call gave {got}")
            if raises and not says:
                problems.append(f"{name}: the call raises {error} and the stub does not say so")
    return problems


def test_every_name_in_all_resolves_once():
    assert len(anneal.__all__) == len(set(anneal.__all__))
    assert [n for n in anneal.__all__ if not hasattr(anneal, n)] == []


def test_every_exception_of_the_module_is_exported_from_the_package():
    exceptions = _exceptions(core)
    assert {"AnnealError", "InvalidArgumentError", "BoundsError"} <= set(exceptions)
    for name, cls in exceptions.items():
        assert name in anneal.__all__, name
        assert getattr(anneal, name) is cls, name
        assert name in core.__all__, name


def test_everything_the_package_takes_from_the_module_is_in_all():
    taken = {
        name
        for name, obj in vars(anneal).items()
        if not name.startswith("_")
        and getattr(obj, "__module__", None) == "anneal._core"
    }
    assert sorted(taken - set(anneal.__all__)) == []


def test_the_errors_the_stub_documents_are_the_ones_the_module_documents():
    assert doc_problems(STUB.read_text()) == []


def test_the_calls_the_stub_says_raise_do_raise():
    assert raises_problems(STUB.read_text()) == []


def test_memory_error_is_named_where_it_is_raised():
    text = STUB.read_text()
    docs = stub_docstrings(text)
    named = {name for name, doc in docs.items() if "MemoryError" in errors_named(doc)}
    assert {"cluster_search", "Config.for_tip4p", "Config.recommended_tip4p",
            "Config.communicating_tip4p"} <= named


def test_bounds_takes_the_arguments_the_stub_describes():
    stub = {
        node.name: node
        for node in ast.parse(STUB.read_text()).body
        if isinstance(node, ast.ClassDef)
    }["Bounds"]
    init = next(
        item for item in stub.body if isinstance(item, ast.FunctionDef) and item.name == "__new__"
    )
    args = init.args.args[1:]
    defaults = [None] * (len(args) - len(init.args.defaults)) + list(init.args.defaults)
    described = [(a.arg, d is not None) for a, d in zip(args, defaults)]
    actual = [
        (p.name, p.default is not inspect.Parameter.empty)
        for p in inspect.signature(anneal.Bounds).parameters.values()
    ]
    assert described == actual


@pytest.mark.parametrize(
    "edit, fragment",
    [
        (
            ("Raises `MemoryError` when the", "Raises `ValueError` when the"),
            "cluster_search: the call raises MemoryError",
        ),
        (
            (
                "    def for_cluster(n: int) -> Config: ...",
                '    def for_cluster(n: int) -> Config:\n        """Raises `MemoryError`."""',
            ),
            "for_cluster: the stub says MemoryError",
        ),
    ],
)
def test_the_comparison_catches_drift_in_what_the_calls_raise(edit, fragment):
    old, new = edit
    text = STUB.read_text()
    assert old in text, old
    problems = raises_problems(text.replace(old, new, 1))
    assert any(fragment in problem for problem in problems), problems


@pytest.mark.parametrize(
    "edit, fragment",
    [
        (
            ("`MemoryError` when the allocator refuses the two", "`ValueError` when the allocator refuses the two"),
            "the stub names",
        ),
        (
            ("in. Raises `NoFiniteStartError`", "in. Raises `FeatureUnavailableError`"),
            "cluster_search: the stub names",
        ),
        (
            ('"""Base class of every error', '"""Root of every error'),
            "AnnealError: the stub says",
        ),
        (
            ("Raises `BoundsError` for sides", "Raises for sides"),
            "make_bounds: the stub names",
        ),
        (
            (
                '    """The call needs a Cargo feature that this build of `anneal` does not include."""\n',
                "    ...\n",
            ),
            "FeatureUnavailableError is an exception of the module with no docstring",
        ),
    ],
)
def test_the_comparison_catches_drift_in_the_docstrings(edit, fragment):
    old_text, new_text = edit
    text = STUB.read_text()
    assert old_text in text, old_text
    problems = doc_problems(text.replace(old_text, new_text, 1))
    assert any(fragment in problem for problem in problems), problems
