# Copyright (c) anneal developers
"""The Errors section of the Python API page agrees with the module.

The page lists the exception classes, says which one each is a kind of, and
gives a table of the errors that each call raises. This file reads that text
and compares it with the classes of ``anneal._core`` and with the docstrings
in the stubs, which ``test_public_names.py`` holds to the runtime.
"""

import re
from pathlib import Path

import pytest
from test_public_names import errors_named, stub_docstrings, STUB

import anneal._core as core

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "orgmode" / "reference" / "python-api.org"
BULLET = re.compile(r"^- =(\w+)= :: (.*(?:\n  .*)*)", re.M)
ROW = re.compile(r"^\| =([\w.]+)=[ ,=\w.]*\|(.*)\|\s*$", re.M)


def errors_section(text):
    start = text.index("\n* Errors\n")
    end = text.index("\n* ", start + 1)
    return text[start:end]


def exceptions():
    return {
        name: obj
        for name, obj in vars(core).items()
        if isinstance(obj, type) and issubclass(obj, core.AnnealError)
    }


def page_problems(text):
    """Every way the Errors section differs from the module, as sentences."""
    section = errors_section(text)
    problems = []
    classes = exceptions()

    listed = {name: body for name, body in BULLET.findall(section)}
    for name in sorted(set(classes) - set(listed)):
        problems.append(f"{name} is an exception of the module and the page does not list it")
    for name in sorted(set(listed) - set(classes)):
        problems.append(f"the page lists {name}, which is not an exception of the module")
    for name, body in listed.items():
        kind = re.search(r"a kind of =(\w+)=", " ".join(body.split()))
        if name in classes and kind:
            parent = classes[name].__bases__[0].__name__
            if kind.group(1) != parent:
                problems.append(f"{name}: the page says a kind of {kind.group(1)}, it is {parent}")
        parent = classes[name].__bases__[0] if name in classes else None
        if parent is not None and name != "AnnealError" and not kind and parent is not core.AnnealError:
            problems.append(f"{name} is a kind of {parent.__name__} and the page does not say so")

    docs = stub_docstrings(STUB.read_text())
    table = {}
    for call, raises in ROW.findall(section):
        table[call.removeprefix("anneal._core.")] = set(re.findall(r"=(\w+)=", raises))
    for call, named in table.items():
        doc = docs.get(call)
        if doc is None:
            problems.append(f"the table lists {call}, which has no docstring in the stub")
        elif errors_named(doc) != named:
            problems.append(
                f"{call}: the table says {sorted(named)}, the stub {sorted(errors_named(doc))}"
            )
    for call, doc in docs.items():
        if call in classes or call in table:
            continue
        if errors_named(doc):
            problems.append(f"{call} names {sorted(errors_named(doc))} and the table omits it")
    return problems


def test_the_errors_section_agrees_with_the_module():
    assert page_problems(PAGE.read_text()) == []


def test_the_table_has_a_row_for_every_call_that_can_run_out_of_memory():
    text = errors_section(PAGE.read_text())
    docs = stub_docstrings(STUB.read_text())
    raising = {call for call, doc in docs.items() if "MemoryError" in errors_named(doc)}
    assert raising
    rows = {call.removeprefix("anneal._core.") for call, _ in ROW.findall(text)}
    assert raising <= rows


@pytest.mark.parametrize(
    "old, new, fragment",
    [
        ("- =FeatureUnavailableError= ::", "- =FeatureUnavailable= ::", "FeatureUnavailableError is an exception"),
        ("- =BoundsError= :: a kind of =InvalidArgumentError=", "- =BoundsError= :: a kind of =AnnealError=", "BoundsError: the page says a kind of AnnealError"),
        ("| =cluster_search=              | =InvalidArgumentError=, =MemoryError=,", "| =cluster_search=              | =InvalidArgumentError=,", "cluster_search: the table says"),
        ("| =Config.for_tip4p=            |", "| =Config.for_nothing=          |", "Config.for_nothing, which has no docstring"),
        ("| =anneal._core.make_bounds=    | =BoundsError=                                               |\n", "", "make_bounds names ['BoundsError'] and the table omits it"),
    ],
)
def test_the_comparison_catches_a_page_that_drifts(old, new, fragment):
    text = PAGE.read_text()
    assert old in text, old
    problems = page_problems(text.replace(old, new, 1))
    assert any(fragment in problem for problem in problems), problems
