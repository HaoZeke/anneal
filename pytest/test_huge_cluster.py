# Copyright (c) anneal developers
"""A size the machine cannot hold is an error, not an abort or a panic.

``cluster_search`` allocates vectors of ``3n`` or ``6n`` numbers. The old
code aborted the process when the allocator said no, and panicked when the
length overflowed a machine word. The checks set no largest ``n``: they ask
the allocator and report its answer. Each case runs in a child process,
because an abort would take the test run with it.
"""

import json
import subprocess
import sys

import pytest

SCRIPT = """
import json
import numpy as np
import anneal

cases = json.loads(__import__("sys").argv[1])
out = []
for case in cases:
    n = case["n"]
    kwargs = case["kwargs"]
    try:
        if case["call"] == "search":
            anneal.cluster_search(
                lambda x: float(np.sum(np.asarray(x) ** 2)),
                lambda x: 2.0 * np.asarray(x),
                n,
                5,
                seed=1,
                **kwargs,
            )
        else:
            getattr(anneal.Config, case["call"])(n)
        out.append("returned")
    except BaseException as err:
        out.append(type(err).__name__)
print(json.dumps(out))
"""

USIZE_MAX = 2**64 - 1
SEARCH_KWARGS = [
    {},
    {"rigid": True},
    {"ras": True},
    {"communicating": True},
    {"derived": True},
    {"recommended": False},
]


def run_cases(cases):
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT, json.dumps(cases)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, f"the process died with {done.returncode}: {done.stderr[-400:]}"
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("n", [10**13, 10**18, 2**61, 2**63])
def test_a_search_over_more_points_than_memory_holds_raises_memory_error(n):
    cases = [{"call": "search", "n": n, "kwargs": kwargs} for kwargs in SEARCH_KWARGS]
    seen = run_cases(cases)
    assert seen == ["MemoryError" if n < 2**63 else "InvalidArgumentError"] * len(cases), seen


def test_a_state_length_that_overflows_raises_a_bad_argument():
    cases = [{"call": "search", "n": USIZE_MAX, "kwargs": kwargs} for kwargs in SEARCH_KWARGS]
    assert run_cases(cases) == ["InvalidArgumentError"] * len(cases)


@pytest.mark.parametrize("preset", ["for_tip4p", "recommended_tip4p", "communicating_tip4p"])
def test_a_water_preset_for_more_waters_than_memory_holds_raises(preset):
    cases = [
        {"call": preset, "n": 10**13, "kwargs": {}},
        {"call": preset, "n": 10**18, "kwargs": {}},
        {"call": preset, "n": USIZE_MAX, "kwargs": {}},
    ]
    assert run_cases(cases) == ["MemoryError", "MemoryError", "InvalidArgumentError"]


def test_a_size_the_machine_holds_is_still_built():
    cases = [
        {"call": "for_tip4p", "n": 64, "kwargs": {}},
        {"call": "recommended", "n": 10**6, "kwargs": {}},
        {"call": "recommended", "n": USIZE_MAX, "kwargs": {}},
        {"call": "search", "n": 6, "kwargs": {"target": 0.0}},
    ]
    assert run_cases(cases) == ["returned"] * len(cases)
