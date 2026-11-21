# Copyright (c) anneal developers
"""A rigid-water search ends on an objective that is not water.

A quadratic objective drives every molecule to the origin, where the
molecules lie on top of one another and the hydrogen-bond cycle move has no
direction to turn toward. That once handed NaN to the rotation fold, which
never returned. Each search runs in a child process so that a search that
does not return fails the test instead of hanging the run.
"""

import json
import subprocess
import sys

import pytest

SCRIPT = """
import json, sys
import numpy as np
import anneal

preset, n, budget = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = getattr(anneal.Config, preset)(n)
out = anneal.cluster_search(
    lambda x: float(np.sum(np.asarray(x) ** 2)),
    lambda x: 2.0 * np.asarray(x),
    n,
    budget,
    seed=1,
    config=cfg,
    full=True,
)
print(json.dumps({"hops": out["hops"], "energy": out["best_energy"], "stop": out["stop"]}))
"""

PRESETS = ["for_tip4p", "recommended_tip4p", "communicating_tip4p"]


def search(preset, n, budget, timeout=120):
    return subprocess.run(
        [sys.executable, "-c", SCRIPT, preset, str(n), str(budget)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("n", [3, 5])
def test_a_quadratic_objective_ends_the_search(preset, n):
    done = search(preset, n, 600)
    assert done.returncode == 0, done.stderr[-2000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["hops"] > 0
    assert report["energy"] == report["energy"]
    assert report["stop"] in {"budget", "max_hops"}


@pytest.mark.parametrize("budget", [1, 50, 5000])
def test_the_budget_decides_when_the_search_ends(budget):
    done = search("for_tip4p", 3, budget)
    assert done.returncode == 0, done.stderr[-2000:]
