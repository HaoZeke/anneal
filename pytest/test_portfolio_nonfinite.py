"""A noisy portfolio run whose every value is not finite ends within its budget."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import anneal

BUDGET = 60


def _run(value, start, replicas):
    calls = []

    def objective(x):
        assert x.shape == (1,)
        assert 0.0 <= x[0] <= 1.0
        calls.append(x.copy())
        return value

    result = anneal.global_optimize(
        objective,
        np.zeros(1),
        np.ones(1),
        BUDGET,
        seed=0,
        x0=np.array([0.5]) if start else None,
        noise_sigma=1e-3,
        replicas=replicas,
    )
    assert 0 < result["n_evals"] == len(calls) <= BUDGET
    assert result["n_grads"] == 0
    assert result["best_val"] == np.inf
    assert 0.0 <= result["best_pos"][0] <= 1.0
    print(json.dumps({"n_evals": result["n_evals"]}))


@pytest.mark.parametrize("replicas", [1, 3])
@pytest.mark.parametrize("start", [True, False])
@pytest.mark.parametrize("value", ["inf", "nan"])
def test_noisy_run_ends_when_no_value_is_finite(value, start, replicas):
    # A run that never returns cannot be stopped inside this process, so
    # each runs in its own.
    script = str(Path(__file__).resolve())
    run = subprocess.run(
        [sys.executable, script, value, str(int(start)), str(replicas)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert json.loads(run.stdout)["n_evals"] > 0


if __name__ == "__main__":
    _run(float(sys.argv[1]), bool(int(sys.argv[2])), int(sys.argv[3]))
