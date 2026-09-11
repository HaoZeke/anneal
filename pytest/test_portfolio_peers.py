"""The public portfolio wrapper keeps scalar-only cooperation in native code."""

import json
from pathlib import Path
import subprocess
import sys

import anneal
import numpy as np
import pytest


def _probe(analytic):
    observations = []
    gradients = []
    optimum = 0.271 + 0.01 * np.arange(8)

    def value(x):
        d = x - optimum
        return float(np.sum(d * d + 10.0 * (1.0 - np.cos(2.0 * np.pi * d))))

    def objective(x):
        assert x.shape == (8,)
        assert np.all(np.isfinite(x)) and np.all(np.abs(x) <= 2.0)
        result = value(x)
        observations.append((x.copy(), result))
        return result

    def gradient(x):
        gradients.append(x.copy())
        d = x - optimum
        return 2.0 * d + 20.0 * np.pi * np.sin(2.0 * np.pi * d)

    result = anneal.global_optimize(
        objective,
        np.full(8, -2.0),
        np.full(8, 2.0),
        budget=8003,
        seed=17,
        grad_fn=gradient if analytic else None,
        replicas=4,
        coverage_radius=0.8,
    )
    assert result["n_evals"] == len(observations)
    assert result["n_grads"] == len(gradients)
    assert result["n_evals"] + result["n_grads"] <= 8003
    if analytic:
        assert gradients
    else:
        assert len(observations) == 8003
        assert not gradients
    assert result["best_val"] == min(v for _, v in observations)
    assert result["best_val"] == value(result["best_pos"])
    assert len(result["replicas"]) == 4
    assert sum(r["n_evals"] for r in result["replicas"]) == result["n_evals"]
    assert sum(r["n_grads"] for r in result["replicas"]) == result["n_grads"]
    assert result["coverage_published_samples"] > 0
    assert result["coverage_applied_foreign_samples"] > 0
    assert result["coverage_repelled_proposals"] > 0
    print(json.dumps({"n_evals": result["n_evals"], "n_grads": result["n_grads"]}))


@pytest.mark.parametrize("analytic", [False, True])
def test_public_portfolio_peers_release_the_interpreter_lock(analytic):
    probe = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(int(analytic))],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    result = json.loads(probe.stdout)
    assert result["n_evals"] > 0
    assert (result["n_grads"] > 0) == analytic


def test_private_portfolio_peers_report_no_foreign_samples():
    calls = []

    def objective(x):
        calls.append(x.copy())
        return float(np.sum((x - 0.375) ** 2))

    result = anneal.global_optimize(
        objective, np.full(3, -2.0), np.full(3, 2.0), budget=257,
        replicas=4, coverage_shared=False,
    )
    assert result["n_evals"] == len(calls) == 257
    assert result["n_grads"] == 0
    assert result["coverage_published_samples"] == 0
    assert result["coverage_applied_foreign_samples"] == 0
    assert result["coverage_repelled_proposals"] == 0


@pytest.mark.parametrize("settings", [
    {"replicas": 0},
    {"replicas": 4, "coverage_radius": 0.0},
    {"replicas": 4, "coverage_radius": float("nan")},
])
def test_invalid_portfolio_peer_settings_cost_no_callbacks(settings):
    calls = []

    def objective(x):
        calls.append(x.copy())
        return 0.0

    with pytest.raises(ValueError):
        anneal.global_optimize(objective, np.zeros(2), np.ones(2), budget=10, **settings)
    assert not calls


if __name__ == "__main__":
    _probe(bool(int(sys.argv[1])))
