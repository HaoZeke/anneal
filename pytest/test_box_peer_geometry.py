"""Distinguish occupied-chain crowding from proposed-position crowding."""

import anneal
import pytest


@pytest.mark.parametrize("entry", ["ensemble_optimize", "minimize"])
@pytest.mark.parametrize("shared", [False, True])
def test_scalar_entries_report_peer_geometry_without_user_derivatives(entry, shared):
    observed = []

    def value_only(x):
        assert len(x) == 1 and -1.0 <= x[0] <= 1.0
        observed.append(float(x[0]))
        return 0.5 * x[0] * x[0]

    options = dict(budget=512, replicas=2, seed=7, history="none", coverage_shared=shared)
    if entry == "minimize":
        result = anneal.minimize(value_only, [0.0], [(-1.0, 1.0)], jac=None, **options)
        statistics = result.diagnostics
        assert result.nfev == len(observed)
        assert result.njev == 0
    else:
        statistics = anneal.ensemble_optimize(
            value_only, [-1.0], [1.0], x0=[0.0], grad_fn=None, **options
        )
    assert statistics["n_evals"] == len(observed) == 512
    assert statistics["n_grads"] == 0
    assert statistics["history_observations"] == 0
    checks = statistics["coverage_sample_peer_checks"]
    anchors = statistics["coverage_sample_anchor_overlaps"]
    anchor_only = statistics["coverage_sample_anchor_only_overlaps"]
    assert 0 <= anchor_only <= anchors <= checks
    assert anchor_only + statistics["coverage_sample_overlaps"] <= checks
    if shared:
        assert checks == statistics["hops"]
        assert anchors > 0 and anchor_only > 0
    else:
        assert checks == anchors == anchor_only == 0
