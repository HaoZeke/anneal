"""SOTA validation: the Tsallis (GenSA) visiting kernel implemented in the Rust
``TsallisVisit`` matches SciPy ``dual_annealing``'s ``VisitingDistribution``.

The Rust kernel and the NumPy reference below both implement the Schuur/Xiang
transform per coordinate,

    dx = sigma(T, q_v) * x / |y|^{(q_v-1)/(3-q_v)},   x, y ~ N(0, 1),

with sigma built from the SciPy ``visit_fn`` Gamma-function constants. This test
confirms the reference matches SciPy's own ``visit_fn`` via a two-sample KS test
over the practical range of q_v (default 2.62). The Rust port is separately
pinned to these statistics in ``tests/tsallis_visit_marginal.rs``.
"""

import numpy as np
import pytest

scipy_da = pytest.importorskip("scipy.optimize._dual_annealing")
from scipy.special import gammaln  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402


def schuur_visit(qv, temperature, n, rng):
    """NumPy mirror of ``TsallisVisit::propose`` (one coordinate per sample)."""
    factor2 = (qv - 1.0) ** (4.0 - qv)
    factor3 = 2.0 ** ((2.0 - qv) / (qv - 1.0))
    factor4_p = np.sqrt(np.pi) * factor2 / (factor3 * (3.0 - qv))
    factor5 = 1.0 / (qv - 1.0) - 0.5
    factor6 = np.exp(gammaln(factor5))  # = Gamma(factor5), Euler reflection
    factor4 = factor4_p * temperature ** (1.0 / (qv - 1.0))
    expo = (qv - 1.0) / (3.0 - qv)
    sigma = (factor4 / factor6) ** expo
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    return sigma * x / np.abs(y) ** expo


@pytest.mark.parametrize("qv", [1.5, 2.0, 2.62, 2.9])
@pytest.mark.parametrize("temperature", [1.0, 5.0])
def test_matches_scipy_visit_fn(qv, temperature):
    VisitingDistribution = scipy_da.VisitingDistribution
    vd = VisitingDistribution(
        np.array([-1e30]), np.array([1e30]), qv, np.random.default_rng(0)
    )
    scipy_samples = np.concatenate([vd.visit_fn(temperature, 2000) for _ in range(50)])
    mine = schuur_visit(qv, temperature, scipy_samples.size, np.random.default_rng(1))
    ks = ks_2samp(scipy_samples, mine)
    assert ks.statistic < 0.01, (
        f"q_v={qv} T={temperature}: KS={ks.statistic:.4f} (p={ks.pvalue:.3f}) "
        "does not match SciPy visit_fn"
    )
