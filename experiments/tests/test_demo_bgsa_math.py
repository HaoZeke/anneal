import numpy as np
import pytest

from experiments.scripts.demo_bgsa import _log_pseudo_marginal_weight, pmsa_metad


def test_pseudo_marginal_weight_is_unbiased_for_gaussian_energy_noise():
    true_energy = 1.3
    temperature = 0.9
    sigma = 0.4
    n_eval = 4
    rng = np.random.default_rng(19)
    means = true_energy + rng.normal(0.0, sigma / np.sqrt(n_eval), size=100_000)
    estimates = np.exp(
        [
            _log_pseudo_marginal_weight(mean, temperature, sigma, n_eval)
            for mean in means
        ]
    )
    target = np.exp(-true_energy / temperature)
    assert abs(estimates.mean() - target) / target < 0.02


def test_pseudo_marginal_driver_rejects_tsallis_acceptance():
    noisy = lambda x: float(np.sum(x * x))
    with pytest.raises(ValueError, match="q_a == 1"):
        pmsa_metad(
            1,
            1,
            1,
            1.0,
            0.1,
            noisy,
            0.1,
            q_v=1.1,
            q_a=1.2,
            n_eval_per_step=1,
        )
