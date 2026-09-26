import numpy as np
import pytest

from experiments.scripts.demo_bgsa import (
    _log_pseudo_marginal_weight,
    continuous_time_tempering,
    _laplace_third_moment_correction,
    trajectory_inla_diagnostic,
    metad_gamma_from_qv,
    pmsa_metad,
    select_bgsa_driver,
    _pilot_t_hot_from_acceptance,
    smc_pt_log_z_estimator,
)


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


def test_continuous_tempering_counts_each_objective_evaluation_once():
    _, calls, beta_history = continuous_time_tempering(
        seed=2,
        n_epochs=3,
        k_inner=2,
        t_min=0.5,
        t_max=2.0,
        q_v=1.1,
    )
    assert calls == 1 + 3 * 2
    assert len(beta_history) == 3


def test_pt_evidence_uses_paired_importance_weights_not_swap_acceptance():
    estimate, _, _, _ = smc_pt_log_z_estimator(
        [0.0, np.log(2.0)],
        [0.0, 0.0],
    )
    assert np.isclose(estimate, np.log(1.5))


def test_pt_evidence_rejects_unpaired_weights():
    with pytest.raises(ValueError, match="equal lengths"):
        smc_pt_log_z_estimator([0.0], [0.0, 1.0])


def test_metad_bias_factor_is_independent_of_tsallis_visiting_index():
    assert metad_gamma_from_qv(1.1) == metad_gamma_from_qv(2.9)
    assert metad_gamma_from_qv(2.9) > 1.0


def test_driver_selector_uses_gradient_sensitivity_ratio():
    assert select_bgsa_driver(
        {"grad_sens": 0.8, "sigma_sens": 0.1, "best_val_cv": 0.2}, 1.1
    ) == "bgsa"
    assert select_bgsa_driver(
        {"grad_sens": 0.2, "sigma_sens": 0.1, "best_val_cv": 1.5}, 1.1
    ) == "bgsa_pt_metad"
    assert select_bgsa_driver(
        {"grad_sens": 0.2, "sigma_sens": 0.1, "best_val_cv": 0.5}, 1.1
    ) == "bgsa_metad"


def test_hot_temperature_inverts_empirical_swap_acceptance_target():
    observations = [{"best_val": value} for value in (0.0, 0.0, 10.0, 10.0)]
    hot = _pilot_t_hot_from_acceptance(observations, 1.0)
    predicted = np.exp((1.0 - 1.0 / hot) * (0.0 - 10.0))
    assert np.isclose(predicted, 0.234, rtol=1e-6)


def test_tierney_kadane_correction_has_the_cubic_posterior_coefficient():
    cubic = 0.8

    def nll(params):
        x = params[0]
        return 0.5 * x * x + cubic * x**3 / 6.0

    shift = _laplace_third_moment_correction(nll, [0.0], h=1e-3)[0]
    assert np.isclose(shift, -cubic / 2.0, rtol=1e-4)


def test_trajectory_diagnostic_does_not_flag_stationary_ar_endpoints():
    rng = np.random.default_rng(4)
    trajectory = np.zeros(2000)
    for i in range(1, len(trajectory)):
        trajectory[i] = 0.8 * trajectory[i - 1] + rng.normal()
    phi, _, sigma_t, flags = trajectory_inla_diagnostic(trajectory)
    assert 0.7 < phi < 0.9
    assert np.allclose(sigma_t, sigma_t[0])
    assert flags.sum() < 100


def test_trajectory_diagnostic_flags_a_sustained_mean_shift():
    rng = np.random.default_rng(5)
    trajectory = rng.normal(size=1000)
    trajectory[500:] += 8.0
    _, _, _, flags = trajectory_inla_diagnostic(trajectory)
    assert flags[500:].sum() > 300
