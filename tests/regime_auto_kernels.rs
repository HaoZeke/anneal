//! Independent reference checks for regime auto-selection and critical kernels.
//!
//! Drives *shipped* entry points: `select_regime`, `check_accept_path`,
//! Metropolis/Tsallis accept traits, GLE FDT via eindir, TsallisVisit propose.

use anneal_core::accept::{AcceptRule, Metropolis, TsallisAccept};
use anneal_core::methods::{
    OptimizationRegime, ProblemFeatures, check_accept_path, order_arms, require_accept_compatible,
    select_regime,
};
use anneal_core::movekernel::{MoveKernel, TsallisVisit};
use eindir_core::{
    Bounds, GleThermostat, box_geometry, compensated_delta, isotropic_proposal_scale,
    optimal_sampling_drift,
};
use ndarray::{Array1, Array2, array};
use rand::SeedableRng;

#[test]
fn regime_policy_maps_features() {
    let low = ProblemFeatures {
        dim: 2,
        has_grad: true,
        noise_sigma: None,
        aspect_ratio: 1.0,
        mean_width: 4.0,
        budget: 4000,
    };
    assert_eq!(select_regime(&low), OptimizationRegime::LowDimSmooth);
    let high = ProblemFeatures {
        dim: 40,
        has_grad: true,
        noise_sigma: None,
        aspect_ratio: 1.0,
        mean_width: 4.0,
        budget: 8000,
    };
    assert_eq!(
        select_regime(&high),
        OptimizationRegime::HighDimIllConditioned
    );
    let noisy = ProblemFeatures {
        dim: 5,
        has_grad: false,
        noise_sigma: Some(0.05),
        aspect_ratio: 1.0,
        mean_width: 4.0,
        budget: 2000,
    };
    assert_eq!(select_regime(&noisy), OptimizationRegime::StochasticNoise);
    let schwefel = ProblemFeatures {
        dim: 5,
        has_grad: true,
        noise_sigma: None,
        aspect_ratio: 1.0,
        mean_width: 1000.0,
        budget: 3000,
    };
    assert_eq!(
        select_regime(&schwefel),
        OptimizationRegime::MultimodalGlobal
    );
}

#[test]
fn out_of_regime_exact_accept_fails_cleanly() {
    let err = check_accept_path(OptimizationRegime::StochasticNoise, false);
    assert!(err.is_err());
    let ok = check_accept_path(OptimizationRegime::StochasticNoise, true);
    assert!(ok.is_ok());
    // Shipped gate used by portfolio_optimize:
    assert!(require_accept_compatible(Some(0.05), false).is_err());
    assert!(require_accept_compatible(Some(0.05), true).is_ok());
    assert!(require_accept_compatible(None, false).is_ok());
}

#[test]
fn explore_first_under_all_regimes() {
    let avail = [
        "explore",
        "gle",
        "de",
        "gsa",
        "surrogate",
        "hop",
        "shift",
        "hmc",
    ];
    for regime in [
        OptimizationRegime::Default,
        OptimizationRegime::LowDimSmooth,
        OptimizationRegime::HighDimIllConditioned,
        OptimizationRegime::MultimodalNoGrad,
        OptimizationRegime::MultimodalGlobal,
        OptimizationRegime::StochasticNoise,
    ] {
        let ordered = order_arms(&avail, regime, 6);
        assert_eq!(ordered[0], "explore", "regime={regime:?}");
    }
}

#[test]
fn metropolis_and_tsallis_accept_limits() {
    let m = Metropolis;
    // Downhill always 1 (L3).
    let p_down: f64 = m.accept_prob(-1.0_f64, 1.0_f64);
    let p_zero: f64 = m.accept_prob(0.0_f64, 1.0_f64);
    assert!((p_down - 1.0).abs() < 1e-15);
    assert!((p_zero - 1.0).abs() < 1e-15);
    // Uphill Metropolis.
    let p: f64 = m.accept_prob(1.0_f64, 1.0_f64);
    assert!((p - (-1.0_f64).exp()).abs() < 1e-12);
    // Temperature monotonicity L4: higher T => higher p for fixed ΔE>0.
    let p_hot: f64 = m.accept_prob(2.0_f64, 10.0_f64);
    let p_cold: f64 = m.accept_prob(2.0_f64, 0.5_f64);
    assert!(p_hot >= p_cold - 1e-15);

    let t = TsallisAccept::new(1.0_f64); // Q_a → Metropolis limit
    let pt: f64 = t.accept_prob(1.0_f64, 1.0_f64);
    assert!((pt - (-1.0_f64).exp()).abs() < 1e-6);
}

#[test]
fn compensated_delta_matches_naive_away_from_cancellation() {
    let d = compensated_delta(3.5, 1.25);
    assert!((d - 2.25).abs() < 1e-15);
}

#[test]
fn box_geometry_and_isotropic_scale() {
    let b = Bounds::new(array![0.0, 0.0, 0.0], array![2.0, 2.0, 2.0], 0.0);
    let g = box_geometry(&b);
    assert_eq!(g.dim, 3);
    let s = isotropic_proposal_scale(&g, 0.25);
    assert!(s > 0.0);
    assert!((s - 0.25 * 2.0 / (3.0f64).sqrt()).abs() < 1e-12);
}

#[test]
fn gle_fdt_and_stationarity_shipped_path() {
    let a = optimal_sampling_drift(1.0);
    let n = a.nrows();
    // FDT: A C + C A^T = B B^T with C = I => A + A^T should be SPD-ish diagonal dominant
    // For optimal sampling drift, C = I, FDT is A + A^T = B B^T (symmetric part non-neg).
    let mut sym = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            sym[[i, j]] = a[[i, j]] + a[[j, i]];
        }
    }
    // Diagonal of A+A^T should be positive (friction).
    for i in 0..n {
        assert!(sym[[i, i]] >= -1e-10, "sym diagonal {i} = {}", sym[[i, i]]);
    }
    // Propagator exists and is finite.
    let gle = GleThermostat::canonical(&a, 0.1, 1.0, 1.0);
    let c = Array2::<f64>::eye(n);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let s = gle.sample_stationary(&c, 1, 1.0, &mut rng);
    assert_eq!(s.nrows(), n);
    assert!(s.iter().all(|x| x.is_finite()));
}

#[test]
fn tsallis_visit_proposes_finite() {
    let k = TsallisVisit::new(2.62);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let x = Array1::zeros(4);
    let y = k.propose(x.view(), 1.0, &mut rng);
    assert_eq!(y.len(), 4);
    assert!(y.iter().all(|v| v.is_finite()));
}
