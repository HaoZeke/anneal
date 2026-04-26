//! Proptest sweeps witnessing the IISE-manuscript composition laws on every
//! shipped concrete component.
//!
//! L1 (symmetry): `Neighborhood::is_symmetric` is true for `ContinuousR_n`
//! and `BoxConstrained`; the kernels (`Gaussian`, `Cauchy`, `TsallisVisit`)
//! are translation-invariant and produce zero-mean perturbations.
//!
//! L3 (downhill always accepts): `Metropolis` and `TsallisAccept` return
//! `1.0` for any `delta_e <= 0` regardless of temperature.
//!
//! L4 (monotonicity in `T`): for fixed `delta_e > 0`, `accept_prob(d, t1)
//! <= accept_prob(d, t2)` whenever `0 < t1 < t2`. Cooling schedules
//! (`LogCool`, `ReciprocalCool`, `TsallisCool` with `q_v > 1`) are
//! non-increasing in the epoch counter on `[0, 1000]`.

use anneal_core::accept::{AcceptRule, Metropolis, TsallisAccept};
use anneal_core::cool::{Cooling, LogCool, ReciprocalCool, TsallisCool};
use anneal_core::movekernel::{Cauchy, Gaussian, MoveKernel, TsallisVisit};
use anneal_core::neigh::{BoxConstrained, ContinuousR_n, Neighborhood};

use eindir_core::Bounds;
use ndarray::Array1;
use proptest::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

const SAMPLE_BUDGET: usize = 1024;

// ----- L1: structural symmetry on neighborhoods --------------------------

proptest! {
    #[test]
    fn l1_continuous_r_n_is_symmetric(dim in 1usize..=8) {
        let n = ContinuousR_n::new(dim);
        prop_assert!(<ContinuousR_n as Neighborhood<f64>>::is_symmetric(&n));
    }

    #[test]
    fn l1_box_constrained_is_symmetric(dim in 1usize..=8) {
        let low = Array1::from_elem(dim, -1.0_f64);
        let high = Array1::from_elem(dim, 1.0_f64);
        let n = BoxConstrained::new(Bounds::new(low, high, 1e-12));
        prop_assert!(n.is_symmetric());
    }

    #[test]
    fn l1_continuous_r_n_contains_is_dim_check(dim in 1usize..=8, seed in any::<u64>()) {
        let n = ContinuousR_n::new(dim);
        let mut rng = StdRng::seed_from_u64(seed);
        let i = Array1::from_iter((0..dim).map(|_| (rng.random::<f64>() - 0.5) * 10.0));
        let j = Array1::from_iter((0..dim).map(|_| (rng.random::<f64>() - 0.5) * 10.0));
        prop_assert!(<ContinuousR_n as Neighborhood<f64>>::contains(&n, i.view(), j.view()));
    }
}

// ----- L1 (kernel symmetry, structural): zero-mean perturbations ---------

#[test]
fn l1_gaussian_perturbation_is_zero_mean() {
    let k = Gaussian::new(1.0);
    let i = Array1::from_elem(3, 0.0_f64);
    let mut rng = StdRng::seed_from_u64(0);
    let mut acc = Array1::<f64>::zeros(3);
    for _ in 0..SAMPLE_BUDGET {
        let p = k.propose(i.view(), 1.0, &mut rng);
        acc = &acc + &p;
    }
    let mean = acc.mapv(|x| x / SAMPLE_BUDGET as f64);
    // 4 sigma / sqrt(N) tolerance.
    let tol = 4.0 / (SAMPLE_BUDGET as f64).sqrt();
    for &m in mean.iter() {
        assert!(
            m.abs() < tol,
            "Gaussian perturbation mean {} out of tolerance {}",
            m,
            tol
        );
    }
}

#[test]
fn l1_cauchy_perturbation_is_dim_preserving() {
    let k = Cauchy::new(0.5);
    let i = Array1::from_elem(4, 1.0_f64);
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..SAMPLE_BUDGET {
        let p = k.propose(i.view(), 1.0, &mut rng);
        assert_eq!(p.len(), i.len());
        for v in p.iter() {
            assert!(v.is_finite(), "Cauchy proposal produced non-finite value");
        }
    }
}

#[test]
fn l1_tsallis_visit_dim_preserving() {
    let k = TsallisVisit::new(2.62);
    let i = Array1::from_elem(2, 0.0_f64);
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..256 {
        let p = k.propose(i.view(), 1.0, &mut rng);
        assert_eq!(p.len(), i.len());
        for v in p.iter() {
            assert!(
                v.is_finite(),
                "Tsallis-visit proposal produced non-finite value"
            );
        }
    }
}

// ----- L3: downhill always accepts ---------------------------------------

proptest! {
    #[test]
    fn l3_metropolis_downhill_accepts(d in -1e6_f64..=0.0, t in 1e-6_f64..1e6) {
        let p = Metropolis.accept_prob(d, t);
        prop_assert_eq!(p, 1.0);
    }

    #[test]
    fn l3_tsallis_downhill_accepts(
        d in -1e6_f64..=0.0,
        t in 1e-6_f64..1e6,
        q_a in 0.5_f64..2.5,
    ) {
        let acc = TsallisAccept::new(q_a);
        let p = acc.accept_prob(d, t);
        prop_assert_eq!(p, 1.0);
    }
}

// ----- L4: T -> p non-decreasing for fixed delta_e > 0 -------------------

proptest! {
    #[test]
    fn l4_metropolis_monotone_in_t(
        d in 1e-3_f64..1e3,
        t1 in 1e-3_f64..1e2,
        t2 in 1e-3_f64..1e2,
    ) {
        prop_assume!(t1 < t2);
        let p1 = Metropolis.accept_prob(d, t1);
        let p2 = Metropolis.accept_prob(d, t2);
        prop_assert!(p1 <= p2 + 1e-12, "p1={} p2={}", p1, p2);
    }

    #[test]
    fn l4_tsallis_monotone_in_t_q_a_one(
        d in 1e-3_f64..1e3,
        t1 in 1e-3_f64..1e2,
        t2 in 1e-3_f64..1e2,
    ) {
        // q_a == 1 dispatches to the Metropolis branch; verifies the
        // L'Hopital fallback inherits L4.
        prop_assume!(t1 < t2);
        let acc = TsallisAccept::new(1.0_f64);
        let p1 = acc.accept_prob(d, t1);
        let p2 = acc.accept_prob(d, t2);
        prop_assert!(p1 <= p2 + 1e-12);
    }
}

// ----- L4: cooling non-increasing in epoch -------------------------------

proptest! {
    #[test]
    fn l4_log_cool_monotone(t_init in 1e-2_f64..1e2, k0 in 1.5_f64..1e3) {
        let c = LogCool::new(t_init, k0);
        prop_assert!(c.is_monotone());
        let mut prev = c.temperature(0);
        for k in 1..=1000usize {
            let t = c.temperature(k);
            prop_assert!(t <= prev + 1e-12, "k={k} prev={prev} t={t}");
            prev = t;
        }
    }

    #[test]
    fn l4_reciprocal_cool_monotone(t_init in 1e-2_f64..1e2) {
        let c = ReciprocalCool::new(t_init);
        prop_assert!(c.is_monotone());
        let mut prev = c.temperature(0);
        for k in 1..=1000usize {
            let t = c.temperature(k);
            prop_assert!(t <= prev + 1e-12);
            prev = t;
        }
    }

    #[test]
    fn l4_tsallis_cool_monotone(t_init in 1e-2_f64..1e2, q_v in 1.01_f64..2.99) {
        let c = TsallisCool::new(t_init, q_v);
        prop_assert!(c.is_monotone());
        let mut prev = c.temperature(0);
        for k in 1..=1000usize {
            let t = c.temperature(k);
            prop_assert!(t <= prev + 1e-9, "q_v={q_v} k={k} prev={prev} t={t}");
            prev = t;
        }
    }
}
