use anneal_core::first_passage::{ExponentialMixture, FirstPassage};

fn synthetic(
    seed: u64,
    n: usize,
    w_fast: f64,
    fast: f64,
    slow: f64,
    budget: f64,
) -> Vec<FirstPassage> {
    // A deterministic linear congruential stream is enough for a fixture.
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut uniform = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let mean = if uniform() < w_fast { fast } else { slow };
            let t = -mean * uniform().ln();
            if t < budget {
                FirstPassage::Hit(t)
            } else {
                FirstPassage::Censored(budget)
            }
        })
        .collect()
}

#[test]
fn a_two_component_fit_recovers_censored_synthetic_data() {
    let data = synthetic(7, 4000, 0.4, 2.0e5, 6.0e6, 4.0e6);
    let fit = ExponentialMixture::fit(&data, 2, 500).unwrap();
    let (fast, slow) = if fit.means[0] < fit.means[1] {
        (0, 1)
    } else {
        (1, 0)
    };
    assert!(
        (fit.weights[fast] - 0.4).abs() < 0.05,
        "weights {:?}",
        fit.weights
    );
    assert!(
        (fit.means[fast] / 2.0e5 - 1.0).abs() < 0.15,
        "means {:?}",
        fit.means
    );
    assert!(
        (fit.means[slow] / 6.0e6 - 1.0).abs() < 0.3,
        "means {:?}",
        fit.means
    );
}

#[test]
fn the_ensemble_probability_is_the_independent_starts_bound() {
    let fit = ExponentialMixture {
        weights: vec![0.5, 0.5],
        means: vec![1.0e5, 1.0e7],
        shifts: vec![0.0, 0.0],
        log_likelihood: 0.0,
        iterations: 0,
    };
    let one = fit.hit_probability(4.0e6);
    let two = fit.ensemble_hit_probability(2, 4.0e6);
    let expected = 1.0 - (1.0 - fit.hit_probability(2.0e6)).powi(2);
    assert!((two - expected).abs() < 1e-12);
    // A mixture of exponentials has a non-increasing hazard, so splitting
    // never loses: with the fast component saturated, many short chains
    // win by sampling more starts.
    assert!(fit.ensemble_hit_probability(48, 4.0e6) > one);
    let (k, _) = fit.best_split(4.0e6, &[1, 2, 4, 8, 16, 48]).unwrap();
    assert_eq!(k, 48);
}

#[test]
fn invalid_inputs_are_refused() {
    assert!(ExponentialMixture::fit(&[], 2, 10).is_err());
    assert!(ExponentialMixture::fit(&[FirstPassage::Hit(1.0)], 2, 10).is_err());
    assert!(
        ExponentialMixture::fit(&[FirstPassage::Hit(-1.0), FirstPassage::Hit(2.0)], 1, 10).is_err()
    );
}

#[test]
fn a_shifted_slow_component_makes_one_long_chain_beat_two_short_ones() {
    // Fast starts fire from the beginning; slow ones need a warm-up of
    // 1.2e6 forces before they can fire at all.
    let mut data = synthetic(11, 3000, 0.1, 1.0e5, 3.0e6, 4.0e6);
    let mut state = 99u64;
    for point in data.iter_mut() {
        if let FirstPassage::Hit(t) = point {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let slow = ((state >> 11) as f64 / (1u64 << 53) as f64) > 0.1;
            if slow {
                let shifted = *t + 1.2e6;
                *point = if shifted < 4.0e6 { FirstPassage::Hit(shifted) } else { FirstPassage::Censored(4.0e6) };
            }
        }
    }
    let grid: Vec<f64> = (0..=10).map(|i| i as f64 * 2.0e5).collect();
    let fit = ExponentialMixture::fit_shifted(&data, &grid, 500).unwrap();
    let shift = fit.shifts[1];
    assert!((shift - 1.2e6).abs() <= 2.0e5, "shift {shift} weights {:?} means {:?}", fit.weights, fit.means);
    let one = fit.ensemble_hit_probability(1, 4.0e6);
    let two = fit.ensemble_hit_probability(2, 4.0e6);
    assert!(one > two, "one chain {one} two chains {two}");
}
