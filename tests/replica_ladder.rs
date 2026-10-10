//! Every rung of a replica ladder hops at its own temperature.
//!
//! The hot end of a ladder is there to take the rises a cold chain declines,
//! and the swap brings what it finds down to the cold end. A ladder whose
//! temperatures reached only the swap test would adopt what a flat ladder
//! adopts. Counted here as adopted steps that climb more than one well depth,
//! on LJ13 at a temperature where a single chain seldom climbs that far.

use anneal_core::dos::EnergyBias;
use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, optimize, optimize_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};

const BUDGET: usize = 30_000;
const SEEDS: u64 = 8;

fn lj_run_with(cfg: &Config, seed: u64, budget: usize, gradient: bool) -> Outcome {
    lj_run_on(cfg, seed, &mut Ledger::new(budget), gradient)
}

fn lj_run_on(cfg: &Config, seed: u64, ledger: &mut Ledger, gradient: bool) -> Outcome {
    let pot = PairPotential::lennard_jones(cfg.n_points);
    let mut opt = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        opt.forget();
        let (f, xr, _) = opt.minimize_watched(
            x,
            iters,
            |v| {
                if !led.charge() {
                    return None;
                }
                Some(pot.value_and_gradient(v))
            },
            |_, _| true,
        );
        (f, xr)
    };
    if gradient {
        let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
            if !led.charge() {
                return None;
            }
            Some(pot.value_and_gradient(x).1)
        };
        optimize_with_gradient(cfg, ledger, &mut relax, Some(&mut grad), seed)
    } else {
        optimize(cfg, ledger, &mut relax, seed)
    }
}

fn lj_run(cfg: &Config, seed: u64) -> Outcome {
    lj_run_with(cfg, seed, BUDGET, false)
}

/// Adopted steps that climbed more than one well depth, over the seeds.
fn large_rises(base: &Config, ladder_top: f64) -> usize {
    let mut cfg = base.clone();
    cfg.replicas = 2;
    cfg.ladder_top = ladder_top;
    (0..SEEDS)
        .map(|seed| {
            lj_run(&cfg, seed)
                .accepted_transitions
                .iter()
                .filter(|t| t.adopted && t.to_energy - t.from_energy > 1.0)
                .count()
        })
        .sum()
}

fn assert_the_hot_rung_climbs(base: &Config) {
    let flat = large_rises(base, 1.0);
    let hot = large_rises(base, 10.0);
    assert!(
        hot >= 3 * flat.max(10),
        "a rung ten times hotter adopted {hot} rises above a well depth \
         against {flat} on a flat ladder"
    );
}

#[test]
fn a_hot_rung_adopts_more_large_rises_than_a_flat_ladder() {
    let mut cfg = Config::recommended(13);
    assert!(!cfg.budget_window && !cfg.statistical_temperature);
    cfg.temperature = 0.2;
    assert_the_hot_rung_climbs(&cfg);
}

/// The law sets the coldest rung's temperature and the hot rung hops at its
/// ratio times it, so the ladder survives the budget window.
#[test]
fn the_ladder_multiplies_the_budget_window_temperature() {
    let cfg = Config::derived(13);
    assert!(cfg.budget_window);
    assert_the_hot_rung_climbs(&cfg);
}

/// The clamped estimate is the coldest rung's temperature and the hot rung
/// hops at its ratio times it.
///
/// The estimate takes over once the rungs at ratio one have recorded
/// `flat_sweep` hops, and on a hot ladder that is the coldest rung alone. At
/// the default turn of 50 hops it holds every other turn, so the estimate sets
/// no temperature before hop 750 of some 970, too late to tell a ladder that
/// drops the ratio under it from one that keeps it. Turns of 450 let the
/// coldest rung fill the sweep of 400 within its first turn, so the hot rung's
/// whole first turn runs under the estimate, without the refits a shorter
/// sweep would add. Measured over these seeds, the hot rung adopts 422 rises
/// against a bar of 195, a ladder that drops the ratio under the estimate 96,
/// and one whose rungs all hop at one temperature 58 against a bar of 195.
#[test]
fn the_ladder_multiplies_the_statistical_temperature() {
    let mut cfg = Config::recommended(13);
    cfg.temperature = 0.2;
    cfg.statistical_temperature = true;
    cfg.swap_period = 450;
    assert_the_hot_rung_climbs(&cfg);
}

/// A rung that takes over the chain takes over what is known about its own
/// state, so every accepted step carries the validation gradient of the state
/// it left, including the first step after a switch.
#[test]
fn a_rung_switch_brings_each_state_its_own_gradient() {
    let mut cfg = Config::recommended(13);
    cfg.replicas = 3;
    cfg.swap_period = 10;
    let pot = PairPotential::lennard_jones(cfg.n_points);
    let mut checked = 0;
    for seed in 0..4 {
        for t in &lj_run_with(&cfg, seed, 10_000, true).accepted_transitions {
            if let Some(gradient) = &t.from_gradient {
                assert!(
                    *gradient == pot.value_and_gradient(t.from_state.view()).1,
                    "seed {seed}: the step at hop {} carries another state's gradient",
                    t.hop
                );
                checked += 1;
            }
        }
    }
    assert!(checked >= 50, "only {checked} steps carried a gradient");
}

/// Quenches minima hopping counted as returns to the basin the chain stood in,
/// as a fraction of all it classified, and swaps refused and tried, over the
/// seeds.
fn returns(cfg: &Config) -> (f64, usize, usize) {
    let (mut same, mut classified, mut refused, mut tried) = (0, 0, 0, 0);
    for seed in 0..SEEDS {
        let out = lj_run_with(cfg, seed, 20_000, false);
        let (s, k, n) = out.visit_counts;
        same += s;
        classified += s + k + n;
        refused += out.swaps_tried - out.swaps_accepted;
        tried += out.swaps_tried;
    }
    (same as f64 / classified as f64, refused, tried)
}

/// A rung that takes over the chain takes over the basin its state stands in,
/// so minima hopping counts a quench back into that basin as a return, the
/// first after a switch included.
///
/// Only a refused swap resumes a rung on a state other than the one that has
/// just hopped, so this ladder refuses most: its rungs deposit by their
/// temperatures, which keeps the cold rung settled and pushes the hot one
/// above it, and it is steep enough that the energies decide the swap. With
/// the screens off no trial is counted without its quench, so every one is
/// classified by the basin it reaches. Measured over these seeds, the ladder
/// counts 0.50 of its quenches as returns and a single chain 0.46, against a
/// bar of 0.41, and a ladder whose rung keeps the basin of the one that hopped
/// before it 0.30.
#[test]
fn a_rung_switch_brings_each_state_its_own_basin() {
    let mut cfg = Config::for_cluster(13);
    assert!(!cfg.return_screen && !cfg.bayes_screen);
    cfg.minima_hopping = true;
    cfg.screen_margin = f64::INFINITY;
    cfg.ladder_top = 100.0;
    cfg.bias_by_rung = true;
    cfg.swap_period = 1;
    let (single, _, _) = returns(&cfg);
    cfg.replicas = 2;
    let (ladder, refused, tried) = returns(&cfg);
    assert!(
        2 * refused >= tried,
        "the ladder refused {refused} of {tried} swaps, too few to resume rungs on their own states"
    );
    assert!(
        ladder >= 0.9 * single,
        "the ladder counted {ladder:.3} of its quenches as returns against {single:.3} on a single chain"
    );
}

/// The energy bias is one function on every rung, so its tempering factor is
/// set at the temperature the coldest rung hops at, whichever rung fills its
/// first sample. Under a held temperature that is `temperature` on any ladder,
/// and `(gamma - 1) T` is the sample's spread.
#[test]
fn the_energy_bias_tempers_alike_whichever_rung_fills_its_sample() {
    let mut cfg = Config::recommended(13);
    assert!(!cfg.budget_window && !cfg.statistical_temperature);
    cfg.replicas = 2;
    cfg.ladder_top = 10.0;
    cfg.energy_bias = true;
    cfg.flat_sweep = 32;
    // The sample fills on about the 32nd hop: on the hot rung when the cold
    // one hands over after 20 hops, and on the cold rung when it holds for 40.
    for swap_period in [20, 40] {
        cfg.swap_period = swap_period;
        let bias = lj_run_with(&cfg, 0, 5_000, false)
            .energy_bias
            .expect("the sample fills well inside the run");
        let spread = bias.w0 * EnergyBias::FILL_DEPOSITS;
        let tempered = (bias.gamma - 1.0) * cfg.temperature;
        assert!(
            (tempered - spread).abs() <= 1e-9 * spread,
            "swap period {swap_period}: (gamma - 1) T is {tempered} against a spread of {spread}"
        );
    }
}

/// Under the budget window a temperature follows the gap of the state it is
/// read at, so a factor set at the temperature of whichever rung fills the
/// sample would follow that rung. Set at the coldest rung's, from the state
/// that rung holds, it is set at one temperature whether the cold rung hands
/// over before the hop that fills the sample or holds through it.
///
/// The sample fills on hop 32, with the cold rung holding the state it reached
/// on hop 31: the hot rung fills it when the cold one offers a swap after 31
/// hops and is refused, and the cold rung when it holds for 32. Measured over
/// these seeds, the five whose swap is refused fill at one temperature to
/// rounding, where the filling rung's temperatures are up to 2.2 times apart.
#[test]
fn the_energy_bias_tempers_at_the_coldest_rung_under_the_budget_window() {
    let mut cfg = Config::derived(13);
    assert!(cfg.budget_window && !cfg.statistical_temperature && !cfg.flat_histogram);
    cfg.replicas = 2;
    cfg.ladder_top = 10.0;
    cfg.energy_bias = true;
    cfg.flat_sweep = 32;
    cfg.max_hops = Some(32);
    let fill = |swap_period: usize, seed: u64| {
        let mut cfg = cfg.clone();
        cfg.swap_period = swap_period;
        let out = lj_run_with(&cfg, seed, 5_000, false);
        let bias = out.energy_bias.expect("the sample fills on the last hop");
        (
            bias.w0 * EnergyBias::FILL_DEPOSITS / (bias.gamma - 1.0),
            out.swaps_accepted,
        )
    };
    let mut compared = 0;
    for seed in 0..SEEDS {
        let (handed, swapped) = fill(31, seed);
        if swapped > 0 {
            continue;
        }
        let (held, _) = fill(32, seed);
        assert!(
            (handed - held).abs() <= 1e-9 * held,
            "seed {seed}: the hot rung filled the sample at {handed} and the cold one at {held}"
        );
        compared += 1;
    }
    assert!(
        compared >= SEEDS / 2,
        "only {compared} seeds refused the swap"
    );
}

/// A hop the surrogate decides is tested on the bare energy and one it
/// abstains on with the biases, so a rung under delayed acceptance hops by no
/// one weight for a swap to exchange. A single chain runs it; a ladder refuses
/// it before the ledger is charged.
#[test]
fn a_ladder_refuses_delayed_acceptance() {
    let mut cfg = Config::recommended(13);
    cfg.delayed_acceptance = true;
    let single = lj_run_with(&cfg, 0, 2_000, false);
    assert!(
        single.hops > 0 && single.delayed.is_some(),
        "a single chain under delayed acceptance did not run"
    );
    cfg.replicas = 2;
    let mut ledger = Ledger::new(2_000);
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        lj_run_on(&cfg, 0, &mut ledger, false)
    }))
    .expect_err("a ladder ran under delayed acceptance");
    let message = refused
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| refused.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    assert!(
        message.starts_with("delayed acceptance needs a single chain"),
        "refused with {message:?}"
    );
    assert_eq!(
        ledger.spent(),
        0,
        "the ledger was charged before the refusal"
    );
}
