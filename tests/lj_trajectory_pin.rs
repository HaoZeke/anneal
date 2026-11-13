//! Short recommended-preset trajectory, pinned by hash.
//!
//! The accepted-transition sequence and the final energies are part of the
//! search result. A faster evaluation path has to leave them unchanged.
//! The recommended full quench is 1000 steps, so this pin is that
//! trajectory, not the earlier 200-step one.
//!
//! The hash is the scalar walk through one `libm`. Host `exp`, `sin`,
//! and `cos`, and the NEON dot product, are a different trajectory.
//! Proposals, the acceptance test, the SOAP cutoff, the fivefold turn,
//! the continuous-symmetry orientation, the energy-lattice softmax, the
//! depth-rewarded arm, and the pair force go through that `libm`, and
//! this test forces rgmin's scalar dots.
//!
//! Without `ira` that hash is `6e277f86…`. With `ira` the permutation
//! is libira, and libira calls host `sincos`, so the walk is the one
//! linked against that library. On glibc it is `02c54952…`.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use rand::SeedableRng;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};

const SEED: u64 = 2000;
const ATOMS: usize = 98;
const BUDGET: usize = 5_000;

fn trajectory_hash() -> String {
    // AVX2, AVX-512 and NEON dots are not the same quench. Scalar `mul_add`
    // is the path every target can run.
    let _ = rgmin::vecops::force_simd_path("scalar");
    let cfg = Config::recommended(ATOMS);
    let potential = PairPotential::lennard_jones(ATOMS);
    let mut rng = StdRng::seed_from_u64(SEED);
    let start = random_cluster_in_radius(ATOMS, cfg.start_radius(), cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(BUDGET);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ndarray::ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |point| {
            if !led.charge() {
                return None;
            }
            Some(potential.value_and_gradient(point))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, point: ndarray::ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        Some(potential.value_and_gradient(point).1)
    };
    let mut rng = StdRng::seed_from_u64(SEED);
    let outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");

    let mut hasher = Sha256::new();
    hasher.update(outcome.best.to_le_bytes());
    hasher.update(outcome.final_energy.to_le_bytes());
    hasher.update((outcome.charged as u64).to_le_bytes());
    hasher.update((outcome.hops as u64).to_le_bytes());
    hasher.update((outcome.accepted_transitions.len() as u64).to_le_bytes());
    for edge in &outcome.accepted_transitions {
        hasher.update((edge.hop as u64).to_le_bytes());
        hasher.update(edge.action.as_bytes());
        hasher.update([0]);
        hasher.update(edge.from_energy.to_le_bytes());
        hasher.update(edge.to_energy.to_le_bytes());
        hasher.update([u8::from(edge.adopted), u8::from(edge.validated)]);
        for value in edge.to_state.iter() {
            hasher.update(value.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

#[test]
fn recommended_lj98_seed_2000_keeps_its_short_trajectory() {
    let hash = trajectory_hash();
    let expected = if cfg!(feature = "ira") {
        // libira calls host `sincos`. This is the glibc-linked walk.
        "02c5495200f0e6b5ac1448f608cca733330a307675c1a97e27099f1bad9e7f55"
    } else {
        "6e277f8611c9988bbb95c72873fb5b044be414e9148ab9571b732f834e3dceb9"
    };
    assert_eq!(hash, expected, "trajectory hash changed: {hash}");
}
