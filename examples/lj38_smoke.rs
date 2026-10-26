//! One recommended-arm run on LJ38.
//!
//! Prints the best energy and the charged force calls. The potential is
//! the reduced Lennard-Jones cluster. This is a smoke, not a paper row.
//! The reference -173.928427 is the truncated octahedron.
//!
//! Usage: `lj38_smoke [budget] [seed]`

use anneal_core::methods::cluster_hopping::{Config, Ledger, run_with_gradient};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

const N: usize = 38;
const REFERENCE: f64 = -173.928427;

fn ico_start(seed: u64) -> Array1<f64> {
    let sites =
        anneal_core::lattice::grow(&anneal_core::structure::Template::Icosahedral.points(), N);
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37).wrapping_add(7));
    let mut x = Array1::zeros(3 * N);
    for (index, site) in sites.iter().take(N).enumerate() {
        for axis in 0..3 {
            let jitter = (rand::Rng::random::<f64>(&mut rng) - 0.5) * 0.1;
            x[3 * index + axis] = site[axis] + jitter;
        }
    }
    x
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let budget: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(800);
    let seed: u64 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);

    let cfg = Config::recommended(N);
    let potential = PairPotential::lennard_jones(N);
    let mut ledger = Ledger::new(budget);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            Some(potential.value_and_gradient(v))
        });
        (value, relaxed)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let out = run_with_gradient(
        &cfg,
        ico_start(seed).view(),
        &mut ledger,
        &mut relax,
        None,
        &mut rng,
    );
    println!("label: smoke, not a paper row");
    println!("arm: recommended");
    println!("atoms: {N}");
    println!("potential: reduced Lennard-Jones");
    println!("reference: {REFERENCE}");
    println!("best_energy: {}", out.best);
    println!("force_calls: {}", ledger.spent());
    println!("hops: {}", out.hops);
    println!("delta_from_reference: {}", out.best - REFERENCE);
}
