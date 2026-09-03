//! What coordination numbers relaxed Lennard-Jones minima actually have.
//!
//! The coordination kernel density estimate needs bin centres, and the first
//! version took 6 and 8 from the `CoordinationHistogram` class of the
//! atomistic-cookbook `metatomic-plumed` recipe, which was written for a
//! different cluster. Measured at LJ38 over 24 seeds that arm solved 15 of 24
//! against 14 of 24 for its control, which is not distinguishable from doing
//! nothing, and a borrowed parameter is the first thing to suspect: the SOAP
//! arm in the same campaign turned out to be measuring a defect of its own
//! rather than the idea it was named for.
//!
//! This derives the bins from the problem instead. It relaxes random clusters
//! with the same optimiser and the same potential the driver uses, pools the
//! per-site coordination numbers of the minima, and prints the distribution
//! with the modes that a kernel density estimate should be centred on.
//!
//! Usage: `cargo run --release --example coordination_spectrum -- [n] [samples]`

use anneal_core::methods::cluster_hopping::random_cluster;
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::morphology::CoordinationKde;
use anneal_core::potentials::PairPotential;
use anneal_core::structure::cna;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(38);
    let samples: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(400);

    let pot = PairPotential::lennard_jones(n);
    let cv = CoordinationKde::for_lj(n, 1.0);
    let mut rng = StdRng::seed_from_u64(20260806);
    let mut opt = WarmLbfgs::default();

    // Pooled over sites and over structures. Half-unit bins, because the
    // question is where the modes sit and whether they fall on integers.
    let mut hist = vec![0usize; 2 * (n + 2)];
    let mut energies: Vec<f64> = Vec::new();
    let mut total_sites = 0usize;

    // The population modes are not the answer on their own. A deposition bin
    // is useful when the two competing morphologies differ there, not when
    // sites are common there, so the same histogram is kept separately for
    // structures the common-neighbour analysis calls close-packed and for
    // those it calls icosahedral.
    let mut hist_fcc = vec![0usize; 2 * (n + 2)];
    let mut hist_ico = vec![0usize; 2 * (n + 2)];
    let (mut sites_fcc, mut sites_ico) = (0usize, 0usize);
    let (mut n_fcc, mut n_ico) = (0usize, 0usize);

    for _ in 0..samples {
        let x0 = random_cluster(n, 0.55, 0.85, &mut rng);
        opt.forget();
        let (e, xr, _) = opt.minimize(x0.view(), 400, |v| Some(pot.value_and_gradient(v)));
        if !e.is_finite() {
            continue;
        }
        energies.push(e);
        let c = cna(xr.view(), n, 1.39);
        let (f555, f421) = (c.fraction((5, 5, 5)), c.fraction((4, 2, 1)));
        let coords = cv.coordination(xr.view());
        for cc in &coords {
            let slot = (cc * 2.0).round() as usize;
            if slot < hist.len() {
                hist[slot] += 1;
                total_sites += 1;
            }
        }
        // Only structures the classifier is confident about, so the contrast
        // is between the two funnels rather than between everything and the
        // disordered middle.
        let (target, sites, count) = if f421 > f555 && f421 > 0.05 {
            (&mut hist_fcc, &mut sites_fcc, &mut n_fcc)
        } else if f555 > f421 && f555 > 0.05 {
            (&mut hist_ico, &mut sites_ico, &mut n_ico)
        } else {
            continue;
        };
        *count += 1;
        for cc in &coords {
            let slot = (cc * 2.0).round() as usize;
            if slot < target.len() {
                target[slot] += 1;
                *sites += 1;
            }
        }
    }

    energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "LJ{n}: {} relaxed minima, energies {:.4} to {:.4}, median {:.4}",
        energies.len(),
        energies.first().copied().unwrap_or(f64::NAN),
        energies.last().copied().unwrap_or(f64::NAN),
        energies[energies.len() / 2]
    );
    println!("cutoff {:.3}, switch from {:.3}", cv.r0, cv.r1);
    println!("{total_sites} site coordinations pooled\n");

    println!("coordination  count   share  histogram");
    for (slot, &c) in hist.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let share = c as f64 / total_sites as f64;
        let bar = "#".repeat((share * 200.0).round() as usize);
        println!("{:>12.1}  {c:>6}  {share:>6.3}  {bar}", slot as f64 / 2.0);
    }

    // Local maxima of the half-unit histogram, which is what a kernel density
    // estimate wants to sit on: a bin centre in a valley resolves nothing.
    let mut modes: Vec<(f64, f64)> = Vec::new();
    for slot in 1..hist.len() - 1 {
        let share = hist[slot] as f64 / total_sites as f64;
        if hist[slot] > hist[slot - 1] && hist[slot] >= hist[slot + 1] && share > 0.02 {
            modes.push((slot as f64 / 2.0, share));
        }
    }
    modes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nmodes above 2 percent, by share:");
    for (c, s) in &modes {
        println!("  {c:>5.1}  {s:.3}");
    }
    let mut top: Vec<f64> = modes.iter().take(2).map(|m| m.0).collect();
    top.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("\ntwo strongest population modes: {top:?}");

    println!(
        "\nBY FUNNEL: {n_fcc} close-packed structures ({sites_fcc} sites), \
         {n_ico} icosahedral ({sites_ico} sites)"
    );
    if sites_fcc == 0 || sites_ico == 0 {
        println!("  one funnel was never reached; no contrast to report");
        return;
    }
    println!("coordination   fcc share   ico share   difference");
    let mut gaps: Vec<(f64, f64)> = Vec::new();
    for slot in 0..hist.len() {
        let a = hist_fcc[slot] as f64 / sites_fcc as f64;
        let b = hist_ico[slot] as f64 / sites_ico as f64;
        if a < 1e-6 && b < 1e-6 {
            continue;
        }
        println!("{:>12.1}   {a:>9.4}   {b:>9.4}   {:>+10.4}", slot as f64 / 2.0, a - b);
        gaps.push((slot as f64 / 2.0, (a - b).abs()));
    }
    gaps.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    println!("\nbins where the two funnels differ most:");
    for (c, g) in gaps.iter().take(5) {
        println!("  {c:>5.1}  {g:.4}");
    }
    let mut chosen: Vec<f64> = gaps.iter().take(2).map(|g| g.0).collect();
    chosen.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("\nderived bins: {chosen:?}");
    println!("recipe bins for comparison: [6.0, 8.0]");
}
