//! Real `optimize` / `run_with_gradient` path for the observed-cloud SOAP hop.
//!
//! Cluster, multi-species molecule, and a slab-shaped frozen frame. The
//! in-crate Lennard-Jones potential stands in for a QC engine.

use anneal_core::methods::cluster_hopping::{ClusterMove, Config, Ledger, run_with_gradient};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn water_dimer() -> (Array1<f64>, Vec<u32>, Vec<Vec<usize>>) {
    let x = Array1::from_vec(vec![
        0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0, 3.10, 0.15, 0.08, 3.98, 0.40, -0.05, 2.82,
        1.05, 0.18,
    ]);
    let z = vec![8, 1, 1, 8, 1, 1];
    let groups = vec![vec![0, 1, 2], vec![3, 4, 5]];
    (x, z, groups)
}

fn charged_relax<'a>(
    pot: &'a PairPotential,
    opt: &'a mut WarmLbfgs,
    frozen: Option<&'a [bool]>,
) -> impl FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>) + 'a {
    move |led, x, iters| {
        opt.forget();
        let pin = x.to_owned();
        let (f, mut xr, _) = opt.minimize_watched(
            x,
            iters,
            |v| {
                if !led.charge() {
                    return None;
                }
                let (e, mut g) = pot.value_and_gradient(v);
                if let Some(mask) = frozen {
                    for (i, &is_frozen) in mask.iter().enumerate() {
                        if is_frozen {
                            for k in 0..3 {
                                g[3 * i + k] = 0.0;
                            }
                        }
                    }
                }
                Some((e, g))
            },
            |_, _| true,
        );
        if let Some(mask) = frozen {
            for (i, &is_frozen) in mask.iter().enumerate() {
                if is_frozen {
                    for k in 0..3 {
                        xr[3 * i + k] = pin[3 * i + k];
                    }
                }
            }
        }
        (f, xr)
    }
}

#[test]
fn hop_shape_from_shipped_constructors() {
    let rec = Config::recommended(13);
    let soap = rec
        .move_library
        .kernels(&rec)
        .into_iter()
        .find(|k| matches!(k, ClusterMove::Soap { .. }));
    match soap {
        Some(ClusterMove::Soap {
            class,
            species,
            mobile,
            ..
        }) => {
            assert!(!class, "recommended SOAP shipped the 555->421 oracle");
            assert!(species.is_none());
            assert!(mobile.is_none());
            println!("rec class={class} species={species:?} mobile={mobile:?}");
        }
        _ => panic!("recommended LeanBurst has no SOAP arm"),
    }

    let mol = Config::recommended_molecular(vec![8, 1, 1], vec![vec![0, 1, 2]], 1.0);
    assert!(!mol.packing_cna_applies());
    assert!(!mol.soap_class_residual);
    let soap = mol
        .move_library
        .kernels(&mol)
        .into_iter()
        .find(|k| matches!(k, ClusterMove::Soap { .. }));
    match soap {
        Some(ClusterMove::Soap { class, species, .. }) => {
            assert!(!class);
            assert_eq!(species.as_deref(), Some(&[8, 1, 1][..]));
            println!("mol class={class} species={species:?}");
        }
        _ => panic!("recommended_molecular has no SOAP arm"),
    }

    let mut slab = Config::recommended_molecular(vec![29, 29, 1, 1], vec![vec![2, 3]], 1.0);
    slab.active_region = Some((vec![2, 3], 0));
    slab.n_points = 4;
    let soap = slab
        .move_library
        .kernels(&slab)
        .into_iter()
        .find(|k| matches!(k, ClusterMove::Soap { .. }));
    match soap {
        Some(ClusterMove::Soap {
            class,
            species,
            mobile,
            ..
        }) => {
            assert!(!class);
            assert_eq!(species.as_deref(), Some(&[29, 29, 1, 1][..]));
            assert_eq!(mobile.as_deref(), Some(&[2, 3][..]));
            println!("slab class={class} species={species:?} mobile={mobile:?}");
        }
        _ => panic!("slab-shaped library has no SOAP arm"),
    }
}

#[test]
fn recommended_molecular_search_hops_on_water_dimer() {
    let (start, species, groups) = water_dimer();
    let n = species.len();
    let cfg = Config::recommended_molecular(species, groups, 1.0);
    assert!(!cfg.packing_cna_applies());
    assert!(
        cfg.move_library
            .kernels(&cfg)
            .iter()
            .any(|k| matches!(k, ClusterMove::Soap { class: false, .. }))
    );
    let pot = PairPotential::lennard_jones(n);
    let mut ledger = Ledger::new(2_500);
    let mut opt = WarmLbfgs::default();
    let mut relax = charged_relax(&pot, &mut opt, None);
    let mut rng = StdRng::seed_from_u64(3);
    let out = run_with_gradient(&cfg, start.view(), &mut ledger, &mut relax, None, &mut rng)
        .expect("finite start");
    assert!(out.best.is_finite(), "best {}", out.best);
    assert!(out.hops > 0, "no hops on recommended_molecular");
    let soap_draws = out
        .arms
        .iter()
        .find(|(name, _, _, _)| name == "soap")
        .map(|(_, d, _, _)| *d)
        .unwrap_or(0);
    println!(
        "mol-rec best={:.6} hops={} charged={} soap_draws={} arms={:?}",
        out.best, out.hops, out.charged, soap_draws, out.arms
    );
}

#[test]
fn recommended_molecular_slab_keeps_frozen_coords() {
    let (start, species, groups) = water_dimer();
    let n = species.len();
    let mut cfg = Config::recommended_molecular(species, groups, 1.0);
    cfg.active_region = Some((vec![0, 1, 2], 0));
    let frozen: Vec<bool> = (0..n).map(|i| i >= 3).collect();
    cfg.frozen = Some(frozen.clone());
    let pot = PairPotential::lennard_jones(n);
    let mut ledger = Ledger::new(2_500);
    let mut opt = WarmLbfgs::default();
    let mut relax = charged_relax(&pot, &mut opt, Some(frozen.as_slice()));
    let mut rng = StdRng::seed_from_u64(5);
    let out = run_with_gradient(&cfg, start.view(), &mut ledger, &mut relax, None, &mut rng)
        .expect("finite start");
    assert!(out.best.is_finite(), "best {}", out.best);
    assert!(out.hops > 0, "no hops on slab-shaped recommended_molecular");
    let end = out
        .final_state
        .as_ref()
        .or(out.best_state.as_ref())
        .expect("search returned no state");
    for i in 3..6 {
        for k in 0..3 {
            assert_eq!(
                end[3 * i + k],
                start[3 * i + k],
                "frozen atom {i} axis {k} moved"
            );
        }
    }
    let soap_draws = out
        .arms
        .iter()
        .find(|(name, _, _, _)| name == "soap")
        .map(|(_, d, _, _)| *d)
        .unwrap_or(0);
    println!(
        "slab-rec best={:.6} hops={} charged={} soap_draws={} frozen_ok=1 arms={:?}",
        out.best, out.hops, out.charged, soap_draws, out.arms
    );
}

#[test]
fn recommended_molecular_spends_no_potential_call_past_the_budget() {
    let (start, species, groups) = water_dimer();
    let n = species.len();
    let cfg = Config::recommended_molecular(species, groups, 1.0);
    let pot = PairPotential::lennard_jones(n);
    let budget = 20_000;
    let mut ledger = Ledger::new(budget);
    let mut opt = WarmLbfgs::default();
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        opt.forget();
        let (f, xr, _) = opt.minimize_watched(
            x,
            iters,
            |v| {
                if !led.charge() {
                    return None;
                }
                calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(pot.value_and_gradient(v))
            },
            |_, _| true,
        );
        (f, xr)
    };
    let mut rng = StdRng::seed_from_u64(3);
    let out = run_with_gradient(&cfg, start.view(), &mut ledger, &mut relax, None, &mut rng)
        .expect("finite start");
    assert!(out.best.is_finite(), "best {}", out.best);
    let calls = calls.into_inner();
    assert!(
        calls <= budget,
        "{calls} potential calls on a budget of {budget}"
    );
}

/// Flexible water in eV and angstrom: harmonic O-H and H-H springs inside
/// each molecule and a Lennard-Jones O-O term between molecules.
struct FlexibleWater {
    molecules: usize,
    bounds: eindir_core::bounds::Bounds<f64>,
}

impl FlexibleWater {
    fn new(molecules: usize) -> Self {
        let dim = 9 * molecules;
        let bounds = eindir_core::bounds::Bounds::new(
            Array1::from_elem(dim, -20.0),
            Array1::from_elem(dim, 20.0),
            0.0,
        );
        Self { molecules, bounds }
    }
}

impl eindir_core::Objective<f64> for FlexibleWater {
    fn dim(&self) -> usize {
        9 * self.molecules
    }

    fn bounds(&self) -> &eindir_core::bounds::Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        eindir_core::gradient::DifferentiableObjective::value_and_gradient(self, x).0
    }
}

impl eindir_core::gradient::Gradient<f64> for FlexibleWater {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        eindir_core::gradient::DifferentiableObjective::value_and_gradient(self, x).1
    }

    fn dim(&self) -> usize {
        9 * self.molecules
    }
}

impl eindir_core::gradient::DifferentiableObjective<f64> for FlexibleWater {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let mut energy = 0.0;
        let mut grad = Array1::zeros(x.len());
        let mut pair = |i: usize, j: usize, energy: &mut f64, f: &dyn Fn(f64) -> (f64, f64)| {
            let d: Vec<f64> = (0..3).map(|k| x[3 * i + k] - x[3 * j + k]).collect();
            let r = d.iter().map(|v| v * v).sum::<f64>().sqrt();
            let (e, de) = f(r);
            *energy += e;
            for k in 0..3 {
                grad[3 * i + k] += de * d[k] / r;
                grad[3 * j + k] -= de * d[k] / r;
            }
        };
        let spring = |k: f64, r0: f64| move |r: f64| (0.5 * k * (r - r0).powi(2), k * (r - r0));
        let (sigma, eps) = (3.15, 0.0067);
        let lj = move |r: f64| {
            let s6 = (sigma / r).powi(6);
            (
                4.0 * eps * (s6 * s6 - s6),
                4.0 * eps * (6.0 * s6 - 12.0 * s6 * s6) / r,
            )
        };
        for m in 0..self.molecules {
            let o = 3 * m;
            pair(o, o + 1, &mut energy, &spring(45.0, 0.96));
            pair(o, o + 2, &mut energy, &spring(45.0, 0.96));
            pair(o + 1, o + 2, &mut energy, &spring(20.0, 1.52));
            for other in m + 1..self.molecules {
                pair(o, 3 * other, &mut energy, &lj);
            }
        }
        (energy, grad)
    }
}

#[test]
fn tric_local_step_relaxes_a_molecular_search_within_budget() {
    use anneal_core::methods::cluster_search::search_from;
    use anneal_core::methods::tric_opt::LocalStep;

    let (dimer, _, _) = water_dimer();
    let mut start = dimer.to_vec();
    for shift in [[0.0, 3.1, 0.2], [3.0, 3.0, -0.3]] {
        for atom in 0..3 {
            for k in 0..3 {
                start.push(dimer[3 * atom + k] + shift[k]);
            }
        }
    }
    let start = Array1::from_vec(start);
    let molecules = 4;
    let species = [8, 1, 1].repeat(molecules);
    let groups: Vec<Vec<usize>> = (0..molecules)
        .map(|m| (3 * m..3 * m + 3).collect())
        .collect();
    let pot = FlexibleWater::new(molecules);
    let budget = 3000;
    let mut runs = Vec::new();
    for step in [LocalStep::Lbfgs, LocalStep::Tric] {
        let mut cfg = Config::recommended_molecular(species.clone(), groups.clone(), 1.0);
        cfg.local_step = step;
        let mut ledger = Ledger::new(budget);
        let (out, stats) =
            search_from(&pot, &cfg, &mut ledger, start.view(), 5).expect("finite start");
        assert!(
            ledger.spent() <= budget,
            "{step:?} spent {} of {budget}",
            ledger.spent()
        );
        let best = out.best_state.expect("a recorded minimum");
        for m in 0..molecules {
            for h in 1..3 {
                let r = (0..3)
                    .map(|k| (best[9 * m + 3 * h + k] - best[9 * m + k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!((r - 0.96).abs() < 0.05, "{step:?} O-H {r}");
            }
        }
        assert!(stats.converged > 0, "{step:?} {stats:?}");
        runs.push((out.best, stats));
    }
    let (lbfgs, tric) = (runs[0], runs[1]);
    assert_ne!(lbfgs.1, tric.1, "the TRIC quench never ran");
    assert!(
        tric.0 <= lbfgs.0 + 1e-3,
        "tric best {} against {}",
        tric.0,
        lbfgs.0
    );
}

/// [`FlexibleWater`] that sleeps two milliseconds a call, as an electronic
/// structure gradient would take hundreds.
struct SlowWater(FlexibleWater);

impl eindir_core::Objective<f64> for SlowWater {
    fn dim(&self) -> usize {
        eindir_core::Objective::dim(&self.0)
    }

    fn bounds(&self) -> &eindir_core::bounds::Bounds<f64> {
        self.0.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        eindir_core::gradient::DifferentiableObjective::value_and_gradient(self, x).0
    }
}

impl eindir_core::gradient::Gradient<f64> for SlowWater {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        eindir_core::gradient::DifferentiableObjective::value_and_gradient(self, x).1
    }

    fn dim(&self) -> usize {
        eindir_core::Objective::dim(&self.0)
    }
}

impl eindir_core::gradient::DifferentiableObjective<f64> for SlowWater {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        std::thread::sleep(std::time::Duration::from_millis(2));
        self.0.value_and_gradient(x)
    }
}

#[test]
fn the_auto_quench_follows_the_cost_of_a_call() {
    use anneal_core::methods::cluster_search::search_from;
    use anneal_core::methods::tric_opt::LocalStep;

    let (dimer, _, _) = water_dimer();
    let mut start = dimer.to_vec();
    for atom in 0..3 {
        for (k, shift) in [0.0, 3.1, 0.2].iter().enumerate() {
            start.push(dimer[3 * atom + k] + shift);
        }
    }
    let start = Array1::from_vec(start);
    let species = [8, 1, 1].repeat(3);
    let groups = vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]];
    let cfg = Config::recommended_molecular(species, groups, 1.0);
    assert_eq!(cfg.local_step, LocalStep::Auto);

    let fast = FlexibleWater::new(3);
    let mut ledger = Ledger::new(600);
    let (_, stats) = search_from(&fast, &cfg, &mut ledger, start.view(), 3).expect("finite start");
    let (call, step) = stats.quench_seconds.expect("the opening quench was timed");
    assert_eq!(
        stats.local_step,
        LocalStep::Lbfgs,
        "{call} s a call, {step} s a step"
    );

    let slow = SlowWater(FlexibleWater::new(3));
    let mut ledger = Ledger::new(150);
    let (_, stats) = search_from(&slow, &cfg, &mut ledger, start.view(), 3).expect("finite start");
    let (call, step) = stats.quench_seconds.expect("the opening quench was timed");
    assert_eq!(
        stats.local_step,
        LocalStep::Tric,
        "{call} s a call, {step} s a step"
    );
    assert!(ledger.spent() <= 150);
}

/// A ten-carbon chain in eV and angstrom: 1.50 Å bonds, 1-3 springs for the
/// 120° angles, and a Lennard-Jones attraction between atoms three or more
/// bonds apart, so a folded chain lies below the zigzag.
struct BondedChain {
    atoms: usize,
    bounds: eindir_core::bounds::Bounds<f64>,
}

impl BondedChain {
    const BOND: f64 = 1.50;

    fn new(atoms: usize) -> Self {
        let bounds = eindir_core::bounds::Bounds::new(
            Array1::from_elem(3 * atoms, -30.0),
            Array1::from_elem(3 * atoms, 30.0),
            0.0,
        );
        Self { atoms, bounds }
    }

    fn zigzag(&self) -> Array1<f64> {
        let mut x = Array1::zeros(3 * self.atoms);
        for i in 0..self.atoms {
            x[3 * i] = 1.299 * i as f64;
            x[3 * i + 1] = if i % 2 == 0 { 0.0 } else { 0.75 };
        }
        x
    }
}

impl eindir_core::Objective<f64> for BondedChain {
    fn dim(&self) -> usize {
        3 * self.atoms
    }

    fn bounds(&self) -> &eindir_core::bounds::Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        eindir_core::gradient::DifferentiableObjective::value_and_gradient(self, x).0
    }
}

impl eindir_core::gradient::Gradient<f64> for BondedChain {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        eindir_core::gradient::DifferentiableObjective::value_and_gradient(self, x).1
    }

    fn dim(&self) -> usize {
        3 * self.atoms
    }
}

impl eindir_core::gradient::DifferentiableObjective<f64> for BondedChain {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let mut energy = 0.0;
        let mut grad = Array1::zeros(x.len());
        for i in 0..self.atoms {
            for j in i + 1..self.atoms {
                let d: Vec<f64> = (0..3).map(|k| x[3 * i + k] - x[3 * j + k]).collect();
                let r = d.iter().map(|v| v * v).sum::<f64>().sqrt();
                let (e, de) = match j - i {
                    1 => (20.0 * (r - Self::BOND).powi(2), 40.0 * (r - Self::BOND)),
                    2 => (5.0 * (r - 2.598).powi(2), 10.0 * (r - 2.598)),
                    _ => {
                        let s6 = (3.4 / r).powi(6);
                        (
                            0.04 * (s6 * s6 - s6),
                            0.04 * (6.0 * s6 - 12.0 * s6 * s6) / r,
                        )
                    }
                };
                energy += e;
                for k in 0..3 {
                    grad[3 * i + k] += de * d[k] / r;
                    grad[3 * j + k] -= de * d[k] / r;
                }
            }
        }
        (energy, grad)
    }
}

#[test]
fn a_flexible_molecule_spends_its_budget_on_torsion_quenches() {
    use anneal_core::methods::cluster_search::search_from;
    let chain = BondedChain::new(10);
    let species = vec![6; 10];
    // One fragment per atom: every bond joins two declared groups.
    let groups: Vec<Vec<usize>> = (0..10).map(|i| vec![i]).collect();
    let cfg = Config::recommended_molecular(species, groups, 0.05);
    let start = chain.zigzag();
    let budget = 3000;
    let mut ledger = Ledger::new(budget);
    let (out, _) = search_from(&chain, &cfg, &mut ledger, start.view(), 11).expect("finite start");
    assert!(
        out.charged + 50 >= budget,
        "{} of {budget} calls charged in {} hops, {} overlaps",
        out.charged,
        out.hops,
        out.overlap_rejected
    );
    assert!(
        out.hops < budget,
        "{} hops on {} calls",
        out.hops,
        out.charged
    );
    let torsion = out
        .arms
        .iter()
        .find(|(name, ..)| name == "torsion")
        .expect("a torsion arm");
    assert!(torsion.1 > 0, "{:?}", out.arms);
    let best = out.best_state.expect("a recorded minimum");
    for i in 0..9 {
        let r = (0..3)
            .map(|k| (best[3 * i + k] - best[3 * i + 3 + k]).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((r - BondedChain::BOND).abs() < 0.05, "bond {i} at {r}");
    }
    let zigzag =
        eindir_core::gradient::DifferentiableObjective::value_and_gradient(&chain, start.view()).0;
    assert!(
        out.best < zigzag - 0.05,
        "best {} zigzag {zigzag}",
        out.best
    );
}
