// Copyright (c) anneal developers

//! Export every validated quench of one current-code landfold campaign seed.
//!
//! ```text
//! landfold_campaign FAMILY N CONFIG BUDGET SEED OUT.min
//! ```
//!
//! `FAMILY` is `lj` or `tip4p`; `CONFIG` is `derived`, `recommended`, or
//! `for_tip4p`. Each output row is the energy followed by the state that the
//! current search certified. The checkpoint is observational: it does not
//! alter the chain or consume potential calls.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use anneal_core::bias::BasinBias;
use anneal_core::grad::Objective;
use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, ClusterFingerprint, Config, Ledger, QuenchStatus,
    random_cluster, run_with_bias_at_checkpoints,
};
use anneal_core::methods::cluster_search::{RelaxStats, warm_relax};
use anneal_core::potentials::{PairPotential, Tip4pCluster, random_tip4p_cluster};
use anneal_core::{DifferentiableObjective, Gradient};
use eindir_core::Bounds;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

enum Potential {
    Lj(PairPotential),
    Tip4p(Tip4pCluster),
}

impl Objective<f64> for Potential {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }

    fn dim(&self) -> usize {
        match self {
            Self::Lj(pot) => Objective::dim(pot),
            Self::Tip4p(pot) => Objective::dim(pot),
        }
    }

    fn bounds(&self) -> &Bounds<f64> {
        match self {
            Self::Lj(pot) => pot.bounds(),
            Self::Tip4p(pot) => pot.bounds(),
        }
    }
}

impl Gradient<f64> for Potential {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }

    fn dim(&self) -> usize {
        Objective::dim(self)
    }
}

impl DifferentiableObjective<f64> for Potential {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        Potential::value_and_gradient(self, x)
    }
}

impl Potential {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        match self {
            Self::Lj(pot) => pot.value_and_gradient(x),
            Self::Tip4p(pot) => pot.value_and_gradient(x),
        }
    }

    fn fold(&self, state: &mut Array1<f64>) {
        if let Self::Tip4p(pot) = self
            && let Some(values) = state.as_slice_mut()
        {
            pot.fold_rotations(values);
        }
    }
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        6,
        "usage: landfold_campaign FAMILY N CONFIG BUDGET SEED OUT.min"
    );
    let family = &args[0];
    let n: usize = args[1].parse().expect("N");
    let config = &args[2];
    let budget: usize = args[3].parse().expect("BUDGET");
    let seed: u64 = args[4].parse().expect("SEED");
    let output = &args[5];

    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(3));
    let (potential, cfg, start) = match (family.as_str(), config.as_str()) {
        ("lj", "derived") => {
            let cfg = Config::derived(n);
            let start = random_cluster(n, 0.7, cfg.min_separation, &mut rng);
            (Potential::Lj(PairPotential::lennard_jones(n)), cfg, start)
        }
        ("lj", "recommended") => {
            let cfg = Config::recommended(n);
            let start = random_cluster(n, 0.7, cfg.min_separation, &mut rng);
            (Potential::Lj(PairPotential::lennard_jones(n)), cfg, start)
        }
        ("tip4p", "for_tip4p") => {
            let cfg = Config::for_tip4p(n);
            let start = random_tip4p_cluster(n, &mut rng);
            (Potential::Tip4p(Tip4pCluster::new(n)), cfg, start)
        }
        _ => panic!("unsupported campaign {family}/{config}"),
    };

    let file = File::create(output).unwrap_or_else(|err| panic!("{output}: {err}"));
    let mut rows = BufWriter::new(file);
    let mut ledger = Ledger::new(budget);
    let mut stats = RelaxStats::default();
    let mut preset_relax = warm_relax(&potential, &cfg, &mut stats, seed);
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| {
        let (energy, mut state) = preset_relax(led, x, steps);
        potential.fold(&mut state);
        (energy, state)
    };
    let mut gradient = |led: &mut Ledger, x: ArrayView1<f64>| {
        led.charge().then(|| potential.value_and_gradient(x).1)
    };
    let mut bias = BasinBias::new(
        ClusterFingerprint::of_config(&cfg, &start),
        cfg.merge_radius,
        cfg.bias_height,
        cfg.bias_gamma,
    );
    let mut written = 0usize;
    let mut energies = Vec::new();
    let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {
        for quench in snapshot.quench_boundaries() {
            if quench.status() != QuenchStatus::Validated {
                continue;
            }
            if energies
                .iter()
                .any(|held: &f64| (*held - quench.energy()).abs() < 1e-6)
            {
                continue;
            }
            energies.push(quench.energy());
            write!(rows, "{:.12}", quench.energy()).expect("write energy");
            for value in quench.state() {
                write!(rows, " {:.10}", value).expect("write state");
            }
            writeln!(rows).expect("finish row");
            written += 1;
        }
        CheckpointAction::Continue
    };
    let outcome = run_with_bias_at_checkpoints(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut gradient),
        &mut bias,
        &mut rng,
        1,
        &mut checkpoint,
    )
    .expect("finite campaign start");
    rows.flush().expect("flush campaign rows");
    eprintln!(
        "{family}/{config} n={n} seed={seed} charged={} hops={} rows={written} best={:.12}",
        ledger.spent(),
        outcome.hops,
        outcome.best
    );
}
