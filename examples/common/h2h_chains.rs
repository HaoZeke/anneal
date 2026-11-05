//! The communicating-chain arms of the head-to-head drivers.
//!
//! `H2H_ARM` picks the arm. `single`, the default, is the driver's own single
//! chain. `arm3-communicating` runs the driver's arm-3 configuration as the
//! chains of one ensemble. `communicating` runs [`Config::into_communicating`]
//! on the driver's base configuration with depth reward on, the full arm of
//! the paper, as the chains of one ensemble.
//!
//! Both ensemble arms switch the superbasin report and the superbasin escape
//! on, share one exact minimum history ([`HistoryMode::Shared`]), and divide
//! the driver's budget among the chains of [`Plan::default`]. The other
//! channels are the defaults of [`config_from_env`]. With the `ira` feature,
//! identity is IRA behind a sorted-pair cache that keeps every structure.
//!
//! The first hit is the aggregate charged count at the checkpoint where a
//! chain first crosses the target. Chains run as threads and share one
//! history and one call counter, so that count, unlike a single chain's,
//! depends on the schedule. Every chain retires at its first checkpoint
//! after the first hit (`stop_at_target`), so a seed that hits spends less
//! than the budget and reports the best energy at retirement.

#![allow(dead_code)]

use super::ensemble_report::config_from_env;
#[cfg(not(feature = "ira"))]
use super::ensemble_report::sorted_pairs_witness;
use anneal_core::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
use anneal_core::methods::cluster_hopping::Config;
use anneal_core::methods::ensemble::{
    EnsembleProblem, HistoryMode, ObjectiveFactory, StartFactory, run_ensemble,
};
use anneal_core::methods::lattice_ensemble::Plan;
use anneal_core::pes_exploration::StructureContext;
use ndarray::{Array1, ArrayView1};

/// Which arm a driver runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    /// The driver's single chain.
    Single,
    /// The arm-3 chain, orbit completion from both cores, communicating.
    OrbitCommunicating,
    /// The full arm, communicating.
    Communicating,
}

impl Arm {
    /// The arm `H2H_ARM` names.
    pub fn from_env() -> Self {
        match std::env::var("H2H_ARM").as_deref() {
            Ok("single") | Err(_) => Self::Single,
            Ok("arm3-communicating") => Self::OrbitCommunicating,
            Ok("communicating") => Self::Communicating,
            Ok(other) => {
                panic!("H2H_ARM={other:?}; expected single, arm3-communicating or communicating")
            }
        }
    }

    /// Record label.
    pub fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::OrbitCommunicating => "arm3-communicating",
            Self::Communicating => "communicating",
        }
    }

    /// The chain configuration of this arm: `arm3` itself for the single
    /// chain, and for the ensembles `arm3`, or `base` with the
    /// communicating switches and depth reward, both with superbasins on.
    pub fn chain_config(self, arm3: &Config, base: Config) -> Config {
        let mut cfg = match self {
            Self::Single => return arm3.clone(),
            Self::OrbitCommunicating => arm3.clone(),
            Self::Communicating => {
                let mut cfg = base.into_communicating();
                cfg.depth_reward = true;
                cfg
            }
        };
        cfg.superbasin_report = true;
        cfg.superbasin_escape = true;
        cfg
    }
}

/// What one ensemble seed reports.
pub struct ChainsRun {
    /// Chains in the ensemble.
    pub replicas: usize,
    /// Lowest energy over the chains.
    pub best: f64,
    /// Aggregate charged calls when a chain first crossed the target.
    pub first_target_calls: Option<usize>,
    /// Aggregate charged calls.
    pub charged: usize,
    /// Hops over the chains.
    pub hops: usize,
    /// Wall seconds.
    pub wall_seconds: f64,
}

/// Runs one seed of a communicating arm on `objective`, one charge per
/// value and gradient, and reports the ensemble.
///
/// A relaxation certifies a minimum below the record gradient of `cfg`,
/// and identity is matched at the Lennard-Jones calibration tolerance in
/// units of its length scale.
pub fn run_chains<O>(
    cfg: &Config,
    budget: usize,
    seed: u64,
    target: f64,
    objective: &O,
    start: StartFactory<'_>,
) -> ChainsRun
where
    O: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Sync,
{
    let mut ens = config_from_env(Plan::default().chains, budget, Some(target));
    ens.history = HistoryMode::Shared;
    ens.stop_at_target = true;
    ens.validate(cfg).unwrap_or_else(|error| panic!("{error}"));
    let geometry =
        DescriptorGeometry::finite(cfg.length_scale).expect("a positive length scale is valid");
    let descriptor = universal_descriptor_space(geometry);
    let context = StructureContext::new(None, None, Some(format!("h2h-n{}", cfg.n_points)));
    let radius = anneal_core::catalog::lj::CALIBRATION_IRA_TOLERANCE * cfg.length_scale;
    #[cfg(feature = "ira")]
    let witness = anneal_core::methods::minima_hopping::SerializedWitness(std::sync::Mutex::new(
        anneal_core::shape::IraStructureWitness {
            kmax_factor: anneal_core::shape::IraMetric::default().kmax_factor,
            radius,
        }
        .with_pair_cache(usize::MAX),
    ));
    #[cfg(not(feature = "ira"))]
    let witness = sorted_pairs_witness(cfg.n_points, radius);
    let objective: ObjectiveFactory<'_> = &|_| Box::new(objective);
    let same_family = |_: &[f64], _: &[f64]| false;
    let problem = EnsembleProblem {
        objective,
        start,
        descriptor: &descriptor,
        context: &context,
        witness: &witness,
        same_family: &same_family,
        certificate: cfg.record_gradient,
        polish_below: cfg.record_gradient,
        callbacks_per_objective: 1,
    };
    let report = run_ensemble(cfg, &ens, seed, &problem)
        .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
    ChainsRun {
        replicas: report.replicas.len(),
        best: report.best,
        first_target_calls: report.first_target_calls,
        charged: report.aggregate_charged,
        hops: report.replicas.iter().map(|run| run.outcome.hops).sum(),
        wall_seconds: report.wall_seconds,
    }
}

/// Name of the identity witness this build matches with.
pub fn witness_name() -> &'static str {
    if cfg!(feature = "ira") {
        "ira"
    } else {
        "sorted-pairs-fallback"
    }
}
