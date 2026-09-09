//! Run the production hop ensemble on a bound-constrained algebraic box.
//!
//! This is [`run_ensemble`]: recommended cluster hop, four replicas,
//! `SharedMinimumHistory`. The state is the design vector padded to 3N so
//! the hop loop's Cartesian library can run. Extra coordinates carry a
//! unit harmonic so they are not a free null space. Identity is exact
//! coordinates, not IRA: a CUTEst box is not a point set.

use std::sync::Mutex;

use eindir_core::{Gradient, Objective};
use ndarray::{Array1, ArrayView1};

use crate::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
use crate::methods::cluster_hopping::{Config, SoapProposalMode};
use crate::methods::ensemble::{EnsembleConfig, EnsembleProblem, HistoryMode, run_ensemble};
use crate::methods::minima_hopping::{HistoryMembership, SerializedWitness};
use crate::pes_exploration::{ExactStructureWitness, StructureContext};

/// Outcome of one production-ensemble seed on a box.
#[derive(Clone, Debug)]
pub struct EnsembleHopResult {
    /// Best design-space point (padding stripped).
    pub best_pos: Array1<f64>,
    /// Objective at [`EnsembleHopResult::best_pos`].
    pub best_val: f64,
    /// Charged hop-ledger calls.
    pub charged: usize,
    /// Distinct exact identities in the shared or largest private history.
    pub history_minima: usize,
}

/// Coordinate identity on the padded hop state.
struct CoordWitness {
    tol: f64,
}

impl ExactStructureWitness for CoordWitness {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right.iter())
                .all(|(a, b)| (a - b).abs() <= self.tol)
    }
}

/// Recommended cluster hop with cluster-only symmetry retries stripped.
///
/// A padded CUTEst/box state is not a point set. Core/orbit/stall
/// symmetrisation can spin on that padding without charging the hop
/// ledger (BIGGS6: 14+ min, 99.9% CPU, 0 rows).
fn box_hop_config(n_points: usize, mean_width: f64, budget: usize) -> Config {
    let mut cfg = Config::recommended(n_points);
    cfg.replicas = 1;
    cfg.point_symmetrise_on_new = false;
    cfg.orbit_complete_on_new = false;
    cfg.symmetrise_on_stall = false;
    cfg.point_symmetrise = false;
    cfg.soap_mode = SoapProposalMode::Off;
    cfg.soap_repel = false;
    cfg.restart_on_stall = false;
    cfg.jump_on_stall = false;
    cfg.path_on_stall = false;
    cfg.md_escape = false;
    cfg.max_hops = Some(budget.max(1));
    cfg.container = cfg
        .container
        .max(mean_width * (n_points as f64).cbrt() * 4.0);
    cfg.min_separation = cfg.min_separation.min(0.05 * mean_width).max(1e-9);
    cfg.length_scale = mean_width.max(1e-6);
    cfg
}

/// Hist75-shaped ensemble on `obj`: four replicas, shared accepted history.
pub fn ensemble_hop_optimize<O, G>(
    obj: &O,
    grad: Option<&G>,
    seed: u64,
    x0: Option<ArrayView1<f64>>,
    budget: usize,
    replicas: usize,
    history: HistoryMode,
    membership: HistoryMembership,
) -> Result<EnsembleHopResult, String>
where
    O: Objective<f64> + Sync,
    G: Gradient<f64> + Sync,
{
    let bounds = obj.bounds().clone();
    let dim = bounds.dims.max(1);
    let n_points = dim.div_ceil(3);
    let state_len = 3 * n_points;
    let mut widths = Array1::zeros(dim);
    for i in 0..dim {
        widths[i] = (bounds.high[i] - bounds.low[i]).max(1e-12);
    }
    let mean_width = widths.iter().sum::<f64>() / dim as f64;
    let cfg = box_hop_config(n_points, mean_width, budget);

    let ens = EnsembleConfig {
        replicas: replicas.max(1),
        budget: budget.max(replicas.max(1)),
        history,
        membership,
        shared_bias: Some(1.0 / replicas.max(1) as f64),
        gossip: None,
        two_choice_stall: None,
        checkpoint_interval: 1_000,
        target: None,
    };

    let geometry =
        DescriptorGeometry::finite(cfg.length_scale).map_err(|error| error.to_string())?;
    let descriptor = universal_descriptor_space(geometry);
    let context = StructureContext::new(
        Some(vec![18; n_points]),
        None,
        Some("cutest-ensemble".into()),
    );
    let witness = SerializedWitness(Mutex::new(CoordWitness {
        tol: 1e-3 * cfg.length_scale,
    }));

    let pad = |x: ArrayView1<f64>| -> Array1<f64> {
        let mut state = Array1::zeros(state_len);
        let n = dim.min(x.len());
        for i in 0..n {
            state[i] = x[i];
        }
        state
    };
    let embed_start = |x: ArrayView1<f64>| pad(bounds.clip(x).view());

    let x0_owned = x0.map(|x| bounds.clip(x));
    let start = |replica: usize, rng: &mut rand::rngs::StdRng| -> Array1<f64> {
        use rand::Rng;
        if replica == 0 {
            if let Some(ref x0) = x0_owned {
                return embed_start(x0.view());
            }
            return embed_start(((&bounds.low + &bounds.high) * 0.5).view());
        }
        let mut draw = Array1::zeros(dim);
        for i in 0..dim {
            draw[i] = bounds.low[i] + widths[i] * rng.random::<f64>();
        }
        embed_start(draw.view())
    };

    let make_obj = |_replica: usize| {
        let bounds = bounds.clone();
        Box::new(move |x: ArrayView1<f64>| -> (f64, Array1<f64>) {
            let mut xdim = Array1::zeros(dim);
            for i in 0..dim {
                xdim[i] = x.get(i).copied().unwrap_or(0.0);
            }
            let clipped = bounds.clip(xdim.view());
            // One hop-ledger unit per Python callback: eval, then grad.
            let mut energy = obj.eval(clipped.view());
            let mut gradient = Array1::zeros(state_len);
            if let Some(g) = grad {
                let gd = g.grad(clipped.view());
                for i in 0..dim.min(gd.len()) {
                    gradient[i] = gd[i];
                }
            }
            for i in dim..state_len {
                let extra = x.get(i).copied().unwrap_or(0.0);
                energy += 0.5 * extra * extra;
                gradient[i] = extra;
            }
            (energy, gradient)
        }) as Box<dyn FnMut(ArrayView1<f64>) -> (f64, Array1<f64>) + Send>
    };

    let same_family = |_a: &[f64], _b: &[f64]| false;
    let callbacks_per_objective = if grad.is_some() { 2 } else { 1 };
    let problem = EnsembleProblem {
        objective: &make_obj,
        start: &start,
        descriptor: &descriptor,
        context: &context,
        witness: &witness,
        same_family: &same_family,
        certificate: 1e-5,
        polish_below: 1e-3,
        callbacks_per_objective,
    };
    let report = run_ensemble(&cfg, &ens, seed, &problem)?;
    let mut best_pos = Array1::zeros(dim);
    // The ensemble reports energy; recover the design point from the
    // winning replica's hop outcome.
    let mut best_val = report.best;
    let mut found = false;
    for replica in &report.replicas {
        if replica.outcome.best.is_finite() && (!found || replica.outcome.best < best_val) {
            let Some(state) = replica.outcome.best_state.as_ref() else {
                continue;
            };
            best_val = replica.outcome.best;
            for i in 0..dim {
                best_pos[i] = state.get(i).copied().unwrap_or(0.0);
            }
            best_pos = bounds.clip(best_pos.view());
            found = true;
        }
    }
    if !found {
        let start = embed_start(((&bounds.low + &bounds.high) * 0.5).view());
        for i in 0..dim {
            best_pos[i] = start[i];
        }
        best_pos = bounds.clip(best_pos.view());
        // No uncharged Python/PES callback on the fallback path.
        best_val = f64::INFINITY;
    }
    let history_minima = report
        .histories
        .iter()
        .map(|(minima, _, _)| *minima)
        .max()
        .unwrap_or(0);
    Ok(EnsembleHopResult {
        best_pos,
        best_val,
        charged: report.aggregate_charged,
        history_minima,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eindir_core::Bounds;
    use ndarray::{Array1, ArrayView1, array};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Sphere {
        bounds: Bounds<f64>,
    }

    impl Sphere {
        fn new() -> Self {
            Self {
                bounds: Bounds::new(array![-2.0, -2.0, -2.0], array![2.0, 2.0, 2.0], 1e-9),
            }
        }
    }

    impl Objective<f64> for Sphere {
        fn dim(&self) -> usize {
            3
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x.dot(&x)
        }
    }

    impl Gradient<f64> for Sphere {
        fn dim(&self) -> usize {
            3
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            2.0 * &x
        }
    }

    #[test]
    fn production_ensemble_runs_on_a_three_coordinate_box() {
        let obj = Sphere::new();
        let start = array![1.0, 1.0, 1.0];
        let out = ensemble_hop_optimize(
            &obj,
            Some(&obj),
            3,
            Some(start.view()),
            400,
            2,
            HistoryMode::Shared,
            HistoryMembership::Accepted,
        )
        .expect("ensemble hop");
        assert!(out.best_val.is_finite());
        assert!(out.best_val <= 3.0);
        assert!(out.charged > 0);
        assert!(out.best_pos.iter().all(|v| v.abs() <= 2.0 + 1e-8));
    }

    struct SixSphere {
        bounds: Bounds<f64>,
    }

    impl SixSphere {
        fn new() -> Self {
            Self {
                bounds: Bounds::new(Array1::from_elem(6, -2.0), Array1::from_elem(6, 2.0), 1e-9),
            }
        }
    }

    impl Objective<f64> for SixSphere {
        fn dim(&self) -> usize {
            6
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            x.dot(&x)
        }
    }

    impl Gradient<f64> for SixSphere {
        fn dim(&self) -> usize {
            6
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            2.0 * &x
        }
    }

    #[test]
    fn production_ensemble_returns_on_a_six_coordinate_box() {
        let obj = SixSphere::new();
        let start = Array1::from_elem(6, 1.0);
        let out = ensemble_hop_optimize(
            &obj,
            Some(&obj),
            3,
            Some(start.view()),
            400,
            2,
            HistoryMode::Shared,
            HistoryMembership::Accepted,
        )
        .expect("ensemble hop");
        assert!(out.best_val.is_finite());
        assert!(out.charged > 0);
        assert!(out.charged <= 400);
        assert_eq!(out.best_pos.len(), 6);
        assert!(out.best_pos.iter().all(|v| v.abs() <= 2.0 + 1e-8));
    }

    #[test]
    fn padded_box_hop_disables_cluster_only_retries() {
        let cfg = box_hop_config(2, 4.0, 400);
        assert!(!cfg.point_symmetrise_on_new);
        assert!(!cfg.orbit_complete_on_new);
        assert!(!cfg.symmetrise_on_stall);
        assert!(!cfg.point_symmetrise);
        assert!(!cfg.soap_repel);
        assert!(!cfg.restart_on_stall);
        assert!(!cfg.jump_on_stall);
        assert!(!cfg.path_on_stall);
        assert!(!cfg.md_escape);
        assert_eq!(cfg.soap_mode, SoapProposalMode::Off);
    }

    struct CountingSphere {
        bounds: Bounds<f64>,
        calls: AtomicUsize,
    }

    impl CountingSphere {
        fn new() -> Self {
            Self {
                bounds: Bounds::new(Array1::from_elem(6, -2.0), Array1::from_elem(6, 2.0), 1e-9),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl Objective<f64> for CountingSphere {
        fn dim(&self) -> usize {
            6
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            self.calls.fetch_add(1, Ordering::SeqCst);
            x.dot(&x)
        }
    }

    impl Gradient<f64> for CountingSphere {
        fn dim(&self) -> usize {
            6
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            2.0 * &x
        }
    }

    #[test]
    fn every_box_callback_charges_the_ledger() {
        let obj = CountingSphere::new();
        let start = Array1::from_elem(6, 1.0);
        let out = ensemble_hop_optimize(
            &obj,
            Some(&obj),
            3,
            Some(start.view()),
            32,
            2,
            HistoryMode::Shared,
            HistoryMembership::Accepted,
        )
        .expect("ensemble hop");
        let n = obj.calls.load(Ordering::SeqCst);
        assert!(n > 0);
        assert!(out.charged > 0);
        assert!(out.charged <= 32);
        assert!(
            n <= out.charged,
            "uncharged callback: n={n} charged={}",
            out.charged
        );
    }
}
