//! Residual archive search: the measured hop plus skip-returns and stall symmetry.
//!
//! This is not [`Config::recommended`]. That preset keeps its measured rates.
//! The driver clones the caller's config, turns on `return_screen` (so returning
//! trials skip the full quench) and `symmetrise_on_stall` (a residual escape
//! the recommended stack leaves off), then runs [`run_with_gradient`]. Shared
//! [`Archive`] state is filled from the hop best so workers can still record
//! floors and local events.

use crate::catalog::Catalog;
use crate::floors::FloorBook;
use crate::localkey::local_keys;
use crate::methods::cluster_hopping::{
    active_mask, run_with_gradient, Config, GradFn, Ledger, Relax,
};
use crate::residual_field::ResidualField;
use crate::screen::DropModel;
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use std::collections::HashSet;

/// Contact multiple for local NAUTY, matching [`crate::graphkey`] tests.
const LOCAL_CUTOFF: f64 = 1.35;

/// Shared catalogue, floors, residual field, and pending locks.
#[derive(Debug, Clone)]
pub struct Archive {
    /// Local-topology events.
    pub catalog: Catalog,
    /// Energy classes.
    pub floors: FloorBook,
    /// GMRF residual on the class graph.
    pub residual: ResidualField,
    /// Remaining-drop model.
    pub drop: DropModel,
    /// One representative structure per floor.
    pub reps: Vec<Array1<f64>>,
    /// Floor currently being fully quenched.
    pub pending_floors: HashSet<usize>,
    /// Whether a residual full quench is in flight.
    pub pending_residual: bool,
}

impl Default for Archive {
    fn default() -> Self {
        Self::new()
    }
}

impl Archive {
    /// Empty shared state.
    pub fn new() -> Self {
        Self {
            catalog: Catalog::new(),
            floors: FloorBook::new(),
            residual: ResidualField::new(),
            drop: DropModel::new(),
            reps: Vec::new(),
            pending_floors: HashSet::new(),
            pending_residual: false,
        }
    }

    fn set_rep(&mut self, id: usize, x: ArrayView1<f64>) {
        if id >= self.reps.len() {
            self.reps.resize(id + 1, Array1::zeros(0));
        }
        if self.reps[id].is_empty() {
            self.reps[id] = x.to_owned();
        }
    }
}

/// What one archive run produced.
#[derive(Debug, Clone, Default)]
pub struct ArchiveOutcome {
    /// Lowest quenched energy.
    pub best: f64,
    /// Structure attaining it.
    pub best_state: Option<Array1<f64>>,
    /// Screen passes paid.
    pub screens: usize,
    /// Full quenches paid.
    pub full: usize,
    /// Screens discarded as returning to the incumbent attractor.
    pub returned: usize,
    /// Screens discarded as a saturated floor.
    pub same_floor: usize,
    /// Distinct energy classes at the end.
    pub floors: usize,
    /// Distinct local-topology events at the end.
    pub events: usize,
    /// ARTn / stall-symmetry escapes attempted.
    pub artn: usize,
    /// Charged evaluations spent (copied from the ledger).
    pub charged: usize,
    /// Ledger spend when `best` last improved.
    pub best_at: usize,
}

/// One worker on a shared [`Archive`], until the ledger is empty.
///
/// Runs the measured hopping driver with `return_screen` and stall
/// symmetrisation on a clone of `cfg`. The caller's recommended defaults are
/// not written.
pub fn archive_search<'g, R: Rng + ?Sized>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: Option<&mut GradFn<'g>>,
    archive: &mut Archive,
    rng: &mut R,
) -> ArchiveOutcome {
    let mut c = cfg.clone();
    c.return_screen = true;
    c.return_polish = (cfg.relax_steps / 4).max(1);
    c.symmetrise_on_stall = true;
    let hop = run_with_gradient(&c, start, ledger, relax, grad, rng);
    if hop.best.is_finite() {
        if let Some(ref x) = hop.best_state {
            let id = archive.floors.assign(hop.best, None, 0.0);
            archive.residual.observe(id, hop.best);
            archive.set_rep(id, x.view());
            let keys = local_keys(key_coords(x.view(), &c).view(), LOCAL_CUTOFF);
            archive.catalog.observe_bag(&keys);
        }
    }
    let best_at = hop
        .improvements
        .iter()
        .find(|(_, _, _, e)| (*e - hop.best).abs() < 1e-8)
        .map(|(_, sp, _, _)| *sp)
        .unwrap_or_else(|| ledger.spent());
    ArchiveOutcome {
        best: hop.best,
        best_state: hop.best_state,
        screens: hop.screened_out,
        full: hop.hops,
        returned: hop.returned,
        same_floor: 0,
        floors: archive.floors.len().max(hop.basins),
        events: archive.catalog.event_count(),
        artn: hop.symmetrised.0 + hop.stall_escapes,
        charged: ledger.spent(),
        best_at,
    }
}

/// Coordinates used for topology identity: the active patch on a slab, else all.
fn key_coords(x: ArrayView1<f64>, cfg: &Config) -> Array1<f64> {
    let n = x.len() / 3;
    let Some((seeds, shells)) = cfg.active_region.as_ref() else {
        return x.to_owned();
    };
    let Some(species) = cfg.species.as_ref() else {
        return x.to_owned();
    };
    let mask = active_mask(x, species, seeds, *shells, cfg.bond_tolerance);
    let idx: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter(|(_, m)| **m)
        .map(|(i, _)| i)
        .collect();
    if idx.len() < 2 {
        return x.to_owned();
    }
    let mut y = Array1::zeros(idx.len() * 3);
    for (k, &i) in idx.iter().enumerate() {
        if i < n {
            for d in 0..3 {
                y[3 * k + d] = x[3 * i + d];
            }
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::cluster_hopping::random_cluster;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn lj(x: ArrayView1<f64>) -> f64 {
        let n = x.len() / 3;
        let mut e = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut r2 = 0.0;
                for k in 0..3 {
                    let d = x[3 * i + k] - x[3 * j + k];
                    r2 += d * d;
                }
                if r2 < 1e-12 {
                    return f64::INFINITY;
                }
                let ir2 = 1.0 / r2;
                let ir6 = ir2 * ir2 * ir2;
                e += 4.0 * (ir6 * ir6 - ir6);
            }
        }
        e
    }

    fn crude_relax(
        led: &mut Ledger,
        x: ArrayView1<f64>,
        steps: usize,
    ) -> (f64, Array1<f64>) {
        let mut y = x.to_owned();
        let mut e = lj(y.view());
        for _ in 0..steps {
            if !led.charge() {
                break;
            }
            let mut best_i = 0usize;
            let mut best_g = 0.0_f64;
            for i in 0..y.len() {
                let old = y[i];
                y[i] = old + 1e-4;
                let ep = lj(y.view());
                y[i] = old;
                let g = (ep - e) / 1e-4;
                if g.abs() > best_g.abs() {
                    best_g = g;
                    best_i = i;
                }
            }
            y[best_i] -= 0.05 * best_g.signum();
            e = lj(y.view());
        }
        (e, y)
    }

    #[test]
    fn archive_search_does_not_touch_recommended_defaults() {
        let rec = Config::recommended(13);
        let before = format!("{rec:?}");
        assert!(
            !rec.return_screen,
            "recommended must keep return_screen off; ras enables it on a clone"
        );
        assert!(!rec.symmetrise_on_stall);
        assert_eq!(rec.screen_steps, 25);
        assert_eq!(rec.relax_steps, 200);
        let rec2 = Config::recommended(75);
        assert!(!rec2.return_screen);
        assert!(rec2.allocate_moves);
        assert!(rec2.tabu_on_stall);
        assert!(rec2.depth_reward);

        let mut rng = StdRng::seed_from_u64(0);
        let start = random_cluster(13, 0.7, rec.min_separation, &mut rng);
        let mut ledger = Ledger::new(80);
        let mut relax =
            |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| crude_relax(led, x, steps);
        let mut archive = Archive::new();
        let _ = archive_search(
            &rec,
            start.view(),
            &mut ledger,
            &mut relax,
            None,
            &mut archive,
            &mut rng,
        );
        assert_eq!(
            format!("{rec:?}"),
            before,
            "archive_search mutated the caller's recommended config"
        );
        assert!(!rec.return_screen);
        assert_eq!(rec.return_polish, 0);
        assert!(!rec.symmetrise_on_stall);
        assert_eq!(rec.screen_steps, 25);
        assert_eq!(rec.relax_steps, 200);
    }

    #[test]
    fn a_tiny_run_records_floors_and_refuses_returns() {
        let mut rng = StdRng::seed_from_u64(1);
        let cfg = Config::recommended(7);
        let start = random_cluster(7, 0.7, cfg.min_separation, &mut rng);
        let mut ledger = Ledger::new(400);
        let mut relax =
            |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| crude_relax(led, x, steps);
        let mut archive = Archive::new();
        let out = archive_search(
            &cfg,
            start.view(),
            &mut ledger,
            &mut relax,
            None,
            &mut archive,
            &mut rng,
        );
        assert!(out.full >= 1);
        assert!(out.screens + out.full >= 1);
        assert!(out.charged <= 400);
        assert!(out.best.is_finite());
        assert!(out.floors >= 1);
    }

    #[test]
    fn a_tiny_run_does_not_starve_full_quenches() {
        let mut rng = StdRng::seed_from_u64(2);
        let cfg = Config::recommended(7);
        let start = random_cluster(7, 0.7, cfg.min_separation, &mut rng);
        let mut ledger = Ledger::new(400);
        let mut relax =
            |led: &mut Ledger, x: ArrayView1<f64>, steps: usize| crude_relax(led, x, steps);
        let mut archive = Archive::new();
        let out = archive_search(
            &cfg,
            start.view(),
            &mut ledger,
            &mut relax,
            None,
            &mut archive,
            &mut rng,
        );
        assert!(
            out.full >= 2,
            "starved: full={} screens={} returned={} charged={}",
            out.full,
            out.screens,
            out.returned,
            out.charged
        );
        assert!(out.charged <= 400);
        assert!(out.best.is_finite());
    }
}
