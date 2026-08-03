//! Multi-step paths between minima, for landscapes where one hop cannot cross.
//!
//! Basin hopping is a breadth-first search of depth one: propose, relax, accept.
//! On a multi-funnel landscape that is the wrong depth, and the measurement says
//! so rather than the intuition. From the structure a 75-point Lennard-Jones
//! search settles into, none of 1800 single moves across the whole kernel set
//! reaches anything lower; every move that leaves the basin lands higher, and
//! the moves that land lowest do not leave. The target is 1.2 below and no
//! single proposal goes there.
//!
//! No bias, temperature law, collective variable or acquisition rule creates a
//! path that is not in the move graph. What is needed is a search of depth
//! greater than one: a sequence of intermediate structures between where the
//! chain is and somewhere structurally different, each relaxed, so the corridor
//! between two funnels is examined rather than jumped.
//!
//! This is the operation behind funnel hopping in the cluster literature, and
//! it is cheap to state: interpolate between two minima, relax the images, and
//! keep any that lands outside the basin it started from. A relaxation from a
//! point partway along lands in whichever basin claims that point, which need
//! not be either endpoint's, and that is the escape a perturbation cannot make.
//!
//! Two things make it work rather than merely run.
//!
//! The endpoints must be structurally different. Interpolating between two
//! structures in one funnel lands back in it, which is consistent with the
//! archive-based escape moves in this campaign scoring zero of eight on LJ75
//! while holding only structures the chain itself had reached.
//!
//! The images must be relaxed individually and charged. An unrelaxed image is
//! not a minimum and says nothing about which basin it belongs to, and a path
//! whose cost is not charged is not comparable with the hopping it replaces.

use ndarray::{Array1, ArrayView1};

/// A minimum found along a path, with where it came from.
#[derive(Debug, Clone)]
pub struct PathPoint {
    /// Interpolation fraction the image started at, in `(0, 1)`.
    pub lambda: f64,
    /// Relaxed value.
    pub energy: f64,
    /// Relaxed structure.
    pub state: Array1<f64>,
}

/// What a path attempt produced.
#[derive(Debug, Clone)]
pub struct PathOutcome {
    /// Every image that relaxed successfully, in order of `lambda`.
    pub points: Vec<PathPoint>,
    /// The deepest of them, if any.
    pub best: Option<usize>,
    /// Images that left the starting basin, by the caller's test.
    pub escapes: Vec<usize>,
}

impl PathOutcome {
    /// The deepest structure that also left the starting basin.
    ///
    /// This, not the deepest overall, is what a stuck chain wants: the deepest
    /// image is usually a relaxation back into the basin the path started in.
    pub fn best_escape(&self) -> Option<&PathPoint> {
        self.escapes
            .iter()
            .map(|i| &self.points[*i])
            .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
    }
}

/// Relaxes images interpolated between two structures.
///
/// `relax` returns the value and relaxed structure, charging the caller's
/// ledger, and returns `None` when the budget is spent, which ends the path
/// where it stands rather than silently returning a shorter one as complete.
///
/// `left_start` decides whether a relaxed image is outside the basin the path
/// began in. It is the caller's because basin identity is: on a cluster it is a
/// shape distance against `a`, and a threshold there is a length.
///
/// Images are taken away from the endpoints. A fraction near zero relaxes back
/// to `a` and one near one to `b`, and neither is a crossing.
pub fn interpolate_path<R, E>(
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    n_images: usize,
    mut relax: R,
    mut left_start: E,
) -> PathOutcome
where
    R: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    E: FnMut(ArrayView1<f64>) -> bool,
{
    let mut out = PathOutcome {
        points: Vec::new(),
        best: None,
        escapes: Vec::new(),
    };
    if a.len() != b.len() || n_images == 0 {
        return out;
    }
    for k in 0..n_images {
        // Interior fractions only, evenly spaced, excluding both endpoints.
        let lambda = (k + 1) as f64 / (n_images + 1) as f64;
        let mut image = Array1::zeros(a.len());
        for i in 0..a.len() {
            image[i] = (1.0 - lambda) * a[i] + lambda * b[i];
        }
        let Some((energy, state)) = relax(image.view()) else {
            break;
        };
        if !energy.is_finite() {
            continue;
        }
        let escaped = left_start(state.view());
        out.points.push(PathPoint {
            lambda,
            energy,
            state,
        });
        let idx = out.points.len() - 1;
        if escaped {
            out.escapes.push(idx);
        }
        match out.best {
            Some(b) if out.points[b].energy <= energy => {}
            _ => out.best = Some(idx),
        }
    }
    out
}

/// Tracks whether a chain has stopped making progress, so a path is attempted
/// when hopping has stalled rather than on a schedule.
///
/// A path costs many relaxations and is worth paying for only when the cheap
/// mechanism has stopped working. The signal is the one this landscape gives
/// plainly: near a deep minimum roughly nineteen proposals in twenty return to
/// where they started, and the incumbent stops moving.
#[derive(Debug, Clone)]
pub struct StallDetector {
    /// Hops without an improvement before the chain counts as stalled.
    pub patience: usize,
    since_improvement: usize,
    best: f64,
    /// Times a stall was reported.
    pub stalls: usize,
}

impl StallDetector {
    /// Detector that reports a stall after `patience` hops without progress.
    pub fn new(patience: usize) -> Self {
        Self {
            patience,
            since_improvement: 0,
            best: f64::INFINITY,
            stalls: 0,
        }
    }

    /// Records a hop's value. Returns whether the chain is now stalled.
    ///
    /// Reporting resets the counter, so a stall is reported once per stretch
    /// rather than on every hop after the threshold.
    pub fn observe(&mut self, energy: f64) -> bool {
        if energy < self.best - 1e-10 {
            self.best = energy;
            self.since_improvement = 0;
            return false;
        }
        self.since_improvement += 1;
        if self.since_improvement >= self.patience {
            self.since_improvement = 0;
            self.stalls += 1;
            return true;
        }
        false
    }

    /// Hops since the last improvement.
    pub fn since_improvement(&self) -> usize {
        self.since_improvement
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-dimensional triple well, so which basin an image falls into is
    /// known in advance and the test checks the search rather than a guess.
    ///
    /// Minima near -3, 0 and +3, with the outer ones deeper.
    fn triple_well(x: f64) -> f64 {
        0.02 * x.powi(4) - 0.35 * x.powi(2) + 0.02 * x
    }

    /// Steepest descent on the well, which is enough in one dimension.
    fn relax_1d(start: f64) -> (f64, f64) {
        let mut x = start;
        for _ in 0..2000 {
            let g = 0.08 * x.powi(3) - 0.7 * x + 0.02;
            x -= 0.01 * g;
        }
        (triple_well(x), x)
    }

    #[test]
    fn a_path_finds_a_basin_neither_endpoint_is_in() {
        // Endpoints in the two outer wells; the middle well lies between them
        // and no interpolation endpoint is in it.
        let a = Array1::from(vec![-3.0]);
        let b = Array1::from(vec![3.0]);
        let out = interpolate_path(
            a.view(),
            b.view(),
            9,
            |v| {
                let (e, x) = relax_1d(v[0]);
                Some((e, Array1::from(vec![x])))
            },
            |s| (s[0] - (-3.0)).abs() > 0.5,
        );
        assert!(!out.points.is_empty(), "no images relaxed");
        // At least one image must land somewhere that is not the left well.
        assert!(
            !out.escapes.is_empty(),
            "no image left the starting basin: {:?}",
            out.points.iter().map(|p| p.state[0]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn images_exclude_the_endpoints() {
        let a = Array1::from(vec![0.0]);
        let b = Array1::from(vec![1.0]);
        let out = interpolate_path(
            a.view(),
            b.view(),
            4,
            |v| Some((v[0], v.to_owned())),
            |_| false,
        );
        assert_eq!(out.points.len(), 4);
        for p in &out.points {
            assert!(
                p.lambda > 0.0 && p.lambda < 1.0,
                "lambda {} is an endpoint, which relaxes back to where it came from",
                p.lambda
            );
        }
    }

    #[test]
    fn the_best_escape_is_not_the_best_point() {
        // The deepest image relaxes back to the start, which is the case
        // best_escape exists to avoid returning.
        let a = Array1::from(vec![0.0]);
        let b = Array1::from(vec![4.0]);
        let out = interpolate_path(
            a.view(),
            b.view(),
            3,
            |v| {
                // Deepest at the first image, which stays in the start basin.
                let e = if v[0] < 1.5 { -10.0 } else { -1.0 };
                Some((e, v.to_owned()))
            },
            |s| s[0] >= 1.5,
        );
        assert_eq!(out.points[out.best.unwrap()].energy, -10.0);
        let esc = out.best_escape().expect("an escape should exist");
        assert_eq!(esc.energy, -1.0, "best_escape returned the deepest instead");
    }

    #[test]
    fn an_exhausted_budget_stops_the_path() {
        let a = Array1::from(vec![0.0]);
        let b = Array1::from(vec![1.0]);
        let mut left = 2;
        let out = interpolate_path(
            a.view(),
            b.view(),
            10,
            |v| {
                if left == 0 {
                    return None;
                }
                left -= 1;
                Some((v[0], v.to_owned()))
            },
            |_| false,
        );
        assert_eq!(out.points.len(), 2, "the path ran past the budget");
    }

    #[test]
    fn a_stall_is_reported_once_per_stretch() {
        let mut d = StallDetector::new(3);
        assert!(!d.observe(-1.0));
        assert!(!d.observe(-0.5));
        assert!(!d.observe(-0.5));
        assert!(d.observe(-0.5), "three flat hops should stall");
        assert!(!d.observe(-0.5), "the stall should not repeat immediately");
        assert_eq!(d.stalls, 1);
    }

    #[test]
    fn improvement_clears_the_stall_counter() {
        let mut d = StallDetector::new(3);
        d.observe(-1.0);
        d.observe(-1.0);
        assert_eq!(d.since_improvement(), 1);
        d.observe(-2.0);
        assert_eq!(d.since_improvement(), 0, "progress must reset the count");
    }
}
