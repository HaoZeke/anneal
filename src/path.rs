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
//! Fully relaxing an interpolated image does not do this, and the measurement is
//! unambiguous. Between the structure a 75-point search settles into and each of
//! four constructed morphologies, every image relaxed either back to the start
//! or onto the endpoint: the deepest structure each path produced was that
//! path's own endpoint, to every printed digit. A fully relaxed image slides
//! into whichever basin owns it and the corridor is never sampled.
//!
//! [`transverse_path`] removes the component of the gradient along the path
//! before descending, so an image falls to the valley floor while staying where
//! it was placed. That projection is what a nudged elastic band uses and the
//! reason it is used.
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

/// Descends an image perpendicular to the path tangent.
///
/// Relaxing an interpolated image fully is not a path method and the
/// measurement says so plainly: between the structure a 75-point search settles
/// into and each of four constructed morphologies, every image relaxed either
/// back to the start or onto the endpoint, and the deepest structure any path
/// produced was the endpoint itself. A fully relaxed image slides into whichever
/// basin owns it, so the corridor is never sampled.
///
/// Removing the tangential component leaves the image free to fall into the
/// valley floor while staying where it was placed along the path, which is the
/// projection a nudged elastic band uses and the reason it is used.
fn project_out_tangent(g: &mut Array1<f64>, tangent: ArrayView1<f64>) {
    let norm2: f64 = tangent.iter().map(|t| t * t).sum();
    if norm2 <= 1e-30 {
        return;
    }
    let dot: f64 = g.iter().zip(tangent.iter()).map(|(a, b)| a * b).sum();
    let coeff = dot / norm2;
    for i in 0..g.len() {
        g[i] -= coeff * tangent[i];
    }
}

/// Relaxes images perpendicular to the path between two structures.
///
/// `grad` returns the value and gradient at a point, charging the caller's
/// ledger, and `None` when the budget is spent.
///
/// Each image descends with the component of the gradient along the path
/// removed, so it settles into the valley floor at its own position rather than
/// sliding to an endpoint. The images that come out are candidate structures
/// along the corridor; a caller relaxes the promising ones fully.
pub fn transverse_path<G>(
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    n_images: usize,
    steps: usize,
    step_size: f64,
    mut grad: G,
) -> Vec<(f64, Array1<f64>)>
where
    G: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
{
    if a.len() != b.len() || n_images == 0 {
        return Vec::new();
    }
    let mut images: Vec<Array1<f64>> = (0..n_images)
        .map(|k| {
            let lambda = (k + 1) as f64 / (n_images + 1) as f64;
            let mut image = Array1::zeros(a.len());
            for i in 0..a.len() {
                image[i] = (1.0 - lambda) * a[i] + lambda * b[i];
            }
            image
        })
        .collect();

    let mut values = vec![f64::INFINITY; n_images];
    'outer: for _ in 0..steps {
        for k in 0..n_images {
            // Tangent from the neighbouring images, and from the endpoints at
            // the ends, so the band keeps its direction rather than curling.
            let prev: Array1<f64> = if k == 0 {
                a.to_owned()
            } else {
                images[k - 1].clone()
            };
            let next: Array1<f64> = if k + 1 == n_images {
                b.to_owned()
            } else {
                images[k + 1].clone()
            };
            let mut tangent = next.clone();
            for i in 0..tangent.len() {
                tangent[i] -= prev[i];
            }
            let Some((f, mut g)) = grad(images[k].view()) else {
                break 'outer;
            };
            values[k] = f;
            project_out_tangent(&mut g, tangent.view());
            let gmax = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            // Scaled by the gradient size: an image between two minima can sit
            // on a repulsive wall where the gradient is enormous, and a fixed
            // step there moves it nowhere useful.
            let scale = if gmax > 1.0 { step_size / gmax } else { step_size };
            for i in 0..images[k].len() {
                images[k][i] -= scale * g[i];
            }
        }
    }
    values.into_iter().zip(images).collect()
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

#[cfg(test)]
mod transverse_tests {
    use super::*;
    use ndarray::Array1;

    /// A two-dimensional surface with a valley between two minima, so an image
    /// placed on the ridge should fall into the valley and stay between them
    /// rather than sliding to an end.
    fn valley(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        // Minima near (-2, 0) and (2, 0); y is a stiff transverse direction.
        let (a, b) = (x[0], x[1]);
        let f = 0.1 * (a * a - 4.0).powi(2) + 3.0 * b * b;
        let g = Array1::from(vec![0.4 * a * (a * a - 4.0), 6.0 * b]);
        (f, g)
    }

    #[test]
    fn tangential_motion_is_removed() {
        let mut g = Array1::from(vec![3.0, 4.0]);
        let t = Array1::from(vec![1.0, 0.0]);
        project_out_tangent(&mut g, t.view());
        assert!(g[0].abs() < 1e-12, "tangential part survived: {}", g[0]);
        assert!((g[1] - 4.0).abs() < 1e-12, "transverse part was altered");
    }

    /// The property the function exists for: images stay spread along the path
    /// instead of collapsing onto the endpoints.
    #[test]
    fn images_stay_spread_along_the_path() {
        let a = Array1::from(vec![-2.0, 0.0]);
        let b = Array1::from(vec![2.0, 0.0]);
        // Displaced off the valley floor, so there is something to relax.
        let out = transverse_path(a.view(), b.view(), 7, 200, 0.05, |v| {
            let mut w = v.to_owned();
            w[1] += 0.8;
            let (f, g) = valley(w.view());
            Some((f, g))
        });
        assert_eq!(out.len(), 7);
        let xs: Vec<f64> = out.iter().map(|(_, s)| s[0]).collect();
        // Every image must still lie strictly between the endpoints.
        for x in &xs {
            assert!(
                *x > -2.0 && *x < 2.0,
                "an image left the interval at {x}, which is the sliding this prevents"
            );
        }
        // And they must remain distinct, not piled at one point.
        let spread = xs.iter().cloned().fold(f64::MIN, f64::max)
            - xs.iter().cloned().fold(f64::MAX, f64::min);
        assert!(spread > 1.0, "images collapsed together, spread {spread}");
    }

    #[test]
    fn an_exhausted_budget_stops_the_band() {
        let a = Array1::from(vec![-2.0, 0.0]);
        let b = Array1::from(vec![2.0, 0.0]);
        let mut left = 3;
        let out = transverse_path(a.view(), b.view(), 5, 100, 0.05, |v| {
            if left == 0 {
                return None;
            }
            left -= 1;
            Some(valley(v))
        });
        assert_eq!(out.len(), 5, "the band should still report its images");
    }
}
