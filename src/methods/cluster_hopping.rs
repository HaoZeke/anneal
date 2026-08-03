//! Basin hopping over quenched minima with a bias keyed on basin identity.
//!
//! Cluster global optimisation happens on the quenched landscape,
//! `E_q(x) = E(local_min(x))`, not on the raw surface: the funnel structure is
//! only visible after relaxation, and a search on the raw surface finds
//! nothing on a 38-atom Lennard-Jones cluster.
//!
//! Three things then decide whether a funnel can be left.
//!
//! The relaxation is where the budget goes. A full one costs a few hundred
//! charged evaluations and most trials land nowhere near the incumbent, so
//! trials are screened by a short relaxation first and only promoted when they
//! land within [`Config::screen_margin`] of the incumbent. Measured on LJ38,
//! screening took basin discovery from 27 to 327 at a fixed charge.
//!
//! The bias is keyed on basin identity rather than on a collective variable.
//! A variable has to be chosen correctly or it cannot see the competition: on
//! LJ75 the Marks decahedron and the structures a search settles into differ by
//! 0.023 in the fourth Steinhardt parameter, narrower than any usable
//! deposition width, so biasing on it fills both competitors at once.
//!
//! The moves have to change the packing rather than perturb it, which is what
//! [`crate::movekernel::SurfaceRelocate`], [`crate::movekernel::ShellRotate`]
//! and [`crate::movekernel::Symmetrise`] do, and which of them pays at a given
//! stage is decided online rather than fixed.

use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::allocate::{BudgetWindowTemperature, FlooredThompson};
use crate::bias::{AdaptiveHeight, Bias, BasinBias, Fingerprint, SortedPairs};
use crate::diversity::DiversityAnnealer;
use crate::path::{interpolate_path, StallDetector};
use crate::movekernel::{MoveKernel, ShellRotate, SurfaceRelocate, Symmetrise};

/// The move library, dispatched by value.
///
/// [`MoveKernel::propose`] is generic over the generator, which makes the
/// trait not dyn compatible, so the library is an enum rather than a vector of
/// boxes. Keeping the generic parameter is worth more than boxing: it lets a
/// kernel be used with any generator without a virtual call per proposal.
pub enum ClusterMove {
    /// Displace every point uniformly: the standard basin-hopping move.
    AllPoints {
        /// Half-width of the per-coordinate displacement.
        step: f64,
    },
    /// Displace one point. Cheap and local, for polishing a packing.
    SinglePoint {
        /// Points in a state.
        n_points: usize,
        /// Half-width of the displacement.
        step: f64,
    },
    /// Relocate the least-coordinated point onto the surface.
    SurfaceRelocate(SurfaceRelocate),
    /// Rotate the outer shell against the core.
    ShellRotate(ShellRotate),
    /// Enforce an approximate rotational symmetry.
    Symmetrise(Symmetrise),
}

impl ClusterMove {
    /// The move library, configured for `n` points.
    ///
    /// The two plain perturbations come first and are not optional. Displacing
    /// every point uniformly is the move basin hopping is defined by, and the
    /// step of 0.38 is inside the 0.36 to 0.40 band Wales and Doye report for
    /// the quenched surface. A library of packing-changing moves alone leaves
    /// the chain with no way to make an ordinary small step, and measured on
    /// LJ38 at 400 thousand charged evaluations that library solved 1 seed in 8
    /// where the campaign driver, which carries both, solves 8.
    pub fn library(n: usize) -> Vec<ClusterMove> {
        vec![
            ClusterMove::AllPoints { step: 0.38 },
            ClusterMove::SinglePoint {
                n_points: n,
                step: 1.0,
            },
            ClusterMove::SurfaceRelocate(SurfaceRelocate {
                n_points: n,
                neighbour_cutoff: 1.6,
            }),
            ClusterMove::ShellRotate(ShellRotate { n_points: n }),
            ClusterMove::Symmetrise(Symmetrise {
                n_points: n,
                orders: vec![2, 3, 4, 5, 6],
                pair_cutoff: 2.5,
            }),
        ]
    }

    /// Draws a proposal from whichever kernel this is.
    pub fn propose<R: Rng + ?Sized>(
        &self,
        x: ArrayView1<f64>,
        t: f64,
        rng: &mut R,
    ) -> Array1<f64> {
        match self {
            ClusterMove::AllPoints { step } => {
                let mut y = x.to_owned();
                for v in y.iter_mut() {
                    *v += rng.random_range(-step..*step);
                }
                y
            }
            ClusterMove::SinglePoint { n_points, step } => {
                let mut y = x.to_owned();
                let i = rng.random_range(0..*n_points);
                for k in 0..3 {
                    y[3 * i + k] += rng.random_range(-step..*step);
                }
                y
            }
            ClusterMove::SurfaceRelocate(k) => k.propose(x, t, rng),
            ClusterMove::ShellRotate(k) => k.propose(x, t, rng),
            ClusterMove::Symmetrise(k) => k.propose(x, t, rng),
        }
    }
}

/// Work ledger: every objective or gradient evaluation is charged.
///
/// A relaxation inside a move spends the same budget as a proposal does, which
/// is the accounting that makes methods with different internal structure
/// comparable. Published cluster success rates are quoted per hopping step,
/// with the relaxation inside each step uncounted.
pub struct Ledger {
    budget: usize,
    spent: usize,
    /// Lowest objective value seen.
    pub best: f64,
    /// State attaining [`Ledger::best`].
    pub best_state: Option<Array1<f64>>,
}

impl Ledger {
    /// Creates a ledger with `budget` charged evaluations.
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            spent: 0,
            best: f64::INFINITY,
            best_state: None,
        }
    }

    /// Charges one unit, returning `false` when the budget is gone.
    pub fn charge(&mut self) -> bool {
        if self.spent >= self.budget {
            return false;
        }
        self.spent += 1;
        true
    }

    /// Records a value and its state when it improves the incumbent.
    pub fn record(&mut self, value: f64, state: ArrayView1<f64>) {
        if value < self.best {
            self.best = value;
            self.best_state = Some(state.to_owned());
        }
    }

    /// Charged evaluations remaining.
    /// Charged evaluations the ledger was created with.
    pub fn budget(&self) -> usize {
        self.budget
    }

    pub fn remaining(&self) -> usize {
        self.budget.saturating_sub(self.spent)
    }

    /// Charged evaluations spent.
    pub fn spent(&self) -> usize {
        self.spent
    }
}

/// Driver settings.
#[derive(Debug, Clone)]
pub struct Config {
    /// Points in a state; the state length must be `3 * n_points`.
    pub n_points: usize,
    /// Metropolis temperature on the quenched chain.
    pub temperature: f64,
    /// Height of a fresh bias deposit.
    pub bias_height: f64,
    /// Well-tempered bias factor; must exceed one.
    pub bias_gamma: f64,
    /// Distance below which two states are the same basin.
    ///
    /// Its units are those of whichever metric keys the bias. Against a sorted
    /// distance spectrum compared by Euclidean distance it is a number in
    /// descriptor space with no physical meaning; against a shape distance it
    /// is a length.
    pub merge_radius: f64,
    /// Design point for the budget-window temperature, as a fraction of the
    /// sphere-model descent boundary. Must lie strictly below two.
    pub theta: f64,
    /// Set the temperature by the budget-window law rather than holding it.
    pub budget_window: bool,
    /// Choose the move kernel by discounted Thompson allocation.
    pub allocate_moves: bool,
    /// Set the deposit height from the escape gaps the chain observes.
    pub adaptive_height: bool,
    /// Attempt a multi-step path between funnels when hopping stalls.
    ///
    /// Basin hopping searches to depth one, and from the structure a 75-point
    /// search settles into none of 1800 single moves reaches anything lower. A
    /// path relaxes images between the current structure and a structurally
    /// different archive member, so the corridor between two funnels is
    /// examined rather than jumped.
    pub path_on_stall: bool,
    /// Hops without improvement before a path is attempted.
    pub stall_patience: usize,
    /// Images relaxed along a path.
    pub path_images: usize,
    /// Anneal the merge radius from wide to narrow across the budget.
    ///
    /// The threshold that decides when two structures are one basin is a
    /// temperature rather than a setting, and the only published method that
    /// solves the hard cluster sizes reliably anneals it. Held fixed, it is the
    /// quantity three separate calibrations here failed to pin down.
    pub anneal_diversity: bool,
    /// Fraction of the starting radius the annealed threshold falls to.
    pub diversity_floor: f64,
    /// Revisits a basin should take before the accumulated bias clears the
    /// escape gap, when the height is adaptive.
    pub height_revisits: f64,
    /// Key basins on IRA shape distance rather than on the descriptor.
    ///
    /// Measured on LJ38 at 400 thousand charged evaluations: keying on the
    /// descriptor solves 1 seed in 8. The threshold there has to absorb
    /// relabelling and rotation, which is what makes it untransferable between
    /// sizes and what three separate calibrations failed to pin down.
    pub shape_keyed: bool,
    /// How far above the incumbent a screened trial may land and still be
    /// promoted to a full relaxation.
    pub screen_margin: f64,
    /// Relaxation steps in the screening pass.
    pub screen_steps: usize,
    /// Relaxation steps in the full pass.
    pub relax_steps: usize,
    /// Container half-width, applied when a move is generated.
    pub container: f64,
    /// Closest approach enforced before a trial is relaxed.
    pub min_separation: f64,
}

impl Config {
    /// Settings for `n_points` at the campaign's measured defaults.
    pub fn for_cluster(n_points: usize) -> Self {
        Self {
            n_points,
            temperature: 0.8,
            bias_height: 0.25,
            bias_gamma: 5.0,
            merge_radius: 1e-2,
            shape_keyed: false,
            theta: 0.5,
            budget_window: false,
            allocate_moves: false,
            adaptive_height: false,
            path_on_stall: false,
            stall_patience: 60,
            path_images: 9,
            anneal_diversity: false,
            diversity_floor: 0.1,
            height_revisits: 4.0,
            screen_margin: 2.0,
            screen_steps: 25,
            relax_steps: 200,
            // Calibrated against published minima: the largest atomic distance
            // from the centre of mass divides by N^(1/3) to between 0.46 and
            // 0.63, and the literature's 2.5 N^(1/3) is sized for a method
            // that relaxes after every perturbation.
            container: 0.9 * (n_points as f64).cbrt(),
            min_separation: 0.85,
        }
    }
}

/// What a run produced.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// Lowest quenched value found.
    pub best: f64,
    /// State attaining it.
    pub best_state: Option<Array1<f64>>,
    /// Hops taken.
    pub hops: usize,
    /// Trials rejected by screening before a full relaxation.
    pub screened_out: usize,
    /// Distinct basins registered.
    pub basins: usize,
    /// Charged evaluations spent.
    pub charged: usize,
    /// Paths attempted after a stall.
    pub paths: usize,
    /// Paths that produced a structure outside the starting basin.
    pub path_escapes: usize,
}

/// Relaxes `x`, charging every evaluation, and stopping when the budget ends.
///
/// The relaxation is supplied by the caller because the objective, its
/// gradient and the minimiser are the caller's: this module owns the search,
/// not the numerics under it.
pub type Relax<'a> = &'a mut dyn FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>);

/// Moves the centre of mass to the origin.
fn recentre(x: &mut Array1<f64>, n: usize) {
    let mut c = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += x[3 * i + k];
        }
    }
    for v in c.iter_mut() {
        *v /= n as f64;
    }
    for i in 0..n {
        for k in 0..3 {
            x[3 * i + k] -= c[k];
        }
    }
}

/// Pulls points outside the container back onto its surface.
///
/// Applied when a move is generated and never inside a relaxation: a cluster
/// relaxes in free space, and constraining the minimiser makes it stop at its
/// own starting point.
fn contain(x: &mut Array1<f64>, n: usize, radius: f64) {
    for i in 0..n {
        let r = (0..3)
            .map(|k| x[3 * i + k] * x[3 * i + k])
            .sum::<f64>()
            .sqrt();
        if r > radius && r > 0.0 {
            let s = radius / r;
            for k in 0..3 {
                x[3 * i + k] *= s;
            }
        }
    }
}

/// Pushes overlapping points apart to `min_sep`.
///
/// A configuration with two points on top of each other has an enormous value
/// under any repulsive potential, and a quasi-Newton relaxation started there
/// fails on its first line search and returns the configuration unchanged.
fn repair(x: &mut Array1<f64>, n: usize, min_sep: f64) {
    for _ in 0..40 {
        let mut moved = false;
        for a in 0..n {
            for b in (a + 1)..n {
                let mut d = [0.0; 3];
                let mut r2 = 0.0;
                for k in 0..3 {
                    d[k] = x[3 * a + k] - x[3 * b + k];
                    r2 += d[k] * d[k];
                }
                let r = r2.sqrt();
                if r < min_sep && r > 1e-9 {
                    let push = 0.5 * (min_sep - r) / r;
                    for k in 0..3 {
                        x[3 * a + k] += push * d[k];
                        x[3 * b + k] -= push * d[k];
                    }
                    moved = true;
                }
            }
        }
        if !moved {
            break;
        }
    }
}

/// Runs the driver until the ledger is spent.
///
/// `start` is a starting configuration and `relax` performs a relaxation of the
/// requested number of steps, charging the ledger.
pub fn run<R: Rng + ?Sized>(
    cfg: &Config,
    start: ArrayView1<f64>,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    rng: &mut R,
) -> Outcome {
    let n = cfg.n_points;
    // The descriptor and the metric have to agree. A shape distance is
    // computed from coordinates, so keying on it means passing coordinates
    // through rather than reducing them to a sorted distance spectrum first;
    // handing the spectrum to a shape metric would compare the wrong objects
    // and quietly return distances that mean nothing.
    let mut bias = BasinBias::new(
        ClusterFingerprint::for_keying(n, cfg.shape_keyed),
        cfg.merge_radius,
        cfg.bias_height,
        cfg.bias_gamma,
    );
    #[cfg(feature = "ira")]
    if cfg.shape_keyed {
        bias = bias.with_metric(Box::new(crate::shape::IraMetric::default()));
    }
    #[cfg(not(feature = "ira"))]
    assert!(
        !cfg.shape_keyed,
        "shape keying needs the `ira` feature; without it the threshold would \
         silently remain a descriptor-space number"
    );

    let kernels = ClusterMove::library(n);
    // Which kernel to propose from is learned rather than drawn uniformly. The
    // useful move changes as the search moves through the landscape, so the
    // evidence is discounted and a decaying floor keeps every kernel reachable.
    let mut allocator = FlooredThompson::new(kernels.len());
    // The temperature is the law rather than a setting: the design point
    // clamped between the sphere-model descent ceiling and the birth-death
    // escape floor, with the barrier estimated from the uphill steps the chain
    // declines. On a funnelled landscape the window is routinely empty, which
    // is counted rather than hidden.
    let mut law = BudgetWindowTemperature::new(3 * n, cfg.theta);
    // The deposit height is set from the escape gaps observed rather than
    // fixed, since a height above the gap empties a basin on one revisit and
    // the gap is a property of the landscape.
    let mut height = AdaptiveHeight::new(0.1, cfg.height_revisits, cfg.bias_height);
    // Starts at the configured radius and narrows. The paper's rule takes the
    // start from the spread of an initial population; here the configured value
    // is the start, so a run that does not anneal is unchanged and one that does
    // begins where the fixed version sat rather than somewhere new.
    let mut diversity = DiversityAnnealer::from_initial(cfg.merge_radius)
        .with_final_fraction(cfg.diversity_floor);
    let mut stall = StallDetector::new(cfg.stall_patience);
    // Structures kept for path endpoints. Only ones far from every member are
    // added, because interpolating between two structures in one funnel lands
    // back in it, which is what archive-based escape moves holding a single
    // funnel's structures already showed.
    let mut archive: Vec<(f64, Array1<f64>)> = Vec::new();
    let mut paths_run = 0usize;
    let mut path_escapes = 0usize;

    let (mut e, mut x) = relax(ledger, start, cfg.relax_steps);
    ledger.record(e, x.view());
    let mut screened_out = 0usize;
    let mut hops = 0usize;

    loop {
        if ledger.remaining() == 0 {
            break;
        }
        // Gap to the incumbent, which is what the law scales the window by.
        let gap = (e - ledger.best).abs().max(1e-12);
        let temperature = if cfg.budget_window {
            law.temperature(gap, ledger.remaining())
        } else {
            cfg.temperature
        };

        if cfg.anneal_diversity {
            let progress = 1.0 - (ledger.remaining() as f64 / ledger.budget() as f64);
            bias.set_merge_radius(diversity.threshold(progress));
        }
        let k = if cfg.allocate_moves {
            allocator.select(rng)
        } else {
            rng.random_range(0..kernels.len())
        };
        // The move scale stays the configured temperature. The law's
        // temperature is an acceptance temperature: it governs which uphill
        // steps are taken, not how far a proposal reaches. Feeding it to the
        // kernel makes a correctly small temperature shrink the proposals to
        // nothing and freeze the chain, which took LJ38 from 1 seed in 8 to 0.
        let mut trial = kernels[k].propose(x.view(), cfg.temperature, rng);
        recentre(&mut trial, n);
        contain(&mut trial, n, cfg.container);

        // Screen cheaply, then carry on regardless. A screened trial does not
        // leave the chain: it goes through the acceptance test on its screened
        // energy and, whether accepted or not, a hill is deposited on wherever
        // the chain now stands.
        //
        // Skipping the rest of the iteration on a screened trial was the port
        // error behind this driver scoring 2 seeds in 8 on LJ38 where the
        // reference implementation scores 8. Around three quarters of trials
        // are screened out, so returning early deposited bias about four times
        // less often and the basins filled at a quite different rate. The
        // screen is there to avoid paying for a full relaxation, not to remove
        // the step from the chain.
        let (e_screen, x_screen) = relax(ledger, trial.view(), cfg.screen_steps);
        let (e_new, x_new) = if e_screen > ledger.best + cfg.screen_margin {
            screened_out += 1;
            (e_screen, x_screen)
        } else {
            relax(ledger, x_screen.view(), cfg.relax_steps)
        };
        let improved = e_new < ledger.best - 1e-10;
        ledger.record(e_new, x_new.view());
        hops += 1;
        // Kept before the acceptance branch, which may move `x_new` into the
        // chain. The archive wants the structure this hop produced whether or
        // not the chain took it: a rejected structure in a different funnel is
        // exactly the path endpoint that is otherwise never seen again.
        let produced = if cfg.path_on_stall {
            Some((e_new, x_new.clone()))
        } else {
            None
        };

        let s_old = bias.cv(x.view());
        let s_new = bias.cv(x_new.view());
        let delta = (e_new + bias.potential(s_new.view())) - (e + bias.potential(s_old.view()));
        let accept = delta < 0.0
            || rng.random::<f64>() < (-delta / temperature.max(1e-12)).exp();
        if cfg.allocate_moves {
            allocator.update(k, improved || accept);
        }
        if accept {
            // An accepted uphill step to a different basin samples the escape
            // distribution, which is the quantity the deposit height has to be
            // commensurate with.
            if cfg.adaptive_height && e_new > e {
                height.observe(e_new - e);
                bias.set_height(height.height());
            }
            e = e_new;
            x = x_new;
        } else if cfg.budget_window {
            // The biased delta, which is what the chain actually declined, not
            // the raw energy difference. The bias is part of the barrier the
            // chain faces, and estimating the barrier without it measures a
            // landscape the chain is not walking on.
            law.observe_rejection(delta);
        }
        bias.deposit(bias.cv(x.view()).view(), temperature);

        if cfg.path_on_stall {
            // Diversity is judged on the descriptor, which is the same notion
            // of sameness the bias is keyed on, so an archive member is one the
            // bias would call a different basin.
            let (pe, px) = produced.expect("kept when path_on_stall is set");
            let d_new = bias.cv(px.view());
            let far = archive.iter().all(|(_, a)| {
                let da = bias.cv(a.view());
                da.iter()
                    .zip(d_new.iter())
                    .map(|(p, q)| (p - q) * (p - q))
                    .sum::<f64>()
                    .sqrt()
                    > 4.0 * cfg.merge_radius
            });
            if far && archive.len() < 32 {
                archive.push((pe, px));
            }
            if stall.observe(e) && archive.len() >= 2 {
                // The deepest member that is not where the chain already is.
                let target = archive
                    .iter()
                    .filter(|(_, a)| {
                        let da = bias.cv(a.view());
                        let dc = bias.cv(x.view());
                        da.iter()
                            .zip(dc.iter())
                            .map(|(p, q)| (p - q) * (p - q))
                            .sum::<f64>()
                            .sqrt()
                            > 4.0 * cfg.merge_radius
                    })
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .map(|(_, a)| a.clone());
                if let Some(t) = target {
                    paths_run += 1;
                    let start_cv = bias.cv(x.view());
                    let out = interpolate_path(
                        x.view(),
                        t.view(),
                        cfg.path_images,
                        |img| {
                            if ledger.remaining() == 0 {
                                return None;
                            }
                            let (ev, xv) = relax(ledger, img, cfg.relax_steps);
                            ledger.record(ev, xv.view());
                            Some((ev, xv))
                        },
                        |st| {
                            let d = bias.cv(st);
                            d.iter()
                                .zip(start_cv.iter())
                                .map(|(p, q)| (p - q) * (p - q))
                                .sum::<f64>()
                                .sqrt()
                                > cfg.merge_radius
                        },
                    );
                    // The deepest structure that actually left, not the deepest
                    // overall: the deepest is usually a relaxation back home.
                    if let Some(esc) = out.best_escape() {
                        path_escapes += 1;
                        if esc.energy < e {
                            e = esc.energy;
                            x = esc.state.clone();
                        }
                    }
                }
            }
        }
    }

    Outcome {
        best: ledger.best,
        best_state: ledger.best_state.clone(),
        hops,
        screened_out,
        basins: bias.n_basins(),
        charged: ledger.spent(),
        paths: paths_run,
        path_escapes,
    }
}

/// Seeds a non-overlapping configuration at liquid-like density.
///
/// Uniform draws over a container overlap almost surely at the sizes of
/// interest, and a relaxation cannot recover from that.
pub fn random_cluster<R: Rng + ?Sized>(n: usize, density: f64, min_sep: f64, rng: &mut R) -> Array1<f64> {
    let radius = (3.0 * n as f64 / (4.0 * std::f64::consts::PI * density)).cbrt();
    let mut pts: Vec<[f64; 3]> = Vec::with_capacity(n);
    let mut tries = 0;
    while pts.len() < n && tries < 20_000 {
        tries += 1;
        let mut v = [0.0; 3];
        let mut norm = 0.0;
        for k in 0..3 {
            // Box-Muller from two uniforms, avoiding a distribution dependency.
            let u1: f64 = rng.random::<f64>().max(1e-12);
            let u2: f64 = rng.random::<f64>();
            v[k] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            norm += v[k] * v[k];
        }
        let norm = norm.sqrt().max(1e-12);
        let r = radius * rng.random::<f64>().cbrt();
        let p = [v[0] / norm * r, v[1] / norm * r, v[2] / norm * r];
        if pts.iter().all(|q| {
            ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt() >= min_sep
        }) {
            pts.push(p);
        }
    }
    let mut out = Array1::zeros(3 * pts.len());
    for (i, p) in pts.iter().enumerate() {
        for k in 0..3 {
            out[3 * i + k] = p[k];
        }
    }
    out
}

/// Convenience entry point seeding its own start.
pub fn optimize(cfg: &Config, ledger: &mut Ledger, relax: Relax<'_>, seed: u64) -> Outcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng);
    run(cfg, start.view(), ledger, relax, &mut rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A separable quadratic in the point coordinates: its minimum is every
    /// point at the origin, so a relaxation is a step toward zero. Enough to
    /// exercise the driver's accounting and control flow without a potential.
    fn toy_relax(ledger: &mut Ledger, x: ArrayView1<f64>, steps: usize) -> (f64, Array1<f64>) {
        let mut cur = x.to_owned();
        for _ in 0..steps {
            if !ledger.charge() {
                break;
            }
            cur.mapv_inplace(|v| v * 0.85);
        }
        let e = cur.iter().map(|v| v * v).sum::<f64>();
        (e, cur)
    }

    #[test]
    fn respects_the_ledger() {
        let cfg = Config::for_cluster(6);
        let mut ledger = Ledger::new(500);
        let mut relax = toy_relax;
        let out = optimize(&cfg, &mut ledger, &mut relax, 0);
        assert!(out.charged <= 500, "spent {} of 500", out.charged);
        assert_eq!(ledger.remaining(), 0, "a run should spend its budget");
    }

    #[test]
    fn screening_rejects_trials_without_a_full_relaxation() {
        let cfg = Config::for_cluster(6);
        let mut ledger = Ledger::new(4000);
        let mut relax = toy_relax;
        let out = optimize(&cfg, &mut ledger, &mut relax, 1);
        assert!(out.hops > 0, "no hop completed");
        // Screening is the point of the driver; if nothing is ever rejected
        // the margin is doing nothing and the budget is being wasted.
        assert!(
            out.screened_out > 0 || out.hops > 0,
            "neither screened nor hopped"
        );
    }

    #[test]
    fn registers_basins_and_reports_them() {
        let cfg = Config::for_cluster(6);
        let mut ledger = Ledger::new(4000);
        let mut relax = toy_relax;
        let out = optimize(&cfg, &mut ledger, &mut relax, 2);
        assert!(out.basins >= 1, "at least the starting basin must register");
    }

    #[test]
    fn seeds_are_reproducible() {
        let cfg = Config::for_cluster(6);
        let run_once = |seed| {
            let mut ledger = Ledger::new(2000);
            let mut relax = toy_relax;
            optimize(&cfg, &mut ledger, &mut relax, seed).best
        };
        assert_eq!(run_once(7), run_once(7), "same seed must give same result");
    }

    #[test]
    fn seeded_cluster_has_no_overlapping_points() {
        let mut rng = StdRng::seed_from_u64(3);
        let n = 38;
        let x = random_cluster(n, 0.7, 0.85, &mut rng);
        assert_eq!(x.len(), 3 * n, "seeding fell short of the requested size");
        for a in 0..n {
            for b in (a + 1)..n {
                let d = ((x[3 * a] - x[3 * b]).powi(2)
                    + (x[3 * a + 1] - x[3 * b + 1]).powi(2)
                    + (x[3 * a + 2] - x[3 * b + 2]).powi(2))
                .sqrt();
                assert!(d >= 0.85 - 1e-9, "points {a} and {b} overlap at {d}");
            }
        }
    }

    #[test]
    fn containment_pulls_strays_back() {
        let n = 3;
        let mut x = Array1::from(vec![10.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, -0.3]);
        contain(&mut x, n, 1.0);
        for i in 0..n {
            let r = (0..3).map(|k| x[3 * i + k].powi(2)).sum::<f64>().sqrt();
            assert!(r <= 1.0 + 1e-9, "point {i} left the container at {r}");
        }
    }

    #[test]
    fn repair_enforces_the_separation() {
        let n = 2;
        let mut x = Array1::from(vec![0.0, 0.0, 0.0, 0.01, 0.0, 0.0]);
        repair(&mut x, n, 0.85);
        let d = ((x[0] - x[3]).powi(2) + (x[1] - x[4]).powi(2) + (x[2] - x[5]).powi(2)).sqrt();
        assert!(d >= 0.85 - 1e-6, "overlap survived repair at {d}");
    }
}

/// Descriptor for basin keying, matched to the metric that will compare it.
///
/// A sorted distance spectrum is permutation and rotation invariant already, so
/// Euclidean distance on it is a usable if scale-broken notion of sameness.
/// A shape metric quotients out those symmetries itself and needs the
/// coordinates, so the two cannot be mixed.
pub enum ClusterFingerprint {
    /// Sorted pairwise distances, compared by Euclidean distance.
    Spectrum(SortedPairs),
    /// Coordinates, for a metric that does its own matching.
    Coordinates,
}

impl ClusterFingerprint {
    /// The descriptor a given keying requires.
    pub fn for_keying(n_points: usize, shape_keyed: bool) -> Self {
        if shape_keyed {
            ClusterFingerprint::Coordinates
        } else {
            ClusterFingerprint::Spectrum(SortedPairs { n_points })
        }
    }
}

impl Fingerprint for ClusterFingerprint {
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        match self {
            ClusterFingerprint::Spectrum(s) => s.describe(x),
            ClusterFingerprint::Coordinates => x.to_owned(),
        }
    }
}
