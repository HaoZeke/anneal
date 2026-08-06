//! A Hamiltonian proposal for the hopping chain, adapted rather than tuned.
//!
//! Basin hopping and the genetic algorithms it competes with propose by
//! displacing every coordinate uniformly and independently. That proposal is
//! isotropic, so it ignores every direction the landscape has, and its
//! amplitude is a constant a human swept until the acceptance ratio looked
//! right. This crate's library uses 0.38 on a Lennard-Jones cluster for exactly
//! that reason. Neither property survives contact with a different potential or
//! a different size.
//!
//! Sampling theory replaced both a decade ago. The direction comes from the
//! gradient through Hamiltonian dynamics, the trajectory length comes from the
//! no-U-turn criterion, and the step size comes from dual averaging against a
//! target acceptance. What is left for a human to set is which mass matrix to
//! use, and this module makes that a measurement rather than a choice.
//!
//! # What this is, and what it is not
//!
//! The chain walks the quenched surface `Ẽ(x) = E(Q(x))`, which is piecewise
//! constant on basins and therefore has no gradient anywhere useful. HMC cannot
//! run on it. So the Hamiltonian dynamics runs on the *underlying* potential
//! `E`, and its endpoint is quenched before the acceptance test, which stays on
//! quenched energies exactly as basin hopping requires. The trajectory replaces
//! the random kick and nothing else.
//!
//! This is a proposal, not a sampler of the quenched measure. Three separate
//! facts, stated plainly because each is a place a reader could be misled:
//!
//! 1. The trajectory is a valid HMC proposal for `exp(-E/T_traj)` on the raw
//!    surface. Reversibility and volume preservation are asserted by test, not
//!    argued.
//! 2. The quench `Q` in front of the acceptance test is many-to-one: a whole
//!    basin of trial points maps to one minimum. So the induced proposal kernel
//!    on minima is not symmetric, and the chain is not a Metropolis chain for
//!    `exp(-Ẽ/T)` however the trial point was generated. That is a property of
//!    basin hopping as a method family, shared with every implementation of it
//!    including the uniform-displacement control, and not something introduced
//!    here.
//! 3. The model-Hessian metric depends on position, so it varies from proposal
//!    to proposal and adds a second asymmetry on top of the quench's. That one
//!    *is* introduced here, so it is measured:
//!    [`HopDiagnostics::metric_drift`] reports how much the kinetic form
//!    changes across a trajectory, and it is subordinate to the quench's
//!    asymmetry rather than comparable to it.
//!
//! The stationary distribution of the resulting chain is the one the acceptance
//! rule defines, and the reason to run it is that it reaches further per
//! charged evaluation, not that it samples a named measure.
//!
//! # What a proposal costs
//!
//! Each leapfrog leaf is one evaluation of the potential and its gradient,
//! charged. On a pairwise potential the two share the pair loop, so a leaf is
//! one charge rather than two. A trajectory of `2^d - 1` leaves therefore costs
//! that many charges against zero for a random kick, and both are followed by a
//! quench costing about twenty-five under the screen. The comparison the
//! campaign makes is at equal *total* charge, so the HMC arm takes fewer hops
//! and each has to be worth more.
//!
//! [`HopConfig::max_depth`] defaults to 5, capping a trajectory at 31 leaves,
//! which puts a hop within a factor of two of the control's cost. That is a
//! budget decision and it silently truncates the no-U-turn criterion whenever
//! it binds, so [`HopDiagnostics::depth_capped`] counts how often, and a rate
//! near one means the arm is running fixed-length HMC with the label of NUTS.
//!
//! # The momentum scale is adapted against reach, not against energy
//!
//! `T_traj` sets the energy the momentum draw injects: with `dim` degrees of
//! freedom the draw carries `dim T_traj / 2` in energy units, so the trajectory
//! runs on an energy shell that far above the minimum it started in. Something
//! has to set it, and the two obvious candidates are both wrong.
//!
//! The chain's Metropolis temperature is wrong by two orders of magnitude and
//! in the direction that destroys the structure. The 0.8 that Wales and Doye
//! use is a threshold on differences between *minima*; 0.8 times 114 halves of
//! a degree of freedom is 45.6 units of kinetic energy on a 38-point cluster,
//! against a pair well depth of 1. That does not perturb a cluster, it
//! evaporates it.
//!
//! Matching the energy the control's kick injects is also wrong, and this one
//! is worth stating because it looks right. Measured on a relaxed 13-point
//! Lennard-Jones structure at the library's own half-width of 0.38, a uniform
//! kick raises the energy by a median of 3.3e3 and a mean of 1.4e5 in units
//! where the pair well is 1. Neither number describes a small perturbation:
//! both are dominated by the kick driving some pair onto the `r^-12` wall,
//! which costs 1.75e3 per pair at 0.6 of the equilibrium separation. Basin
//! hopping tolerates that because the trial configuration is never evaluated
//! for acceptance, only its quench is, so the wall energy is absorbed by the
//! relaxation and never appears in the chain. An HMC trajectory carrying that
//! energy as kinetic energy would have nothing to absorb it.
//!
//! What the two proposals genuinely have in common is how far they move the
//! structure, which is also what the campaign is about: the mechanisms that
//! paid in this crate changed what a hop can reach. So `T_traj` is adapted
//! until the trajectory's rigid-free displacement matches the control kick's,
//! which for a uniform draw of half-width `h` is `h sqrt(dim / 3)` exactly.
//! Displacement scales as `sqrt(T_traj)`, so the update is a damped Newton step
//! in `log T_traj`, run during warmup alongside dual averaging and frozen with
//! it. It costs no charged evaluations at all, carries no constant that is
//! specific to Lennard-Jones or to a cluster size, and leaves the two arms
//! matched on reach and differing only in direction, which is the comparison
//! worth making.

use ndarray::{Array1, ArrayView1};
use rand::Rng;

use crate::hmc::dual_average::{DualAverage, WarmupSchedule};
use crate::hmc::metric::{Metric, MetricAdaptation, MetricKind};
use crate::methods::cluster_hopping::Ledger;

/// Hamiltonian change beyond which a trajectory is called divergent.
///
/// Stan's threshold, and it is not arbitrary here: on a Lennard-Jones surface a
/// single pair at 0.4 of the mean spacing already contributes 6.1e4 to the
/// energy, so a trajectory that has climbed the `r^-12` wall is over any
/// threshold in this range by orders of magnitude and the exact number does not
/// matter. What matters is that such a trajectory is counted and reported
/// rather than quietly retried.
pub const MAX_DELTA_H: f64 = 1000.0;

/// Settings for the Hamiltonian proposal. Shared and immutable across chains.
#[derive(Debug, Clone)]
pub struct HopConfig {
    /// Points in a structure.
    pub n_points: usize,
    /// Which mass matrix to run.
    pub metric: MetricKind,
    /// Warmup hops, during which the step size and metric adapt.
    pub warmup_hops: usize,
    /// Largest doubling depth; a trajectory holds at most `2^d - 1` leaves.
    pub max_depth: u32,
    /// Target acceptance statistic for dual averaging.
    pub target_accept: f64,
    /// Half-width of the control displacement the reach is matched to.
    ///
    /// The comparison is against this crate's own
    /// [`crate::methods::cluster_hopping::ClusterMove::AllPoints`], whose
    /// library value is 0.38.
    pub control_step: f64,
    /// Rigid-free distance a proposal should cover, when the caller wants to
    /// set it rather than take the control's.
    ///
    /// The default is `control_step * sqrt(dim / 3)`, which is the exact root
    /// mean square displacement of a uniform draw of that half-width over `dim`
    /// coordinates.
    pub target_reach: Option<f64>,
    /// Proposals between metric-drift measurements.
    ///
    /// The measurement rebuilds the metric at the endpoint, which is a dense
    /// factorisation, so it runs on a schedule rather than every hop. It
    /// charges nothing.
    pub drift_period: usize,
    /// Trajectory temperature, when the caller wants to set it rather than have
    /// it calibrated.
    pub trajectory_temperature: Option<f64>,
}

impl HopConfig {
    /// Defaults for a cluster of `n_points` under `metric`.
    pub fn new(n_points: usize, metric: MetricKind) -> Self {
        Self {
            n_points,
            metric,
            warmup_hops: 150,
            max_depth: 5,
            target_accept: 0.8,
            control_step: 0.38,
            target_reach: None,
            drift_period: 64,
            trajectory_temperature: None,
        }
    }

    /// Rigid-free distance a proposal should cover.
    ///
    /// A uniform draw of half-width `h` on each of `dim` coordinates has
    /// `E|d|^2 = dim h^2 / 3`, so the root mean square displacement is
    /// `h sqrt(dim / 3)` with no approximation.
    pub fn reach(&self) -> f64 {
        match self.target_reach {
            Some(d) => d,
            None => self.control_step * (self.n_points as f64).sqrt(),
        }
    }
}

/// One proposal and everything measured while making it.
#[derive(Debug, Clone)]
pub struct HopProposal {
    /// The trajectory endpoint, to be quenched by the caller.
    pub x: Array1<f64>,
    /// Mean Metropolis acceptance probability over the trajectory's leaves.
    ///
    /// This is the statistic dual averaging consumes, and it is *not* the
    /// chain's acceptance rate: the chain accepts on quenched energies.
    pub accept_stat: f64,
    /// Doubling depth reached.
    pub depth: u32,
    /// Leapfrog leaves evaluated, which is what the ledger was charged.
    pub leaves: usize,
    /// Whether the doubling stopped because it hit [`HopConfig::max_depth`].
    pub capped: bool,
    /// Whether any leaf exceeded [`MAX_DELTA_H`].
    pub diverged: bool,
    /// Smallest pair separation reached, in units of the mean spacing.
    ///
    /// The structural coordinate of a divergence: values below about 0.7 are on
    /// the repulsive wall, which is where an explicit integrator fails on this
    /// potential.
    pub min_separation: f64,
}

/// Per-arm counters, which are the result rather than an error path.
///
/// Every field here answers "did the mechanism fire", which a solve count
/// cannot. Eleven mechanisms in this campaign executed without acting and each
/// was invisible until an instrument aimed at the mechanism existed.
#[derive(Debug, Clone, Default)]
pub struct HopDiagnostics {
    /// Proposals made.
    pub proposals: u64,
    /// Proposals with a divergent leaf.
    pub divergences: u64,
    /// Proposals stopped by the depth cap rather than by a U-turn.
    pub depth_capped: u64,
    /// Sum of depths, for the mean.
    pub depth_sum: u64,
    /// Count of proposals at each depth, indexed by depth.
    pub depth_histogram: Vec<u64>,
    /// Sum of acceptance statistics, for the mean against the target.
    pub accept_sum: f64,
    /// Leapfrog leaves evaluated in total, which is the charge the arm spent on
    /// trajectories.
    pub leaves: u64,
    /// Sum of the minimum pair separations at divergent proposals.
    pub divergent_separation_sum: f64,
    /// Relative change in the kinetic form across a trajectory, summed.
    pub metric_drift_sum: f64,
    /// Metric-drift measurements taken.
    pub metric_drift_count: u64,
    /// Step size the chain froze at.
    pub epsilon_final: f64,
    /// Trajectory temperature the calibration produced.
    pub trajectory_temperature: f64,
    /// Condition number of the adapted diagonal metric, one when it carries no
    /// anisotropy.
    pub metric_condition: f64,
    /// Sum of rigid-free distances covered, for the mean.
    pub reach_sum: f64,
    /// Distance the reach adaptation is aiming at.
    pub reach_target: f64,
}

impl HopDiagnostics {
    /// Mean doubling depth.
    pub fn mean_depth(&self) -> f64 {
        if self.proposals == 0 {
            0.0
        } else {
            self.depth_sum as f64 / self.proposals as f64
        }
    }

    /// Fraction of proposals that diverged.
    pub fn divergence_rate(&self) -> f64 {
        if self.proposals == 0 {
            0.0
        } else {
            self.divergences as f64 / self.proposals as f64
        }
    }

    /// Fraction of proposals the depth cap truncated.
    pub fn cap_rate(&self) -> f64 {
        if self.proposals == 0 {
            0.0
        } else {
            self.depth_capped as f64 / self.proposals as f64
        }
    }

    /// Mean acceptance statistic, which dual averaging should have driven to
    /// the target.
    pub fn mean_accept(&self) -> f64 {
        if self.proposals == 0 {
            0.0
        } else {
            self.accept_sum / self.proposals as f64
        }
    }

    /// Mean minimum pair separation among divergent proposals, in units of the
    /// mean spacing.
    pub fn divergent_separation(&self) -> f64 {
        if self.divergences == 0 {
            0.0
        } else {
            self.divergent_separation_sum / self.divergences as f64
        }
    }

    /// Mean relative change in the kinetic form across a trajectory.
    pub fn metric_drift(&self) -> f64 {
        if self.metric_drift_count == 0 {
            0.0
        } else {
            self.metric_drift_sum / self.metric_drift_count as f64
        }
    }

    /// Mean rigid-free distance a proposal covered.
    pub fn mean_reach(&self) -> f64 {
        if self.proposals == 0 {
            0.0
        } else {
            self.reach_sum / self.proposals as f64
        }
    }

    /// One line, for a campaign log.
    ///
    /// Every field is an instrument aimed at whether the mechanism fired.
    /// `cap` near one says NUTS is truncated to fixed-length HMC; `cond` near
    /// one says the estimated metric carries no anisotropy; `reach` far from
    /// its target says the momentum adaptation never converged.
    pub fn report(&self, name: &str) -> String {
        format!(
            "hmc {name}  proposals {}  leaves {}  depth {:.2} (cap {:.3})  \
             accept {:.3}  diverge {:.4} at sep {:.3}  eps {:.4e}  \
             T_traj {:.4e}  reach {:.3}/{:.3}  cond {:.3}  drift {:.4}",
            self.proposals,
            self.leaves,
            self.mean_depth(),
            self.cap_rate(),
            self.mean_accept(),
            self.divergence_rate(),
            self.divergent_separation(),
            self.epsilon_final,
            self.trajectory_temperature,
            self.mean_reach(),
            self.reach_target,
            self.metric_condition,
            self.metric_drift(),
        )
    }

    /// The depth distribution, as counts per depth.
    ///
    /// A mean depth and a cap rate do not say whether the trajectory length is
    /// varying at all. This does: a histogram piled on the cap is fixed-length
    /// HMC, and a spread one is the no-U-turn criterion choosing.
    pub fn depth_report(&self) -> String {
        let mut s = String::from("depths");
        for (d, c) in self.depth_histogram.iter().enumerate() {
            s.push_str(&format!(" {d}:{c}"));
        }
        s
    }
}

/// Value and gradient of the underlying potential, charged to the ledger.
///
/// One closure returns both because on a pairwise potential they share the pair
/// loop, and splitting them doubles the charge for arithmetic already done.
/// `None` means the ledger is spent.
pub type Energy<'a> =
    &'a mut dyn FnMut(&mut Ledger, ArrayView1<f64>) -> Option<(f64, Array1<f64>)>;

/// One chain's adaptation state.
///
/// Owned per chain and never shared. A replica ladder needs one of these per
/// rung: a hot rung and a cold rung converge to different step sizes because
/// they traverse differently conditioned regions, and a swap moves
/// configurations between rungs while this stays with the temperature.
pub struct HopChain {
    /// Step-size adaptation.
    pub step: DualAverage,
    /// Metric estimation and the metric choice.
    pub metric: MetricAdaptation,
    /// Where the chain sits in Stan's warmup schedule.
    pub schedule: WarmupSchedule,
    /// Counters.
    pub diag: HopDiagnostics,
    /// Energy the momentum draw injects, per degree of freedom.
    traj_temp: f64,
    /// Whether the caller pinned the trajectory temperature.
    traj_temp_fixed: bool,
    /// Reach observations folded into the momentum scale.
    reach_updates: u64,
    /// Proposals since the last metric-drift measurement.
    since_drift: usize,
}

impl HopChain {
    /// A fresh chain under `cfg`.
    pub fn new(cfg: &HopConfig) -> Self {
        // A shell one energy unit above the minimum, which for a pairwise
        // potential is a fraction of one well depth spread over every degree of
        // freedom. Cold enough that a first trajectory cannot diverge before
        // the adaptation has seen anything, and the adaptation is multiplicative
        // so it climbs from here geometrically.
        let dim = (3 * cfg.n_points).max(1) as f64;
        Self {
            // The initial step is replaced by the reasonable-epsilon search on
            // the first proposal, so this value only has to be positive.
            step: DualAverage::new(0.05),
            metric: MetricAdaptation::new(cfg.metric, cfg.n_points),
            schedule: WarmupSchedule::new(cfg.warmup_hops),
            diag: HopDiagnostics {
                metric_condition: 1.0,
                reach_target: cfg.reach(),
                ..Default::default()
            },
            traj_temp: cfg.trajectory_temperature.unwrap_or(2.0 / dim),
            traj_temp_fixed: cfg.trajectory_temperature.is_some(),
            reach_updates: 0,
            since_drift: 0,
        }
    }

    /// The step size the chain is integrating with.
    pub fn epsilon(&self) -> f64 {
        self.step.epsilon()
    }

    /// The energy the momentum draw injects, per degree of freedom.
    pub fn trajectory_temperature(&self) -> f64 {
        self.traj_temp
    }

    /// Folds one observed reach into the momentum scale.
    ///
    /// Displacement scales as the square root of the trajectory temperature at
    /// fixed step and leaf count, so `d log D / d log T = 1/2` and the Newton
    /// step is `log T += 2 log(D_target / D)`. The gain decays as `1 / (m + 8)`
    /// so the recursion converges, and each update is capped at a factor of two
    /// so one anomalous trajectory cannot move the scale by orders of
    /// magnitude. Charges nothing.
    ///
    /// Two timescales: the step size adapts on every hop and the momentum scale
    /// on a decaying gain beneath it, so dual averaging tracks the shell rather
    /// than fighting it.
    fn learn_reach(&mut self, cfg: &HopConfig, reach: f64) {
        if self.traj_temp_fixed || !self.schedule.warming() {
            return;
        }
        if !(reach > 0.0) || !reach.is_finite() {
            return;
        }
        let target = cfg.reach();
        self.reach_updates += 1;
        let gain = 1.0 / (self.reach_updates as f64 + 8.0);
        let step = (2.0 * (target / reach).ln() * gain).clamp(-(2.0f64).ln(), (2.0f64).ln());
        self.traj_temp = (self.traj_temp.ln() + step).exp().clamp(1e-12, 1e6);
        self.diag.trajectory_temperature = self.traj_temp;
    }

    /// Draws one Hamiltonian proposal from `x`.
    ///
    /// `energy` is the potential at `x`, already paid for by the caller's
    /// quench. Returns `None` when the ledger runs out mid-trajectory, which
    /// leaves the chain where it was.
    pub fn propose<R: Rng + ?Sized>(
        &mut self,
        cfg: &HopConfig,
        ledger: &mut Ledger,
        x: ArrayView1<f64>,
        energy: f64,
        eval: &mut Energy<'_>,
        rng: &mut R,
    ) -> Option<HopProposal> {
        let n = cfg.n_points;
        if x.len() != 3 * n || !energy.is_finite() {
            return None;
        }
        let temp = self.traj_temp;

        // The metric is built once here and held for the whole trajectory, so
        // the Hamiltonian stays separable and leapfrog stays symplectic.
        self.metric.observe(x);
        let metric = self.metric.freeze(x);

        // The initial step comes from Hoffman and Gelman's reasonable-epsilon
        // search rather than from a constant, and only on the first proposal.
        if self.step.count() == 0 && !self.step.is_frozen() {
            let eps0 = reasonable_epsilon(ledger, x, energy, temp, &metric, eval, rng)?;
            self.step = DualAverage::new(eps0);
            self.step.delta = cfg.target_accept;
        }

        let eps = self.step.epsilon();
        let out = nuts(
            ledger, x, energy, temp, eps, &metric, cfg.max_depth, eval, rng,
        )?;

        // How far the proposal actually moved the structure, with rigid motion
        // removed because the basin descriptor is invariant to it and a
        // trajectory that only translated the cluster has reached nothing.
        let mut d = &out.x - &x.to_owned();
        for z in &metric.rigid().z {
            let c = d.dot(z);
            d.scaled_add(-c, z);
        }
        let reach = d.dot(&d).sqrt();
        self.diag.reach_sum += reach;
        // A divergent trajectory's endpoint is on the repulsive wall and its
        // displacement says nothing about what the momentum scale buys, so it
        // is not fed to the reach adaptation.
        if !out.diverged {
            self.learn_reach(cfg, reach);
        }

        // Adaptation, then the schedule. Stan restarts the step-size recursion
        // whenever the metric changes, because the step size that hits the
        // target under one metric is not the one that hits it under another.
        if self.schedule.warming() {
            self.step.learn(out.accept_stat);
            let closed = self.schedule.advance();
            if closed {
                self.metric.close_window();
                self.diag.metric_condition = self.metric.condition();
                self.step.restart();
            }
            if !self.schedule.warming() {
                self.step.freeze();
                self.metric.close_window();
                self.diag.metric_condition = self.metric.condition();
            }
        }
        self.diag.epsilon_final = self.step.epsilon();

        self.diag.proposals += 1;
        self.diag.leaves += out.leaves as u64;
        self.diag.depth_sum += out.depth as u64;
        if self.diag.depth_histogram.len() <= out.depth as usize {
            self.diag.depth_histogram.resize(out.depth as usize + 1, 0);
        }
        self.diag.depth_histogram[out.depth as usize] += 1;
        self.diag.accept_sum += out.accept_stat;
        if out.capped {
            self.diag.depth_capped += 1;
        }
        if out.diverged {
            self.diag.divergences += 1;
            self.diag.divergent_separation_sum += out.min_separation;
        }

        // How much the metric moved over the trajectory. Charges nothing, and
        // it is the instrument for the one asymmetry this module introduces
        // beyond the quench's.
        self.since_drift += 1;
        if self.since_drift >= cfg.drift_period.max(1) {
            self.since_drift = 0;
            let end = self.metric.freeze(out.x.view());
            let p = metric.sample(rng);
            let a = metric.kinetic(p.view());
            let b = end.kinetic(p.view());
            if a > 0.0 && b.is_finite() {
                self.diag.metric_drift_sum += ((b - a) / a).abs();
                self.diag.metric_drift_count += 1;
            }
        }

        Some(out)
    }
}

/// One leapfrog step under a frozen metric, on `U(x) = E(x) / temp`.
///
/// Kick, drift, kick. Each step is self-contained rather than sharing half
/// kicks with its neighbour, which costs one extra gradient per trajectory and
/// buys exact reversibility of a single step, the property the tests assert.
fn leapfrog(
    ledger: &mut Ledger,
    x: ArrayView1<f64>,
    g: ArrayView1<f64>,
    p: ArrayView1<f64>,
    eps: f64,
    temp: f64,
    metric: &Metric,
    eval: &mut Energy<'_>,
) -> Option<(Array1<f64>, Array1<f64>, Array1<f64>, f64)> {
    let mut p_new = p.to_owned();
    p_new.scaled_add(-0.5 * eps / temp, &g);
    let v = metric.velocity(p_new.view());
    let mut x_new = x.to_owned();
    x_new.scaled_add(eps, &v);
    let (e_new, g_new) = eval(ledger, x_new.view())?;
    p_new.scaled_add(-0.5 * eps / temp, &g_new);
    Some((x_new, p_new, g_new, e_new))
}

/// Hoffman and Gelman 2014, algorithm 4: double or halve the step until one
/// leapfrog step crosses an acceptance of one half.
///
/// The point is that no constant is carried in. The search charges the ledger
/// for the steps it takes, which is reported as part of the arm's cost.
fn reasonable_epsilon<R: Rng + ?Sized>(
    ledger: &mut Ledger,
    x: ArrayView1<f64>,
    energy: f64,
    temp: f64,
    metric: &Metric,
    eval: &mut Energy<'_>,
    rng: &mut R,
) -> Option<f64> {
    let (_, g0) = eval(ledger, x)?;
    let p0 = metric.sample(rng);
    let h0 = energy / temp + metric.kinetic(p0.view());
    let mut eps = 1.0f64;
    let mut direction = 0i32;
    for _ in 0..40 {
        let (_, p1, _, e1) =
            leapfrog(ledger, x, g0.view(), p0.view(), eps, temp, metric, eval)?;
        let h1 = e1 / temp + metric.kinetic(p1.view());
        let log_ratio = h0 - h1;
        let d = if log_ratio > -(2.0f64).ln() { 1 } else { -1 };
        if direction == 0 {
            direction = d;
        } else if d != direction {
            break;
        }
        if !log_ratio.is_finite() {
            eps *= 0.5;
            direction = -1;
            continue;
        }
        eps *= if direction > 0 { 2.0 } else { 0.5 };
        if !(1e-12..=1e6).contains(&eps) {
            break;
        }
    }
    Some(eps.clamp(1e-12, 1e6))
}

/// A leaf of the trajectory.
#[derive(Clone)]
struct Leaf {
    x: Array1<f64>,
    p: Array1<f64>,
    g: Array1<f64>,
    h: f64,
}

/// A doubling's worth of trajectory.
struct SubTree {
    left: Leaf,
    right: Leaf,
    candidate: Array1<f64>,
    /// `log sum exp(-H)` over the leaves.
    log_w: f64,
    /// Sum of the momenta, the vector the U-turn criterion is taken against.
    rho: Array1<f64>,
    leaves: usize,
    /// Sum over leaves of `min(1, exp(H0 - H))`, for the acceptance statistic.
    accept_sum: f64,
    turned: bool,
    diverged: bool,
    min_sep: f64,
}

fn log_sum_exp(a: f64, b: f64) -> f64 {
    if !a.is_finite() {
        return b;
    }
    if !b.is_finite() {
        return a;
    }
    let m = a.max(b);
    m + (-(a - b).abs()).exp().ln_1p()
}

/// Stan's generalised no-U-turn criterion.
///
/// Hoffman and Gelman's original test is `(x_plus - x_minus) . p < 0` at either
/// end. Under a non-identity metric the vector conjugate to a displacement is
/// the velocity `M^-1 p`, not the momentum, so the products are taken against
/// velocities; and the displacement is replaced by `rho`, the sum of the
/// momenta over the trajectory, which is the same quantity to leading order and
/// is what Stan accumulates. Betancourt, "A Conceptual Introduction to
/// Hamiltonian Monte Carlo", arXiv:1701.02434, appendix A.4.
///
/// The extra sub-trajectory checks Stan added later are not implemented. Their
/// absence costs gradients rather than correctness: a trajectory that makes a
/// U-turn strictly inside a subtree runs to the end of that doubling instead of
/// stopping at it.
fn turned(metric: &Metric, left: &Leaf, right: &Leaf, rho: &Array1<f64>) -> bool {
    let vl = metric.velocity(left.p.view());
    let vr = metric.velocity(right.p.view());
    vl.dot(rho) <= 0.0 || vr.dot(rho) <= 0.0
}

/// Smallest pair separation in units of the mean nearest-neighbour spacing.
fn min_separation(x: ArrayView1<f64>, n: usize) -> f64 {
    if n < 2 {
        return f64::INFINITY;
    }
    let mut lo = f64::INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut r2 = 0.0;
            for k in 0..3 {
                let d = x[3 * i + k] - x[3 * j + k];
                r2 += d * d;
            }
            lo = lo.min(r2);
        }
    }
    let scale = crate::model_hessian::spacing(x, n);
    if scale > 0.0 { lo.sqrt() / scale } else { lo.sqrt() }
}

#[allow(clippy::too_many_arguments)]
fn build_tree<R: Rng + ?Sized>(
    ledger: &mut Ledger,
    from: &Leaf,
    h0: f64,
    depth: u32,
    sign: f64,
    eps: f64,
    temp: f64,
    metric: &Metric,
    n: usize,
    eval: &mut Energy<'_>,
    rng: &mut R,
) -> Option<SubTree> {
    if depth == 0 {
        let (x1, p1, g1, e1) = leapfrog(
            ledger,
            from.x.view(),
            from.g.view(),
            from.p.view(),
            sign * eps,
            temp,
            metric,
            eval,
        )?;
        let h = e1 / temp + metric.kinetic(p1.view());
        let dh = h0 - h;
        let diverged = !h.is_finite() || (h - h0).abs() > MAX_DELTA_H;
        let leaf = Leaf {
            x: x1,
            p: p1,
            g: g1,
            h,
        };
        let rho = leaf.p.clone();
        let sep = if diverged {
            min_separation(leaf.x.view(), n)
        } else {
            f64::INFINITY
        };
        return Some(SubTree {
            candidate: leaf.x.clone(),
            log_w: if h.is_finite() { -h } else { f64::NEG_INFINITY },
            rho,
            leaves: 1,
            accept_sum: if dh.is_finite() {
                dh.exp().min(1.0)
            } else {
                0.0
            },
            turned: diverged,
            diverged,
            min_sep: sep,
            left: leaf.clone(),
            right: leaf,
        });
    }
    let a = build_tree(
        ledger,
        from,
        h0,
        depth - 1,
        sign,
        eps,
        temp,
        metric,
        n,
        eval,
        rng,
    )?;
    if a.turned || a.diverged {
        return Some(a);
    }
    let start = if sign < 0.0 { &a.left } else { &a.right };
    let b = build_tree(
        ledger,
        start,
        h0,
        depth - 1,
        sign,
        eps,
        temp,
        metric,
        n,
        eval,
        rng,
    )?;
    let (left, right) = if sign < 0.0 {
        (b.left.clone(), a.right.clone())
    } else {
        (a.left.clone(), b.right.clone())
    };
    let log_w = log_sum_exp(a.log_w, b.log_w);
    // Progressive multinomial within the doubling: the new half is taken with
    // probability equal to its share of the weight.
    let pick = (b.log_w - log_w).exp();
    let candidate = if rng.random::<f64>() < pick {
        b.candidate.clone()
    } else {
        a.candidate.clone()
    };
    let mut rho = a.rho.clone();
    rho += &b.rho;
    let turned = a.turned || b.turned || turned(metric, &left, &right, &rho);
    Some(SubTree {
        left,
        right,
        candidate,
        log_w,
        rho,
        leaves: a.leaves + b.leaves,
        accept_sum: a.accept_sum + b.accept_sum,
        turned,
        diverged: a.diverged || b.diverged,
        min_sep: a.min_sep.min(b.min_sep),
    })
}

/// One no-U-turn trajectory.
#[allow(clippy::too_many_arguments)]
fn nuts<R: Rng + ?Sized>(
    ledger: &mut Ledger,
    x: ArrayView1<f64>,
    energy: f64,
    temp: f64,
    eps: f64,
    metric: &Metric,
    max_depth: u32,
    eval: &mut Energy<'_>,
    rng: &mut R,
) -> Option<HopProposal> {
    let n = x.len() / 3;
    let (_, g0) = eval(ledger, x)?;
    let p0 = metric.sample(rng);
    let h0 = energy / temp + metric.kinetic(p0.view());
    let root = Leaf {
        x: x.to_owned(),
        p: p0,
        g: g0,
        h: h0,
    };
    let mut left = root.clone();
    let mut right = root.clone();
    let mut rho = root.p.clone();
    let mut candidate = root.x.clone();
    let mut log_w = -h0;
    let mut leaves = 0usize;
    let mut accept_sum = 0.0;
    let mut depth = 0u32;
    let mut diverged = false;
    let mut min_sep = f64::INFINITY;
    let mut capped = false;

    loop {
        if depth >= max_depth {
            capped = true;
            break;
        }
        let sign = if rng.random::<f64>() < 0.5 { -1.0 } else { 1.0 };
        let from = if sign < 0.0 { &left } else { &right };
        let sub = build_tree(
            ledger, from, h0, depth, sign, eps, temp, metric, n, eval, rng,
        )?;
        leaves += sub.leaves;
        accept_sum += sub.accept_sum;
        min_sep = min_sep.min(sub.min_sep);
        if sub.diverged {
            diverged = true;
            depth += 1;
            break;
        }
        if !sub.turned {
            // Biased progressive sampling across doublings: the new half is
            // taken with probability min(1, w_new / w_old), which pushes the
            // draw away from the starting point and is what Stan does.
            let pick = (sub.log_w - log_w).exp().min(1.0);
            if rng.random::<f64>() < pick {
                candidate = sub.candidate.clone();
            }
        }
        log_w = log_sum_exp(log_w, sub.log_w);
        if sign < 0.0 {
            left = sub.left;
        } else {
            right = sub.right;
        }
        rho += &sub.rho;
        depth += 1;
        if sub.turned || turned(metric, &left, &right, &rho) {
            break;
        }
    }

    let accept_stat = if leaves == 0 {
        0.0
    } else {
        accept_sum / leaves as f64
    };
    Some(HopProposal {
        x: candidate,
        accept_stat,
        depth,
        leaves,
        capped,
        diverged,
        min_separation: if min_sep.is_finite() { min_sep } else { 0.0 },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmc::metric::RigidModes;
    use ndarray::Array1;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// An anisotropic quadratic: `U = 1/2 sum k_i x_i^2` with the stiffnesses
    /// spread over three decades. Everything the integrator has to satisfy is
    /// checkable in closed form here.
    struct Quad {
        k: Array1<f64>,
    }

    impl Quad {
        fn spread(dim: usize, ratio: f64) -> Self {
            let k = Array1::from_iter((0..dim).map(|i| {
                ratio.powf(i as f64 / (dim.max(2) - 1) as f64)
            }));
            Self { k }
        }
        fn value(&self, x: ArrayView1<f64>) -> f64 {
            0.5 * (0..x.len()).map(|i| self.k[i] * x[i] * x[i]).sum::<f64>()
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            Array1::from_iter((0..x.len()).map(|i| self.k[i] * x[i]))
        }
    }

    fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = x.len() / 3;
        let mut e = 0.0;
        let mut g = Array1::zeros(x.len());
        for i in 0..n {
            for j in (i + 1)..n {
                let d = [
                    x[3 * i] - x[3 * j],
                    x[3 * i + 1] - x[3 * j + 1],
                    x[3 * i + 2] - x[3 * j + 2],
                ];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let inv2 = 1.0 / r2;
                let inv6 = inv2 * inv2 * inv2;
                let inv12 = inv6 * inv6;
                e += 4.0 * (inv12 - inv6);
                let coef = 24.0 * inv2 * (2.0 * inv12 - inv6);
                for k in 0..3 {
                    g[3 * i + k] -= coef * d[k];
                    g[3 * j + k] += coef * d[k];
                }
            }
        }
        (e, g)
    }

    fn blob(n: usize) -> Array1<f64> {
        let mut x = Array1::zeros(3 * n);
        for i in 0..n {
            x[3 * i] = (i % 3) as f64 * 1.12;
            x[3 * i + 1] = ((i / 3) % 3) as f64 * 1.12;
            x[3 * i + 2] = (i / 9) as f64 * 1.12;
        }
        x
    }

    /// Runs `steps` leapfrog steps, ignoring the ledger.
    fn run_steps(
        q: &Quad,
        x0: &Array1<f64>,
        p0: &Array1<f64>,
        eps: f64,
        steps: usize,
        metric: &Metric,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut led = Ledger::new(usize::MAX);
        let mut x = x0.clone();
        let mut p = p0.clone();
        let mut g = q.grad(x.view());
        let mut ev = |l: &mut Ledger, v: ArrayView1<f64>| {
            l.charge();
            Some((q.value(v), q.grad(v)))
        };
        let mut e: Energy<'_> = &mut ev;
        for _ in 0..steps {
            let (x1, p1, g1, _) =
                leapfrog(&mut led, x.view(), g.view(), p.view(), eps, 1.0, metric, &mut e)
                    .unwrap();
            x = x1;
            p = p1;
            g = g1;
        }
        (x, p)
    }

    /// The property that makes the proposal valid, asserted rather than argued:
    /// integrating forward, negating the momentum and integrating the same
    /// number of steps returns the starting point.
    ///
    /// Checked under every metric, because a metric that is not symmetric
    /// positive definite breaks reversibility and nothing else in the pipeline
    /// would notice.
    #[test]
    fn the_integrator_is_time_reversible() {
        let n = 6;
        let dim = 3 * n;
        let x0 = blob(n);
        let q = Quad::spread(dim, 100.0);
        for kind in [
            MetricKind::Identity,
            MetricKind::Diagonal,
            MetricKind::ModelHessian,
        ] {
            let ad = MetricAdaptation::new(kind, n);
            let metric = ad.freeze(x0.view());
            let mut rng = StdRng::seed_from_u64(4);
            let p0 = metric.sample(&mut rng);
            let steps = 24;
            let eps = 0.01;
            let (x1, p1) = run_steps(&q, &x0, &p0, eps, steps, &metric);
            let (x2, p2) = run_steps(&q, &x1, &(-&p1), eps, steps, &metric);
            let dx = (0..dim)
                .map(|i| (x2[i] - x0[i]).abs())
                .fold(0.0f64, f64::max);
            let dp = (0..dim)
                .map(|i| (p2[i] + p0[i]).abs())
                .fold(0.0f64, f64::max);
            let scale = p0.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1.0);
            assert!(
                dx < 1e-10,
                "{}: the return missed the start by {dx}",
                kind.name()
            );
            assert!(
                dp < 1e-10 * scale,
                "{}: the returned momentum missed by {dp}",
                kind.name()
            );
        }
    }

    /// The Hamiltonian error of a Strang splitting is second order in the step,
    /// so halving the step at fixed trajectory length has to cut the drift by
    /// about four. A first-order integrator would give two, and an
    /// implementation with a misplaced half kick gives one.
    #[test]
    fn the_energy_error_falls_as_the_square_of_the_step() {
        let n = 6;
        let dim = 3 * n;
        let x0 = blob(n);
        let q = Quad::spread(dim, 25.0);
        let ad = MetricAdaptation::new(MetricKind::Identity, n);
        let metric = ad.freeze(x0.view());
        let mut rng = StdRng::seed_from_u64(17);
        let p0 = metric.sample(&mut rng);
        let h0 = q.value(x0.view()) + metric.kinetic(p0.view());
        let drift = |eps: f64, steps: usize| -> f64 {
            let (x, p) = run_steps(&q, &x0, &p0, eps, steps, &metric);
            (q.value(x.view()) + metric.kinetic(p.view()) - h0).abs()
        };
        // Fixed trajectory length: the step halves and the count doubles.
        let coarse = drift(0.02, 40);
        let fine = drift(0.01, 80);
        let ratio = coarse / fine;
        assert!(
            (3.0..5.0).contains(&ratio),
            "halving the step cut the drift by {ratio}, wanted about four \
             (coarse {coarse:.3e}, fine {fine:.3e})"
        );
    }

    /// Steepest descent with a backtracking step, so a test can start from a
    /// relaxed structure without a minimiser or a ledger.
    fn descend(mut x: Array1<f64>, iters: usize) -> Array1<f64> {
        let mut step = 1e-3;
        let (mut e, mut g) = lj(x.view());
        for _ in 0..iters {
            let mut trial = x.clone();
            trial.scaled_add(-step, &g);
            let (e1, g1) = lj(trial.view());
            if e1 < e {
                x = trial;
                e = e1;
                g = g1;
                step *= 1.2;
            } else {
                step *= 0.5;
                if step < 1e-12 {
                    break;
                }
            }
        }
        x
    }

    /// Mean rigid-free distance and mean Hamiltonian drift over `draws`
    /// trajectories of `steps` leaves at step `eps`.
    fn distance_and_drift(
        x0: &Array1<f64>,
        metric: &Metric,
        rigid: &RigidModes,
        eps: f64,
        steps: usize,
        draws: usize,
        seed: u64,
    ) -> (f64, f64) {
        let mut led = Ledger::new(usize::MAX);
        let mut ev = |l: &mut Ledger, v: ArrayView1<f64>| {
            l.charge();
            Some(lj(v))
        };
        let mut e: Energy<'_> = &mut ev;
        let mut rng = StdRng::seed_from_u64(seed);
        let (e0, g0) = lj(x0.view());
        let mut dist_sum = 0.0;
        let mut drift_sum = 0.0;
        for _ in 0..draws {
            let p0 = metric.sample(&mut rng);
            let h0 = e0 + metric.kinetic(p0.view());
            let mut x = x0.clone();
            let mut p = p0;
            let mut g = g0.clone();
            let mut energy = e0;
            let mut broke = false;
            for _ in 0..steps {
                match leapfrog(
                    &mut led,
                    x.view(),
                    g.view(),
                    p.view(),
                    eps,
                    1.0,
                    metric,
                    &mut e,
                ) {
                    Some((x1, p1, g1, e1)) => {
                        x = x1;
                        p = p1;
                        g = g1;
                        energy = e1;
                    }
                    None => {
                        broke = true;
                        break;
                    }
                }
            }
            if broke || !energy.is_finite() {
                drift_sum += 1e9;
                continue;
            }
            drift_sum += (energy + metric.kinetic(p.view()) - h0).abs();
            let mut d = &x - x0;
            // Rigid motion is not distance covered: the basin descriptor is
            // invariant to it, so it must not be counted as reach.
            for z in &rigid.z {
                let c = d.dot(z);
                d.scaled_add(-c, z);
            }
            dist_sum += d.dot(&d).sqrt();
        }
        (dist_sum / draws as f64, drift_sum / draws as f64)
    }

    /// The step that puts the mean Hamiltonian drift at `target`, found by
    /// geometric scan. Each metric gets its own, which is the whole reason dual
    /// averaging exists: the step that hits a given acceptance under one metric
    /// is not the one that hits it under another.
    fn eps_for_drift(
        x0: &Array1<f64>,
        metric: &Metric,
        rigid: &RigidModes,
        steps: usize,
        target: f64,
    ) -> f64 {
        let mut lo = 1e-6f64;
        let mut hi = 1.0f64;
        for _ in 0..40 {
            let mid = (lo * hi).sqrt();
            let (_, drift) = distance_and_drift(x0, metric, rigid, mid, steps, 12, 101);
            if drift > target {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        (lo * hi).sqrt()
    }

    /// The claim the model Hessian is in the campaign to test: at equal
    /// Hamiltonian drift and equal gradient count, its metric moves the
    /// structure further than the unit metric does.
    ///
    /// The comparison has to be made at matched drift, not at matched step. A
    /// metric that happens to produce smaller velocities covers less distance
    /// and less drift at the same `eps`, and the ratio of the two is then
    /// dominated by the effective step size rather than by the conditioning:
    /// distance grows linearly in `eps` and drift quadratically, so
    /// distance-per-drift is proportional to `1/eps` and rewards whichever arm
    /// is taking the smaller step. Fixing the drift and comparing distance is
    /// the scale-free question, and it is also the question dual averaging
    /// answers in the running sampler, since it gives each metric the step size
    /// that hits one common acceptance target.
    ///
    /// The anisotropic problem is Lennard-Jones itself, at a relaxed structure.
    /// That is the anisotropy the method has to exploit: contacts are stiff,
    /// transverse motion is soft, and distant pairs are soft, which is what the
    /// stretch operator with an exponential falloff encodes.
    #[test]
    fn the_model_hessian_metric_buys_distance_per_unit_drift() {
        let n = 13;
        let x0 = descend(blob(n), 4000);
        let rigid = RigidModes::at(x0.view(), n);
        let steps = 24;
        let target = 0.05;
        let mut reach = Vec::new();
        for kind in [MetricKind::Identity, MetricKind::ModelHessian] {
            let metric = MetricAdaptation::new(kind, n).freeze(x0.view());
            let eps = eps_for_drift(&x0, &metric, &rigid, steps, target);
            let (dist, drift) =
                distance_and_drift(&x0, &metric, &rigid, eps, steps, 64, 777);
            reach.push((kind, eps, dist, drift));
        }
        for (kind, eps, dist, drift) in &reach {
            assert!(
                (drift / target).abs() < 4.0,
                "{}: the scan left the drift at {drift} against a target of \
                 {target} (eps {eps}, distance {dist})",
                kind.name()
            );
        }
        assert!(
            reach[1].2 > reach[0].2,
            "at a matched drift of {target} over {steps} gradients, the \
             model-Hessian metric reached {:.4} at eps {:.3e} against {:.4} at \
             eps {:.3e} for the unit metric, so the preconditioning is not \
             paying",
            reach[1].2,
            reach[1].1,
            reach[0].2,
            reach[0].1,
        );
    }


    /// A trajectory driven into the repulsive wall has to be reported as
    /// divergent rather than returned as an ordinary proposal, and the
    /// structural coordinate has to come back with it.
    #[test]
    fn a_trajectory_into_the_wall_is_reported_divergent() {
        let n = 8;
        let x0 = blob(n);
        let mut led = Ledger::new(100_000);
        let ad = MetricAdaptation::new(MetricKind::Identity, n);
        let metric = ad.freeze(x0.view());
        let mut ev = |l: &mut Ledger, v: ArrayView1<f64>| {
            if !l.charge() {
                return None;
            }
            Some(lj(v))
        };
        let mut e: Energy<'_> = &mut ev;
        let (e0, _) = lj(x0.view());
        let mut rng = StdRng::seed_from_u64(31);
        // A step far past any stability limit for this potential.
        let out = nuts(
            &mut led, x0.view(), e0, 0.05, 5.0, &metric, 4, &mut e, &mut rng,
        )
        .unwrap();
        assert!(out.diverged, "a step of 5 did not diverge on LJ");
        assert!(
            out.min_separation < 1.0,
            "a divergence reported a minimum separation of {} spacings, \
             which is not on the wall",
            out.min_separation
        );
    }

    /// Dual averaging has to drive the acceptance statistic to the target on a
    /// real potential, without the step size being set by hand anywhere.
    #[test]
    fn adaptation_reaches_the_target_acceptance_on_lj() {
        let n = 8;
        let x0 = blob(n);
        let mut cfg = HopConfig::new(n, MetricKind::Identity);
        cfg.warmup_hops = 120;
        cfg.max_depth = 3;
        let mut chain = HopChain::new(&cfg);
        let mut led = Ledger::new(400_000);
        let mut rng = StdRng::seed_from_u64(2);
        let mut x = x0.clone();
        let mut ev = |l: &mut Ledger, v: ArrayView1<f64>| {
            if !l.charge() {
                return None;
            }
            Some(lj(v))
        };
        let mut e: Energy<'_> = &mut ev;
        let mut late = Vec::new();
        for hop in 0..300 {
            let (energy, _) = lj(x.view());
            let Some(out) = chain.propose(&cfg, &mut led, x.view(), energy, &mut e, &mut rng)
            else {
                break;
            };
            if !out.diverged {
                x = out.x.clone();
            }
            if hop >= 150 {
                late.push(out.accept_stat);
            }
        }
        assert!(chain.step.is_frozen(), "warmup never ended");
        assert!(!late.is_empty(), "no post-warmup proposals were made");
        let mean = late.iter().sum::<f64>() / late.len() as f64;
        assert!(
            (mean - cfg.target_accept).abs() < 0.2,
            "post-warmup acceptance settled at {mean} against a target of {}",
            cfg.target_accept
        );
        assert!(
            chain.epsilon() > 0.0 && chain.epsilon().is_finite(),
            "the adapted step is {}",
            chain.epsilon()
        );
    }

    /// The reason the momentum scale is not matched on energy, kept as a test
    /// because it is the measurement that decided the design.
    ///
    /// A uniform kick at the library's own half-width does not perturb a
    /// relaxed Lennard-Jones structure gently. It drives pairs onto the `r^-12`
    /// wall, so its energy rise is enormous and heavy tailed, and basin hopping
    /// only survives it because the quench absorbs the wall energy before
    /// anything looks at it. A trajectory carrying that as kinetic energy has
    /// nothing to absorb it.
    #[test]
    fn the_controls_energy_rise_is_dominated_by_the_repulsive_wall() {
        let n = 13;
        let x0 = descend(blob(n), 4000);
        let (e0, _) = lj(x0.view());
        let mut rng = StdRng::seed_from_u64(8);
        let mut rises: Vec<f64> = Vec::new();
        for _ in 0..4000 {
            let mut y = x0.clone();
            for v in y.iter_mut() {
                *v += rng.random_range(-0.38..0.38);
            }
            rises.push(lj(y.view()).0 - e0);
        }
        rises.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = rises[rises.len() / 2];
        let mean = rises.iter().sum::<f64>() / rises.len() as f64;
        assert!(
            median > 100.0,
            "the median rise came to {median} well depths, so the kick is a \
             small perturbation after all and matching energy would have worked"
        );
        assert!(
            mean > 3.0 * median,
            "the rise is no longer heavy tailed: mean {mean}, median {median}"
        );
    }

    /// The momentum scale has to converge to one that covers the control's
    /// distance, since that is what the two arms are matched on.
    #[test]
    fn the_momentum_scale_adapts_to_the_controls_reach() {
        let n = 13;
        let x0 = descend(blob(n), 4000);
        let mut cfg = HopConfig::new(n, MetricKind::Identity);
        cfg.warmup_hops = 400;
        cfg.max_depth = 4;
        let mut chain = HopChain::new(&cfg);
        let mut led = Ledger::new(2_000_000);
        let mut rng = StdRng::seed_from_u64(19);
        let mut ev = |l: &mut Ledger, v: ArrayView1<f64>| {
            if !l.charge() {
                return None;
            }
            Some(lj(v))
        };
        let mut e: Energy<'_> = &mut ev;
        // Held at one structure, so what is measured is the adaptation and not
        // the chain wandering to a different part of the landscape.
        let mut late = Vec::new();
        for hop in 0..600 {
            let (energy, _) = lj(x0.view());
            let Some(out) = chain.propose(&cfg, &mut led, x0.view(), energy, &mut e, &mut rng)
            else {
                break;
            };
            if hop >= 400 && !out.diverged {
                let mut d = &out.x - &x0;
                let rigid = crate::hmc::metric::RigidModes::at(x0.view(), n);
                for z in &rigid.z {
                    let c = d.dot(z);
                    d.scaled_add(-c, z);
                }
                late.push(d.dot(&d).sqrt());
            }
        }
        assert!(!late.is_empty(), "no post-warmup proposals were made");
        let mean = late.iter().sum::<f64>() / late.len() as f64;
        let target = cfg.reach();
        assert!(
            (mean / target).abs() > 0.4 && (mean / target) < 2.5,
            "the adapted reach settled at {mean:.4} against a control reach of \
             {target:.4} (T_traj {:.3e})",
            chain.trajectory_temperature()
        );
    }

    /// The depth cap has to be reported when it binds, since a capped
    /// trajectory is a silently truncated proposal and an arm that caps every
    /// hop is running fixed-length HMC under the name of NUTS.
    #[test]
    fn the_depth_cap_is_counted_when_it_binds() {
        let n = 8;
        let x0 = blob(n);
        let mut cfg = HopConfig::new(n, MetricKind::Identity);
        cfg.max_depth = 1;
        cfg.warmup_hops = 20;
        let mut chain = HopChain::new(&cfg);
        let mut led = Ledger::new(60_000);
        let mut rng = StdRng::seed_from_u64(12);
        let mut ev = |l: &mut Ledger, v: ArrayView1<f64>| {
            if !l.charge() {
                return None;
            }
            Some(lj(v))
        };
        let mut e: Energy<'_> = &mut ev;
        let mut x = x0.clone();
        for _ in 0..60 {
            let (energy, _) = lj(x.view());
            let Some(out) = chain.propose(&cfg, &mut led, x.view(), energy, &mut e, &mut rng)
            else {
                break;
            };
            if !out.diverged {
                x = out.x;
            }
        }
        assert!(
            chain.diag.cap_rate() > 0.5,
            "a depth cap of 1 bound on only {:.2} of proposals",
            chain.diag.cap_rate()
        );
    }
}
