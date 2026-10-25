//! Lattice-searching chains that pool their funnel bottoms in one bank.
//!
//! Each chain owns an equal share of the force budget on its own
//! [`Ledger`](crate::methods::cluster_hopping::Ledger) and spends it one trial
//! at a time. A trial starts from a random cluster, from a bank member with a
//! few surface atoms placed on vacant hollow sites, or from a cut-and-splice of
//! two members (Deaven and Ho, Phys. Rev. Lett. 75, 288 (1995)). It is
//! quenched and taken down the lattice descent of
//! [`crate::methods::lattice_search`], and the minimum it ends in is offered
//! to a bank of conformational space annealing (Lee, Scheraga and Rackovsky,
//! J. Comput. Chem. 18, 1222 (1997)) under the rule of Lee, Lee and Lee,
//! Phys. Rev. Lett. 91, 080201 (2003), arXiv cond-mat/0307690: it replaces
//! the member it resembles when it is lower, and otherwise displaces the
//! highest member when it resembles none and is lower than that member.
//!
//! [`Sharing::Shared`](crate::methods::lattice_ensemble::Sharing::Shared)
//! gives the chains one bank of `chains * slots` members, so a funnel any
//! chain enters is refined by every chain and splices mix structures found by
//! different chains.
//! [`Sharing::Private`](crate::methods::lattice_ensemble::Sharing::Private)
//! gives each chain a bank of `slots` members, with the same trial rules,
//! budget shares and stream seeds. Splicing needs a second member to donate
//! the other half: [`choose`](crate::methods::lattice_ensemble) draws a splice
//! only when the bank holds at least two members. A one-slot private bank
//! therefore never splices and never draws the splice coin, and that chain's
//! random stream parts from its stream under `Sharing::Shared` at its first
//! bank draw, or sooner if the shared bank is not yet full when the private
//! one is, since only a full bank draws the coin for a random start. The
//! private ablation arm uses more than one slot, so the bank has a donor and
//! the splice rate is the only thing that decides whether a draw splices.
//! With one slot the private bank removes cross-chain splicing together with
//! the exchange of members. A margin over another method that a one-slot
//! private run matches belongs to what the two share, the trial rules, the
//! lattice descent and its quench, and not to communication between chains.
//!
//! Chains advance in synchronous generations. Parents are drawn and offers
//! admitted in chain order between generations; only the trials run in
//! parallel, each on its own ledger and random stream, so a run replays bit
//! for bit at any thread count.
//!
//! Resemblance defaults to the mean absolute difference between the sorted
//! distances of the atoms from their centroid. The other key is the shell
//! histogram of Lee, Lee and Lee, Phys. Rev. Lett. 91, 080201 (2003), eq. (2),
//! arXiv cond-mat/0307690. The merge distance is a fraction of the mean
//! pairwise resemblance of the first full bank, D_ave. By default that
//! fraction is held at one half, D_ave/2. Lee, Lee and Lee instead reduce it
//! by a fixed ratio so that it equals D_ave/5 after 10_000 local
//! minimizations, and hold it there. A shrinking cutoff lets the variants of
//! one funnel fill the bank as distinct members, which is why the default
//! holds.
//!
//! Parents are drawn least-used first, and a member that improves is fresh
//! again, so the effort follows the funnels that are still descending. A
//! member drawn
//! [`Plan::retire`](crate::methods::lattice_ensemble::Plan::retire) times
//! without improving gives its slot to the next random start that resembles
//! no member, whatever that start's energy. Without it, a bank whose members
//! have all stopped descending rejects every start higher than its worst
//! member, which on a funnelled surface is nearly every start. Nothing in the
//! run reads a reference energy or a structure class.
//!
//! The retirement rule, the radial resemblance key, the held merge distance
//! and the counted accounting are this work's. Retirement of zero, the shell
//! histogram, the fixed-ratio cutoff and a run with the bank switched off are
//! the published choices those four replace, selected per arm.
//! [`Plan::banked`](crate::methods::lattice_ensemble::Plan::banked) false is
//! the dynamic lattice search of Shao, Cheng and Cai on its own: every trial
//! is a fresh cluster, quenched and taken down the lattice, and nothing is
//! spliced or stored. The bank rule, cut-and-splice and the lattice search
//! are the cited methods, the last in the variant
//! [`crate::methods::lattice_search`] describes. The counted force total is
//! accounting. It is not a switch.

use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::methods::cluster_hopping::{Config, Ledger, random_cluster, random_cluster_in_radius};
use crate::methods::lattice_search::{Lattice, Quench};
use crate::methods::splice::cut_and_splice;
use crate::potentials::PairKind;

/// Whether the chains see one bank or each its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sharing {
    /// One bank of `chains * slots` members, drawn from and offered to by
    /// every chain.
    Shared,
    /// A bank of `slots` members per chain, seen by that chain alone. With one
    /// slot the bank has no donor, so it never splices.
    Private,
}

/// How a trial structure was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// A random cluster.
    Fresh,
    /// A bank member with surface atoms moved onto vacant sites.
    Moved,
    /// A cut-and-splice of two bank members.
    Spliced,
}

impl Origin {
    /// Position in per-origin tallies.
    pub fn index(self) -> usize {
        match self {
            Self::Fresh => 0,
            Self::Moved => 1,
            Self::Spliced => 2,
        }
    }

    /// Short label for logs.
    pub fn label(self) -> &'static str {
        match self {
            Self::Fresh => "fresh",
            Self::Moved => "moved",
            Self::Spliced => "spliced",
        }
    }
}

/// Which distance the bank treats as resemblance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resemblance {
    /// Sorted distances from the centroid. The difference is the mean
    /// absolute deviation. This is the default.
    Radial,
    /// Eq. (2) of Lee, Lee and Lee, Phys. Rev. Lett. 91, 080201 (2003).
    LeeCsa,
}

/// How the merge distance moves after the first bank fills.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutoffSchedule {
    /// Linear in the fraction of the force budget already spent, from
    /// [`Plan::merge`](crate::methods::lattice_ensemble::Plan::merge) `.0` to
    /// `.1`. Equal ends hold the cutoff. The default ends are both one half.
    LinearBudget,
    /// Lee, Lee and Lee, Phys. Rev. Lett. 91, 080201 (2003). The cutoff
    /// starts at D_ave/2 and is reduced by a fixed ratio so that it equals
    /// D_ave/5 after [`LEE_ANNEAL_MINIMIZATIONS`] local minimizations, then
    /// stays at D_ave/5. [`Plan::merge`](crate::methods::lattice_ensemble::Plan::merge)
    /// is not read.
    LeeFixedRatio,
}

/// Where a fresh trial places its points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StartRule {
    /// The sphere [`Config::recommended`] draws in.
    ///
    /// Radius `0.9 * L0 * N^(1/3)`, with `L0` the preset length scale
    /// ([`Config::start_radius`]).
    PresetSphere,
    /// The liquid-density ball of [`random_cluster`] at this number density.
    Density(f64),
}

/// The settings of one ensemble.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Chains, each holding an equal share of the budget.
    pub chains: usize,
    /// Bank members per chain.
    pub slots: usize,
    /// Probability that a trial starts from a random cluster once the bank
    /// is full.
    pub fresh: f64,
    /// Probability that a trial drawn from the bank splices two members
    /// instead of moving atoms of one.
    pub splice: f64,
    /// Fewest and most surface atoms a move places on vacant sites.
    pub moved: (usize, usize),
    /// Merge distance when the first bank fills and when the budget ends, as
    /// fractions of that bank's mean pairwise resemblance.
    pub merge: (f64, f64),
    /// Draws without improvement after which a member gives its slot to the
    /// next random start that resembles no member, whatever its energy.
    /// Zero keeps every member until a lower minimum displaces it.
    pub retire: usize,
    /// Distribution of a fresh trial. The default is the preset sphere.
    pub start: StartRule,
    /// Closest approach allowed in random starts and splices.
    ///
    /// A [`StartRule::PresetSphere`] draw uses the recommended preset's own
    /// separation, which is this value for a reduced Lennard-Jones cluster.
    pub min_separation: f64,
    /// One bank for all chains, or one per chain.
    pub sharing: Sharing,
    /// Radial key, or Lee, Lee and Lee's shell histogram.
    pub resemblance: Resemblance,
    /// Held or linear cutoff, or their fixed-ratio schedule.
    pub cutoff: CutoffSchedule,
    /// When false, every trial is a fresh lattice descent and the bank is
    /// not drawn, spliced or updated. That is the dynamic lattice search of
    /// Shao, Cheng and Cai on its own. The default keeps the bank.
    pub banked: bool,
}

impl Default for Plan {
    /// Frozen field values for the tabulated lattice runs.
    ///
    /// `plan_default_is_frozen` pins every field. A different default is
    /// a different plan.
    fn default() -> Self {
        Self {
            chains: 40,
            slots: 1,
            fresh: 0.3,
            splice: 0.2,
            moved: (1, 4),
            merge: (0.5, 0.5),
            retire: 20,
            start: StartRule::PresetSphere,
            min_separation: 0.85,
            sharing: Sharing::Shared,
            resemblance: Resemblance::Radial,
            cutoff: CutoffSchedule::LinearBudget,
            banked: true,
        }
    }
}

impl Plan {
    /// A fresh cluster under [`Plan::start`].
    ///
    /// [`StartRule::PresetSphere`] is the draw [`Config::recommended`] uses,
    /// so a one-chain ensemble and that chain share a start at the same seed.
    pub fn fresh_cluster<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Array1<f64> {
        match self.start {
            StartRule::PresetSphere => {
                let cfg = Config::recommended(n);
                random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, rng)
            }
            StartRule::Density(density) => random_cluster(n, density, self.min_separation, rng),
        }
    }
}

/// What happened to a minimum offered to a bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// The bank had room and the minimum was new.
    Added(usize),
    /// Lower than the member it resembles, and took its slot.
    Improved(usize),
    /// Resembles no member, and displaced the highest one.
    Displaced(usize),
    /// A random start that resembles no member, given the slot of the member
    /// drawn most often without improving.
    Recycled(usize),
    /// Resembles a member and is not lower.
    Duplicate(usize),
    /// Resembles no member and is higher than every member of a full bank.
    Rejected,
}

/// A minimum held by a bank.
#[derive(Debug, Clone)]
pub struct Member {
    /// Its energy.
    pub energy: f64,
    /// Its coordinates, flattened `3N`.
    pub state: Vec<f64>,
    /// Trials drawn from it since it last improved.
    pub draws: usize,
    /// How the trial that set it was made.
    pub origin: Origin,
    key: Vec<f64>,
}

/// Distances of the atoms from their centroid, sorted.
pub fn radial_key(x: &[f64]) -> Vec<f64> {
    let n = x.len() / 3;
    let mut centre = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            centre[k] += x[3 * i + k];
        }
    }
    let scale = (n as f64).max(1.0);
    for value in &mut centre {
        *value /= scale;
    }
    let mut radii: Vec<f64> = (0..n)
        .map(|i| {
            let d = [
                x[3 * i] - centre[0],
                x[3 * i + 1] - centre[1],
                x[3 * i + 2] - centre[2],
            ];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        })
        .collect();
    radii.sort_by(f64::total_cmp);
    radii
}

/// Mean absolute difference of two radial keys of the same length.
pub fn resemblance(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b).map(|(u, v)| (u - v).abs()).sum::<f64>() / a.len() as f64
}

/// First-shell radius in Lee, Lee and Lee, Phys. Rev. Lett. 91, 080201
/// (2003), eq. (2), in reduced units with sigma = 1.
pub const LEE_SHELL_FIRST: f64 = 1.35;
/// Second-shell radius in the same equation, same units.
pub const LEE_SHELL_SECOND: f64 = 1.70;
/// Local minimizations at which their cutoff reaches D_ave/5.
///
/// "The D_cut is reduced by a fixed ratio ... in such a way that D_cut
/// reaches D_ave/5 after 10000 local minimizations. ... The value of D_cut
/// is kept constant after it reaches the final value."
pub const LEE_ANNEAL_MINIMIZATIONS: u64 = 10_000;

/// Fraction of D_ave on that fixed-ratio schedule.
///
/// `(1/2) * (2/5) ^ min(m / 10000, 1)`. Zero minimizations gives one half.
/// Ten thousand and every count after it gives one fifth. Up to the horizon
/// the ratio of successive fractions is constant.
pub fn lee_cut_fraction(minimizations: u64) -> f64 {
    let horizon = LEE_ANNEAL_MINIMIZATIONS as f64;
    let t = (minimizations as f64 / horizon).min(1.0);
    0.5 * (2.0_f64 / 5.0).powf(t)
}

/// Shell radii for `kind`.
///
/// Lennard-Jones keeps 1.35 and 1.70. Any other pair multiplies both by
/// `r_min(kind) / r_min(Lennard-Jones)`, which is 1 for Lennard-Jones, so a
/// neighbour at the pair minimum sits in the same bin it occupies in the
/// paper's units.
pub fn lee_shell_radii(kind: PairKind) -> (f64, f64) {
    let scale = kind.r_min() / PairKind::LennardJones.r_min();
    (LEE_SHELL_FIRST * scale, LEE_SHELL_SECOND * scale)
}

/// Eq. (2) of Lee, Lee and Lee.
///
/// `H(1, n)` counts atoms with `n` neighbours inside the sphere of radius
/// `shell1`, and `H(2, n)` the same for `shell2`. A neighbour on the boundary
/// counts. The second sphere includes the first. The distance is
/// `sum_n n * (2 |H(1,n) - H'(1,n)| + |H(2,n) - H'(2,n)|)`.
pub fn lee_csa_distance(a: &[f64], b: &[f64], shell1: f64, shell2: f64) -> f64 {
    let ha = lee_histograms(a, shell1, shell2);
    let hb = lee_histograms(b, shell1, shell2);
    histogram_distance(&ha, &hb)
}

fn lee_histograms(x: &[f64], shell1: f64, shell2: f64) -> Vec<f64> {
    let n = x.len() / 3;
    let mut hist = vec![0.0; n.saturating_mul(2)];
    if n == 0 {
        return hist;
    }
    let s1 = shell1 * shell1;
    let s2 = shell2 * shell2;
    for i in 0..n {
        let mut c1 = 0usize;
        let mut c2 = 0usize;
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = x[3 * i] - x[3 * j];
            let dy = x[3 * i + 1] - x[3 * j + 1];
            let dz = x[3 * i + 2] - x[3 * j + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 <= s1 {
                c1 += 1;
            }
            if r2 <= s2 {
                c2 += 1;
            }
        }
        hist[c1] += 1.0;
        hist[n + c2] += 1.0;
    }
    hist
}

fn histogram_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || !a.len().is_multiple_of(2) {
        return f64::INFINITY;
    }
    let n = a.len() / 2;
    let mut total = 0.0;
    for k in 0..n {
        let weight = k as f64;
        total += weight * (2.0 * (a[k] - b[k]).abs() + (a[n + k] - b[n + k]).abs());
    }
    total
}

/// Same energy and same radial key: one minimum found twice.
const SAME_ENERGY: f64 = 1e-6;
const SAME_KEY: f64 = 1e-4;
/// An offer must undercut the member it resembles by this much.
const IMPROVEMENT: f64 = 1e-7;

/// Minima held apart by an annealed resemblance cutoff.
#[derive(Debug, Clone)]
pub struct FunnelBank {
    members: Vec<Member>,
    capacity: usize,
    /// Mean pairwise resemblance of the first full bank.
    scale: Option<f64>,
    /// Current merge distance; zero until the bank first fills.
    merge: f64,
    /// Draws without improvement that make a member's slot free for a
    /// random start; zero never frees one.
    retire: usize,
    /// Radial key, or the shell histogram.
    resemblance: Resemblance,
    /// Radii passed to the shell histogram. The radial key ignores them.
    shells: (f64, f64),
}

impl FunnelBank {
    /// An empty bank of at most `capacity` members.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "a bank holds at least one minimum");
        Self {
            members: Vec::with_capacity(capacity),
            capacity,
            scale: None,
            merge: 0.0,
            retire: 0,
            resemblance: Resemblance::Radial,
            shells: (LEE_SHELL_FIRST, LEE_SHELL_SECOND),
        }
    }

    /// The same bank, freeing the slot of a member drawn `retire` times
    /// without improving for the next random start that resembles no member.
    pub fn with_retirement(mut self, retire: usize) -> Self {
        self.retire = retire;
        self
    }

    /// The same bank, comparing offers with `resemblance`.
    ///
    /// The default is the radial key, which ignores `shells`.
    pub fn with_resemblance(mut self, resemblance: Resemblance, shells: (f64, f64)) -> Self {
        self.resemblance = resemblance;
        self.shells = shells;
        self
    }

    /// The members in slot order.
    pub fn members(&self) -> &[Member] {
        &self.members
    }

    /// Whether every slot is taken.
    pub fn is_full(&self) -> bool {
        self.members.len() >= self.capacity
    }

    /// Current merge distance.
    pub fn merge(&self) -> f64 {
        self.merge
    }

    /// Mean pairwise resemblance of the first full bank, once it filled.
    pub fn scale(&self) -> Option<f64> {
        self.scale
    }

    /// Moves the merge distance along `merge` for a budget `progress` in
    /// `[0, 1]`.
    pub fn set_progress(&mut self, merge: (f64, f64), progress: f64) {
        if let Some(scale) = self.scale {
            let t = progress.clamp(0.0, 1.0);
            self.merge = scale * (merge.0 + (merge.1 - merge.0) * t);
        }
    }

    /// Moves the merge distance under `schedule`.
    ///
    /// [`CutoffSchedule::LinearBudget`] is [`FunnelBank::set_progress`].
    /// [`CutoffSchedule::LeeFixedRatio`] uses [`lee_cut_fraction`] of the
    /// bank's scale and does not read `merge` or `budget_progress`.
    /// `minimizations` counts local minimizations already finished.
    pub fn set_cutoff(
        &mut self,
        schedule: CutoffSchedule,
        merge: (f64, f64),
        budget_progress: f64,
        minimizations: u64,
    ) {
        match schedule {
            CutoffSchedule::LinearBudget => self.set_progress(merge, budget_progress),
            CutoffSchedule::LeeFixedRatio => {
                if let Some(scale) = self.scale {
                    self.merge = scale * lee_cut_fraction(minimizations);
                }
            }
        }
    }

    /// Offers a minimum under the replacement rule.
    ///
    /// Until the bank first fills, every minimum that is not one already held
    /// takes a free slot; the spread of that first bank sets the scale of the
    /// merge distance, so the first bank cannot itself be filtered by it.
    pub fn offer(&mut self, energy: f64, state: &[f64], origin: Origin) -> Admission {
        let key = self.key_of(state);
        let nearest = self
            .members
            .iter()
            .enumerate()
            .map(|(i, m)| (i, self.distance(&key, &m.key)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let member = |energy: f64, key: Vec<f64>| Member {
            energy,
            state: state.to_vec(),
            draws: 0,
            origin,
            key,
        };
        if let Some((i, d)) = nearest {
            let same = d < SAME_KEY && (energy - self.members[i].energy).abs() < SAME_ENERGY;
            if same || (self.scale.is_some() && d <= self.merge) {
                if energy < self.members[i].energy - IMPROVEMENT {
                    self.members[i] = member(energy, key);
                    return Admission::Improved(i);
                }
                return Admission::Duplicate(i);
            }
        }
        if !self.is_full() {
            self.members.push(member(energy, key));
            if self.is_full() && self.scale.is_none() {
                self.calibrate();
            }
            return Admission::Added(self.members.len() - 1);
        }
        if self.retire > 0 && origin == Origin::Fresh {
            let stale = self
                .members
                .iter()
                .enumerate()
                .filter(|(_, m)| m.draws >= self.retire)
                .max_by(|a, b| {
                    a.1.draws
                        .cmp(&b.1.draws)
                        .then_with(|| a.1.energy.total_cmp(&b.1.energy))
                })
                .map(|(i, _)| i);
            if let Some(s) = stale {
                self.members[s] = member(energy, key);
                return Admission::Recycled(s);
            }
        }
        let worst = self
            .members
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.energy.total_cmp(&b.1.energy))
            .map(|(i, _)| i);
        match worst {
            Some(w) if energy < self.members[w].energy => {
                self.members[w] = member(energy, key);
                Admission::Displaced(w)
            }
            _ => Admission::Rejected,
        }
    }

    fn key_of(&self, state: &[f64]) -> Vec<f64> {
        match self.resemblance {
            Resemblance::Radial => radial_key(state),
            Resemblance::LeeCsa => lee_histograms(state, self.shells.0, self.shells.1),
        }
    }

    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        match self.resemblance {
            Resemblance::Radial => resemblance(a, b),
            Resemblance::LeeCsa => histogram_distance(a, b),
        }
    }

    fn calibrate(&mut self) {
        let mut total = 0.0;
        let mut pairs = 0usize;
        for i in 0..self.members.len() {
            for j in (i + 1)..self.members.len() {
                total += self.distance(&self.members[i].key, &self.members[j].key);
                pairs += 1;
            }
        }
        // A one-slot bank has no spread; its rule is plain descent.
        if pairs > 0 && total > 0.0 {
            self.scale = Some(total / pairs as f64);
        }
    }

    /// The least-drawn member, lowest first among equals, marked as drawn.
    pub fn draw(&mut self) -> Option<usize> {
        let pick = self
            .members
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1.draws
                    .cmp(&b.1.draws)
                    .then_with(|| a.1.energy.total_cmp(&b.1.energy))
            })
            .map(|(i, _)| i)?;
        self.members[pick].draws += 1;
        Some(pick)
    }

    /// The lowest member.
    pub fn best(&self) -> Option<&Member> {
        self.members
            .iter()
            .min_by(|a, b| a.energy.total_cmp(&b.energy))
    }
}

/// A new ensemble best: the charged total of the whole ensemble when it was
/// admitted, its energy, and how its trial started.
#[derive(Debug, Clone, Copy)]
pub struct Record {
    /// Ensemble force calls up to and including the trial that found it,
    /// counting the trials of one generation in chain order. Lattice pair work
    /// not yet settled into whole calls, under one call per chain, is not in it.
    pub charged: usize,
    /// Oracle calls up to and including that trial. The bank term is zero.
    pub force_calls: usize,
    /// Wall time from the start of the ensemble to the admission, in seconds.
    pub wall_seconds: f64,
    /// Energy of the new best.
    pub energy: f64,
    /// How the trial that found it started.
    pub origin: Origin,
}

/// What one ensemble did with its budget.
#[derive(Debug, Clone)]
pub struct Run {
    /// Lowest energy reached.
    pub best: f64,
    /// Coordinates of that minimum.
    pub best_state: Vec<f64>,
    /// Force calls charged across all chains.
    pub charged: usize,
    /// Potential calls summed over every chain and the bank.
    ///
    /// A chain call is one energy-and-gradient evaluation, or one whole unit
    /// the lattice search settled for its pair work. The bank does not
    /// evaluate the potential when it admits a member, so its term is zero
    /// and is still added. This is the count a matched budget uses.
    pub force_calls: usize,
    /// Wall time of the ensemble, in seconds.
    pub wall_seconds: f64,
    /// Generations run.
    pub generations: usize,
    /// Every new ensemble best, in order.
    pub trace: Vec<Record>,
    /// Trials by origin: fresh, moved, spliced.
    pub trials: [usize; 3],
    /// Force calls by origin.
    pub calls: [usize; 3],
    /// Minima taken into a bank (added, improved, displaced or recycled) by
    /// origin.
    pub admitted: [usize; 3],
    /// Members at the end, every bank in chain order.
    pub bank: Vec<Member>,
}

/// Random stream of chain `chain` in ensemble `seed`.
pub fn chain_seed(seed: u64, chain: usize) -> u64 {
    seed.wrapping_mul(0x9E37_79B9)
        .wrapping_add(chain as u64)
        .wrapping_add(7)
}

struct Chain {
    rng: StdRng,
    ledger: Ledger,
}

enum Start {
    Fresh,
    Moved { state: Vec<f64>, count: usize },
    Spliced { a: Vec<f64>, b: Vec<f64> },
}

struct Trial {
    origin: Origin,
    calls: usize,
    oracle_calls: usize,
    minimum: Option<(f64, Vec<f64>)>,
}

fn choose(bank: &mut FunnelBank, rng: &mut StdRng, plan: &Plan) -> Start {
    if !plan.banked || !bank.is_full() || rng.random::<f64>() < plan.fresh {
        return Start::Fresh;
    }
    let Some(parent) = bank.draw() else {
        return Start::Fresh;
    };
    let len = bank.members.len();
    if len >= 2 && rng.random::<f64>() < plan.splice {
        let mut other = rng.random_range(0..len - 1);
        if other >= parent {
            other += 1;
        }
        return Start::Spliced {
            a: bank.members[parent].state.clone(),
            b: bank.members[other].state.clone(),
        };
    }
    let (lo, hi) = plan.moved;
    let count = rng.random_range(lo.max(1)..=hi.max(lo.max(1)));
    Start::Moved {
        state: bank.members[parent].state.clone(),
        count,
    }
}

impl Chain {
    fn trial(
        &mut self,
        n: usize,
        start: Start,
        plan: &Plan,
        quench: &Quench,
        lattice: &Lattice,
    ) -> Trial {
        let before = self.ledger.spent();
        crate::methods::lattice_search::oracle_begin();
        let (origin, x) = match start {
            Start::Fresh => (Origin::Fresh, plan.fresh_cluster(n, &mut self.rng).to_vec()),
            Start::Moved { state, count } => {
                (Origin::Moved, lattice.shuffle(&state, count, &mut self.rng))
            }
            Start::Spliced { a, b } => (
                Origin::Spliced,
                cut_and_splice(
                    ArrayView1::from(&a),
                    ArrayView1::from(&b),
                    None,
                    plan.min_separation,
                    &mut self.rng,
                )
                .to_vec(),
            ),
        };
        let minimum = quench
            .relax(&mut self.ledger, &x)
            .filter(|first| first.converged)
            .map(|first| {
                let (energy, state, _) =
                    lattice.descend(quench, &mut self.ledger, first.energy, &first.state);
                (energy, state)
            });
        Trial {
            origin,
            calls: self.ledger.spent() - before,
            oracle_calls: crate::methods::lattice_search::oracle_end(),
            minimum,
        }
    }
}

/// Runs one ensemble of `n` atoms on `budget` force calls in total.
///
/// The budget is split over the chains, the first `budget % chains` chains
/// taking one call more, so the ledgers sum to `budget` exactly.
/// Site geometry is [`Lattice::for_pair`] of `kind`.
pub fn run(
    n: usize,
    budget: usize,
    seed: u64,
    plan: &Plan,
    quench: &Quench,
    kind: PairKind,
) -> Run {
    let lattice = Lattice::for_pair(kind);
    let mut quench = quench.clone();
    quench.kind = kind;
    run_with_lattice(n, budget, seed, plan, &quench, &lattice)
}

/// Runs one ensemble on a caller-built lattice.
///
/// [`run`] builds that lattice from the pair kind. A harness that only
/// changes the candidate count keeps the same lengths through this entry.
pub fn run_with_lattice(
    n: usize,
    budget: usize,
    seed: u64,
    plan: &Plan,
    quench: &Quench,
    lattice: &Lattice,
) -> Run {
    let chains = plan.chains.max(1);
    let slots = plan.slots.max(1);
    let share = budget / chains;
    let extra = budget % chains;
    let caller_stop = crate::caller_stop::bound();
    let mut workers: Vec<Chain> = (0..chains)
        .map(|c| {
            let budget = share + usize::from(c < extra);
            Chain {
                rng: StdRng::seed_from_u64(chain_seed(seed, c)),
                ledger: match &caller_stop {
                    Some(flag) => Ledger::with_stop(budget, std::sync::Arc::clone(flag)),
                    None => Ledger::new(budget),
                },
            }
        })
        .collect();
    let (banks, capacity) = match plan.sharing {
        Sharing::Shared => (1, chains * slots),
        Sharing::Private => (chains, slots),
    };
    let bank_of = |c: usize| match plan.sharing {
        Sharing::Shared => 0,
        Sharing::Private => c,
    };
    let shells = lee_shell_radii(lattice.kind);
    let mut banks: Vec<FunnelBank> = (0..banks)
        .map(|_| {
            FunnelBank::new(capacity)
                .with_retirement(plan.retire)
                .with_resemblance(plan.resemblance, shells)
        })
        .collect();
    let mut quenches = vec![0u64; banks.len()];
    let started = std::time::Instant::now();
    let bank_calls = 0usize;
    let mut out = Run {
        best: f64::INFINITY,
        best_state: Vec::new(),
        charged: 0,
        force_calls: 0,
        wall_seconds: 0.0,
        generations: 0,
        trace: Vec::new(),
        trials: [0; 3],
        calls: [0; 3],
        admitted: [0; 3],
        bank: Vec::new(),
    };
    loop {
        if crate::caller_stop::raised() {
            break;
        }
        let active: Vec<usize> = (0..chains)
            .filter(|&c| workers[c].ledger.remaining() > 0)
            .collect();
        if active.is_empty() {
            break;
        }
        out.generations += 1;
        for (b, bank) in banks.iter_mut().enumerate() {
            let (spent, total) = workers
                .iter()
                .enumerate()
                .filter(|(c, _)| bank_of(*c) == b)
                .fold((0usize, 0usize), |(s, t), (_, w)| {
                    (s + w.ledger.spent(), t + w.ledger.budget())
                });
            bank.set_cutoff(
                plan.cutoff,
                plan.merge,
                spent as f64 / total.max(1) as f64,
                quenches[b],
            );
        }
        let starts: Vec<Start> = active
            .iter()
            .map(|&c| choose(&mut banks[bank_of(c)], &mut workers[c].rng, plan))
            .collect();
        let jobs: Vec<(&mut Chain, Start)> = workers
            .iter_mut()
            .filter(|w| w.ledger.remaining() > 0)
            .zip(starts)
            .collect();
        let trials: Vec<Trial> = jobs
            .into_par_iter()
            .map(|(chain, start)| chain.trial(n, start, plan, quench, lattice))
            .collect();
        for (&c, trial) in active.iter().zip(trials) {
            let o = trial.origin.index();
            out.charged += trial.calls;
            out.force_calls += trial.oracle_calls;
            out.trials[o] += 1;
            out.calls[o] += trial.calls;
            let Some((energy, state)) = trial.minimum else {
                continue;
            };
            if energy < out.best {
                out.best = energy;
                out.best_state.clone_from(&state);
                out.trace.push(Record {
                    charged: out.charged,
                    force_calls: out.force_calls,
                    wall_seconds: started.elapsed().as_secs_f64(),
                    energy,
                    origin: trial.origin,
                });
            }
            quenches[bank_of(c)] += 1;
            if !plan.banked {
                continue;
            }
            match banks[bank_of(c)].offer(energy, &state, trial.origin) {
                Admission::Added(_)
                | Admission::Improved(_)
                | Admission::Displaced(_)
                | Admission::Recycled(_) => {
                    out.admitted[o] += 1;
                }
                Admission::Duplicate(_) | Admission::Rejected => {}
            }
        }
    }
    out.bank = banks.into_iter().flat_map(|b| b.members).collect();
    // Admission compares stored energies. It does not call the potential.
    out.force_calls += bank_calls;
    out.wall_seconds = started.elapsed().as_secs_f64();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::cluster_hopping::{Config, random_cluster_in_radius};
    use crate::potentials::PairKind;
    use ndarray::Array1;

    fn line(points: usize, spacing: f64) -> Vec<f64> {
        (0..points)
            .flat_map(|i| [i as f64 * spacing, 0.0, 0.0])
            .collect()
    }

    #[test]
    fn a_lower_minimum_replaces_the_member_it_resembles() {
        let mut bank = FunnelBank::new(2);
        assert_eq!(
            bank.offer(-1.0, &line(4, 1.0), Origin::Fresh),
            Admission::Added(0)
        );
        assert_eq!(
            bank.offer(-2.0, &line(4, 3.0), Origin::Fresh),
            Admission::Added(1)
        );
        let scale = bank.scale().expect("a full bank has a scale");
        bank.set_progress((0.5, 0.1), 0.0);
        assert!((bank.merge() - 0.5 * scale).abs() < 1e-12);
        // Close to the first member, lower than it: takes its slot.
        assert_eq!(
            bank.offer(-1.5, &line(4, 1.01), Origin::Moved),
            Admission::Improved(0)
        );
        assert_eq!(bank.members()[0].origin, Origin::Moved);
        // Close to the first member again, not lower: discarded.
        assert_eq!(
            bank.offer(-1.2, &line(4, 1.02), Origin::Moved),
            Admission::Duplicate(0)
        );
        // Unlike both and lower than the highest: displaces it.
        assert_eq!(
            bank.offer(-1.8, &line(4, 9.0), Origin::Spliced),
            Admission::Displaced(0)
        );
        // Unlike both and higher than both: rejected.
        assert_eq!(
            bank.offer(-0.5, &line(4, 20.0), Origin::Fresh),
            Admission::Rejected
        );
    }

    #[test]
    fn the_same_minimum_does_not_take_two_slots() {
        let mut bank = FunnelBank::new(3);
        bank.offer(-1.0, &line(4, 1.0), Origin::Fresh);
        assert_eq!(
            bank.offer(-1.0, &line(4, 1.0), Origin::Fresh),
            Admission::Duplicate(0)
        );
        assert_eq!(bank.members().len(), 1);
        assert!(bank.scale().is_none());
    }

    #[test]
    fn draws_go_to_the_least_drawn_member_first() {
        let mut bank = FunnelBank::new(2);
        bank.offer(-1.0, &line(4, 1.0), Origin::Fresh);
        bank.offer(-2.0, &line(4, 3.0), Origin::Fresh);
        assert_eq!(bank.draw(), Some(1));
        assert_eq!(bank.draw(), Some(0));
        assert_eq!(bank.draw(), Some(1));
    }

    #[test]
    fn a_member_drawn_without_improving_yields_to_a_random_start() {
        let mut bank = FunnelBank::new(2).with_retirement(2);
        bank.offer(-1.0, &line(4, 1.0), Origin::Fresh);
        bank.offer(-2.0, &line(4, 3.0), Origin::Fresh);
        bank.set_progress((0.5, 0.5), 0.0);
        assert_eq!(bank.draw(), Some(1));
        assert_eq!(bank.draw(), Some(0));
        assert_eq!(bank.draw(), Some(1));
        // Higher than both members and like neither: a refinement is
        // rejected, a random start takes the stalled member's slot.
        assert_eq!(
            bank.offer(-0.5, &line(4, 9.0), Origin::Moved),
            Admission::Rejected
        );
        assert_eq!(
            bank.offer(-0.5, &line(4, 9.0), Origin::Fresh),
            Admission::Recycled(1)
        );
        assert_eq!(bank.members()[1].draws, 0);
        assert_eq!(bank.members()[1].origin, Origin::Fresh);
        // Member 0 has one draw, short of the threshold.
        assert_eq!(
            bank.offer(-0.7, &line(4, 30.0), Origin::Fresh),
            Admission::Displaced(1)
        );
    }

    fn small_plan(sharing: Sharing) -> Plan {
        Plan {
            chains: 6,
            slots: 2,
            sharing,
            ..Plan::default()
        }
    }

    #[test]
    fn the_ledgers_sum_to_the_budget_and_never_exceed_it() {
        let budget = 20_003;
        let run = run(
            13,
            budget,
            3,
            &small_plan(Sharing::Shared),
            &Quench::default(),
            PairKind::LennardJones,
        );
        assert_eq!(run.charged, budget, "chains stop only on an empty ledger");
        assert_eq!(run.calls.iter().sum::<usize>(), run.charged);
        assert_eq!(
            run.force_calls, run.charged,
            "the reported count must equal the oracle calls, bank term included"
        );
        assert!(run.wall_seconds >= 0.0);
        assert!(
            run.trace
                .windows(2)
                .all(|w| w[1].energy < w[0].energy && w[1].charged >= w[0].charged)
        );
    }

    #[test]
    fn reported_force_count_equals_the_sum_of_oracle_calls() {
        for sharing in [Sharing::Shared, Sharing::Private] {
            let run = run(
                13,
                900,
                2,
                &small_plan(sharing),
                &Quench::default(),
                PairKind::LennardJones,
            );
            assert_eq!(
                run.force_calls, run.charged,
                "{sharing:?}: reported {} oracle {}",
                run.force_calls, run.charged
            );
            assert!(run.force_calls > 0, "{sharing:?} recorded no calls");
        }
    }

    #[test]
    fn thirteen_atoms_reach_the_icosahedron() {
        for sharing in [Sharing::Shared, Sharing::Private] {
            let run = run(
                13,
                30_000,
                5,
                &small_plan(sharing),
                &Quench::default(),
                PairKind::LennardJones,
            );
            assert!(
                (run.best - -44.326801).abs() < 1e-5,
                "{sharing:?} ended at {}",
                run.best
            );
        }
    }

    #[test]
    fn a_run_replays_bit_for_bit_at_any_thread_count() {
        let plan = small_plan(Sharing::Shared);
        let go = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| {
                    run(
                        19,
                        40_000,
                        11,
                        &plan,
                        &Quench::default(),
                        PairKind::LennardJones,
                    )
                })
        };
        let one = go(1);
        let three = go(3);
        assert_eq!(one.best.to_bits(), three.best.to_bits());
        assert_eq!(one.charged, three.charged);
        assert_eq!(one.generations, three.generations);
        assert_eq!(one.trace.len(), three.trace.len());
        for (a, b) in one.trace.iter().zip(&three.trace) {
            assert_eq!(a.charged, b.charged);
            assert_eq!(a.energy.to_bits(), b.energy.to_bits());
        }
        assert!(
            one.best_state
                .iter()
                .zip(&three.best_state)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    fn quench_lj(x: &Array1<f64>, steps: usize) -> f64 {
        let n = x.len() / 3;
        let pot = crate::potentials::PairPotential::lennard_jones(n);
        let mut opt = crate::methods::warm_lbfgs::WarmLbfgs::default();
        let (energy, _, _) = opt.minimize(x.view(), steps, |v| Some(pot.value_and_gradient(v)));
        energy
    }

    #[test]
    fn serial_arm_matches_recommended_first_quench() {
        let n = 13usize;
        let seed = 11u64;
        let cfg = Config::recommended(n);
        let radius = 0.9 * cfg.length_scale * (n as f64).cbrt();
        assert_eq!(cfg.start_radius().to_bits(), radius.to_bits());
        let plan = Plan {
            chains: 1,
            ..Plan::default()
        };
        assert_eq!(plan.start, StartRule::PresetSphere);
        let mut recommended_rng = StdRng::seed_from_u64(seed);
        let recommended = random_cluster_in_radius(
            n,
            cfg.start_radius(),
            cfg.min_separation,
            &mut recommended_rng,
        );
        let mut serial_rng = StdRng::seed_from_u64(seed);
        let serial = plan.fresh_cluster(n, &mut serial_rng);
        let recommended_energy = quench_lj(&recommended, cfg.relax_steps);
        let serial_energy = quench_lj(&serial, cfg.relax_steps);
        assert_eq!(serial_energy.to_bits(), recommended_energy.to_bits());
    }

    #[test]
    fn plan_default_is_frozen() {
        let plan = Plan::default();
        assert_eq!(plan.chains, 40);
        assert_eq!(plan.slots, 1);
        assert_eq!(plan.fresh.to_bits(), 0.3_f64.to_bits());
        assert_eq!(plan.splice.to_bits(), 0.2_f64.to_bits());
        assert_eq!(plan.moved, (1, 4));
        assert_eq!(plan.merge.0.to_bits(), 0.5_f64.to_bits());
        assert_eq!(plan.merge.1.to_bits(), 0.5_f64.to_bits());
        assert_eq!(plan.retire, 20);
        assert_eq!(plan.start, StartRule::PresetSphere);
        assert_eq!(plan.min_separation.to_bits(), 0.85_f64.to_bits());
        assert_eq!(plan.sharing, Sharing::Shared);
        assert_eq!(plan.resemblance, Resemblance::Radial);
        assert_eq!(plan.cutoff, CutoffSchedule::LinearBudget);
        assert!(plan.banked);
    }

    #[test]
    fn a_zero_splice_rate_never_splices() {
        let plan = Plan {
            chains: 4,
            slots: 2,
            fresh: 0.0,
            splice: 0.0,
            ..Plan::default()
        };
        let run = run(
            13,
            400,
            1,
            &plan,
            &Quench::default(),
            PairKind::LennardJones,
        );
        assert_eq!(run.trials[Origin::Spliced.index()], 0);
        assert_eq!(run.admitted[Origin::Spliced.index()], 0);
    }

    fn cheap_quench() -> Quench {
        Quench {
            max_iterations: 2,
            rms_tolerance: 1.0e6,
            ..Quench::default()
        }
    }

    #[test]
    fn retire_zero_never_frees_a_slot() {
        let mut bank = FunnelBank::new(2).with_retirement(0);
        bank.offer(-1.0, &line(4, 1.0), Origin::Fresh);
        bank.offer(-2.0, &line(4, 3.0), Origin::Fresh);
        bank.set_progress((0.5, 0.5), 0.0);
        for _ in 0..6 {
            bank.draw();
        }
        assert_eq!(
            bank.offer(-0.5, &line(4, 9.0), Origin::Fresh),
            Admission::Rejected
        );
    }

    fn tetrahedron() -> Vec<f64> {
        let height = (6.0_f64).sqrt() / 3.0;
        let mid = (3.0_f64).sqrt() / 6.0;
        let side = (3.0_f64).sqrt() / 2.0;
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, side, 0.0],
            [0.5, mid, height],
        ];
        points.into_iter().flatten().collect()
    }

    #[test]
    fn lee_distance_matches_equation_2() {
        let straight = line(4, 1.0);
        let tetra = tetrahedron();
        // Line: two ends with coordination 1 and two interiors with
        // coordination 2, in both shells, because the second neighbour sits
        // at 2, past 1.70. Tetrahedron: every atom has coordination 3 in
        // both shells. Eq. (2) then totals 54.
        let distance = lee_csa_distance(&straight, &tetra, LEE_SHELL_FIRST, LEE_SHELL_SECOND);
        assert_eq!(distance, 54.0);
        assert_eq!(
            lee_csa_distance(&straight, &straight, LEE_SHELL_FIRST, LEE_SHELL_SECOND),
            0.0
        );
        let stretched = line(4, 1.2);
        assert_eq!(
            lee_csa_distance(&straight, &stretched, LEE_SHELL_FIRST, LEE_SHELL_SECOND),
            0.0
        );
        assert!(resemblance(&radial_key(&straight), &radial_key(&stretched)) > 1e-3);
        let mut lee = FunnelBank::new(2)
            .with_resemblance(Resemblance::LeeCsa, (LEE_SHELL_FIRST, LEE_SHELL_SECOND));
        assert_eq!(
            lee.offer(-1.0, &straight, Origin::Fresh),
            Admission::Added(0)
        );
        assert_eq!(
            lee.offer(-1.0, &stretched, Origin::Fresh),
            Admission::Duplicate(0)
        );
        let mut radial = FunnelBank::new(2);
        assert_eq!(
            radial.offer(-1.0, &straight, Origin::Fresh),
            Admission::Added(0)
        );
        assert_eq!(
            radial.offer(-1.0, &stretched, Origin::Fresh),
            Admission::Added(1)
        );
    }

    #[test]
    fn lee_cutoff_goes_from_half_to_a_fifth_by_a_fixed_ratio() {
        assert!((lee_cut_fraction(0) - 0.5).abs() < 1e-15);
        assert!((lee_cut_fraction(LEE_ANNEAL_MINIMIZATIONS) - 0.2).abs() < 1e-12);
        assert!((lee_cut_fraction(LEE_ANNEAL_MINIMIZATIONS * 2) - 0.2).abs() < 1e-12);
        let step = lee_cut_fraction(1) / lee_cut_fraction(0);
        assert!((lee_cut_fraction(2) / lee_cut_fraction(1) - step).abs() < 1e-12);
        let published = (2.0_f64 / 5.0).powf(1.0 / LEE_ANNEAL_MINIMIZATIONS as f64);
        assert!((step - published).abs() < 1e-12);
        let lj = lee_shell_radii(PairKind::LennardJones);
        assert_eq!(lj.0.to_bits(), LEE_SHELL_FIRST.to_bits());
        assert_eq!(lj.1.to_bits(), LEE_SHELL_SECOND.to_bits());
        let morse = lee_shell_radii(PairKind::Morse { rho: 14.0 });
        let scale = 1.0 / PairKind::LennardJones.r_min();
        assert!((morse.0 - LEE_SHELL_FIRST * scale).abs() < 1e-12);
        assert!((morse.1 - LEE_SHELL_SECOND * scale).abs() < 1e-12);

        let mut bank = FunnelBank::new(2);
        bank.offer(-1.0, &line(4, 1.0), Origin::Fresh);
        bank.offer(-2.0, &line(4, 3.0), Origin::Fresh);
        let scale = bank.scale().expect("full bank");
        bank.set_cutoff(CutoffSchedule::LeeFixedRatio, (0.9, 0.9), 1.0, 0);
        assert!((bank.merge() - 0.5 * scale).abs() < 1e-12);
        bank.set_cutoff(
            CutoffSchedule::LeeFixedRatio,
            (0.9, 0.9),
            0.0,
            LEE_ANNEAL_MINIMIZATIONS,
        );
        assert!((bank.merge() - 0.2 * scale).abs() < 1e-9 * scale.max(1.0));
        bank.set_cutoff(CutoffSchedule::LinearBudget, (0.5, 0.5), 0.7, 10_000);
        assert!((bank.merge() - 0.5 * scale).abs() < 1e-12);
    }

    #[test]
    fn a_one_slot_private_bank_does_not_splice_and_two_slots_do() {
        let quench = cheap_quench();
        let base = Plan {
            chains: 4,
            fresh: 0.0,
            splice: 1.0,
            sharing: Sharing::Private,
            ..Plan::default()
        };
        let one = run(
            13,
            600,
            1,
            &Plan {
                slots: 1,
                ..base.clone()
            },
            &quench,
            PairKind::LennardJones,
        );
        let two = run(
            13,
            600,
            1,
            &Plan { slots: 2, ..base },
            &quench,
            PairKind::LennardJones,
        );
        let spliced = Origin::Spliced.index();
        println!(
            "splice_probe slots1 {} slots2 {}",
            one.trials[spliced], two.trials[spliced]
        );
        assert_eq!(one.trials[spliced], 0);
        assert!(
            two.trials[spliced] > 0,
            "two-slot private spliced {}",
            two.trials[spliced]
        );
    }

    #[test]
    fn an_unbanked_run_is_a_fresh_lattice_descent() {
        let plan = Plan {
            chains: 3,
            slots: 2,
            fresh: 0.0,
            splice: 1.0,
            banked: false,
            ..Plan::default()
        };
        let run = run(13, 400, 4, &plan, &cheap_quench(), PairKind::LennardJones);
        assert_eq!(run.trials[Origin::Spliced.index()], 0);
        assert_eq!(run.trials[Origin::Moved.index()], 0);
        assert!(run.trials[Origin::Fresh.index()] > 0);
        assert!(run.bank.is_empty());
        assert!(run.best.is_finite());
    }
}
