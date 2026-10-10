//! Lattice-searching chains that pool their funnel bottoms in one bank.
//!
//! Each chain owns an equal share of the force budget on its own [`Ledger`]
//! and spends it one trial at a time. A trial starts from a random cluster,
//! from a bank member with a few surface atoms placed on vacant hollow sites,
//! or from a cut-and-splice of two members (Deaven and Ho, Phys. Rev. Lett.
//! 75, 288 (1995)). It is quenched and taken down the lattice descent of
//! [`crate::methods::lattice_search`], and the minimum it ends in is offered
//! to a bank of conformational space annealing (Lee, Scheraga and Rackovsky,
//! J. Comput. Chem. 18, 1222 (1997)) under the rule of Lee, Lee and Lee,
//! Phys. Rev. Lett. 91, 080201 (2003), arXiv cond-mat/0307690: it replaces
//! the member it resembles when it is lower, and otherwise displaces the
//! highest member when it resembles none and is lower than that member.
//!
//! [`Sharing::Shared`] gives the chains one bank of `chains * slots` members,
//! so a funnel any chain enters is refined by every chain and splices mix
//! structures found by different chains. [`Sharing::Private`] gives each chain
//! a bank of `slots` members, with the same trial rules, budget shares and
//! stream seeds. A one-slot private bank never splices and never draws the
//! splice coin, so a chain's random stream parts from its stream under
//! [`Sharing::Shared`] at its first bank draw, or sooner if the shared bank is
//! not yet full when the private one is, since only a full bank draws the coin
//! for a random start. With one slot the private ablation therefore removes
//! cross-chain splicing together with the exchange of members. A margin over
//! another method that the private ablation matches belongs to what the two
//! share, the trial rules, the lattice descent and its quench, and not to
//! communication between chains.
//!
//! Chains advance in synchronous generations. Parents are drawn and offers
//! admitted in chain order between generations; only the trials run in
//! parallel, each on its own ledger and random stream, so a run replays bit
//! for bit at any thread count.
//!
//! Resemblance is the mean absolute difference between the sorted distances
//! of the atoms from their centroid. The merge distance is a fraction of the
//! mean pairwise resemblance of the first full bank, by default a half, the
//! D_ave/2 of Lee, Lee and Lee. Conformational space annealing shrinks it as
//! the search goes on, and here it may shrink linearly as the budget is spent;
//! by default it holds, because a shrinking cutoff lets the variants of one
//! funnel fill the bank as distinct members.
//!
//! Parents are drawn least-used first, and a member that improves is fresh
//! again, so the effort follows the funnels that are still descending. A
//! member drawn [`Plan::retire`] times without improving gives its slot to the
//! next random start that resembles no member, whatever that start's energy.
//! Without it, a bank whose members have all stopped descending rejects every
//! start higher than its worst member, which on a funnelled surface is nearly
//! every start. Nothing in the run reads a reference energy or a structure
//! class.
//!
//! The retirement rule, the radial resemblance key, the held merge distance
//! and the counted accounting are this work's. The bank rule, cut-and-splice
//! and the lattice search are the cited methods, the last in the variant
//! [`crate::methods::lattice_search`] describes.

use ndarray::ArrayView1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::methods::cluster_hopping::{Ledger, random_cluster};
use crate::methods::lattice_search::{Lattice, Quench};
use crate::methods::splice::cut_and_splice;

/// Whether the chains see one bank or each its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sharing {
    /// One bank of `chains * slots` members, drawn from and offered to by
    /// every chain.
    Shared,
    /// A bank of `slots` members per chain, seen by that chain alone. With one
    /// slot it never splices.
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
    /// Number density of the random starts.
    pub density: f64,
    /// Closest approach allowed in random starts and splices.
    pub min_separation: f64,
    /// One bank for all chains, or one per chain.
    pub sharing: Sharing,
}

impl Default for Plan {
    fn default() -> Self {
        Self {
            chains: 40,
            slots: 1,
            fresh: 0.3,
            splice: 0.2,
            moved: (1, 4),
            merge: (0.5, 0.5),
            retire: 20,
            density: 0.7,
            min_separation: 0.85,
            sharing: Sharing::Shared,
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
        }
    }

    /// The same bank, freeing the slot of a member drawn `retire` times
    /// without improving for the next random start that resembles no member.
    pub fn with_retirement(mut self, retire: usize) -> Self {
        self.retire = retire;
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

    /// Offers a minimum under the replacement rule.
    ///
    /// Until the bank first fills, every minimum that is not one already held
    /// takes a free slot; the spread of that first bank sets the scale of the
    /// merge distance, so the first bank cannot itself be filtered by it.
    pub fn offer(&mut self, energy: f64, state: &[f64], origin: Origin) -> Admission {
        let key = radial_key(state);
        let nearest = self
            .members
            .iter()
            .enumerate()
            .map(|(i, m)| (i, resemblance(&key, &m.key)))
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

    fn calibrate(&mut self) {
        let mut total = 0.0;
        let mut pairs = 0usize;
        for i in 0..self.members.len() {
            for j in (i + 1)..self.members.len() {
                total += resemblance(&self.members[i].key, &self.members[j].key);
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
    minimum: Option<(f64, Vec<f64>)>,
}

fn choose(bank: &mut FunnelBank, rng: &mut StdRng, plan: &Plan) -> Start {
    if !bank.is_full() || rng.random::<f64>() < plan.fresh {
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
        let (origin, x) = match start {
            Start::Fresh => (
                Origin::Fresh,
                random_cluster(n, plan.density, plan.min_separation, &mut self.rng).to_vec(),
            ),
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
            minimum,
        }
    }
}

/// Runs one ensemble of `n` atoms on `budget` force calls in total.
///
/// The budget is split over the chains, the first `budget % chains` chains
/// taking one call more, so the ledgers sum to `budget` exactly.
pub fn run(
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
    let mut workers: Vec<Chain> = (0..chains)
        .map(|c| Chain {
            rng: StdRng::seed_from_u64(chain_seed(seed, c)),
            ledger: Ledger::new(share + usize::from(c < extra)),
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
    let mut banks: Vec<FunnelBank> = (0..banks)
        .map(|_| FunnelBank::new(capacity).with_retirement(plan.retire))
        .collect();
    let mut out = Run {
        best: f64::INFINITY,
        best_state: Vec::new(),
        charged: 0,
        generations: 0,
        trace: Vec::new(),
        trials: [0; 3],
        calls: [0; 3],
        admitted: [0; 3],
        bank: Vec::new(),
    };
    loop {
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
            bank.set_progress(plan.merge, spent as f64 / total.max(1) as f64);
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
                    energy,
                    origin: trial.origin,
                });
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
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
            &Lattice::default(),
        );
        assert_eq!(run.charged, budget, "chains stop only on an empty ledger");
        assert_eq!(run.calls.iter().sum::<usize>(), run.charged);
        assert!(
            run.trace
                .windows(2)
                .all(|w| w[1].energy < w[0].energy && w[1].charged >= w[0].charged)
        );
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
                &Lattice::default(),
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
                        &Lattice::default(),
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
}
