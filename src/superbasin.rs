//! Leaving a superbasin by solving the absorbing chain the run already built.
//!
//! Measured at 98 points: twelve million charged evaluations buy 401177 hops
//! over 11366 distinct basins, about 35 visits each, and the run ends at
//! -539.81 against a reference of -543.665361. The chain is not wandering, it
//! is cycling. Eleven mechanisms have been measured against that trap by
//! reweighting where the chain walks, and the three that helped all changed
//! what a hop can *reach* instead.
//!
//! A set of basins the chain cycles within, with rare transitions out of it, is
//! a superbasin, and kinetic Monte Carlo has exact machinery for one. The
//! transitions are already recorded, so the machinery costs no force
//! evaluations at all: it is linear algebra on data the run collected while
//! failing.
//!
//! # The flat exit
//!
//! Order the states so the transient ones (the superbasin) come first and the
//! absorbing ones (its boundary) last, giving the transition matrix in
//! canonical form
//!
//! ```text
//!   P = [ Q  R ]
//!       [ 0  I ]
//! ```
//!
//! The fundamental matrix `N = (I - Q)^-1` counts expected visits, `B = N R`
//! holds the absorption probabilities and `t = N 1` the expected steps to
//! absorption. Row `i` of `B` is the distribution over exits for a chain
//! standing in basin `i`, and the escape move samples it and jumps there,
//! rather than diffusing there over the thousands of hops the same algebra says
//! it would take.
//!
//! Nothing here forms `N`. Two facts do the work.
//!
//! Row `i` of `B` and entry `i` of `t` come from *one* transposed solve rather
//! than from `|A|` of them: with `(I - Q)^T y = e_i` the vector `y` is row `i`
//! of `N`, so `B_i = y^T R` and `t_i = y^T 1`. Verified symbolically over a
//! general 3-by-2 canonical form.
//!
//! And the solve itself is done by node elimination, the graph transformation
//! of Trygubenko and Wales (doi:10.1063/1.2198806), which renormalises
//!
//! ```text
//!   P'_ij  = P_ij + P_ix P_xj / (1 - P_xx)
//!   tau'_i = tau_i + P_ix tau_x / (1 - P_xx)
//! ```
//!
//! as each transient state `x` is removed. Every quantity stays nonnegative and
//! the one subtraction, `1 - P_xx`, is evaluated as the sum of the other
//! outgoing probabilities, so no cancellation occurs anywhere in the
//! elimination. That matters because `I - Q` is worst conditioned exactly when
//! the superbasin traps hardest, which is when the method is wanted.
//!
//! # Conditioning, exactly rather than by rule of thumb
//!
//! `N >= 0` elementwise, so `||N||_inf` is exactly `max_i t_i`, and therefore
//!
//! ```text
//!   cond_inf(I - Q) = ||I - Q||_inf * max_i t_i,   with ||I - Q||_inf <= 2.
//! ```
//!
//! The condition number is the trapping: a superbasin that takes ten thousand
//! jumps to leave presents a system with condition number about twenty
//! thousand. Checked against `numpy.linalg.cond` on 200 random substochastic
//! blocks, agreeing to 1e-6 relative. It is reported per solve rather than
//! assumed, because this crate has already had one numerical failure read as a
//! modelling failure until somebody computed the rank: a design matrix at
//! condition 1.6e71 with numerical rank 2 of 11.
//!
//! # The same operator is the successor representation
//!
//! `N = (I - Q)^-1` is the successor representation of reinforcement learning,
//! `M = (I - gamma P)^-1`, which Dayan introduced as the expected discounted
//! future occupancy of each state (doi:10.1162/neco.1993.5.4.613). One operator,
//! two boundary treatments: discounting by `gamma` there, absorption at the
//! superbasin boundary here. Saying so is not decoration. It names the limit of
//! what this module can do and points at the literature that addressed it.
//!
//! Forming `N` exactly requires having visited the states. That is enough to
//! leave a superbasin already mapped and useless for reaching one never seen,
//! and a 98-point run visits 11366 basins on a landscape whose minimum count
//! grows like `exp(alpha N)`. Reinforcement learning met the same wall and went
//! around it by *learning* the successor representation with function
//! approximation over state features rather than tabulating it over enumerated
//! states, which generalises occupancy predictions to states never visited
//! (successor features, doi:10.1073/pnas.1907370117). The corresponding
//! question here is whether the exit distribution of a basin is predictable
//! from structural features of that basin rather than from its row in a visited
//! transition matrix. The features exist in this crate already, in
//! [`crate::structure`]. This module does not answer that question; it makes
//! the exact object whose approximation the question is about.
//!
//! Two further correspondences, stated because they are exact rather than
//! suggestive. The coarse jump built here is an option in the sense of Sutton,
//! Precup and Singh (doi:10.1016/S0004-3702(99)00052-1): a temporally extended
//! action with an initiation set, an internal policy and a termination
//! condition. Hierarchical reinforcement learning discovers options rather than
//! detecting them from a timescale-separation threshold, which is what
//! [`LumpParams::min_separation`] does. And archiving states, returning to a
//! promising one and exploring from there, without relying on diffusion to get
//! back, is the structure of Go-Explore (doi:10.1038/s41586-020-03157-9); the
//! archive here is keyed by basin and the returning is what this module adds.
//!
//! # Lumping, and then recursion
//!
//! The flat exit is half the method. A set `S` whose internal transitions are
//! fast against its exits reaches quasi-equilibrium before it escapes, so it
//! can be replaced by one coarse state with effective exit rates
//!
//! ```text
//!   k_eff(S -> j) = sum_{i in S} pi_i^S k_ij
//! ```
//!
//! with `pi^S` the stationary vector of the chain restricted to `S`. That is
//! the mean rate method of Chatterjee and Voter (doi:10.1063/1.3409606), in the
//! graph framing of Stamatakis and Vlachos (doi:10.1063/1.3596751); Novotny's
//! absorbing chains (doi:10.1103/PhysRevLett.74.1) and the basin finding of
//! Puchala, Falk and Garikipati (doi:10.1063/1.3369627) are the same family.
//! Lumping the coarse graph again gives a hierarchy, and that hierarchy is a
//! funnel decomposition derived from the run's own transitions.
//!
//! Lumping is an approximation and it is checked rather than trusted. The
//! quasi-equilibrium has to actually hold: a set that is merely well connected
//! is not lumpable. The test is the timescale ratio, escape time over internal
//! equilibration time, both measured in jumps, and a set is lumped only when
//! that ratio clears [`LumpParams::min_separation`]. The residual error is
//! `O(1 / ratio)`: on a three-state trap with separation 1040 the lumped exit
//! distribution differs from the exact absorbing-chain answer by 3.3e-4 in
//! total variation.
//!
//! # The counts are biased, and the bias is divided out
//!
//! The chain that records these transitions is running well-tempered
//! metadynamics, which deliberately distorts the visit distribution, so raw
//! accepted-hop counts estimate the transition matrix of a landscape that
//! changes as the run proceeds. Building a quasi-equilibrium from them gives an
//! occupancy belonging to no distribution. This crate has that fault
//! catalogued once already, in replica exchange computing swap acceptances on
//! raw energies between chains carrying different biases, which produced a
//! ladder that never stratified.
//!
//! The correction costs nothing. The move kernel does not see the bias, so a
//! proposal from `i` to `j` is drawn at the same frequency whatever has been
//! deposited, and the unbiased Metropolis chain would accept it with
//! probability `min(1, exp(-(E_j - E_i) / T))`, a number already available at
//! the hop. Accumulating that over proposals, accepted or not, estimates the
//! unbiased rate `q_ij a_ij` directly. This is waste recycling: every proposal
//! contributes, the bias appears nowhere in the estimator, and no evaluation is
//! paid for it. The raw accepted counts are kept alongside as a diagnostic, so
//! the size of the distortion is visible rather than assumed away.
//!
//! # What it did
//!
//! Measured on Elja against `thompson,rscreen`, 24 seeds an arm, paired on the
//! seed: 12/24 against 12/24 at 75 points and three million evaluations, 2/24
//! against 0/24 at 98 points and three million, 5/24 against 6/24 at 98 points
//! and twelve million. No pair separates.
//!
//! The mechanism ran. At 98 points and twelve million evaluations it took 822
//! jumps across the arm, standing in for 3.04 million hops of diffusion by its
//! own accounting, a third of the 9.1 million the arm spent, with 278 of them
//! landing lower than they left. Conditioning stayed mild, a median of 214 and
//! a maximum of 359, and every solve reached its 1e-10 tolerance. Paired, it
//! flipped five seeds into a solve at 75 points and five out of one.
//!
//! The limit is the operator rather than the code. `N` exists over states that
//! were visited, so the exit it names is a basin the run has already entered,
//! and reaching a visited basin sooner does not enlarge what the search can
//! reach. The coarse-graining did recover structure the descriptors here could
//! not: every run built a hierarchy, median depth two and up to four, with
//! lump separations of 13 to 16 typically and up to 119, and the coarse states
//! are separable by polyhedral template fractions at F = 128 pooled. The
//! partition is real; the move built on it is not the one that pays.
//!
//! # What is left out, stated
//!
//! A proposal landing in a basin the run has never registered contributes to
//! the escape rate but names no target. Such mass is counted, carried through
//! the algebra as an extra absorbing column ([`UNKNOWN`]), and reported,
//! because a basin with no stored structure cannot be jumped to: the exit
//! distribution the move samples is conditional on landing somewhere the run
//! holds coordinates for, and the conditioning mass is reported on the jump.

use std::collections::{BTreeMap, BTreeSet};

use ndarray::{Array1, ArrayView1};
use rand::Rng;

/// Absorbing-state identifier standing for a basin the run never registered.
///
/// Carried as a real column so the exit distribution is a proper distribution
/// and the unreachable share is visible, rather than being silently folded into
/// the escape rate.
pub const UNKNOWN: usize = usize::MAX;

// ---------------------------------------------------------------------------
// Observed transitions
// ---------------------------------------------------------------------------

/// Directed transition counts between basins, with the deposited bias divided
/// out.
///
/// Two sets of counts are kept. The reweighted one accumulates the unbiased
/// Metropolis acceptance probability of every proposal and is what the algebra
/// uses; the accepted one counts what the biased chain actually did, and exists
/// so the distortion can be measured instead of assumed small.
#[derive(Debug, Default, Clone)]
pub struct HopCounts {
    /// Reweighted mass from a source to each named destination.
    out: BTreeMap<usize, BTreeMap<usize, f64>>,
    /// Reweighted mass leaving a source towards a basin with no identifier.
    leak: BTreeMap<usize, f64>,
    /// Time spent in a source, in hops.
    time: BTreeMap<usize, f64>,
    /// Hops the biased chain accepted, per ordered pair.
    accepted: BTreeMap<(usize, usize), u64>,
    /// Proposals recorded in total.
    pub observations: u64,
}

impl HopCounts {
    /// Empty counts.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one proposal out of `from`.
    ///
    /// `to` is the destination basin when the run holds an identifier for it,
    /// `None` when the quench landed somewhere never registered. `accept` is
    /// the acceptance probability of the *unbiased* chain, which is what
    /// removes the deposited bias from the estimate; pass `0.0` for a hop that
    /// produced no usable destination, which then counts only as time spent.
    pub fn observe(&mut self, from: usize, to: Option<usize>, accept: f64) {
        let a = accept.clamp(0.0, 1.0);
        self.observations += 1;
        *self.time.entry(from).or_insert(0.0) += 1.0;
        match to {
            Some(j) if j != from => {
                *self.out.entry(from).or_default().entry(j).or_insert(0.0) += a;
            }
            // A proposal that returns to the same basin is time, not a
            // transition. Recording it as an edge would load the diagonal, and
            // near a deep minimum roughly nineteen proposals in twenty return.
            Some(_) => {}
            None => {
                *self.leak.entry(from).or_insert(0.0) += a;
            }
        }
    }

    /// Adds transition mass directly, for building a coarse graph whose weights
    /// are effective rates rather than observed proposals.
    pub fn observe_weighted(&mut self, from: usize, to: Option<usize>, weight: f64) {
        if !(weight > 0.0) {
            return;
        }
        match to {
            Some(j) if j != from => {
                *self.out.entry(from).or_default().entry(j).or_insert(0.0) += weight;
            }
            Some(_) => {}
            None => {
                *self.leak.entry(from).or_insert(0.0) += weight;
            }
        }
    }

    /// Adds holding time to a state, in hops.
    pub fn add_time(&mut self, state: usize, hops: f64) {
        *self.time.entry(state).or_insert(0.0) += hops;
    }

    /// Records that the biased chain accepted a hop, for the distortion
    /// diagnostic only.
    pub fn observe_accepted(&mut self, from: usize, to: usize) {
        if from != to {
            *self.accepted.entry((from, to)).or_insert(0) += 1;
        }
    }

    /// Basins seen as a source or a destination.
    pub fn nodes(&self) -> BTreeSet<usize> {
        let mut s: BTreeSet<usize> = self.out.keys().copied().collect();
        for tos in self.out.values() {
            s.extend(tos.keys().copied());
        }
        s.extend(self.time.keys().copied());
        s
    }

    /// Ordered pairs carrying reweighted mass.
    pub fn n_edges(&self) -> usize {
        self.out.values().map(|m| m.len()).sum()
    }

    /// How far the biased counts sit from the reweighted ones.
    ///
    /// Total variation between the two edge distributions over the pairs either
    /// records. Zero means the deposits changed nothing about which
    /// transitions were seen; large means the raw counts describe a landscape
    /// the reweighted estimate does not, which is the fault this correction
    /// exists to avoid.
    pub fn bias_distortion(&self) -> f64 {
        let mut keys: BTreeSet<(usize, usize)> = self.accepted.keys().copied().collect();
        for (i, tos) in &self.out {
            for j in tos.keys() {
                keys.insert((*i, *j));
            }
        }
        let raw_total: f64 = self.accepted.values().map(|v| *v as f64).sum();
        let rw_total: f64 = self.out.values().flat_map(|m| m.values()).sum();
        if raw_total <= 0.0 || rw_total <= 0.0 {
            return f64::NAN;
        }
        let mut tv = 0.0;
        for (i, j) in keys {
            let raw = self.accepted.get(&(i, j)).copied().unwrap_or(0) as f64 / raw_total;
            let rw = self
                .out
                .get(&i)
                .and_then(|m| m.get(&j))
                .copied()
                .unwrap_or(0.0)
                / rw_total;
            tv += (raw - rw).abs();
        }
        0.5 * tv
    }
}

impl HopCounts {
    /// Recorded transitions as `(from, to, reweighted mass)`.
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.out
            .iter()
            .flat_map(|(i, tos)| tos.iter().map(move |(j, w)| (*i, *j, *w)))
    }

    /// Hops the chain spent in a basin.
    pub fn time_of(&self, basin: usize) -> f64 {
        self.time.get(&basin).copied().unwrap_or(0.0)
    }

    /// Mass that left a basin towards a destination never registered.
    pub fn leak_of(&self, basin: usize) -> f64 {
        self.leak.get(&basin).copied().unwrap_or(0.0)
    }

    /// Total hops recorded.
    pub fn total_time(&self) -> f64 {
        self.time.values().sum()
    }
}

// ---------------------------------------------------------------------------
// The jump chain
// ---------------------------------------------------------------------------

/// One outgoing transition: `(target, probability, mass)`.
///
/// The probability is conditional on the jump going to a named basin; the mass
/// is the unnormalised weight the probability came from, which is what an
/// execution-count criterion needs.
pub type Edge = (usize, f64, f64);

/// The observed chain with returns eliminated analytically.
///
/// A Metropolis chain on a cluster landscape returns to where it stands on
/// roughly nineteen proposals in twenty, so the self-transition probability
/// dominates the diagonal and would dominate the conditioning for a reason that
/// has nothing to do with trapping. Eliminating it in closed form, which is one
/// step of the same graph transformation used later, leaves a chain whose
/// states are basins, whose steps are genuine transitions, and whose holding
/// time carries the returns as a number of hops. What remains ill conditioned
/// after that is the superbasin itself.
#[derive(Debug, Clone)]
pub struct JumpChain {
    /// State identifiers, ascending.
    nodes: Vec<usize>,
    /// Position of each identifier in `nodes`.
    index: BTreeMap<usize, usize>,
    /// Outgoing transitions, by local source index.
    adj: Vec<Vec<Edge>>,
    /// Probability that a jump out of a state goes to a basin with no
    /// identifier.
    leak: Vec<f64>,
    /// Expected hops spent in a state per jump out of it.
    hold: Vec<f64>,
    /// Total hops observed in a state.
    time: Vec<f64>,
}

impl JumpChain {
    /// Builds the jump chain from observed counts.
    pub fn from_counts(counts: &HopCounts) -> Self {
        let nodes: Vec<usize> = counts.nodes().into_iter().collect();
        let index: BTreeMap<usize, usize> =
            nodes.iter().enumerate().map(|(i, b)| (*b, i)).collect();
        let mut adj = vec![Vec::new(); nodes.len()];
        let mut leak = vec![0.0; nodes.len()];
        let mut hold = vec![1.0; nodes.len()];
        let mut time = vec![0.0; nodes.len()];
        for (i, b) in nodes.iter().enumerate() {
            let named: f64 = counts
                .out
                .get(b)
                .map(|m| m.values().sum::<f64>())
                .unwrap_or(0.0);
            let unnamed = counts.leak.get(b).copied().unwrap_or(0.0);
            let escape = named + unnamed;
            time[i] = counts.time.get(b).copied().unwrap_or(0.0);
            let observed = time[i];
            if escape <= 0.0 {
                // Never observed to leave. It stays a node so that it can be an
                // absorbing target, with the whole of its observed time as a
                // holding time.
                hold[i] = observed.max(1.0);
                continue;
            }
            hold[i] = (observed / escape).max(1.0);
            leak[i] = unnamed / escape;
            if let Some(m) = counts.out.get(b) {
                for (j, w) in m {
                    if let Some(t) = index.get(j) {
                        adj[i].push((*t, w / escape, *w));
                    }
                }
            }
        }
        Self {
            nodes,
            index,
            adj,
            leak,
            hold,
            time,
        }
    }

    /// States in the chain.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when the chain holds no state.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// State identifiers, ascending.
    pub fn nodes(&self) -> &[usize] {
        &self.nodes
    }

    /// Position of an identifier, when it is present.
    pub fn position(&self, id: usize) -> Option<usize> {
        self.index.get(&id).copied()
    }

    /// Outgoing transitions of a state.
    pub fn edges(&self, local: usize) -> &[Edge] {
        &self.adj[local]
    }

    /// Expected hops spent in a state per jump out of it.
    pub fn hold(&self, local: usize) -> f64 {
        self.hold[local]
    }

    /// Hops the chain has spent in a state.
    ///
    /// The residence the trapping test is built on. A superbasin is where the
    /// chain keeps coming back to, and this is that quantity directly, rather
    /// than through any one transition having been repeated.
    pub fn residence(&self, local: usize) -> f64 {
        self.time[local]
    }

    /// States around `local`, ordered by how much time the chain has spent in
    /// them.
    ///
    /// Grown outwards one state at a time, always taking the neighbour of the
    /// current set where the chain has spent the most hops. Two properties
    /// matter. The set stays connected to where the chain is standing, so it
    /// describes the region the chain occupies rather than the busiest states
    /// anywhere. And the ordering is by residence, not by any transition
    /// having been executed repeatedly: measured on a 75-point run at a million
    /// evaluations, the mean reweighted mass on a transition is 2.0 with half
    /// the edges below it, so a criterion built on repeated transitions
    /// describes almost nothing while the chain is demonstrably revisiting
    /// basins 25 times each.
    ///
    /// States never observed to leave are skipped: they cannot be transient,
    /// since a transient state has to reach the boundary, and they make
    /// perfectly good absorbing ones.
    pub fn neighbourhood(&self, local: usize, max_states: usize) -> Vec<usize> {
        if self.adj[local].is_empty() {
            return Vec::new();
        }
        let mut chosen = vec![local];
        let mut inside: BTreeSet<usize> = BTreeSet::new();
        inside.insert(local);
        let mut frontier: BTreeSet<usize> = BTreeSet::new();
        let push = |frontier: &mut BTreeSet<usize>, inside: &BTreeSet<usize>, v: usize| {
            for (t, _, _) in &self.adj[v] {
                if !inside.contains(t) && !self.adj[*t].is_empty() {
                    frontier.insert(*t);
                }
            }
        };
        push(&mut frontier, &inside, local);
        while chosen.len() < max_states && !frontier.is_empty() {
            let next = *frontier
                .iter()
                .max_by(|a, b| {
                    self.time[**a]
                        .partial_cmp(&self.time[**b])
                        .expect("residences are finite")
                        .then(b.cmp(a))
                })
                .expect("frontier is non-empty");
            frontier.remove(&next);
            inside.insert(next);
            chosen.push(next);
            push(&mut frontier, &inside, next);
        }
        chosen
    }

    /// Probability that a jump leaves towards an unnamed basin.
    pub fn leak(&self, local: usize) -> f64 {
        self.leak[local]
    }

    /// Directed transitions recorded.
    pub fn n_edges(&self) -> usize {
        self.adj.iter().map(|a| a.len()).sum()
    }

    /// Strongly connected components of the fast subgraph, largest first.
    ///
    /// An edge is fast when it carries at least `min_probability` of its
    /// state's escape flux and has been executed at least `min_mass` times.
    /// Both thresholds guard a different failure. Without the flux share, one
    /// basin with hundreds of rarely used neighbours pulls the whole graph into
    /// a single component; without the execution count, a transition seen once
    /// counts as a fast process, which is what Chatterjee and Voter's criterion
    /// exists to exclude. A coarse graph carries effective rates rather than
    /// counts, so `min_mass` applies to the observed chain only.
    ///
    /// Strong connectivity is required rather than convenient: the internal
    /// quasi-equilibrium of a set is the stationary vector of the chain
    /// restricted to it, and that vector is unique only on a strongly connected
    /// set. A set that is merely well connected in the undirected sense can
    /// have no such vector at all.
    pub fn components(&self, min_probability: f64, min_mass: f64) -> Vec<Vec<usize>> {
        let n = self.nodes.len();
        let succ: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                self.adj[i]
                    .iter()
                    .filter(|(_, p, m)| *p >= min_probability && *m >= min_mass)
                    .map(|(t, _, _)| *t)
                    .collect()
            })
            .collect();
        // Tarjan, iterative: a run registers thousands of basins and a
        // recursive walk over them overflows the stack.
        let mut idx = vec![usize::MAX; n];
        let mut low = vec![0usize; n];
        let mut on = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        let mut out: Vec<Vec<usize>> = Vec::new();
        let mut counter = 0usize;
        // Each frame is a node and how far through its successors it is.
        let mut work: Vec<(usize, usize)> = Vec::new();
        for root in 0..n {
            if idx[root] != usize::MAX {
                continue;
            }
            idx[root] = counter;
            low[root] = counter;
            counter += 1;
            stack.push(root);
            on[root] = true;
            work.push((root, 0));
            while !work.is_empty() {
                let (v, pos) = *work.last().expect("work is non-empty");
                if pos < succ[v].len() {
                    work.last_mut().expect("work is non-empty").1 += 1;
                    let w = succ[v][pos];
                    if idx[w] == usize::MAX {
                        idx[w] = counter;
                        low[w] = counter;
                        counter += 1;
                        stack.push(w);
                        on[w] = true;
                        work.push((w, 0));
                    } else if on[w] {
                        low[v] = low[v].min(idx[w]);
                    }
                } else {
                    work.pop();
                    if low[v] == idx[v] {
                        let mut comp = Vec::new();
                        while let Some(w) = stack.pop() {
                            on[w] = false;
                            comp.push(w);
                            if w == v {
                                break;
                            }
                        }
                        comp.sort_unstable();
                        out.push(comp);
                    }
                    if let Some((p, _)) = work.last() {
                        let p = *p;
                        low[p] = low[p].min(low[v]);
                    }
                }
            }
        }
        out.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));
        out
    }
}

// ---------------------------------------------------------------------------
// Canonical form and the absorbing-chain solve
// ---------------------------------------------------------------------------

/// A sub-chain in canonical form: transient block `Q`, absorbing block `R`.
///
/// Rows of `[Q R]` sum to one exactly, the unnamed-destination column included,
/// so nothing is normalised away downstream.
#[derive(Debug, Clone)]
pub struct Canonical {
    /// Basin identifiers of the transient states, in row order.
    pub transient: Vec<usize>,
    /// Basin identifiers of the absorbing states, in column order.
    /// [`UNKNOWN`] stands for destinations the run never registered.
    pub absorbing: Vec<usize>,
    /// Transient-to-transient probabilities, sparse by row.
    q: Vec<Vec<(usize, f64)>>,
    /// Transient-to-absorbing probabilities, sparse by row.
    r: Vec<Vec<(usize, f64)>>,
    /// Expected hops per jump for each transient state.
    hold: Vec<f64>,
    /// Mean share of a transient state's escape mass with no named
    /// destination.
    pub unknown_mass: f64,
}

/// Why a canonical form could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalError {
    /// Fewer transient states than a superbasin needs.
    TooSmall(usize),
    /// Transient states from which no absorbing state is reachable, which
    /// leaves `I - Q` singular.
    Closed(usize),
    /// No absorbing state at all: the set is the whole observed graph.
    NoBoundary,
}

impl std::fmt::Display for CanonicalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CanonicalError::TooSmall(n) => {
                write!(f, "a superbasin needs at least 2 states, got {n}")
            }
            CanonicalError::Closed(n) => write!(
                f,
                "{n} transient states reach no boundary; I - Q would be singular there"
            ),
            CanonicalError::NoBoundary => {
                write!(f, "the transient set has no observed boundary")
            }
        }
    }
}

impl std::error::Error for CanonicalError {}

/// What the absorbing chain says about one starting state.
#[derive(Debug, Clone)]
pub struct Absorption {
    /// Probability of leaving through each absorbing state, aligned to
    /// [`Canonical::absorbing`].
    pub exit: Vec<f64>,
    /// Expected hops before absorption, which is the search cost the jump
    /// replaces.
    pub hops: f64,
    /// Expected jumps before absorption, which is what the trapping test uses.
    pub jumps: f64,
    /// Largest update of the last sweep, when the sparse solve produced this.
    pub residual: f64,
    /// Whether node elimination produced this, which carries no residual.
    pub exact: bool,
}

impl Canonical {
    /// Canonical form for the states at local positions `set` inside `chain`.
    ///
    /// Everything reachable from the set in one jump and outside it becomes
    /// absorbing. The unnamed-destination column is added whenever any
    /// transient state leaks into it.
    pub fn new(chain: &JumpChain, set: &[usize]) -> Result<Self, CanonicalError> {
        if set.len() < 2 {
            return Err(CanonicalError::TooSmall(set.len()));
        }
        let inside: BTreeSet<usize> = set.iter().copied().collect();
        let local: BTreeMap<usize, usize> = set.iter().enumerate().map(|(i, s)| (*s, i)).collect();
        let mut boundary: BTreeSet<usize> = BTreeSet::new();
        let mut leaks = false;
        for s in set {
            for (t, _, _) in chain.edges(*s) {
                if !inside.contains(t) {
                    boundary.insert(*t);
                }
            }
            if chain.leak(*s) > 0.0 {
                leaks = true;
            }
        }
        if boundary.is_empty() && !leaks {
            return Err(CanonicalError::NoBoundary);
        }
        let mut absorbing: Vec<usize> = boundary.iter().map(|b| chain.nodes[*b]).collect();
        let boundary_local: BTreeMap<usize, usize> =
            boundary.iter().enumerate().map(|(i, b)| (*b, i)).collect();
        let unknown_col = if leaks {
            absorbing.push(UNKNOWN);
            Some(absorbing.len() - 1)
        } else {
            None
        };

        let mut q = vec![Vec::new(); set.len()];
        let mut r = vec![Vec::new(); set.len()];
        let mut hold = vec![1.0; set.len()];
        let mut unknown_mass = 0.0;
        for (i, s) in set.iter().enumerate() {
            hold[i] = chain.hold(*s);
            for (t, p, _) in chain.edges(*s) {
                match local.get(t) {
                    Some(j) => q[i].push((*j, *p)),
                    None => r[i].push((boundary_local[t], *p)),
                }
            }
            if let Some(c) = unknown_col {
                let l = chain.leak(*s);
                if l > 0.0 {
                    r[i].push((c, l));
                    unknown_mass += l;
                }
            }
        }
        unknown_mass /= set.len() as f64;

        // Every transient state must be able to reach the boundary or the
        // system is singular. Reverse reachability from the rows that already
        // touch R, which is a sweep rather than a solve.
        let mut reaches = vec![false; set.len()];
        let mut frontier: Vec<usize> = (0..set.len()).filter(|i| !r[*i].is_empty()).collect();
        for i in &frontier {
            reaches[*i] = true;
        }
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); set.len()];
        for (i, row) in q.iter().enumerate() {
            for (j, _) in row {
                preds[*j].push(i);
            }
        }
        while let Some(v) = frontier.pop() {
            for p in &preds[v] {
                if !reaches[*p] {
                    reaches[*p] = true;
                    frontier.push(*p);
                }
            }
        }
        let closed = reaches.iter().filter(|r| !**r).count();
        if closed > 0 {
            return Err(CanonicalError::Closed(closed));
        }

        Ok(Self {
            transient: set.iter().map(|s| chain.nodes[*s]).collect(),
            absorbing,
            q,
            r,
            hold,
            unknown_mass,
        })
    }

    /// Transient states.
    pub fn n_transient(&self) -> usize {
        self.transient.len()
    }

    /// Absorbing states, the unnamed column included when present.
    pub fn n_absorbing(&self) -> usize {
        self.absorbing.len()
    }

    /// Exit distribution and expected time to absorption, from `source`.
    ///
    /// Node elimination, not a linear solve, and not an inverse. States other
    /// than the source are removed one at a time with
    ///
    /// ```text
    ///   P'_ij  = P_ij + P_ix P_xj / (1 - P_xx)
    ///   tau'_i = tau_i + P_ix tau_x / (1 - P_xx)
    /// ```
    ///
    /// and `1 - P_xx` is evaluated as the sum of the remaining outgoing
    /// probabilities of `x` rather than by subtraction, so the elimination
    /// touches no cancelling difference at any point. All quantities stay
    /// nonnegative, which is why this stays accurate where a residual-based
    /// solve on the same matrix loses digits.
    ///
    /// The elimination order is by ascending fill-in, `|preds| * |succs|`,
    /// recomputed as the graph changes: the graphs a search produces are
    /// locally tree-like, where a bad order fills in far more than a good one.
    pub fn absorb(&self, source: usize) -> Absorption {
        let n = self.transient.len();
        let m = self.absorbing.len();
        let mut tq: Vec<BTreeMap<usize, f64>> = self
            .q
            .iter()
            .map(|row| {
                let mut acc = BTreeMap::new();
                for (j, p) in row {
                    *acc.entry(*j).or_insert(0.0) += *p;
                }
                acc
            })
            .collect();
        let mut tr: Vec<Vec<f64>> = vec![vec![0.0; m]; n];
        for (i, row) in self.r.iter().enumerate() {
            for (j, p) in row {
                tr[i][*j] += *p;
            }
        }
        // Waiting time carried on each state, in hops and in jumps.
        let mut tau_hops: Vec<f64> = self.hold.clone();
        let mut tau_jumps: Vec<f64> = vec![1.0; n];
        let mut preds: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for (i, row) in tq.iter().enumerate() {
            for j in row.keys() {
                preds[*j].insert(i);
            }
        }
        let mut alive: BTreeSet<usize> = (0..n).collect();
        alive.remove(&source);

        while !alive.is_empty() {
            let x = *alive
                .iter()
                .min_by_key(|i| preds[**i].len().saturating_mul(tq[**i].len()))
                .expect("alive is non-empty");
            alive.remove(&x);
            tq[x].remove(&x);
            preds[x].remove(&x);
            // 1 - P_xx as the sum of what is left, never as a subtraction.
            let stay: f64 = tq[x].values().sum::<f64>() + tr[x].iter().sum::<f64>();
            if !(stay > 0.0) {
                // Nothing leaves x, so no path through it absorbs. Drop the
                // edges into it rather than dividing by zero.
                for p in std::mem::take(&mut preds[x]) {
                    tq[p].remove(&x);
                }
                continue;
            }
            let inv = 1.0 / stay;
            let out_t: Vec<(usize, f64)> = tq[x].iter().map(|(j, p)| (*j, *p)).collect();
            let out_a: Vec<(usize, f64)> = tr[x]
                .iter()
                .enumerate()
                .filter(|(_, p)| **p > 0.0)
                .map(|(j, p)| (j, *p))
                .collect();
            let tau_x_hops = tau_hops[x];
            let tau_x_jumps = tau_jumps[x];
            for p in std::mem::take(&mut preds[x]) {
                let pix = match tq[p].remove(&x) {
                    Some(v) => v,
                    None => continue,
                };
                let scale = pix * inv;
                for (j, pxj) in &out_t {
                    *tq[p].entry(*j).or_insert(0.0) += scale * pxj;
                    preds[*j].insert(p);
                }
                for (j, pxj) in &out_a {
                    tr[p][*j] += scale * pxj;
                }
                tau_hops[p] += scale * tau_x_hops;
                tau_jumps[p] += scale * tau_x_jumps;
            }
            tq[x].clear();
            tr[x].iter_mut().for_each(|v| *v = 0.0);
        }

        // The source's own self-loop is what remains of every excursion that
        // returned; dividing it out is the last elimination step.
        tq[source].remove(&source);
        let stay: f64 = tq[source].values().sum::<f64>() + tr[source].iter().sum::<f64>();
        let inv = if stay > 0.0 { 1.0 / stay } else { 0.0 };
        let exit: Vec<f64> = tr[source].iter().map(|p| p * inv).collect();
        Absorption {
            exit,
            hops: tau_hops[source] * inv,
            jumps: tau_jumps[source] * inv,
            residual: 0.0,
            exact: true,
        }
    }

    /// The same answer by a sparse solve, for sets too large to eliminate.
    ///
    /// Elimination is cubic in the worst case, and the region a run cycles in
    /// is the size of its visited set: 1200 basins at 75 points on a million
    /// evaluations, 11366 at 98 points on twelve million. Node elimination
    /// cannot be asked for that, and the row of `N` it would produce is
    /// available from one sparse solve.
    ///
    /// Row `i` of `N` is `y` with `(I - Q)^T y = e_i`, verified symbolically,
    /// and then `B_i = y^T R` and `t_i = y^T 1`. Gauss-Seidel on that system
    /// from `y = 0` is monotone increasing, because `Q^T >= 0` and the
    /// right-hand side is nonnegative, so a truncated run underestimates every
    /// quantity it produces: the exit distribution is renormalised, the escape
    /// time is a lower bound, and the trapping test built on it errs towards
    /// refusing. That one-sidedness is why this is safe to truncate and the
    /// residual is reported rather than assumed small.
    ///
    /// Convergence stalls in proportion to the trapping, which is the same
    /// statement as the condition number identity.
    pub fn absorb_sparse(&self, source: usize, max_sweeps: usize, tol: f64) -> Absorption {
        let n = self.transient.len();
        // Rows of Q^T: which transient states lead into each one.
        let mut qt: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for (i, row) in self.q.iter().enumerate() {
            for (j, p) in row {
                qt[*j].push((i, *p));
            }
        }
        let mut y = vec![0.0; n];
        let mut residual = f64::INFINITY;
        for _ in 0..max_sweeps {
            let mut delta: f64 = 0.0;
            for i in 0..n {
                let mut acc = if i == source { 1.0 } else { 0.0 };
                for (j, p) in &qt[i] {
                    acc += p * y[*j];
                }
                delta = delta.max((acc - y[i]).abs());
                y[i] = acc;
            }
            residual = delta;
            if delta <= tol {
                break;
            }
        }
        let mut exit = vec![0.0; self.absorbing.len()];
        for (i, row) in self.r.iter().enumerate() {
            for (c, p) in row {
                exit[*c] += y[i] * p;
            }
        }
        let mass: f64 = exit.iter().sum();
        if mass > 0.0 {
            for v in exit.iter_mut() {
                *v /= mass;
            }
        }
        Absorption {
            exit,
            hops: y.iter().zip(self.hold.iter()).map(|(a, b)| a * b).sum(),
            jumps: y.iter().sum(),
            residual,
            exact: false,
        }
    }

    /// Exit distribution and escape time by whichever solver the size calls
    /// for.
    pub fn absorb_at_scale(
        &self,
        source: usize,
        elimination_cap: usize,
        max_sweeps: usize,
        tol: f64,
    ) -> Absorption {
        if self.transient.len() <= elimination_cap {
            self.absorb(source)
        } else {
            self.absorb_sparse(source, max_sweeps, tol)
        }
    }

    /// Expected jumps to absorption from every transient state, by monotone
    /// Gauss-Seidel on `(I - Q) u = 1`.
    ///
    /// Returned with the largest update of the last sweep and the sweeps taken.
    /// `Q >= 0` and the diagonal of the jump chain is zero, so iterating
    /// `u_i <- 1 + sum_j Q_ij u_j` from `u = 0` increases monotonically towards
    /// the solution and can only underestimate it. That makes the reported
    /// residual one-sided rather than hopeful, and makes this the honest
    /// instrument for how hard the system is: convergence stalls in proportion
    /// to the trapping.
    ///
    /// The escape move does not use this. It uses [`Canonical::absorb`], which
    /// is exact.
    pub fn expected_jumps(&self, max_sweeps: usize, tol: f64) -> (Vec<f64>, f64, usize) {
        let n = self.transient.len();
        let mut u = vec![0.0; n];
        let mut sweeps = 0;
        let mut residual = f64::INFINITY;
        for s in 0..max_sweeps {
            let mut delta: f64 = 0.0;
            for i in 0..n {
                let mut acc = 1.0;
                for (j, p) in &self.q[i] {
                    acc += p * u[*j];
                }
                delta = delta.max((acc - u[i]).abs());
                u[i] = acc;
            }
            sweeps = s + 1;
            residual = delta;
            if delta <= tol {
                break;
            }
        }
        (u, residual, sweeps)
    }

    /// Infinity-norm condition number of `I - Q`.
    ///
    /// `N = (I - Q)^-1 >= 0` elementwise, so `||N||_inf = max_i (N 1)_i`, which
    /// is the expected-jumps vector. The identity is exact, not a bound, and it
    /// says the conditioning of the escape problem *is* the trapping it
    /// describes.
    ///
    /// Returned with the Gauss-Seidel residual, since an unconverged `u`
    /// underestimates the condition number and the reader should be told by how
    /// much.
    pub fn condition_inf(&self, max_sweeps: usize, tol: f64) -> (f64, f64) {
        let (u, residual, _) = self.expected_jumps(max_sweeps, tol);
        let row = self
            .q
            .iter()
            .map(|r| r.iter().map(|(_, p)| p.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let umax = u.iter().copied().fold(0.0_f64, f64::max);
        ((1.0 + row) * umax, residual)
    }
}

// ---------------------------------------------------------------------------
// Lumping
// ---------------------------------------------------------------------------

/// Thresholds for deciding whether a set of states is one coarse state.
#[derive(Debug, Clone)]
pub struct LumpParams {
    /// Smallest share of a state's escape flux an edge may carry and still be
    /// considered at all.
    ///
    /// A floor on the threshold ladder rather than the criterion itself. One
    /// fixed flux share cannot be the criterion: on a nested landscape the
    /// value that separates a tight group from its neighbour merges the two
    /// levels above it, which is measurable as a hierarchy that stops one level
    /// short.
    pub min_probability: f64,
    /// Flux-share thresholds tried between the strongest edge and the floor.
    ///
    /// Components merge monotonically as the threshold falls, so scanning from
    /// strict to loose and keeping the largest set that still passes the
    /// separation test finds the coarsest lumpable set without a threshold
    /// having to be chosen.
    pub threshold_ladder: usize,
    /// Largest set the equilibration test is run on.
    ///
    /// Power iteration costs a sweep over the component's edges, and a
    /// component covering a whole run's graph is not going to be in
    /// quasi-equilibrium anyway.
    pub max_lump_states: usize,
    /// Times a transition must have been executed to count as fast.
    ///
    /// Chatterjee and Voter identify the superbasin as the subnetwork of
    /// processes executed often enough to be fast. Applied to the observed
    /// chain only: a coarse graph carries effective rates, not counts.
    pub min_executions: f64,
    /// Smallest set worth replacing by one coarse state.
    pub min_states: usize,
    /// Timescale separation a set must clear before it is lumped.
    ///
    /// Escape time over internal equilibration time, both in jumps. The lumping
    /// error is `O(1 / separation)`; ten is one decade, which puts the induced
    /// error in the exit distribution near a per cent, against exit
    /// probabilities that differ between funnels by far more.
    pub min_separation: f64,
    /// Jumps allowed for the internal distribution to equilibrate before the
    /// set is declared not lumpable.
    pub equilibration_cap: usize,
    /// Power iterations allowed for the internal stationary vector.
    pub occupancy_sweeps: usize,
    /// Levels the recursion may build.
    pub max_levels: usize,
}

impl Default for LumpParams {
    fn default() -> Self {
        Self {
            min_probability: 1e-6,
            threshold_ladder: 12,
            max_lump_states: 8192,
            min_executions: 2.0,
            min_states: 2,
            min_separation: 10.0,
            equilibration_cap: 256,
            occupancy_sweeps: 1024,
            max_levels: 8,
        }
    }
}

/// One coarse state: a set of finer states in quasi-equilibrium.
#[derive(Debug, Clone)]
pub struct Lump {
    /// Identifiers of the member states at the level below.
    pub states: Vec<usize>,
    /// Internal quasi-equilibrium occupancy, aligned to `states`.
    pub occupancy: Vec<f64>,
    /// Expected jumps inside the set before it is left.
    pub escape_jumps: f64,
    /// Jumps for the internal distribution to reach quasi-equilibrium.
    pub relax_jumps: f64,
    /// Expected hops inside the set before it is left, which is the cost the
    /// search would otherwise pay to cross it.
    pub escape_hops: f64,
    /// Timescale separation, escape over equilibration.
    pub separation: f64,
}

/// One level of the coarse-graining hierarchy.
#[derive(Debug, Clone)]
pub struct Level {
    /// The coarse states formed at this level.
    pub lumps: Vec<Lump>,
    /// Which coarse state each state of the level below belongs to.
    pub of: BTreeMap<usize, usize>,
    /// The chain over the coarse states.
    pub chain: JumpChain,
    /// Coarse-state identifiers, lumps first and singletons after.
    pub ids: Vec<usize>,
}

/// The coarse-graining hierarchy of the observed transition graph.
///
/// Level zero is the basins themselves. Each further level lumps the one below.
/// Reported whether or not the escape move is used, since the graph is recorded
/// anyway and the hierarchy is direct evidence about whether the funnel
/// structure is recoverable from transitions at all. Four descriptors have
/// failed to recover it on this landscape: a fourth-order bond-order parameter
/// separating the two 75-point funnels by 0.023, sorted distances needing a
/// merge radius on a knife edge, a spectral partition of the same graph, and a
/// tensor invariant.
#[derive(Debug, Clone)]
pub struct Hierarchy {
    /// The chain over basins.
    pub base: JumpChain,
    /// Coarse-graining levels, finest first.
    pub levels: Vec<Level>,
}

/// Internal quasi-equilibrium of a set, and the jumps needed to reach it.
///
/// The occupancy is the stationary vector of the lazy chain restricted to the
/// set and renormalised. Laziness costs a factor of two on the relaxation time
/// and buys aperiodicity, without which power iteration on a set whose
/// transitions alternate does not converge at all.
///
/// The relaxation time is measured rather than read off an eigenvalue: the
/// chain is not reversible, so its second eigenvalue is complex in general and
/// its modulus is not the quantity that decides whether quasi-equilibrium is
/// reached. What decides that is how many jumps the internal distribution needs
/// to come within `1/e` of the stationary vector in total variation, which is
/// what is measured here. `None` means it did not, within the cap, which is a
/// refusal to lump rather than a failure.
fn quasi_equilibrium(
    chain: &JumpChain,
    set: &[usize],
    params: &LumpParams,
) -> Option<(Vec<f64>, f64)> {
    let n = set.len();
    if n < 2 {
        return None;
    }
    let local: BTreeMap<usize, usize> = set.iter().enumerate().map(|(i, s)| (*s, i)).collect();
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (i, s) in set.iter().enumerate() {
        let mut acc = Vec::new();
        let mut total = 0.0;
        for (t, p, _) in chain.edges(*s) {
            if let Some(j) = local.get(t) {
                acc.push((*j, *p));
                total += *p;
            }
        }
        if total <= 0.0 {
            return None;
        }
        for (_, p) in acc.iter_mut() {
            *p /= total;
        }
        rows[i] = acc;
    }
    let step = |p: &[f64]| -> Vec<f64> {
        let mut q = vec![0.0; n];
        for (i, pi) in p.iter().enumerate() {
            // Lazy: half the mass stays, which makes the chain aperiodic
            // without moving its stationary vector.
            q[i] += 0.5 * pi;
            for (j, w) in &rows[i] {
                q[*j] += 0.5 * pi * w;
            }
        }
        q
    };
    let mut pi = vec![1.0 / n as f64; n];
    for _ in 0..params.occupancy_sweeps {
        let next = step(&pi);
        let delta: f64 = next
            .iter()
            .zip(pi.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            * 0.5;
        pi = next;
        if delta < 1e-13 {
            break;
        }
    }
    let total: f64 = pi.iter().sum();
    if !(total > 0.0) {
        return None;
    }
    for v in pi.iter_mut() {
        *v /= total;
    }
    // Equilibration from the worst start available, the state carrying the most
    // occupancy, so the reported time is not flattered by starting near pi.
    let start = pi
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("occupancies are finite"))
        .map(|(i, _)| i)?;
    let mut p = vec![0.0; n];
    p[start] = 1.0;
    let mut relax = f64::INFINITY;
    for k in 1..=params.equilibration_cap {
        p = step(&p);
        let tv: f64 = p
            .iter()
            .zip(pi.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            * 0.5;
        if tv < std::f64::consts::E.recip() {
            // Laziness doubled the time; report the underlying chain's.
            relax = 0.5 * k as f64;
            break;
        }
    }
    if !relax.is_finite() {
        return None;
    }
    Some((pi, relax.max(1.0)))
}

/// Escape rate per jump and mean hops per jump for a set, under its
/// quasi-equilibrium occupancy.
fn escape_rate(chain: &JumpChain, set: &[usize], occupancy: &[f64]) -> (f64, f64) {
    let inside: BTreeSet<usize> = set.iter().copied().collect();
    let mut k_esc = 0.0;
    let mut hops_per_jump = 0.0;
    for (i, s) in set.iter().enumerate() {
        let out: f64 = chain
            .edges(*s)
            .iter()
            .filter(|(t, _, _)| !inside.contains(t))
            .map(|(_, p, _)| *p)
            .sum::<f64>()
            + chain.leak(*s);
        k_esc += occupancy[i] * out;
        hops_per_jump += occupancy[i] * chain.hold(*s);
    }
    (k_esc, hops_per_jump)
}

/// Flux-share thresholds to try, strictest first.
///
/// Quantiles of the edge-probability distribution rather than a geometric
/// ladder, so the thresholds land where the edges actually are.
fn threshold_ladder(chain: &JumpChain, params: &LumpParams, min_mass: f64) -> Vec<f64> {
    let mut probs: Vec<f64> = Vec::new();
    for i in 0..chain.len() {
        for (_, p, m) in chain.edges(i) {
            if *m >= min_mass && *p >= params.min_probability {
                probs.push(*p);
            }
        }
    }
    if probs.is_empty() {
        return Vec::new();
    }
    probs.sort_by(|a, b| a.partial_cmp(b).expect("probabilities are finite"));
    let k = params.threshold_ladder.max(1);
    let mut out: Vec<f64> = Vec::with_capacity(k);
    for q in 0..k {
        // Descending: the strictest threshold first, so components merge as the
        // scan proceeds and a superset always arrives after its subsets. The
        // order is load bearing, not cosmetic: reversed, a subset overwrites
        // the coarser set that contains it and the hierarchy loses a level.
        let idx = (q * (probs.len() - 1)) / (k - 1).max(1);
        let t = probs[probs.len() - 1 - idx];
        if out
            .last()
            .map(|l: &f64| (*l - t).abs() > 1e-15)
            .unwrap_or(true)
        {
            out.push(t);
        }
    }
    out
}

/// The set is trapping: it equilibrates internally well before it is left.
///
/// Returns the occupancy, the escape rate and the separation when it passes.
fn separation_of(
    chain: &JumpChain,
    comp: &[usize],
    params: &LumpParams,
) -> Option<(Vec<f64>, f64, f64, f64)> {
    let (pi, relax) = quasi_equilibrium(chain, comp, params)?;
    let (k_esc, hops_per_jump) = escape_rate(chain, comp, &pi);
    if !(k_esc > 0.0) {
        return None;
    }
    let escape_jumps = 1.0 / k_esc;
    let separation = escape_jumps / relax;
    if separation < params.min_separation {
        return None;
    }
    Some((pi, escape_jumps, hops_per_jump, separation))
}

/// Builds one coarse level over `chain`, or `None` when nothing is lumpable.
///
/// Candidates are the strongly connected components of the fast subgraph, taken
/// over a ladder of flux-share thresholds rather than over one. Each is
/// accepted only when its escape time clears its equilibration time by
/// [`LumpParams::min_separation`], which is the assumption lumping makes stated
/// as a test: a set that is well connected but not equilibrated is not
/// lumpable, and this rejects it. Because components only merge as the
/// threshold falls, a set accepted later contains any accepted earlier, so
/// overwriting leaves the coarsest lumpable sets.
pub fn lump_once(chain: &JumpChain, params: &LumpParams, min_mass: f64) -> Option<Level> {
    let mut lumps: Vec<Lump> = Vec::new();
    let mut owner: Vec<Option<usize>> = vec![None; chain.len()];
    let mut tested: BTreeSet<(usize, usize)> = BTreeSet::new();
    for tau in threshold_ladder(chain, params, min_mass) {
        for comp in chain.components(tau, min_mass) {
            if comp.len() < params.min_states.max(2) || comp.len() > params.max_lump_states {
                continue;
            }
            // Components repeat across thresholds; each shape is judged once.
            if !tested.insert((comp[0], comp.len())) {
                continue;
            }
            let (pi, escape_jumps, hops_per_jump, separation) =
                match separation_of(chain, &comp, params) {
                    Some(v) => v,
                    None => continue,
                };
            let idx = lumps.len();
            for s in &comp {
                owner[*s] = Some(idx);
            }
            lumps.push(Lump {
                states: comp.iter().map(|s| chain.nodes[*s]).collect(),
                occupancy: pi,
                escape_jumps,
                relax_jumps: escape_jumps / separation,
                escape_hops: hops_per_jump * escape_jumps,
                separation,
            });
        }
    }
    // Sets fully covered by a later, larger one own nothing and are dropped.
    let mut keep: Vec<usize> = Vec::new();
    let mut renumber: BTreeMap<usize, usize> = BTreeMap::new();
    for o in owner.iter().flatten() {
        if !renumber.contains_key(o) {
            renumber.insert(*o, keep.len());
            keep.push(*o);
        }
    }
    let lumps: Vec<Lump> = keep.iter().map(|k| lumps[*k].clone()).collect();
    let mut of: BTreeMap<usize, usize> = BTreeMap::new();
    for (s, o) in owner.iter().enumerate() {
        if let Some(o) = o {
            of.insert(chain.nodes[s], renumber[o]);
        }
    }
    if lumps.is_empty() {
        return None;
    }
    // States that joined no lump stay as themselves, so the coarse chain covers
    // the same graph rather than a subset of it.
    let mut ids: Vec<usize> = (0..lumps.len()).collect();
    let mut coarse_of_local: Vec<usize> = vec![usize::MAX; chain.len()];
    for (k, l) in lumps.iter().enumerate() {
        for s in &l.states {
            if let Some(p) = chain.position(*s) {
                coarse_of_local[p] = k;
            }
        }
    }
    let mut next = lumps.len();
    for p in 0..chain.len() {
        if coarse_of_local[p] == usize::MAX {
            coarse_of_local[p] = next;
            ids.push(next);
            of.insert(chain.nodes[p], next);
            next += 1;
        }
    }
    if next < 2 {
        // One coarse state covering everything says nothing and cannot be
        // recursed on.
        return None;
    }

    // Mean rate method: k_eff(S -> j) = sum_i pi_i^S P_ij, with the internal
    // transitions dropped because they are what the coarse state absorbs.
    let mut counts = HopCounts::new();
    for (k, l) in lumps.iter().enumerate() {
        for (i, s) in l.states.iter().enumerate() {
            let p = match chain.position(*s) {
                Some(p) => p,
                None => continue,
            };
            for (t, w, _) in chain.edges(p) {
                let c = coarse_of_local[*t];
                if c == k {
                    continue;
                }
                counts.observe_weighted(k, Some(c), l.occupancy[i] * w);
            }
            let leak = chain.leak(p);
            if leak > 0.0 {
                counts.observe_weighted(k, None, l.occupancy[i] * leak);
            }
        }
        // The coarse state holds the chain for the escape time of the set, which
        // is what the mean rate method says a lumped set does.
        counts.add_time(k, l.escape_hops);
    }
    for p in 0..chain.len() {
        let c = coarse_of_local[p];
        if c < lumps.len() {
            continue;
        }
        for (t, w, _) in chain.edges(p) {
            let d = coarse_of_local[*t];
            if d == c {
                continue;
            }
            counts.observe_weighted(c, Some(d), *w);
        }
        let leak = chain.leak(p);
        if leak > 0.0 {
            counts.observe_weighted(c, None, leak);
        }
        counts.add_time(c, chain.hold(p));
    }

    Some(Level {
        lumps,
        of,
        chain: JumpChain::from_counts(&counts),
        ids,
    })
}

impl Hierarchy {
    /// Coarse-grains the observed graph until nothing further lumps.
    pub fn build(counts: &HopCounts, params: &LumpParams) -> Self {
        let base = JumpChain::from_counts(counts);
        let mut levels: Vec<Level> = Vec::new();
        let mut current = base.clone();
        for level in 0..params.max_levels {
            // The execution-count criterion applies to observed counts only.
            let min_mass = if level == 0 {
                params.min_executions
            } else {
                0.0
            };
            match lump_once(&current, params, min_mass) {
                Some(level) => {
                    current = level.chain.clone();
                    levels.push(level);
                }
                None => break,
            }
        }
        Self { base, levels }
    }

    /// Levels above the basins themselves.
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// The coarse state a basin belongs to at `level`, counting from one.
    ///
    /// `None` when the basin was never seen, or when the level does not exist.
    pub fn state_at(&self, level: usize, basin: usize) -> Option<usize> {
        let mut id = basin;
        for l in self.levels.iter().take(level) {
            id = *l.of.get(&id)?;
        }
        Some(id)
    }

    /// Basins under each coarse state at the top level, largest first.
    ///
    /// This is the funnel partition the transitions imply, if they imply one.
    pub fn top_partition(&self) -> Vec<Vec<usize>> {
        let depth = self.levels.len();
        if depth == 0 {
            return Vec::new();
        }
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for b in self.base.nodes() {
            if let Some(top) = self.state_at(depth, *b) {
                groups.entry(top).or_default().push(*b);
            }
        }
        let mut out: Vec<Vec<usize>> = groups.into_values().collect();
        out.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));
        out
    }
}

// ---------------------------------------------------------------------------
// The escape move
// ---------------------------------------------------------------------------

/// Why an escape was refused.
///
/// Refusal is the common case and is not a failure. Jumping out of a region the
/// chain has not finished exploring is worse than staying, so every condition
/// the algebra needs is checked before a jump is offered.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// The observed graph is too small to show a superbasin.
    TooFewNodes(usize),
    /// The chain is standing somewhere the graph does not cover.
    Unseen(usize),
    /// No set around the chain is trapping: the chain revisits its states this
    /// many times on the way out, against the threshold it has to clear. A jump
    /// here would push the chain out of a region it is still exploring.
    WellMixed(f64),
    /// A superbasin was identified but has no observed boundary.
    NoBoundary,
    /// The exits are all to basins with no stored structure, so there is
    /// nothing to land on.
    NoArchivedExit,
    /// The sampled exit has no usable landing structure. Its probability
    /// remains assigned to local exploration rather than to a known exit.
    UnresolvedExit(f64),
    /// The transient set is larger than the exact elimination is allowed to be.
    TooLarge(usize),
    /// The canonical form could not be built.
    Closed(CanonicalError),
}

impl Refusal {
    /// Position of this kind in the refusal breakdown.
    pub fn kind(&self) -> usize {
        match self {
            Refusal::TooFewNodes(_) => 0,
            Refusal::Unseen(_) => 1,
            Refusal::WellMixed(_) => 2,
            Refusal::NoBoundary => 3,
            Refusal::NoArchivedExit => 4,
            Refusal::TooLarge(_) => 5,
            Refusal::Closed(_) => 6,
            Refusal::UnresolvedExit(_) => 7,
        }
    }

    /// Names of the refusal kinds, in breakdown order.
    pub const KINDS: [&'static str; 8] = [
        "small",
        "unseen",
        "mixed",
        "no-boundary",
        "no-exit",
        "too-large",
        "closed",
        "unresolved-exit",
    ];
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::TooFewNodes(n) => write!(f, "{n} basins is too few to identify a superbasin"),
            Refusal::Unseen(b) => write!(f, "basin {b} is not in the transition graph"),
            Refusal::WellMixed(r) => write!(f, "no trapping set: {r:.2} revisits per state"),
            Refusal::NoBoundary => write!(f, "the superbasin has no observed boundary"),
            Refusal::NoArchivedExit => write!(f, "no exit basin has a stored structure"),
            Refusal::UnresolvedExit(p) => write!(f, "exit mass {p:.6} requires local exploration"),
            Refusal::TooLarge(n) => write!(f, "{n} transient states exceeds the elimination cap"),
            Refusal::Closed(e) => write!(f, "{e}"),
        }
    }
}

/// A jump the absorbing chain proposes.
#[derive(Debug, Clone)]
pub struct Jump {
    /// Basin jumped to.
    pub basin: usize,
    /// Quenched energy of the stored structure there.
    pub energy: f64,
    /// The structure itself.
    pub state: Array1<f64>,
    /// Hierarchy level the exit was computed at; zero is the basin graph.
    pub level: usize,
    /// Unconditional probability the absorbing chain gives this exit.
    pub probability: f64,
    /// Expected hops the chain would have spent leaving by diffusion.
    pub expected_hops: f64,
    /// Transient states in the superbasin solved.
    pub transient: usize,
    /// Absorbing states on its boundary.
    pub absorbing: usize,
    /// Infinity-norm condition number of `I - Q` for that solve.
    pub condition: f64,
    /// Residual of the Gauss-Seidel run that produced the condition number.
    pub condition_residual: f64,
    /// Residual of the solve that produced the exit distribution, zero when
    /// node elimination produced it.
    pub solve_residual: f64,
    /// Whether the exit came from the exact elimination.
    pub exact: bool,
    /// Legacy diagnostic: mass discarded by conditioning. Sampling retains
    /// unresolved mass as a refusal, so this value is zero.
    pub conditioned_away: f64,
}

/// Running totals for the report.
#[derive(Debug, Clone, Default)]
pub struct EscapeStats {
    /// Jumps taken.
    pub jumps: usize,
    /// Escapes refused.
    pub refusals: usize,
    /// Refusals by kind, indexed by [`Refusal::kind`].
    pub refusals_by_kind: [usize; 8],
    /// Largest revisits-per-state ratio a well-mixed refusal reported.
    ///
    /// The scientific content of a negative result: if the observed graph never
    /// reaches the trapping threshold, the chain is not cycling inside a
    /// superbasin at the resolution basin identity provides, and no exit
    /// algebra applies however good the algebra is.
    pub mixed_ratio_max: f64,
    /// Largest condition number seen.
    pub condition_max: f64,
    /// Sum of condition numbers, for the mean.
    condition_sum: f64,
    /// Largest Gauss-Seidel residual behind a reported condition number.
    pub condition_residual_max: f64,
    /// Largest residual of a solve that produced an exit distribution.
    pub solve_residual_max: f64,
    /// Jumps whose exit came from the exact elimination.
    pub exact_solves: usize,
    /// Expected hops the jumps replaced, summed.
    pub hops_saved: f64,
    /// Energy gained by jumps that landed lower than they left.
    pub gain: f64,
    /// Jumps that landed lower than they left.
    pub improvements: usize,
}

impl EscapeStats {
    /// Mean condition number over the jumps taken.
    pub fn condition_mean(&self) -> f64 {
        if self.jumps == 0 {
            f64::NAN
        } else {
            self.condition_sum / self.jumps as f64
        }
    }
}

/// Records transitions and structures, and proposes exits from superbasins.
#[derive(Debug)]
pub struct SuperbasinEscape {
    /// Observed transitions, bias divided out.
    pub counts: HopCounts,
    /// Deepest structure stored per basin.
    store: BTreeMap<usize, (f64, Array1<f64>)>,
    /// Structures held at once.
    pub archive_capacity: usize,
    /// Lumping thresholds.
    pub params: LumpParams,
    /// Basins the graph needs before an escape is considered.
    pub min_nodes: usize,
    /// Transient states an escape may consider.
    ///
    /// The region a run cycles in is the size of its visited set, so this is
    /// thousands rather than tens. Sets above
    /// [`SuperbasinEscape::elimination_cap`] are solved sparsely instead of by
    /// elimination.
    pub max_transient: usize,
    /// Transient states the exact node elimination is allowed to handle.
    ///
    /// Elimination is cubic in the worst case; above this the sparse solve
    /// takes over, which is linear per sweep and one-sided when truncated.
    pub elimination_cap: usize,
    /// Sweeps the sparse solve may take before it reports what it has.
    pub solve_sweeps: usize,
    /// Mean visits per transient state that count as cycling rather than
    /// exploring.
    ///
    /// Expected jumps to absorption divided by the size of the transient set,
    /// which is the mean number of times each state is entered before the set
    /// is left. One means the chain sweeps through and leaves, which is
    /// exploration and the case where jumping would throw away a region still
    /// being mapped. Two means every state is entered twice on average, so the
    /// chain is demonstrably returning.
    ///
    /// Deliberately not [`LumpParams::min_separation`], which answers a
    /// different question. That threshold bounds an approximation error, since
    /// lumping is only valid under quasi-equilibrium and its error is
    /// `O(1 / separation)`. The exit computed here involves no approximation at
    /// any ratio, so the only thing its threshold has to decide is whether the
    /// chain is returning at all.
    pub min_revisits: f64,
    /// Gauss-Seidel sweeps allowed for the condition-number report.
    pub condition_sweeps: usize,
    /// Proposals between rebuilds of the hierarchy.
    ///
    /// Coarse-graining costs a sweep over the graph per threshold, which is
    /// cheap against a hop but not against nothing. A hierarchy is a statement
    /// about the shape of a run, and that shape does not change in a hundred
    /// hops.
    pub rebuild_every: u64,
    /// The hierarchy, and the observation count it was built at.
    cache: Option<(u64, Hierarchy)>,
    /// Totals for the report.
    pub stats: EscapeStats,
}

impl Default for SuperbasinEscape {
    fn default() -> Self {
        Self::new()
    }
}

impl SuperbasinEscape {
    /// A recorder with no observations.
    pub fn new() -> Self {
        Self {
            counts: HopCounts::new(),
            store: BTreeMap::new(),
            archive_capacity: 4096,
            params: LumpParams::default(),
            min_nodes: 24,
            max_transient: 4096,
            elimination_cap: 384,
            solve_sweeps: 4096,
            min_revisits: 2.0,
            condition_sweeps: 4096,
            rebuild_every: 20_000,
            cache: None,
            stats: EscapeStats::default(),
        }
    }

    /// The hierarchy, rebuilt when it has gone stale.
    fn current_hierarchy(&mut self) -> Hierarchy {
        let obs = self.counts.observations;
        let stale = match &self.cache {
            Some((at, _)) => obs.saturating_sub(*at) >= self.rebuild_every,
            None => true,
        };
        if stale {
            self.cache = Some((obs, Hierarchy::build(&self.counts, &self.params)));
        }
        self.cache
            .as_ref()
            .expect("the cache was just filled")
            .1
            .clone()
    }

    /// Records one proposal, with the unbiased acceptance probability that
    /// removes the deposited bias from the estimate.
    pub fn observe(&mut self, from: usize, to: Option<usize>, unbiased_accept: f64) {
        self.counts.observe(from, to, unbiased_accept);
    }

    /// Records a hop the biased chain accepted, for the distortion diagnostic.
    pub fn observe_accepted(&mut self, from: usize, to: usize) {
        self.counts.observe_accepted(from, to);
    }

    /// Stores a structure as the representative of its basin.
    ///
    /// The deepest structure seen in a basin is kept, because that is the one
    /// worth landing on. When the archive is full the shallowest entry is
    /// dropped, for the same reason.
    pub fn keep(&mut self, basin: usize, energy: f64, state: ArrayView1<f64>) {
        match self.store.get_mut(&basin) {
            Some(slot) if energy < slot.0 => {
                slot.0 = energy;
                slot.1 = state.to_owned();
                return;
            }
            Some(_) => return,
            None => {}
        }
        if self.store.len() >= self.archive_capacity {
            let worst = self
                .store
                .iter()
                .max_by(|a, b| a.1.0.partial_cmp(&b.1.0).expect("energies are finite"))
                .map(|(k, v)| (*k, v.0));
            match worst {
                Some((k, e)) if e > energy => {
                    self.store.remove(&k);
                }
                _ => return,
            }
        }
        self.store.insert(basin, (energy, state.to_owned()));
    }

    /// Structures stored.
    pub fn archived(&self) -> usize {
        self.store.len()
    }

    /// The coarse-graining hierarchy of what has been observed.
    pub fn hierarchy(&self) -> Hierarchy {
        Hierarchy::build(&self.counts, &self.params)
    }

    /// Proposes an exit from the superbasin the chain is standing in.
    ///
    /// Costs no objective evaluations: the transition matrix, the structures
    /// and the algebra are all data the run already holds.
    ///
    /// The level is the coarsest one at which the chain sits inside a lump
    /// small enough to eliminate exactly. A coarse exit crosses in one move
    /// what the same algebra says would take the expected number of hops
    /// reported on the jump.
    pub fn propose<R: Rng + ?Sized>(&mut self, here: usize, rng: &mut R) -> Result<Jump, Refusal> {
        let mut hierarchy = self.current_hierarchy();
        if hierarchy.base.position(here).is_none() {
            // A basin registered since the cached hierarchy was built. The
            // chain standing somewhere the graph does not cover is a statement
            // about the cache rather than about the landscape, so the answer is
            // to rebuild rather than to refuse. Measured on a 75-point run,
            // refusing here threw away 12 of 15 attempts.
            self.cache = None;
            hierarchy = self.current_hierarchy();
        }
        if hierarchy.base.len() < self.min_nodes {
            return Err(self.refuse(Refusal::TooFewNodes(hierarchy.base.len())));
        }
        if hierarchy.base.position(here).is_none() {
            return Err(self.refuse(Refusal::Unseen(here)));
        }
        let (level, canonical, absorption) = match self.solve_at_best_level(&hierarchy, here) {
            Ok(v) => v,
            Err(e) => return Err(self.refuse(e)),
        };

        // Only archived exits can provide coordinates. Unnamed exits,
        // evicted structures, and unconverged sparse-solve mass remain a
        // probability of continuing local exploration.
        let mut targets: Vec<(usize, f64, f64)> = Vec::new();
        let mut kept = 0.0;
        for (c, id) in canonical.absorbing.iter().enumerate() {
            let p = absorption.exit[c];
            if !(p > 0.0) || *id == UNKNOWN {
                continue;
            }
            if let Some((basin, energy)) = self.representative(&hierarchy, level, *id) {
                kept += p;
                targets.push((basin, energy, p));
            }
        }
        if targets.is_empty() || !(kept > 0.0) {
            return Err(self.refuse(Refusal::NoArchivedExit));
        }
        let mut u = rng.random::<f64>();
        if u >= kept {
            return Err(self.refuse(Refusal::UnresolvedExit((1.0 - kept).max(0.0))));
        }
        let mut chosen = targets.len() - 1;
        for (k, t) in targets.iter().enumerate() {
            u -= t.2;
            if u <= 0.0 {
                chosen = k;
                break;
            }
        }
        let (basin, energy, p) = targets[chosen];
        let (condition, condition_residual) = canonical.condition_inf(self.condition_sweeps, 1e-10);
        let state = self.store[&basin].1.clone();
        self.stats.jumps += 1;
        self.stats.condition_sum += condition;
        self.stats.condition_max = self.stats.condition_max.max(condition);
        self.stats.condition_residual_max =
            self.stats.condition_residual_max.max(condition_residual);
        self.stats.solve_residual_max = self.stats.solve_residual_max.max(absorption.residual);
        if absorption.exact {
            self.stats.exact_solves += 1;
        }
        self.stats.hops_saved += absorption.hops;
        Ok(Jump {
            basin,
            energy,
            state,
            level,
            probability: p,
            expected_hops: absorption.hops,
            transient: canonical.n_transient(),
            absorbing: canonical.n_absorbing(),
            condition,
            condition_residual,
            solve_residual: absorption.residual,
            exact: absorption.exact,
            conditioned_away: 0.0,
        })
    }

    /// Counts a refusal by kind and hands it back.
    ///
    /// Kept as a breakdown rather than a total because the kinds say different
    /// things: a graph too small is a run that has not walked far enough, a
    /// well-mixed one is a run that is not trapped, and no archived exit is a
    /// trap whose boundary was never entered.
    fn refuse(&mut self, why: Refusal) -> Refusal {
        self.stats.refusals += 1;
        self.stats.refusals_by_kind[why.kind()] += 1;
        if let Refusal::WellMixed(r) = why {
            if r.is_finite() {
                self.stats.mixed_ratio_max = self.stats.mixed_ratio_max.max(r);
            }
        }
        why
    }

    /// Records that a jump landed lower than where it left.
    pub fn observe_gain(&mut self, gain: f64) {
        if gain > 0.0 {
            self.stats.gain += gain;
            self.stats.improvements += 1;
        }
    }

    /// The superbasin around `here`, solved at the coarsest level that fits.
    fn solve_at_best_level(
        &self,
        hierarchy: &Hierarchy,
        here: usize,
    ) -> Result<(usize, Canonical, Absorption), Refusal> {
        let mut last_error = Refusal::WellMixed(f64::NAN);
        // Coarsest first: one move at a high level crosses what would be
        // thousands of hops at the bottom.
        for level in (1..=hierarchy.depth()).rev() {
            let l = &hierarchy.levels[level - 1];
            let state_below = match hierarchy.state_at(level - 1, here) {
                Some(s) => s,
                None => continue,
            };
            let lump_idx = match l.of.get(&state_below) {
                Some(k) if *k < l.lumps.len() => *k,
                _ => continue,
            };
            let below = if level == 1 {
                &hierarchy.base
            } else {
                &hierarchy.levels[level - 2].chain
            };
            let lump = &l.lumps[lump_idx];
            if lump.states.len() > self.max_transient {
                last_error = Refusal::TooLarge(lump.states.len());
                continue;
            }
            let set: Vec<usize> = lump
                .states
                .iter()
                .filter_map(|s| below.position(*s))
                .collect();
            if set.len() < 2 {
                continue;
            }
            let canonical = match Canonical::new(below, &set) {
                Ok(c) => c,
                Err(CanonicalError::NoBoundary) => {
                    last_error = Refusal::NoBoundary;
                    continue;
                }
                Err(_) => continue,
            };
            let source = match canonical.transient.iter().position(|t| *t == state_below) {
                Some(s) => s,
                None => continue,
            };
            let absorption =
                canonical.absorb_at_scale(source, self.elimination_cap, self.solve_sweeps, 1e-10);
            return Ok((level - 1, canonical, absorption));
        }
        // Flat fallback: the trapping component the chain stands in, on the
        // basin graph itself, for a run that has not yet cycled enough for a
        // hierarchy to form.
        //
        // Trapping is tested on the exact answer rather than on a
        // quasi-equilibrium, which the flat exit does not need: a set of m
        // states crossed in about m jumps is being explored, and one that takes
        // ten times that to leave is being cycled in. The measured trap is 35
        // visits per basin.
        let chain = &hierarchy.base;
        let here_local = chain.position(here).ok_or(Refusal::Unseen(here))?;
        // The largest neighbourhood around the chain that is still trapping.
        //
        // Grown by residence, tested by the exact algebra. A set of m states
        // crossed in about m jumps is being explored; one that takes ten times
        // that to leave is being cycled in, which is the condition the exit
        // exists for. The measured trap is 35 visits per basin.
        //
        // Sizes are tried on a doubling ladder rather than one at a time: each
        // test is an elimination, and the answer wanted is an order of
        // magnitude rather than an exact boundary.
        let order = chain.neighbourhood(here_local, self.max_transient);
        if order.len() < self.params.min_states.max(2) {
            return Err(last_error);
        }
        let mut best: Option<(Canonical, Absorption)> = None;
        let mut best_ratio = f64::NAN;
        let mut k = self.params.min_states.max(2);
        loop {
            let take = k.min(order.len());
            let mut set: Vec<usize> = order[..take].to_vec();
            set.sort_unstable();
            match Canonical::new(chain, &set) {
                Ok(canonical) => {
                    if let Some(source) = canonical.transient.iter().position(|t| *t == here) {
                        let absorption = canonical.absorb_at_scale(
                            source,
                            self.elimination_cap,
                            self.solve_sweeps,
                            1e-10,
                        );
                        let ratio = absorption.jumps / take as f64;
                        if best_ratio.is_nan() || ratio > best_ratio {
                            best_ratio = ratio;
                        }
                        if ratio >= self.min_revisits {
                            best = Some((canonical, absorption));
                        }
                    }
                }
                Err(CanonicalError::NoBoundary) => last_error = Refusal::NoBoundary,
                Err(e) => last_error = Refusal::Closed(e),
            }
            if take >= order.len() {
                break;
            }
            k *= 2;
        }
        if let Some((canonical, absorption)) = best {
            return Ok((0, canonical, absorption));
        }
        if !best_ratio.is_nan() {
            last_error = Refusal::WellMixed(best_ratio);
        }
        Err(last_error)
    }

    /// The deepest archived structure under a coarse state.
    fn representative(
        &self,
        hierarchy: &Hierarchy,
        level: usize,
        state: usize,
    ) -> Option<(usize, f64)> {
        if level == 0 {
            return self.store.get(&state).map(|(e, _)| (state, *e));
        }
        let mut best: Option<(usize, f64)> = None;
        for (basin, (energy, _)) in &self.store {
            if hierarchy.state_at(level, *basin) == Some(state) {
                match best {
                    Some((_, e)) if e <= *energy => {}
                    _ => best = Some((*basin, *energy)),
                }
            }
        }
        best
    }

    /// The report for this run.
    pub fn report(&self) -> SuperbasinReport {
        let h = self.hierarchy();
        let levels: Vec<(usize, usize, f64)> = h
            .levels
            .iter()
            .map(|l| {
                let largest = l
                    .lumps
                    .iter()
                    .max_by_key(|k| k.states.len())
                    .map(|k| (k.states.len(), k.separation))
                    .unwrap_or((0, f64::NAN));
                (l.ids.len(), largest.0, largest.1)
            })
            .collect();
        let top: Vec<usize> = h
            .top_partition()
            .into_iter()
            .take(8)
            .map(|g| g.len())
            .collect();
        SuperbasinReport {
            nodes: h.base.len(),
            edges: h.base.n_edges(),
            distortion: self.counts.bias_distortion(),
            levels,
            top,
            jumps: self.stats.jumps,
            refusals: self.stats.refusals,
            refusals_by_kind: self.stats.refusals_by_kind,
            mixed_ratio_max: self.stats.mixed_ratio_max,
            condition_max: self.stats.condition_max,
            condition_mean: self.stats.condition_mean(),
            condition_residual_max: self.stats.condition_residual_max,
            solve_residual_max: self.stats.solve_residual_max,
            exact_solves: self.stats.exact_solves,
            hops_saved: self.stats.hops_saved,
            improvements: (self.stats.improvements, self.stats.gain),
            archived: self.store.len(),
            separability: None,
            quotient: None,
        }
    }
}

/// What a run has to say about the superbasin structure it walked through.
///
/// Produced whether or not the escape move fired, because the graph is recorded
/// regardless and the hierarchy is the cheap evidence about whether the funnel
/// decomposition is recoverable from transitions.
#[derive(Debug, Clone, Default)]
pub struct SuperbasinReport {
    /// Basins in the transition graph.
    pub nodes: usize,
    /// Directed transitions carrying reweighted mass.
    pub edges: usize,
    /// Total variation between the raw accepted counts and the reweighted ones.
    pub distortion: f64,
    /// Per level: coarse states, largest lump, and its timescale separation.
    pub levels: Vec<(usize, usize, f64)>,
    /// Basins under each top-level coarse state, largest first, truncated to
    /// the largest eight.
    pub top: Vec<usize>,
    /// Jumps taken.
    pub jumps: usize,
    /// Escapes refused.
    pub refusals: usize,
    /// Refusals by kind, indexed by [`Refusal::kind`].
    pub refusals_by_kind: [usize; 7],
    /// Largest revisits-per-state ratio a well-mixed refusal reported.
    ///
    /// The scientific content of a negative result: if the observed graph never
    /// reaches the trapping threshold, the chain is not cycling inside a
    /// superbasin at the resolution basin identity provides, and no exit
    /// algebra applies however good the algebra is.
    pub mixed_ratio_max: f64,
    /// Largest condition number seen on a jump.
    pub condition_max: f64,
    /// Mean condition number over the jumps.
    pub condition_mean: f64,
    /// Largest Gauss-Seidel residual behind a reported condition number.
    pub condition_residual_max: f64,
    /// Largest residual of a solve that produced an exit distribution.
    pub solve_residual_max: f64,
    /// Jumps whose exit came from the exact elimination.
    pub exact_solves: usize,
    /// Expected hops the jumps replaced, summed.
    pub hops_saved: f64,
    /// Jumps that landed lower than they left, and the depth they gained.
    pub improvements: (usize, f64),
    /// Structures held for landing on.
    pub archived: usize,
    /// Structural separability of the top-level coarse states, when asked for.
    pub separability: Option<Separability>,
    /// What taking basin identity modulo the symmetry orbit does to the graph.
    pub quotient: Option<QuotientReport>,
}

/// How well a set of features separates the coarse states the transitions
/// imply.
///
/// A one-way analysis of variance on the funnel labels, pooled over feature
/// dimensions: between-group variance over within-group variance, with the
/// usual degrees of freedom. It answers one question and it is the question the
/// successor-representation framing raises.
///
/// `N = (I - Q)^-1` can only be formed over states that were visited, so the
/// exit it computes is available for a superbasin already mapped and not for
/// one never seen, and a 98-point run visits 11366 basins on a landscape whose
/// minimum count grows like `exp(alpha N)`. Reinforcement learning went around
/// that by approximating the same operator as a function of state features
/// rather than tabulating it over states (successor features,
/// doi:10.1073/pnas.1907370117). That works exactly when basins in the same
/// coarse state look alike and basins in different ones do not.
///
/// `f` well above one says the transition-derived partition is structurally
/// predictable, so an exit could be learned for basins never entered. `f` near
/// one says it is not, and that a learned exit would have nothing to learn
/// from these features.
#[derive(Debug, Clone)]
pub struct Separability {
    /// Pooled F statistic: between-group over within-group variance.
    pub f: f64,
    /// Coarse states carrying at least two structures.
    pub groups: usize,
    /// Structures the statistic was computed on.
    pub points: usize,
    /// Per-dimension F, so a single informative feature is visible behind a
    /// pooled average.
    pub per_dimension: Vec<f64>,
}

impl SuperbasinEscape {
    /// Structural separability of the top-level coarse states.
    ///
    /// `features` maps a structure to a descriptor. Returns `None` when fewer
    /// than two coarse states carry two structures each, which is not a
    /// negative result but an absence of one.
    pub fn separability<F>(&self, features: F) -> Option<Separability>
    where
        F: Fn(ArrayView1<f64>) -> Array1<f64>,
    {
        let h = self.hierarchy();
        if h.depth() == 0 {
            return None;
        }
        let depth = h.depth();
        let mut groups: BTreeMap<usize, Vec<Array1<f64>>> = BTreeMap::new();
        for (basin, (_, state)) in &self.store {
            if let Some(top) = h.state_at(depth, *basin) {
                groups.entry(top).or_default().push(features(state.view()));
            }
        }
        groups.retain(|_, v| v.len() >= 2);
        if groups.len() < 2 {
            return None;
        }
        let dim = groups.values().next()?.first()?.len();
        let points: usize = groups.values().map(|v| v.len()).sum();
        let mut per_dimension = Vec::with_capacity(dim);
        for d in 0..dim {
            let grand: f64 = groups
                .values()
                .flat_map(|v| v.iter().map(|x| x[d]))
                .sum::<f64>()
                / points as f64;
            let mut between = 0.0;
            let mut within = 0.0;
            for v in groups.values() {
                let mean: f64 = v.iter().map(|x| x[d]).sum::<f64>() / v.len() as f64;
                between += v.len() as f64 * (mean - grand) * (mean - grand);
                within += v.iter().map(|x| (x[d] - mean) * (x[d] - mean)).sum::<f64>();
            }
            let df_b = (groups.len() - 1) as f64;
            let df_w = (points - groups.len()) as f64;
            let f = if within > 0.0 && df_w > 0.0 {
                (between / df_b) / (within / df_w)
            } else {
                f64::NAN
            };
            per_dimension.push(f);
        }
        let finite: Vec<f64> = per_dimension
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        let f = if finite.is_empty() {
            f64::NAN
        } else {
            finite.iter().sum::<f64>() / finite.len() as f64
        };
        Some(Separability {
            f,
            groups: groups.len(),
            points,
            per_dimension,
        })
    }
}

// ---------------------------------------------------------------------------
// Quotienting the graph by the symmetry orbit
// ---------------------------------------------------------------------------

/// What quotienting the transition graph by the symmetry orbit does to it.
///
/// The graph's states are descriptor classes, and a sorted distance spectrum
/// distinguishes a structure from its own relabelling and from its own
/// point-group images. Those are one state. A chain re-entering one minimum
/// under twenty labels therefore registers as twenty states entered once each,
/// which is indistinguishable, in the expected-visits statistic, from a chain
/// sweeping through twenty different minima. The two readings differ by exactly
/// this quotient, so it is measured rather than argued.
///
/// Only labels the run actually visited are merged. Generating the unvisited
/// members of an orbit from the point group would add states that were never
/// entered, which cannot raise a visits-per-state ratio; filling the interior
/// analytically is what an escape would want, not what this measurement needs.
#[derive(Debug, Clone)]
pub struct QuotientReport {
    /// Distinct basins before quotienting.
    pub basins_raw: usize,
    /// Distinct basins after.
    pub basins_quotiented: usize,
    /// Equivalence classes holding more than one label.
    pub orbits_nontrivial: usize,
    /// Labels in the largest class.
    pub largest_orbit: usize,
    /// Structures the archive held, which bounds what could be merged.
    pub archived: usize,
    /// Energy buckets holding more than one structure.
    pub energy_buckets: usize,
    /// Shape comparisons actually run.
    pub comparisons: usize,
    /// Largest shape distance accepted as the same structure.
    pub matched_max: f64,
    /// Smallest shape distance rejected inside an energy bucket.
    ///
    /// With `matched_max` this is the gap the merge threshold sits in, so the
    /// threshold is auditable rather than asserted.
    pub rejected_min: f64,
    /// Median and maximum expected visits per state, before quotienting.
    pub revisits_raw: (f64, f64),
    /// The same after quotienting. This is the number that decides it.
    pub revisits_quotiented: (f64, f64),
    /// Source basins the statistic was computed from.
    pub sources: usize,
    /// Coarse-graining depth before and after.
    pub depth: (usize, usize),
    /// Share of basins that joined a lump, before and after.
    pub lumped_fraction: (f64, f64),
}

/// Expected visits per state, over a sample of starting basins.
///
/// The same quantity the escape's trapping test uses, computed the same way:
/// grow the neighbourhood by residence, try sizes on a doubling ladder, take
/// the largest ratio the source reaches. Reported as the median and maximum
/// over sources, so one recurrent corner cannot carry the summary.
fn revisit_profile(
    chain: &JumpChain,
    sources: &[usize],
    max_states: usize,
    elimination_cap: usize,
    sweeps: usize,
    min_states: usize,
) -> (f64, f64) {
    let mut best: Vec<f64> = Vec::new();
    for src in sources {
        let order = chain.neighbourhood(*src, max_states);
        if order.len() < min_states.max(2) {
            continue;
        }
        let id = chain.nodes()[*src];
        let mut top = f64::NAN;
        let mut k = min_states.max(2);
        loop {
            let take = k.min(order.len());
            let mut set: Vec<usize> = order[..take].to_vec();
            set.sort_unstable();
            if let Ok(c) = Canonical::new(chain, &set) {
                if let Some(local) = c.transient.iter().position(|t| *t == id) {
                    let a = c.absorb_at_scale(local, elimination_cap, sweeps, 1e-10);
                    let ratio = a.jumps / take as f64;
                    if top.is_nan() || ratio > top {
                        top = ratio;
                    }
                }
            }
            if take >= order.len() {
                break;
            }
            k *= 2;
        }
        if top.is_finite() {
            best.push(top);
        }
    }
    if best.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    best.sort_by(|a, b| a.partial_cmp(b).expect("ratios are finite"));
    (
        best[best.len() / 2],
        *best.last().expect("best is non-empty"),
    )
}

/// Depth of the hierarchy and the share of states that joined a lump.
fn lump_coverage(counts: &HopCounts, params: &LumpParams, nodes: usize) -> (usize, f64) {
    let h = Hierarchy::build(counts, params);
    let inside: usize = h
        .levels
        .first()
        .map(|l| l.lumps.iter().map(|k| k.states.len()).sum())
        .unwrap_or(0);
    (h.depth(), inside as f64 / nodes.max(1) as f64)
}

/// Path compression over the label classes.
fn find(parent: &mut BTreeMap<usize, usize>, x: usize) -> usize {
    let mut r = x;
    while parent[&r] != r {
        r = parent[&r];
    }
    let mut c = x;
    while parent[&c] != c {
        let n = parent[&c];
        parent.insert(c, r);
        c = n;
    }
    r
}

impl SuperbasinEscape {
    /// Rebuilds the graph with basin identity taken modulo the symmetry orbit,
    /// and reports what changes.
    ///
    /// `distance` returns the shape distance between two archived structures,
    /// which is zero for one structure against its own relabelling and rotation
    /// and order one between different minima. It is only ever asked about
    /// structures whose quenched energies already agree to `energy_tol`, and
    /// that filter is exact rather than heuristic: an orbit is a level set of
    /// the energy, so two structures at different energies cannot be orbit
    /// members, and the filter removes the quadratic cost without removing a
    /// single true pair.
    pub fn quotient<D>(
        &self,
        distance: D,
        merge_tol: f64,
        energy_tol: f64,
        sources: usize,
    ) -> QuotientReport
    where
        D: Fn(ArrayView1<f64>, ArrayView1<f64>) -> f64,
    {
        let raw_chain = JumpChain::from_counts(&self.counts);
        // Archived basins ordered by energy, so an orbit is a contiguous run.
        let mut by_energy: Vec<(usize, f64)> =
            self.store.iter().map(|(b, (e, _))| (*b, *e)).collect();
        by_energy.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("energies are finite"));

        let mut parent: BTreeMap<usize, usize> = by_energy.iter().map(|(b, _)| (*b, *b)).collect();
        let mut comparisons = 0usize;
        let mut buckets = 0usize;
        let mut matched_max = 0.0_f64;
        let mut rejected_min = f64::INFINITY;
        let mut i = 0usize;
        while i < by_energy.len() {
            // A bucket runs while consecutive energies stay within the
            // tolerance, which is how a degenerate level presents.
            let mut j = i + 1;
            while j < by_energy.len() && (by_energy[j].1 - by_energy[j - 1].1).abs() <= energy_tol {
                j += 1;
            }
            if j - i > 1 {
                buckets += 1;
                for a in i..j {
                    for b in (a + 1)..j {
                        let (ba, bb) = (by_energy[a].0, by_energy[b].0);
                        if find(&mut parent, ba) == find(&mut parent, bb) {
                            continue;
                        }
                        comparisons += 1;
                        let d = distance(self.store[&ba].1.view(), self.store[&bb].1.view());
                        if d <= merge_tol {
                            matched_max = matched_max.max(d);
                            let ra = find(&mut parent, ba);
                            let rb = find(&mut parent, bb);
                            let (lo, hi) = (ra.min(rb), ra.max(rb));
                            parent.insert(hi, lo);
                        } else {
                            rejected_min = rejected_min.min(d);
                        }
                    }
                }
            }
            i = j;
        }

        let mut map: BTreeMap<usize, usize> = BTreeMap::new();
        for b in raw_chain.nodes() {
            let rep = if parent.contains_key(b) {
                find(&mut parent, *b)
            } else {
                *b
            };
            map.insert(*b, rep);
        }
        let mut sizes: BTreeMap<usize, usize> = BTreeMap::new();
        for rep in map.values() {
            *sizes.entry(*rep).or_insert(0) += 1;
        }

        // The same counts under the quotient. A transition between two labels
        // of one class becomes time spent rather than a transition, which is
        // what it always was.
        let mut folded = HopCounts::new();
        for (i, tos) in &self.counts.out {
            let ri = map[i];
            for (j, w) in tos {
                let rj = map.get(j).copied().unwrap_or(*j);
                folded.observe_weighted(ri, Some(rj), *w);
            }
        }
        for (i, w) in &self.counts.leak {
            folded.observe_weighted(map[i], None, *w);
        }
        for (i, t) in &self.counts.time {
            folded.add_time(map[i], *t);
        }
        let quot_chain = JumpChain::from_counts(&folded);

        let pick = |c: &JumpChain| -> Vec<usize> {
            let mut idx: Vec<usize> = (0..c.len()).collect();
            idx.sort_by(|a, b| {
                c.residence(*b)
                    .partial_cmp(&c.residence(*a))
                    .expect("residences are finite")
            });
            idx.truncate(sources);
            idx
        };
        let raw_sources = pick(&raw_chain);
        let quot_sources = pick(&quot_chain);
        let revisits_raw = revisit_profile(
            &raw_chain,
            &raw_sources,
            self.max_transient,
            self.elimination_cap,
            self.solve_sweeps,
            self.params.min_states,
        );
        let revisits_quotiented = revisit_profile(
            &quot_chain,
            &quot_sources,
            self.max_transient,
            self.elimination_cap,
            self.solve_sweeps,
            self.params.min_states,
        );
        let (depth_raw, frac_raw) = lump_coverage(&self.counts, &self.params, raw_chain.len());
        let (depth_q, frac_q) = lump_coverage(&folded, &self.params, quot_chain.len());

        QuotientReport {
            basins_raw: raw_chain.len(),
            basins_quotiented: quot_chain.len(),
            orbits_nontrivial: sizes.values().filter(|v| **v > 1).count(),
            largest_orbit: sizes.values().copied().max().unwrap_or(0),
            archived: self.store.len(),
            energy_buckets: buckets,
            comparisons,
            matched_max,
            rejected_min: if rejected_min.is_finite() {
                rejected_min
            } else {
                f64::NAN
            },
            revisits_raw,
            revisits_quotiented,
            sources: raw_sources.len(),
            depth: (depth_raw, depth_q),
            lumped_fraction: (frac_raw, frac_q),
        }
    }
}

/// The graph rebuilt with basins merged according to `map`.
///
/// A transition between two labels of one class becomes time spent rather than
/// a transition, which is what it always was if the two labels are one state.
/// Labels absent from `map` stay as themselves.
pub fn regroup(counts: &HopCounts, map: &BTreeMap<usize, usize>) -> HopCounts {
    let rep = |b: usize| map.get(&b).copied().unwrap_or(b);
    let mut folded = HopCounts::new();
    for (i, tos) in &counts.out {
        for (j, w) in tos {
            folded.observe_weighted(rep(*i), Some(rep(*j)), *w);
        }
    }
    for (i, w) in &counts.leak {
        folded.observe_weighted(rep(*i), None, *w);
    }
    for (i, t) in &counts.time {
        folded.add_time(rep(*i), *t);
    }
    folded
}

/// The statistic that decides whether a graph holds a trap to escape.
#[derive(Debug, Clone)]
pub struct Profile {
    /// States in the graph.
    pub states: usize,
    /// Median expected visits per state over the sampled sources.
    pub revisits_median: f64,
    /// Largest expected visits per state over the sampled sources.
    pub revisits_max: f64,
    /// Sources the statistic was computed from.
    pub sources: usize,
    /// Coarse-graining depth.
    pub depth: usize,
    /// Share of states that joined a lump.
    pub lumped_fraction: f64,
}

/// Expected visits per state and lumping coverage for a graph.
///
/// One is the sweeping case: each state entered once on the way through. Above
/// one the chain is returning, which is what a superbasin means and what an
/// absorbing exit needs in order to have anything to act on.
pub fn profile(
    counts: &HopCounts,
    params: &LumpParams,
    sources: usize,
    max_states: usize,
    elimination_cap: usize,
    sweeps: usize,
) -> Profile {
    let chain = JumpChain::from_counts(counts);
    let mut idx: Vec<usize> = (0..chain.len()).collect();
    idx.sort_by(|a, b| {
        chain
            .residence(*b)
            .partial_cmp(&chain.residence(*a))
            .expect("residences are finite")
    });
    idx.truncate(sources);
    let (median, max) = revisit_profile(
        &chain,
        &idx,
        max_states,
        elimination_cap,
        sweeps,
        params.min_states,
    );
    let (depth, lumped_fraction) = lump_coverage(counts, params, chain.len());
    Profile {
        states: chain.len(),
        revisits_median: median,
        revisits_max: max,
        sources: idx.len(),
        depth,
        lumped_fraction,
    }
}

impl SuperbasinEscape {
    /// The archived structures, for analysis outside the driver.
    pub fn archive_entries(&self) -> Vec<(usize, f64, Array1<f64>)> {
        self.store
            .iter()
            .map(|(b, (e, x))| (*b, *e, x.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Builds a jump chain directly from transition probabilities, for the
    /// tests that assert against a closed form.
    fn chain_from(rows: &[(usize, Vec<(usize, f64)>)]) -> JumpChain {
        let mut counts = HopCounts::new();
        for (i, row) in rows {
            let total: f64 = row.iter().map(|(_, p)| *p).sum();
            for (j, p) in row {
                counts.observe_weighted(*i, Some(*j), *p);
            }
            // One hop per jump, so expected jumps and expected hops agree and
            // the closed forms apply to both.
            counts.add_time(*i, total);
        }
        JumpChain::from_counts(&counts)
    }

    /// A birth-death chain on `0..=m` with `p` up, absorbing at both ends.
    fn birth_death(m: usize, p: f64) -> (JumpChain, Vec<usize>) {
        let mut rows = Vec::new();
        for i in 1..m {
            rows.push((i, vec![(i + 1, p), (i - 1, 1.0 - p)]));
        }
        (chain_from(&rows), (1..m).collect())
    }

    fn positions(chain: &JumpChain, ids: &[usize]) -> Vec<usize> {
        ids.iter()
            .map(|s| chain.position(*s).expect("state is in the chain"))
            .collect()
    }

    #[test]
    fn absorption_matches_gamblers_ruin() {
        // P(absorbed at m | start i) = (1 - (q/p)^i) / (1 - (q/p)^m).
        let m = 6;
        let p = 2.0 / 3.0;
        let (chain, set) = birth_death(m, p);
        let canonical = Canonical::new(&chain, &positions(&chain, &set)).expect("boundary exists");
        let ratio = (1.0 - p) / p;
        let top = canonical
            .absorbing
            .iter()
            .position(|a| *a == m)
            .expect("upper absorbing state");
        for (k, i) in (1..m).enumerate() {
            let a = canonical.absorb(k);
            let closed = (1.0 - ratio.powi(i as i32)) / (1.0 - ratio.powi(m as i32));
            assert!(
                (a.exit[top] - closed).abs() < 1e-12,
                "start {i}: got {}, closed form {closed}",
                a.exit[top]
            );
            assert!((a.exit.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn expected_steps_match_symmetric_walk() {
        // Expected steps to absorption from i is i(m - i) for the symmetric
        // walk, and one jump is one hop in this construction.
        let m = 7;
        let (chain, set) = birth_death(m, 0.5);
        let canonical = Canonical::new(&chain, &positions(&chain, &set)).expect("boundary exists");
        for (k, i) in (1..m).enumerate() {
            let a = canonical.absorb(k);
            let closed = (i * (m - i)) as f64;
            assert!(
                (a.jumps - closed).abs() < 1e-9,
                "start {i}: got {}, closed form {closed}",
                a.jumps
            );
            assert!(
                (a.hops - closed).abs() < 1e-9,
                "hops {} at start {i}",
                a.hops
            );
        }
    }

    #[test]
    fn elimination_agrees_with_iteration() {
        // The exact elimination and the iterative solve are different
        // algorithms on the same system; they have to agree where the iteration
        // converges.
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..20 {
            let n = 6usize;
            let mut counts = HopCounts::new();
            for i in 0..n {
                let mut row = Vec::new();
                let mut total = 0.0;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let w: f64 = rng.random::<f64>();
                    row.push((j, w));
                    total += w;
                }
                let exit: f64 = rng.random::<f64>() * 0.1;
                for (j, w) in row {
                    counts.observe_weighted(i, Some(j), w / (total + exit));
                }
                counts.observe_weighted(i, Some(900 + i % 2), exit / (total + exit));
                counts.add_time(i, 1.0);
            }
            let chain = JumpChain::from_counts(&counts);
            let set: Vec<usize> = positions(&chain, &(0..n).collect::<Vec<usize>>());
            let canonical = Canonical::new(&chain, &set).expect("boundary exists");
            let a = canonical.absorb(0);
            let (u, residual, _) = canonical.expected_jumps(500_000, 1e-13);
            assert!(residual < 1e-9, "residual {residual:.2e}");
            assert!(
                (u[0] - a.jumps).abs() < 1e-6 * a.jumps.max(1.0),
                "elimination {} against iteration {}",
                a.jumps,
                u[0]
            );
            assert!((a.exit.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn sparse_solve_matches_the_closed_form_and_the_elimination() {
        // The two solvers share no code. On the symmetric walk both must give
        // i(m - i) expected steps and the same exit distribution, and the
        // sparse one has to report a residual that bounds its own error.
        let m = 9;
        let (chain, set) = birth_death(m, 0.55);
        let canonical = Canonical::new(&chain, &positions(&chain, &set)).expect("boundary exists");
        for k in 0..canonical.n_transient() {
            let exact = canonical.absorb(k);
            let sparse = canonical.absorb_sparse(k, 500_000, 1e-14);
            assert!(exact.exact && !sparse.exact);
            assert!(sparse.residual <= 1e-13, "residual {:.2e}", sparse.residual);
            assert!(
                (sparse.jumps - exact.jumps).abs() < 1e-8 * exact.jumps,
                "state {k}: sparse {} exact {}",
                sparse.jumps,
                exact.jumps
            );
            for (a, b) in sparse.exit.iter().zip(exact.exit.iter()) {
                assert!((a - b).abs() < 1e-9, "exit {a} against {b}");
            }
        }
        // Truncated, it underestimates rather than overshooting, which is what
        // makes a capped sweep count safe to act on.
        let short = canonical.absorb_sparse(0, 3, 0.0);
        let exact = canonical.absorb(0);
        assert!(
            short.jumps < exact.jumps && short.residual > 1e-6,
            "truncated {} against exact {} residual {:.2e}",
            short.jumps,
            exact.jumps,
            short.residual
        );
    }

    #[test]
    fn condition_number_is_the_trapping() {
        // A three-state trap leaving with probability eps per jump. The
        // infinity-norm condition number of I - Q is ||I - Q||_inf times the
        // largest expected number of jumps, exactly, and it grows as the trap
        // tightens.
        let mut previous = 0.0;
        for eps in [1e-2, 1e-3, 1e-4] {
            let rows = vec![
                (
                    0usize,
                    vec![(1, 0.5 - eps / 2.0), (2, 0.5 - eps / 2.0), (100, eps)],
                ),
                (
                    1,
                    vec![(0, 0.5 - eps / 2.0), (2, 0.5 - eps / 2.0), (101, eps)],
                ),
                (2, vec![(0, 0.5), (1, 0.5)]),
            ];
            let chain = chain_from(&rows);
            let set = positions(&chain, &[0, 1, 2]);
            let canonical = Canonical::new(&chain, &set).expect("boundary exists");
            let (kappa, residual) = canonical.condition_inf(4_000_000, 1e-10);
            let longest = (0..3)
                .map(|k| canonical.absorb(k).jumps)
                .fold(0.0_f64, f64::max);
            assert!(residual < 1e-6, "residual {residual:.3e} at eps {eps}");
            assert!(
                (kappa - 2.0 * longest).abs() < 1e-3 * kappa,
                "eps {eps}: kappa {kappa:.4e} against 2 * {longest:.4e}"
            );
            assert!(kappa > 0.5 / eps, "eps {eps} gave kappa {kappa:.3e}");
            assert!(
                kappa > previous,
                "the conditioning must worsen with the trap"
            );
            previous = kappa;
        }
    }

    #[test]
    fn single_exit_is_certain() {
        // A trap with exactly one way out gives that exit with probability one,
        // whatever the internal structure.
        let rows = vec![
            (0usize, vec![(1, 0.6), (2, 0.4)]),
            (1, vec![(0, 0.7), (2, 0.3)]),
            (2, vec![(0, 0.45), (1, 0.45), (9, 0.1)]),
        ];
        let chain = chain_from(&rows);
        let canonical =
            Canonical::new(&chain, &positions(&chain, &[0, 1, 2])).expect("boundary exists");
        assert_eq!(canonical.absorbing, vec![9]);
        for k in 0..3 {
            let a = canonical.absorb(k);
            assert!(
                (a.exit[0] - 1.0).abs() < 1e-13,
                "state {k} gave {}",
                a.exit[0]
            );
        }
    }

    #[test]
    fn unknown_destinations_stay_visible() {
        // Mass leaving to a basin the run never registered is carried as its
        // own absorbing column rather than folded into the escape rate.
        let mut counts = HopCounts::new();
        for _ in 0..100 {
            counts.observe(0, Some(1), 1.0);
            counts.observe(1, Some(0), 1.0);
        }
        counts.observe(0, None, 1.0);
        counts.observe(1, Some(2), 1.0);
        let chain = JumpChain::from_counts(&counts);
        let canonical =
            Canonical::new(&chain, &positions(&chain, &[0, 1])).expect("boundary exists");
        assert!(canonical.absorbing.contains(&UNKNOWN));
        assert!(canonical.absorbing.contains(&2));
        let a = canonical.absorb(0);
        assert!((a.exit.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let unknown = canonical
            .absorbing
            .iter()
            .position(|x| *x == UNKNOWN)
            .expect("unknown column");
        assert!(
            a.exit[unknown] > 0.4 && a.exit[unknown] < 0.6,
            "the two exits are symmetric: {:?}",
            a.exit
        );
    }

    #[test]
    fn lumping_reproduces_the_exact_exit_when_separated() {
        // The mean rate method is an approximation and its error is
        // O(1 / separation). Measured against the exact absorbing-chain answer
        // on the same set.
        let eps = 1e-3;
        let rows = vec![
            (0usize, vec![(1, 0.5 - eps), (2, 0.5), (100, eps)]),
            (1, vec![(0, 0.5), (2, 0.5 - eps), (101, eps)]),
            (2, vec![(0, 0.5), (1, 0.5)]),
        ];
        let chain = chain_from(&rows);
        let set = positions(&chain, &[0, 1, 2]);
        let params = LumpParams::default();
        let (pi, relax) = quasi_equilibrium(&chain, &set, &params).expect("strongly connected");
        let inside: BTreeSet<usize> = set.iter().copied().collect();
        let mut k_eff: BTreeMap<usize, f64> = BTreeMap::new();
        let (k_esc, _) = escape_rate(&chain, &set, &pi);
        for (i, s) in set.iter().enumerate() {
            for (t, p, _) in chain.edges(*s) {
                if !inside.contains(t) {
                    *k_eff.entry(chain.nodes()[*t]).or_insert(0.0) += pi[i] * p;
                }
            }
        }
        let canonical = Canonical::new(&chain, &set).expect("boundary exists");
        let exact = canonical.absorb(0);
        let mut tv = 0.0;
        for (c, id) in canonical.absorbing.iter().enumerate() {
            let lumped = k_eff.get(id).copied().unwrap_or(0.0) / k_esc;
            tv += (lumped - exact.exit[c]).abs();
        }
        tv *= 0.5;
        let separation = (1.0 / k_esc) / relax;
        assert!(separation > 100.0, "separation {separation}");
        assert!(
            tv < 10.0 / separation,
            "total variation {tv:.3e} against separation {separation:.1}"
        );
    }

    #[test]
    fn well_mixed_set_is_not_lumpable() {
        // Twelve states, each reachable from every other, with exits as fast as
        // the internal transitions. Quasi-equilibrium is never established
        // before the set is left, so the criterion has to refuse.
        let mut counts = HopCounts::new();
        let n = 12usize;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    counts.observe_weighted(i, Some(j), 3.0);
                }
            }
            // Escape mass equal to the internal mass: leaving is as fast as
            // moving inside.
            counts.observe_weighted(i, Some(1000 + i), 33.0);
            counts.add_time(i, 1.0);
        }
        let chain = JumpChain::from_counts(&counts);
        let params = LumpParams::default();
        let set = positions(&chain, &(0..n).collect::<Vec<usize>>());
        let (pi, relax) = quasi_equilibrium(&chain, &set, &params).expect("strongly connected");
        let (k_esc, _) = escape_rate(&chain, &set, &pi);
        let separation = (1.0 / k_esc) / relax;
        assert!(
            separation < params.min_separation,
            "separation {separation:.2} must fall below the threshold"
        );
        assert!(
            lump_once(&chain, &params, params.min_executions).is_none(),
            "a well-mixed set with fast exits must not be lumped"
        );
    }

    #[test]
    fn slow_internal_mixing_is_not_lumpable() {
        // Two sub-wells joined by a rare internal transition. The union is well
        // connected and emphatically not in quasi-equilibrium, which is the
        // failure mode lumping has to refuse; each half on its own is lumpable,
        // and the criterion has to tell the two situations apart.
        let eps = 1e-3;
        let mut rows = Vec::new();
        for i in 0..4usize {
            let mut row: Vec<(usize, f64)> = (0..4)
                .filter(|j| *j != i)
                .map(|j| (j, (1.0 - 2.0 * eps) / 3.0))
                .collect();
            row.push((4, eps));
            row.push((100 + i, eps));
            rows.push((i, row));
        }
        for i in 4..8usize {
            let mut row: Vec<(usize, f64)> = (4..8)
                .filter(|j| *j != i)
                .map(|j| (j, (1.0 - 2.0 * eps) / 3.0))
                .collect();
            row.push((0, eps));
            row.push((100 + i, eps));
            rows.push((i, row));
        }
        let chain = chain_from(&rows);
        let params = LumpParams::default();
        let union = positions(&chain, &(0..8).collect::<Vec<usize>>());
        assert!(
            quasi_equilibrium(&chain, &union, &params).is_none(),
            "the union does not equilibrate within the cap and must be refused"
        );
        let half = positions(&chain, &[0, 1, 2, 3]);
        let (pi, relax) = quasi_equilibrium(&chain, &half, &params).expect("half equilibrates");
        let (k_esc, _) = escape_rate(&chain, &half, &pi);
        assert!(
            (1.0 / k_esc) / relax > params.min_separation,
            "each half on its own is a superbasin"
        );
    }

    #[test]
    fn hierarchy_recovers_two_funnels() {
        // Two trapping sets joined by one rare transition. The transitions alone
        // must recover the partition: this is the object four descriptors
        // failed to separate on the real landscape.
        let mut counts = HopCounts::new();
        let a: Vec<usize> = (0..8).collect();
        let b: Vec<usize> = (8..16).collect();
        for group in [&a, &b] {
            for i in group.iter() {
                for j in group.iter() {
                    if i != j {
                        counts.observe_weighted(*i, Some(*j), 1000.0);
                    }
                }
                counts.add_time(*i, 1.0);
            }
        }
        counts.observe_weighted(0, Some(8), 2.0);
        counts.observe_weighted(8, Some(0), 2.0);
        let params = LumpParams::default();
        let h = Hierarchy::build(&counts, &params);
        assert_eq!(h.depth(), 1, "levels {:?}", h.depth());
        let top = h.top_partition();
        assert_eq!(top.len(), 2, "top partition {top:?}");
        assert_eq!(top[0].len(), 8);
        assert_eq!(top[1].len(), 8);
        let sides: BTreeSet<usize> = top[0].iter().copied().collect();
        assert!(
            sides == a.iter().copied().collect::<BTreeSet<usize>>()
                || sides == b.iter().copied().collect::<BTreeSet<usize>>(),
            "the split does not follow the construction: {top:?}"
        );
    }

    #[test]
    fn hierarchy_recurses_on_nested_traps() {
        // Four tight groups, paired into two, joined once. One level of lumping
        // cannot express that; the recursion has to.
        let mut counts = HopCounts::new();
        let groups: Vec<Vec<usize>> = (0..4)
            .map(|g| ((g * 6)..(g * 6 + 6)).collect::<Vec<usize>>())
            .collect();
        for g in &groups {
            for i in g {
                for j in g {
                    if i != j {
                        counts.observe_weighted(*i, Some(*j), 1e6);
                    }
                }
                counts.add_time(*i, 1.0);
            }
        }
        counts.observe_weighted(0, Some(6), 1e2);
        counts.observe_weighted(6, Some(0), 1e2);
        counts.observe_weighted(12, Some(18), 1e2);
        counts.observe_weighted(18, Some(12), 1e2);
        counts.observe_weighted(1, Some(13), 2.0);
        counts.observe_weighted(13, Some(1), 2.0);
        let h = Hierarchy::build(&counts, &LumpParams::default());
        assert!(h.depth() >= 2, "depth {} on a nested landscape", h.depth());
        let top = h.top_partition();
        assert_eq!(top.len(), 2, "top partition {top:?}");
        assert_eq!(top[0].len(), 12);
        assert_eq!(top[1].len(), 12);
    }

    #[test]
    fn refuses_a_graph_too_small_to_judge() {
        let mut esc = SuperbasinEscape::new();
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..50 {
            esc.observe(0, Some(1), 1.0);
            esc.observe(1, Some(0), 1.0);
        }
        esc.keep(0, -1.0, Array1::zeros(3).view());
        esc.keep(1, -2.0, Array1::zeros(3).view());
        match esc.propose(0, &mut rng) {
            Err(Refusal::TooFewNodes(n)) => assert_eq!(n, 2),
            other => panic!("expected a refusal on a two-node graph, got {other:?}"),
        }
        assert_eq!(esc.stats.jumps, 0);
    }

    #[test]
    fn refuses_a_well_mixed_graph() {
        // Forty basins, every one reachable from every other at the same rate.
        // The chain is exploring, not trapped, and jumping would push it out of
        // a region it has not finished.
        let mut esc = SuperbasinEscape::new();
        let mut rng = StdRng::seed_from_u64(2);
        let n = 40usize;
        for _ in 0..3 {
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        esc.observe(i, Some(j), 1.0);
                    }
                }
            }
        }
        for i in 0..n {
            esc.keep(i, -(i as f64), Array1::zeros(3).view());
        }
        match esc.propose(0, &mut rng) {
            Err(Refusal::WellMixed(_)) | Err(Refusal::NoBoundary) => {}
            other => panic!("expected a refusal on a well-mixed graph, got {other:?}"),
        }
        assert_eq!(esc.stats.jumps, 0);
    }

    /// Thirty basins the chain cycles inside, with one rare exit.
    fn trap_with_one_exit(archive_exit: bool) -> SuperbasinEscape {
        let mut esc = SuperbasinEscape::new();
        esc.min_nodes = 8;
        for _ in 0..2 {
            for i in 0..30usize {
                for j in 0..30usize {
                    if i != j {
                        esc.observe(i, Some(j), 1.0);
                    }
                }
            }
        }
        esc.observe(7, Some(500), 0.5);
        for i in 0..30usize {
            esc.keep(i, -1.0, Array1::from(vec![i as f64, 0.0, 0.0]).view());
        }
        if archive_exit {
            esc.keep(500, -9.0, Array1::from(vec![42.0, 0.0, 0.0]).view());
        }
        esc
    }

    #[test]
    fn jumps_to_a_stored_structure_outside_the_trap() {
        let mut esc = trap_with_one_exit(true);
        let mut rng = StdRng::seed_from_u64(3);
        let jump = esc.propose(0, &mut rng).expect("a trap with one exit");
        assert_eq!(jump.basin, 500);
        assert_eq!(jump.state, Array1::from(vec![42.0, 0.0, 0.0]));
        assert!((jump.probability - 1.0).abs() < 1e-12);
        assert!(
            jump.expected_hops > 30.0,
            "the diffusive crossing this replaces is {} hops",
            jump.expected_hops
        );
        assert!(jump.condition > 1.0, "condition {}", jump.condition);
        assert_eq!(jump.transient, 30);
        assert_eq!(esc.stats.jumps, 1);
        assert!(esc.stats.hops_saved > 30.0);
    }

    #[test]
    fn refuses_when_no_exit_has_a_structure() {
        let mut esc = trap_with_one_exit(false);
        let mut rng = StdRng::seed_from_u64(4);
        match esc.propose(0, &mut rng) {
            Err(Refusal::NoArchivedExit) => {}
            other => panic!("expected a refusal with no landing site, got {other:?}"),
        }
    }

    #[test]
    fn unresolved_exit_mass_preserves_the_probability_of_continuing_exploration() {
        let mut esc = trap_with_one_exit(true);
        esc.observe(7, None, 1.0);
        esc.observe(7, None, 0.5);
        let mut rng = StdRng::seed_from_u64(79);
        let mut reference_rng = rng.clone();
        let expected_jumps = (0..256)
            .filter(|_| reference_rng.random::<f64>() < 0.25)
            .count();
        let mut jumps = 0;
        for _ in 0..256 {
            if let Ok(jump) = esc.propose(0, &mut rng) {
                jumps += 1;
                assert_eq!(jump.basin, 500);
            }
        }
        assert_eq!(jumps, expected_jumps);
        assert_eq!(esc.stats.jumps + esc.stats.refusals, 256);
    }

    #[test]
    fn reweighting_removes_the_deposited_bias() {
        // Two basins, the second higher by one temperature unit. A bias sitting
        // on the first makes the chain accept the uphill hop far more often
        // than the unbiased chain would, and the reweighted estimate has to
        // recover the unbiased rate whatever the deposit does.
        let t = 1.0;
        let de = 1.0_f64;
        let unbiased = (-de / t).exp();
        let mut counts = HopCounts::new();
        for _ in 0..10_000 {
            counts.observe(0, Some(1), unbiased);
            counts.observe(1, Some(0), 1.0);
        }
        let chain = JumpChain::from_counts(&counts);
        let i = chain.position(0).expect("basin zero");
        // Holding time is hops over escaping mass, which for a Metropolis chain
        // accepting with probability a is exactly 1/a hops per jump.
        assert!(
            (chain.hold(i) - 1.0 / unbiased).abs() < 1e-9,
            "hold {} against 1/a = {}",
            chain.hold(i),
            1.0 / unbiased
        );
        // A deposit of one temperature unit on basin zero flattens the uphill
        // step, so the biased chain takes it every time. The diagnostic has to
        // show that gap rather than let it pass.
        let mut biased = counts.clone();
        for _ in 0..10_000 {
            biased.observe_accepted(0, 1);
            biased.observe_accepted(1, 0);
        }
        let d = biased.bias_distortion();
        assert!(
            d > 0.1,
            "the raw counts differ from the reweighted ones by {d:.3}, which the \
             diagnostic must show"
        );
    }

    #[test]
    fn the_default_basin_descriptor_is_already_orbit_invariant() {
        // The premise behind quotienting the graph by the symmetry orbit is
        // that the descriptor splits one minimum across its relabellings and
        // its point-group images, so a recurrent chain reads as a sweeping one.
        // For this crate's default keying that premise is false, and it is
        // false exactly rather than approximately.
        //
        // The descriptor is the sorted multiset of all pairwise distances.
        // Relabelling permutes which pair contributes which distance and leaves
        // the multiset alone; rotation, reflection and translation leave every
        // distance alone. So every member of an orbit has a bit-comparable
        // descriptor, and no orbit can be split into distinct basins.
        use crate::bias::{Fingerprint, SortedPairs};
        use rand::seq::SliceRandom;

        let n = 38usize;
        let mut rng = StdRng::seed_from_u64(20260806);
        // An irregular structure, so the test cannot pass by accidental
        // symmetry of a lattice.
        let mut x = Array1::<f64>::zeros(3 * n);
        for v in x.iter_mut() {
            *v = rng.random::<f64>() * 6.0 - 3.0;
        }
        let fp = SortedPairs { n_points: n };
        let base = fp.describe(x.view());

        // A random rotation from a QR-like construction, a reflection, a
        // translation, and a random relabelling of the points.
        let (a, b, c) = (
            rng.random::<f64>() * 6.283,
            rng.random::<f64>() * 6.283,
            rng.random::<f64>() * 6.283,
        );
        let (ca, sa) = (a.cos(), a.sin());
        let (cb, sb) = (b.cos(), b.sin());
        let (cc, sc) = (c.cos(), c.sin());
        let rot = [
            [ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
            [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
            [-sb, cb * sc, cb * cc],
        ];
        let shift = [1.7, -0.4, 2.9];
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);

        let mut y = Array1::<f64>::zeros(3 * n);
        for (new, old) in order.iter().enumerate() {
            let p = [x[3 * old], x[3 * old + 1], x[3 * old + 2]];
            for d in 0..3 {
                // The reflection is the sign on the first component, applied
                // before the rotation, so the image is improper.
                let q = [-p[0], p[1], p[2]];
                y[3 * new + d] = rot[d][0] * q[0] + rot[d][1] * q[1] + rot[d][2] * q[2] + shift[d];
            }
        }
        let image = fp.describe(y.view());

        let gap: f64 = base
            .iter()
            .zip(image.iter())
            .map(|(p, q)| (p - q) * (p - q))
            .sum::<f64>()
            .sqrt();
        // The merge radius the campaign runs at is 0.7, and the closest two
        // genuinely different 75-point minima come is 0.9212.
        assert!(
            gap < 1e-9,
            "an improper, rotated, translated, relabelled copy sits {gap:.3e} from the \
             original in descriptor space; the orbit is not collapsed"
        );

        // And the same descriptor still separates a genuinely different
        // structure, so the invariance is not bought by the descriptor being
        // blind.
        let mut z = x.clone();
        z[0] += 0.9;
        z[7] -= 0.8;
        let other = fp.describe(z.view());
        let far: f64 = base
            .iter()
            .zip(other.iter())
            .map(|(p, q)| (p - q) * (p - q))
            .sum::<f64>()
            .sqrt();
        assert!(
            far > 0.1,
            "the descriptor separates different structures: {far:.3e}"
        );
    }

    #[test]
    fn archive_keeps_the_deepest_and_drops_the_shallowest() {
        let mut esc = SuperbasinEscape::new();
        esc.archive_capacity = 3;
        for i in 0..6usize {
            esc.keep(i, -(i as f64), Array1::from(vec![i as f64]).view());
        }
        assert_eq!(esc.archived(), 3);
        for i in 3..6usize {
            assert!(esc.store.contains_key(&i), "basin {i} was dropped");
        }
        esc.keep(5, -99.0, Array1::from(vec![7.0]).view());
        assert_eq!(esc.store[&5].0, -99.0);
    }
}
