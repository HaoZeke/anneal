//! Replica exchange: the swap ratio each rung's own target implies, the
//! non-reversible sweep, and a ladder set by the run rather than by hand.
//!
//! Three things are separable here and are separated, because the campaign has
//! to be able to say which one paid.
//!
//! 1. *What the swap ratio is.* A rung that carries a bias is not sampling the
//!    canonical distribution at its temperature, and a swap acceptance built
//!    from raw energies exchanges between two distributions neither rung
//!    samples. The general form is written once, in terms of each rung's own
//!    log density, and every scheme below uses it.
//! 2. *Which pairs are offered.* Offering one uniformly chosen adjacent pair
//!    makes the rung index of a given configuration a random walk, which needs
//!    O(N^2) sweeps to cross N rungs. Offering all even pairs and all odd pairs
//!    on alternating sweeps, deterministically, makes it a persistent walk that
//!    crosses in O(N). This is the Syed, Bouchard-Cote, Deligiannidis and
//!    Doucet scheme.
//! 3. *Where the rungs sit.* A geometric ladder with a hand-set top is a guess
//!    about the energy scale. The ladder here is built from the cold chain's
//!    own energy fluctuation and then moved so that every neighbouring pair
//!    costs the same rejection.
//!
//! # The swap ratio
//!
//! Let rung `k` target `pi_k` on a common state space, and let the ladder hold
//! `x_k` at rung `k`. The product measure is `pi(x_0, ..., x_{N-1}) = prod_k
//! pi_k(x_k)`. A proposal that exchanges the occupants of rungs `i` and `j` is
//! its own inverse and has unit Jacobian, so the Metropolis-Hastings ratio is
//!
//! ```text
//!   a = [pi_i(x_j) pi_j(x_i)] / [pi_i(x_i) pi_j(x_j)]
//! ```
//!
//! and in logs, which is what [`swap_log_ratio`] returns,
//!
//! ```text
//!   ln a = [ln pi_i(x_j) - ln pi_i(x_i)] + [ln pi_j(x_i) - ln pi_j(x_j)].
//! ```
//!
//! Every term is one rung's own target evaluated at both configurations. No
//! quantity appears that belongs to neither rung. With
//! `ln pi_k(x) = -(E(x) + V_k(x)) / T_k` this is
//!
//! ```text
//!   ln a = (1/T_i)[U_i(x_i) - U_i(x_j)] + (1/T_j)[U_j(x_j) - U_j(x_i)],
//!          U_k(x) = E(x) + V_k(x)
//! ```
//!
//! which is the Piana and Laio bias-exchange acceptance. When the two rungs
//! carry the same bias it collapses to the canonical swap
//! `(1/T_i - 1/T_j)(U_i - U_j)` on the shared effective energy, and to the
//! textbook `(1/T_i - 1/T_j)(E_i - E_j)` only when the bias is also flat. A
//! bias the two rungs share still moves with the configuration and does not
//! cancel, which is the step that is easy to get wrong. The collapse is a test
//! ([`tests::equal_biases_collapse_to_the_canonical_swap`]),
//! not a remark, and so is the failure mode: exchanging on raw energies between
//! rungs whose biases differ leaves a measurably wrong stationary distribution
//! ([`tests::raw_energy_exchange_between_biased_rungs_is_not_the_product_measure`]).
//!
//! # Why a deterministic sweep moves configurations faster
//!
//! Track one configuration by a tag and watch which rung holds it. Under
//! [`SwapScheme::RandomPair`] the tag takes an unbiased step only when its own
//! pair is the one drawn, so its rung index is a random walk on `0..N` that is
//! offered a step once every `N` sweeps: the expected number of sweeps for that
//! tag to reach the hottest rung and return grows as `N^3`. Offering a whole
//! parity class at once removes one factor and leaves `N^2`.
//!
//! Under [`SwapScheme::DeterministicEvenOdd`] the parity of the offered pairs
//! alternates with the sweep index. A tag that moved up on an even sweep is
//! offered the pair above it on the next odd sweep, so it keeps moving up until
//! a rejection turns it around. Its index is a persistent walk with mean free
//! path `1/r` rungs, `r` being the local rejection rate. Writing `Lambda = sum_k
//! r_k` for the total barrier, the expected sweeps for one traverse of the
//! ladder is, for a telegraph process of unit speed and mean free path `N /
//! Lambda`,
//!
//! ```text
//!   E[traverse] ~ N + N^2 / (2 * (N / Lambda)) = N (1 + Lambda / 2)
//! ```
//!
//! so with `Lambda` a property of the problem and the ladder adapted to spread
//! it evenly, the time for one tag to cross and return is linear in `N`.
//!
//! Two round-trip numbers follow and they are not the same number.
//! [`Ladder::round_trip_rate`] counts trips per sweep aggregated over every tag
//! on the ladder, which is what a search consumes and which stays bounded in
//! `N` under the deterministic sweep because all `N` tags travel at once.
//! [`Ladder::mean_round_trip_time`] divides that back out to one tag, and it is
//! the quantity whose growth separates the schemes. Measured on the idealised
//! index process at a fixed barrier of 3, in sweeps per tagged round trip:
//!
//! | rungs | deterministic | stochastic | one drawn pair |
//! |-------|---------------|------------|----------------|
//! | 8     | 99            | 197        | 675            |
//! | 16    | 153           | 568        | 4241           |
//! | 32    | 281           | 2175       | 36863          |
//! | 64    | 560           | 8619       | 273792         |
//!
//! which fits `N^0.84`, `N^1.83` and `N^2.91`. This is the scaling the tests
//! measure rather than assert; see
//! [`tests::round_trip_time_is_linear_for_the_deterministic_sweep_and_quadratic_otherwise`].
//!
//! The same measurement inside the cluster driver, where the rungs share one
//! budget instead of each having their own, says the scaling is real and says
//! why it does not rescue the method. LJ38 at 4e5 charged evaluations, a swap
//! every ten hops, 24 seeds, round trips completed per run:
//!
//! | rungs | sweeps the budget buys | stochastic | deterministic |
//! |-------|------------------------|------------|---------------|
//! | 4     | 270                    | 29.0       | 34.0          |
//! | 8     | 139                    | 7.1        | 17.5          |
//! | 16    | 69                     | 0.4        | 4.7           |
//! | 32    | 35                     | 0.0        | 0.0           |
//!
//! The deterministic sweep is ahead by a factor of 2.5 at eight rungs and 12 at
//! sixteen, which is the advantage the scaling predicts. It is fought by the
//! first column: a run of fixed cost divides its hops among the rungs, so the
//! sweeps it can afford fall as `1/N` while the per-tag round-trip time rises
//! as `N`. Round trips per run therefore fall as `1/N^2` for the deterministic
//! sweep and `1/N^3` for the reversible one, and by 32 rungs neither completes
//! any. A longer ladder is not available to a fixed-budget optimiser whatever
//! the sweep does.
//!
//! [`SwapScheme::StochasticEvenOdd`] offers the same pairs but draws the parity
//! from a coin, which makes the sweep reversible and the index process
//! diffusive again. It is the honest reversible control for the comparison: it
//! differs from the non-reversible scheme in one coin flip and in nothing else.
//!
//! # Where the rungs sit
//!
//! Two mechanisms, in this order.
//!
//! *The first ladder* comes from the cold chain's energy fluctuation. For
//! neighbouring inverse temperatures separated by `d` and energies that are
//! near-Gaussian with standard deviation `s` at both rungs, the difference
//! `d * (E_i - E_j)` is Gaussian with variance `2 d^2 s^2` and, by the
//! fluctuation identity that pins its mean to half its variance, the mean swap
//! acceptance is `2 Phi(-d s / sqrt(2))`. Inverting for a target acceptance
//! `a*` gives [`beta_step_for_acceptance`], `d = -sqrt(2) Phi^{-1}(a*/2) / s`,
//! which is the whole ladder once `s` is measured. Nothing here is a multiple
//! of the cold temperature.
//!
//! *Afterwards* the ladder is moved by the communication-barrier estimator of
//! the same literature. Accumulate the mean acceptance probability of each pair
//! (the probability, not the accept indicator: it is the conditional
//! expectation and has the smaller variance), form the local rejection rates
//! `r_k = 1 - alpha_k` and the cumulative barrier `Lambda_k = sum_{m<k} r_m`.
//! `Lambda` is monotone in the inverse temperature, so the new rungs are the
//! solutions of `Lambda(beta'_k) = (k / (N-1)) Lambda`, taken by inverse linear
//! interpolation on the measured knots. Equal barriers between neighbours means
//! equal rejection, which is the same condition as equal round-trip
//! contribution from every rung. [`Ladder::equalise`] is that step.
//!
//! Moving the rungs makes the chain adaptive, and the adaptation here is not
//! diminishing: the windows are a fixed length, so the ladder keeps moving for
//! as long as the run lasts. Each fixed ladder is stationary for its own
//! product measure, which is what the tests check; the sequence of them is not,
//! and no claim is made that it is. What this component is scored on is whether
//! the search reaches the published minimum, not on the quality of the samples
//! it produces along the way. A run that needs exact stationarity should stop
//! adapting, which is
//! [`crate::methods::cluster_hopping::LadderMode::Reversible`] with the rungs
//! fixed.
//!
//! The hot endpoint is the one thing the barrier estimator cannot place, since
//! it is the edge of the measured range. [`Ladder::retune_top`] moves it by a
//! multiplicative controller on the mean rejection, so a ladder whose pairs
//! reject more than the target is pulled in and one that rejects less is
//! stretched out. The target is a rejection rate, not a temperature, so it does
//! not carry units from one potential to another.
//!
//! # Composition with the within-temperature move
//!
//! [`ReplicaMove`] is the plug. A rung owns a `Chain`, which is whatever the
//! move adapts per temperature (a step size, a metric, a bias), and the ladder
//! owns a `Point`, which is what a swap exchanges. Swaps move points between
//! rungs and never move chains: a step size adapted at `T_k` belongs to `T_k`
//! and not to the configuration that happens to be sitting there.
//!
//! [`MetropolisWalk`] is the default and is what the shipped move set reduces
//! to. A Hamiltonian chain with its own dual-averaged step size and its own
//! metric satisfies the same trait with `Chain` being the per-rung adaptation
//! state. The shape it takes, for a chain object owning a dual-averaged step
//! size and a metric estimate and proposing through a charged trajectory:
//!
//! ```text
//! struct HopReplica<'a> {
//!     cfg: &'a HopConfig,                 // shared, immutable
//!     ledger: &'a mut Ledger,             // the trajectory is charged
//!     eval: &'a mut dyn FnMut(&mut Ledger, ArrayView1<f64>)
//!                         -> Option<(f64, Array1<f64>)>,
//! }
//!
//! impl ReplicaMove for HopReplica<'_> {
//!     type Point = Array1<f64>;           // moved by a swap
//!     type Chain = HopChain;              // one per rung, never moved
//!     fn advance<R>(&mut self, rung, temperature, point, chain, rng) -> bool {
//!         chain.propose(self.cfg, self.ledger, point.view(), .., self.eval, rng)
//!     }
//! }
//! ```
//!
//! The split is the whole contract: `Point` is exchanged and `Chain` is not,
//! because a step size and a metric are adapted against a temperature and stay
//! with the rung when the configuration standing there leaves.
//!
//! The interface exists; the combination is measured and is not worth its cost.
//! Both mechanisms are paid out of the same quantity, hops, and neither returns
//! anything on whether the search finds the minimum. The Hamiltonian numbers
//! here are measured on the `feat/hmc-hop` arms, whose sampler is
//! `src/hmc/hop.rs` and whose per-rung chain is the `HopChain` above. A
//! Hamiltonian rung costs 76 charged evaluations per hop against 41 for the
//! shipped move library, and a run of 4e5 takes 5216 hops against 9781. At a
//! swap every ten hops over four rungs that is 130 sweeps against 245, and 130
//! is already down where sixteen displacement rungs sit, at 4.7 round trips per
//! run. Against that cost the exchange is worth 0.10 in solve rate at Fisher
//! 0.13, and the Hamiltonian proposal loses to the shipped library at Fisher
//! 0.039. Two effects that are individually absent do not have a product worth
//! a budget.
//!
//! Nor would a ladder recover the one thing a trajectory sampler with a U-turn
//! criterion is built to give. The depth cap binds on 86.8 per cent of
//! proposals under an identity metric and 78.8 per cent under a model-Hessian
//! one even when the cap is raised to ten, so the rungs would carry effectively
//! fixed-length trajectories and there is no variable trajectory length to
//! trade against the sweep count.

use rand::Rng;

// ---------------------------------------------------------------------------
// The target and the swap ratio
// ---------------------------------------------------------------------------

/// A ladder of targets on one state space.
///
/// `log_density` returns `ln pi_k(x)` up to a constant that may depend on the
/// rung but not on the configuration, which is all a swap ratio needs: the
/// per-rung constants cancel between the two orderings.
pub trait ReplicaTarget {
    /// The configuration a swap exchanges.
    type Point;
    /// Per-rung state the swap leaves in place, such as an accumulated bias.
    type Chain;

    /// `ln pi_k(x)`, evaluated with rung `k`'s own chain.
    fn log_density(
        &self,
        rung: usize,
        temperature: f64,
        point: &Self::Point,
        chain: &Self::Chain,
    ) -> f64;
}

/// The log Metropolis-Hastings ratio for exchanging the occupants of two rungs.
///
/// Both terms are one rung's own target at both configurations. Returns
/// `f64::NEG_INFINITY` when either target vanishes, which is a refusal rather
/// than an error.
pub fn swap_log_ratio<S: ReplicaTarget>(
    target: &S,
    temps: &[f64],
    i: usize,
    j: usize,
    points: &[S::Point],
    chains: &[S::Chain],
) -> f64 {
    let li_xi = target.log_density(i, temps[i], &points[i], &chains[i]);
    let li_xj = target.log_density(i, temps[i], &points[j], &chains[i]);
    let lj_xj = target.log_density(j, temps[j], &points[j], &chains[j]);
    let lj_xi = target.log_density(j, temps[j], &points[i], &chains[j]);
    (li_xj - li_xi) + (lj_xi - lj_xj)
}

/// The acceptance probability `min(1, a)` for the same exchange.
pub fn swap_probability<S: ReplicaTarget>(
    target: &S,
    temps: &[f64],
    i: usize,
    j: usize,
    points: &[S::Point],
    chains: &[S::Chain],
) -> f64 {
    let l = swap_log_ratio(target, temps, i, j, points, chains);
    if l >= 0.0 { 1.0 } else { l.exp() }
}

/// The same ratio written for the common case: one energy, one bias per rung.
///
/// `u_i_at_i` is `E(x_i) + V_i(x_i)` and so on. Kept separate from
/// [`swap_log_ratio`] so a caller holding four numbers does not have to build a
/// target to use the correct algebra.
pub fn biased_swap_log_ratio(
    t_i: f64,
    t_j: f64,
    u_i_at_i: f64,
    u_i_at_j: f64,
    u_j_at_j: f64,
    u_j_at_i: f64,
) -> f64 {
    (u_i_at_i - u_i_at_j) / t_i.max(f64::MIN_POSITIVE)
        + (u_j_at_j - u_j_at_i) / t_j.max(f64::MIN_POSITIVE)
}

// ---------------------------------------------------------------------------
// The within-temperature move
// ---------------------------------------------------------------------------

/// One rung's move between swaps.
///
/// The split of `Point` and `Chain` is the interface contract: a swap moves
/// points and never moves chains. Whatever a move adapts against a temperature
/// (a step size, a mass matrix, a deposited bias) lives in `Chain` and stays
/// with the rung.
pub trait ReplicaMove {
    /// The configuration, exchanged by swaps.
    type Point;
    /// The per-rung adaptation state, which swaps leave alone.
    type Chain;

    /// Advances rung `rung` by one move at `temperature`. Returns whether the
    /// point moved, which the ladder reports but does not act on.
    fn advance<R: Rng + ?Sized>(
        &mut self,
        rung: usize,
        temperature: f64,
        point: &mut Self::Point,
        chain: &mut Self::Chain,
        rng: &mut R,
    ) -> bool;
}

// ---------------------------------------------------------------------------
// Which pairs a sweep offers
// ---------------------------------------------------------------------------

/// Which neighbouring pairs a swap sweep offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SwapScheme {
    /// One uniformly drawn adjacent pair per sweep. Reversible, and the index
    /// process is a random walk.
    #[default]
    RandomPair,
    /// All even pairs or all odd pairs, parity drawn from a fair coin.
    /// Reversible, and the index process is a random walk with a longer step.
    StochasticEvenOdd,
    /// All even pairs or all odd pairs, parity alternating with the sweep
    /// index. Not reversible; leaves the product measure invariant all the
    /// same, because each sweep is a composition of Metropolis moves against
    /// it. The index process is a persistent walk.
    DeterministicEvenOdd,
}

// ---------------------------------------------------------------------------
// The index process
// ---------------------------------------------------------------------------

/// Where each tagged configuration sits on the ladder, and how often one has
/// gone from the coldest rung to the hottest and back.
///
/// This is the instrument the scaling claim is made on. Swap counts do not
/// distinguish a ladder that transports configurations from one that shuffles
/// two neighbours forever, and the campaign has already recorded a ladder that
/// swapped busily and never stratified.
#[derive(Clone, Debug)]
pub struct IndexProcess {
    /// `tag_at[k]` is the tag currently held by rung `k`.
    tag_at: Vec<usize>,
    /// `+1` once a tag has touched the coldest rung, `-1` once it has touched
    /// the hottest, `0` before either.
    label: Vec<i8>,
    round_trips: usize,
    hot_arrivals: usize,
}

impl IndexProcess {
    /// A fresh ladder: tag `k` starts at rung `k`.
    pub fn new(n_rungs: usize) -> Self {
        let mut label = vec![0i8; n_rungs];
        if n_rungs > 0 {
            label[0] = 1;
        }
        if n_rungs > 1 {
            label[n_rungs - 1] = -1;
        }
        Self {
            tag_at: (0..n_rungs).collect(),
            label,
            round_trips: 0,
            hot_arrivals: 0,
        }
    }

    /// Applies an accepted exchange between two rungs.
    pub fn swap(&mut self, i: usize, j: usize) {
        self.tag_at.swap(i, j);
    }

    /// Records arrivals at the two ends. Call once per sweep, after the
    /// sweep's accepted swaps have been applied.
    pub fn observe_ends(&mut self) {
        let n = self.tag_at.len();
        if n < 2 {
            return;
        }
        let cold = self.tag_at[0];
        if self.label[cold] == -1 {
            self.round_trips += 1;
        }
        self.label[cold] = 1;
        let hot = self.tag_at[n - 1];
        if self.label[hot] == 1 {
            self.hot_arrivals += 1;
        }
        self.label[hot] = -1;
    }

    /// Completed cold-to-hot-to-cold trips.
    pub fn round_trips(&self) -> usize {
        self.round_trips
    }

    /// Arrivals at the hot end by a tag that last touched the cold end. Half a
    /// round trip, reported because a ladder can carry configurations up and
    /// fail to bring them back.
    pub fn hot_arrivals(&self) -> usize {
        self.hot_arrivals
    }

    /// The tag at each rung.
    pub fn tags(&self) -> &[usize] {
        &self.tag_at
    }
}

// ---------------------------------------------------------------------------
// The ladder
// ---------------------------------------------------------------------------

/// A temperature ladder, its sweep scheme, its pair statistics and its index
/// process.
///
/// Rung 0 is the coldest. Inverse temperatures decrease with the rung index.
#[derive(Clone, Debug)]
pub struct Ladder {
    betas: Vec<f64>,
    scheme: SwapScheme,
    sweep: usize,
    /// Per pair `k`, meaning the pair `(k, k+1)`.
    offered: Vec<u64>,
    accepted: Vec<u64>,
    alpha_sum: Vec<f64>,
    /// Kept across an adaptation so a run reports its whole history.
    total_offered: u64,
    total_accepted: u64,
    index: IndexProcess,
    clamped: bool,
}

/// The hottest rung a fluctuation ladder will place, as a fraction of the cold
/// inverse temperature. A ladder is not allowed to reach infinite temperature,
/// where nothing the hot chain holds carries information about the cold one.
pub const MIN_HOT_BETA_FRACTION: f64 = 0.05;

impl Ladder {
    /// Builds from temperatures, coldest first.
    pub fn from_temperatures(temps: &[f64], scheme: SwapScheme) -> Self {
        assert!(temps.len() >= 2, "a ladder needs at least two rungs");
        assert!(
            temps.windows(2).all(|w| w[1] > w[0]),
            "temperatures must increase with the rung index"
        );
        let n = temps.len();
        Self {
            betas: temps.iter().map(|t| 1.0 / t).collect(),
            scheme,
            sweep: 0,
            offered: vec![0; n - 1],
            accepted: vec![0; n - 1],
            alpha_sum: vec![0.0; n - 1],
            total_offered: 0,
            total_accepted: 0,
            index: IndexProcess::new(n),
            clamped: false,
        }
    }

    /// The ladder the cold chain's own energy fluctuation implies.
    ///
    /// `sigma_e` is the standard deviation of the energies the coldest chain
    /// visits. Rungs are equally spaced in inverse temperature at
    /// [`beta_step_for_acceptance`], so the top is wherever `n_rungs - 1` of
    /// those steps land, and no multiple of the cold temperature enters.
    ///
    /// The step is clamped so the hot rung stays at
    /// [`MIN_HOT_BETA_FRACTION`] of the cold inverse temperature. The clamp
    /// binds whenever the energy spread is large against the cold temperature,
    /// which on a quenched cluster surface it is: at LJ38 the quenched
    /// energies a chain at 0.8 visits have a spread of order one, and a
    /// 0.23-acceptance step of 1.7 per rung would take a four-rung ladder past
    /// infinite temperature. When it binds the first ladder is the widest one
    /// available and the barrier estimator does the rest;
    /// [`Ladder::clamped_at_construction`] reports which case fired.
    pub fn from_fluctuation(
        t_cold: f64,
        sigma_e: f64,
        n_rungs: usize,
        target_accept: f64,
        scheme: SwapScheme,
    ) -> Self {
        assert!(n_rungs >= 2, "a ladder needs at least two rungs");
        assert!(t_cold > 0.0, "the cold temperature must be positive");
        let b0 = 1.0 / t_cold;
        let want = beta_step_for_acceptance(sigma_e, target_accept);
        let widest = b0 * (1.0 - MIN_HOT_BETA_FRACTION) / (n_rungs - 1) as f64;
        let clamped = !(want > 0.0) || want > widest;
        let step = if clamped { widest } else { want };
        let temps: Vec<f64> = (0..n_rungs).map(|k| 1.0 / (b0 - step * k as f64)).collect();
        let mut ladder = Self::from_temperatures(&temps, scheme);
        ladder.clamped = clamped;
        ladder
    }

    /// Whether the fluctuation step was clamped by
    /// [`MIN_HOT_BETA_FRACTION`] when the ladder was built.
    pub fn clamped_at_construction(&self) -> bool {
        self.clamped
    }

    /// Temperatures, coldest first.
    pub fn temperatures(&self) -> Vec<f64> {
        self.betas.iter().map(|b| 1.0 / b).collect()
    }

    /// Inverse temperatures, coldest first.
    pub fn betas(&self) -> &[f64] {
        &self.betas
    }

    /// Rungs on the ladder.
    pub fn len(&self) -> usize {
        self.betas.len()
    }

    /// Whether the ladder has no rungs. Never true after construction.
    pub fn is_empty(&self) -> bool {
        self.betas.is_empty()
    }

    /// Sweeps offered so far.
    pub fn sweeps(&self) -> usize {
        self.sweep
    }

    /// The index process.
    pub fn index(&self) -> &IndexProcess {
        &self.index
    }

    /// Round trips per sweep, aggregated over every tag on the ladder.
    ///
    /// This is the operational number: it is how often the ladder delivers a
    /// configuration that has been to the hot end back to the coldest rung, and
    /// it is what a search cares about. Under a deterministic even-odd sweep it
    /// stays bounded as rungs are added, because every rung carries a tag and
    /// all of them are in flight at once.
    pub fn round_trip_rate(&self) -> f64 {
        if self.sweep == 0 {
            0.0
        } else {
            self.index.round_trips() as f64 / self.sweep as f64
        }
    }

    /// Sweeps for one *tagged* configuration to cross the ladder and return.
    ///
    /// `n_rungs / round_trip_rate`, since the aggregate counts every tag. This
    /// is the quantity whose growth in the number of rungs separates the
    /// schemes: linear for the deterministic sweep, quadratic for a reversible
    /// one. Infinite when nothing has completed a trip.
    pub fn mean_round_trip_time(&self) -> f64 {
        let trips = self.index.round_trips();
        if trips == 0 {
            f64::INFINITY
        } else {
            self.len() as f64 * self.sweep as f64 / trips as f64
        }
    }

    /// Swaps offered and accepted since construction.
    pub fn swap_counts(&self) -> (u64, u64) {
        (self.total_offered, self.total_accepted)
    }

    /// Which pairs the next sweep offers, as indices `k` meaning `(k, k+1)`.
    pub fn next_pairs<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<usize> {
        let np = self.betas.len() - 1;
        match self.scheme {
            SwapScheme::RandomPair => vec![rng.random_range(0..np)],
            SwapScheme::StochasticEvenOdd => {
                let parity = usize::from(rng.random::<bool>());
                (0..np).filter(|k| k % 2 == parity).collect()
            }
            SwapScheme::DeterministicEvenOdd => {
                let parity = self.sweep % 2;
                (0..np).filter(|k| k % 2 == parity).collect()
            }
        }
    }

    /// Offers one sweep with acceptance probabilities supplied by the caller,
    /// and returns the pairs that accepted.
    ///
    /// The caller applies the exchange to its own configurations; the ladder
    /// applies it to the index process and to the pair statistics. Exposed
    /// because it is also the idealised index process on its own: hand it a
    /// constant rejection rate and it is the persistent-walk model the round
    /// trip scaling is derived from, with no target and no configurations.
    pub fn offer<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        mut alpha: impl FnMut(usize) -> f64,
    ) -> Vec<usize> {
        let pairs = self.next_pairs(rng);
        let mut taken = Vec::with_capacity(pairs.len());
        for k in pairs {
            let a = alpha(k).clamp(0.0, 1.0);
            self.offered[k] += 1;
            self.total_offered += 1;
            self.alpha_sum[k] += a;
            if rng.random::<f64>() < a {
                self.accepted[k] += 1;
                self.total_accepted += 1;
                self.index.swap(k, k + 1);
                taken.push(k);
            }
        }
        self.index.observe_ends();
        self.sweep += 1;
        taken
    }

    /// Offers one sweep against a target, exchanging the points it accepts.
    pub fn exchange<S: ReplicaTarget, R: Rng + ?Sized>(
        &mut self,
        target: &S,
        points: &mut [S::Point],
        chains: &[S::Chain],
        rng: &mut R,
    ) -> Vec<usize> {
        let temps = self.temperatures();
        // The acceptance of every offered pair is computed before any is
        // applied, which is safe because a sweep's pairs are disjoint.
        let taken = {
            let points_ref: &[S::Point] = points;
            self.offer(rng, |k| {
                swap_probability(target, &temps, k, k + 1, points_ref, chains)
            })
        };
        for &k in &taken {
            points.swap(k, k + 1);
        }
        taken
    }

    /// Mean acceptance probability per pair since the last reset, `None` for a
    /// pair that has not been offered.
    pub fn pair_acceptance(&self) -> Vec<Option<f64>> {
        self.offered
            .iter()
            .zip(&self.alpha_sum)
            .map(|(&n, &s)| if n == 0 { None } else { Some(s / n as f64) })
            .collect()
    }

    /// Local rejection rates `r_k = 1 - alpha_k`, with unobserved pairs
    /// carrying the mean of the observed ones.
    pub fn rejection_rates(&self) -> Vec<f64> {
        let obs: Vec<f64> = self
            .pair_acceptance()
            .into_iter()
            .flatten()
            .map(|a| 1.0 - a)
            .collect();
        let fill = if obs.is_empty() {
            0.5
        } else {
            obs.iter().sum::<f64>() / obs.len() as f64
        };
        self.pair_acceptance()
            .into_iter()
            .map(|a| a.map(|a| 1.0 - a).unwrap_or(fill))
            .collect()
    }

    /// The communication barrier, `sum_k r_k`.
    ///
    /// The number of rejections a configuration pays to cross the ladder once,
    /// and the quantity the adapted ladder spreads evenly.
    pub fn barrier(&self) -> f64 {
        self.rejection_rates().iter().sum()
    }

    /// Clears the pair statistics, leaving the index process and the rungs.
    pub fn reset_statistics(&mut self) {
        self.offered.iter_mut().for_each(|v| *v = 0);
        self.accepted.iter_mut().for_each(|v| *v = 0);
        self.alpha_sum.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Moves the interior rungs so every neighbouring pair carries the same
    /// share of the barrier. Endpoints stay put.
    ///
    /// The measured knots are `(Lambda_k, beta_k)` with `Lambda_k` the
    /// cumulative rejection up to rung `k`, monotone increasing as `beta`
    /// decreases. The new rungs are the inverse-interpolated solutions of
    /// `Lambda(beta) = (k / (N-1)) Lambda`. Statistics are cleared, since they
    /// were measured on a ladder that no longer exists.
    ///
    /// Returns the barrier the move was computed from.
    pub fn equalise(&mut self) -> f64 {
        let n = self.betas.len();
        let r = self.rejection_rates();
        let mut cum = vec![0.0; n];
        for k in 1..n {
            cum[k] = cum[k - 1] + r[k - 1];
        }
        let total = cum[n - 1];
        if !(total > 1e-12) {
            // Every pair accepts always: the ladder is narrower than it needs
            // to be and there is no barrier profile to equalise against.
            self.reset_statistics();
            return total;
        }
        let mut fresh = self.betas.clone();
        for k in 1..n - 1 {
            let want = total * k as f64 / (n - 1) as f64;
            fresh[k] = invert_monotone(&cum, &self.betas, want);
        }
        // A degenerate profile can produce a non-monotone solve; refuse it
        // rather than ship a ladder whose rungs are out of order.
        if fresh.windows(2).all(|w| w[1] < w[0]) {
            self.betas = fresh;
        }
        self.reset_statistics();
        total
    }

    /// Moves the hot endpoint so the mean rejection per pair approaches
    /// `target_reject`, then re-equalises the interior.
    ///
    /// The barrier estimator cannot place the endpoint, because the endpoint is
    /// the edge of what was measured. A multiplicative controller on the log
    /// inverse temperature does it instead: a ladder rejecting more than the
    /// target is too wide and is pulled in, one rejecting less is stretched.
    /// `gain` is the controller's step; 0.5 moves the top by about 20 per cent
    /// when the mean rejection misses the target by 0.4.
    pub fn retune_top(&mut self, target_reject: f64, gain: f64) {
        let n = self.betas.len();
        let r = self.rejection_rates();
        let mean = r.iter().sum::<f64>() / r.len() as f64;
        let factor = (gain * (mean - target_reject)).exp();
        let b_cold = self.betas[0];
        let b_top = (self.betas[n - 1] * factor).clamp(1e-6 * b_cold, 0.9 * b_cold);
        self.betas[n - 1] = b_top;
        // The interior is re-spaced geometrically inside the new range before
        // the barrier profile is applied again, so a large endpoint move cannot
        // leave rungs stacked at one end.
        for (k, b) in self.betas.iter_mut().enumerate().take(n - 1).skip(1) {
            let f = k as f64 / (n - 1) as f64;
            *b = b_cold * (b_top / b_cold).powf(f);
        }
        self.reset_statistics();
    }
}

/// Inverse linear interpolation of a monotone increasing table.
///
/// `x` increases, `y` decreases, and `want` is clamped into the table's range.
fn invert_monotone(x: &[f64], y: &[f64], want: f64) -> f64 {
    let n = x.len();
    if want <= x[0] {
        return y[0];
    }
    if want >= x[n - 1] {
        return y[n - 1];
    }
    for k in 1..n {
        if want <= x[k] {
            let span = x[k] - x[k - 1];
            let f = if span > 1e-15 {
                (want - x[k - 1]) / span
            } else {
                0.0
            };
            return y[k - 1] + f * (y[k] - y[k - 1]);
        }
    }
    y[n - 1]
}

// ---------------------------------------------------------------------------
// The Gaussian ladder step
// ---------------------------------------------------------------------------

/// The inverse-temperature step that a Gaussian energy of spread `sigma_e`
/// accepts at rate `target_accept`.
///
/// From `mean acceptance = 2 Phi(-d sigma / sqrt(2))`, so
/// `d = -sqrt(2) Phi^{-1}(a*/2) / sigma`. At the classical target of 0.23 the
/// numerator is 1.6976.
pub fn beta_step_for_acceptance(sigma_e: f64, target_accept: f64) -> f64 {
    let a = target_accept.clamp(1e-6, 0.999);
    let z = inverse_normal_cdf(a / 2.0);
    let num = -std::f64::consts::SQRT_2 * z;
    if sigma_e > 1e-12 { num / sigma_e } else { 0.0 }
}

/// The classical swap-acceptance target for a temperature ladder, from Kone
/// and Kofke's analysis of the entropy production per swap.
pub const TARGET_SWAP_ACCEPT: f64 = 0.23;

/// Standard normal quantile, Acklam's rational approximation. Absolute error
/// below 1.2e-9 over the open unit interval.
pub fn inverse_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

// ---------------------------------------------------------------------------
// The default within-temperature move
// ---------------------------------------------------------------------------

/// Metropolis random walk on a finite state space, the move the shipped set
/// reduces to when nothing else is switched on.
///
/// Present so the ladder has a default and so the tests have a kernel whose
/// stationary distribution is known exactly. It owns the energy, since the
/// energy is the mover's and not the ladder's; the ladder holds only what a
/// swap needs.
#[derive(Clone, Debug)]
pub struct MetropolisWalk {
    /// The state space this walk moves on.
    pub system: DiscreteSystem,
}

/// A rung's chain under [`MetropolisWalk`]: the bias it carries, indexed the
/// same way as the state space.
#[derive(Clone, Debug, Default)]
pub struct BiasVector {
    /// Per-state bias, added to the energy inside this rung's target.
    pub v: Vec<f64>,
}

impl BiasVector {
    /// A rung with no bias over `n` states.
    pub fn flat(n: usize) -> Self {
        Self { v: vec![0.0; n] }
    }

    /// The bias at a state.
    pub fn at(&self, s: usize) -> f64 {
        self.v.get(s).copied().unwrap_or(0.0)
    }
}

/// A finite-state tempered target: one energy vector, one bias per rung.
#[derive(Clone, Debug)]
pub struct DiscreteSystem {
    /// Energy of each state.
    pub energy: Vec<f64>,
}

impl ReplicaTarget for DiscreteSystem {
    type Point = usize;
    type Chain = BiasVector;

    fn log_density(
        &self,
        _rung: usize,
        temperature: f64,
        point: &usize,
        chain: &BiasVector,
    ) -> f64 {
        -(self.energy[*point] + chain.at(*point)) / temperature
    }
}

impl ReplicaMove for MetropolisWalk {
    type Point = usize;
    type Chain = BiasVector;

    fn advance<R: Rng + ?Sized>(
        &mut self,
        _rung: usize,
        temperature: f64,
        point: &mut usize,
        bias: &mut BiasVector,
        rng: &mut R,
    ) -> bool {
        let sys = &self.system;
        let n = sys.energy.len();
        if n < 2 {
            return false;
        }
        let mut y = rng.random_range(0..n);
        while y == *point {
            y = rng.random_range(0..n);
        }
        let du = (sys.energy[y] + bias.at(y)) - (sys.energy[*point] + bias.at(*point));
        let p = if du <= 0.0 {
            1.0
        } else {
            (-du / temperature).exp()
        };
        if rng.random::<f64>() < p {
            *point = y;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn boltzmann(energy: &[f64], bias: &BiasVector, t: f64) -> Vec<f64> {
        let w: Vec<f64> = energy
            .iter()
            .enumerate()
            .map(|(s, e)| (-(e + bias.at(s)) / t).exp())
            .collect();
        let z: f64 = w.iter().sum();
        w.into_iter().map(|v| v / z).collect()
    }

    // -----------------------------------------------------------------
    // 1. The swap ratio
    // -----------------------------------------------------------------

    /// With equal biases the general ratio has to be the canonical one, since
    /// that is the claim that lets a ladder without bias use either.
    #[test]
    fn equal_biases_collapse_to_the_canonical_swap() {
        let sys = DiscreteSystem {
            energy: vec![0.0, 1.0, 2.5, -0.5],
        };
        let shared = BiasVector {
            v: vec![0.3, -1.2, 0.7, 2.0],
        };
        let temps = [0.5, 2.0];
        let chains = [shared.clone(), shared.clone()];
        for (i, j) in [(0usize, 1usize), (2, 3), (1, 3)] {
            let points = [i, j];
            let got = swap_log_ratio(&sys, &temps, 0, 1, &points, &chains);
            // Equal biases means both rungs sample the same effective energy
            // U = E + V, so the general ratio has to be the canonical swap on
            // U. It is not the canonical swap on E: a bias the two rungs share
            // still moves with the configuration and does not cancel.
            let u = |s: usize| sys.energy[s] + shared.at(s);
            let canonical = (1.0 / temps[0] - 1.0 / temps[1]) * (u(i) - u(j));
            assert!(
                (got - canonical).abs() < 1e-12,
                "biased form {got} against canonical {canonical} at equal biases"
            );
        }
    }

    /// The four-term form, against numbers worked out by hand, so a sign slip
    /// in the algebra has somewhere to fail.
    #[test]
    fn the_biased_ratio_is_the_four_term_expression() {
        // Rung 0 at T=0.5 with bias v0, rung 1 at T=2.0 with bias v1.
        let sys = DiscreteSystem {
            energy: vec![-3.0, -1.0],
        };
        let v0 = BiasVector { v: vec![2.0, 0.0] };
        let v1 = BiasVector { v: vec![0.0, 1.0] };
        let temps = [0.5, 2.0];
        let chains = [v0, v1];
        let points = [0usize, 1usize];
        // U_0(x_0) = -3 + 2 = -1 ; U_0(x_1) = -1 + 0 = -1
        // U_1(x_1) = -1 + 1 =  0 ; U_1(x_0) = -3 + 0 = -3
        // ln a = (-1 - -1)/0.5 + (0 - -3)/2.0 = 0 + 1.5
        let got = swap_log_ratio(&sys, &temps, 0, 1, &points, &chains);
        assert!((got - 1.5).abs() < 1e-12, "log ratio {got} against 1.5");
        let direct = biased_swap_log_ratio(0.5, 2.0, -1.0, -1.0, 0.0, -3.0);
        assert!((direct - 1.5).abs() < 1e-12, "direct form {direct}");
        // And the raw-energy rule, which is what the fault used, gives a
        // different number on the same pair: (1/0.5 - 1/2)(-3 - -1) = -3.
        let raw = (1.0 / 0.5 - 1.0 / 2.0) * (sys.energy[0] - sys.energy[1]);
        assert!((raw - -3.0).abs() < 1e-12);
        assert!(
            (raw - got).abs() > 1.0,
            "the two rules must not coincide here"
        );
    }

    /// The four-number form the driver calls and the target form the tests
    /// simulate have to be the same function, or the thing that is checked is
    /// not the thing that runs.
    #[test]
    fn the_driver_form_and_the_target_form_agree() {
        let sys = DiscreteSystem {
            energy: vec![-2.0, 0.5, 3.0],
        };
        let v0 = BiasVector {
            v: vec![0.0, 1.3, -0.7],
        };
        let v1 = BiasVector {
            v: vec![2.2, -0.1, 0.4],
        };
        let temps = [0.7, 2.3];
        let chains = [v0.clone(), v1.clone()];
        for a in 0..3 {
            for b in 0..3 {
                let points = [a, b];
                let general = swap_log_ratio(&sys, &temps, 0, 1, &points, &chains);
                let driver = biased_swap_log_ratio(
                    temps[0],
                    temps[1],
                    sys.energy[a] + v0.at(a),
                    sys.energy[b] + v0.at(b),
                    sys.energy[b] + v1.at(b),
                    sys.energy[a] + v1.at(a),
                );
                assert!(
                    (general - driver).abs() < 1e-12,
                    "states ({a}, {b}): {general} against {driver}"
                );
            }
        }
    }

    /// A biased pair exchanges at the rate the algebra predicts.
    ///
    /// The prediction is the mean of the acceptance probability under the
    /// product of the two rungs' own targets, computed by enumeration. The
    /// measurement is the fraction of offered swaps a running two-rung ladder
    /// accepts. They agree only if both the swap ratio and the local kernels
    /// target what they are supposed to.
    #[test]
    fn a_biased_pair_exchanges_at_the_predicted_rate() {
        let sys = DiscreteSystem {
            energy: vec![0.0, 0.7, 1.4, 2.1, 2.8],
        };
        let v0 = BiasVector {
            v: vec![1.5, 0.0, 0.4, -0.3, 0.9],
        };
        let v1 = BiasVector {
            v: vec![-0.6, 0.8, 0.0, 1.1, -0.2],
        };
        let temps = [0.6, 1.7];
        let mut chains = [v0.clone(), v1.clone()];

        // Predicted: E_{pi_0 x pi_1}[min(1, a)].
        let p0 = boltzmann(&sys.energy, &v0, temps[0]);
        let p1 = boltzmann(&sys.energy, &v1, temps[1]);
        let mut want = 0.0;
        for (a, &pa) in p0.iter().enumerate() {
            for (b, &pb) in p1.iter().enumerate() {
                let points = [a, b];
                want += pa * pb * swap_probability(&sys, &temps, 0, 1, &points, &chains);
            }
        }

        // Measured: run the ladder.
        let mut rng = StdRng::seed_from_u64(20260806);
        let mut ladder = Ladder::from_temperatures(&temps, SwapScheme::RandomPair);
        let mut points = [0usize, 0usize];
        let mut walk = MetropolisWalk {
            system: sys.clone(),
        };
        let sweeps = 300_000;
        for _ in 0..sweeps {
            for k in 0..2 {
                for _ in 0..4 {
                    walk.advance(k, temps[k], &mut points[k], &mut chains[k], &mut rng);
                }
            }
            ladder.exchange(&sys, &mut points, &chains, &mut rng);
        }
        let (offered, accepted) = ladder.swap_counts();
        let got = accepted as f64 / offered as f64;
        assert!(
            (got - want).abs() < 0.01,
            "measured swap rate {got:.4} against the predicted {want:.4}"
        );
    }

    // -----------------------------------------------------------------
    // 2. The stationary distribution of a small analytic ladder
    // -----------------------------------------------------------------

    /// Runs a three-rung biased ladder and returns each rung's empirical
    /// occupancy. `raw` selects the recorded fault: swapping on raw energies
    /// while the rungs carry different biases.
    fn ladder_occupancy(
        sys: &DiscreteSystem,
        biases: &[BiasVector],
        temps: &[f64],
        scheme: SwapScheme,
        raw: bool,
        sweeps: usize,
    ) -> Vec<Vec<f64>> {
        let n_rungs = temps.len();
        let n_states = sys.energy.len();
        let mut rng = StdRng::seed_from_u64(4242);
        let mut ladder = Ladder::from_temperatures(temps, scheme);
        let mut points: Vec<usize> = vec![0; n_rungs];
        let mut walk = MetropolisWalk {
            system: sys.clone(),
        };
        let mut chains: Vec<BiasVector> = biases.to_vec();
        let mut count = vec![vec![0usize; n_states]; n_rungs];
        for _ in 0..sweeps {
            for k in 0..n_rungs {
                for _ in 0..3 {
                    walk.advance(k, temps[k], &mut points[k], &mut chains[k], &mut rng);
                }
            }
            if raw {
                let e = sys.energy.clone();
                let pts = points.clone();
                let taken = ladder.offer(&mut rng, |k| {
                    let l = (1.0 / temps[k] - 1.0 / temps[k + 1]) * (e[pts[k]] - e[pts[k + 1]]);
                    if l >= 0.0 { 1.0 } else { l.exp() }
                });
                for k in taken {
                    points.swap(k, k + 1);
                }
            } else {
                ladder.exchange(sys, &mut points, biases, &mut rng);
            }
            for k in 0..n_rungs {
                count[k][points[k]] += 1;
            }
        }
        count
            .into_iter()
            .map(|c| c.into_iter().map(|v| v as f64 / sweeps as f64).collect())
            .collect()
    }

    fn analytic_ladder() -> (DiscreteSystem, Vec<BiasVector>, Vec<f64>) {
        let sys = DiscreteSystem {
            energy: vec![0.0, 0.4, 1.1, 1.9],
        };
        let biases = vec![
            BiasVector {
                v: vec![0.0, 0.0, 0.0, 0.0],
            },
            BiasVector {
                v: vec![0.8, -0.4, 0.2, 0.0],
            },
            BiasVector {
                v: vec![1.6, -0.8, 0.4, 0.0],
            },
        ];
        let temps = vec![0.5, 0.9, 1.6];
        (sys, biases, temps)
    }

    #[test]
    fn the_ladder_is_stationary_for_the_product_measure() {
        let (sys, biases, temps) = analytic_ladder();
        for scheme in [
            SwapScheme::RandomPair,
            SwapScheme::StochasticEvenOdd,
            SwapScheme::DeterministicEvenOdd,
        ] {
            let got = ladder_occupancy(&sys, &biases, &temps, scheme, false, 400_000);
            for k in 0..temps.len() {
                let want = boltzmann(&sys.energy, &biases[k], temps[k]);
                for s in 0..sys.energy.len() {
                    assert!(
                        (got[k][s] - want[s]).abs() < 0.01,
                        "{scheme:?} rung {k} state {s}: occupancy {:.4} against target {:.4}; \
                         the ladder is not stationary for the product measure",
                        got[k][s],
                        want[s]
                    );
                }
            }
        }
    }

    /// The recorded fault, kept as a test so that the correct rule is not the
    /// only thing that is checked. Swapping on raw energies between rungs whose
    /// biases differ has to leave the marginals wrong; if this ever passes, the
    /// test above is passing for want of the bias mattering.
    #[test]
    fn raw_energy_exchange_between_biased_rungs_is_not_the_product_measure() {
        let (sys, biases, temps) = analytic_ladder();
        let got = ladder_occupancy(
            &sys,
            &biases,
            &temps,
            SwapScheme::DeterministicEvenOdd,
            true,
            400_000,
        );
        let mut worst: f64 = 0.0;
        for k in 0..temps.len() {
            let want = boltzmann(&sys.energy, &biases[k], temps[k]);
            for s in 0..sys.energy.len() {
                worst = worst.max((got[k][s] - want[s]).abs());
            }
        }
        assert!(
            worst > 0.02,
            "the raw-energy swap reproduced the product measure to {worst:.4}, \
             so this ladder cannot tell the two rules apart"
        );
    }

    // -----------------------------------------------------------------
    // 3. Round trips
    // -----------------------------------------------------------------

    /// Sweeps for one tagged configuration to cross the ladder and return, in
    /// the idealised index process: `n_rungs` rungs, total barrier `barrier`
    /// spread evenly, so the local rejection is `barrier / (n_rungs - 1)`.
    fn sweeps_per_round_trip(n_rungs: usize, barrier: f64, scheme: SwapScheme, seed: u64) -> f64 {
        let temps: Vec<f64> = (0..n_rungs).map(|k| 1.0 + k as f64).collect();
        let mut ladder = Ladder::from_temperatures(&temps, scheme);
        let mut rng = StdRng::seed_from_u64(seed);
        let r = barrier / (n_rungs - 1) as f64;
        let alpha = 1.0 - r;
        let target_trips = 400usize;
        let cap = 40_000 * n_rungs;
        let mut sweeps = 0usize;
        while ladder.index().round_trips() < target_trips && sweeps < cap {
            ladder.offer(&mut rng, |_| alpha);
            sweeps += 1;
        }
        ladder.mean_round_trip_time()
    }

    /// The claim the non-reversible scheme is adopted for, measured rather than
    /// asserted: with the barrier held fixed and the ladder refined, round-trip
    /// time grows linearly for the deterministic sweep and quadratically for a
    /// reversible one.
    ///
    /// The fit is a log-log slope across four ladder sizes. Nothing is tuned:
    /// the barrier is the same for every size, which is what an adapted ladder
    /// maintains as rungs are added.
    #[test]
    fn round_trip_time_is_linear_for_the_deterministic_sweep_and_quadratic_otherwise() {
        let sizes = [8usize, 16, 32, 64];
        let barrier = 3.0;
        let slope = |scheme: SwapScheme| -> f64 {
            let xs: Vec<f64> = sizes.iter().map(|n| (*n as f64).ln()).collect();
            let ys: Vec<f64> = sizes
                .iter()
                .map(|n| sweeps_per_round_trip(*n, barrier, scheme, 7 + *n as u64).ln())
                .collect();
            let xm = xs.iter().sum::<f64>() / xs.len() as f64;
            let ym = ys.iter().sum::<f64>() / ys.len() as f64;
            let num: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - xm) * (y - ym)).sum();
            let den: f64 = xs.iter().map(|x| (x - xm) * (x - xm)).sum();
            num / den
        };
        let deo = slope(SwapScheme::DeterministicEvenOdd);
        let seo = slope(SwapScheme::StochasticEvenOdd);
        let rp = slope(SwapScheme::RandomPair);
        assert!(
            deo < 1.25,
            "deterministic even-odd round-trip time scaled as N^{deo:.2}, not linearly"
        );
        assert!(
            (1.7..2.3).contains(&seo),
            "stochastic even-odd round-trip time scaled as N^{seo:.2}, not quadratically"
        );
        // One pair per sweep costs another factor of N on top of the diffusive
        // index process, which is what the shipped scheme was paying.
        assert!(
            rp > 2.6,
            "random-pair round-trip time scaled as N^{rp:.2}, not cubically"
        );
        assert!(
            seo - deo > 0.6,
            "the two even-odd schemes did not separate: {deo:.2} against {seo:.2}"
        );
    }

    /// A round trip is a tag reaching the hot end and returning, not a swap
    /// count. On a two-rung ladder that always accepts, every second sweep
    /// completes one.
    #[test]
    fn a_round_trip_is_a_tag_reaching_both_ends() {
        let temps = [1.0, 2.0];
        let mut ladder = Ladder::from_temperatures(&temps, SwapScheme::DeterministicEvenOdd);
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..100 {
            ladder.offer(&mut rng, |_| 1.0);
        }
        // Tag 0 alternates between the two rungs, so it returns to the cold
        // rung every second sweep.
        assert_eq!(ladder.index().round_trips(), 50);
        assert!((ladder.round_trip_rate() - 0.5).abs() < 1e-12);
    }

    /// A ladder that never accepts transports nothing, whatever it swaps.
    #[test]
    fn a_ladder_that_never_accepts_makes_no_round_trips() {
        let temps = [1.0, 2.0, 4.0, 8.0];
        let mut ladder = Ladder::from_temperatures(&temps, SwapScheme::DeterministicEvenOdd);
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..2_000 {
            ladder.offer(&mut rng, |_| 0.0);
        }
        assert_eq!(ladder.index().round_trips(), 0);
        assert!((ladder.barrier() - 3.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------
    // 4. The adapted ladder
    // -----------------------------------------------------------------

    /// Mean swap acceptance between two Gaussian rungs of spread `sigma`
    /// separated by `d` in inverse temperature, by simulation.
    fn gaussian_pair_acceptance(d: f64, sigma: f64, seed: u64) -> f64 {
        use rand_distr::{Distribution, Normal};
        let mut rng = StdRng::seed_from_u64(seed);
        // Under the canonical distribution the mean energy shifts with beta by
        // the fluctuation identity, which is what puts the mean of the log
        // ratio at half its variance.
        let hot = Normal::new(d * sigma * sigma, sigma).unwrap();
        let cold = Normal::new(0.0, sigma).unwrap();
        let n = 400_000;
        let mut acc = 0.0;
        for _ in 0..n {
            let e_c = cold.sample(&mut rng);
            let e_h = hot.sample(&mut rng);
            // ln a = (beta_cold - beta_hot)(E_cold - E_hot) = -d (E_hot - E_cold).
            let l = -d * (e_h - e_c);
            acc += if l >= 0.0 { 1.0 } else { l.exp() };
        }
        acc / n as f64
    }

    /// The step the ladder is built from does what it says: a pair separated by
    /// it accepts at the target rate.
    #[test]
    fn the_gaussian_step_hits_its_acceptance_target() {
        for sigma in [0.5_f64, 2.0] {
            for target in [0.23_f64, 0.4] {
                let d = beta_step_for_acceptance(sigma, target);
                let got = gaussian_pair_acceptance(d, sigma, 99);
                assert!(
                    (got - target).abs() < 0.01,
                    "sigma {sigma} target {target}: measured acceptance {got:.4}"
                );
            }
        }
    }

    #[test]
    fn the_quantile_matches_known_values() {
        assert!((inverse_normal_cdf(0.975) - 1.959_963_98).abs() < 1e-7);
        assert!((inverse_normal_cdf(0.5)).abs() < 1e-12);
        assert!((inverse_normal_cdf(0.115) + 1.200_359).abs() < 1e-5);
        // The constant quoted in the module documentation.
        assert!((beta_step_for_acceptance(1.0, 0.23) - 1.6976).abs() < 1e-3);
    }

    /// A ladder built from a fluctuation has no hand-set top: the top is
    /// wherever the steps land, and it moves when the fluctuation moves.
    #[test]
    fn the_first_ladder_comes_from_the_fluctuation() {
        let narrow = Ladder::from_fluctuation(0.8, 6.0, 5, 0.23, SwapScheme::DeterministicEvenOdd);
        let wide = Ladder::from_fluctuation(0.8, 12.0, 5, 0.23, SwapScheme::DeterministicEvenOdd);
        let tn = narrow.temperatures();
        let tw = wide.temperatures();
        assert!(!narrow.clamped_at_construction() && !wide.clamped_at_construction());
        assert!((tn[0] - 0.8).abs() < 1e-12 && (tw[0] - 0.8).abs() < 1e-12);
        // A larger spread needs smaller steps in beta, so its top is colder.
        assert!(
            tw[4] < tn[4],
            "top {:.3} against {:.3}: a wider energy spread must give a shorter ladder",
            tw[4],
            tn[4]
        );
        // And the clamp fires at the spreads a quenched cluster surface
        // actually shows, where the target step exceeds the whole beta range.
        let cluster = Ladder::from_fluctuation(0.8, 1.0, 4, 0.23, SwapScheme::DeterministicEvenOdd);
        assert!(cluster.clamped_at_construction());
        let t = cluster.temperatures();
        assert!((t[0] - 0.8).abs() < 1e-12);
        assert!(
            (t[3] - 0.8 / MIN_HOT_BETA_FRACTION).abs() < 1e-9,
            "top {}",
            t[3]
        );
    }

    /// Barrier equalisation on a ladder whose rejections are deliberately
    /// lopsided. The profile has to flatten, and the endpoints have to stay.
    ///
    /// Rejection between neighbours is modelled as `1 - exp(-c |d beta|)`,
    /// monotone in the spacing, which is the qualitative behaviour of any
    /// overlap-driven acceptance and enough to exercise the estimator.
    #[test]
    fn equalising_flattens_the_rejection_profile() {
        let temps: Vec<f64> = vec![0.4, 0.45, 0.5, 1.2, 3.0, 8.0];
        let mut ladder = Ladder::from_temperatures(&temps, SwapScheme::DeterministicEvenOdd);
        let mut rng = StdRng::seed_from_u64(31337);
        let spread = |l: &Ladder| {
            let r = l.rejection_rates();
            let m = r.iter().sum::<f64>() / r.len() as f64;
            (r.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / r.len() as f64).sqrt()
        };
        let reject = |betas: &[f64], k: usize| {
            let d = (betas[k] - betas[k + 1]).abs();
            1.0 - (-0.9 * d).exp()
        };
        // Measure the starting profile.
        for _ in 0..4_000 {
            let b = ladder.betas().to_vec();
            ladder.offer(&mut rng, |k| 1.0 - reject(&b, k));
        }
        let before = spread(&ladder);
        let barrier_before = ladder.barrier();
        let t_cold = ladder.temperatures()[0];
        let t_hot = *ladder.temperatures().last().unwrap();
        for _ in 0..25 {
            ladder.equalise();
            for _ in 0..4_000 {
                let b = ladder.betas().to_vec();
                ladder.offer(&mut rng, |k| 1.0 - reject(&b, k));
            }
        }
        let after = spread(&ladder);
        assert!(
            after < before * 0.35,
            "rejection spread went {before:.4} to {after:.4}; the estimator did not flatten it"
        );
        let t = ladder.temperatures();
        assert!((t[0] - t_cold).abs() < 1e-9, "the cold endpoint moved");
        assert!(
            (t[t.len() - 1] - t_hot).abs() < 1e-9,
            "the hot endpoint moved"
        );
        // Flattening redistributes the barrier; it does not invent one.
        assert!(
            (ladder.barrier() - barrier_before).abs() < 0.5 * barrier_before,
            "the barrier changed from {barrier_before:.3} to {:.3}",
            ladder.barrier()
        );
    }

    /// The endpoint controller pulls a ladder in when its pairs reject too
    /// much and stretches it when they reject too little.
    #[test]
    fn the_top_controller_moves_toward_the_target_rejection() {
        let temps: Vec<f64> = vec![0.5, 1.0, 2.0, 4.0, 8.0];
        let mut rng = StdRng::seed_from_u64(5);
        let mut hot = Ladder::from_temperatures(&temps, SwapScheme::DeterministicEvenOdd);
        for _ in 0..1_000 {
            hot.offer(&mut rng, |_| 0.05);
        }
        let top_before = *hot.temperatures().last().unwrap();
        hot.retune_top(1.0 - TARGET_SWAP_ACCEPT, 0.5);
        let top_after = *hot.temperatures().last().unwrap();
        assert!(
            top_after < top_before,
            "a ladder rejecting 0.95 was not pulled in: {top_before:.3} to {top_after:.3}"
        );

        let mut cool = Ladder::from_temperatures(&temps, SwapScheme::DeterministicEvenOdd);
        for _ in 0..1_000 {
            cool.offer(&mut rng, |_| 0.99);
        }
        let before = *cool.temperatures().last().unwrap();
        cool.retune_top(1.0 - TARGET_SWAP_ACCEPT, 0.5);
        let after = *cool.temperatures().last().unwrap();
        assert!(
            after > before,
            "a ladder rejecting 0.01 was not stretched: {before:.3} to {after:.3}"
        );
    }

    /// The pair statistics use the acceptance probability rather than the
    /// accept indicator, so a pair offered a hundred times at probability 0.3
    /// reports 0.3 and not whatever the coin did.
    #[test]
    fn pair_acceptance_is_the_probability_not_the_indicator() {
        let temps = [1.0, 2.0];
        let mut ladder = Ladder::from_temperatures(&temps, SwapScheme::RandomPair);
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..100 {
            ladder.offer(&mut rng, |_| 0.3);
        }
        let a = ladder.pair_acceptance()[0].unwrap();
        assert!((a - 0.3).abs() < 1e-12, "reported acceptance {a}");
        assert!((ladder.barrier() - 0.7).abs() < 1e-12);
    }
}
