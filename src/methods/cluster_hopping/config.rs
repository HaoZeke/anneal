//! Scale-aware cluster-search configuration and named presets.

use super::preset::{LennardJonesPreset, MolecularPreset};
use super::*;

/// Which descriptor a run keys basins on.
///
/// Named rather than a boolean because there are now three and the choice is
/// the lever: at 75 points the merge radius on a distance spectrum is sharply
/// sensitive, 13 seeds in 24 at 0.7 against 0 in 8 at 0.95, and a descriptor
/// that separates distinct structures more cleanly is what would widen that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Keying {
    /// Sorted pairwise distances.
    #[default]
    Distances,
    /// Prospective superbasin: the Weisfeiler--Lehman key of the coloured
    /// primitive-ring graph of the structure's core
    /// ([`crate::corekey::core_key`]). Shelf isomers that differ by surface
    /// relocations share one key; distinct packings do not.
    Core,
    /// Coordinates, matched by a shape metric.
    Shape,
    /// Sorted per-point pair energies.
    Sites,
    /// Coordinates canonically ordered against a fixed reference.
    ///
    /// The only keying here that does not throw correspondence away. Sorting
    /// buys invariance by discarding which point holds which value; a canonical
    /// order keeps it, so two structures with the same multiset of distances
    /// and a different arrangement separate.
    ///
    /// It is also what makes shape keying affordable. A shape distance costs an
    /// IRA call, so keying on it directly pays one call per basin comparison
    /// and a bias holding thousands of basins cannot be queried at hop rate.
    /// Matching each structure once against a reference costs one call per hop
    /// and leaves every comparison Euclidean.
    Canonical,
    /// High-`l` mean SOAP. The packing superbasin, not an isomer.
    ///
    /// SortedPairs at 0.7 merges a one-hop return and splits distinct
    /// minima, so bias fills each Mackay isomer and never the funnel.
    /// Unit mean SOAP puts isomers of one packing inside `0.10` and
    /// LJ75 ico-Marks at `0.163`. Filling that one well is the
    /// Chatterjee-Voter move under a force ledger: many recrossings
    /// of the occupied packing, then an exit. No clock, no FPTA.
    SoapPacking,
    /// Sorted distances with the kernel spectra of
    /// [`crate::tensor_id::TripletSpectrum`] appended: strictly richer than
    /// [`Keying::Distances`], adding the weighted triangle sum a multiset of
    /// distances cannot hold, with no reference structure and no chosen
    /// coordinate. Its merge radius is a different number from the distance
    /// keying's, larger by the length of the appended block.
    Triplet,
}

/// Constraint applied to SOAP proposals on grouped systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SoapProposalMode {
    /// Apply the ambient Cartesian pullback. Internal geometry may change.
    Flexible,
    /// Retract each declared group onto its nearest rigid motion.
    Rigid,
    /// Do not include a SOAP proposal in the move library.
    Off,
}

/// Deterministic continuous-symmetry proposal inserted into basin hopping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(tag = "group", rename_all = "snake_case")]
pub enum ContinuousSymmetry {
    /// No continuous-symmetry proposal.
    #[default]
    Off,
    /// Project onto the inversion group `C_i` at quench indices divisible by
    /// `interval`, quench the group average, and adopt it only when it is
    /// lower. The supplied minimum is quench one and each projection quench
    /// advances the same counter as an ordinary hopping quench.
    Inversion {
        /// Divisor of the basin-hopping quench count.
        interval: usize,
    },
}

/// Driver settings.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Config {
    /// Points in a state. Cartesian libraries use length `3 * n_points`.
    /// [`MoveLibrary::RigidBody`] appends a rotation vector per point, so
    /// the objective dimension is `6 * n_points`.
    pub n_points: usize,
    /// Declared coordinate length scale.
    pub length_scale: f64,
    /// Declared objective-energy scale.
    pub energy_scale: f64,
    /// Proposal library hosted by this configuration.
    pub move_library: MoveLibrary,
    /// Separation below which two points count as neighbours.
    pub neighbour_cutoff: f64,
    /// Pair cutoff used by the symmetry proposal.
    pub symmetrise_cutoff: f64,
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
    /// Kernel width for [`Keying::Triplet`], in the units of the coordinates.
    ///
    /// Ignored by every other keying. Defaults to the Lennard-Jones value; a
    /// potential with a different pair minimum needs it scaled the way
    /// `container` and `min_separation` are.
    pub keying_sigma: f64,
    /// Design point for the budget-window temperature, as a fraction of the
    /// sphere-model descent boundary. Must lie strictly below two.
    pub theta: f64,
    /// Set the temperature by the budget-window law rather than holding it.
    pub budget_window: bool,
    /// Choose the move kernel by discounted Thompson allocation.
    pub allocate_moves: bool,
    /// Set the deposit height from the escape gaps the chain observes.
    pub adaptive_height: bool,
    /// Hops a single `run` may take before returning, when set.
    ///
    /// Used by the replica ladder to advance one chain by a slice; a plain run
    /// leaves it unset and stops only when the ledger does.
    pub max_hops: Option<usize>,
    /// Replicas run on a temperature ladder, with periodic swaps.
    ///
    /// One is the plain chain. Above one, the driver runs a ladder and offers
    /// swaps through [`crate::exchange::Exchange`], which is the crate's own
    /// operator and satisfies detailed balance by construction.
    ///
    /// This is the standard non-local mechanism for a multi-funnel landscape
    /// and the measurements here say why it is the right one to reach for: no
    /// single move from the plateau reaches anything lower, so a cold chain
    /// cannot leave it, while a hot chain crosses freely and finds nothing
    /// precise. A swap moves a hot chain's crossing down to a cold chain that
    /// can polish it, which neither temperature achieves alone.
    pub replicas: usize,
    /// Hops between swap attempts.
    pub swap_period: usize,
    /// Drive the escape scale and the acceptance threshold from the history,
    /// after Goedecker's minima hopping, instead of a Metropolis temperature.
    ///
    /// Revisiting a known minimum makes the *next escape* harder rather than
    /// the *current basin* less attractive, which is a different use of the
    /// same history the bias keeps. The transition region between funnels is
    /// left crossable, which Goedecker argues is why flooding it is the wrong
    /// response to a revisit.
    ///
    /// This is the scaled-move form: the escape scale multiplies the move
    /// amplitude and the acceptance threshold replaces Metropolis. Soft-mode
    /// climbs are *not* taken every hop under this flag; they are the separate
    /// [`Config::escape_on_stall`] path. Measured: activating every hop under a
    /// gradient cost ~687 charged evaluations per hop on LJ38 and bought 291
    /// hops from 200k, which is not a search. The controller and the climb are
    /// complementary and must stay separable.
    pub minima_hopping: bool,
    /// Lanczos steps for the soft-mode escape.
    ///
    /// Each costs two gradient evaluations, charged. Eight resolves the softest
    /// mode of a cluster well enough to displace along, against about forty
    /// charged evaluations for the relaxation that follows.
    pub escape_lanczos_steps: usize,
    /// Finite-difference step for the Hessian-vector product.
    pub escape_epsilon: f64,
    /// Distance moved along the softest mode per climbing step.
    pub escape_amplitude: f64,
    /// Push past the saddle, in units of the climbing step, before the
    /// feedback scale multiplies it.
    pub escape_overshoot: f64,
    /// Climbing steps before a climb is abandoned.
    pub escape_max_climb: usize,
    /// Climb out of the basin when the chain stops improving.
    ///
    /// The escape and the plain chain have opposite economics and this is how
    /// they are combined. A climb is a guaranteed way out of a funnel and costs
    /// 637 charged evaluations against 30 for an ordinary hop, so running one
    /// every hop buys 471 hops where the plain chain buys a hundred thousand
    /// and loses LJ38 outright. Running one only when the chain has stopped
    /// improving costs a few per cent and supplies the one thing a biased
    /// random walk has no mechanism for: leaving a funnel on purpose.
    pub escape_on_stall: bool,
    /// Leave a stalled basin back through the door the chain came in.
    ///
    /// The screen state of the accepted trial that entered the current basin
    /// is an evaluated waypoint of the descent, partway down from the ridge.
    /// Restarting from it with an outward push costs only the requench,
    /// against 637 evaluations for a Lanczos climb, and starts from a point
    /// the landscape has already priced. The climb remains the fallback when
    /// no entry is on record.
    pub trail_on_stall: bool,
    /// Smallest number of hops without improvement before a climb is
    /// triggered.
    ///
    /// A floor, not the trigger. The trigger is
    /// `escape_stall_factor` times the longest quiet stretch this chain has
    /// already survived, so a climb fires only when the chain is stuck longer
    /// than it has ever been stuck before.
    ///
    /// A fixed patience cannot be set. Traced on 75 points, the runs that
    /// succeed cross at 42 and 55 per cent of the way in, after 1500 to 1900
    /// basins, and go tens of thousands of hops between improvements on the
    /// way. A patience of 400 fires about 180 climbs into that and the chain
    /// never accumulates: the arm scored 2 seeds in 8 against 9 in 16 without
    /// it.
    pub escape_stall_patience: usize,
    /// Multiple of the longest quiet stretch so far that counts as stuck.
    pub escape_stall_factor: f64,
    /// Track the funnel partition the search's own transitions imply.
    ///
    /// A stall is currently detected from energy: so many hops without a new
    /// best. That conflates two situations a search should treat differently, a
    /// chain polishing inside a region it can leave, and a chain that cannot
    /// leave at all. The transition graph tells them apart: when the accepted
    /// hops split into two parts with few edges between them and the chain sits
    /// in one, that is a funnel and not slow progress.
    ///
    /// Steer with spectral (Fiedler) well-tempered bias on the hop graph.
    ///
    /// When true, accepted hops build a transition graph on basin identity
    /// (the same fingerprint as the per-basin bias; with `ira` and
    /// [`Keying::Canonical`] that identity is IRA, not SortedPairs). The
    /// second eigenvector of the normalised Laplacian is the continuous CV
    /// for an extra well-tempered bias ([`crate::spectral::SpectralBias`]).
    /// That is the algorithm: identity supplies resolution, the spectrum
    /// supplies the funnel coordinate — no hand-chosen Q4. Refits cost an
    /// eigendecomposition of a matrix the size of the basin count, on a
    /// schedule ([`Config::funnel_period`]). See [`crate::funnel_spectral`]
    /// and [`crate::spectral`].
    pub track_funnels: bool,
    /// Accepted hops between Laplacian refits / Fiedler updates.
    pub funnel_period: usize,
    /// Symmetrise onto the symmetry the structure nearly has, on a stall.
    ///
    /// Oakley, Johnston and Wales report the mean first encounter time for the
    /// 98-point cluster, whose global minimum is tetrahedral, improving by more
    /// than seventyfold under a scheme of this kind. That is the case this
    /// driver is weakest on: 3 seeds in 8 at twelve million evaluations against
    /// 8 in 8 at 75 points.
    ///
    /// Applied when the chain is stuck rather than as an allocator arm, because
    /// it is not a perturbation competing with the others. It either finds an
    /// approximate symmetry and lands the structure on it, or finds none and
    /// leaves the chain alone. See [`crate::symmetrise`].
    pub symmetrise_on_stall: bool,
    /// Published continuous-symmetry move, independent of stall detection.
    ///
    /// This is distinct from [`Config::symmetrise_on_stall`]. The
    /// continuous-symmetry construction solves a global atom assignment for
    /// each group image and quenches the averaged geometry on a fixed
    /// schedule. The current implementation provides `C_i`, whose inversion
    /// operation makes the orientation objective rotation independent.
    pub continuous_symmetry: ContinuousSymmetry,
    /// Largest deviation at which an approximate symmetry is worth using.
    pub symmetry_tolerance: f64,
    /// Coordinate-space radius used to merge points after symmetrisation.
    pub symmetry_merge_radius: f64,
    /// Hops without improvement before a symmetrisation is considered.
    ///
    /// Separate from the escape patience because the two answer different
    /// questions: an escape is for a chain that cannot leave, this is for a
    /// chain that has stopped finding anything and may be near a symmetric
    /// answer without being on it.
    pub symmetrise_patience: usize,
    /// Wales and Doye's angular move, applied when a point is loose.
    ///
    /// "If the highest pair energy rose above a fraction R of the lowest pair
    /// energy then an angular move was employed for the atom in question with
    /// all other atoms fixed" (J. Phys. Chem. A 101, 5111). This is the move
    /// their unbiased search used to reach the decahedral minima at 75 and 102
    /// points, and it is not in this crate's library: the nearest thing,
    /// surface relocation, picks the least-coordinated point rather than the
    /// worst-bound one and places it near the surface rather than at the far
    /// edge.
    ///
    /// It replaces the allocator's choice on the steps where it fires, rather
    /// than being one arm among many, because the criterion decides when it is
    /// the right move.
    pub angular_moves: bool,
    /// Acceptance rate the pair-energy ratio is tuned to.
    ///
    /// "R was adjusted to give an acceptance ratio for angular moves of 0.5 and
    /// generally converged to between 0.40 and 0.44."
    pub angular_target: f64,
    /// Which descriptor basins are keyed on.
    ///
    /// Takes precedence over `shape_keyed`, which stays for callers that only
    /// need the two-way choice.
    pub keying: Keying,
    /// Choose the move from where the chain is standing.
    ///
    /// The allocator learns one success rate per move, which is the right model
    /// when a move has a rate and the wrong one when it has a precondition. The
    /// angular move is the clear case: it is not applied at a frequency, it is
    /// applied when a point crosses a pair-energy criterion, and a rate learned
    /// across the times it was and was not appropriate describes no situation
    /// the chain is ever in. See [`crate::contextual`].
    pub contextual_moves: bool,
    /// Rate at which the contextual allocator picks uniformly regardless.
    pub contextual_floor: f64,
    /// Decide whether to finish a relaxation from a posterior rather than
    /// from a fixed margin.
    ///
    /// The margin screen is the one mechanism here measured to be worth having,
    /// at 13 seeds in 24 against 2 in 8 without it, and what it does is spend
    /// numerical effort where it is likely to pay. That is a decision under
    /// uncertainty about a quantity not yet computed, and a constant is a poor
    /// way to make it. See [`crate::screen`].
    pub bayes_screen: bool,
    /// Accept against the density of minima rather than against the energy.
    ///
    /// The Metropolis rule targets `g(E~) exp(-E~ / T)` in quenched energy, and
    /// on a multi-funnel landscape `g` decides the outcome: the chain sits in
    /// whichever funnel holds the most minima. Weighting by `1 / g` makes the
    /// sampled energy histogram flat, so the deep and rare energies get the
    /// same share of the run as the shallow and abundant ones. See
    /// [`crate::dos`].
    pub flat_histogram: bool,
    /// Trials between weight refreshes. The weight is frozen across a sweep so
    /// each sweep is an exact chain for its own target rather than an adaptive
    /// one whose invariance has to be argued.
    pub flat_sweep: usize,
    /// Quantile of a sweep's visited energies that sets the cut below which the
    /// target is flat. Lower is greedier: the flat region shrinks toward the
    /// deepest energies the chain has reached.
    pub flat_quantile: f64,
    /// Take the temperature from the entropy's own slope rather than from a
    /// constant.
    ///
    /// The Metropolis rule and the basin bias both measure well and both stand;
    /// what this replaces is the one hand-set number they sit on. See
    /// [`crate::dos::DensityOfStates::temperature`].
    pub statistical_temperature: bool,
    /// Deposit a well-tempered bias in quenched energy.
    ///
    /// The per-basin bias fills the basin the chain stands in, and the trap is
    /// a funnel holding exponentially many basins. Energy separates the funnel
    /// where a coordinate length cannot. All scales come from the run's own
    /// quenched-energy distribution. See [`crate::dos::EnergyBias`].
    pub energy_bias: bool,
    /// Reward move arms by the depth they reach, not by acceptance.
    ///
    /// See [`crate::allocate::DepthAllocator`].
    pub depth_reward: bool,
    /// Perturb in the soft subspace of the incumbent's own curvature.
    ///
    /// An isotropic step in `3n` dimensions puts nearly all of its norm on
    /// stiff directions, and the quench relaxes those components straight back
    /// into the basin they came from: only the projection onto the
    /// low-curvature subspace survives. Confining the draw to that subspace
    /// with per-mode thermal amplitudes is the local Gaussian `N(0, T H^{-1})`
    /// truncated to the modes that carry displacement at `T`, computed
    /// matrix-free by shifted Lanczos and charged to the ledger like any other
    /// work. Nothing here mentions a morphology: the subspace is the
    /// structure's own.
    pub soft_perturb: bool,
    /// Soft modes kept in the subspace.
    pub soft_modes: usize,
    /// Lanczos steps per subspace computation.
    pub soft_steps: usize,
    /// Perturb from a covariance learned from this run's accepted moves.
    ///
    /// The soft-subspace arm computes the directions that matter from the
    /// Hessian and pays charged evaluations for them. This arm learns the same
    /// object free: accepted minimum-to-minimum displacements concentrate in
    /// the directions basins actually connect along, so their shrunk empirical
    /// covariance, `(1 - gamma) sigma0^2 I + gamma S`, is a proposal fitted to
    /// the run's own successes. Shrinkage toward isotropy covers the cold
    /// start, and the weight ramps with evidence, which is the Ledoit-Wolf
    /// compromise rather than a schedule. Sampling needs no factorisation:
    /// `sqrt(1-gamma) sigma0 z0 + sqrt(gamma/m) sum z_i d_i` has exactly the
    /// mixture covariance. Nothing morphological enters; the buffer is this
    /// run's history.
    pub cov_perturb: bool,
    /// Stage the quench: settle the moved atoms in the frozen environment
    /// before the screening relaxation.
    ///
    /// The measured-productive moves displace one to three atoms while every
    /// trial pays a full-system screen. A k-atom settle costs k of the
    /// n(n-1)/2 pair rows per evaluation, charged fractionally through
    /// [`Ledger::charge_frac`], so the cheap stage absorbs the descent the
    /// full screen would otherwise spend whole evaluations on.
    pub staged_quench: bool,
    /// Descent steps in the settle stage.
    pub settle_iters: usize,
    /// Inter-group contact cutoff for the molecular library.
    pub group_cutoff: f64,
    /// Bonding cutoff below which two atoms are one molecule, for deriving
    /// the groups from the structure's own connectivity each hop. Used only
    /// when no species are declared; with species the bond-matrix rule over
    /// covalent radii replaces it.
    pub covalent_cutoff: f64,
    /// Atomic numbers, one per point. With these set the connectivity uses
    /// the species-aware bond matrix, which a single length cannot replace on
    /// a system holding more than one element.
    pub species: Option<Vec<u32>>,
    /// Bond-matrix tolerance on the covalent radii sum.
    pub bond_tolerance: f64,
    /// Frozen mask, one flag per point. A frozen point is environment: groups
    /// made entirely of frozen points never enter the move library, and the
    /// structure is not recentred or contained, since the frozen frame IS the
    /// frame. The caller's objective is expected to return zero force on
    /// frozen points so the quench leaves them where they stand.
    pub frozen: Option<Vec<bool>>,
    /// Dynamic active region: seed atoms and the number of bond-matrix
    /// neighbour shells around them that stay mobile, recomputed from the
    /// current structure each hop. Everything outside the region is treated
    /// as frozen for that hop, so the mobile patch follows the seeds.
    /// Requires `species`.
    pub active_region: Option<(Vec<usize>, usize)>,
    /// Trials relaxed regardless of the posterior, to keep the model's training
    /// set from being censored by the rule it trains.
    pub bayes_exploration: f64,
    /// Posterior probability of improvement above which a trial is relaxed.
    pub bayes_threshold: f64,
    /// Observations before the posterior is consulted at all.
    pub bayes_warmup: usize,
    /// Forbid the funnel the chain is stuck in, rather than making it
    /// expensive.
    ///
    /// Wales and Doye record the lockout directly: once the lowest icosahedral
    /// minimum is reached at 75 points, the decahedron is never found later in
    /// that run. Two responses were measured here and both failed. A
    /// well-tempered bias raises the potential where the chain has been, and
    /// runs that fail register as many basins as runs that succeed, so the
    /// filling is not what decides it. Restarting the walker from a random
    /// configuration failed too, nineteen times per run: a random start
    /// descends into the icosahedral funnel again because that funnel's basin
    /// of attraction is far wider. The lockout is entropic and a soft
    /// penalty cannot outrun it.
    ///
    /// This rejects outright. Structures within the merge radius of a
    /// quarantined one are refused whatever their energy, so the chain cannot
    /// return to a funnel it has been declared stuck in. The ledger still
    /// records them, so a quarantine that turns out to cover the answer costs
    /// the search nothing it had already found.
    pub tabu_on_stall: bool,
    /// Quarantined structures held at once, oldest dropped first.
    pub tabu_capacity: usize,
    /// Restart the walker from a fresh configuration on a stall, keeping the
    /// bias.
    ///
    /// What is stuck is the walker, not the landscape memory. Traced at 75
    /// points, a run that fails stops improving at 2 to 26 per cent of the way
    /// in and spends the rest inside the icosahedral funnel, while the runs
    /// that succeed cross at 42 to 91 per cent; so a chain that has not crossed
    /// early is unlikely to, and the thing worth keeping from its remaining
    /// budget is what it has already filled in.
    ///
    /// Different from the climb, which moves the walker a short way and leaves
    /// it in the same funnel, and from a bank, which splits the budget. This
    /// spends nothing and discards nothing: the bias the old chain built is
    /// what steers the new one away from where the old one was.
    pub restart_on_stall: bool,
    /// Charged calls without a new best before a stalled chain restarts
    /// from a fresh random cluster.
    pub restart_patience: usize,
    /// Set the merge radius from how far an accepted hop actually reaches.
    ///
    /// A radius chosen by hand does not transfer: one calibrated at 38 points
    /// is wrong at 75, and one calibrated in a sorted-distance spectrum is
    /// wrong in a shape metric. Two structures are the same basin when a single
    /// accepted hop can carry the chain between them, and the search reports
    /// that step length for free. See [`crate::calibrate`].
    pub calibrate_radius: bool,
    /// Quantile of the accepted-hop step length the radius tracks.
    pub calibrate_quantile: f64,
    /// Accepted hops required before the calibrated radius is used.
    pub calibrate_warmup: u64,
    /// Scale the deposit height with rung temperature.
    ///
    /// A bias pushes a chain out of where it sits and a low temperature keeps
    /// it in, so a cold rung carrying a full bias is evicted from good basins
    /// and cannot return. Measured on LJ75, that inverts the ladder: the
    /// coldest rung held -391.3 while the hottest held -396.0, where a working
    /// ladder has the deepest structure at the cold end.
    ///
    /// Scaling the height by the rung's temperature ratio leaves the coldest
    /// rung nearly a plain hopping chain, which polishes, and the hottest
    /// carrying the full bias, which crosses. The swap then moves a crossing
    /// down to a chain that can refine it, which is the division of labour the
    /// ladder exists for.
    pub bias_by_rung: bool,
    /// Hottest temperature on the ladder, as a multiple of `temperature`.
    pub ladder_top: f64,
    /// Abandon a trial whose short relaxation is heading back to the current
    /// basin, before paying for the full one.
    ///
    /// The energy screen passes a returning trial, because a perturbation that
    /// falls straight back carries the incumbent's energy and looks like a
    /// success. Near a deep minimum roughly nineteen proposals in twenty
    /// return, so most of the budget buys relaxations into the basin the chain
    /// already occupies. Measured on the shape distance after a partial
    /// relaxation, returns and escapes separate cleanly: 0.160 against 1.846
    /// with 97 per cent of pairs ordered correctly at thirty iterations.
    pub return_screen: bool,
    /// SOAP hop uses the 555→421 / fcc-prototype oracle. Off (recommended)
    /// is the observed-cloud residual `2p − μ`.
    pub soap_class_residual: bool,
    /// Molecular constraint mode for the SOAP proposal. Recommended uses the
    /// flexible Cartesian pullback; `Off` removes the arm for the control.
    pub soap_mode: SoapProposalMode,
    /// Extra relaxation steps on a returning trial when `return_screen` is on.
    ///
    /// Zero (the recommended default) skips the full quench entirely, which is
    /// how a hop opts out of return polish. A positive value finishes every
    /// returning trial in that hop at a fraction of `relax_steps` so a
    /// near-incumbent that is actually a new isomer can still settle.
    pub return_polish: usize,
    /// Ledger spend that must be reached before `return_polish` fires.
    ///
    /// Zero polishes every returning trial. A positive value keeps the first
    /// part of the hop as skip-return and only finishes returns after that
    /// many charged evaluations, so one chain can cover both the ico GM and
    /// the later Marks funnel.
    pub return_polish_after: usize,
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
    ///
    /// Bounded below by what basin identity needs, which is not what a
    /// population diversity threshold needs. A merge radius under the distance
    /// a single hop covers, 0.4766 on 75-point minima, stops recognising a
    /// structure already visited: annealing 0.7 down to 0.07 took a run from
    /// 250 basins at 25 revisits to 4423 at 2.6, and the best found from
    /// -396.282 to -394.629.
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
    /// Calibrated by sweep on the corrected relaxer, LJ38 at 4e5 charged
    /// evaluations, four seeds each:
    ///
    /// | steps | solved | charged per hop | hops |
    /// |-------|--------|-----------------|------|
    /// | 6     | 0/4    | 11              | 149392 |
    /// | 10    | 0/4    | 16              | 94412 |
    /// | 15    | 1/4    | 21              | 66437 |
    /// | 25    | 4/4    | 33              | 49728 |
    /// | 40    | 4/4    | 47              | 33396 |
    ///
    /// Three times the hops buys nothing when the quench is short. The chain
    /// moves on the transformed landscape, and a screened energy that has not
    /// reached its basin is not a point on it, so a proposal is compared
    /// against the incumbent on a quantity that is not the one being
    /// minimised. 25 is the knee: 40 solves as often and costs 1.4 times as
    /// much per hop, 15 costs less and solves once in four.
    ///
    /// This is the same wall the adaptive screening quench hit from the other
    /// side. There the extrapolated energy was wrong by 1e4 at the step where
    /// its rule fired; here a genuinely shorter quench is simply not enough
    /// quench. Both say the screening pass is the quench rather than overhead
    /// around it.
    pub screen_steps: usize,
    /// Whether the screening pass stops on a decision instead of `screen_steps`.
    ///
    /// The fixed length is where the budget goes: measured on 38 points, 89 to
    /// 92 per cent of charged evaluations were spent screening, against 8 per
    /// cent on the relaxations that screening exists to avoid. Every mechanism
    /// in this crate that tried to change *where* the chain goes was measured
    /// and failed; this one changes what a hop costs, which is the axis the
    /// only successful mechanism so far, the return screen, also moved.
    pub adaptive_screen: bool,
    /// Gradient below which a structure may be recorded as the run's best.
    ///
    /// Loose enough that a genuine minimum passes, since a quenched cluster
    /// comes back near 1e-6, and tight enough to bar a partial quench, which
    /// comes back near 1e-1 or worse.
    pub record_gradient: f64,
    /// Whether every quenched energy is kept, not only the improving ones:
    /// the sample [`crate::tail`] fits an endpoint to. Truncated by
    /// [`Config::screen_margin`], since the full relaxation runs only where
    /// the partial energy sits within the margin of the incumbent.
    pub trace_quenched: bool,
    /// Extra relaxation steps spent polishing a new best to share tolerance.
    ///
    /// Screened hopping's economy is that almost no hop is fully relaxed, so
    /// almost no state meets the tolerance a shared census identifies basins
    /// at, and a cooperative run has nothing valid to offer. Polishing only
    /// the states that improve the record bounds the cost to the number of
    /// improvements, a few hundred evaluations each over tens of events,
    /// about one percent of a campaign budget, all charged to the ledger.
    /// Zero disables the polish and leaves the solo behaviour untouched.
    pub polish_records: usize,
    /// Predictive spread, in units of the temperature, above which the first
    /// stage abstains rather than deciding.
    ///
    /// See [`crate::delayed::Surrogate::predict_at`].
    pub surrogate_tolerance: f64,
    /// Whether acceptance is delayed behind a learned surrogate.
    ///
    /// A first stage decides on a surrogate for the quenched energy, costing
    /// one evaluation and no gradient, and only survivors are quenched; a
    /// second stage subtracts the surrogate difference back out. The composite
    /// step is reversible with respect to the true target for any surrogate,
    /// so a poor one costs acceptance rate rather than correctness. This is
    /// what the screen was reaching for and does not have. See
    /// [`crate::delayed`].
    pub delayed_acceptance: bool,
    /// Candidates built and scored per growth proposal.
    ///
    /// Costs no charged evaluations, since scoring is structural.
    pub construct_width: usize,
    /// Whether to score the quench extrapolation without acting on it.
    ///
    /// Runs the screening pass to its full length and records what an adaptive
    /// stop would have claimed, which is the only way to separate "the model is
    /// wrong" from "the model is right and the search needs the precision".
    pub probe_screen: bool,
    /// Descent steps before the quench predictor may speak.
    ///
    /// The first steps of a quench from a perturbed cluster are nowhere near
    /// the quadratic region: atoms sit close enough that energies run to 1e5,
    /// and a log-linear fit through three such decrements extrapolates a tail
    /// that has nothing to do with the basin. Measured, a stop at step 4 missed
    /// the full pass by 1.0e4 on a landscape whose minima are 0.5 apart.
    pub quench_warmup: usize,
    /// Standard deviations of separation a verdict needs.
    pub quench_confidence: f64,
    /// Relaxation steps in the full pass.
    pub relax_steps: usize,
    /// First-phase transform of every relaxation: relax on a compacted
    /// surface first, then on the plain potential from that minimum, judging
    /// the plain energy. `None` relaxes on the plain potential only.
    pub two_phase: Option<crate::methods::two_phase::TwoPhase>,
    /// Reoccupation move: every `reoccupy_interval` charged calls the chain
    /// rebuilds its surface on the lattice grown from its interior with
    /// [`crate::methods::lattice_search::reoccupy`], relaxes the result and
    /// adopts it when it is lower than the current minimum. `None` disables
    /// the move; the lattice settings name the pair form the site energies
    /// are read from.
    #[serde(skip)]
    pub reoccupy: Option<crate::methods::lattice_search::LatticeSearchConfig>,
    /// Charged calls between reoccupation moves.
    pub reoccupy_interval: usize,
    /// Learned portfolio of relaxation surfaces: the plain surface plus every
    /// transform listed, one drawn per hop by depth-rewarded Thompson
    /// sampling. Empty leaves the choice to `two_phase`.
    pub surfaces: Vec<crate::methods::two_phase::TwoPhase>,
    /// Container half-width, applied when a move is generated.
    pub container: f64,
    /// Closest approach enforced before a trial is relaxed.
    pub min_separation: f64,
}

impl Config {
    /// Canonical, versioned JSON for the fully resolved configuration.
    pub fn resolved_json(&self) -> Result<String, serde_json::Error> {
        #[derive(serde::Serialize)]
        struct ResolvedConfig<'a> {
            schema: &'static str,
            #[serde(flatten)]
            config: &'a Config,
        }

        serde_json::to_string(&ResolvedConfig {
            schema: "anneal-cluster-config-v1",
            config: self,
        })
    }

    /// SHA-256 of [`Config::resolved_json`] as lowercase hexadecimal.
    pub fn resolved_sha256(&self) -> Result<String, serde_json::Error> {
        use sha2::{Digest, Sha256};

        let digest = Sha256::digest(self.resolved_json()?.as_bytes());
        Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
    }

    /// Settings for `n_points` at the campaign's measured defaults.
    /// The measured configuration: the stack every layer of which beat or
    /// matched its paired control across four cluster morphologies.
    ///
    /// Composed surface relocations paying one acceptance test (LJ75 49/144
    /// against 17/144, Bayes factor 3104 with the arm allocator), Normal-Gamma
    /// Thompson allocation rewarded by depth, and tabu on stall (LJ98 40/72
    /// against 20/72, Bayes factor 43.8). Neutral where its mechanisms are not
    /// needed: 55/72 against 55/72 on the 38-point double funnel and 47-48 of
    /// 48 on the 55-point single funnel. Reference GMIN at matched
    /// potential-call budgets: 37/48, 0/48, 0/48.
    ///
    /// [`Config::for_cluster`] remains the plain Wales-Doye protocol, kept as
    /// the comparison baseline; this is what a caller who wants answers should
    /// start from.
    ///
    /// LeanBurst includes the SOAP pullback (analytic \(J^{+}\) of stacked
    /// local power spectra). The hop target is the observed-cloud residual
    /// `2p − μ`, the same map used on molecules and slabs: partitioned by
    /// observed species, never by a CNA class or an fcc prototype.
    /// Thompson allocates SOAP with surface, single, burst and sym. The
    /// return screen and stall symmetrisation are on; Ih-dominated stalls
    /// withhold symmetrise rather than invent a missing packing.
    ///
    /// Basin identity stays the measured pair-spectrum merge at 0.7.
    /// [`Config::packing_superbasin`] is the unmeasured SOAP-packing
    /// keying and adaptive-height stack.
    pub fn recommended(n_points: usize) -> Self {
        let mut cfg = Self::for_cluster(n_points);
        cfg.move_library = MoveLibrary::LeanBurst;
        cfg.allocate_moves = true;
        cfg.depth_reward = true;
        cfg.tabu_on_stall = true;
        cfg.return_screen = true;
        cfg.symmetrise_on_stall = true;
        cfg.soap_class_residual = false;
        cfg.soap_mode = SoapProposalMode::Flexible;
        cfg
    }

    /// Unmeasured SOAP-packing superbasin on top of [`Config::recommended`].
    ///
    /// Unit high-`l` mean SOAP merge 0.10 plus adaptive height with
    /// twenty revisits. Hit rates are not the recommended LJ38/LJ75
    /// campaign numbers.
    pub fn packing_superbasin(n_points: usize) -> Self {
        let mut cfg = Self::recommended(n_points);
        // One deposit of 0.25 exceeds the measured LJ75 intra-funnel
        // gap (~0.09-0.18). That empties a basin on the first revisit
        // and the next start is another ico draw. Adaptive height with
        // twenty revisits is the AS-KMC N_f analogue: fill the occupied
        // packing before the exit is cheap.
        cfg.adaptive_height = true;
        cfg.height_revisits = 20.0;
        #[cfg(feature = "featomic")]
        {
            cfg.keying = Keying::SoapPacking;
            cfg.merge_radius = crate::featomic_hop::SOAP_PACK_MERGE;
        }
        cfg
    }

    /// CNA 555 / Ih packing diagnostics apply only to a monoatomic
    /// cluster. A molecule or a slab has species or a frozen frame;
    /// those are not Honeycutt-Andersen environments.
    pub fn packing_cna_applies(&self) -> bool {
        self.species.is_none()
            && self.active_region.is_none()
            && self.frozen.is_none()
            && !self.move_library.is_molecular()
            && !self.move_library.is_rigid_body()
    }

    /// Recommended flags, with the two hand-set scalars replaced by
    /// derived ones: budget-window temperature (`θ = 1/2` inside the
    /// descent window) and the cost-asymmetric Bayes screen
    /// `τ = (R-S)/(2R-S)` from [`crate::screen::cost_asymmetric_threshold`].
    ///
    /// This is not the measured `recommended` configuration. Hit rates
    /// for this stack are not claimed until a campaign records them.
    pub fn derived(n_points: usize) -> Self {
        let mut cfg = Self::recommended(n_points);
        cfg.budget_window = true;
        cfg.bayes_screen = true;
        cfg.bayes_threshold =
            crate::screen::cost_asymmetric_threshold(cfg.screen_steps, cfg.relax_steps);
        cfg
    }

    /// Creates the reduced-unit Lennard-Jones cluster configuration.
    pub fn for_cluster(n_points: usize) -> Self {
        Self::with_scales(
            n_points,
            LennardJonesPreset::REDUCED_SCALE,
            LennardJonesPreset::REDUCED_SCALE,
        )
    }

    /// Lennard-Jones preset expressed against declared physical scales.
    pub fn with_scales(n_points: usize, length_scale: f64, energy_scale: f64) -> Self {
        assert!(
            length_scale.is_finite() && length_scale > 0.0,
            "length_scale must be finite and positive"
        );
        assert!(
            energy_scale.is_finite() && energy_scale > 0.0,
            "energy_scale must be finite and positive"
        );
        Self {
            n_points,
            length_scale,
            energy_scale,
            move_library: MoveLibrary::Atomic,
            neighbour_cutoff: LennardJonesPreset::NEIGHBOUR_CUTOFF * length_scale,
            symmetrise_cutoff: LennardJonesPreset::SYMMETRISE_CUTOFF * length_scale,
            temperature: LennardJonesPreset::TEMPERATURE * energy_scale,
            bias_height: LennardJonesPreset::BIAS_HEIGHT * energy_scale,
            bias_gamma: 5.0,
            // Calibrated against the descriptor it is compared with, not
            // guessed. Over 75-point minima the sorted-pair distance between
            // independent minima is 0.9212 at the closest with a median of
            // 3.28, while a structure one hop away sits at 0.4766 to 0.58.
            // The multiplier separates a return from a genuinely different
            // minimum while remaining proportional to the declared scale.
            merge_radius: LennardJonesPreset::MERGE_RADIUS * length_scale,
            keying_sigma: 2.5 * length_scale,
            shape_keyed: false,
            theta: 0.5,
            budget_window: false,
            allocate_moves: false,
            adaptive_height: false,
            max_hops: None,
            replicas: 1,
            swap_period: 50,
            bias_by_rung: false,
            keying: Keying::Distances,
            contextual_moves: false,
            contextual_floor: 0.1,
            bayes_screen: false,
            flat_histogram: false,
            flat_sweep: 400,
            flat_quantile: 0.5,
            statistical_temperature: false,
            energy_bias: false,
            depth_reward: false,
            soft_perturb: false,
            soft_modes: 6,
            soft_steps: 30,
            cov_perturb: false,
            staged_quench: false,
            settle_iters: 20,
            group_cutoff: LennardJonesPreset::GROUP_CUTOFF * length_scale,
            covalent_cutoff: LennardJonesPreset::COVALENT_CUTOFF * length_scale,
            species: None,
            bond_tolerance: 1.25,
            frozen: None,
            active_region: None,
            bayes_exploration: 0.1,
            bayes_threshold: 0.05,
            bayes_warmup: 300,
            track_funnels: false,
            funnel_period: 20_000,
            symmetrise_on_stall: false,
            continuous_symmetry: ContinuousSymmetry::Off,
            symmetry_tolerance: LennardJonesPreset::SYMMETRY_TOLERANCE * length_scale,
            symmetry_merge_radius: LennardJonesPreset::SYMMETRY_MERGE_RADIUS * length_scale,
            symmetrise_patience: 2_000,
            tabu_on_stall: false,
            tabu_capacity: 8,
            angular_moves: false,
            angular_target: 0.5,
            restart_on_stall: false,
            restart_patience: 5_000,
            calibrate_radius: false,
            calibrate_quantile: 0.9,
            calibrate_warmup: 200,
            minima_hopping: false,
            escape_lanczos_steps: 16,
            escape_epsilon: LennardJonesPreset::ESCAPE_EPSILON * length_scale,
            escape_amplitude: LennardJonesPreset::ESCAPE_AMPLITUDE * length_scale,
            escape_overshoot: 1.5,
            escape_max_climb: 24,
            escape_on_stall: false,
            trail_on_stall: false,
            escape_stall_patience: 5_000,
            escape_stall_factor: 2.0,
            ladder_top: 4.0,
            return_screen: false,
            soap_class_residual: false,
            soap_mode: SoapProposalMode::Flexible,
            return_polish: 0,
            return_polish_after: 0,
            path_on_stall: false,
            stall_patience: 60,
            path_images: 9,
            anneal_diversity: false,
            diversity_floor: 0.75,
            height_revisits: 4.0,
            screen_margin: LennardJonesPreset::SCREEN_MARGIN * energy_scale,
            screen_steps: 25,
            adaptive_screen: false,
            record_gradient: LennardJonesPreset::RECORD_GRADIENT * energy_scale / length_scale,
            trace_quenched: false,
            polish_records: 0,
            surrogate_tolerance: 0.5,
            delayed_acceptance: false,
            construct_width: 4,
            probe_screen: false,
            quench_warmup: 4,
            quench_confidence: 2.0,
            relax_steps: 200,
            two_phase: None,
            reoccupy: None,
            reoccupy_interval: 5_000,
            surfaces: Vec::new(),
            // Calibrated against published minima: the largest atomic distance
            // from the centre of mass divides by N^(1/3) to between 0.46 and
            // 0.63, and the literature's 2.5 N^(1/3) is sized for a method
            container: LennardJonesPreset::CONTAINER_RADIUS
                * length_scale
                * (n_points as f64).cbrt(),
            min_separation: LennardJonesPreset::MIN_SEPARATION * length_scale,
        }
    }

    /// Basin-hopping settings for a rigid TIP4P water cluster.
    ///
    /// `n_molecules` is the number of waters. The state is six coordinates
    /// per molecule (centre of mass plus an exponential-map rotation). The
    /// move library is Wales--Hodges translation and rotation, each with
    /// its own step, not the atomic Cartesian kernels.
    pub fn for_tip4p(n_molecules: usize) -> Self {
        assert!(
            n_molecules >= 2,
            "a TIP4P cluster needs at least two waters"
        );
        let length_scale = crate::potentials::SIGMA;
        let mut cfg = Self::with_scales(n_molecules, length_scale, 1.0);
        cfg.move_library = MoveLibrary::RigidBody {
            n_molecules,
            translate_step: 0.35,
            rotate_step: 0.40,
        };
        cfg.temperature = 2.0;
        cfg.bias_height = 2.0;
        cfg.screen_margin = 25.0;
        cfg.merge_radius = 0.6;
        cfg.record_gradient = 1.0e-3;
        cfg.neighbour_cutoff = 3.5;
        cfg.min_separation = 2.70;
        cfg.container = 3.5 * 2.75 * (n_molecules as f64).cbrt();
        cfg.soap_mode = SoapProposalMode::Off;
        cfg.angular_moves = false;
        cfg.allocate_moves = true;
        cfg.relax_steps = 80;
        cfg.screen_steps = 15;
        cfg
    }

    /// Species-aware molecular preset with rigid groups.
    pub fn for_molecular(species: Vec<u32>, groups: Vec<Vec<usize>>, energy_scale: f64) -> Self {
        assert!(!species.is_empty(), "species must not be empty");
        let length_scale = MolecularPreset::COVALENT_DIAMETER
            * species
                .iter()
                .copied()
                .map(covalent_radius)
                .fold(0.0_f64, f64::max);
        assert!(length_scale > 0.0, "species must have known covalent radii");
        let mut cfg = Self::with_scales(species.len(), length_scale, energy_scale);
        cfg.group_cutoff = MolecularPreset::GROUP_CUTOFF * length_scale;
        cfg.move_library = MoveLibrary::Molecular {
            groups: groups.clone(),
            reactive: false,
        };
        cfg.species = Some(species);
        cfg
    }

    /// Measured allocation and stall controls over a molecular move library.
    ///
    /// SOAP is the same observed-cloud residual as the cluster hop:
    /// `2p − μ` within each observed atomic number. No CNA, no fcc
    /// prototype. A slab sets `active_region`; frozen atoms stay as
    /// SOAP neighbours and do not move.
    pub fn recommended_molecular(
        species: Vec<u32>,
        groups: Vec<Vec<usize>>,
        energy_scale: f64,
    ) -> Self {
        let mut cfg = Self::for_molecular(species, groups, energy_scale);
        cfg.allocate_moves = true;
        cfg.depth_reward = true;
        cfg.tabu_on_stall = true;
        cfg.soap_class_residual = false;
        cfg.soap_mode = SoapProposalMode::Flexible;
        cfg.adaptive_height = true;
        cfg.height_revisits = 20.0;
        #[cfg(feature = "featomic")]
        {
            cfg.keying = Keying::SoapPacking;
            cfg.merge_radius = crate::featomic_hop::SOAP_PACK_MERGE;
        }
        cfg
    }

    /// Radius of the preset's initial cluster sphere.
    pub fn start_radius(&self) -> f64 {
        self.container
    }

    /// Proposal mixture for use by the general [`crate::sampler::Sampler`].
    pub fn proposal_kernel(&self) -> ClusterProposal {
        ClusterProposal::new(self.move_library.kernels(self))
    }
}
