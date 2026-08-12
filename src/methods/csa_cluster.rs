//! A bank of chains under an annealed diversity threshold.
//!
//! Ties [`crate::methods::bank`] to the cluster driver: the bank decides where
//! to search from and what to keep, [`crate::methods::cluster_hopping::run`]
//! does the searching, and [`crate::diversity`] narrows the threshold as the
//! budget runs down.
//!
//! The division of labour is the point. A single biased chain at 75 points
//! reaches the global minimum in 4 runs of 8 at three million charged
//! evaluations, and the four that fail all fail the same way: they settle in
//! the icosahedral funnel and the decahedron is never seen. Nothing in a single
//! chain notices that, because from inside the funnel the search looks healthy.
//! A bank does notice, because it holds solutions from more than one funnel at
//! once and spends its next start on the one it knows least about. With
//! featomic the distance and the acquisition morphology are the SOAP
//! unit mean high-`l` SOAP, not a leftover-RMS profile or a CNA class.
//!
//! The budget is split, not multiplied. A bank of eight running eight chains of
//! an eighth the length costs what one chain costs, which is the comparison
//! that matters; a bank that quietly spends eight times the budget has proved
//! nothing.

use crate::bias::{BasinBias, Fingerprint, SortedPairs};
use crate::diversity::DiversityAnnealer;
use crate::funnel_bo::FunnelModel;
use crate::methods::bank::{Admission, Bank};
use crate::methods::cluster_hopping::ClusterFingerprint;
use crate::methods::cluster_hopping::{Config, GradFn, Ledger, Outcome, Relax, random_cluster};
use crate::methods::splice::cut_and_splice;
#[cfg(not(feature = "featomic"))]
use crate::structure::cna_descriptor;

/// Morphology the acquisition model fits. SOAP leftover when featomic
/// is on; CNA fractions otherwise.
fn morphology(x: ArrayView1<f64>, cfg: &Config, bond_cutoff: f64) -> Array1<f64> {
    #[cfg(feature = "featomic")]
    {
        let _ = bond_cutoff;
        crate::featomic_hop::soap_morphology(
            x,
            3.5 * cfg.length_scale,
            cfg.species.as_deref(),
            None,
        )
    }
    #[cfg(not(feature = "featomic"))]
    {
        cna_descriptor(x, cfg.n_points, bond_cutoff)
    }
}
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// How the bank is run.
#[derive(Debug, Clone)]
pub struct BankConfig {
    /// Solutions held at once.
    pub capacity: usize,
    /// Charged evaluations given to each chain before it reports back.
    ///
    /// Short enough that a bad region is abandoned quickly and long enough that
    /// a chain settles into a minimum: a slice that ends mid-descent offers the
    /// bank a structure that says nothing about where it was going.
    pub slice: usize,
    /// Chains run from random starts before the bank is first used, to give
    /// `Dcut` a scale taken from the data.
    pub seeding: usize,
    /// Floor of the `Dcut` schedule, as a fraction of its start.
    pub dcut_floor: f64,
    /// Share of rounds spent mixing two members instead of searching from one.
    ///
    /// Without mixing the bank does not hold funnels apart, it holds variants
    /// apart. Measured at 75 points with a bank of thirty and a thousand
    /// slices, every member ended between -396.28 and -396.19: thirty
    /// icosahedral structures, each shape-distinct under the threshold and all
    /// in the same funnel, because each member's chain descends on its own and
    /// nothing ever puts two of them together. The published method mixes by
    /// cutting one solution and splicing in part of another; that is
    /// [`crate::methods::splice`], and it is the mix used when this fraction
    /// fires. The trial is not a minimum: the caller quenches it and offers
    /// the relaxed structure to the bank.
    pub mix_fraction: f64,
    /// Independent splice trials drawn from one pair in a mixing round.
    ///
    /// Each trial is a different plane and a separate quench, so one mix can
    /// offer several children without starting a chain slice for each.
    pub mix_images: usize,
    /// Independent random kicks of a seed in a mixing round, quenched
    /// like the splice images. Lee, Lee and Scheraga draw twenty
    /// splices and ten random perturbations a seed.
    pub random_images: usize,
    /// Complete passes over the bank before a deadlock enlargement.
    ///
    /// One pass is every member used once as a seed. After this many
    /// passes Lee, Lee and Scheraga add random minima to both banks
    /// and reset `Dcut` to `Dave/2`.
    pub deadlock_iters: usize,
    /// Random minima injected on deadlock. Zero turns the enlargement off.
    pub deadlock_inject: usize,
    /// Choose the next member to search from by expected improvement over
    /// morphology, rather than by which has been used least.
    ///
    /// Least-used is a round robin, and a round robin over a bank that has
    /// collapsed into one funnel searches that funnel evenly. Measured at 75
    /// points, every member ended between -396.28 and -396.19: thirty
    /// icosahedral variants, each distinct under the threshold, and the
    /// selection spread its starts across all of them. Expected improvement
    /// scores a morphology the search has no evidence about above one it has
    /// sampled repeatedly and found mediocre, which is the distinction a round
    /// robin cannot make. See [`crate::funnel_bo`].
    pub acquisition: bool,
}

impl Default for BankConfig {
    fn default() -> Self {
        Self {
            capacity: 8,
            slice: 60_000,
            seeding: 8,
            // Dave/5 relative to a start of Dave/2, which is the published
            // schedule. A floor of a tenth anneals four times too far.
            dcut_floor: 0.4,
            mix_fraction: 0.5,
            mix_images: 7,
            random_images: 0,
            deadlock_iters: 3,
            deadlock_inject: 0,
            acquisition: false,
        }
    }
}

impl BankConfig {
    /// The published CSA schedule: bank 50, Dave/2 to Dave/5, twenty
    /// splices and ten random images a mix, deadlock after three
    /// passes with fifty injected minima.
    pub fn published() -> Self {
        Self {
            capacity: 50,
            slice: 60_000,
            seeding: 50,
            dcut_floor: 0.4,
            mix_fraction: 0.5,
            mix_images: 20,
            random_images: 10,
            deadlock_iters: 3,
            deadlock_inject: 50,
            acquisition: false,
        }
    }
}

/// What a bank run did.
#[derive(Debug, Clone)]
pub struct BankOutcome {
    /// Lowest value found anywhere.
    pub best: f64,
    /// The structure attaining it.
    pub best_state: Option<Array1<f64>>,
    /// Chains run.
    pub slices: usize,
    /// Hops summed over the chains, so the report says what the run did rather
    /// than reporting a default.
    pub hops: usize,
    /// Basins summed over the chains. Not distinct basins across the run: each
    /// chain keeps its own bias and its own numbering.
    pub basins: usize,
    /// Charged evaluations spent.
    pub charged: usize,
    /// Candidates that improved the member they resembled.
    pub improved: usize,
    /// Candidates that resembled nothing in the bank.
    pub novel: usize,
    /// Candidates discarded as near-copies.
    pub duplicates: usize,
    /// Screened trials summed over slices.
    pub screened_out: usize,
    /// Return-screened trials summed over slices.
    pub returned: usize,
    /// Hop, charged evaluations, basins and value at each new campaign best.
    ///
    /// Merged from the slices with each slice's spend offset by what the
    /// campaign had already spent, because a slice keeps its own sub-ledger and
    /// its counts start from zero. Without the merge a caller asking this arm
    /// for a first encounter time gets a censored answer from every run,
    /// including the ones that found the answer.
    pub improvements: Vec<(usize, usize, usize, f64)>,
    /// Morphologies the acquisition model holds.
    pub morphologies: usize,
    /// Rounds spent mixing two members.
    pub mixes: usize,
    /// Images from those rounds that were admitted to the bank.
    pub mix_admitted: usize,
    /// Images from those rounds that beat both of their endpoints.
    pub mix_below_both: usize,
    /// `Dcut` at the start and at the end.
    pub dcut: (f64, f64),
    /// Energies held in the bank at the end, ascending.
    pub bank: Vec<f64>,
    /// Deadlock enlargements that fired.
    pub deadlocks: usize,
    /// Random minima those enlargements admitted.
    pub injected: usize,
}

/// One chain against its own sub-ledger, settled up against the caller's.
///
/// The sub-ledger is not an accounting nicety. Handing the campaign ledger to
/// the inner run makes every budget-aware mechanism inside it, the
/// budget-window temperature above all, see the whole campaign's budget rather
/// than the slice it was given.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn slice_run_inner<'g>(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: &mut Option<&mut GradFn<'g>>,
    bias: &mut BasinBias<ClusterFingerprint>,
    start: ArrayView1<f64>,
    budget: usize,
    rng: &mut StdRng,
) -> (Outcome, usize) {
    let before = ledger.spent();
    let mut slice_ledger = Ledger::new(budget.min(ledger.remaining()));
    let out = crate::methods::cluster_hopping::run_with_bias(
        cfg,
        start,
        &mut slice_ledger,
        relax,
        grad.as_deref_mut(),
        bias,
        rng,
    );
    ledger.charge_many(slice_ledger.spent());
    if let Some(st) = slice_ledger.best_state.as_ref() {
        ledger.record(slice_ledger.best, st.view());
    }
    (out, before)
}

/// As above, returning only the outcome for callers that do not merge traces.
#[allow(clippy::too_many_arguments)]
fn slice_run<'g>(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    grad: &mut Option<&mut GradFn<'g>>,
    bias: &mut BasinBias<ClusterFingerprint>,
    start: ArrayView1<f64>,
    budget: usize,
    rng: &mut StdRng,
) -> (Outcome, usize) {
    slice_run_inner(cfg, ledger, relax, grad, bias, start, budget, rng)
}

/// Runs a bank of chains until the ledger is spent.
///
/// `distance` decides when two solutions are the same, and is the caller's:
/// with a shape distance `Dcut` is a length and transfers between sizes, which
/// a threshold in descriptor space does not.
pub fn run<'g, D>(
    cfg: &Config,
    bank_cfg: &BankConfig,
    ledger: &mut Ledger,
    relax: Relax<'_>,
    mut grad: Option<&mut GradFn<'g>>,
    mut distance: D,
    seed: u64,
) -> BankOutcome
where
    D: FnMut(ArrayView1<f64>, ArrayView1<f64>) -> f64,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let total = ledger.remaining();
    let mut slices = 0usize;
    let mut improved = 0usize;
    let mut duplicates = 0usize;
    let mut hops = 0usize;
    let mut basins = 0usize;
    let mut screened_out = 0usize;
    let mut returned = 0usize;
    // A model of how low each morphology goes, over the share of points in
    // each local environment. Five numbers, so a Gaussian process over it is
    // exact and cheap; the coordinates would not be.
    let mut funnel = FunnelModel::new(0.15, 20.0, 1e-2);
    // Between the first and second neighbour shells, where the radial
    // distribution is near zero so the bond set is insensitive to the exact
    // value. Expressed through the configured minimum separation so it follows
    // the potential's length scale rather than assuming Lennard-Jones.
    let bond_cutoff = cfg.min_separation * (1.39 / 0.85);
    let mut improvements: Vec<(usize, usize, usize, f64)> = Vec::new();
    let mut campaign_best = f64::INFINITY;
    let mut mixes = 0usize;
    let mut mix_admitted = 0usize;
    let mut mix_below_both = 0usize;

    // The threshold is set from the seeding population below. Nothing is
    // judged against this one: the seeding phase bypasses the rule entirely.
    let mut bank = Bank::new(bank_cfg.capacity, 1.0);
    // One bias for the whole run, not one per chain. What it holds is where the
    // landscape has already been filled in, which does not belong to whichever
    // chain did the filling.
    let mut bias = BasinBias::new(
        ClusterFingerprint::of_config(cfg, &ndarray::Array1::zeros(0)),
        cfg.merge_radius,
        cfg.bias_height,
        cfg.bias_gamma,
    );

    // Seeding: independent chains from random starts, which is also how the
    // threshold gets a scale.
    for _ in 0..bank_cfg.seeding {
        if ledger.remaining() == 0 {
            break;
        }
        let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng);
        let (out, spent_before) = slice_run(
            cfg,
            ledger,
            relax,
            &mut grad,
            &mut bias,
            start.view(),
            bank_cfg.slice,
            &mut rng,
        );
        // Offset each slice's counts by what the campaign had already spent,
        // since a slice keeps its own sub-ledger and starts from zero.
        for &(h, c, b, e) in &out.improvements {
            if e < campaign_best - 1e-12 && improvements.len() < 512 {
                campaign_best = e;
                improvements.push((hops + h, spent_before + c, b, e));
            }
        }
        slices += 1;
        hops += out.hops;
        basins += out.basins;
        screened_out += out.screened_out;
        returned += out.returned;
        if let Some(s) = out.best_state.as_ref() {
            if bank_cfg.acquisition {
                funnel.observe(morphology(s.view(), cfg, bond_cutoff).view(), out.best);
            }
            bank.seed(s.view(), out.best);
        }
    }

    let mean = bank.mean_distance(&mut distance);
    let fallback = {
        #[cfg(feature = "featomic")]
        {
            crate::featomic_hop::SOAP_DCUT_FALLBACK
        }
        #[cfg(not(feature = "featomic"))]
        {
            cfg.merge_radius
        }
    };
    let mut schedule = match mean {
        Some(m) => DiversityAnnealer::from_initial(0.5 * m),
        // Seeding landed in one packing: Dave is zero. merge_radius is a
        // length (0.7) and SOAP distances live on the unit sphere
        // (ico-Marks = 0.163). Using the length as Dcut makes Marks a
        // duplicate of Mackay. The SOAP floor is below that gap.
        None => DiversityAnnealer::from_initial(fallback),
    }
    .with_final_fraction(bank_cfg.dcut_floor);
    let dcut0 = schedule.current();
    bank.dcut = dcut0;
    let mut deadlocks = 0usize;
    let mut injected = 0usize;
    let mut passes = 0usize;

    while ledger.remaining() > 0 {
        let progress = 1.0 - ledger.remaining() as f64 / total.max(1) as f64;
        bank.dcut = schedule.threshold(progress);

        let i = if bank_cfg.acquisition {
            // The member whose morphology the model rates most promising,
            // counting uncertainty: a region never sampled scores on its
            // variance, which is how the search reaches a funnel it has no
            // evidence about.
            let mut best = None;
            let mut best_ei = f64::NEG_INFINITY;
            for (k, m) in bank.members().iter().enumerate() {
                let d = morphology(m.state.view(), cfg, bond_cutoff);
                let ei = funnel.expected_improvement(d.view());
                if ei > best_ei {
                    best_ei = ei;
                    best = Some(k);
                }
            }
            match best {
                Some(k) => k,
                None => break,
            }
        } else {
            match bank.next_start() {
                Some(i) => i,
                None => break,
            }
        };
        bank.mark_used(i);

        let min_hits = bank.members().iter().map(|m| m.hits).min().unwrap_or(0);
        if min_hits > passes {
            passes = min_hits;
            if bank_cfg.deadlock_inject > 0
                && bank_cfg.deadlock_iters > 0
                && passes % bank_cfg.deadlock_iters == 0
            {
                deadlocks += 1;
                bank.grow(bank_cfg.deadlock_inject);
                for _ in 0..bank_cfg.deadlock_inject {
                    if ledger.remaining() == 0 {
                        break;
                    }
                    let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng);
                    let (e, x) = relax(ledger, start.view(), cfg.relax_steps);
                    ledger.record(e, x.view());
                    if bank.inject(x.view(), e) {
                        injected += 1;
                    }
                }
                if let Some(m) = bank.mean_distance(&mut distance) {
                    schedule = DiversityAnnealer::from_initial(0.5 * m)
                        .with_final_fraction(bank_cfg.dcut_floor);
                    bank.dcut = schedule.current();
                }
            }
        }

        // A mixing round: cut-and-splice this member with another, quench the
        // trial, and offer what comes back. One quench per trial, charged on
        // the shared ledger, so a bank of k does not spend k times the budget.
        if bank.len() >= 2 && rng.random::<f64>() < bank_cfg.mix_fraction {
            // The partner comes from the working bank or from the first bank,
            // as in Lee, Lee and Scheraga. Drawing only from the working bank
            // is what let every partner end up in the same funnel as every
            // member; the first bank is the half of the population that cannot
            // collapse.
            let from_first = !bank.first_bank().is_empty() && rng.random::<bool>();
            let (b, eb) = if from_first {
                let j = rng.random_range(0..bank.first_bank().len());
                let m = &bank.first_bank()[j];
                (m.state.clone(), m.energy)
            } else {
                let mut j = rng.random_range(0..bank.len());
                if j == i {
                    j = (j + 1) % bank.len();
                }
                let m = &bank.members()[j];
                (m.state.clone(), m.energy)
            };
            let a = bank.members()[i].state.clone();
            let ea = bank.members()[i].energy;
            let species = cfg.species.as_deref();
            let n_trials = bank_cfg.mix_images.max(1);
            mixes += 1;
            for _ in 0..n_trials {
                if ledger.remaining() == 0 {
                    break;
                }
                let trial =
                    cut_and_splice(a.view(), b.view(), species, cfg.min_separation, &mut rng);
                let (e, x) = relax(ledger, trial.view(), cfg.relax_steps);
                ledger.record(e, x.view());
                if !matches!(
                    bank.offer(x.view(), e, &mut distance),
                    Admission::Duplicate(_) | Admission::Rejected
                ) {
                    mix_admitted += 1;
                }
                if e < ea.min(eb) {
                    mix_below_both += 1;
                }
            }
            // Lee, Lee and Scheraga also draw random perturbations of
            // the seed, not only splices. Small kicks, then the same
            // quench and the same replacement rule.
            for _ in 0..bank_cfg.random_images {
                if ledger.remaining() == 0 {
                    break;
                }
                let mut kick = a.clone();
                for v in kick.iter_mut() {
                    *v += rng.random_range(-0.15..0.15);
                }
                let (e, x) = relax(ledger, kick.view(), cfg.relax_steps);
                ledger.record(e, x.view());
                if !matches!(
                    bank.offer(x.view(), e, &mut distance),
                    Admission::Duplicate(_) | Admission::Rejected
                ) {
                    mix_admitted += 1;
                }
                if e < ea.min(eb) {
                    mix_below_both += 1;
                }
            }
            continue;
        }

        let start = bank.members()[i].state.clone();
        // Started from the member, not from its exact coordinates: a chain
        // begun at a minimum with a bias that has not seen it yet spends its
        // first hops rediscovering where it is.
        let mut kick = start.clone();
        for v in kick.iter_mut() {
            *v += rng.random_range(-0.15..0.15);
        }

        let (out, spent_before) = slice_run(
            cfg,
            ledger,
            relax,
            &mut grad,
            &mut bias,
            kick.view(),
            bank_cfg.slice,
            &mut rng,
        );
        // Offset each slice's counts by what the campaign had already spent,
        // since a slice keeps its own sub-ledger and starts from zero.
        for &(h, c, b, e) in &out.improvements {
            if e < campaign_best - 1e-12 && improvements.len() < 512 {
                campaign_best = e;
                improvements.push((hops + h, spent_before + c, b, e));
            }
        }
        slices += 1;
        hops += out.hops;
        basins += out.basins;
        screened_out += out.screened_out;
        returned += out.returned;
        if let Some(s) = out.best_state.as_ref() {
            if bank_cfg.acquisition {
                funnel.observe(morphology(s.view(), cfg, bond_cutoff).view(), out.best);
            }
            match bank.offer(s.view(), out.best, &mut distance) {
                Admission::Improved(_) => improved += 1,
                Admission::Duplicate(_) => duplicates += 1,
                _ => {}
            }
        }
    }

    let mut energies: Vec<f64> = bank.members().iter().map(|m| m.energy).collect();
    energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    BankOutcome {
        best: ledger.best,
        best_state: ledger.best_state.clone(),
        slices,
        improvements,
        morphologies: funnel.len(),
        mixes,
        mix_admitted,
        mix_below_both,
        hops,
        basins,
        screened_out,
        returned,
        charged: ledger.spent(),
        improved,
        novel: bank.novel,
        duplicates,
        dcut: (dcut0, bank.dcut),
        bank: energies,
        deadlocks,
        injected,
    }
}

/// Euclidean distance between sorted pairwise-distance spectra.
///
/// The default when no shape matching is available. It is a distance in
/// descriptor space, so its threshold does not transfer between sizes; with the
/// `ira` feature the caller should pass a shape distance instead.
pub fn spectrum_distance(n_points: usize) -> impl FnMut(ArrayView1<f64>, ArrayView1<f64>) -> f64 {
    let f = SortedPairs { n_points };
    move |a, b| {
        let da = f.describe(a);
        let db = f.describe(b);
        if da.len() != db.len() {
            return f64::INFINITY;
        }
        da.iter()
            .zip(db.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }
}

/// Lee, Lee and Scheraga coordination-histogram distance.
///
/// First- and second-neighbour shells at `r1` and `r2` (1.35 and 1.70
/// in reduced LJ units). \(H(s,n)\) is how many atoms have \(n\)
/// neighbours in shell \(s\). The published \(D\) weights the core:
///
/// \[
/// D(k,k')=\sum_n n\bigl(2|H^k(1,n)-H^{k'}(1,n)|+|H^k(2,n)-H^{k'}(2,n)|\bigr).
/// \]
pub fn coordination_histogram_distance(
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    r1: f64,
    r2: f64,
) -> f64 {
    let ha = shell_histograms(a, r1, r2);
    let hb = shell_histograms(b, r1, r2);
    let m = ha.0.len().max(hb.0.len()).max(ha.1.len()).max(hb.1.len());
    let mut d = 0.0;
    for n in 0..m {
        let h1a = *ha.0.get(n).unwrap_or(&0) as f64;
        let h1b = *hb.0.get(n).unwrap_or(&0) as f64;
        let h2a = *ha.1.get(n).unwrap_or(&0) as f64;
        let h2b = *hb.1.get(n).unwrap_or(&0) as f64;
        d += n as f64 * (2.0 * (h1a - h1b).abs() + (h2a - h2b).abs());
    }
    d
}

fn shell_histograms(x: ArrayView1<f64>, r1: f64, r2: f64) -> (Vec<usize>, Vec<usize>) {
    let n = x.len() / 3;
    let r1sq = r1 * r1;
    let r2sq = r2 * r2;
    let mut h1 = vec![0usize; n];
    let mut h2 = vec![0usize; n];
    for i in 0..n {
        let mut n1 = 0usize;
        let mut n2 = 0usize;
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = x[3 * i] - x[3 * j];
            let dy = x[3 * i + 1] - x[3 * j + 1];
            let dz = x[3 * i + 2] - x[3 * j + 2];
            let r2ij = dx * dx + dy * dy + dz * dz;
            if r2ij <= r1sq {
                n1 += 1;
            } else if r2ij <= r2sq {
                n2 += 1;
            }
        }
        if n1 < h1.len() {
            h1[n1] += 1;
        }
        if n2 < h2.len() {
            h2[n2] += 1;
        }
    }
    (h1, h2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A separable washboard: `sum -cos(2 pi v) + 0.01 v^2`, so there is a
    /// minimum near every integer coordinate and the tilt makes them differ in
    /// energy. Enough to exercise the accounting and the control flow without a
    /// potential.
    ///
    /// A single-minimum objective will not do here even though it serves the
    /// other tests. Every chain converges to the same answer, so a bank holding
    /// one member is correct behaviour and a test asking for two is testing its
    /// own premise.
    fn washboard(x: ArrayView1<f64>) -> f64 {
        x.iter()
            .map(|v| -(2.0 * std::f64::consts::PI * v).cos() + 0.01 * v * v)
            .sum()
    }

    fn toy_relax(ledger: &mut Ledger, x: ArrayView1<f64>, steps: usize) -> (f64, Array1<f64>) {
        let mut cur = x.to_owned();
        for _ in 0..steps {
            if !ledger.charge() {
                break;
            }
            for v in cur.iter_mut() {
                let g = 2.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * *v).sin()
                    + 0.02 * *v;
                *v -= 0.02 * g;
            }
        }
        let e = washboard(cur.view());
        (e, cur)
    }

    fn small_config(n: usize) -> Config {
        let mut cfg = Config::for_cluster(n);
        cfg.max_hops = Some(40);
        cfg
    }

    /// The accounting claim the comparison rests on: a bank of chains costs
    /// what one chain costs, because the budget is split rather than handed to
    /// each of them.
    #[test]
    fn the_bank_spends_one_budget_not_one_per_chain() {
        let cfg = small_config(8);
        let bank_cfg = BankConfig {
            capacity: 4,
            slice: 400,
            seeding: 4,
            ..BankConfig::default()
        };
        let mut ledger = Ledger::new(5_000);
        let out = run(
            &cfg,
            &bank_cfg,
            &mut ledger,
            &mut toy_relax,
            None,
            spectrum_distance(8),
            11,
        );
        assert!(
            out.charged <= 5_000,
            "spent {} against a budget of 5000",
            out.charged
        );
        assert!(out.slices > 1, "only {} chains ran", out.slices);
    }

    #[test]
    fn the_threshold_is_taken_from_the_seeding_and_narrows() {
        let cfg = small_config(8);
        let bank_cfg = BankConfig {
            capacity: 6,
            slice: 300,
            seeding: 6,
            dcut_floor: 0.1,
            ..BankConfig::default()
        };
        let mut ledger = Ledger::new(12_000);
        let out = run(
            &cfg,
            &bank_cfg,
            &mut ledger,
            &mut toy_relax,
            None,
            spectrum_distance(8),
            3,
        );
        assert!(out.dcut.0 > 0.0, "no starting threshold was set");
        assert!(
            out.dcut.1 <= out.dcut.0 + 1e-12,
            "the threshold went from {} to {}",
            out.dcut.0,
            out.dcut.1
        );
    }

    /// Mixing has to actually run and actually offer what it finds, or the
    /// bank holds variants of one region rather than distinct regions.
    #[test]
    fn mixing_rounds_offer_spliced_children() {
        let cfg = small_config(8);
        let bank_cfg = BankConfig {
            capacity: 6,
            slice: 300,
            seeding: 6,
            dcut_floor: 0.5,
            mix_fraction: 1.0,
            mix_images: 5,
            acquisition: false,
            ..BankConfig::default()
        };
        let mut ledger = Ledger::new(20_000);
        let out = run(
            &cfg,
            &bank_cfg,
            &mut ledger,
            &mut toy_relax,
            None,
            spectrum_distance(8),
            13,
        );
        assert!(out.mixes > 0, "no mixing round ran");
        assert!(
            out.charged <= 20_000,
            "mixing spent past the budget: {}",
            out.charged
        );
    }

    #[test]
    fn mixing_can_be_turned_off() {
        let cfg = small_config(8);
        let bank_cfg = BankConfig {
            capacity: 5,
            slice: 300,
            seeding: 5,
            dcut_floor: 0.5,
            mix_fraction: 0.0,
            mix_images: 5,
            acquisition: false,
            ..BankConfig::default()
        };
        let mut ledger = Ledger::new(9_000);
        let out = run(
            &cfg,
            &bank_cfg,
            &mut ledger,
            &mut toy_relax,
            None,
            spectrum_distance(8),
            2,
        );
        assert_eq!(out.mixes, 0);
    }

    /// A bank that ends holding one solution has not held anything apart.
    #[test]
    fn the_bank_ends_holding_more_than_one_solution() {
        let cfg = small_config(8);
        let bank_cfg = BankConfig {
            capacity: 5,
            slice: 300,
            seeding: 5,
            ..BankConfig::default()
        };
        let mut ledger = Ledger::new(9_000);
        let out = run(
            &cfg,
            &bank_cfg,
            &mut ledger,
            &mut toy_relax,
            None,
            spectrum_distance(8),
            7,
        );
        assert!(
            out.bank.len() >= 2,
            "the bank ended with {} members",
            out.bank.len()
        );
        // Ascending, so a caller reading the first entry gets the best.
        assert!(out.bank.windows(2).all(|w| w[0] <= w[1] + 1e-12));
    }

    #[test]
    fn an_exhausted_budget_stops_the_run() {
        let cfg = small_config(8);
        let bank_cfg = BankConfig {
            capacity: 4,
            slice: 1_000,
            seeding: 4,
            ..BankConfig::default()
        };
        let mut ledger = Ledger::new(500);
        let out = run(
            &cfg,
            &bank_cfg,
            &mut ledger,
            &mut toy_relax,
            None,
            spectrum_distance(8),
            5,
        );
        assert!(out.charged <= 500, "spent {}", out.charged);
    }

    fn ico13() -> Array1<f64> {
        let p = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let verts: [[f64; 3]; 12] = [
            [0.0, 1.0, p],
            [0.0, 1.0, -p],
            [0.0, -1.0, p],
            [0.0, -1.0, -p],
            [1.0, p, 0.0],
            [1.0, -p, 0.0],
            [-1.0, p, 0.0],
            [-1.0, -p, 0.0],
            [p, 0.0, 1.0],
            [-p, 0.0, 1.0],
            [p, 0.0, -1.0],
            [-p, 0.0, -1.0],
        ];
        let s = 1.0 / (1.0 + p * p).sqrt() * 2.0_f64.powf(1.0 / 6.0);
        let mut x = Array1::<f64>::zeros(3 * 13);
        for (i, v) in verts.iter().enumerate() {
            for k in 0..3 {
                x[3 * (i + 1) + k] = s * v[k];
            }
        }
        x
    }

    fn cuboct13() -> Array1<f64> {
        let s = 2.0_f64.powf(1.0 / 6.0);
        let verts = [
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, 1.0],
            [0.0, -1.0, -1.0],
        ];
        let mut y = Array1::<f64>::zeros(3 * 13);
        for (i, v) in verts.iter().enumerate() {
            for k in 0..3 {
                y[3 * (i + 1) + k] = s * v[k];
            }
        }
        y
    }

    /// Lee, Lee and Scheraga eq. (2): a structure is zero from itself,
    /// and two closed packings are not.
    #[test]
    fn published_d_vanishes_on_a_copy_and_separates_packings() {
        let x = ico13();
        let d0 = coordination_histogram_distance(x.view(), x.view(), 1.35, 1.70);
        assert_eq!(d0, 0.0);
        let y = cuboct13();
        let d = coordination_histogram_distance(x.view(), y.view(), 1.35, 1.70);
        assert!(d > 0.0, "ico13 and cuboct13 have published D={d}");
    }

    /// After every member has been a seed once, the published
    /// enlargement adds random minima to both banks.
    #[test]
    fn deadlock_enlarges_the_bank_after_one_pass() {
        let cfg = small_config(8);
        let bank_cfg = BankConfig {
            capacity: 2,
            slice: 80,
            seeding: 2,
            mix_fraction: 0.0,
            deadlock_iters: 1,
            deadlock_inject: 2,
            ..BankConfig::default()
        };
        let mut ledger = Ledger::new(20_000);
        let out = run(
            &cfg,
            &bank_cfg,
            &mut ledger,
            &mut toy_relax,
            None,
            spectrum_distance(8),
            11,
        );
        assert!(out.deadlocks >= 1, "no deadlock fired, bank {:?}", out.bank);
        assert!(
            out.injected >= 1,
            "deadlock injected nothing, bank {:?}",
            out.bank
        );
        assert!(
            out.bank.len() >= 3,
            "the bank did not grow, holding {:?}",
            out.bank
        );
    }
}
