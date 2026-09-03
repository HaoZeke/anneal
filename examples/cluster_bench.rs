//! Cluster global optimisation across more than one potential.
//!
//! One benchmark family is not a result. This runs the same driver, the same
//! charged ledger and the same mechanisms against Lennard-Jones and Morse
//! clusters, so a mechanism that only helps on one of them is visible as such.
//!
//! Morse is the useful second family because its range parameter is a dial on
//! how hard the landscape is, with the same code path either side. Doye and
//! Wales, *Structural consequences of the range of the interatomic potential*,
//! survey 20 to 80 points across that dial; the short-range cases are the ones
//! the literature reports as out of reach for unbiased methods.
//!
//! Usage:
//! `cargo run --release --example cluster_bench -- <potential> <n> <budget> <seeds> [mechanisms]`
//! where `<potential>` is `lj`, or `morse:RHO` such as `morse:6`.

use anneal_core::hmc::hop::HopConfig;
use anneal_core::hmc::metric::MetricKind;
use anneal_core::methods::cluster_hopping::{Config, Keying, LadderMode, Ledger, MoveLibrary};
use anneal_core::methods::cluster_search::{
    Encounter, first_encounter, median_encounter, search, verify,
};
use anneal_core::methods::csa_cluster::{self, BankConfig};
use anneal_core::methods::csa_cluster::{self, BankConfig};
use anneal_core::methods::csa_cluster::{self, BankConfig};
use anneal_core::potentials::{PairKind, PairPotential};
use anneal_core::structure::{cna, ptm, ptm_fractions};

/// The potential named on the command line.
fn parse_potential(spec: &str, n: usize) -> Option<PairPotential> {
    if spec == "lj" {
        return Some(PairPotential::lennard_jones(n));
    }
    let rho: f64 = spec.strip_prefix("morse:")?.parse().ok()?;
    Some(PairPotential::morse(n, rho))
}

/// A name for the report, from what the caller asked for.
fn potential_name(spec: &str) -> String {
    if spec == "lj" {
        "LJ".into()
    } else {
        spec.replace("morse:", "Morse rho=")
    }
}

/// Published global minima, for reporting only; nothing steers by these.
///
/// Lennard-Jones from the Cambridge Cluster Database, Morse from Doye and
/// Wales' table for the same database, in units of the pair well depth.
fn reference(p: PairKind, n: usize) -> Option<f64> {
    match p {
        PairKind::LennardJones => Some(match n {
            13 => -44.326801,
            38 => -173.928427,
            55 => -279.248470,
            75 => -397.492331,
            98 => -543.665361,
            _ => return None,
        }),
        PairKind::Morse { rho } => {
            let r = (rho * 2.0).round() as i64;
            // Taken from the bold entries of Doye and Wales' table, which are
            // the global minima; the table also lists competing minima for the
            // same size and range, and the global one changes structure along
            // a row. Two values here were first taken from non-bold rows, so
            // the success bar was set below the global minimum: at rho = 6 and
            // N = 38 the minimum is 38E at -157.477108, not 38D at -157.406902,
            // and at rho = 14 and N = 55 it is 55C at -220.646208, not 55B at
            // -213.523774. The runs found the true minima and were scored
            // against the wrong ones.
            Some(match (r, n) {
                (12, 38) => -157.477108,
                (12, 55) => -250.286609,
                (12, 75) => -351.472365,
                (20, 38) => -145.849817,
                (20, 55) => -225.814286,
                (20, 75) => -322.643558,
                (28, 38) => -144.321054,
                (28, 55) => -220.646208,
                (28, 75) => -318.407330,
                _ => return None,
            })
        }
    }
}

/// Is the archive made of minima, and do its basins hold distinct structures.
///
/// Three merges are built and the deciding statistic recomputed under each.
/// The point is not the merge but the number after it: expected visits per
/// state, where one means the chain enters each state once on the way through
/// and above one means it is returning.
#[allow(clippy::too_many_arguments)]
fn archive_analysis(
    pot: &PairPotential,
    n: usize,
    counts: &anneal_core::superbasin::HopCounts,
    archive: &[(usize, f64, ndarray::Array1<f64>)],
    shape_sample: usize,
    seed: u64,
) {
    use anneal_core::superbasin::{LumpParams, profile, regroup};
    use std::collections::BTreeMap;

    let params = LumpParams::default();
    let base = profile(counts, &params, 16, 4096, 384, 4096);
    println!(
        "    archive analysis: {} basins in the graph, {} structures stored",
        base.states,
        archive.len()
    );

    // 0. Are these minima at all. The chain carries accepted states, and under
    //    the return screen an accepted state can be a partial quench, so this
    //    has to be established before any statement about the landscape rests
    //    on it. Re-quenched hard, off the ledger, and the drop is the answer.
    let mut opt = anneal_core::methods::warm_lbfgs::WarmLbfgs::default();
    let mut polished: Vec<(usize, f64, ndarray::Array1<f64>)> = Vec::with_capacity(archive.len());
    let mut drops: Vec<f64> = Vec::with_capacity(archive.len());
    let mut grads: Vec<f64> = Vec::with_capacity(archive.len());
    for (b, e, x) in archive {
        opt.forget();
        let (f, xr, _) = opt.minimize(x.view(), 4000, |v| Some(pot.value_and_gradient(v)));
        let g = pot.value_and_gradient(xr.view()).1;
        grads.push(g.iter().fold(0.0_f64, |a, q| a.max(q.abs())));
        drops.push(e - f);
        polished.push((*b, f, xr));
    }
    let quantile = |v: &mut Vec<f64>, q: f64| -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f64 * q) as usize]
    };
    let mut d = drops.clone();
    let mut g = grads.clone();
    println!(
        "      stored states re-quenched: energy fell by {:.4} median, {:.4} at the ninth \
         decile, {:.4} worst; final gradient {:.1e} median, {:.1e} worst",
        quantile(&mut d.clone(), 0.5),
        quantile(&mut d.clone(), 0.9),
        quantile(&mut d, 1.0),
        quantile(&mut g.clone(), 0.5),
        quantile(&mut g, 1.0)
    );

    // 1. Shape distance on all pairs, no energy precondition. The distribution
    //    rather than a threshold count, because the question is whether the
    //    basins are near duplicates and that is a property of the distribution.
    #[cfg(feature = "ira")]
    {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;
        let metric = anneal_core::shape::IraMetric::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut idx: Vec<usize> = (0..polished.len()).collect();
        idx.shuffle(&mut rng);
        idx.truncate(shape_sample);
        idx.sort_unstable();
        let mut dists: Vec<f64> = Vec::new();
        let mut close: Vec<(usize, usize)> = Vec::new();
        for a in 0..idx.len() {
            for b in (a + 1)..idx.len() {
                let (i, j) = (idx[a], idx[b]);
                let dd = metric.distance(polished[i].2.view(), polished[j].2.view());
                dists.push(dd);
                if dd < 0.1 {
                    close.push((polished[i].0, polished[j].0));
                }
            }
        }
        let under = |v: &[f64], t: f64| v.iter().filter(|x| **x < t).count();
        let n_pairs = dists.len();
        let mut sorted = dists.clone();
        println!(
            "      shape distance over {} structures, {} pairs, no energy filter: \
             median {:.3}, lower decile {:.3}, minimum {:.3}; {} pairs below 0.7 ({:.4}), \
             {} below 0.1 ({:.5})",
            idx.len(),
            n_pairs,
            quantile(&mut sorted.clone(), 0.5),
            quantile(&mut sorted.clone(), 0.1),
            quantile(&mut sorted, 0.0),
            under(&dists, 0.7),
            under(&dists, 0.7) as f64 / n_pairs.max(1) as f64,
            under(&dists, 0.1),
            under(&dists, 0.1) as f64 / n_pairs.max(1) as f64
        );
        // Merge by shape and recompute the deciding statistic.
        let mut parent: BTreeMap<usize, usize> =
            polished.iter().map(|(b, _, _)| (*b, *b)).collect();
        fn find(p: &mut BTreeMap<usize, usize>, x: usize) -> usize {
            let mut r = x;
            while p[&r] != r {
                r = p[&r];
            }
            r
        }
        for (i, j) in &close {
            let (ri, rj) = (find(&mut parent, *i), find(&mut parent, *j));
            if ri != rj {
                let (lo, hi) = (ri.min(rj), ri.max(rj));
                parent.insert(hi, lo);
            }
        }
        let map: BTreeMap<usize, usize> = polished
            .iter()
            .map(|(b, _, _)| (*b, find(&mut parent, *b)))
            .collect();
        let merged = profile(&regroup(counts, &map), &params, 16, 4096, 384, 4096);
        println!(
            "      merging shape-identical basins: {} -> {} states, expected visits \
             {:.2} median {:.2} max (unmerged {:.2} / {:.2}), lumped share {:.3} -> {:.3}",
            base.states,
            merged.states,
            merged.revisits_median,
            merged.revisits_max,
            base.revisits_median,
            base.revisits_max,
            base.lumped_fraction,
            merged.lumped_fraction
        );
    }
    #[cfg(not(feature = "ira"))]
    let _ = (shape_sample, seed);

    // 2. Structural type. If many basins carry one type, the graph's states are
    //    distinguishing distortions rather than structures.
    let cut = 1.39 * pot.kind().r_min() / 2.0_f64.powf(1.0 / 6.0);
    let label_of = |x: ndarray::ArrayView1<f64>| -> (Vec<i64>, Vec<i64>) {
        let f = ptm_fractions(x, n, 0.12);
        let c = cna(x, n, cut);
        (
            f.iter().map(|v| (v * 20.0).round() as i64).collect(),
            [(5, 5, 5), (4, 2, 1), (4, 2, 2), (5, 4, 4), (4, 4, 4)]
                .iter()
                .map(|k| (c.fraction(*k) * 20.0).round() as i64)
                .collect(),
        )
    };
    let mut classes: BTreeMap<(Vec<i64>, Vec<i64>), Vec<(usize, f64)>> = BTreeMap::new();
    for (b, e, x) in &polished {
        classes
            .entry(label_of(x.view()))
            .or_default()
            .push((*b, *e));
    }
    let mut sizes: Vec<usize> = classes.values().map(|v| v.len()).collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    println!(
        "      structural types: {} distinct labels over {} structures, largest class {}, \
         top five {:?}",
        classes.len(),
        polished.len(),
        sizes.first().copied().unwrap_or(0),
        &sizes[..sizes.len().min(5)]
    );

    // 3. Energy spread inside a structural class. A class holding many basins
    //    over a narrow range is the near-distortion case.
    let mut spreads: Vec<(usize, f64)> = classes
        .values()
        .filter(|v| v.len() > 1)
        .map(|v| {
            let lo = v.iter().map(|(_, e)| *e).fold(f64::INFINITY, f64::min);
            let hi = v.iter().map(|(_, e)| *e).fold(f64::NEG_INFINITY, f64::max);
            (v.len(), hi - lo)
        })
        .collect();
    spreads.sort_by(|a, b| b.0.cmp(&a.0));
    println!(
        "      energy spread inside a class: {:?}",
        &spreads[..spreads.len().min(6)]
    );

    // And the statistic under the structural merge, which is the coarsest of
    // the three and therefore the most favourable to the trap reading.
    let mut map: BTreeMap<usize, usize> = BTreeMap::new();
    let mut class_info: BTreeMap<usize, (usize, f64, [f64; 3], [f64; 3])> = BTreeMap::new();
    for (label, v) in &classes {
        let rep = v.iter().map(|(b, _)| *b).min().unwrap_or(0);
        for (b, _) in v {
            map.insert(*b, rep);
        }
        // Template and common-neighbour fractions of the class, recovered from
        // the quantised label, and the deepest structure it holds. This is the
        // correspondence test: an icosahedral funnel shows in the 555 bond
        // count, a decahedral one in 421 and 422 together.
        let deepest = v.iter().map(|(_, e)| *e).fold(f64::INFINITY, f64::min);
        let p = &label.0;
        let c = &label.1;
        class_info.insert(
            rep,
            (
                v.len(),
                deepest,
                [
                    p.first().copied().unwrap_or(0) as f64 / 20.0,
                    p.get(1).copied().unwrap_or(0) as f64 / 20.0,
                    p.get(2).copied().unwrap_or(0) as f64 / 20.0,
                ],
                [
                    c.first().copied().unwrap_or(0) as f64 / 20.0,
                    c.get(1).copied().unwrap_or(0) as f64 / 20.0,
                    c.get(2).copied().unwrap_or(0) as f64 / 20.0,
                ],
            ),
        );
    }
    confinement(counts, &map, &class_info);
    let merged = profile(&regroup(counts, &map), &params, 16, 4096, 384, 4096);
    println!(
        "      merging by structural type: {} -> {} states, expected visits {:.2} median \
         {:.2} max (unmerged {:.2} / {:.2}), depth {} -> {}, lumped share {:.3} -> {:.3}",
        base.states,
        merged.states,
        merged.revisits_median,
        merged.revisits_max,
        base.revisits_median,
        base.revisits_max,
        base.depth,
        merged.depth,
        base.lumped_fraction,
        merged.lumped_fraction
    );
}

/// Confinement rather than recurrence.
///
/// Expected visits per state cannot see a superbasin whose interior is larger
/// than the sample: a funnel holding order exp(alpha N) minima, entered eleven
/// thousand times, gives about one visit per state whether the chain is free or
/// completely confined. Confinement is the property that separates those, and
/// it is visible without any state being revisited: how much of the recorded
/// transition mass stays inside a morphological class against how much leaves
/// it, and how the chain's time is distributed over classes.
fn confinement(
    counts: &anneal_core::superbasin::HopCounts,
    labels: &std::collections::BTreeMap<usize, usize>,
    class_info: &std::collections::BTreeMap<usize, (usize, f64, [f64; 3], [f64; 3])>,
) {
    use std::collections::BTreeMap;
    let cls = |b: usize| labels.get(&b).copied().unwrap_or(usize::MAX);

    let mut internal: BTreeMap<usize, f64> = BTreeMap::new();
    let mut escaping: BTreeMap<usize, f64> = BTreeMap::new();
    let mut entries: BTreeMap<usize, f64> = BTreeMap::new();
    let mut time_in: BTreeMap<usize, f64> = BTreeMap::new();
    for (i, j, w) in counts.edges() {
        let (a, b) = (cls(i), cls(j));
        if a == usize::MAX {
            continue;
        }
        if a == b {
            *internal.entry(a).or_insert(0.0) += w;
        } else {
            *escaping.entry(a).or_insert(0.0) += w;
            if b != usize::MAX {
                *entries.entry(b).or_insert(0.0) += w;
            }
        }
    }
    for b in counts.nodes() {
        let c = cls(b);
        if c != usize::MAX {
            *time_in.entry(c).or_insert(0.0) += counts.time_of(b);
            *escaping.entry(c).or_insert(0.0) += counts.leak_of(b);
        }
    }
    let total_time: f64 = time_in.values().sum();

    // Occupancy concentration: does the run happen inside a few classes.
    let mut by_time: Vec<(usize, f64)> = time_in.iter().map(|(c, t)| (*c, *t)).collect();
    by_time.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let share = |k: usize| -> f64 {
        by_time.iter().take(k).map(|(_, t)| *t).sum::<f64>() / total_time.max(1.0)
    };
    println!(
        "      confinement: {} classes carry the run, top 1 holds {:.3} of the hops, \
         top 3 {:.3}, top 5 {:.3}, top 10 {:.3}",
        by_time.len(),
        share(1),
        share(3),
        share(5),
        share(10)
    );

    // Timescale separation without recurrence: mass that stays against mass
    // that leaves, and the hops between class changes.
    let mut ratios: Vec<f64> = Vec::new();
    for (c, i) in &internal {
        let e = escaping.get(c).copied().unwrap_or(0.0);
        if e > 0.0 {
            ratios.push(i / e);
        }
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |v: &Vec<f64>, f: f64| -> f64 {
        if v.is_empty() {
            f64::NAN
        } else {
            v[((v.len() - 1) as f64 * f) as usize]
        }
    };
    println!(
        "      internal mass over escaping mass, per class: {:.2} median, {:.2} at the \
         ninth decile, {:.2} max, over {} classes with an exit",
        q(&ratios, 0.5),
        q(&ratios, 0.9),
        q(&ratios, 1.0),
        ratios.len()
    );

    println!(
        "      largest classes by occupancy: {}",
        by_time
            .iter()
            .take(6)
            .map(|(c, t)| {
                let e = escaping.get(c).copied().unwrap_or(0.0);
                let dwell = if e > 0.0 { t / e } else { f64::INFINITY };
                let (n_b, e_min, ptm, cna) =
                    class_info
                        .get(c)
                        .copied()
                        .unwrap_or((0, f64::NAN, [0.0; 3], [0.0; 3]));
                format!(
                    "[{} basins, {:.3} of hops, dwell {:.0} hops, deepest {:.4}, \
                     ptm fcc/hcp/ico {:.2}/{:.2}/{:.2}, cna 555/421/422 {:.2}/{:.2}/{:.2}]",
                    n_b,
                    t / total_time.max(1.0),
                    dwell,
                    e_min,
                    ptm[0],
                    ptm[1],
                    ptm[2],
                    cna[0],
                    cna[1],
                    cna[2]
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let spec = args.get(1).cloned().unwrap_or_else(|| "lj".into());
    let n: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(38);
    let pot = match parse_potential(&spec, n) {
        Some(p) => p,
        None => {
            eprintln!("unknown potential {spec}; expected lj or morse:RHO");
            std::process::exit(2);
        }
    };
    let budget: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(400_000);
    let seeds: u64 = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(8);
    let seed0: u64 = std::env::var("SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let reference = reference(pot.kind(), n);
    println!(
        "{} N={n}, budget {budget} charged evaluations, {seeds} seeds{}",
        potential_name(&spec),
        reference
            .map(|r| format!(", reference {r:.6}"))
            .unwrap_or_else(|| ", no published reference".into())
    );

    let mut cfg = Config::for_cluster(n);
    // The move library and the container are set from the potential's own
    // length scale rather than from Lennard-Jones numbers. A Morse cluster at
    // rho = 14 sits at r_0 = 1 against 1.12 for Lennard-Jones, and a container
    // or a minimum separation carried over from the other potential is a
    // different problem.
    let scale = pot.kind().r_min() / 2.0_f64.powf(1.0 / 6.0);
    cfg.min_separation *= scale;
    cfg.container *= scale;

    let opts: Vec<&str> = args
        .get(5)
        .map(|v| v.split(',').collect())
        .unwrap_or_default();
    cfg.allocate_moves = opts.contains(&"thompson");
    cfg.return_screen = opts.contains(&"rscreen");
    cfg.adaptive_screen = opts.contains(&"aq");
    cfg.probe_screen = opts.contains(&"probe");
    cfg.track_funnels = opts.contains(&"funnel");
    // The hierarchy is reported whenever it is recorded, escape or not, so a
    // run says what funnel structure its own transitions imply.
    cfg.superbasin_report = opts.contains(&"sbreport");
    cfg.superbasin_escape = opts.contains(&"sbasin");
    if cfg.superbasin_escape {
        cfg.superbasin_report = true;
    }
    cfg.superbasin_quotient = opts.contains(&"sbquot");
    if cfg.superbasin_quotient {
        cfg.superbasin_report = true;
    }
    cfg.energy_trace = opts.contains(&"trace");
    cfg.superbasin_features = opts.contains(&"sbfeat");
    if cfg.superbasin_features {
        cfg.superbasin_report = true;
    }
    cfg.delayed_acceptance = opts.contains(&"da");
    let selected: Vec<MoveLibrary> = [
        ("reseed", MoveLibrary::Reseed),
        ("twin", MoveLibrary::Twin),
        ("learn", MoveLibrary::LearnedReseed),
    ]
    .into_iter()
    .filter_map(|(name, library)| opts.contains(&name).then_some(library))
    .collect();
    assert!(selected.len() <= 1, "select at most one move library");
    if let Some(library) = selected.into_iter().next() {
        cfg.move_library = library;
    }
    if let Ok(v) = std::env::var("SCREEN_STEPS") {
        cfg.screen_steps = v.parse().unwrap_or(25);
    }
    if let Ok(v) = std::env::var("TEMP") {
        cfg.temperature = v.parse().unwrap_or(0.8);
    }
    if let Ok(v) = std::env::var("QUENCH_WARMUP") {
        cfg.quench_warmup = v.parse().unwrap_or(4);
    }
    if let Ok(v) = std::env::var("QUENCH_CONF") {
        cfg.quench_confidence = v.parse().unwrap_or(2.0);
    }
    // The delayed-acceptance first stage abstains above this many temperatures
    // of predictive spread. `inf` makes it always speak, which is what
    // separates a surrogate that is silent from one that is wrong.
    if let Ok(v) = std::env::var("SURROGATE_TOL") {
        cfg.surrogate_tolerance = v.parse().unwrap_or(0.5);
    }
    // Above the energy range the screen never refuses, so every hop pays a
    // full relaxation and the quenched sample is untruncated. The comparator
    // for what the screen's own truncation does to a tail fit.
    if let Ok(v) = std::env::var("SCREEN_MARGIN") {
        cfg.screen_margin = v.parse().unwrap_or(2.0);
    }
    cfg.contextual_moves = opts.contains(&"ctx");
    cfg.bayes_screen = opts.contains(&"bayes");
    cfg.angular_moves = opts.contains(&"angular");
    // Symmetrise onto the structure's own approximate symmetry when stuck.
    cfg.symmetrise_on_stall = opts.contains(&"sym");
    cfg.budget_window = opts.contains(&"bfwt");
    // The like-for-like control for the Hamiltonian arms: the kick alone,
    // without the three packing-changing kernels the default library carries.
    cfg.displacement_only = opts.contains(&"kick");
    if opts.contains(&"sites") {
        cfg.keying = Keying::Sites;
    }
    if opts.contains(&"canon") {
        cfg.keying = Keying::Canonical;
        cfg.merge_radius = 0.3;
    }
    // Every quenched energy, to a file, when the caller asks for one. The
    // charged count travels with each so a prefix of the file is what the run
    // had seen at that point in its budget.
    let trace_dir = std::env::var("QUENCH_TRACE").ok();
    cfg.trace_quenched = trace_dir.is_some();
    if opts.contains(&"triplet") {
        cfg.keying = Keying::Triplet;
        // The descriptor appends two kernel spectra to the distance spectrum,
        // so its distances run larger than the default keying's; 0.95 is the
        // response to a quench-scale displacement, and MERGE_RADIUS overrides.
        cfg.merge_radius = 0.95;
        cfg.keying_sigma *= scale;
    }
    if let Ok(v) = std::env::var("MERGE_RADIUS")
        && let Ok(r) = v.parse::<f64>()
    {
        cfg.merge_radius = r;
    }
    // The Hamiltonian proposal, and which mass matrix it runs. Three arms and a
    // control: the metric is the thing being measured, so it is named on the
    // command line rather than chosen here.
    let hmc_metric = if opts.contains(&"hmc-hess") {
        Some(MetricKind::ModelHessian)
    } else if opts.contains(&"hmc-diag") {
        Some(MetricKind::Diagonal)
    } else if opts.contains(&"hmc") {
        Some(MetricKind::Identity)
    } else {
        None
    };
    if let Some(kind) = hmc_metric {
        let mut h = HopConfig::new(n, kind);
        // Warmup and the depth cap are budget decisions and are reported as
        // such: the cap rate says how often the no-U-turn criterion was
        // truncated, and a run whose cap rate is near one is running
        // fixed-length HMC under the name of NUTS.
        if let Ok(v) = std::env::var("HMC_WARMUP") {
            if let Ok(w) = v.parse::<usize>() {
                h.warmup_hops = w;
            }
        }
        if let Ok(v) = std::env::var("HMC_MAX_DEPTH") {
            if let Ok(d) = v.parse::<u32>() {
                h.max_depth = d;
            }
        }
        println!(
            "  hamiltonian proposal: metric {}, warmup {} hops, depth cap {} \
             ({} leaves), target accept {}, reach {:.3}",
            kind.name(),
            h.warmup_hops,
            h.max_depth,
            (1usize << h.max_depth) - 1,
            h.target_accept,
            h.reach(),
        );
        cfg.hmc = Some(h);
    }
    // The replica ladder, one budget shared across the rungs rather than one
    // budget each. The four names are the four arms: the ladder as it ran,
    // the same ladder with each rung at its own temperature, whole parity
    // classes with a coin-flipped parity, and the non-reversible sweep on a
    // ladder placed by the barrier the run measures.
    let ladder_mode = if opts.contains(&"indep") {
        Some(LadderMode::Independent)
    } else if opts.contains(&"nrpt") {
        Some(LadderMode::NonReversible)
    } else if opts.contains(&"rpt") {
        Some(LadderMode::Reversible)
    } else if opts.contains(&"ptt") {
        Some(LadderMode::Cyclic)
    } else if opts.contains(&"pt") {
        Some(LadderMode::Shipped)
    } else {
        None
    };
    if let Some(mode) = ladder_mode {
        cfg.replicas = std::env::var("REPLICAS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        cfg.ladder_mode = mode;
        // The swap period is the ladder's unit of time and the budget decides
        // how many units there are. At LJ38 with 4e5 charged evaluations a run
        // takes about 12700 hops, so a period of 50 over four rungs buys 60
        // sweeps, and a ladder transports nothing in 60 sweeps.
        if let Ok(v) = std::env::var("SWAP_PERIOD") {
            if let Ok(p) = v.parse::<usize>() {
                cfg.swap_period = p.max(1);
            }
        }
        if let Ok(v) = std::env::var("LADDER_ACCEPT") {
            if let Ok(a) = v.parse::<f64>() {
                cfg.ladder_target_accept = a.clamp(0.01, 0.95);
            }
        }
        cfg.bias_by_rung = opts.contains(&"rungbias");
        println!(
            "  replica exchange: {} chains, {mode:?}, swap every {} hops, \
             rung temperatures {}, ladder target accept {:.2}",
            cfg.replicas,
            cfg.swap_period,
            mode.tempers(),
            cfg.ladder_target_accept
        );
    }
    if !opts.is_empty() {
        println!("  mechanisms: {}", opts.join(", "));
    }

    // The bank arm, with the next start chosen by expected improvement over
    // morphology rather than by a round robin.
    let use_bank = opts.contains(&"bank");
    let bank_cfg = BankConfig {
        capacity: 30,
        slice: budget / 400,
        seeding: 30,
        dcut_floor: 0.4,
        mix_fraction: 0.5,
        mix_images: 7,
        random_images: 0,
        deadlock_iters: 3,
        deadlock_inject: 0,
        acquisition: opts.contains(&"ei"),
    };
    if use_bank {
        println!(
            "  bank of {} , slice {}, acquisition {}",
            bank_cfg.capacity, bank_cfg.slice, bank_cfg.acquisition
        );
    }

    let mut solved = 0usize;
    let mut encounters: Vec<Encounter> = Vec::new();
    let mut deepest = f64::INFINITY;
    let mut total_charged = 0usize;
    let mut total_hops = 0usize;
    for seed in seed0..(seed0 + seeds) {
        let mut ledger = Ledger::new(budget);
        // The whole of the plumbing, in one call: the relaxation, the charged
        // gradient and the convergence count all come from the crate now, so
        // this example cannot quietly run a different potential or a different
        // relaxation from any other caller.
        let (out, stats) = if use_bank {
            let mut opt2 = anneal_core::methods::warm_lbfgs::WarmLbfgs::default();
            let mut relax = |led: &mut Ledger, x: ndarray::ArrayView1<f64>, iters: usize| {
                opt2.forget();
                let (f, xr, _) = opt2.minimize(x, iters, |v| {
                    if !led.charge() {
                        return None;
                    }
                    Some(pot.value_and_gradient(v))
                });
                (f, xr)
            };
            let b = csa_cluster::run(
                &cfg,
                &bank_cfg,
                &mut ledger,
                &mut relax,
                None,
                anneal_core::methods::csa_cluster::spectrum_distance(n),
                seed,
            );
            println!(
                "      bank: {} slices, {} morphologies, Dcut {:.3} -> {:.3}, {} mixes",
                b.slices, b.morphologies, b.dcut.0, b.dcut.1, b.mixes
            );
            (
                anneal_core::methods::cluster_hopping::Outcome {
                    best: b.best,
                    best_state: b.best_state,
                    hops: b.hops,
                    basins: b.basins,
                    // Carried, or the encounter time censors every run of this
                    // arm including the ones that found the answer.
                    improvements: b.improvements.clone(),
                    ..Default::default()
                },
                anneal_core::methods::cluster_search::RelaxStats::default(),
            )
        } else {
            search(&pot, &cfg, &mut ledger, seed)
        };

        // The reported value is checked against a fresh evaluation of the
        // structure it claims to come from, off the ledger and outside the
        // driver, and the structure has to be a minimum rather than merely
        // carry the right energy.
        let verified = verify(&pot, &out);
        if let Some((e, gmax)) = verified {
            assert!(
                gmax < 1e-3,
                "seed {seed} returned a structure with gradient {gmax:.2e}, \
                 which is not a minimum"
            );
            assert!(
                (e - out.best).abs() < 1e-6,
                "seed {seed} reported {:.6} but its structure is {e:.6}",
                out.best
            );
        }
        // What morphology the run actually ended in, which a success count
        // does not say. Cheap enough to do once per seed on the answer.
        let morphology = out.best_state.as_ref().map(|st| {
            let c = cna(
                st.view(),
                n,
                1.39 * pot.kind().r_min() / 2.0_f64.powf(1.0 / 6.0),
            );
            let f = ptm_fractions(st.view(), n, 0.12);
            // The residual distribution, because a classification that reports
            // nothing is either a structure with no order or a cutoff set in
            // the wrong place, and the fractions alone cannot say which.
            let mut r: Vec<f64> = ptm(st.view(), n, f64::INFINITY)
                .iter()
                .map(|m| m.rmsd)
                .collect();
            r.sort_by(|a, b| a.partial_cmp(b).unwrap());
            format!(
                "cna 555 {:.3} 421 {:.3} 422 {:.3} | ptm fcc {:.2} hcp {:.2} ico {:.2} other {:.2}",
                c.fraction((5, 5, 5)),
                c.fraction((4, 2, 1)),
                c.fraction((4, 2, 2)),
                f[0],
                f[1],
                f[2],
                f[4]
            ) + &format!(
                " | rmsd min {:.3} median {:.3}",
                r.first().copied().unwrap_or(f64::NAN),
                r.get(r.len() / 2).copied().unwrap_or(f64::NAN)
            )
        });
        let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
        if let Some(tr) = &out.energy_trace {
            // One file per seed, one quenched energy per line, in order. The
            // name carries the outcome because which region a run sat in is the
            // thing a reader has to condition on.
            let dir = std::env::var("TRACE_DIR").unwrap_or_else(|_| ".".into());
            let tag = if hit { "solved" } else { "failed" };
            let path = format!("{dir}/trace_lj{n}_s{seed}_{tag}.txt");
            let body: String = tr
                .iter()
                .map(|v| format!("{v:.6}\n"))
                .collect::<Vec<_>>()
                .concat();
            if let Err(e) = std::fs::write(&path, body) {
                eprintln!("could not write {path}: {e}");
            } else {
                println!(
                    "    energy trace: {} full quenches written to {path}, \
                     lowest {:.6}, highest {:.6}",
                    tr.len(),
                    tr.iter().copied().fold(f64::INFINITY, f64::min),
                    tr.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                );
            }
        }
        if hit {
            solved += 1;
        }
        // The work to first reach the published minimum, which is the
        // statistic worth comparing. A run that never reached it contributes a
        // lower bound rather than being dropped.
        if let Some(r) = reference {
            encounters.push(first_encounter(&out, r, 1e-4, ledger.spent()));
        }
        deepest = deepest.min(out.best);
        total_charged += ledger.spent();
        total_hops += out.hops;
        // Transport, on every seed. A ladder can improve mixing and still not
        // find the minimum, and a solve count reports neither half.
        if let Some((trips, sw, barrier)) = out.transport {
            let per_tag = if trips > 0 {
                out.rungs.len().max(cfg.replicas) as f64 * sw as f64 / trips as f64
            } else {
                f64::INFINITY
            };
            println!(
                "    ladder: round trips {trips} in {sw} sweeps \
                 (rate {:.4}/sweep, {per_tag:.0} sweeps per tag), barrier {barrier:.2}",
                trips as f64 / sw.max(1) as f64
            );
            for (t, b, en) in &out.rungs {
                println!("      rung T={t:.3}  basins {b:>5}  energy {en:>11.4}");
            }
        }
        if let Some((s1, s1r, s2, s2r)) = out.delayed {
            println!(
                "    delayed: stage1 {s1} rejected {s1r} ({:.3}), stage2 {s2} rejected {s2r} ({:.3})",
                s1r as f64 / s1.max(1) as f64,
                s2r as f64 / s2.max(1) as f64
            );
        }
        if let (Some(counts), Some(archive)) = (&out.superbasin_counts, &out.superbasin_archive) {
            archive_analysis(&pot, n, counts, archive, 320, seed);
        }
        if let Some(sb) = &out.superbasin {
            println!(
                "    superbasin: {} basins {} transitions, {} archived, bias distortion {:.3}",
                sb.nodes, sb.edges, sb.archived, sb.distortion
            );
            for (k, (states, largest, separation)) in sb.levels.iter().enumerate() {
                println!(
                    "      level {}: {states} coarse states, largest lump {largest}, \
                     separation {separation:.1}",
                    k + 1
                );
            }
            let refusals: Vec<String> = anneal_core::superbasin::Refusal::KINDS
                .iter()
                .zip(sb.refusals_by_kind.iter())
                .filter(|(_, n)| **n > 0)
                .map(|(k, n)| format!("{k} {n}"))
                .collect();
            println!(
                "      top partition {:?}   jumps {} refused {} [{}] worst revisits {:.2}   \
                 condition max {:.3e} mean {:.3e} residual {:.1e}   \
                 solve residual {:.1e} exact {}   \
                 hops replaced {:.0}   improved {} by {:.4}",
                sb.top,
                sb.jumps,
                sb.refusals,
                refusals.join(", "),
                sb.mixed_ratio_max,
                sb.condition_max,
                sb.condition_mean,
                sb.condition_residual_max,
                sb.solve_residual_max,
                sb.exact_solves,
                sb.hops_saved,
                sb.improvements.0,
                sb.improvements.1
            );
            if let Some(q) = &sb.quotient {
                println!(
                    "      orbit quotient: {} basins -> {} ({:.2}x), {} classes above one, \
                     largest {}, from {} archived over {} energy buckets and {} comparisons \
                     (matched <= {:.2e}, rejected >= {:.3})",
                    q.basins_raw,
                    q.basins_quotiented,
                    q.basins_raw as f64 / q.basins_quotiented.max(1) as f64,
                    q.orbits_nontrivial,
                    q.largest_orbit,
                    q.archived,
                    q.energy_buckets,
                    q.comparisons,
                    q.matched_max,
                    q.rejected_min
                );
                println!(
                    "      expected visits per state over {} sources: raw {:.2} median \
                     {:.2} max, quotiented {:.2} median {:.2} max   |   \
                     hierarchy depth {} -> {}, lumped share {:.3} -> {:.3}",
                    q.sources,
                    q.revisits_raw.0,
                    q.revisits_raw.1,
                    q.revisits_quotiented.0,
                    q.revisits_quotiented.1,
                    q.depth.0,
                    q.depth.1,
                    q.lumped_fraction.0,
                    q.lumped_fraction.1
                );
            }
            if let Some(sep) = &sb.separability {
                println!(
                    "      structure separates the coarse states: F {:.2} over {} states, \
                     {} structures, per template {:?}",
                    sep.f,
                    sep.groups,
                    sep.points,
                    sep.per_dimension
                        .iter()
                        .map(|v| format!("{v:.2}"))
                        .collect::<Vec<_>>()
                );
            }
        }
        for (name, draws, accepts, best) in &out.arms {
            if *draws > 0 {
                println!(
                    "    arm {name:<14} draws {draws:>7}  accepts {accepts:>7} ({:.3})  best {:.4}",
                    *accepts as f64 / *draws as f64,
                    best
                );
            }
        }
        println!(
            "  seed {seed}: best {:.6}  hops {}  basins {}  relaxed {}/{}  sym {}/{:.2}  \
             charged screen {} full {} check {} ({:.0}% screen)  accept {:.3}  qsteps {:.1}  probe {} at {:.1} err {:.4}  verified {}{}",
            out.best,
            out.hops,
            out.basins,
            stats.converged,
            stats.total(),
            out.symmetrised.0,
            out.symmetrised.1,
            stats.screen_charged,
            stats.full_charged,
            stats.check_charged,
            100.0 * stats.screen_share(),
            out.accepted as f64 / out.hops.max(1) as f64,
            stats.screen_steps_taken as f64 / stats.screens.max(1) as f64,
            stats.probe_stops,
            stats.probe_steps as f64 / stats.probe_stops.max(1) as f64,
            stats.probe_error / stats.probe_stops.max(1) as f64,
            verified
                .map(|(e, gmax)| format!("{e:.6} |g| {gmax:.1e}"))
                .unwrap_or_else(|| "NO STATE".into()),
            if hit { "  SOLVED" } else { "" }
        );
        if let Some(m) = morphology {
            println!("      {m}");
        }
        if let Some(dir) = trace_dir.as_deref() {
            let path = format!("{dir}/{}_{n}_s{seed}.trace", spec.replace(':', ""));
            let mut body = format!(
                "# charged basins energy converged\n# budget {budget} solved {} \
                 first_encounter {} screen_margin {}\n",
                hit as u8,
                encounters
                    .last()
                    .filter(|e| e.found())
                    .map(|e| e.charged().to_string())
                    .unwrap_or_else(|| "censored".into()),
                cfg.screen_margin
            );
            for (c, b, e, k) in &out.quenched {
                body.push_str(&format!("{c} {b} {e:.9} {}\n", *k as u8));
            }
            match std::fs::write(&path, body) {
                Ok(()) => println!("      trace {} rows -> {path}", out.quenched.len()),
                Err(e) => eprintln!("      trace write failed: {e}"),
            }
        }
        // The sampler's own diagnostics, one line per rung. Reported for every
        // run rather than only when something looks wrong: a divergence rate
        // says which configurations the integrator cannot traverse, and a
        // mechanism that runs without acting is invisible to a solve count.
        for (k, d) in out.hmc.iter().enumerate() {
            if d.proposals > 0 {
                println!("      rung {k} {}", d.report(""));
                println!("      rung {k} {}", d.depth_report());
            }
        }
    }
    println!(
        "{solved}/{seeds} solved, deepest {deepest:.6}   mean hops {}, force per hop {}",
        total_hops / seeds.max(1) as usize,
        total_charged / total_hops.max(1)
    );
    if let Some(r) = reference {
        println!("gap to reference {:+.6}", deepest - r);
    }
    if !encounters.is_empty() {
        let found: Vec<usize> = encounters
            .iter()
            .filter(|e| e.found())
            .map(|e| e.charged())
            .collect();
        let censored = encounters.len() - found.len();
        match median_encounter(&encounters) {
            Some(m) => println!(
                "first encounter: median {m} charged evaluations \
                 ({} reached, {censored} censored)",
                found.len()
            ),
            None => println!(
                "first encounter: no median, {} of {} runs censored; \
                 the median has not been observed",
                censored,
                encounters.len()
            ),
        }
        if !found.is_empty() {
            let lo = found.iter().copied().min().unwrap_or(0);
            let hi = found.iter().copied().max().unwrap_or(0);
            println!("  observed encounters span {lo} to {hi}");
        }
    }
}
