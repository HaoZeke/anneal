//! Does the endpoint posterior tell a run that will fail from one that will not.
//!
//! Reads quenched-energy traces written by `cluster_bench` under
//! `QUENCH_TRACE`, fits the peaks-over-threshold endpoint posterior of
//! [`anneal_core::tail`] to a *prefix* of each, and asks one question of the
//! posterior: does it put the published reference outside the region the run is
//! sampling.
//!
//! The prefix is what makes this a prediction. A run that succeeds has the
//! reference in its own sample, and the support constraint of the generalised
//! Pareto likelihood then forces the endpoint at or below it, so a call made on
//! the whole trace of a successful run is arithmetic rather than evidence.
//! Successful runs therefore contribute only at prefixes ending before their
//! first encounter with the reference; the count of usable pairs is reported
//! alongside the rate so that this is visible rather than buried.
//!
//! Usage:
//! `cargo run --release --example tail_classifier -- <reference> <trace>...`
//! with `LADDER=1` to print the threshold-stability diagnostic per trace.

use anneal_core::tail::{GridSpec, Prior, Rung, ladder, mean_excess_slope, select_threshold};

/// One run's trace.
struct Trace {
    name: String,
    budget: usize,
    solved: bool,
    first_encounter: Option<usize>,
    rows: Vec<(usize, usize, f64, bool)>,
}

fn read(path: &str) -> Option<Trace> {
    let body = std::fs::read_to_string(path).ok()?;
    let mut budget = 0usize;
    let mut solved = false;
    let mut first_encounter = None;
    let mut rows = Vec::new();
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("# budget ") {
            let f: Vec<&str> = rest.split_whitespace().collect();
            budget = f.first()?.parse().ok()?;
            solved = f.get(2).map(|v| *v == "1").unwrap_or(false);
            first_encounter = f.get(4).and_then(|v| v.parse().ok());
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 4 {
            continue;
        }
        rows.push((
            f[0].parse().ok()?,
            f[1].parse().ok()?,
            f[2].parse().ok()?,
            f[3] == "1",
        ));
    }
    Some(Trace {
        name: std::path::Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.into()),
        budget,
        solved,
        first_encounter,
        rows,
    })
}

/// Energies seen up to `charged`, one per first visit to a basin.
///
/// Repeat visits are the same minimum read twice and are not a second draw
/// from the density of states; the basin counter already in the trace
/// separates them at no cost.
fn prefix(
    t: &Trace,
    charged: usize,
    first_visit: bool,
    converged_only: bool,
    dedup: bool,
) -> Vec<f64> {
    let mut out = Vec::new();
    let mut seen = usize::MAX;
    let mut values: std::collections::HashSet<i64> = std::collections::HashSet::new();
    for &(c, b, e, k) in &t.rows {
        if c > charged {
            break;
        }
        if converged_only && !k {
            continue;
        }
        if first_visit {
            if b == seen {
                continue;
            }
            seen = b;
        }
        // One entry per distinct minimum. A chain in a funnel re-quenches to
        // the same structure repeatedly: 1589 relaxations at 38 points return
        // 934 distinct energies, and the deepest value of a run arrives six to
        // nineteen times. Those repeats are not further draws from the density
        // of states, and the tie they put at the sample maximum is what a
        // generalised Pareto fit reads as a hard cutoff: the shape runs to the
        // prior floor near -1 and the endpoint pins to the value already found.
        // Two minima agreeing to 1e-6 in energy are the same minimum; the
        // deepest four at 38 points are separated by 0.68 and more.
        if dedup && !values.insert((e * 1e6).round() as i64) {
            continue;
        }
        out.push(e);
    }
    out
}

/// Jeffreys Beta(1/2, 1/2) posterior for a rate, as `scripts/posterior.py`.
fn rate(k: usize, n: usize) -> (f64, f64, f64) {
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let (a, b) = (0.5 + k as f64, 0.5 + (n - k) as f64);
    let mean = a / (a + b);
    // Equal-tailed 95 per cent interval by bisection on the regularised
    // incomplete beta, which is enough for a two-decimal report.
    let q = |p: f64| {
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..200 {
            let m = 0.5 * (lo + hi);
            if beta_cdf(m, a, b) < p {
                lo = m;
            } else {
                hi = m;
            }
        }
        0.5 * (lo + hi)
    };
    (mean, q(0.025), q(0.975))
}

/// Regularised incomplete beta by its continued fraction (Lentz).
fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lbeta =
        |p: f64, q: f64| libm_lgamma(p) + libm_lgamma(q) - libm_lgamma(p + q);
    let front = (a * x.ln() + b * (1.0 - x).ln() - lbeta(a, b)).exp() / a;
    let cf = |a: f64, b: f64, x: f64| {
        let (mut c, mut d) = (1.0f64, 1.0f64 - (a + b) * x / (a + 1.0));
        if d.abs() < 1e-300 {
            d = 1e-300;
        }
        d = 1.0 / d;
        let mut h = d;
        for m in 1..300 {
            let m = m as f64;
            let num = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
            d = 1.0 + num * d;
            if d.abs() < 1e-300 {
                d = 1e-300;
            }
            c = 1.0 + num / c;
            if c.abs() < 1e-300 {
                c = 1e-300;
            }
            d = 1.0 / d;
            h *= d * c;
            let num = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
            d = 1.0 + num * d;
            if d.abs() < 1e-300 {
                d = 1e-300;
            }
            c = 1.0 + num / c;
            if c.abs() < 1e-300 {
                c = 1e-300;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-14 {
                break;
            }
        }
        h
    };
    if x < (a + 1.0) / (a + b + 2.0) {
        front * cf(a, b, x)
    } else {
        1.0 - (b * (1.0 - x).ln() + a * x.ln() - lbeta(a, b)).exp() / b * cf(b, a, 1.0 - x)
    }
}

/// Lanczos log gamma, g = 7, n = 9; relative error below 1e-13 on the positive
/// reals, which is far inside what a credible interval is quoted to.
fn libm_lgamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - libm_lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Probability that a failed run's statistic exceeds a solved run's, over all
/// pairs, with a bootstrap interval resampling runs rather than pairs.
///
/// The pairs are not independent, so a Beta posterior on a concordance count
/// would report an interval several times too narrow. Resampling the runs
/// carries the dependence through.
fn auc_with_interval<F: Fn(&(bool, f64, f64)) -> f64>(
    rows: &[(bool, f64, f64)],
    stat: F,
    seed: u64,
) -> (f64, f64, f64) {
    let point = auc(rows, &stat);
    let mut s = seed;
    let mut next = || {
        // xorshift64star, enough for a bootstrap index stream.
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let mut reps = Vec::with_capacity(4000);
    for _ in 0..4000 {
        let boot: Vec<(bool, f64, f64)> = (0..rows.len())
            .map(|_| rows[(next() % rows.len() as u64) as usize])
            .collect();
        if boot.iter().any(|r| r.0) && boot.iter().any(|r| !r.0) {
            reps.push(auc(&boot, &stat));
        }
    }
    reps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if reps.is_empty() {
        return (point, f64::NAN, f64::NAN);
    }
    let q = |p: f64| reps[((p * (reps.len() - 1) as f64) as usize).min(reps.len() - 1)];
    (point, q(0.025), q(0.975))
}

fn auc<F: Fn(&(bool, f64, f64)) -> f64>(rows: &[(bool, f64, f64)], stat: &F) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for a in rows.iter().filter(|r| !r.0) {
        for b in rows.iter().filter(|r| r.0) {
            let (x, y) = (stat(a), stat(b));
            num += if x > y {
                1.0
            } else if x == y {
                0.5
            } else {
                0.0
            };
            den += 1.0;
        }
    }
    if den > 0.0 { num / den } else { f64::NAN }
}

fn report_ladder(name: &str, rungs: &[Rung], pick: usize, reference: f64) {
    println!("  {name}: threshold ladder");
    println!(
        "    {:>5} {:>8} {:>7} {:>8} {:>8} {:>11} {:>11} {:>11} {:>8} {:>8} {:>5}",
        "q", "u", "k", "mexc", "E[xi]", "endpoint", "2.5%", "97.5%", "P(unb)", "overlap", "pick"
    );
    for (i, r) in rungs.iter().enumerate() {
        let p = &r.posterior;
        println!(
            "    {:>5.2} {:>8.3} {:>7} {:>8.4} {:>8.4} {:>11.4} {:>11.4} {:>11.4} {:>8.3} {:>8.3} {:>5}",
            r.quantile,
            p.energy_threshold,
            p.n_exceedances,
            r.mean_excess,
            p.xi_mean(),
            p.endpoint_quantile(0.5),
            p.endpoint_quantile(0.025),
            p.endpoint_quantile(0.975),
            p.p_unbounded,
            r.overlap_top,
            if i == pick { "*" } else { "" }
        );
    }
    let p = &rungs[pick].posterior;
    println!(
        "    selected q {:.2}: P(endpoint above reference {reference:.6}) = {:.4}, \
         shape mean {:.4}, floor mass {:.2e}, box leak {:.2e}",
        rungs[pick].quantile,
        p.prob_endpoint_above(reference),
        p.xi_mean(),
        p.p_xi_floor,
        p.box_leak
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reference: f64 = args
        .get(1)
        .and_then(|v| v.parse().ok())
        .expect("usage: tail_classifier <reference> <trace>...");
    let paths: Vec<&String> = args[2..].iter().collect();
    let show_ladder = std::env::var("LADDER").is_ok();
    let alpha: f64 = std::env::var("ALPHA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.95);
    let gap: usize = std::env::var("DECLUSTER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    // Off by default. The basin counter is keyed by the driver's merge radius,
    // which at 38 points collapses about thirty hops into one registered
    // basin, so first-visit filtering there discards most of the sample rather
    // than deduplicating it. Serial dependence is handled by declustering,
    // which cuts the sample by a stated factor instead.
    let first_visit = std::env::var("FIRST_VISIT").is_ok();
    // A relaxation that stopped at its iteration cap sits a little above the
    // minimum it was heading for. Keeping only the gradient-verified ones
    // removes that, at the cost of most of the sample.
    let converged_only = std::env::var("CONVERGED_ONLY").is_ok();
    let dedup = std::env::var("NO_DEDUP").is_err();
    let k_min: usize = std::env::var("KMIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let quantiles: Vec<f64> = vec![0.50, 0.35, 0.25, 0.15, 0.10, 0.06, 0.04, 0.025];
    let prior = Prior::default();
    let grid = GridSpec::default();
    let fractions: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0];

    println!(
        "reference {reference:.6}, decision P(endpoint above reference) > {alpha}, \
         thresholds {quantiles:?}, k_min {k_min}, first-visit {first_visit}, \
         converged-only {converged_only}, distinct-minima {dedup}, decluster gap {gap}"
    );

    let traces: Vec<Trace> = paths.iter().filter_map(|p| read(p)).collect();
    println!("{} traces read", traces.len());

    // Per prefix: (failed excluding, failed usable, succeeded including,
    // succeeded usable).
    let mut tally = vec![(0usize, 0usize, 0usize, 0usize); fractions.len()];
    // Per prefix: (solved, P(endpoint above reference), best so far) per run,
    // for the discrimination curve and for the control against which the
    // endpoint posterior has to earn its cost.
    let mut stats: Vec<Vec<(bool, f64, f64)>> = vec![Vec::new(); fractions.len()];
    let mut pinning: Vec<f64> = Vec::new();
    #[allow(clippy::type_complexity)]
    let mut shape_report: Vec<(String, bool, f64, f64, f64, f64, f64, f64, usize)> = Vec::new();

    for t in &traces {
        for (fi, &f) in fractions.iter().enumerate() {
            let cut = (f * t.budget as f64) as usize;
            // A successful run's own answer inside the window makes the call
            // arithmetic rather than a prediction.
            if t.solved && t.first_encounter.map(|c| cut >= c).unwrap_or(false) {
                continue;
            }
            let e = prefix(t, cut, first_visit, converged_only, dedup);
            if e.len() < k_min * 3 {
                continue;
            }
            let rungs = ladder(&e, &quantiles, k_min, gap, &prior, &grid);
            let Some(pick) = select_threshold(&rungs, 0.5) else {
                continue;
            };
            let p = &rungs[pick].posterior;
            let pe = p.prob_endpoint_above(reference);
            let excludes = pe > alpha;
            if show_ladder && (f - 1.0).abs() < 1e-12 {
                report_ladder(&t.name, &rungs, pick, reference);
            }
            // How far the upper end of the endpoint posterior sits below the
            // deepest energy in the sample. The likelihood's support constraint
            // forbids an endpoint above an observed minimum, so a small gap
            // here means the posterior is pinned by that constraint rather than
            // located by the tail's shape, and the decision statistic is then a
            // restatement of the running minimum.
            if (f - 1.0).abs() < 1e-12 {
                pinning.push(p.endpoint_quantile(0.975) - e.iter().copied().fold(f64::INFINITY, f64::min));
            }
            if (f - 0.5).abs() < 1e-12 {
                shape_report.push((
                    t.name.clone(),
                    t.solved,
                    e.iter().copied().fold(f64::INFINITY, f64::min),
                    p.xi_mean(),
                    p.p_unbounded,
                    p.endpoint_quantile(0.5),
                    p.endpoint_quantile(0.975),
                    pe,
                    p.n_exceedances,
                ));
            }
            stats[fi].push((t.solved, pe, e.iter().copied().fold(f64::INFINITY, f64::min)));
            if t.solved {
                tally[fi].3 += 1;
                if !excludes {
                    tally[fi].2 += 1;
                }
            } else {
                tally[fi].1 += 1;
                if excludes {
                    tally[fi].0 += 1;
                }
            }
        }
    }

    // The model-free check, before any fit is believed. A mean excess flat in
    // the threshold is the exponential case, where the sample lies in the
    // Gumbel domain and there is no endpoint to estimate; one falling with the
    // threshold is a negative shape and a floor.
    let mut slopes: Vec<(bool, f64, f64)> = Vec::new();
    for t in &traces {
        let e = prefix(t, t.budget, first_visit, converged_only, dedup);
        if let Some((s, x)) = mean_excess_slope(&e, &quantiles, k_min) {
            slopes.push((t.solved, s, x));
        }
    }
    if !slopes.is_empty() {
        let summarise = |sel: bool| {
            let mut v: Vec<f64> = slopes
                .iter()
                .filter(|r| r.0 == sel)
                .map(|r| r.2)
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if v.is_empty() {
                return (f64::NAN, f64::NAN, f64::NAN, 0usize);
            }
            let q = |p: f64| v[((p * (v.len() - 1) as f64) as usize).min(v.len() - 1)];
            (q(0.5), q(0.05), q(0.95), v.len())
        };
        println!("\nmean-excess slope over the whole trace, and the shape it implies");
        for (label, sel) in [("failed", false), ("solved", true)] {
            let (m, lo, hi, n) = summarise(sel);
            println!(
                "  {label:>7}: n {n}, implied xi median {m:.4}, 5th to 95th [{lo:.4}, {hi:.4}]"
            );
        }
        // The model checked against data it was fitted to. With that shape and
        // the mean excess at each threshold, the moment endpoint is a floor the
        // model asserts; a floor above an energy the run has already reached is
        // a refutation, not an estimate.
        let mut viol = 0usize;
        let mut tot = 0usize;
        let mut worst = 0.0f64;
        for t in &traces {
            let e = prefix(t, t.budget, first_visit, converged_only, dedup);
            let Some((_, xi)) = mean_excess_slope(&e, &quantiles, k_min) else {
                continue;
            };
            let deepest = e.iter().copied().fold(f64::INFINITY, f64::min);
            for &q in &quantiles {
                let u = anneal_core::tail::quantile_of(&e, q);
                let Some((k, m)) = anneal_core::tail::mean_excess(&e, u) else {
                    continue;
                };
                if k < k_min {
                    continue;
                }
                let Some(theta) = anneal_core::tail::moment_endpoint(u, m, xi) else {
                    continue;
                };
                tot += 1;
                if theta > deepest {
                    viol += 1;
                    worst = worst.max(theta - deepest);
                }
            }
        }
        println!(
            "  moment endpoint above an energy the run had already reached: \
             {viol}/{tot} threshold fits, worst by {worst:.4}"
        );
    }

    if !pinning.is_empty() {
        pinning.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| pinning[((p * (pinning.len() - 1) as f64) as usize).min(pinning.len() - 1)];
        println!(
            "  endpoint 97.5 per cent quantile below the deepest sampled energy: \
             median {:.4}, 5th to 95th [{:.4}, {:.4}] over {} whole-trace fits",
            -q(0.5),
            -q(0.95),
            -q(0.05),
            pinning.len()
        );
    }

    println!("\nendpoint posterior at half budget, per run");
    println!(
        "  {:>20} {:>6} {:>7} {:>11} {:>9} {:>9} {:>11} {:>11} {:>9}",
        "run", "solved", "k", "best so far", "E[xi]", "P(xi>=0)", "endpoint", "97.5%", "P(excl)"
    );
    for (n, s, b, x, u, m, h, pe, k) in &shape_report {
        println!(
            "  {n:>20} {:>6} {k:>7} {b:>11.4} {x:>9.4} {u:>9.4} {m:>11.4} {h:>11.4} {pe:>9.4}",
            if *s { "yes" } else { "no" }
        );
    }

    println!("\nclassifier by budget prefix");
    println!(
        "  {:>6} {:>16} {:>22} {:>16} {:>22}",
        "prefix", "excl | failed", "95% credible", "incl | solved", "95% credible"
    );
    for (fi, &f) in fractions.iter().enumerate() {
        let (ke, ne, ki, ni) = tally[fi];
        let (me, le, he) = rate(ke, ne);
        let (mi, li, hi) = rate(ki, ni);
        println!(
            "  {f:>6.2} {:>7} {:>8.3} {:>10.3} {:>11.3} {:>7} {:>8.3} {:>10.3} {:>11.3}",
            format!("{ke}/{ne}"),
            me,
            le,
            he,
            format!("{ki}/{ni}"),
            mi,
            li,
            hi
        );
    }

    // Discrimination, and the control the endpoint posterior has to beat.
    //
    // A one-alpha call hides the shape of the statistic, so the whole ordering
    // is reported: the probability that a run which failed carries a higher
    // exclusion probability than one which succeeded, over all pairs, with
    // ties at a half. Chance is 0.5. The same quantity for the deepest energy
    // found so far is the control, because the endpoint posterior is anchored
    // just below that value by the support constraint and would look
    // informative merely by restating it.
    println!("\ndiscrimination, P(failed run ranks above solved run), chance 0.5");
    println!(
        "  {:>6} {:>7} {:>7} {:>10} {:>18} {:>10} {:>18}",
        "prefix", "n_fail", "n_solve", "endpoint", "95% bootstrap", "best-so-far", "95% bootstrap"
    );
    for (fi, &f) in fractions.iter().enumerate() {
        let s = &stats[fi];
        let nf = s.iter().filter(|r| !r.0).count();
        let ns = s.len() - nf;
        if nf == 0 || ns == 0 {
            continue;
        }
        let (ae, le, he) = auc_with_interval(s, |r| r.1, 20260806);
        // A run that has already gone deeper is the one nearer the reference,
        // so a shallower best is the "failing" direction and enters negated.
        let (ab, lb, hb) = auc_with_interval(s, |r| -r.2, 20260807);
        println!(
            "  {f:>6.2} {nf:>7} {ns:>7} {ae:>10.3} {:>18} {ab:>10.3} {:>18}",
            format!("[{le:.3}, {he:.3}]"),
            format!("[{lb:.3}, {hb:.3}]")
        );
    }

    println!("\nexclusion rate against the decision bar, failed and solved");
    println!(
        "  {:>6} {:>28} {:>28}",
        "prefix", "P(excl | failed) at alpha", "P(excl | solved) at alpha"
    );
    let bars = [0.5f64, 0.8, 0.9, 0.95, 0.99];
    println!(
        "  {:>6} {:>28} {:>28}",
        "",
        bars.iter()
            .map(|b| format!("{b:>5.2}"))
            .collect::<Vec<_>>()
            .join(" "),
        bars.iter()
            .map(|b| format!("{b:>5.2}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    for (fi, &f) in fractions.iter().enumerate() {
        let s = &stats[fi];
        let nf = s.iter().filter(|r| !r.0).count();
        let ns = s.len() - nf;
        if nf == 0 || ns == 0 {
            continue;
        }
        let row = |solved: bool, n: usize| {
            bars.iter()
                .map(|b| {
                    let k = s.iter().filter(|r| r.0 == solved && r.1 > *b).count();
                    format!("{:>5.2}", k as f64 / n as f64)
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        println!("  {f:>6.2} {:>28} {:>28}", row(false, nf), row(true, ns));
    }

    println!("\nfor scripts/posterior.py, treatment = failed, control = solved:");
    for (fi, &f) in fractions.iter().enumerate() {
        let (ke, ne, ki, ni) = tally[fi];
        if ne == 0 || ni == 0 {
            continue;
        }
        println!("  excl-at-{f:.2}:{ke}/{ne}:{}/{ni}", ni - ki);
    }
}
