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

use anneal_core::tail::{GridSpec, Prior, Rung, ladder, select_threshold};

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
fn prefix(t: &Trace, charged: usize, first_visit: bool, converged_only: bool) -> Vec<f64> {
    let mut out = Vec::new();
    let mut seen = usize::MAX;
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

fn report_ladder(name: &str, rungs: &[Rung], pick: usize, reference: f64) {
    println!("  {name}: threshold ladder");
    println!(
        "    {:>5} {:>8} {:>7} {:>11} {:>11} {:>11} {:>8} {:>8} {:>7}",
        "q", "u", "k", "endpoint", "2.5%", "97.5%", "P(unb)", "overlap", "pick"
    );
    for (i, r) in rungs.iter().enumerate() {
        let p = &r.posterior;
        println!(
            "    {:>5.2} {:>8.3} {:>7} {:>11.4} {:>11.4} {:>11.4} {:>8.3} {:>8.3} {:>7}",
            r.quantile,
            p.energy_threshold,
            p.n_exceedances,
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
         converged-only {converged_only}, decluster gap {gap}"
    );

    let traces: Vec<Trace> = paths.iter().filter_map(|p| read(p)).collect();
    println!("{} traces read", traces.len());

    // Per prefix: (failed excluding, failed usable, succeeded including,
    // succeeded usable).
    let mut tally = vec![(0usize, 0usize, 0usize, 0usize); fractions.len()];
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
            let e = prefix(t, cut, first_visit, converged_only);
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

    println!("\nfor scripts/posterior.py, treatment = failed, control = solved:");
    for (fi, &f) in fractions.iter().enumerate() {
        let (ke, ne, ki, ni) = tally[fi];
        if ne == 0 || ni == 0 {
            continue;
        }
        println!("  excl-at-{f:.2}:{ke}/{ne}:{}/{ni}", ni - ki);
    }
}
