//! Does curvature measured at one Lennard-Jones minimum predict the quench
//! depth at another, and how far can it be carried?
//!
//! The depth `E(x) - E(Q(x))` has the leading form `1/2 g^T H^-1 g`. The model
//! Hessian is free but omits every transverse term; the exact Hessian is what
//! the omission costs, and it is expensive. If the correction `H_true - H_model`
//! measured once at a relaxed minimum can be carried to nearby structures by
//! the permutation and rotation IRA reports, one measurement serves many
//! predictions.
//!
//! Five predictors are compared against the depth an actual relaxation
//! recovers:
//!
//! - `model`, the stretch-only operator rebuilt at the target for free;
//! - `transported`, that operator plus the correction carried from the
//!   reference across the IRA correspondence;
//! - `naive`, the same correction applied with no permutation and no rotation.
//!   This is the control that decides whether IRA earns its cost: if a
//!   correspondence that ignores the structure does as well, the matching is
//!   not what is doing the work;
//! - `exact`, the true Hessian at the target, which is the harmonic limit and
//!   bounds what any transport could reach;
//! - `gradnorm`, `1/2 |g|^2`, the curvature-free baseline.
//!
//! # Metric
//!
//! Every predictor is scored after one global multiplicative scale, fitted as
//! the median of `true / predicted` over the whole sample. That is how the
//! number is consumed: the surrogate downstream regresses on the depth and
//! absorbs any constant, and the model Hessian's force constant `K0` is
//! declared arbitrary for exactly that reason. Scoring the raw value would
//! rank the predictors by whose units happen to match Lennard-Jones, which is
//! not a property anyone uses. Spearman rank correlation is reported alongside,
//! since it needs no fit at all.
//!
//! Two families of target. Family `same` perturbs the reference and stays in
//! its basin; family `other` relaxes a large perturbation into a different
//! minimum and perturbs that. The reported curve is depth error against IRA
//! match distance, in equal-count bins, which is what says when transport is
//! worth its cost.

use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::model_hessian;
use anneal_core::potentials::{PairKind, PairPotential};
use anneal_core::shape::match_shapes;
use anneal_core::sym_hessian::{Transport, TransportedCurvature, dense_model, pair_hessian};
use ndarray::{Array1, ArrayView1};
use std::time::Instant;

/// Deterministic uniform stream, so a rerun reproduces the table.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn centred(&mut self) -> f64 {
        self.unit() - 0.5
    }
}

/// Pushes apart any pair closer than `min_sep`.
///
/// A uniform perturbation occasionally lands two points on top of each other,
/// and the `r^-12` term then produces a gradient near 1e13 that no line search
/// recovers from. Without this the relaxations being measured are failures
/// rather than relaxations.
fn repair(x: &mut Array1<f64>, min_sep: f64) {
    let n = x.len() / 3;
    for _ in 0..50 {
        let mut worst = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut d = [0.0f64; 3];
                let mut r2 = 0.0;
                for k in 0..3 {
                    d[k] = x[3 * i + k] - x[3 * j + k];
                    r2 += d[k] * d[k];
                }
                let r = r2.sqrt();
                if r < min_sep && r > 1e-12 {
                    let push = 0.5 * (min_sep - r) / r;
                    for k in 0..3 {
                        x[3 * i + k] += push * d[k];
                        x[3 * j + k] -= push * d[k];
                    }
                    worst = worst.max(min_sep - r);
                }
            }
        }
        if worst < 1e-9 {
            break;
        }
    }
}

/// Relaxes to a minimum and reports the value, the point and the gradients used.
fn relax(pot: &PairPotential, x: ArrayView1<f64>, max_iter: usize) -> (f64, Array1<f64>, usize) {
    let mut opt = WarmLbfgs::default();
    opt.minimize(x, max_iter, |p| Some(pot.value_and_gradient(p)))
}

/// Median of a slice, which is the statistic a heavy-tailed error ratio needs.
fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// Names of the predictors, in the order `Row::predictions` holds them.
const NAMES: [&str; 5] = ["model", "transported", "naive", "exact", "gradnorm"];

/// One target structure and what each predictor said about it.
struct Row {
    family: &'static str,
    ira_distance: f64,
    true_depth: f64,
    predictions: [f64; 5],
    /// `||M^T H_A M - H_y||_F / ||H_y||_F`, the transport error on the operator
    /// itself, free of anharmonicity.
    operator_error: f64,
    /// The same with the identity correspondence: no permutation, no rotation.
    operator_error_naive: f64,
    /// `||H_model(y) - H_y||_F / ||H_y||_F`, what the free operator costs in the
    /// same currency. The bar transport has to clear.
    operator_error_model: f64,
}

/// Relative Frobenius distance between a transported reference Hessian and the
/// target's own.
///
/// The depth error mixes two things: how well the operator transports, and how
/// far the harmonic model is from the depth a real relaxation recovers. This
/// separates them, and it is the quantity a caller deciding whether to pay for
/// a correspondence actually needs.
fn operator_transfer_error(
    h_ref: &ndarray::Array2<f64>,
    h_target: &ndarray::Array2<f64>,
    t: &Transport,
    n: usize,
) -> f64 {
    let dim = 3 * n;
    let mut num = 0.0;
    let mut den = 0.0;
    let mut e = Array1::<f64>::zeros(dim);
    for c in 0..dim {
        e[c] = 1.0;
        // Column `c` of `M^T H_A M`, built through the transport rather than by
        // forming `M`, so the convention under test is the one shipped.
        let mapped = t.to_reference(e.view());
        let col = t.from_reference(h_ref.dot(&mapped).view());
        for r in 0..dim {
            let d = col[r] - h_target[[r, c]];
            num += d * d;
            den += h_target[[r, c]] * h_target[[r, c]];
        }
        e[c] = 0.0;
    }
    (num / den.max(1e-300)).sqrt()
}

/// Spearman rank correlation, which needs no scale fit and no distributional
/// assumption; the depths span three orders of magnitude and their errors are
/// heavy tailed, so a Pearson coefficient would be reporting the largest few.
fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 3 {
        return f64::NAN;
    }
    let rank = |v: &[f64]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(std::cmp::Ordering::Equal));
        let mut r = vec![0.0; n];
        let mut i = 0;
        while i < n {
            // Ties share the average rank, or a predictor that saturates would
            // be scored on the order its ties happened to arrive in.
            let mut j = i;
            while j + 1 < n && (v[idx[j + 1]] - v[idx[i]]).abs() < 1e-15 {
                j += 1;
            }
            let avg = (i + j) as f64 / 2.0;
            for &k in &idx[i..=j] {
                r[k] = avg;
            }
            i = j + 1;
        }
        r
    };
    let (ra, rb) = (rank(a), rank(b));
    let mean = (n as f64 - 1.0) / 2.0;
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (x, y) = (ra[i] - mean, rb[i] - mean);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da <= 0.0 || db <= 0.0 {
        return f64::NAN;
    }
    num / (da * db).sqrt()
}

fn main() {
    let n = 38;
    let pot = PairPotential::lennard_jones(n);
    let kind = PairKind::LennardJones;
    let solve_iters = 200;
    let kmax = 1.8;

    // A reference minimum, from a compact random start. Which minimum it is does
    // not matter; that it is genuinely relaxed does.
    let mut rng = Rng(20260806);
    let mut start = Array1::<f64>::zeros(3 * n);
    for v in start.iter_mut() {
        *v = rng.centred() * 4.4;
    }
    repair(&mut start, 0.85);
    let (e_ref, reference, ref_evals) = relax(&pot, start.view(), 4000);
    let ref_grad_norm = pot
        .value_and_gradient(reference.view())
        .1
        .iter()
        .fold(0.0f64, |a, z| a.max(z.abs()));
    println!(
        "reference minimum: E = {e_ref:.6}, |g|_inf = {ref_grad_norm:.2e}, {ref_evals} evaluations to reach it"
    );

    let t0 = Instant::now();
    let h_ref = pair_hessian(reference.view(), n, kind);
    let carried = TransportedCurvature::from_hessian(reference.view(), h_ref.view());
    let build_us = t0.elapsed().as_secs_f64() * 1e6;
    println!(
        "correction at the reference: ||H_true - H_model||_F = {:.3}, ||H_true||_F = {:.3}, built in {build_us:.0} us",
        carried.correction_norm(),
        h_ref.iter().map(|v| v * v).sum::<f64>().sqrt()
    );

    // The two operators' conditioning, which is what the depth solve inverts.
    // Reported because a better operator can predict the depth worse: the depth
    // is a quadratic form in the inverse, so a soft mode that transports badly
    // is amplified rather than averaged.
    {
        use anneal_core::sym_hessian::{restricted_condition, restricted_spectrum};
        let dim = 3 * n;
        let mut free = ndarray::Array2::<f64>::zeros((dim, dim - 3));
        // Any orthonormal basis of the complement of the translations; built by
        // Gram-Schmidt on the coordinate axes against the three uniform shifts.
        let mut cols: Vec<Array1<f64>> = Vec::new();
        let mut trans = Vec::new();
        for k in 0..3 {
            let mut t = Array1::<f64>::zeros(dim);
            for i in 0..n {
                t[3 * i + k] = 1.0 / (n as f64).sqrt();
            }
            trans.push(t);
        }
        for c in 0..dim {
            let mut v = Array1::<f64>::zeros(dim);
            v[c] = 1.0;
            for t in &trans {
                let d = v.dot(t);
                v.scaled_add(-d, t);
            }
            for u in &cols {
                let d = v.dot(u);
                v.scaled_add(-d, u);
            }
            let nrm = v.dot(&v).sqrt();
            if nrm > 1e-8 && cols.len() < dim - 3 {
                v /= nrm;
                cols.push(v);
            }
        }
        for (c, v) in cols.iter().enumerate() {
            for r in 0..dim {
                free[[r, c]] = v[r];
            }
        }
        let h_model = dense_model(reference.view(), n);
        let s_true = restricted_spectrum(h_ref.view(), free.view());
        let s_model = restricted_spectrum(h_model.view(), free.view());
        let show = |s: &[f64]| {
            s[..4]
                .iter()
                .map(|v| format!("{v:.2e}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!(
            "off the translations, four softest modes:\n  exact Hessian  {}  (cond {:?})\n  model Hessian  {}  (cond {:?})",
            show(&s_true),
            restricted_condition(h_ref.view(), free.view()).map(|v| format!("{v:.1}")),
            show(&s_model),
            restricted_condition(h_model.view(), free.view()).map(|v| format!("{v:.1}"))
        );
        println!(
            "  the depth solve deflates translations only, so the exact operator's three\n\
             \x20 rigid rotations stay in the space it inverts; the model's FLOOR shift is\n\
             \x20 what keeps them out of the denominator."
        );
    }

    // Other basins, from large perturbations relaxed back down.
    let mut basins: Vec<Array1<f64>> = Vec::new();
    while basins.len() < 12 {
        let mut y = reference.clone();
        for v in y.iter_mut() {
            *v += rng.centred() * 1.4;
        }
        repair(&mut y, 0.85);
        let (e, m, _) = relax(&pot, y.view(), 4000);
        if e.is_finite() && (e - e_ref).abs() > 1e-6 {
            basins.push(m);
        }
    }

    let mut rows: Vec<Row> = Vec::new();
    let mut ira_calls = 0usize;
    let mut ira_time = 0.0f64;
    let mut refused = 0usize;

    for (family, seeds) in [("same", 0..120usize), ("other", 120..240usize)] {
        for k in seeds {
            let base = if family == "same" {
                reference.clone()
            } else {
                basins[k % basins.len()].clone()
            };
            // Geometric amplitude sweep from 0.002 to 0.6, so the curve has a
            // clean low end where the correspondence is nearly exact and the
            // transport error must vanish. A linear sweep put every target in
            // the regime where it has already failed.
            let amp = 0.002 * (0.6f64 / 0.002).powf((k % 12) as f64 / 11.0);
            let mut y = base.clone();
            for v in y.iter_mut() {
                *v += rng.centred() * amp;
            }
            repair(&mut y, 0.85);

            let (e_y, g_y) = pot.value_and_gradient(y.view());
            let (e_min, _, _) = relax(&pot, y.view(), 4000);
            if !e_min.is_finite() || e_y - e_min <= 1e-9 {
                continue;
            }
            let true_depth = e_y - e_min;

            let t = Instant::now();
            let m = match_shapes(reference.view(), y.view(), kmax);
            ira_time += t.elapsed().as_secs_f64();
            ira_calls += 1;
            let m = match m {
                Ok(v) => v,
                Err(_) => {
                    refused += 1;
                    continue;
                }
            };
            let transport = match Transport::from_match(&m) {
                Some(v) => v,
                None => {
                    refused += 1;
                    continue;
                }
            };

            // The identity correspondence: no permutation, no rotation. Used
            // twice, for two different purposes. Against the target's own
            // Hessian it is the harmonic limit; against the reference's
            // correction it is the control that says whether IRA is doing the
            // work.
            let ident = Transport::new((0..n).collect(), [1., 0., 0., 0., 1., 0., 0., 0., 1.])
                .expect("the identity is a bijection");

            let model = model_hessian::depth(y.view(), n, g_y.view(), solve_iters);
            let transported = carried.depth(y.view(), g_y.view(), &transport, solve_iters);
            let naive = carried.depth(y.view(), g_y.view(), &ident, solve_iters);
            let h_y = pair_hessian(y.view(), n, kind);
            let here = TransportedCurvature::from_hessian(y.view(), h_y.view());
            let exact = here.depth(y.view(), g_y.view(), &ident, solve_iters);
            let gradnorm = 0.5 * g_y.dot(&g_y);

            rows.push(Row {
                family,
                ira_distance: m.distance,
                true_depth,
                predictions: [model, transported, naive, exact, gradnorm],
                operator_error: operator_transfer_error(&h_ref, &h_y, &transport, n),
                operator_error_naive: operator_transfer_error(&h_ref, &h_y, &ident, n),
                operator_error_model: {
                    let hm = dense_model(y.view(), n);
                    let num: f64 = (&hm - &h_y).iter().map(|v| v * v).sum();
                    let den: f64 = h_y.iter().map(|v| v * v).sum();
                    (num / den.max(1e-300)).sqrt()
                },
            });
        }
    }

    println!(
        "\n{} targets kept, {refused} refused for a correspondence IRA would not \
         give; {ira_calls} IRA calls at {:.2} ms each",
        rows.len(),
        1e3 * ira_time / ira_calls.max(1) as f64
    );

    // One global scale per predictor, fitted on the whole sample, because that
    // is the constant the downstream regression absorbs. Fitted once and reused
    // in every bin, so a bin cannot flatter a predictor by refitting inside it.
    let scale: Vec<f64> = (0..NAMES.len())
        .map(|p| {
            let mut ratios: Vec<f64> = rows
                .iter()
                .filter(|r| r.predictions[p].abs() > 1e-30)
                .map(|r| r.true_depth / r.predictions[p])
                .collect();
            median(&mut ratios)
        })
        .collect();
    println!("\nfitted global scale, true / predicted:");
    for (i, name) in NAMES.iter().enumerate() {
        println!("  {name:>12}  {:.4e}", scale[i]);
    }

    let err = |r: &Row, p: usize| {
        (scale[p] * r.predictions[p] - r.true_depth).abs() / r.true_depth.abs().max(1e-12)
    };

    // Equal-count bins in IRA distance, so every point on the curve rests on the
    // same number of targets. Fixed edges put ninety-five targets in one bin and
    // two in another, which is not a curve.
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&i, &j| {
        rows[i]
            .ira_distance
            .partial_cmp(&rows[j].ira_distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let bins = 10usize;
    println!(
        "\nmedian relative depth error after the global scale, and the transport \
         error on the operator itself, by IRA match distance"
    );
    println!(
        "{:>18} {:>6} {:>8} {:>12} {:>8} {:>8} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "IRA distance",
        "count",
        "model",
        "transported",
        "naive",
        "exact",
        "gradnorm",
        "op transp",
        "op naive",
        "op model",
        "rho(transp)"
    );
    for b in 0..bins {
        let lo = b * order.len() / bins;
        let hi = ((b + 1) * order.len() / bins).max(lo + 1).min(order.len());
        if lo >= hi {
            continue;
        }
        let sel: Vec<&Row> = order[lo..hi].iter().map(|&i| &rows[i]).collect();
        let med: Vec<f64> = (0..NAMES.len())
            .map(|p| {
                let mut e: Vec<f64> = sel.iter().map(|r| err(r, p)).collect();
                median(&mut e)
            })
            .collect();
        let truths: Vec<f64> = sel.iter().map(|r| r.true_depth).collect();
        let preds: Vec<f64> = sel.iter().map(|r| r.predictions[1]).collect();
        let mut op: Vec<f64> = sel.iter().map(|r| r.operator_error).collect();
        let mut op_naive: Vec<f64> = sel.iter().map(|r| r.operator_error_naive).collect();
        let mut op_model: Vec<f64> = sel.iter().map(|r| r.operator_error_model).collect();
        println!(
            "{:>18} {:>6} {:>8.3} {:>12.3} {:>8.3} {:>8.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>10.3}",
            format!(
                "[{:.4}, {:.4}]",
                sel[0].ira_distance,
                sel[sel.len() - 1].ira_distance
            ),
            sel.len(),
            med[0],
            med[1],
            med[2],
            med[3],
            med[4],
            median(&mut op),
            median(&mut op_naive),
            median(&mut op_model),
            spearman(&truths, &preds)
        );
    }

    println!("\nover the whole sample and by family, with Spearman rank correlation");
    for family in ["all", "same", "other"] {
        let sel: Vec<&Row> = rows
            .iter()
            .filter(|r| family == "all" || r.family == family)
            .collect();
        if sel.is_empty() {
            continue;
        }
        let mut d: Vec<f64> = sel.iter().map(|r| r.ira_distance).collect();
        println!(
            "  {family:>5}: {:>3} targets, median IRA distance {:.3}",
            sel.len(),
            median(&mut d)
        );
        let truths: Vec<f64> = sel.iter().map(|r| r.true_depth).collect();
        for (p, name) in NAMES.iter().enumerate() {
            let mut e: Vec<f64> = sel.iter().map(|r| err(r, p)).collect();
            let preds: Vec<f64> = sel.iter().map(|r| r.predictions[p]).collect();
            println!(
                "      {name:>12}  median error {:.3}  rho {:.3}",
                median(&mut e),
                spearman(&truths, &preds)
            );
        }
    }

    // Cost, in the currency the ledger charges.
    println!(
        "\ncost per prediction: the exact Hessian at a target is {} gradients by \
         central difference, against one IRA call at {:.2} ms and no gradients for \
         the transported one. The correction is built once per reference, in \
         {build_us:.0} us here.",
        6 * n,
        1e3 * ira_time / ira_calls.max(1) as f64
    );
}
