//! Replay a recorded chain through this crate's relaxation.
//!
//! The trials come from a run of the reference implementation, so both see
//! identical inputs and any difference in the screened or relaxed energy is
//! this crate's numerics rather than a different search path.

use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::{Array1, ArrayView1};

fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let d = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let inv2 = 1.0 / r2;
            let inv6 = inv2 * inv2 * inv2;
            let inv12 = inv6 * inv6;
            e += 4.0 * (inv12 - inv6);
            let coef = 24.0 * inv2 * (2.0 * inv12 - inv6);
            for k in 0..3 {
                g[3 * i + k] -= coef * d[k];
                g[3 * j + k] += coef * d[k];
            }
        }
    }
    (e, g)
}

/// Pulls the numeric arrays out of the recorded JSON without a parser crate.
fn field_array(rec: &str, key: &str) -> Option<Vec<f64>> {
    let pat = format!("\"{key}\": [");
    let start = rec.find(&pat)? + pat.len();
    let end = start + rec[start..].find(']')?;
    Some(
        rec[start..end]
            .split(',')
            .filter_map(|v| v.trim().parse::<f64>().ok())
            .collect(),
    )
}

fn field_num(rec: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\": ");
    let start = rec.find(&pat)? + pat.len();
    let rest = &rec[start..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse::<f64>().ok()
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "py_trace.json".into());
    let raw = std::fs::read_to_string(&path).expect("cannot read the trace");
    // Records are separated by "},{" in the dumped array.
    let records: Vec<&str> = raw.split("}, {").collect();
    println!(
        "{:>5} {:>13} {:>13} {:>13} {:>8}",
        "step", "py e_new", "rust screen", "rust full", "py full"
    );
    let (mut agree, mut total) = (0usize, 0usize);
    let mut screen_pass_rust = 0usize;
    let mut screen_pass_py = 0usize;
    for (i, rec) in records.iter().enumerate() {
        let (Some(trial), Some(py_e), Some(best)) = (
            field_array(rec, "trial"),
            field_num(rec, "e_new"),
            field_num(rec, "best"),
        ) else {
            continue;
        };
        let py_full = rec.contains("\"full\": true");
        if py_full {
            screen_pass_py += 1;
        }
        let x = Array1::from(trial);
        let n = x.len() / 3;
        // The reference reports shifted energies; this compares like with like
        // by relaxing the same coordinates with the same step counts.
        let mut opt = WarmLbfgs::default();
        let (e_screen, x_screen, _) = opt.minimize(x.view(), 25, |v| Some(lj(v)));
        let passes = e_screen <= best + 2.0;
        if passes {
            screen_pass_rust += 1;
        }
        let mut opt2 = WarmLbfgs::default();
        let (e_full, _, _) = opt2.minimize(x_screen.view(), 200, |v| Some(lj(v)));
        total += 1;
        if (e_full - py_e).abs() < 1e-3 {
            agree += 1;
        }
        if i < 12 {
            println!("{i:>5} {py_e:>13.5} {e_screen:>13.5} {e_full:>13.5} {py_full:>8}  n={n}");
        }
    }
    println!("\nfull relaxation agrees with the reference on {agree}/{total} trials");
    println!(
        "screen promotes: rust {screen_pass_rust}/{total}, reference {screen_pass_py}/{total}"
    );
}
