//! What is the sorted-pair descriptor distance between LJ75 minima?
//!
//! Every basin-keyed mechanism here compares that distance against
//! `merge_radius`. If the threshold is far below the distance between distinct
//! minima then every structure is its own basin, the per-basin bias never
//! accumulates, an escape test is always true and a return test never is. The
//! scale was calibrated for a shape metric and never for this descriptor.

use anneal_core::bias::{Fingerprint, SortedPairs};
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
            let c = 24.0 * inv2 * (2.0 * inv12 - inv6);
            for k in 0..3 {
                g[3 * i + k] -= c * d[k];
                g[3 * j + k] += c * d[k];
            }
        }
    }
    (e, g)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

fn main() {
    let n = 75;
    let fp = SortedPairs { n_points: n };
    let mut rng = Rng(7);
    let relax = |x: ArrayView1<f64>| {
        let mut o = WarmLbfgs::default();
        let (f, xr, _) = o.minimize(x, 3000, |v| Some(lj(v)));
        (f, xr)
    };

    // Independent minima from random starts.
    let radius = 0.9 * (n as f64).cbrt();
    let mut indep = Vec::new();
    for _ in 0..10 {
        let mut s = Array1::zeros(3 * n);
        for v in s.iter_mut() {
            *v = rng.uniform(-radius, radius);
        }
        indep.push(relax(s.view()).1);
    }
    // Perturbation neighbours of one of them, which is what a chain compares.
    let base = indep[0].clone();
    let mut neigh = Vec::new();
    for _ in 0..10 {
        let mut t = base.clone();
        for v in t.iter_mut() {
            *v += rng.uniform(-0.38, 0.38);
        }
        neigh.push(relax(t.view()).1);
    }

    let dist = |a: &Array1<f64>, b: &Array1<f64>| {
        let da = fp.describe(a.view());
        let db = fp.describe(b.view());
        da.iter()
            .zip(db.iter())
            .map(|(p, q)| (p - q) * (p - q))
            .sum::<f64>()
            .sqrt()
    };

    let mut across = Vec::new();
    for i in 0..indep.len() {
        for j in (i + 1)..indep.len() {
            across.push(dist(&indep[i], &indep[j]));
        }
    }
    let mut hops: Vec<f64> = neigh.iter().map(|x| dist(&base, x)).collect();
    across.sort_by(|a, b| a.partial_cmp(b).unwrap());
    hops.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q = |v: &Vec<f64>, f: f64| v[((v.len() - 1) as f64 * f) as usize];
    println!("sorted-pair descriptor distance");
    println!(
        "  between independent minima: min {:.4}  median {:.4}  max {:.4}",
        across[0],
        q(&across, 0.5),
        across[across.len() - 1]
    );
    println!(
        "  one hop from a minimum:     min {:.4}  median {:.4}  max {:.4}",
        hops[0],
        q(&hops, 0.5),
        hops[hops.len() - 1]
    );
    println!("\n  merge_radius in use: 0.0100");
    println!(
        "  fraction of independent pairs the threshold calls the same basin: {:.0}%",
        100.0 * across.iter().filter(|d| **d <= 0.01).count() as f64 / across.len() as f64
    );
    println!(
        "  fraction of one-hop neighbours it calls the same basin:           {:.0}%",
        100.0 * hops.iter().filter(|d| **d <= 0.01).count() as f64 / hops.len() as f64
    );
}
