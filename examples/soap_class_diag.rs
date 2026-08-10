//! Class residual vs mean residual on Mackay ico, then 200-trial escape
//! from a quenched icosahedral start.

use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::soap::{
    SoapSpec, class_residual_rms, ih_dominated, mean_residual_rms, step_away, step_away_mean,
};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

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

fn relax(x: ArrayView1<f64>, iters: usize) -> (f64, Array1<f64>) {
    let mut opt = WarmLbfgs::default();
    let (f, xr, _) = opt.minimize(x, iters, |v| Some(lj(v)));
    (f, xr)
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
    let s = 1.0 / (1.0 + p * p).sqrt();
    let mut x = Array1::<f64>::zeros(3 * 13);
    for (i, v) in verts.iter().enumerate() {
        for k in 0..3 {
            x[3 * (i + 1) + k] = s * v[k];
        }
    }
    x
}

fn ico_sites(n: usize) -> Array1<f64> {
    let offsets = anneal_core::structure::Template::Icosahedral.points();
    let pts = anneal_core::lattice::grow(&offsets, n);
    let mut x = Array1::zeros(3 * n);
    for (i, p) in pts.iter().enumerate().take(n) {
        x[3 * i] = p[0];
        x[3 * i + 1] = p[1];
        x[3 * i + 2] = p[2];
    }
    x
}

fn escape_table(label: &str, start: ArrayView1<f64>, trials: usize, class: bool) {
    let spec = SoapSpec::default();
    let (e0, x0) = relax(start, 400);
    let mut rng = StdRng::seed_from_u64(20260810);
    let mut below = 0usize;
    let mut left = 0usize;
    let mut best = e0;
    for _ in 0..trials {
        let y = if class {
            step_away(x0.view(), &[], spec, 0.35, &mut rng)
        } else {
            step_away_mean(x0.view(), spec, 0.35, &mut rng)
        };
        let (e1, _) = relax(y.view(), 200);
        if (e1 - e0).abs() > 1e-4 {
            left += 1;
        }
        if e1 < e0 - 1e-4 {
            below += 1;
        }
        if e1 < best {
            best = e1;
        }
    }
    println!(
        "{label} class={class} start {e0:.6}  left {left}/{trials}  below {below}/{trials}  best {best:.6}"
    );
}

fn main() {
    let spec = SoapSpec::default();
    let x13 = ico13();
    let (e13, x13) = relax(x13.view(), 400);
    let mean = mean_residual_rms(x13.view(), spec);
    let class = class_residual_rms(x13.view(), spec);
    println!("LJ13 ico energy {e13:.6}  mean_rms {mean:.6}  class_rms {class:.6}  ih {}", ih_dominated(x13.view(), spec));

    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(55);
    let trials: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let raw = ico_sites(n);
    let (mut e, mut x) = relax(raw.view(), 2000);
    // LJ75 ico competitor is -396.282249. Lattice grow alone sits much higher.
    if n >= 75 {
        let plateau = -396.282249;
        let mut rng = RngWalk(20260810);
        let (mut be, mut bx) = (e, x.clone());
        for _ in 0..8000 {
            let mut t = x.clone();
            for v in t.iter_mut() {
                *v += rng.uniform(-0.38, 0.38);
            }
            let (e2, x2) = relax(t.view(), 200);
            if e2 < e || rng.next() < ((e - e2) / 0.8).exp() {
                e = e2;
                x = x2;
            }
            if e < be {
                be = e;
                bx = x.clone();
            }
            if be <= plateau + 1e-3 {
                break;
            }
        }
        e = be;
        x = bx;
        println!("LJ{n} hop-to-plateau {e:.6} (target {plateau})");
    }
    println!(
        "LJ{n} start {e:.6}  mean_rms {:.6}  class_rms {:.6}  ih {}",
        mean_residual_rms(x.view(), spec),
        class_residual_rms(x.view(), spec),
        ih_dominated(x.view(), spec)
    );
    if trials > 0 {
        escape_table(&format!("LJ{n}"), x.view(), trials, false);
        escape_table(&format!("LJ{n}"), x.view(), trials, true);
    }
}

struct RngWalk(u64);
impl RngWalk {
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
