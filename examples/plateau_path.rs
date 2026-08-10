//! Does a path leave the plateau that no single move escapes?
//!
//! The criterion is direct: from the structure a 75-point search settles into,
//! reach anything below it. None of 1800 single moves across the whole kernel
//! set does, so this is the one test that says whether depth greater than one
//! buys what depth one cannot.
//!
//! Endpoints come from constructed morphologies, since a path between two
//! structures in the same funnel lands back in it, and a chain that has settled
//! has only its own funnel to offer.

use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::path::{interpolate_path, transverse_path};
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

fn relax(x: ArrayView1<f64>, iters: usize) -> (f64, Array1<f64>) {
    let mut opt = WarmLbfgs::default();
    let (f, xr, _) = opt.minimize(x, iters, |v| Some(lj(v)));
    (f, xr)
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

/// Lattice sites of a family, nearest `n` to the origin, jittered.
fn seed(n: usize, family: &str, scale: f64, rng: &mut Rng) -> Option<Array1<f64>> {
    let mut pts: Vec<[f64; 3]> = Vec::new();
    match family {
        "icosahedral" => {
            let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
            let base: Vec<[f64; 3]> = {
                let mut v = Vec::new();
                for s1 in [-1.0_f64, 1.0] {
                    for s2 in [-1.0_f64, 1.0] {
                        v.push([0.0, s1, s2 * phi]);
                        v.push([s1, s2 * phi, 0.0]);
                        v.push([s1 * phi, 0.0, s2]);
                    }
                }
                v
            };
            pts.push([0.0, 0.0, 0.0]);
            for shell in 1..=3 {
                for b in &base {
                    pts.push([
                        b[0] * shell as f64,
                        b[1] * shell as f64,
                        b[2] * shell as f64,
                    ]);
                    for k in 1..shell {
                        let t = k as f64 / shell as f64;
                        pts.push([b[0] * t * 2.0, b[1] * t * 2.0, b[2] * t * 2.0]);
                    }
                }
            }
        }
        "decahedral" => {
            for iz in -4..=4 {
                let z = iz as f64 * (2.0_f64 / 3.0).sqrt();
                for ir in 0..=4 {
                    if ir == 0 {
                        pts.push([0.0, 0.0, z]);
                        continue;
                    }
                    for sector in 0..5 {
                        let base = 2.0 * std::f64::consts::PI * sector as f64 / 5.0;
                        for k in 0..ir {
                            let t = k as f64 / ir as f64;
                            let a = [base.cos() * ir as f64, base.sin() * ir as f64];
                            let nb = base + 2.0 * std::f64::consts::PI / 5.0;
                            let b = [nb.cos() * ir as f64, nb.sin() * ir as f64];
                            pts.push([(1.0 - t) * a[0] + t * b[0], (1.0 - t) * a[1] + t * b[1], z]);
                        }
                    }
                }
            }
        }
        _ => {
            // Face-centred cubic.
            for i in -3..=3 {
                for j in -3..=3 {
                    for k in -3..=3 {
                        for b in [
                            [0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.0],
                            [0.5, 0.0, 0.5],
                            [0.0, 0.5, 0.5],
                        ] {
                            pts.push([
                                (i as f64 + b[0]) * 2.0_f64.sqrt(),
                                (j as f64 + b[1]) * 2.0_f64.sqrt(),
                                (k as f64 + b[2]) * 2.0_f64.sqrt(),
                            ]);
                        }
                    }
                }
            }
        }
    }
    for p in pts.iter_mut() {
        for c in p.iter_mut() {
            *c *= scale;
        }
    }
    pts.sort_by(|a, b| {
        let ra = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
        let rb = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
        ra.partial_cmp(&rb).unwrap()
    });
    // A lattice can yield fewer sites than the cluster needs, and taking the
    // first n regardless indexes past the end.
    if pts.len() < n {
        return None;
    }
    let mut v = Vec::with_capacity(3 * n);
    for p in pts.iter().take(n) {
        for c in p {
            v.push(c + rng.uniform(-0.02, 0.02));
        }
    }
    let mut x = Array1::from(v);
    let mut centre = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            centre[k] += x[3 * i + k];
        }
    }
    for c in centre.iter_mut() {
        *c /= n as f64;
    }
    for i in 0..n {
        for k in 0..3 {
            x[3 * i + k] -= centre[k];
        }
    }
    Some(x)
}

fn main() {
    let n = 75;
    let plateau_target = -396.282247;
    let reference = -397.492331;
    let mut rng = Rng(20260803);

    // Reach the plateau by ordinary monotonic hopping.
    let start = seed(n, "fcc", 1.09, &mut rng).expect("fcc lattice is large enough");
    let (mut e, mut x) = relax(start.view(), 4000);
    // Metropolis rather than monotonic, at the temperature the literature uses
    // for the quenched surface. Monotonic descent from one seed reached only
    // -394.41, which is not the plateau this test is about.
    let (mut be, mut bx) = (e, x.clone());
    for _ in 0..20000 {
        let mut t = x.clone();
        for v in t.iter_mut() {
            *v += rng.uniform(-0.38, 0.38);
        }
        let (e2, x2) = relax(t.view(), 400);
        if e2 < e || rng.next() < ((e - e2) / 0.8).exp() {
            e = e2;
            x = x2;
        }
        if e < be {
            be = e;
            bx = x.clone();
        }
        if be <= plateau_target + 1e-3 {
            break;
        }
    }
    let (e, x) = (be, bx);
    println!("start {e:.6}  (plateau {plateau_target}, reference {reference})");

    // Endpoints read from the validated constructor, which reproduces the
    // LJ13, LJ38 and LJ55 references exactly. The Rust lattice builder used
    // before yielded fewer than 75 icosahedral sites, so the family the
    // plateau actually belongs to was the one family never tested.
    let mut targets: Vec<(String, f64, Array1<f64>)> = Vec::new();
    let seed_file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "seeds_75.txt".into());
    let text = std::fs::read_to_string(&seed_file).expect("seed file");
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let Some(name) = parts.next() else { continue };
        let Some(energy) = parts.next().and_then(|v| v.parse::<f64>().ok()) else {
            continue;
        };
        let coords: Vec<f64> = parts.filter_map(|v| v.parse().ok()).collect();
        if coords.len() == 3 * n {
            targets.push((name.to_string(), energy, Array1::from(coords)));
        }
    }
    println!("{} endpoints read", targets.len());

    // Two methods on the same endpoints: relax each image fully, which the
    // previous measurement showed slides every image onto an endpoint, and
    // relax perpendicular to the path so an image stays where it was placed.
    let mut best_escape = f64::INFINITY;
    let mut best_from = String::new();
    let mut best_transverse = f64::INFINITY;
    let mut best_t_from = String::new();
    for (name, te, tx) in &targets {
        let out = interpolate_path(
            x.view(),
            tx.view(),
            15,
            |img| Some(relax(img, 400)),
            |st| {
                let (ev, _) = lj(st);
                (ev - e).abs() > 1e-4
            },
        );
        let esc = out.best_escape().map(|p| p.energy).unwrap_or(f64::NAN);
        if esc < best_escape {
            best_escape = esc;
            best_from = name.clone();
        }

        // The band, then a full relaxation of each settled image: the band
        // places an image in the corridor, and the relaxation says which basin
        // that place belongs to.
        let band = transverse_path(x.view(), tx.view(), 15, 300, 0.02, |v| Some(lj(v)));
        let mut t_best = f64::INFINITY;
        for (_, img) in &band {
            let (fe, _) = relax(img.view(), 600);
            if fe < t_best {
                t_best = fe;
            }
        }
        if t_best < best_transverse {
            best_transverse = t_best;
            best_t_from = name.clone();
        }
        println!(
            "  {name:<14} target {te:>10.4}  full-relax escape {esc:>10.4}  \
             transverse best {t_best:>10.4}"
        );
    }
    println!("\nfull relaxation  best {best_escape:.6} from {best_from}");
    println!("transverse band  best {best_transverse:.6} from {best_t_from}");
    println!(
        "criterion E < -396.5 from the plateau: full {}, transverse {}",
        if best_escape < -396.5 {
            "MET"
        } else {
            "not met"
        },
        if best_transverse < -396.5 {
            "MET"
        } else {
            "not met"
        }
    );
}
