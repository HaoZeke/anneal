//! Drive the shipped fivefold hop on a quenched LJ75 icosahedral competitor.
//!
//! Grows a Mackay lattice, basin-hops to the −396.282249 shelf, then
//! applies `step_away_fivefold` twice and quenches. Prints whether the
//! hop is identity and whether the quench is still on the shelf.
//!
//! Usage: ico75_hop [xyz-out]

use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::soap::{
    fivefold_axis_table, fivefold_length, fivefold_probe, step_away_fivefold,
    step_away_fivefold_about,
};
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

const PLATEAU: f64 = -396.282249;
const RMSD: f64 = 0.35;

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

fn energy(x: ArrayView1<f64>) -> f64 {
    lj(x).0
}

fn hop_rms(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    let n = (x.len() / 3).max(1) as f64;
    let mut s = 0.0;
    for i in 0..x.len() {
        let d = y[i] - x[i];
        s += d * d;
    }
    (s / n).sqrt()
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

fn to_plateau() -> (f64, Array1<f64>) {
    let raw = ico_sites(75);
    let (mut e, mut x) = relax(raw.view(), 2000);
    println!("grow_quench {e:.9}");
    if e <= PLATEAU + 1e-3 {
        return (e, x);
    }
    let mut rng = RngWalk(20260810);
    let (mut be, mut bx) = (e, x.clone());
    for hop in 0..8000 {
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
            println!("best {be:.9} hop {hop}");
        }
        if be <= PLATEAU + 1e-3 {
            break;
        }
    }
    (be, bx)
}

fn report(tag: &str, start_e: f64, start: ArrayView1<f64>, y: ArrayView1<f64>) {
    let rms = hop_rms(start, y);
    let ident = rms < 1e-8;
    let d5_y = fivefold_length(y);
    let (eq, xq) = relax(y, 400);
    let d5_q = fivefold_length(xq.view());
    let on_shelf = (eq - PLATEAU).abs() < 1e-4;
    println!(
        "{tag} hop_rms {rms:.8} identity {ident} d5_hop {d5_y:.6} quench {eq:.9} d5_q {d5_q:.6} on_shelf {on_shelf} delta_e {:+.6}",
        eq - start_e
    );
}

fn dump_xyz(path: &str, x: ArrayView1<f64>, e: f64) {
    use std::io::Write;
    let n = x.len() / 3;
    let mut f = std::fs::File::create(path).expect(path);
    writeln!(f, "{n}").unwrap();
    writeln!(f, "LJ75 ico competitor e={e:.9}").unwrap();
    for i in 0..n {
        writeln!(
            f,
            "X {:.12} {:.12} {:.12}",
            x[3 * i],
            x[3 * i + 1],
            x[3 * i + 2]
        )
        .unwrap();
    }
}

fn load_xyz(path: &str) -> Array1<f64> {
    let text = std::fs::read_to_string(path).expect(path);
    let mut vals = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = t.split_whitespace().collect();
        let start = if parts.len() >= 4 { 1 } else { 0 };
        if parts.len() < start + 3 {
            continue;
        }
        if let Ok(_n) = parts[0].parse::<usize>() {
            if parts.len() == 1 {
                continue;
            }
        }
        for k in 0..3 {
            if let Ok(v) = parts[start + k].parse::<f64>() {
                vals.push(v);
            }
        }
    }
    assert_eq!(vals.len(), 225, "{path} has {} coords, want 225", vals.len());
    Array1::from(vals)
}

fn main() {
    let path = std::env::args().nth(1);
    let (e0, x0) = if let Some(ref p) = path {
        if std::path::Path::new(p).is_file() {
            let x = load_xyz(p);
            let e = energy(x.view());
            println!("loaded {p} energy {e:.9}");
            (e, x)
        } else {
            to_plateau()
        }
    } else {
        to_plateau()
    };
    println!("plateau {e0:.9} target {PLATEAU}");
    for (i, (ax, d)) in fivefold_axis_table(x0.view()).iter().enumerate() {
        println!(
            "axis {i} d5 {d:.6} n {:.3} {:.3} {:.3}",
            ax[0], ax[1], ax[2]
        );
    }
    let p = fivefold_probe(x0.view(), RMSD);
    println!(
        "probe d5 {:.6} n_axes {} residual_rms {:.6} gated {} pentagon {} top5 {:.3} top12 {:.3}",
        p.d5,
        p.n_axes,
        p.residual_rms,
        p.gated,
        p.used_pentagon,
        p.top5_share,
        p.top12_share
    );
    #[cfg(feature = "featomic")]
    {
        let leftover = anneal_core::featomic_hop::leftover_rms(x0.view(), 3.5, None, None);
        println!(
            "descriptor calculator {} lmin {} leftover_rms {:.6}",
            anneal_core::featomic_hop::CALCULATOR,
            anneal_core::featomic_hop::LMIN,
            leftover
        );
        let spec = anneal_core::soap::SoapSpec {
            n_max: 3,
            l_max: 6,
            rcut_nn: 3.5,
        };
        for (tag, seed) in [("trial1", 1u64), ("trial2", 2)] {
            let mut rng = StdRng::seed_from_u64(seed);
            let y = anneal_core::soap::step_away_cloud(
                x0.view(),
                spec,
                RMSD,
                None,
                None,
                None,
                &mut rng,
            );
            report(tag, e0, x0.view(), y.view());
        }
        for cap in [
            0.20_f64, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.00,
        ] {
            let mut rng = StdRng::seed_from_u64(3);
            let y = anneal_core::featomic_hop::step_away_featomic_at(
                x0.view(),
                cap,
                3.5,
                None,
                None,
                &mut rng,
            );
            let rms = hop_rms(x0.view(), y.view());
            let (eq, _) = relax(y.view(), 400);
            println!(
                "featomic_amp {cap:.2} hop_rms {rms:.6} quench {eq:.9} on_shelf {} dE {:+.4}",
                (eq - PLATEAU).abs() < 1e-4,
                eq - e0
            );
        }
    }
    if let Some(ref pth) = path {
        if !std::path::Path::new(pth).is_file() {
            dump_xyz(pth, x0.view(), e0);
            println!("wrote {pth}");
        }
    }
    println!("trial 1");
    let mut rng1 = StdRng::seed_from_u64(1);
    let y1 = step_away_fivefold(x0.view(), RMSD, &mut rng1);
    report("trial1", e0, x0.view(), y1.view());
    println!("trial 2");
    let mut rng2 = StdRng::seed_from_u64(2);
    let y2 = step_away_fivefold(x0.view(), RMSD, &mut rng2);
    report("trial2", e0, x0.view(), y2.view());
    for cap in [0.35_f64, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.20] {
        let mut rng = StdRng::seed_from_u64(3);
        let y = step_away_fivefold(x0.view(), cap, &mut rng);
        let rms = hop_rms(x0.view(), y.view());
        let (eq, _) = relax(y.view(), 400);
        println!(
            "amp {cap:.2} hop_rms {rms:.6} quench {eq:.9} on_shelf {} dE {:+.4}",
            (eq - PLATEAU).abs() < 1e-4,
            eq - e0
        );
    }
    let axes = fivefold_axis_table(x0.view());
    for (i, (ax, d5)) in axes.iter().enumerate() {
        let y = step_away_fivefold_about(x0.view(), 0.75, *ax);
        let (eq, _) = relax(y.view(), 400);
        println!(
            "ax {i} d5 {d5:.3} cap 0.75 quench {eq:.9} dE {:+.4} on_shelf {}",
            eq - e0,
            (eq - PLATEAU).abs() < 1e-4
        );
    }
    // Forced walk: always take the quench. If this never goes below
    // the shelf, Metropolis cannot either.
    let mut rngw = StdRng::seed_from_u64(17);
    let mut ew = e0;
    let mut xw = x0.clone();
    let mut below = 0usize;
    for hop in 0..12 {
        let y = step_away_fivefold(xw.view(), RMSD, &mut rngw);
        let (e2, x2) = relax(y.view(), 400);
        ew = e2;
        xw = x2;
        if ew < PLATEAU - 1e-4 {
            below += 1;
        }
        println!(
            "walk {hop} e {ew:.9} on_shelf {} below_ico {}",
            (ew - PLATEAU).abs() < 1e-4,
            ew < PLATEAU - 1e-4
        );
    }
    println!("walk_below {below}/12 best_walk {ew:.9}");
    // 40 Metropolis hops from the shelf at T=0.8, the production temperature.
    let mut rng = StdRng::seed_from_u64(11);
    let mut e = e0;
    let mut x = x0.clone();
    let mut best = e0;
    let mut off = 0usize;
    for hop in 0..40 {
        let y = step_away_fivefold(x.view(), RMSD, &mut rng);
        let (e2, x2) = relax(y.view(), 400);
        let acc = e2 < e || rng.random::<f64>() < ((e - e2) / 0.8).exp();
        if acc {
            e = e2;
            x = x2;
        }
        if e < best {
            best = e;
        }
        if (e - PLATEAU).abs() > 1e-4 {
            off += 1;
        }
        println!(
            "mh {hop} e {e:.9} acc {acc} on_shelf {} best {best:.9}",
            (e - PLATEAU).abs() < 1e-4
        );
    }
    println!("mh_off {off}/40 best {best:.9}");

    #[cfg(feature = "featomic")]
    {
        const MARKS: f64 = -397.492331;
        let spec = anneal_core::soap::SoapSpec {
            n_max: 3,
            l_max: 6,
            rcut_nn: 3.5,
        };
        let classify = |e: f64| {
            if (e - MARKS).abs() < 1e-4 {
                "marks"
            } else if (e - PLATEAU).abs() < 1e-4 {
                "ico"
            } else if e < PLATEAU - 1e-4 {
                "below_ico"
            } else if e > PLATEAU + 3.0 {
                "melt"
            } else {
                "other"
            }
        };
        // Force the high-l leftover at 0.75: that is the melt hop.
        let mut rngm = StdRng::seed_from_u64(99);
        let ym = anneal_core::featomic_hop::step_away_featomic_at(
            x0.view(),
            0.75,
            3.5,
            None,
            None,
            &mut rngm,
        );
        let (em, xm) = relax(ym.view(), 800);
        println!(
            "melt_start e {em:.9} class {} dE {:+.4}",
            classify(em),
            em - e0
        );
        // Always take the shipped hop quench. From melt, downhill is free.
        let mut e = em;
        let mut x = xm.clone();
        let mut best = em;
        let mut n_marks = 0usize;
        let mut n_ico = 0usize;
        let mut n_below = 0usize;
        for hop in 0..40 {
            let y = anneal_core::soap::step_away_cloud(
                x.view(),
                spec,
                RMSD,
                None,
                None,
                None,
                &mut rngm,
            );
            let (e2, x2) = relax(y.view(), 400);
            e = e2;
            x = x2;
            if e < best {
                best = e;
            }
            let c = classify(e);
            if c == "marks" {
                n_marks += 1;
            }
            if c == "ico" {
                n_ico += 1;
            }
            if c == "below_ico" || c == "marks" {
                n_below += 1;
            }
            println!(
                "melt_walk {hop} e {e:.9} class {c} best {best:.9}"
            );
        }
        println!(
            "melt_walk_summary marks {n_marks}/40 ico {n_ico}/40 below {n_below}/40 best {best:.9}"
        );
        // Metropolis at production T=0.8 from the melt, shipped hop.
        let mut rng = StdRng::seed_from_u64(101);
        e = em;
        x = xm.clone();
        best = em;
        let mut acc_n = 0usize;
        n_marks = 0;
        n_ico = 0;
        n_below = 0;
        for hop in 0..40 {
            let y = anneal_core::soap::step_away_cloud(
                x.view(),
                spec,
                RMSD,
                None,
                None,
                None,
                &mut rng,
            );
            let (e2, x2) = relax(y.view(), 400);
            let acc = e2 < e || rng.random::<f64>() < ((e - e2) / 0.8).exp();
            if acc {
                e = e2;
                x = x2;
                acc_n += 1;
            }
            if e < best {
                best = e;
            }
            let c = classify(e);
            if c == "marks" {
                n_marks += 1;
            }
            if c == "ico" {
                n_ico += 1;
            }
            if c == "below_ico" || c == "marks" {
                n_below += 1;
            }
            println!(
                "melt_mh {hop} e {e:.9} acc {acc} class {c} best {best:.9}"
            );
        }
        println!(
            "melt_mh_summary acc {acc_n}/40 marks {n_marks}/40 ico {n_ico}/40 below {n_below}/40 best {best:.9}"
        );
        // Wales–Doye random step from the same melt, T=0.8, amplitude 0.38.
        let mut rng = StdRng::seed_from_u64(103);
        e = em;
        x = xm.clone();
        best = em;
        acc_n = 0;
        n_marks = 0;
        n_ico = 0;
        n_below = 0;
        for hop in 0..80 {
            let mut y = x.clone();
            for v in y.iter_mut() {
                *v += rng.random::<f64>() * 0.76 - 0.38;
            }
            let (e2, x2) = relax(y.view(), 400);
            let acc = e2 < e || rng.random::<f64>() < ((e - e2) / 0.8).exp();
            if acc {
                e = e2;
                x = x2;
                acc_n += 1;
            }
            if e < best {
                best = e;
            }
            let c = classify(e);
            if c == "marks" {
                n_marks += 1;
            }
            if c == "ico" {
                n_ico += 1;
            }
            if c == "below_ico" || c == "marks" {
                n_below += 1;
            }
            if hop % 5 == 4 || c == "marks" || c == "below_ico" {
                println!(
                    "melt_bh {hop} e {e:.9} acc {acc} class {c} best {best:.9}"
                );
            }
        }
        println!(
            "melt_bh_summary acc {acc_n}/80 marks {n_marks}/80 ico {n_ico}/80 below {n_below}/80 best {best:.9}"
        );
        // Same random BH, but Metropolis against the incumbent ico energy
        // after a forced melt (the production question: take the melt, then
        // explore at T=0.8 with ico as the accepted best).
        let mut rng = StdRng::seed_from_u64(107);
        e = em;
        x = xm.clone();
        best = e0.min(em);
        acc_n = 0;
        n_marks = 0;
        let mut landed_ico = 0usize;
        for hop in 0..80 {
            let mut y = x.clone();
            for v in y.iter_mut() {
                *v += rng.random::<f64>() * 0.76 - 0.38;
            }
            let (e2, x2) = relax(y.view(), 400);
            let acc = e2 < e || rng.random::<f64>() < ((e - e2) / 0.8).exp();
            if acc {
                e = e2;
                x = x2;
                acc_n += 1;
            }
            if e < best {
                best = e;
            }
            let c = classify(e);
            if c == "marks" {
                n_marks += 1;
            }
            if c == "ico" {
                landed_ico += 1;
            }
            if hop % 5 == 4 || c == "marks" || c == "below_ico" {
                println!(
                    "melt_rec {hop} e {e:.9} acc {acc} class {c} best {best:.9}"
                );
            }
        }
        println!(
            "melt_rec_summary acc {acc_n}/80 marks {n_marks}/80 on_ico {landed_ico}/80 best {best:.9}"
        );
    }
}
