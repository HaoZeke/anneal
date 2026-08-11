//! Recommended leftover hop in a featomic SOAP power spectrum.
//!
//! The calculator is featomic `soap_power_spectrum`. The leftover is
//! the per-centre *high-`l`* spectrum (`l ≥ 5`) minus the
//! species-conditioned mean of the mobile set, restricted to a random
//! defect patch. Low-`l` channels are a core-versus-surface breathing.
//! A global high-`l` leftover is one deterministic ray. The patch is
//! drawn with probability `‖Δp_i‖²` and pulled back through featomic's
//! analytic position gradient. No Marks, fcc, or 421 target.

use std::cell::RefCell;
use std::collections::HashMap;

use featomic::systems::UnitCell;
use featomic::types::Vector3D;
use featomic::{CalculationOptions, Calculator, SimpleSystem, System};
use ndarray::{Array1, ArrayView1};
use rand::Rng;

/// Calculator name passed to [`Calculator::new`].
pub const CALCULATOR: &str = "soap_power_spectrum";
/// Power-spectrum angular channels kept in the leftover.
pub const LMIN: i32 = 5;
/// Leftover RMS below which the hop yields.
const DEFECT: f64 = 1e-4;
const LAMBDA: f64 = 1e-3;

thread_local! {
    static CALCULATORS: RefCell<HashMap<String, Calculator>> =
        RefCell::new(HashMap::new());
}

fn with_calculator<T>(json: &str, f: impl FnOnce(&mut Calculator) -> T) -> T {
    CALCULATORS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let calc = cache.entry(json.to_string()).or_insert_with(|| {
            Calculator::new(CALCULATOR, json.to_string())
                .expect("featomic rejected the hyperparameters")
        });
        f(calc)
    })
}

fn hypers(rcut: f64) -> String {
    let width = (0.5_f64).min(0.2 * rcut);
    let sigma = (0.35_f64).min(0.15 * rcut);
    format!(
        "{{\"cutoff\": {{\"radius\": {rcut}, \
           \"smoothing\": {{\"type\": \"ShiftedCosine\", \"width\": {width}}}}}, \
           \"density\": {{\"type\": \"Gaussian\", \"width\": {sigma}, \
           \"center_atom_weight\": 0.0}}, \
           \"basis\": {{\"type\": \"TensorProduct\", \"max_angular\": 6, \
           \"radial\": {{\"type\": \"Gto\", \"max_radial\": 3}}}}}}"
    )
}

fn system_of(x: ArrayView1<f64>, species: Option<&[u32]>) -> Vec<Box<dyn System>> {
    let n = x.len() / 3;
    let mut sys = SimpleSystem::new(UnitCell::infinite());
    for i in 0..n {
        let z = species.and_then(|s| s.get(i).copied()).unwrap_or(1) as i32;
        sys.add_atom(
            z,
            Vector3D::new(x[3 * i], x[3 * i + 1], x[3 * i + 2]),
        );
    }
    vec![Box::new(sys) as Box<dyn System>]
}

/// Per-centre SOAP rows and the Jacobian of the stacked leftover.
struct Spectrum {
    /// `n_at * n_feat` leftover `p_i − μ_{z(i)}` (frozen centres zero).
    leftover: Array1<f64>,
    /// Rows of leftover, columns of coordinates.
    jacobian: ndarray::Array2<f64>,
    /// Species-conditioned mean spectrum. This is the packing label:
    /// leftover RMS is almost the same on every closed shell.
    cloud_mean: Array1<f64>,
    n_at: usize,
    n_feat: usize,
}

fn spectrum(
    x: ArrayView1<f64>,
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
) -> Spectrum {
    let n = x.len() / 3;
    let json = hypers(rcut);
    let mut systems = system_of(x, species);
    let options = CalculationOptions {
        gradients: &["positions"],
        ..Default::default()
    };
    let out = with_calculator(&json, |calc| {
        calc.compute(&mut systems, options)
            .expect("featomic soap_power_spectrum failed")
    });

    // Concatenate blocks. Each block is one (center_type, neighbor types) key.
    // Only `l ≥ LMIN` columns enter the leftover: the radial/low-l block is
    // a breathing mode on a closed packing shell.
    let mut rows: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut j_raw: Vec<Vec<f64>> = vec![Vec::new(); n];
    let m = 3 * n;
    for idx in 0..out.keys().count() {
        let block = out.block_by_id(idx);
        let array = block.values().to_array();
        let shape = array.shape().to_vec();
        let flat: Vec<f64> = array.iter().copied().collect();
        let n_s = shape[0];
        let n_f = shape[shape.len() - 1];
        let samples = block.samples();
        let sample_names = samples.names();
        let atom_col = sample_names
            .iter()
            .position(|&s| s == "atom")
            .unwrap_or(0);
        let props = block.properties();
        let names = props.names();
        let l_idx = names.iter().position(|&s| s == "l");
        let keep: Vec<usize> = (0..n_f)
            .filter(|&f| match l_idx {
                Some(li) if f < props.count() => props[f][li].i32() >= LMIN,
                None => true,
                _ => false,
            })
            .collect();
        let n_keep = keep.len();
        if n_keep == 0 {
            continue;
        }
        for row in 0..n_s {
            let centre = samples[row][atom_col].usize();
            if centre >= n {
                continue;
            }
            if rows[centre].is_empty() {
                rows[centre] = vec![0.0; n_keep];
                j_raw[centre] = vec![0.0; n_keep * m];
            }
            let nf = n_keep.min(rows[centre].len());
            for (tf, &f) in keep.iter().take(nf).enumerate() {
                rows[centre][tf] += flat[row * n_f + f];
            }
        }
        if let Some(g) = block.gradient("positions") {
            let garray = g.values().to_array();
            let gshape = garray.shape().to_vec();
            let gflat: Vec<f64> = garray.iter().copied().collect();
            let (g_rows, g_dirs, g_f) = (gshape[0], gshape[1], gshape[2]);
            let gsamp = g.samples();
            for grow in 0..g_rows {
                // ("sample", "system", "atom")
                let sample = gsamp[grow][0].usize();
                let atom = gsamp[grow][2].usize();
                if sample >= n_s || atom >= n {
                    continue;
                }
                let centre = samples[sample][atom_col].usize();
                if centre >= n || j_raw[centre].is_empty() {
                    continue;
                }
                let nf = n_keep.min(rows[centre].len());
                for d in 0..g_dirs {
                    for (tf, &f) in keep.iter().take(nf).enumerate() {
                        if f >= g_f {
                            continue;
                        }
                        let v = gflat[(grow * g_dirs + d) * g_f + f];
                        j_raw[centre][tf * m + 3 * atom + d] += v;
                    }
                }
            }
        }
    }
    let n_feat = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    for r in &mut rows {
        r.resize(n_feat, 0.0);
    }
    for j in &mut j_raw {
        j.resize(n_feat * m, 0.0);
    }

    let keep = mobile_mask(n, mobile);
    let zi = |i: usize| species.and_then(|z| z.get(i).copied()).unwrap_or(0);
    let mut labels: Vec<u32> = Vec::new();
    for i in 0..n {
        if keep[i] && !labels.contains(&zi(i)) {
            labels.push(zi(i));
        }
    }
    let nlab = labels.len().max(1);
    let mut mu = vec![vec![0.0; n_feat]; nlab];
    let mut cnt = vec![0.0; nlab];
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        let k = labels.iter().position(|&z| z == zi(i)).unwrap_or(0);
        cnt[k] += 1.0;
        for f in 0..n_feat {
            mu[k][f] += rows[i][f];
        }
    }
    for k in 0..nlab {
        if cnt[k] > 0.0 {
            for f in 0..n_feat {
                mu[k][f] /= cnt[k];
            }
        }
    }

    let mut leftover = Array1::zeros(n * n_feat);
    let mut jac = ndarray::Array2::zeros((n * n_feat, m));
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        let k = labels.iter().position(|&z| z == zi(i)).unwrap_or(0);
        for f in 0..n_feat {
            leftover[i * n_feat + f] = rows[i][f] - mu[k][f];
            for atom in 0..n {
                if !keep[atom] {
                    continue;
                }
                for d in 0..3 {
                    let c = 3 * atom + d;
                    jac[[i * n_feat + f, c]] = j_raw[i][f * m + c];
                }
            }
        }
    }
    let mut cloud_mean = Array1::zeros(nlab * n_feat);
    for k in 0..nlab {
        for f in 0..n_feat {
            cloud_mean[k * n_feat + f] = mu[k][f];
        }
    }
    Spectrum {
        leftover,
        jacobian: jac,
        cloud_mean,
        n_at: n,
        n_feat,
    }
}

fn mobile_mask(n: usize, mobile: Option<&[usize]>) -> Vec<bool> {
    match mobile {
        None => vec![true; n],
        Some(idx) => {
            let mut k = vec![false; n];
            for &i in idx {
                if i < n {
                    k[i] = true;
                }
            }
            k
        }
    }
}

/// Per-atom leftover RMS of the high-`l` power spectrum.
pub fn atom_leftover_rms(
    x: ArrayView1<f64>,
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
) -> Array1<f64> {
    let s = spectrum(x, rcut, species, mobile);
    let mut w = Array1::zeros(s.n_at);
    let nf = s.n_feat.max(1) as f64;
    for i in 0..s.n_at {
        let mut q = 0.0;
        for f in 0..s.n_feat {
            let v = s.leftover[i * s.n_feat + f];
            q += v * v;
        }
        w[i] = (q / nf).sqrt();
    }
    w
}

fn unit(v: &Array1<f64>) -> Array1<f64> {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
    v / n
}

/// Species-conditioned mean high-`l` SOAP. The packing label.
pub fn soap_cloud_mean(
    x: ArrayView1<f64>,
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
) -> Array1<f64> {
    unit(&spectrum(x, rcut, species, mobile).cloud_mean)
}

/// Mean-spectrum morphology the bank acquisition model fits.
pub fn soap_morphology(
    x: ArrayView1<f64>,
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
) -> Array1<f64> {
    soap_cloud_mean(x, rcut, species, mobile)
}

/// Distance between two packings in mean SOAP, not leftover RMS.
///
/// Leftover RMS is a core-versus-surface magnitude. It is almost the
/// same on every closed shell, so Dcut collapsed to 10^{-3} and the
/// bank treated every Mackay isomer as a new basin. The mean spectrum
/// is the packing.
pub fn soap_bank_distance(
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
) -> f64 {
    let ua = soap_cloud_mean(a, rcut, species, mobile);
    let ub = soap_cloud_mean(b, rcut, species, mobile);
    if ua.len() != ub.len() || ua.is_empty() {
        return f64::INFINITY;
    }
    ua.iter()
        .zip(ub.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// RMS of the featomic leftover on the mobile, species-conditioned cloud.
pub fn leftover_rms(
    x: ArrayView1<f64>,
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
) -> f64 {
    let s = spectrum(x, rcut, species, mobile);
    if s.leftover.is_empty() {
        return 0.0;
    }
    let n = s.leftover.len() as f64;
    (s.leftover.iter().map(|v| v * v).sum::<f64>() / n).sqrt()
}

fn scale_to_cap(x: ArrayView1<f64>, mut dr: Array1<f64>, rmsd: f64) -> Array1<f64> {
    let n = (x.len() / 3).max(1) as f64;
    let cap = rmsd.max(1e-6);
    let cur = (dr.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    if cur < 1e-15 {
        return x.to_owned();
    }
    dr *= cap / cur;
    &x.to_owned() + &dr
}

fn pin_frozen(x: ArrayView1<f64>, mut y: Array1<f64>, mobile: Option<&[usize]>) -> Array1<f64> {
    let n = x.len() / 3;
    let keep = mobile_mask(n, mobile);
    for i in 0..n {
        if !keep[i] {
            for d in 0..3 {
                y[3 * i + d] = x[3 * i + d];
            }
        }
    }
    y
}

fn tikhonov(j: &ndarray::Array2<f64>, dp: ArrayView1<f64>, lambda: f64) -> Array1<f64> {
    let n = j.ncols();
    let mut a = vec![0.0; n * n];
    let mut b = vec![0.0; n];
    for c in 0..n {
        for r in 0..j.nrows() {
            b[c] += j[[r, c]] * dp[r];
            for d in 0..n {
                a[c * n + d] += j[[r, c]] * j[[r, d]];
            }
        }
        a[c * n + c] += lambda;
    }
    // Dense Cholesky.
    for i in 0..n {
        for k in 0..i {
            a[i * n + i] -= a[i * n + k] * a[i * n + k];
        }
        if a[i * n + i] <= 0.0 {
            a[i * n + i] = lambda;
        }
        a[i * n + i] = a[i * n + i].sqrt();
        for jcol in (i + 1)..n {
            for k in 0..i {
                a[jcol * n + i] -= a[jcol * n + k] * a[i * n + k];
            }
            a[jcol * n + i] /= a[i * n + i];
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * y[k];
        }
        y[i] = s / a[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= a[k * n + i] * x[k];
        }
        x[i] = s / a[i * n + i];
    }
    Array1::from(x)
}

/// Restrict leftover and Jacobian to a random defect patch.
///
/// Atom `a` is drawn with probability proportional to `‖Δp_a‖²`. The
/// patch is `a` and every mobile neighbour inside `rcut`. A global
/// leftover on a closed shell is one breathing ray; a patch is one
/// local reconstruction, and the draw is a different site each hop.
fn focus_patch<R: Rng + ?Sized>(s: &mut Spectrum, x: ArrayView1<f64>, rcut: f64, rng: &mut R) {
    let n = s.n_at;
    let nf = s.n_feat;
    if n == 0 || nf == 0 {
        return;
    }
    let mut w = vec![0.0; n];
    for i in 0..n {
        let mut q = 0.0;
        for f in 0..nf {
            let v = s.leftover[i * nf + f];
            q += v * v;
        }
        w[i] = q;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| w[j].total_cmp(&w[i]));
    if w[order[0]] < 1e-18 {
        return;
    }
    // A unique core leftover is a breathing mode. Drop it and hop a
    // surface site so two seeds reconstruct different patches.
    let mut start = 0usize;
    if n >= 2 && w[order[0]] > 3.0 * w[order[1]] {
        start = 1;
    }
    let pool = (n / 8).max(6).min(n - start).max(1);
    let pick = rng.random_range(0..pool);
    let a = order[start + pick];
    let r2 = rcut * rcut;
    let mut patch = vec![false; n];
    for j in 0..n {
        let mut d2 = 0.0;
        for k in 0..3 {
            let d = x[3 * j + k] - x[3 * a + k];
            d2 += d * d;
        }
        patch[j] = d2 <= r2;
    }
    let m = 3 * n;
    for i in 0..n {
        if !patch[i] {
            for f in 0..nf {
                s.leftover[i * nf + f] = 0.0;
            }
        }
        for f in 0..nf {
            for j in 0..n {
                if patch[j] {
                    continue;
                }
                for d in 0..3 {
                    s.jacobian[[i * nf + f, 3 * j + d]] = 0.0;
                }
            }
            let _ = m;
        }
    }
}

/// Leftover hop through featomic `soap_power_spectrum`.
pub fn step_away_featomic<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    rmsd: f64,
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
    rng: &mut R,
) -> Array1<f64> {
    let mut s = spectrum(x, rcut, species, mobile);
    let nnu = (s.n_at * s.n_feat).max(1) as f64;
    let rms = (s.leftover.iter().map(|v| v * v).sum::<f64>() / nnu).sqrt();
    if rms < DEFECT {
        return x.to_owned();
    }
    focus_patch(&mut s, x, rcut, rng);
    let dr = tikhonov(&s.jacobian, s.leftover.view(), LAMBDA);
    // A patch hop at the caller cap is a local reconstruction.
    // The old global 0.44 floor was a whole-cluster distortion.
    pin_frozen(x, scale_to_cap(x, dr, rmsd), mobile)
}

/// Same leftover direction as [`step_away_featomic`], at `rmsd` with no floor.
pub fn step_away_featomic_at<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    rmsd: f64,
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
    rng: &mut R,
) -> Array1<f64> {
    let mut s = spectrum(x, rcut, species, mobile);
    let nnu = (s.n_at * s.n_feat).max(1) as f64;
    let rms = (s.leftover.iter().map(|v| v * v).sum::<f64>() / nnu).sqrt();
    if rms < DEFECT {
        return x.to_owned();
    }
    focus_patch(&mut s, x, rcut, rng);
    let dr = tikhonov(&s.jacobian, s.leftover.view(), LAMBDA);
    pin_frozen(x, scale_to_cap(x, dr, rmsd), mobile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

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

    #[test]
    fn calculator_is_soap_power_spectrum() {
        assert_eq!(CALCULATOR, "soap_power_spectrum");
        assert_eq!(LMIN, 5);
    }

    #[test]
    fn leftover_on_ico13_is_nonzero() {
        let x = ico13();
        let rms = leftover_rms(x.view(), 3.5, None, None);
        assert!(
            rms > DEFECT,
            "featomic leftover on ico13 is {rms}, want a packing defect"
        );
        let d = soap_bank_distance(x.view(), x.view(), 3.5, None, None);
        assert!(d < 1e-12, "SOAP bank distance of a structure to itself is {d}");
        let morph = soap_morphology(x.view(), 3.5, None, None);
        assert!(!morph.is_empty());
        assert!(morph.iter().all(|v| v.is_finite()));
        // A cuboctahedral shell is a different packing. Leftover RMS
        // magnitude does not see that; the mean spectrum must.
        let p = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let s = 1.0 / (1.0 + p * p).sqrt() * 2.0_f64.powf(1.0 / 6.0);
        let fcc = {
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
        };
        let d_pack = soap_bank_distance(x.view(), fcc.view(), 3.5, None, None);
        assert!(
            d_pack > 0.05,
            "mean SOAP must separate ico13 from cuboct13, got {d_pack}"
        );
    }

    #[test]
    fn hop_moves_ico13() {
        let x = ico13();
        let mut rng = StdRng::seed_from_u64(1);
        let y = step_away_featomic(x.view(), 0.35, 3.5, None, None, &mut rng);
        let n = 13.0;
        let mut s = 0.0;
        for i in 0..x.len() {
            let d = y[i] - x[i];
            s += d * d;
        }
        let rms = (s / n).sqrt();
        assert!(rms > 1e-8, "featomic hop was identity, rms {rms}");
    }

    #[test]
    fn two_seeds_leave_on_different_patches() {
        let x = ico13();
        let mut hops = Vec::new();
        for seed in 1u64..=8 {
            let mut rng = StdRng::seed_from_u64(seed);
            // ico13 fits inside the production 3.5 cutoff, so every patch
            // is the whole cluster. A first-shell cutoff splits vertices.
            hops.push(step_away_featomic(x.view(), 0.35, 1.5, None, None, &mut rng));
        }
        let mut max_rms = 0.0_f64;
        for i in 0..hops.len() {
            for j in (i + 1)..hops.len() {
                let mut s = 0.0;
                for k in 0..x.len() {
                    let d = hops[i][k] - hops[j][k];
                    s += d * d;
                }
                max_rms = max_rms.max((s / 13.0).sqrt());
            }
        }
        assert!(
            max_rms > 1e-8,
            "eight leftover seeds produced one hop, max rms {max_rms}"
        );
    }

    #[test]
    fn leftover_on_water_is_nonzero() {
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0, 3.10, 0.15, 0.08, 3.98, 0.40, -0.05,
            2.82, 1.05, 0.18,
        ]);
        let z = [8u32, 1, 1, 8, 1, 1];
        let rms = leftover_rms(x.view(), 4.0, Some(&z), None);
        assert!(rms > DEFECT, "featomic leftover on water dimer is {rms}");
    }

    #[test]
    fn frozen_atoms_do_not_move() {
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0, 3.10, 0.15, 0.08, 3.98, 0.40, -0.05,
            2.82, 1.05, 0.18,
        ]);
        let z = [8u32, 1, 1, 8, 1, 1];
        let mobile = [0usize, 1, 2];
        let mut rng = StdRng::seed_from_u64(2);
        let y = step_away_featomic(x.view(), 0.35, 4.0, Some(&z), Some(&mobile), &mut rng);
        for i in 9..x.len() {
            assert_eq!(y[i], x[i], "frozen coordinate {i} moved");
        }
    }
}
