//! Recommended hop in a featomic SOAP power spectrum.
//!
//! The calculator is featomic `soap_power_spectrum`. High-`l`
//! (`l ≥ 5`) leftover `p_i − μ` is a defect patch when some centres
//! stick out of the species mean. On a closed shell that leftover is
//! a core-versus-surface breath: the quench returns the same packing.
//! The packing label is the unit species mean `μ` (the bank Dcut).
//! When leftover is a shell mode the hop is a kick of `μ` along a
//! random direction orthogonal to the occupied mean, pulled back
//! through `∂μ/∂x`. No Marks, fcc, or 421 target.

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
/// `Dcut` when every seed is the same packing.
///
/// High-`l` mean SOAP of LJ75 Mackay vs Marks is 0.163; of a structure
/// to itself is 0. The bank used to fall back to `merge_radius` (0.7),
/// which is a length, and 0.163 < 0.7 so Marks was a duplicate of
/// Mackay. This floor sits below the packing gap and above numerical
/// zero, so a one-funnel seed bank still admits the other funnel.
pub const SOAP_DCUT_FALLBACK: f64 = 0.05;
/// Merge radius on unit mean SOAP: isomers of one packing, not Marks.
///
/// LJ75 ico-isomer high-`l` sits below `0.4` of the ico-Marks gap
/// (0.163). This floor is above that isomer band and below the
/// packing gap, so a well-tempered deposit fills the occupied
/// superbasin and not the other funnel.
pub const SOAP_PACK_MERGE: f64 = 0.10;

/// Cloud-mean distance a hole step walks before it stops.
///
/// Measured on the LJ75 icosahedral fixture, twelve seeds, sweeping
/// this from one to five merge radii: the walk reaches 0.33 to 0.55 in
/// the cloud mean and the quench that follows returns to the same
/// DECAF family every time, at every distance. Distance is not what
/// separates packings here, so there is nothing to buy by walking
/// further, and the extra Cartesian displacement is not free.
/// Occupancy Leave uses this walk as the archive-hole start.
pub const SOAP_PACK_ESCAPE: f64 = SOAP_PACK_MERGE;
/// Packing-class gap on unit mean SOAP. Ico isomers sit under
/// [`SOAP_PACK_MERGE`]. Mackay vs Marks is 0.163. A bank sample
/// closer than this is the same funnel, not a class another chain
/// opened, and adopting it restamps the walk onto ico.
pub const SOAP_PACK_GAP: f64 = 0.15;
thread_local! {
    /// Shared-bank packing means. The hop steps in the SOAP null
    /// space of this archive (MAP-Elites coverage + null-space
    /// motion): directions the bank already spans are known.
    static PACK_ARCHIVE: RefCell<Vec<Array1<f64>>> = const { RefCell::new(Vec::new()) };
}

/// Replace the packing archive used by [`step_away_featomic`].
pub fn set_packing_archive(wells: Vec<Array1<f64>>) {
    PACK_ARCHIVE.with(|a| *a.borrow_mut() = wells);
}

/// Occupied packing means last stored by [`set_packing_archive`].
pub fn packing_archive() -> Vec<Array1<f64>> {
    PACK_ARCHIVE.with(|a| a.borrow().clone())
}

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
        sys.add_atom(z, Vector3D::new(x[3 * i], x[3 * i + 1], x[3 * i + 2]));
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
    /// Jacobian of [`cloud_mean`] with respect to coordinates.
    mean_jac: ndarray::Array2<f64>,
    n_at: usize,
    n_feat: usize,
}

fn spectrum(
    x: ArrayView1<f64>,
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
) -> Spectrum {
    spectrum_keep(x, rcut, species, mobile, LMIN)
}

fn spectrum_keep(
    x: ArrayView1<f64>,
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
    lmin: i32,
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
        let atom_col = sample_names.iter().position(|&s| s == "atom").unwrap_or(0);
        let props = block.properties();
        let names = props.names();
        let l_idx = names.iter().position(|&s| s == "l");
        let keep: Vec<usize> = (0..n_f)
            .filter(|&f| match l_idx {
                Some(li) if f < props.count() => props[f][li].i32() >= lmin,
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
    let mut mean_jac = ndarray::Array2::zeros((nlab * n_feat, m));
    for k in 0..nlab {
        for f in 0..n_feat {
            cloud_mean[k * n_feat + f] = mu[k][f];
            if cnt[k] <= 0.0 {
                continue;
            }
            for i in 0..n {
                if !keep[i] {
                    continue;
                }
                if labels.iter().position(|&z| z == zi(i)).unwrap_or(0) != k {
                    continue;
                }
                for c in 0..m {
                    mean_jac[[k * n_feat + f, c]] += j_raw[i][f * m + c] / cnt[k];
                }
            }
        }
    }
    Spectrum {
        leftover,
        jacobian: jac,
        cloud_mean,
        mean_jac,
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

fn soap_l2_pack(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
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
    soap_bank_distance_lmin(a, b, rcut, species, mobile, LMIN)
}

/// Mean-SOAP L2 with a chosen angular floor. `lmin = 0` is the full
/// spectrum; production leftover hops keep [`LMIN`].
pub fn soap_bank_distance_lmin(
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
    lmin: i32,
) -> f64 {
    let ua = unit(&spectrum_keep(a, rcut, species, mobile, lmin).cloud_mean);
    let ub = unit(&spectrum_keep(b, rcut, species, mobile, lmin).cloud_mean);
    if ua.len() != ub.len() || ua.is_empty() {
        return f64::INFINITY;
    }
    ua.iter()
        .zip(ub.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Sorted per-atom leftover-RMS L2. This is the metric that collapsed.
///
/// Closed shells share a core-versus-surface leftover magnitude, so
/// Mackay and Marks sit almost on top of each other. Kept so a test
/// can name the failure; the bank must not call this.
pub fn leftover_profile_distance(
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
) -> f64 {
    let mut wa: Vec<f64> = atom_leftover_rms(a, rcut, species, mobile)
        .iter()
        .copied()
        .collect();
    let mut wb: Vec<f64> = atom_leftover_rms(b, rcut, species, mobile)
        .iter()
        .copied()
        .collect();
    if wa.len() != wb.len() || wa.is_empty() {
        return f64::INFINITY;
    }
    wa.sort_by(|x, y| y.total_cmp(x));
    wb.sort_by(|x, y| y.total_cmp(x));
    wa.iter()
        .zip(wb.iter())
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

/// True when leftover is a shell breath, not a local defect.
///
/// The largest per-atom leftover is then only a few times the median,
/// so `p_i − μ` is the same core-versus-surface mode on every closed
/// packing. Pulling that mode back does not change `μ`.
fn shell_leftover(s: &Spectrum) -> bool {
    let n = s.n_at;
    let nf = s.n_feat;
    if n == 0 || nf == 0 {
        return true;
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
    w.sort_by(|a, b| a.total_cmp(b));
    let med = w[n / 2].max(1e-18);
    w[n - 1] < 4.0 * med
}

/// Orthonormalise `basis` in place (modified Gram-Schmidt), drop zeros.
fn orthonormal(basis: &mut Vec<Array1<f64>>) {
    let mut out: Vec<Array1<f64>> = Vec::new();
    for v in basis.drain(..) {
        let mut w = v;
        for b in &out {
            if b.len() != w.len() {
                continue;
            }
            let p = w.iter().zip(b.iter()).map(|(a, c)| a * c).sum::<f64>();
            for (x, c) in w.iter_mut().zip(b.iter()) {
                *x -= p * *c;
            }
        }
        let n = w.iter().map(|z| z * z).sum::<f64>().sqrt();
        if n > 1e-12 {
            for x in w.iter_mut() {
                *x /= n;
            }
            out.push(w);
        }
        if out.len() == 8 {
            break;
        }
    }
    *basis = out;
}

/// A packing on the unit sphere that is far from every well and from `mu`.
///
/// The shared bank is a point cloud of occupied means. The next start
/// is a hole in that cloud, not a random Cartesian shove and not a
/// louder version of the last kick. Forty-eight unit samples, kept in
/// the open hemisphere away from the well centroid, scored by the
/// nearest well.
fn hole_on_sphere<R: Rng + ?Sized>(
    mu: &Array1<f64>,
    wells: &[Array1<f64>],
    rng: &mut R,
) -> Array1<f64> {
    let mut centroid = Array1::<f64>::zeros(mu.len());
    let mut n = 0.0;
    for w in wells {
        if w.len() != mu.len() || w.iter().all(|v| *v == 0.0) {
            continue;
        }
        let u = unit(w);
        for (c, x) in centroid.iter_mut().zip(u.iter()) {
            *c += *x;
        }
        n += 1.0;
    }
    if n > 0.0 {
        centroid /= n;
        centroid = unit(&centroid);
    } else {
        centroid = mu.clone();
    }
    let mut best = mu.clone();
    let mut best_score = -1.0_f64;
    for _ in 0..48 {
        let mut u = Array1::<f64>::zeros(mu.len());
        for v in u.iter_mut() {
            let a: f64 = rng.random();
            let b: f64 = rng.random();
            let r = (-2.0 * a.max(1e-15).ln()).sqrt();
            *v = r * (2.0 * std::f64::consts::PI * b).cos();
        }
        let nn = u.iter().map(|z| z * z).sum::<f64>().sqrt();
        if nn < 1e-15 {
            continue;
        }
        u /= nn;
        let toward_cloud: f64 = u.iter().zip(centroid.iter()).map(|(a, c)| a * c).sum();
        if toward_cloud > 0.0 {
            for v in u.iter_mut() {
                *v = -*v;
            }
        }
        let mut score = soap_l2_pack(u.view(), mu.view());
        for w in wells {
            if w.len() == u.len() {
                score = score.min(soap_l2_pack(u.view(), w.view()));
            }
        }
        if score > best_score {
            best_score = score;
            best = u;
        }
    }
    best
}

/// Walk coordinates so the packing mean moves toward a hole in the
/// shared SOAP archive.
///
/// `J = ∂μ/∂x` is already built for the leftover hop. The Cartesian
/// 0.35 cap keeps `μ` inside one packing and the quench is a projector
/// back onto that packing. The step size here is in SOAP: each
/// microstep asks for `soap_step` of mean-spectrum change, with only a
/// loose Cartesian guard. Repeat until `μ` sits outside every well.
pub fn step_into_hole<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    wells: &[Array1<f64>],
    soap_step: f64,
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
    rng: &mut R,
) -> Array1<f64> {
    step_into_hole_escaping(
        x,
        wells,
        soap_step,
        SOAP_PACK_ESCAPE,
        rcut,
        species,
        mobile,
        rng,
    )
}

/// [`step_into_hole`] with the escape distance supplied, so the
/// distance a Leave has to cover can be measured rather than assumed.
#[allow(clippy::too_many_arguments)]
pub fn step_into_hole_escaping<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    wells: &[Array1<f64>],
    soap_step: f64,
    escape: f64,
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
    rng: &mut R,
) -> Array1<f64> {
    let mut cur = x.to_owned();
    let step = soap_step.max(SOAP_PACK_MERGE);
    let cart_guard = 2.0;
    let target = {
        let s0 = spectrum(cur.view(), rcut, species, mobile);
        let mu = unit(&s0.cloud_mean);
        if mu.is_empty() {
            return cur;
        }
        hole_on_sphere(&mu, wells, rng)
    };
    for _ in 0..8 {
        let s = spectrum(cur.view(), rcut, species, mobile);
        let mu = unit(&s.cloud_mean);
        if mu.is_empty() || s.mean_jac.nrows() != mu.len() {
            break;
        }
        // Stop on the separation, not on sameness. Breaking as soon as
        // the point is no longer within the merge radius ends the walk
        // on the boundary of the well it is leaving, which is exactly
        // where a quench returns from.
        let nearest = wells
            .iter()
            .filter(|w| w.len() == mu.len())
            .map(|w| soap_l2_pack(mu.view(), w.view()))
            .fold(f64::INFINITY, f64::min);
        if !wells.is_empty() && nearest >= escape {
            break;
        }
        let mut dp = Array1::<f64>::zeros(mu.len());
        for i in 0..mu.len() {
            dp[i] = target[i] - mu[i];
        }
        let dn = dp.iter().map(|z| z * z).sum::<f64>().sqrt();
        if dn < 1e-12 {
            break;
        }
        dp *= step / dn;
        let mut dr = tikhonov(&s.mean_jac, dp.view(), LAMBDA);
        let n = (cur.len() / 3).max(1) as f64;
        let cart = (dr.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
        if cart > cart_guard && cart > 1e-15 {
            dr *= cart_guard / cart;
        }
        cur = pin_frozen(cur.view(), &cur + &dr, mobile);
    }
    cur
}

/// Surplus Hyperband start: step the current packing mean into a SOAP
/// hole of the occupied cloud. Empty wells yield `None` so the caller
/// may draw a random cluster. Not a parent clone and not a named
/// morphology.
pub fn surplus_reseed<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    wells: &[Array1<f64>],
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
    rng: &mut R,
) -> Option<Array1<f64>> {
    if wells.is_empty() {
        return None;
    }
    Some(step_into_hole(
        x,
        wells,
        SOAP_PACK_MERGE,
        rcut,
        species,
        mobile,
        rng,
    ))
}

/// Whether the structure's leftover-SOAP packing mean sits in a stored well.
pub fn in_stored_well(
    x: ArrayView1<f64>,
    wells: &[Array1<f64>],
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
) -> bool {
    let mu = unit(&soap_cloud_mean(x, rcut, species, mobile));
    !mu.is_empty()
        && wells
            .iter()
            .any(|w| w.len() == mu.len() && soap_l2_pack(mu.view(), w.view()) <= SOAP_PACK_MERGE)
}

/// Leave an occupied packing: hole, quench, keep the quench only when
/// DECAF says it is a different family. Leftover-SOAP off-well is not
/// a family change. Failed family attempts kick mean SOAP off the
/// occupied packing. A dead kick stays at the origin so
/// [`crate::catalog::occupancy_leave_adopt`] Refuse does not install a
/// same-family hole.
pub fn leave_occupied_packing<R, Q>(
    x: ArrayView1<f64>,
    wells: &[Array1<f64>],
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
    mut quench: Q,
    rng: &mut R,
) -> Array1<f64>
where
    R: Rng + ?Sized,
    Q: FnMut(ArrayView1<f64>) -> Array1<f64>,
{
    if wells.is_empty() {
        return x.to_owned();
    }
    let mut cur = x.to_owned();
    for attempt in 0..6 {
        let scale = SOAP_PACK_MERGE * (1.0 + 0.5 * f64::from(attempt));
        let y = step_into_hole(cur.view(), wells, scale, rcut, species, mobile, rng);
        let q = quench(y.view());
        let origin = x.as_slice().unwrap_or(&[]);
        let trial = q.as_slice().unwrap_or(&[]);
        if crate::catalog::different_decaf_family(origin, trial) {
            return q;
        }
        cur = q;
    }
    // Same-family leftover hole is not a Leave. Kick off the occupied
    // packing; packing_kick returns x when the kick cannot move.
    let s = spectrum(x, rcut, species, mobile);
    packing_kick(x, &s, 0.35, mobile, rng)
}

/// Kick mean SOAP in the null space of the occupied packing *and*
/// of the shared-bank archive (known packings).
fn packing_kick<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    s: &Spectrum,
    rmsd: f64,
    mobile: Option<&[usize]>,
    rng: &mut R,
) -> Array1<f64> {
    let mu = unit(&s.cloud_mean);
    if mu.is_empty() || s.mean_jac.nrows() != mu.len() {
        return x.to_owned();
    }
    let mut basis = vec![mu.clone()];
    PACK_ARCHIVE.with(|a| {
        let merge = SOAP_PACK_MERGE;
        for w in a.borrow().iter() {
            if w.len() != mu.len() {
                continue;
            }
            // One vector a packing. Thirty ico wells would otherwise
            // fill the Gram-Schmidt budget with the same funnel.
            if basis
                .iter()
                .any(|b| soap_l2_pack(b.view(), w.view()) <= merge)
            {
                continue;
            }
            basis.push(w.clone());
        }
    });
    orthonormal(&mut basis);
    let mut u = Array1::<f64>::zeros(mu.len());
    for v in u.iter_mut() {
        let a: f64 = rng.random();
        let b: f64 = rng.random();
        let r = (-2.0 * a.max(1e-15).ln()).sqrt();
        *v = r * (2.0 * std::f64::consts::PI * b).cos();
    }
    for b in &basis {
        let p = u.iter().zip(b.iter()).map(|(a, c)| a * c).sum::<f64>();
        for (v, c) in u.iter_mut().zip(b.iter()) {
            *v -= p * *c;
        }
    }
    let nrm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    if nrm < 1e-15 {
        return x.to_owned();
    }
    u /= nrm;
    let dr = tikhonov(&s.mean_jac, u.view(), LAMBDA);
    pin_frozen(x, scale_to_cap(x, dr, rmsd), mobile)
}

/// SOAP hop through featomic `soap_power_spectrum`.
///
/// A local leftover patch when some centres stick out of the species
/// mean. A packing-mean kick when leftover is a closed-shell breath
/// or numerically gone.
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
    if rms < DEFECT || shell_leftover(&s) {
        let archive = PACK_ARCHIVE.with(|a| a.borrow().clone());
        if !archive.is_empty() {
            return step_into_hole(x, &archive, SOAP_PACK_MERGE, rcut, species, mobile, rng);
        }
        // A uniform leftover on a free cluster means a closed packing
        // with nothing local to follow, and the kick is the escape from
        // it. On a masked structure it means something else: an
        // adsorbate of a few atoms is all surface and all alike, so the
        // veto fires whatever the geometry does, and kicking on it
        // spends a quench and a move slot on a proposal the residual
        // never asked for. Decline instead, and let the arm the
        // allocator would otherwise have skipped take the hop.
        if mobile.is_some_and(|set| set.len() < x.len() / 3) {
            return x.to_owned();
        }
        return packing_kick(x, &s, rmsd, mobile, rng);
    }
    focus_patch(&mut s, x, rcut, rng);
    let dr = tikhonov(&s.jacobian, s.leftover.view(), LAMBDA);
    pin_frozen(x, scale_to_cap(x, dr, rmsd), mobile)
}

/// Same hop as [`step_away_featomic`], at `rmsd` with no extra floor.
pub fn step_away_featomic_at<R: Rng + ?Sized>(
    x: ArrayView1<f64>,
    rmsd: f64,
    rcut: f64,
    species: Option<&[u32]>,
    mobile: Option<&[usize]>,
    rng: &mut R,
) -> Array1<f64> {
    step_away_featomic(x, rmsd, rcut, species, mobile, rng)
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
        assert!(
            d < 1e-12,
            "SOAP bank distance of a structure to itself is {d}"
        );
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
    fn archive_null_still_moves_ico13() {
        let x = ico13();
        let mu = unit(&spectrum(x.view(), 3.5, None, None).cloud_mean);
        set_packing_archive(vec![mu]);
        let mut rng = StdRng::seed_from_u64(2);
        let y = step_away_featomic(x.view(), 0.35, 3.5, None, None, &mut rng);
        set_packing_archive(Vec::new());
        let mut s = 0.0;
        for i in 0..x.len() {
            let d = y[i] - x[i];
            s += d * d;
        }
        assert!((s / 13.0).sqrt() > 1e-8, "archive-null hop was identity");
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
            hops.push(step_away_featomic(
                x.view(),
                0.35,
                1.5,
                None,
                None,
                &mut rng,
            ));
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

    fn load_xyz(text: &str) -> Array1<f64> {
        let mut vals = Vec::new();
        for (i, line) in text.lines().enumerate() {
            if i < 2 {
                continue;
            }
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            let parts: Vec<&str> = t.split_whitespace().collect();
            let start = if parts.len() >= 4 { 1 } else { 0 };
            for k in 0..3 {
                vals.push(parts[start + k].parse::<f64>().expect(t));
            }
        }
        Array1::from(vals)
    }

    fn rotate_z(x: &Array1<f64>) -> Array1<f64> {
        let mut y = x.clone();
        for i in 0..x.len() / 3 {
            let xx = x[3 * i];
            let yy = x[3 * i + 1];
            y[3 * i] = -yy;
            y[3 * i + 1] = xx;
        }
        y
    }

    /// The bank distance has to separate the paper funnels. Leftover RMS
    /// does not: it is a core-versus-surface magnitude shared by every
    /// closed shell. Mean SOAP has to, or Dcut is still a one-funnel
    /// threshold.
    #[test]
    fn paper_funnels_mean_soap_not_leftover_rms() {
        let ico38 = load_xyz(include_str!("../tests/fixtures/lj38_ico.xyz"));
        let fcc38 = load_xyz(include_str!("../tests/fixtures/lj38_fcc.xyz"));
        let ico75 = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
        let marks = load_xyz(include_str!("../tests/fixtures/lj75_marks.xyz"));
        assert_eq!(ico38.len(), 38 * 3);
        assert_eq!(fcc38.len(), 38 * 3);
        assert_eq!(ico75.len(), 75 * 3);
        assert_eq!(marks.len(), 75 * 3);

        let pairs = [
            ("LJ38 ico-fcc", ico38.view(), fcc38.view()),
            ("LJ75 ico-Marks", ico75.view(), marks.view()),
        ];
        for (label, a, b) in pairs {
            let d_left = leftover_profile_distance(a, b, 3.5, None, None);
            let d_high = soap_bank_distance(a, b, 3.5, None, None);
            let d_full = soap_bank_distance_lmin(a, b, 3.5, None, None, 0);
            let rms_a = leftover_rms(a, 3.5, None, None);
            let rms_b = leftover_rms(b, 3.5, None, None);
            eprintln!(
                "{label}: leftover_profile={d_left:.6} high_l={d_high:.6} \
                 full_l={d_full:.6} leftover_rms={rms_a:.6}/{rms_b:.6}"
            );
            assert!(
                d_left < 0.05,
                "{label}: leftover profile was supposed to collapse, got {d_left}"
            );
            assert!(
                d_high > 0.05,
                "{label}: high-l mean SOAP must separate the funnels, got {d_high}"
            );
        }

        let rot = rotate_z(&ico75);
        let d_rot = soap_bank_distance(ico75.view(), rot.view(), 3.5, None, None);
        let d_self = soap_bank_distance(ico75.view(), ico75.view(), 3.5, None, None);
        eprintln!("LJ75 ico-rotated high_l={d_rot:.8} self={d_self:.8}");
        assert!(d_self < 1e-12, "self distance is {d_self}");
        assert!(
            d_rot < 0.01,
            "a rotation of the same packing must stay below Dcut scale, got {d_rot}"
        );

        // A surface kick quenched back onto the ico shelf is the isomer
        // leftover treated as a new basin. Dcut is half the seed-pair
        // mean; if that isomer sits near ico-Marks, the bank still fills
        // with Mackay variants.
        let mut kicked = ico75.clone();
        kicked[0] += 0.35;
        kicked[1] -= 0.20;
        let (e_iso, iso) = quench_lj(kicked.view());
        let d_iso_high = soap_bank_distance(ico75.view(), iso.view(), 3.5, None, None);
        let d_iso_full = soap_bank_distance_lmin(ico75.view(), iso.view(), 3.5, None, None, 0);
        let d_iso_left = leftover_profile_distance(ico75.view(), iso.view(), 3.5, None, None);
        let d_pack = soap_bank_distance(ico75.view(), marks.view(), 3.5, None, None);
        eprintln!(
            "LJ75 ico-isomer E={e_iso:.6} leftover={d_iso_left:.6} \
             high_l={d_iso_high:.6} full_l={d_iso_full:.6} vs_marks={d_pack:.6}"
        );
        assert!(
            d_pack > SOAP_DCUT_FALLBACK,
            "ico-Marks {d_pack} is below the SOAP Dcut fallback {SOAP_DCUT_FALLBACK}"
        );
        assert!(
            d_pack < 0.7,
            "ico-Marks {d_pack} must stay below merge_radius or the old fallback was not the bug"
        );
        assert!(
            (e_iso + 396.282249).abs() < 0.5,
            "surface kick left the ico shelf, E={e_iso}"
        );
        assert!(
            d_iso_high < 0.4 * d_pack,
            "ico isomer high-l {d_iso_high} is too close to ico-Marks {d_pack}"
        );
    }

    #[test]
    fn soap_pack_merge_is_the_funnel_not_the_isomer() {
        let ico75 = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
        let marks = load_xyz(include_str!("../tests/fixtures/lj75_marks.xyz"));
        let mut kicked = ico75.clone();
        kicked[0] += 0.35;
        kicked[1] -= 0.20;
        let (_e, iso) = quench_lj(kicked.view());
        let d_iso = soap_bank_distance(ico75.view(), iso.view(), 3.5, None, None);
        let d_marks = soap_bank_distance(ico75.view(), marks.view(), 3.5, None, None);
        assert!(
            d_iso < SOAP_PACK_MERGE,
            "ico isomer {d_iso} must merge into the packing well {SOAP_PACK_MERGE}"
        );
        assert!(
            d_marks > SOAP_PACK_MERGE,
            "ico-Marks {d_marks} must stay outside the packing well {SOAP_PACK_MERGE}"
        );
        assert!(
            d_iso < SOAP_PACK_GAP && d_marks > SOAP_PACK_GAP,
            "adopt gap {SOAP_PACK_GAP} must sit between ico isomer {d_iso} and Marks {d_marks}"
        );
    }

    /// Whether the leftover has anything to say on a small mobile set.
    ///
    /// step_away_featomic follows the leftover residual, and when its
    /// rms falls under DEFECT it stops following anything and kicks
    /// instead. A slab adsorbate is six mobile atoms in a frozen
    /// environment, which is the smallest mobile set the recommended
    /// molecular config is used on, so this is where a residual would
    /// run out first. Printed rather than asserted at a threshold,
    /// because the number is the finding.
    #[test]
    fn the_slab_adsorbate_leftover_against_the_defect_floor() {
        let proto = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]];
        let origins = [
            [0.0, 0.0, 3.3],
            [1.7, 1.7, 4.1],
            [0.0, 0.0, 0.0],
            [3.4, 0.0, 0.0],
        ];
        let mut coordinates = Vec::with_capacity(36);
        for origin in &origins {
            for point in &proto {
                for axis in 0..3 {
                    coordinates.push(point[axis] + origin[axis]);
                }
            }
        }
        let x = Array1::from(coordinates);
        let species = [8u32, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1];
        let mobile = [0usize, 1, 2, 3, 4, 5];
        let s = spectrum(x.view(), 3.5, Some(&species), Some(&mobile));
        let nnu = (s.n_at * s.n_feat).max(1) as f64;
        let rms = (s.leftover.iter().map(|v| v * v).sum::<f64>() / nnu).sqrt();
        println!(
            "slab adsorbate leftover rms {rms:.3e} against DEFECT {DEFECT:.3e}; \
             shell_leftover {}",
            shell_leftover(&s)
        );
        assert!(rms.is_finite(), "the slab leftover is not a number");
    }

    /// What it takes to leave a packing, measured.
    ///
    /// A single hole step does not do it. Sweeping the escape distance
    /// from one to five merge radii on the LJ75 icosahedral fixture,
    /// the walk reaches 0.33 to 0.55 in the cloud mean and the quench
    /// returns to the same DECAF family in every one of twelve seeds
    /// at every distance. What leaves is the requench loop: step,
    /// quench, continue from the quenched structure with a wider
    /// scale, and kick in the occupied packing's null space if six
    /// attempts all come home.
    ///
    /// This is the contrast a regression would break, so it is what
    /// the test asserts. Run with --nocapture to read the counts.
    #[test]
    fn leaving_a_packing_takes_the_requench_loop_not_one_hole_step() {
        const SEEDS: u64 = 12;
        let ico = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
        let origin = ico.as_slice().expect("fixture is contiguous");
        let rcut = 3.5;
        let well = unit(&spectrum(ico.view(), rcut, None, None).cloud_mean);
        let mut single = 0usize;
        let mut looped = 0usize;
        for seed in 0..SEEDS {
            let mut rng = StdRng::seed_from_u64(seed);
            let one = step_into_hole(
                ico.view(),
                std::slice::from_ref(&well),
                SOAP_PACK_MERGE,
                rcut,
                None,
                None,
                &mut rng,
            );
            let (_, quenched) = quench_lj(one.view());
            if crate::catalog::different_decaf_family(
                origin,
                quenched.as_slice().expect("quench is contiguous"),
            ) {
                single += 1;
            }

            let mut rng = StdRng::seed_from_u64(seed);
            let many = leave_occupied_packing(
                ico.view(),
                std::slice::from_ref(&well),
                rcut,
                None,
                None,
                |v| quench_lj(v).1,
                &mut rng,
            );
            if crate::catalog::different_decaf_family(
                origin,
                many.as_slice().expect("leave is contiguous"),
            ) {
                looped += 1;
            }
        }
        println!("single hole step {single}/{SEEDS}, requench loop {looped}/{SEEDS}");
        assert!(
            looped > single,
            "the requench loop left the packing {looped}/{SEEDS} against one step's \
             {single}/{SEEDS}; if one step is enough the loop is dead weight, and if \
             neither leaves then a Leave cannot leave"
        );
    }

    /// A hole step that stops at the merge radius has not left the
    /// packing: it ends on the boundary of the well, which is where
    /// the quench that follows it comes straight back from. The walk
    /// has to clear the separation between packings.
    #[test]
    fn a_hole_step_clears_the_escape_distance_not_the_merge_radius() {
        let ico = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
        let rcut = 3.5;
        let well = unit(&spectrum(ico.view(), rcut, None, None).cloud_mean);
        let mut rng = StdRng::seed_from_u64(11);
        let left = step_into_hole(
            ico.view(),
            std::slice::from_ref(&well),
            SOAP_PACK_MERGE,
            rcut,
            None,
            None,
            &mut rng,
        );
        let moved = unit(&spectrum(left.view(), rcut, None, None).cloud_mean);
        let distance = soap_l2_pack(moved.view(), well.view());
        assert!(
            distance >= SOAP_PACK_ESCAPE,
            "hole step stopped {distance} from the well it left, inside the escape {SOAP_PACK_ESCAPE}"
        );
    }

    #[test]
    fn packing_kick_moves_lj75_ico_mean_soap() {
        let ico75 = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
        let mut rng = StdRng::seed_from_u64(7);
        let y = step_away_featomic(ico75.view(), 0.35, 3.5, None, None, &mut rng);
        let mut s = 0.0;
        for i in 0..ico75.len() {
            let d = y[i] - ico75[i];
            s += d * d;
        }
        let rms = (s / 75.0).sqrt();
        assert!(rms > 0.1, "packing hop was a twitch, rms {rms}");
        let d = soap_bank_distance(ico75.view(), y.view(), 3.5, None, None);
        assert!(
            d > SOAP_DCUT_FALLBACK,
            "packing hop left mean SOAP unchanged, d={d}"
        );
    }

    #[test]
    fn surplus_reseed_uses_a_soap_hole_when_the_packing_is_occupied() {
        let ico75 = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
        let mu = soap_cloud_mean(ico75.view(), 3.5, None, None);
        let mut rng = StdRng::seed_from_u64(11);
        let y = surplus_reseed(
            ico75.view(),
            std::slice::from_ref(&mu),
            3.5,
            None,
            None,
            &mut rng,
        )
        .expect("occupied packing must yield a hole start");
        let d = soap_bank_distance(ico75.view(), y.view(), 3.5, None, None);
        assert!(
            d > SOAP_PACK_MERGE * 0.5,
            "surplus reseed stayed in the ico packing, d={d}"
        );
    }

    #[test]
    fn surplus_reseed_is_none_when_no_occupied_packing_is_known() {
        let ico75 = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
        let mut rng = StdRng::seed_from_u64(3);
        assert!(surplus_reseed(ico75.view(), &[], 3.5, None, None, &mut rng).is_none());
    }

    #[test]
    fn leave_accepts_a_quench_that_opens_a_new_decaf_family() {
        let ico75 = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
        let marks = load_xyz(include_str!("../tests/fixtures/lj75_marks.xyz"));
        let mu = soap_cloud_mean(ico75.view(), 3.5, None, None);
        let mut rng = StdRng::seed_from_u64(11);
        let wells = [mu];
        let left = leave_occupied_packing(
            ico75.view(),
            &wells,
            3.5,
            None,
            None,
            |_y| marks.clone(),
            &mut rng,
        );
        assert_eq!(left, marks);
        assert!(crate::catalog::different_decaf_family(
            ico75.as_slice().unwrap(),
            left.as_slice().unwrap()
        ));
    }

    #[test]
    fn leave_occupied_packing_kicks_when_every_quench_stays_ico() {
        let ico75 = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
        let mu = soap_cloud_mean(ico75.view(), 3.5, None, None);
        let mut rng = StdRng::seed_from_u64(11);
        let wells = [mu];
        let left = leave_occupied_packing(
            ico75.view(),
            &wells,
            3.5,
            None,
            None,
            |_y| ico75.clone(),
            &mut rng,
        );
        assert_ne!(left, ico75);
        let mut s = 0.0;
        for i in 0..ico75.len() {
            let d = left[i] - ico75[i];
            s += d * d;
        }
        let rms = (s / 75.0).sqrt();
        assert!(
            (rms - 0.35).abs() < 1e-9,
            "leave must packing_kick off ico, not leftover SOAP, rms={rms}"
        );
        let hole = step_into_hole(
            ico75.view(),
            &wells,
            SOAP_PACK_MERGE * 4.0,
            3.5,
            None,
            None,
            &mut StdRng::seed_from_u64(11),
        );
        assert_ne!(left, hole);
    }

    #[test]
    fn leave_occupied_packing_requenches_until_the_family_changes() {
        let ico75 = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
        let marks = load_xyz(include_str!("../tests/fixtures/lj75_marks.xyz"));
        let mu = soap_cloud_mean(ico75.view(), 3.5, None, None);
        let mut rng = StdRng::seed_from_u64(11);
        let wells = [mu];
        let mut n = 0u32;
        let left = leave_occupied_packing(
            ico75.view(),
            &wells,
            3.5,
            None,
            None,
            |_y| {
                n += 1;
                if n < 3 { ico75.clone() } else { marks.clone() }
            },
            &mut rng,
        );
        assert!(
            n >= 3,
            "leave must quench more than once when the first snaps stay ico, n={n}"
        );
        assert_eq!(left, marks);
        assert!(crate::catalog::different_decaf_family(
            ico75.as_slice().unwrap(),
            left.as_slice().unwrap()
        ));
    }

    #[test]
    fn hole_flow_moves_lj75_packing_mean_past_merge() {
        let ico75 = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
        let mu = soap_cloud_mean(ico75.view(), 3.5, None, None);
        let mut rng = StdRng::seed_from_u64(11);
        let y = step_into_hole(
            ico75.view(),
            std::slice::from_ref(&mu),
            SOAP_PACK_MERGE,
            3.5,
            None,
            None,
            &mut rng,
        );
        let d = soap_bank_distance(ico75.view(), y.view(), 3.5, None, None);
        assert!(
            d > SOAP_PACK_MERGE * 0.5,
            "hole flow stayed in the ico packing, d={d}"
        );
        for i in 0..y.len() {
            assert!(
                y[i].is_finite(),
                "packing hop produced a non-finite coordinate"
            );
        }
    }

    fn quench_lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
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
        let mut opt = crate::methods::warm_lbfgs::WarmLbfgs::default();
        let (f, xr, _) = opt.minimize(x, 400, |v| Some(lj(v)));
        (f, xr)
    }
}
