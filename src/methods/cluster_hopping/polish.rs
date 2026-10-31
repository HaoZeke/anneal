//! Final polish of a molecular search's recorded bests.
//!
//! A molecular search records a quench once its largest gradient component is
//! below [`Config::record_gradient`]. On a soft surface, hydrogen-bonded or
//! torsional, that leaves the record well above the minimum it lies in. Under
//! [`Config::record_polish`] the search ends a reserve of calls early, and
//! what is left relaxes its records, lowest first, with a quasi-Newton
//! optimizer that steps in the approximate normal coordinates of a model
//! Hessian and stops only on a small energy change and gradient norm.
//!
//! - Model Hessian: Lindh, Bernhardsson, Karlström and Malmqvist, Chem. Phys.
//!   Lett. 241, 423 (1995), with its published constants. Stretch and bend
//!   B-vectors as in Wilson, Decius and Cross (1955); torsion B-vectors as in
//!   Blondel and Karplus, J. Comput. Chem. 17, 1132 (1996).
//! - Step: rational function optimization (Banerjee, Adams, Simons and
//!   Shepard, J. Phys. Chem. 89, 52 (1985)) on a BFGS-updated Hessian in
//!   those coordinates.
//! - Curvature clamp, step cap, rebuild period and stop: the defaults and the
//!   tight level of xtb's ANCOPT.
//!
//! The optimizer runs in atomic units, which the rational-function step is not
//! invariant to. The search's coordinates and energies are the molecular
//! preset's angstrom and electronvolt.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use super::{
    Config, EnergyGradFn, Ledger, Outcome, answer_gradient_ok, bind_graph_reference, quench_is_sane,
};
use crate::spectral::symmetric_eigen;

/// Bohr radius in angstrom (CODATA 2018).
pub const BOHR: f64 = 0.529_177_210_903;
/// Hartree in electronvolt (CODATA 2018).
pub const HARTREE: f64 = 27.211_386_245_988;

/// Reference distances (bohr) by row pair.
const R_REF: [[f64; 3]; 3] = [[1.35, 2.10, 2.53], [2.10, 2.87, 3.40], [2.53, 3.40, 3.40]];
/// Weight exponents (bohr⁻²) by row pair.
const ALPHA: [[f64; 3]; 3] = [
    [1.0000, 0.3949, 0.3949],
    [0.3949, 0.2800, 0.2800],
    [0.3949, 0.2800, 0.2800],
];
/// Stretch force constant, hartree per bohr².
const K_STRETCH: f64 = 0.45;
/// Bend force constant, hartree per rad².
const K_BEND: f64 = 0.15;
/// Torsion force constant, hartree per rad².
const K_TORSION: f64 = 0.005;
/// Terms whose weight product is below this are left out.
const WEIGHT_FLOOR: f64 = 1e-8;
/// A bend with a smaller sine has no defined B-vector, nor a torsion through
/// it.
const LINEAR_SINE: f64 = 1e-2;
/// Lowest and highest model curvature, hartree per bohr².
const CURVATURE_LOW: f64 = 0.01;
const CURVATURE_HIGH: f64 = 5.0;
/// Longest step, bohr.
const MAX_STEP: f64 = 1.0;
/// Iterations between model Hessian rebuilds.
const MICRO: usize = 20;
/// Converged when the energy change is below this (hartree), the gradient
/// norm below [`GRADIENT_TOLERANCE`], and the energy fell.
const ENERGY_TOLERANCE: f64 = 1e-6;
/// Projected gradient norm, hartree per bohr.
const GRADIENT_TOLERANCE: f64 = 8e-4;
const JACOBI_SWEEPS: usize = 100;

/// One record's polish.
#[derive(Debug, Clone, PartialEq)]
pub struct PolishedRecord {
    /// Energy the search recorded.
    pub from: f64,
    /// Lowest energy the polish reached at a recordable structure, `+inf`
    /// when it reached none.
    pub to: f64,
    /// Charged evaluations it spent.
    pub calls: usize,
    /// Whether it met the stop rather than running out of calls or failing.
    pub converged: bool,
    /// Norm of the last projected gradient, eV/Å.
    pub gradient_norm: f64,
    /// Whether its structure became the ledger's best.
    pub recorded: bool,
}

/// What the polish under [`Config::record_polish`] did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PolishReport {
    /// Calls held back from the search.
    pub reserve: usize,
    /// Ledger spend when the search ended.
    pub search_charged: usize,
    /// Charged evaluations the polish spent.
    pub calls: usize,
    /// The records polished, lowest first.
    pub records: Vec<PolishedRecord>,
}

fn row(z: u32) -> usize {
    match z {
        0..=2 => 0,
        3..=10 => 1,
        _ => 2,
    }
}

type Vec3 = [f64; 3];

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn scale(a: Vec3, f: f64) -> Vec3 {
    [a[0] * f, a[1] * f, a[2] * f]
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

fn stretch_vector(a: Vec3, b: Vec3) -> Option<[Vec3; 2]> {
    let u = sub(a, b);
    let r = norm(u);
    if !(r > 0.0) {
        return None;
    }
    let e = scale(u, 1.0 / r);
    Some([e, scale(e, -1.0)])
}

/// The bend at `j` between `i` and `k`.
fn bend_vector(i: Vec3, j: Vec3, k: Vec3) -> Option<[Vec3; 3]> {
    let (u, v) = (sub(i, j), sub(k, j));
    let (lu, lv) = (norm(u), norm(v));
    if !(lu > 0.0 && lv > 0.0) {
        return None;
    }
    let (eu, ev) = (scale(u, 1.0 / lu), scale(v, 1.0 / lv));
    let sine = norm(cross(eu, ev));
    if sine < LINEAR_SINE {
        return None;
    }
    let cosine = dot(eu, ev);
    let bi = scale(sub(scale(eu, cosine), ev), 1.0 / (lu * sine));
    let bk = scale(sub(scale(ev, cosine), eu), 1.0 / (lv * sine));
    Some([bi, scale(add(bi, bk), -1.0), bk])
}

/// The torsion of `i-j-k-l` about `j-k`.
fn torsion_vector(i: Vec3, j: Vec3, k: Vec3, l: Vec3) -> Option<[Vec3; 4]> {
    let (f, g, h) = (sub(i, j), sub(j, k), sub(l, k));
    let (a, b) = (cross(f, g), cross(h, g));
    let (la, lb, lg) = (norm(a), norm(b), norm(g));
    if !(lg > 0.0) || la < LINEAR_SINE * norm(f) * lg || lb < LINEAR_SINE * norm(h) * lg {
        return None;
    }
    let (a2, b2) = (la * la, lb * lb);
    let (fg, hg) = (dot(f, g) / (a2 * lg), dot(h, g) / (b2 * lg));
    let bi = scale(a, -lg / a2);
    let bl = scale(b, lg / b2);
    let bj = add(scale(a, lg / a2 + fg), scale(b, -hg));
    let bk = add(scale(b, hg - lg / b2), scale(a, -fg));
    Some([bi, bj, bk, bl])
}

fn accumulate(h: &mut Array2<f64>, k: f64, atoms: &[usize], b: &[Vec3]) {
    for (bi, &a) in atoms.iter().enumerate() {
        for (bj, &c) in atoms.iter().enumerate() {
            for p in 0..3 {
                for q in 0..3 {
                    h[[3 * a + p, 3 * c + q]] += k * b[bi][p] * b[bj][q];
                }
            }
        }
    }
}

/// The Lindh model Hessian of `x` (angstrom), hartree per bohr².
fn model_hessian_au(x: ArrayView1<f64>, species: &[u32]) -> Array2<f64> {
    let n = species.len().min(x.len() / 3);
    let p: Vec<Vec3> = (0..n)
        .map(|i| [x[3 * i] / BOHR, x[3 * i + 1] / BOHR, x[3 * i + 2] / BOHR])
        .collect();
    let rows: Vec<usize> = species.iter().map(|&z| row(z)).collect();
    let mut rho = vec![vec![0.0_f64; n]; n];
    let mut rho_max = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = sub(p[i], p[j]);
            let (a, b) = (rows[i], rows[j]);
            let w = (ALPHA[a][b] * (R_REF[a][b] * R_REF[a][b] - dot(d, d))).exp();
            rho[i][j] = w;
            rho[j][i] = w;
            rho_max = rho_max.max(w);
        }
    }
    // Every factor of a kept product is at least the floor over the largest
    // two other factors, so pruning neighbours at that bound loses no term.
    let cut = WEIGHT_FLOOR / rho_max.max(1.0).powi(2);
    let near: Vec<Vec<usize>> = (0..n)
        .map(|i| (0..n).filter(|&j| j != i && rho[i][j] >= cut).collect())
        .collect();
    let mut h = Array2::<f64>::zeros((3 * n, 3 * n));
    for i in 0..n {
        for j in (i + 1)..n {
            let w = rho[i][j];
            if w < WEIGHT_FLOOR {
                continue;
            }
            if let Some(b) = stretch_vector(p[i], p[j]) {
                accumulate(&mut h, K_STRETCH * w, &[i, j], &b);
            }
        }
    }
    for j in 0..n {
        for (a, &i) in near[j].iter().enumerate() {
            for &k in &near[j][a + 1..] {
                let w = rho[i][j] * rho[j][k];
                if w < WEIGHT_FLOOR {
                    continue;
                }
                if let Some(b) = bend_vector(p[i], p[j], p[k]) {
                    accumulate(&mut h, K_BEND * w, &[i, j, k], &b);
                }
            }
        }
    }
    for j in 0..n {
        for &k in near[j].iter().filter(|&&k| k > j) {
            for &i in near[j].iter().filter(|&&i| i != k) {
                for &l in near[k].iter().filter(|&&l| l != i && l != j) {
                    let w = rho[i][j] * rho[j][k] * rho[k][l];
                    if w < WEIGHT_FLOOR {
                        continue;
                    }
                    if let Some(b) = torsion_vector(p[i], p[j], p[k], p[l]) {
                        accumulate(&mut h, K_TORSION * w, &[i, j, k, l], &b);
                    }
                }
            }
        }
    }
    h
}

/// The Lindh et al. (1995) model Hessian of `x` (angstrom), in eV/Å².
pub fn model_hessian(x: ArrayView1<f64>, species: &[u32]) -> Array2<f64> {
    model_hessian_au(x, species) * (HARTREE / (BOHR * BOHR))
}

/// Approximate normal coordinates and their clamped curvatures.
struct Frame {
    /// Orthonormal columns over the full Cartesian vector.
    basis: Array2<f64>,
    /// Hartree per bohr².
    curvature: Array1<f64>,
}

fn outer_add(m: &mut Array2<f64>, f: f64, v: ArrayView1<f64>) {
    let d = v.len();
    for a in 0..d {
        for b in 0..d {
            m[[a, b]] += f * v[a] * v[b];
        }
    }
}

fn frame(x: ArrayView1<f64>, species: &[u32], mobile: &[bool]) -> Option<Frame> {
    let h = model_hessian_au(x, species);
    let d = x.len();
    let (values, vectors, skip, idx): (Array1<f64>, Array2<f64>, usize, Vec<usize>) =
        if mobile.iter().all(|&m| m) {
            let rigid = crate::curvature::rigid_basis(x);
            let mut projector = Array2::<f64>::eye(d);
            for t in &rigid {
                outer_add(&mut projector, -1.0, t.view());
            }
            let mut m = projector.dot(&h).dot(&projector);
            let bound = m
                .rows()
                .into_iter()
                .map(|r| r.iter().map(|v| v.abs()).sum::<f64>())
                .fold(0.0_f64, f64::max);
            // The rigid modes are moved below every other eigenvalue, so the
            // first `rigid.len()` eigenvectors are exactly them.
            for t in &rigid {
                outer_add(&mut m, -2.0 * (bound + 1.0), t.view());
            }
            let m = (&m + &m.t()) * 0.5;
            let (values, vectors) = symmetric_eigen(m.view(), JACOBI_SWEEPS);
            (values, vectors, rigid.len(), (0..d).collect())
        } else {
            let idx: Vec<usize> = (0..d).filter(|&c| mobile[c]).collect();
            let m = Array2::from_shape_fn((idx.len(), idx.len()), |(a, b)| h[[idx[a], idx[b]]]);
            let (values, vectors) = symmetric_eigen(m.view(), JACOBI_SWEEPS);
            (values, vectors, 0, idx)
        };
    let kept = values.len().saturating_sub(skip);
    if kept == 0 {
        return None;
    }
    let mut basis = Array2::<f64>::zeros((d, kept));
    for (a, &c) in idx.iter().enumerate() {
        basis
            .row_mut(c)
            .assign(&vectors.slice(s![a, skip..skip + kept]));
    }
    let curvature = values
        .slice(s![skip..])
        .mapv(|v| v.clamp(CURVATURE_LOW, CURVATURE_HIGH));
    Some(Frame { basis, curvature })
}

/// The rational-function step: the lowest eigenvector of the gradient-
/// augmented Hessian, scaled so its last component is one.
fn rfo_step(h: ArrayView2<f64>, g: ArrayView1<f64>) -> Option<Array1<f64>> {
    let n = g.len();
    let mut a = Array2::<f64>::zeros((n + 1, n + 1));
    a.slice_mut(s![..n, ..n]).assign(&h);
    a.slice_mut(s![..n, n]).assign(&g);
    a.slice_mut(s![n, ..n]).assign(&g);
    let (_, vectors) = symmetric_eigen(a.view(), JACOBI_SWEEPS);
    let last = vectors[[n, 0]];
    if !(last.abs() >= 1e-10) {
        return None;
    }
    let step = vectors.slice(s![..n, 0]).mapv(|v| v / last);
    step.iter().all(|v| v.is_finite()).then_some(step)
}

fn bfgs(h: &mut Array2<f64>, step: ArrayView1<f64>, change: ArrayView1<f64>) {
    let ys = change.dot(&step);
    let hs = h.dot(&step);
    let shs = step.dot(&hs);
    let (ly, ls) = (change.dot(&change).sqrt(), step.dot(&step).sqrt());
    if !(ys > 1e-8 * ly * ls) || !(shs > 0.0) {
        return;
    }
    outer_add(h, -1.0 / shs, hs.view());
    outer_add(h, 1.0 / ys, change);
}

/// A recordable structure, its energy and the ledger spend when it was seen.
type Candidate = (f64, Array1<f64>, usize);

struct Relaxed {
    calls: usize,
    converged: bool,
    gradient_norm: f64,
    /// Energy of the first iterate, NaN when none was evaluated.
    first: f64,
    best: Option<Candidate>,
}

/// Relaxes `start` until the stop, a refused or non-finite evaluation, or a
/// failed step.
fn relax_record(
    cfg: &Config,
    species: &[u32],
    mobile: &[bool],
    start: ArrayView1<f64>,
    evaluate: &mut dyn FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>, usize)>,
) -> Relaxed {
    let mut out = Relaxed {
        calls: 0,
        converged: false,
        gradient_norm: f64::NAN,
        first: f64::NAN,
        best: None,
    };
    let mut x = start.to_owned();
    let mut current: Option<Frame> = None;
    let mut h = Array2::<f64>::zeros((0, 0));
    let mut since = 0usize;
    // Energy (hartree), projected gradient and step of the last iterate, and
    // whether the frame has been rebuilt since.
    let mut previous: Option<(f64, Array1<f64>, Array1<f64>)> = None;
    let mut fresh = true;
    loop {
        if current.is_none() || since >= MICRO {
            let Some(f) = frame(x.view(), species, mobile) else {
                break;
            };
            h = Array2::from_diag(&f.curvature);
            current = Some(f);
            since = 0;
            fresh = true;
        }
        let Some(f) = current.as_ref() else {
            break;
        };
        let Some((energy, gradient, spent)) = evaluate(x.view()) else {
            break;
        };
        out.calls += 1;
        if out.calls == 1 {
            out.first = energy;
        }
        if !energy.is_finite()
            || gradient.len() != x.len()
            || gradient.iter().any(|v| !v.is_finite())
        {
            break;
        }
        if quench_is_sane(cfg, energy, x.view())
            && answer_gradient_ok(cfg, gradient.view(), x.len())
            && out.best.as_ref().is_none_or(|b| energy < b.0)
        {
            out.best = Some((energy, x.clone(), spent));
        }
        let projected = f.basis.t().dot(&(&gradient * (BOHR / HARTREE)));
        let gradient_norm = projected.dot(&projected).sqrt();
        out.gradient_norm = gradient_norm * HARTREE / BOHR;
        let e = energy / HARTREE;
        if let Some((e_last, g_last, step_last)) = previous.as_ref() {
            let change = e - e_last;
            if change.abs() < ENERGY_TOLERANCE && gradient_norm < GRADIENT_TOLERANCE && change < 0.0
            {
                out.converged = true;
                break;
            }
            if !fresh {
                bfgs(&mut h, step_last.view(), (&projected - g_last).view());
            }
        }
        let Some(mut step) = rfo_step(h.view(), projected.view()) else {
            break;
        };
        let length = step.dot(&step).sqrt();
        if length > MAX_STEP {
            step *= MAX_STEP / length;
        }
        x += &(f.basis.dot(&step) * BOHR);
        previous = Some((e, projected, step));
        since += 1;
        fresh = false;
    }
    out
}

/// Mobile Cartesians: those of points [`Config::frozen`] does not hold.
fn mobile_coordinates(cfg: &Config, dims: usize) -> Vec<bool> {
    (0..dims)
        .map(|c| {
            !cfg.frozen
                .as_deref()
                .is_some_and(|f| f.get(c / 3).copied().unwrap_or(false))
        })
        .collect()
}

/// Calls held back from a search for the polish of its records.
///
/// The dimension of the polish's coordinates: the mobile Cartesians less the
/// rigid-body modes, which are removed only when nothing is frozen. That is
/// the number of steps in which a Broyden-class method with exact line
/// searches terminates on a quadratic. Capped at a quarter of `remaining`.
/// Zero when the setting is off, without species, or for a rigid-body move
/// library, whose relaxation is not Cartesian.
pub(crate) fn reserve(cfg: &Config, start: ArrayView1<f64>, remaining: usize) -> usize {
    let Some(species) = cfg.species.as_deref() else {
        return 0;
    };
    if !cfg.record_polish || cfg.move_library.is_rigid_body() || start.len() != 3 * species.len() {
        return 0;
    }
    let mobile = mobile_coordinates(cfg, start.len());
    let free = mobile.iter().filter(|&&m| m).count();
    let rigid = if free == start.len() {
        crate::curvature::rigid_basis(start).len()
    } else {
        0
    };
    free.saturating_sub(rigid).min(remaining / 4)
}

/// Polishes `records` (in the order the ledger accepted them), lowest first,
/// with every call left on `ledger`, and folds the result into `outcome`.
pub(super) fn polish_records(
    cfg: &Config,
    ledger: &mut Ledger,
    energy_grad: &mut EnergyGradFn<'_>,
    records: Vec<(f64, Array1<f64>)>,
    reserve: usize,
    outcome: &mut Outcome,
) {
    let Some(species) = cfg.species.as_deref() else {
        return;
    };
    let mut report = PolishReport {
        reserve,
        search_charged: ledger.spent(),
        ..PolishReport::default()
    };
    for (from, state) in records.into_iter().rev() {
        if ledger.remaining() == 0 {
            break;
        }
        let mobile = mobile_coordinates(cfg, state.len());
        let relaxed = {
            let mut evaluate = |x: ArrayView1<f64>| -> Option<(f64, Array1<f64>, usize)> {
                let (energy, gradient) = energy_grad(ledger, x)?;
                Some((energy, gradient, ledger.spent()))
            };
            relax_record(cfg, species, &mobile, state.view(), &mut evaluate)
        };
        let mut record = PolishedRecord {
            from,
            to: relaxed.best.as_ref().map_or(f64::INFINITY, |b| b.0),
            calls: relaxed.calls,
            converged: relaxed.converged,
            gradient_norm: relaxed.gradient_norm,
            recorded: false,
        };
        if let Some((energy, x, spent)) = relaxed.best
            && energy < ledger.best
        {
            ledger.record(energy, x.view());
            if outcome.improvements.len() < 512 {
                outcome
                    .improvements
                    .push((outcome.hops, spent, outcome.basins, energy));
            }
            record.recorded = true;
        }
        report.records.push(record);
    }
    report.calls = ledger.spent() - report.search_charged;
    outcome.best = ledger.best;
    outcome.best_state = ledger.best_state.clone();
    outcome.charged = ledger.spent();
    outcome.polish = Some(report);
}

/// Polishes one structure with every call `ledger` has, as the final polish
/// would, and returns its report and the lowest recordable structure.
///
/// Under [`Config::keep_bond_graph`] the graph of `state` is the reference.
/// `from` is the energy of the first iterate, and `recorded` is always false:
/// the ledger only charges. `None` without species or for a state of the
/// wrong length.
pub fn polish_structure(
    cfg: &Config,
    ledger: &mut Ledger,
    energy_grad: &mut EnergyGradFn<'_>,
    state: ArrayView1<f64>,
) -> Option<(PolishedRecord, Option<(f64, Array1<f64>)>)> {
    let species = cfg.species.as_deref()?;
    if state.len() != 3 * species.len() {
        return None;
    }
    let bound = bind_graph_reference(cfg, state);
    let cfg = bound.as_ref().unwrap_or(cfg);
    let mobile = mobile_coordinates(cfg, state.len());
    let relaxed = {
        let mut evaluate = |x: ArrayView1<f64>| -> Option<(f64, Array1<f64>, usize)> {
            let (energy, gradient) = energy_grad(ledger, x)?;
            Some((energy, gradient, ledger.spent()))
        };
        relax_record(cfg, species, &mobile, state, &mut evaluate)
    };
    let record = PolishedRecord {
        from: relaxed.first,
        to: relaxed.best.as_ref().map_or(f64::INFINITY, |b| b.0),
        calls: relaxed.calls,
        converged: relaxed.converged,
        gradient_norm: relaxed.gradient_norm,
        recorded: false,
    };
    Some((record, relaxed.best.map(|(energy, x, _)| (energy, x))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn points(rng: &mut StdRng, n: usize) -> Vec<Vec3> {
        (0..n)
            .map(|_| {
                [
                    rng.random_range(-2.0..2.0),
                    rng.random_range(-2.0..2.0),
                    rng.random_range(-2.0..2.0),
                ]
            })
            .collect()
    }

    fn angle(i: Vec3, j: Vec3, k: Vec3) -> f64 {
        let (u, v) = (sub(i, j), sub(k, j));
        norm(cross(u, v)).atan2(dot(u, v))
    }

    fn dihedral(i: Vec3, j: Vec3, k: Vec3, l: Vec3) -> f64 {
        let (f, g, h) = (sub(i, j), sub(j, k), sub(l, k));
        let (a, b) = (cross(f, g), cross(h, g));
        (dot(cross(b, a), g) / norm(g)).atan2(dot(a, b))
    }

    /// Central differences of `q` over each atom's coordinates.
    fn numeric(q: &dyn Fn(&[Vec3]) -> f64, p: &[Vec3]) -> Vec<Vec3> {
        let step = 1e-6;
        (0..p.len())
            .map(|a| {
                let mut out = [0.0; 3];
                for c in 0..3 {
                    let mut up = p.to_vec();
                    let mut down = p.to_vec();
                    up[a][c] += step;
                    down[a][c] -= step;
                    out[c] = (q(&up) - q(&down)) / (2.0 * step);
                }
                out
            })
            .collect()
    }

    fn assert_close(analytic: &[Vec3], numeric: &[Vec3]) {
        for (a, n) in analytic.iter().zip(numeric) {
            for c in 0..3 {
                assert!(
                    (a[c] - n[c]).abs() < 1e-6 * (1.0 + n[c].abs()),
                    "analytic {a:?} numeric {n:?}"
                );
            }
        }
    }

    /// Each B-vector is the gradient of its coordinate.
    #[test]
    fn b_vectors_match_central_differences() {
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..50 {
            let p = points(&mut rng, 4);
            if let Some(b) = stretch_vector(p[0], p[1]) {
                let num = numeric(&|q| norm(sub(q[0], q[1])), &p[..2]);
                assert_close(&b, &num);
            }
            if let Some(b) = bend_vector(p[0], p[1], p[2]) {
                let num = numeric(&|q| angle(q[0], q[1], q[2]), &p[..3]);
                assert_close(&b, &num);
            }
            if let Some(b) = torsion_vector(p[0], p[1], p[2], p[3]) {
                let num = numeric(&|q| dihedral(q[0], q[1], q[2], q[3]), &p);
                assert_close(&b, &num);
            }
        }
    }

    fn water_dimer() -> (Array1<f64>, Vec<u32>) {
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0, 2.9, 0.1, 0.2, 3.2, 0.9, 0.3, 3.2,
            -0.9, -0.1,
        ]);
        (x, vec![8, 1, 1, 8, 1, 1])
    }

    /// The model is symmetric, positive semidefinite, and blind to rigid
    /// motion, since every term is built on an internal coordinate.
    #[test]
    fn model_hessian_is_symmetric_and_annihilates_rigid_modes() {
        let (x, species) = water_dimer();
        let h = model_hessian(x.view(), &species);
        let largest = h.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        for a in 0..h.nrows() {
            for b in 0..h.ncols() {
                assert!((h[[a, b]] - h[[b, a]]).abs() <= 1e-12 * largest);
            }
        }
        for t in crate::curvature::rigid_basis(x.view()) {
            let ht = h.dot(&t);
            assert!(ht.dot(&ht).sqrt() < 1e-9 * largest, "rigid residual {ht}");
        }
        let (values, _) = symmetric_eigen(h.view(), JACOBI_SWEEPS);
        assert!(values.iter().all(|&v| v > -1e-9 * largest));
        assert!(values.iter().filter(|&&v| v > 1e-6 * largest).count() == 12);
    }

    /// The approximate normal coordinates are orthonormal, exclude the rigid
    /// modes, and keep frozen coordinates out.
    #[test]
    fn frame_is_orthonormal_and_respects_frozen_points() {
        let (x, species) = water_dimer();
        let all = vec![true; x.len()];
        let f = frame(x.view(), &species, &all).expect("frame");
        assert_eq!(f.basis.ncols(), 12);
        let gram = f.basis.t().dot(&f.basis);
        for a in 0..12 {
            for b in 0..12 {
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((gram[[a, b]] - want).abs() < 1e-9);
            }
        }
        for t in crate::curvature::rigid_basis(x.view()) {
            assert!(f.basis.t().dot(&t).iter().all(|v| v.abs() < 1e-9));
        }
        assert!(
            f.curvature
                .iter()
                .all(|&c| (CURVATURE_LOW..=CURVATURE_HIGH).contains(&c))
        );
        let mut mobile = all.clone();
        for c in 0..9 {
            mobile[c] = false;
        }
        let f = frame(x.view(), &species, &mobile).expect("frame");
        assert_eq!(f.basis.ncols(), 9);
        assert!(f.basis.slice(s![..9, ..]).iter().all(|&v| v == 0.0));
    }

    /// A Morse cluster in angstrom and electronvolt.
    fn morse(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let (depth, width, r0) = (1.0, 1.4, 1.5);
        let n = x.len() / 3;
        let mut e = 0.0;
        let mut g = Array1::<f64>::zeros(x.len());
        for i in 0..n {
            for j in (i + 1)..n {
                let d: Vec3 = [
                    x[3 * i] - x[3 * j],
                    x[3 * i + 1] - x[3 * j + 1],
                    x[3 * i + 2] - x[3 * j + 2],
                ];
                let r = norm(d);
                let q = (-width * (r - r0)).exp();
                e += depth * (q * q - 2.0 * q);
                let de = depth * 2.0 * width * (q - q * q) / r;
                for c in 0..3 {
                    g[3 * i + c] += de * d[c];
                    g[3 * j + c] -= de * d[c];
                }
            }
        }
        (e, g)
    }

    fn tetrahedron(rng: &mut StdRng) -> Array1<f64> {
        let base = [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ];
        let mut x = Vec::with_capacity(12);
        for p in base {
            for v in p {
                x.push(v * 0.53 + rng.random_range(-0.08..0.08));
            }
        }
        Array1::from(x)
    }

    fn molecular_config() -> Config {
        let mut cfg = Config::recommended_molecular(vec![6; 4], vec![(0..4).collect()], 1.0);
        cfg.keep_bond_graph = false;
        cfg
    }

    /// From a perturbed tetrahedron the polish meets the tight stop within
    /// twice the coordinate dimension, and lands on the Morse minimum.
    #[test]
    fn relax_record_meets_the_tight_stop_on_a_morse_cluster() {
        let cfg = molecular_config();
        let species = cfg.species.clone().expect("species");
        let mut rng = StdRng::seed_from_u64(3);
        let start = tetrahedron(&mut rng);
        let mobile = vec![true; start.len()];
        let mut spent = 0usize;
        let mut evaluate = |x: ArrayView1<f64>| {
            spent += 1;
            let (e, g) = morse(x);
            Some((e, g, spent))
        };
        let out = relax_record(&cfg, &species, &mobile, start.view(), &mut evaluate);
        assert!(
            out.converged,
            "calls {} norm {}",
            out.calls, out.gradient_norm
        );
        assert!(out.calls <= 12, "calls {}", out.calls);
        let (e, _, _) = out.best.expect("a recordable structure");
        assert!((e - (-6.0)).abs() < 1e-4, "energy {e}");
    }

    /// A refused evaluation ends the polish at once.
    #[test]
    fn relax_record_stops_when_refused() {
        let cfg = molecular_config();
        let species = cfg.species.clone().expect("species");
        let mut rng = StdRng::seed_from_u64(4);
        let start = tetrahedron(&mut rng);
        let mobile = vec![true; start.len()];
        let mut left = 2usize;
        let mut evaluate = |x: ArrayView1<f64>| {
            if left == 0 {
                return None;
            }
            left -= 1;
            let (e, g) = morse(x);
            Some((e, g, 2 - left))
        };
        let out = relax_record(&cfg, &species, &mobile, start.view(), &mut evaluate);
        assert_eq!(out.calls, 2);
        assert!(!out.converged);
    }

    struct MorseObjective {
        dim: usize,
        bounds: eindir_core::Bounds<f64>,
    }

    impl MorseObjective {
        fn new(dim: usize) -> Self {
            Self {
                dim,
                bounds: eindir_core::Bounds::new(
                    Array1::from_elem(dim, -50.0),
                    Array1::from_elem(dim, 50.0),
                    0.0,
                ),
            }
        }
    }

    impl eindir_core::Objective<f64> for MorseObjective {
        fn dim(&self) -> usize {
            self.dim
        }
        fn bounds(&self) -> &eindir_core::Bounds<f64> {
            &self.bounds
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            morse(x).0
        }
    }

    impl eindir_core::Gradient<f64> for MorseObjective {
        fn dim(&self) -> usize {
            self.dim
        }
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            morse(x).1
        }
    }

    impl eindir_core::gradient::DifferentiableObjective<f64> for MorseObjective {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            morse(x)
        }
    }

    /// A slightly puckered hexagon with Morse-length edges: a saddle.
    fn hexagon() -> Array1<f64> {
        let mut x = Vec::with_capacity(18);
        for k in 0..6 {
            let t = k as f64 * std::f64::consts::PI / 3.0;
            x.extend([1.5 * t.cos(), 1.5 * t.sin(), 0.05 * (k % 2) as f64]);
        }
        Array1::from(x)
    }

    /// The search ends a reserve early, the polish spends at most what is
    /// left, and the run never ends above the search part's best. The
    /// search part is the search that the budget less the reserve runs.
    #[test]
    fn a_polishing_search_keeps_its_budget_and_only_lowers() {
        let budget = 300;
        let start = hexagon();
        let objective = MorseObjective::new(start.len());
        let mut cfg = Config::recommended_molecular(vec![6; 6], vec![(0..6).collect()], 1.0);
        cfg.keep_bond_graph = false;
        let mut ledger = Ledger::new(budget);
        let (out, _) = crate::methods::cluster_search::search_from(
            &objective,
            &cfg,
            &mut ledger,
            start.view(),
            1,
        )
        .expect("finite start");
        let polish = out.polish.as_ref().expect("the setting polishes");
        assert_eq!(polish.reserve, 12);
        assert!(out.charged <= budget);
        assert_eq!(out.charged, ledger.spent());
        assert_eq!(polish.search_charged + polish.calls, out.charged);
        assert!(polish.search_charged <= budget - polish.reserve);
        let lowest = polish.records.first().expect("a record").from;
        assert!(out.best <= lowest);

        cfg.record_polish = false;
        let mut ledger = Ledger::new(budget - polish.reserve);
        let (plain, _) = crate::methods::cluster_search::search_from(
            &objective,
            &cfg,
            &mut ledger,
            start.view(),
            1,
        )
        .expect("finite start");
        assert!(plain.polish.is_none());
        assert_eq!(plain.charged, polish.search_charged);
        assert_eq!(plain.best, lowest);
        assert_eq!(
            plain.improvements[..],
            out.improvements[..plain.improvements.len()]
        );
    }

    /// Without species the setting does nothing.
    #[test]
    fn a_search_without_species_is_unchanged() {
        let start = hexagon();
        let objective = MorseObjective::new(start.len());
        let mut on = Config::for_cluster(6);
        on.record_polish = true;
        let mut off = on.clone();
        off.record_polish = false;
        let run = |cfg: &Config| {
            let mut ledger = Ledger::new(200);
            crate::methods::cluster_search::search_from(
                &objective,
                cfg,
                &mut ledger,
                start.view(),
                2,
            )
            .expect("finite start")
            .0
        };
        let (a, b) = (run(&on), run(&off));
        assert!(a.polish.is_none() && b.polish.is_none());
        assert_eq!(a.best, b.best);
        assert_eq!(a.charged, b.charged);
        assert_eq!(a.best_state, b.best_state);
        assert_eq!(a.improvements, b.improvements);
    }

    /// The single-structure polish charges every call, starts its report at
    /// the first iterate, and records nothing.
    #[test]
    fn polish_structure_charges_and_reports_the_first_iterate() {
        let cfg = molecular_config();
        let mut rng = StdRng::seed_from_u64(6);
        let start = tetrahedron(&mut rng);
        let mut ledger = Ledger::new(40);
        let mut energy_grad = |led: &mut Ledger, x: ArrayView1<f64>| {
            if !led.charge() {
                return None;
            }
            Some(morse(x))
        };
        let (record, best) =
            polish_structure(&cfg, &mut ledger, &mut energy_grad, start.view()).expect("species");
        assert_eq!(record.from, morse(start.view()).0);
        assert_eq!(record.calls, ledger.spent());
        assert!(record.converged && !record.recorded);
        let (energy, _) = best.expect("a recordable structure");
        assert_eq!(energy, record.to);
        assert!(energy < record.from);
        let lj = Config::recommended(4);
        assert!(polish_structure(&lj, &mut ledger, &mut energy_grad, start.view()).is_none());
    }

    #[test]
    fn reserve_is_the_coordinate_dimension_capped_at_a_quarter() {
        let mut rng = StdRng::seed_from_u64(5);
        let start = tetrahedron(&mut rng);
        let mut cfg = molecular_config();
        assert!(cfg.record_polish);
        assert_eq!(reserve(&cfg, start.view(), 1000), 6);
        assert_eq!(reserve(&cfg, start.view(), 20), 5);
        cfg.frozen = Some(vec![true, false, false, false]);
        assert_eq!(reserve(&cfg, start.view(), 1000), 9);
        cfg.record_polish = false;
        assert_eq!(reserve(&cfg, start.view(), 1000), 0);
        let lj = Config::recommended(4);
        assert_eq!(reserve(&lj, start.view(), 1000), 0);
    }
}
