//! Molecular relaxation in approximate normal coordinates.
//!
//! A quasi-Newton relaxation in Cartesian coordinates starts from a unit
//! Hessian and has to learn, one curvature pair at a time, that a bond
//! stretch is a hundred times stiffer than a torsion. On a flexible molecule
//! that is most of the relaxation. The molecular optimisers that converge in
//! tens of cycles start from a model Hessian instead, built from the geometry
//! alone, and step in the coordinates that diagonalise it.
//!
//! # What is implemented
//!
//! This is the scheme of the ANCopt optimiser in xtb (Grimme and co-workers;
//! `src/optimizer.f90`, `src/type/anc.f90`, `src/bfgs.f90` and
//! `src/model_hessian.f90` of xtb 6.7.1), written from those sources and the
//! papers below rather than translated:
//!
//! 1. A model Hessian at the current geometry. [`ModelHessian::Lindh`] is the
//!    Lindh 1995 valence model: stretch, bend and torsion force constants
//!    `k exp(alpha_ij (r_ref,ij^2 - r_ij^2))` per row pair, applied through the
//!    Wilson B-vectors of each internal coordinate, so the Hessian is
//!    Cartesian and needs no internal-coordinate bookkeeping. The prefactors
//!    are xtb's retuned 0.40, 0.13 and 0.0075 rather than the paper's 0.45,
//!    0.15 and 0.005. The stretch of every pair also carries xtb's damped C6
//!    term, with the D2 coefficients scaled by 20: the Cartesian Hessian of
//!    the attraction `V = -s6 c6 f(r) / r^6` is subtracted from the block, as
//!    `ddvopt` subtracts it (`Hxx = gmm x x / r^2 - vdw`). Beyond the damping
//!    radius that gives a pair the positive radial curvature `-V''`, which is
//!    what keeps a hydrogen-bonded or dispersion-bound complex from looking
//!    like separate molecules to the model.
//!    [`ModelHessian::Swart`] swaps the weight for the Swart--Bickelhaupt
//!    covalent-radius form `exp(1 - r_ij / (R_i + R_j))`.
//! 2. Approximate normal coordinates. The model is projected onto the
//!    complement of the rigid translations and rotations (onto the mobile
//!    coordinates when atoms are frozen), diagonalised, its spectrum shifted
//!    so the softest mode sits at `hlow` and clamped to `[hlow, hmax]`. The
//!    eigenvectors are the coordinates and the clamped eigenvalues their
//!    starting Hessian.
//! 3. Rational function optimisation (Banerjee, Adams, Simons and Shepard) in
//!    those coordinates: the step is the lowest eigenvector of the gradient-
//!    augmented Hessian, each component capped at `max_component` bohr.
//! 4. A quasi-Newton update of the coordinate Hessian after every step, BFGS
//!    with xtb's diagonal floor or the Bofill SR1/PSB mixture.
//! 5. A rebuild of the coordinates from a fresh model Hessian every
//!    `micro_cycles` steps, the interval growing by a tenth per rebuild up to
//!    twice the start. With the trust radius off, a proposed step longer than
//!    2 bohr after the second in a basis also forces a rebuild; with it on,
//!    that step is scaled to the radius instead. Linear coordinates drift out
//!    of date as torsions turn, and the rebuild is what lets the relaxation
//!    follow a large conformational change.
//!
//! Three departures from xtb. The convergence test is the caller's: the
//! largest Cartesian gradient component under `gtol`, the test
//! [`crate::methods::warm_lbfgs::WarmLbfgs`] applies, so the two count calls
//! to the same answer. [`HessianUpdate::Bofill`] is Bofill's update, where
//! xtb's alternative to BFGS is Powell's. And [`AncOpt::trust_radius`], off
//! by default, puts a trust radius on the step with a ratio test: a step that
//! raises the energy by more than [`AncOpt::energy_rise`] is rejected, and
//! the coordinate Hessian is updated only with the steps that are kept. xtb
//! takes every step it proposes.
//!
//! # Measured
//!
//! GFN2-xTB through the `xtb` 6.7.1 executable with `MKL_CBWR=COMPATIBLE`,
//! one counted call per value and gradient, from RDKit ETKDG starts with no
//! force-field cleanup, four seeds each (`examples/local_opt_bench.rs`).
//! Both optimisers stop at a largest gradient component of 5e-2 eV/A, the
//! molecular presets' `record_gradient`. Calls summed over the four starts:
//!
//! | system | atoms | L-BFGS | ANC | xtb `--opt loose` cycles |
//! |---|---|---|---|---|
//! | met-enkephalin | 75 | 419 | 172 | 182 |
//! | trofinetide | 43 | 486 | 154 | 172 |
//! | N-methylacetamide and two waters | 18 | 357 | 165 | 181 |
//!
//! ANC ended lower than L-BFGS on eleven of the twelve starts, by 8.7 to
//! 219 meV, and 2.8 meV higher on the twelfth. At that gate neither end point
//! is a converged minimum. At 5e-3 eV/A the sums are 4906 against 882, 2103
//! against 388 and 1452 against 349, but eight of the twelve pairs end at
//! different minima (406.5 meV apart on one met-enkephalin start), so the
//! fewer calls do not buy the same answer.
//!
//! # Units
//!
//! The model Hessian is in hartree per bohr squared and its reference lengths
//! in bohr, so the optimiser has to know what the objective's units are.
//! [`AncUnits`] carries that; the default is electronvolt and angstrom, the
//! units of the molecular presets and of GFN-class engines behind them.
//!
//! # Survey
//!
//! - Lindh, Bernhardsson, Karlstrom and Malmqvist, Chem. Phys. Lett. 241,
//!   423 (1995), doi:10.1016/0009-2614(95)00646-L: the model Hessian.
//! - Swart and Bickelhaupt, Int. J. Quantum Chem. 106, 2536 (2006),
//!   doi:10.1002/qua.21049: the covalent-radius weight and the case for
//!   optimising weak and strong coordinates together.
//! - Bakken and Helgaker, J. Chem. Phys. 117, 9160 (2002),
//!   doi:10.1063/1.1515483: redundant internal coordinates and why the model
//!   Hessian, not the coordinate system, carries most of the gain over
//!   Cartesian quasi-Newton.
//! - Banerjee, Adams, Simons and Shepard, J. Phys. Chem. 89, 52 (1985),
//!   doi:10.1021/j100247a015: the rational function step.
//! - Bofill, J. Comput. Chem. 15, 1 (1994), doi:10.1002/jcc.540150102: the
//!   SR1/PSB update that stays correct on an indefinite Hessian.
//! - Wang and Song, J. Chem. Phys. 144, 214108 (2016), doi:10.1063/1.4952956:
//!   geomeTRIC, translation-rotation internal coordinates for complexes, and
//!   the trust-radius control with step rejection used here.
//! - Packwood, Kermode, Mones, Bernstein, Woolley, Gould, Ortner and Csanyi,
//!   J. Chem. Phys. 144, 164109 (2016), doi:10.1063/1.4947024: the same
//!   pairwise exponential used as a preconditioner for L-BFGS instead (ASE
//!   `PreconLBFGS`, the `Exp` precon in rgmin); [`crate::model_hessian`]
//!   holds that operator.
//! - Bannwarth, Ehlert and Grimme, J. Chem. Theory Comput. 15, 1652 (2019),
//!   doi:10.1021/acs.jctc.8b01176, and Bannwarth et al., WIREs Comput. Mol.
//!   Sci. 11, e1493 (2021), doi:10.1002/wcms.1493: GFN2-xTB and the xtb
//!   program that ships ANCopt.

use ndarray::{Array1, ArrayView1};

/// Hartree in electronvolt (CODATA 2018).
pub const HARTREE_EV: f64 = 27.211_386_245_988;
/// Bohr in angstrom (CODATA 2018).
pub const BOHR_ANGSTROM: f64 = 0.529_177_210_903;

/// Units of the objective, as multiples of the atomic units the model is in.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct AncUnits {
    /// Objective energy unit per hartree: 27.211 for electronvolt.
    pub energy_per_hartree: f64,
    /// Objective length unit per bohr: 0.529 for angstrom.
    pub length_per_bohr: f64,
}

impl AncUnits {
    /// Electronvolt and angstrom.
    pub const EV_ANGSTROM: Self = Self {
        energy_per_hartree: HARTREE_EV,
        length_per_bohr: BOHR_ANGSTROM,
    };
    /// Hartree and bohr.
    pub const ATOMIC: Self = Self {
        energy_per_hartree: 1.0,
        length_per_bohr: 1.0,
    };
}

impl Default for AncUnits {
    fn default() -> Self {
        Self::EV_ANGSTROM
    }
}

/// Which empirical force constants build the model Hessian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelHessian {
    /// Lindh 1995, row-pair exponentials, with xtb's constants and C6 term.
    #[default]
    Lindh,
    /// Swart--Bickelhaupt 2006 covalent-radius weights, same terms.
    Swart,
}

/// Quasi-Newton update of the coordinate Hessian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HessianUpdate {
    /// BFGS, skipped on non-positive curvature, diagonal floored at `hlow`.
    #[default]
    Bfgs,
    /// Bofill's mixture of the symmetric rank-one and Powell updates.
    Bofill,
}

/// Which local optimiser a cluster search relaxes each structure with
/// ([`crate::methods::cluster_hopping::Config::local_step`]).
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LocalStep {
    /// Warm-started L-BFGS with a Wolfe line search.
    #[default]
    Lbfgs,
    /// [`AncOpt`] at xtb's settings, gated at the search's `record_gradient`.
    Anc {
        /// Model Hessian the coordinates come from.
        model: ModelHessian,
        /// Units of the objective's energy and coordinates.
        units: AncUnits,
    },
}

impl LocalStep {
    /// Lindh-model ANC for an objective in electronvolt and angstrom.
    pub const ANC_EV_ANGSTROM: Self = Self::Anc {
        model: ModelHessian::Lindh,
        units: AncUnits::EV_ANGSTROM,
    };

    pub(crate) fn is_lbfgs(&self) -> bool {
        matches!(self, Self::Lbfgs)
    }
}

/// Approximate-normal-coordinate rational function relaxation.
#[derive(Debug, Clone)]
pub struct AncOpt {
    /// Atomic number of every atom, in coordinate order.
    pub species: Vec<u32>,
    /// Atoms held fixed. With any atom frozen the rigid motions are not
    /// projected out, since the frozen atoms are the frame.
    pub frozen: Option<Vec<bool>>,
    /// Largest Cartesian gradient component at convergence, objective units.
    pub gtol: f64,
    /// The objective's units.
    pub units: AncUnits,
    /// Model Hessian the coordinates are built from.
    pub model: ModelHessian,
    /// Update applied to the coordinate Hessian after each step.
    pub update: HessianUpdate,
    /// Softest model eigenvalue, hartree per bohr squared (xtb: 0.01).
    pub hlow: f64,
    /// Stiffest model eigenvalue, hartree per bohr squared (xtb: 5).
    pub hmax: f64,
    /// Cap on each coordinate component of a step, bohr (xtb: 1).
    pub max_component: f64,
    /// Steps between coordinate rebuilds at the start (xtb: 20).
    pub micro_cycles: usize,
    /// Starting trust radius on the step norm, bohr. `None` takes every
    /// proposed step, as xtb does.
    pub trust_radius: Option<f64>,
    /// Largest trust radius, bohr.
    pub trust_max: f64,
    /// Energy rise, hartree, above which a trust-radius step is rejected.
    pub energy_rise: f64,
}

impl AncOpt {
    /// Settings for `species` in the given units, at xtb's defaults.
    pub fn new(species: Vec<u32>, units: AncUnits) -> Self {
        Self {
            species,
            frozen: None,
            gtol: 1e-3,
            units,
            model: ModelHessian::Lindh,
            update: HessianUpdate::Bfgs,
            hlow: 0.01,
            hmax: 5.0,
            max_component: 1.0,
            micro_cycles: 20,
            trust_radius: None,
            trust_max: 1.0,
            energy_rise: 1e-6,
        }
    }

    /// Length of the coordinate vector this optimiser relaxes.
    pub fn dim(&self) -> usize {
        3 * self.species.len()
    }

    /// Relaxes `x0`, calling `fg` for value and gradient.
    ///
    /// Same contract as [`crate::methods::warm_lbfgs::WarmLbfgs::minimize`]:
    /// `fg` returns `None` once the caller's budget is spent, and the result
    /// is the value, the point, and the evaluations used.
    pub fn minimize<F>(
        &self,
        x0: ArrayView1<f64>,
        max_iter: usize,
        fg: F,
    ) -> (f64, Array1<f64>, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        self.minimize_watched(x0, max_iter, fg, |_, _| true)
    }

    /// Relaxes `x0`, offering each accepted iterate to `watch`.
    ///
    /// `watch` receives the step index and the value there; `false` ends the
    /// relaxation at that point. `max_iter` bounds the steps, each of which is
    /// one evaluation, after the evaluation at `x0`. The point returned is the
    /// converged one, or the one `watch` stopped at; otherwise, when the
    /// steps or the budget run out, the lowest one evaluated.
    pub fn minimize_watched<F, W>(
        &self,
        x0: ArrayView1<f64>,
        max_iter: usize,
        mut fg: F,
        mut watch: W,
    ) -> (f64, Array1<f64>, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
        W: FnMut(usize, f64) -> bool,
    {
        assert_eq!(x0.len(), self.dim(), "coordinates do not match the species");
        let mut evals = 0usize;
        let Some((f0, g0)) = fg(x0) else {
            return (f64::INFINITY, x0.to_owned(), evals);
        };
        evals += 1;
        let mut best = (f0, x0.to_owned());
        if !f0.is_finite() || self.converged(g0.view()) || self.species.len() < 2 {
            return (best.0, best.1, evals);
        }

        let e_unit = self.units.energy_per_hartree;
        let l_unit = self.units.length_per_bohr;
        let to_au_grad = l_unit / e_unit;

        let mut x = x0.to_owned();
        let mut f = f0;
        let mut g = g0;
        let mut steps = 0usize;
        let base_micro = self.micro_cycles.max(1);
        let mut micro = base_micro;
        let mut trust = self.trust_radius;

        'rebuild: loop {
            let anc = Anc::build(self, x.view());
            let nvar = anc.nvar;
            if nvar == 0 {
                return (best.0, best.1, evals);
            }
            let mut hq = vec![0.0; nvar * nvar];
            for i in 0..nvar {
                hq[i * nvar + i] = anc.eigenvalues[i];
            }
            let x_ref = x.clone();
            let mut q = vec![0.0; nvar];
            let mut gq = anc.project(g.view(), to_au_grad);

            let cycles = micro;
            for cycle in 0..cycles {
                if steps >= max_iter {
                    return (best.0, best.1, evals);
                }
                let mut dq = rfo_step(&hq, &gq, nvar, self.hlow);
                for v in dq.iter_mut() {
                    *v = v.clamp(-self.max_component, self.max_component);
                }
                let mut norm = dot(&dq, &dq).sqrt();
                if trust.is_none() && cycle > 1 && norm > 2.0 {
                    micro = next_micro(micro, base_micro);
                    continue 'rebuild;
                }
                if let Some(radius) = trust
                    && norm > radius
                {
                    let s = radius / norm;
                    for v in dq.iter_mut() {
                        *v *= s;
                    }
                    norm = radius;
                }
                let gnorm = dot(&gq, &gq).sqrt();
                if trust.is_none() && gnorm < 0.002 {
                    // xtb lengthens the step by half once the gradient is
                    // small, where the floored model overstates the curvature.
                    for v in dq.iter_mut() {
                        *v *= 1.5;
                    }
                }
                let predicted = dot(&gq, &dq) + 0.5 * quad(&hq, &dq, nvar);

                let mut q_trial = q.clone();
                for (a, b) in q_trial.iter_mut().zip(&dq) {
                    *a += b;
                }
                let x_trial = anc.cartesian(x_ref.view(), &q_trial, l_unit);
                let Some((f_trial, g_trial)) = fg(x_trial.view()) else {
                    return (best.0, best.1, evals);
                };
                evals += 1;
                steps += 1;
                if !f_trial.is_finite() {
                    match trust.as_mut() {
                        Some(radius) => {
                            *radius = (0.25 * norm).max(1e-4);
                            continue;
                        }
                        None => return (best.0, best.1, evals),
                    }
                }
                let actual = (f_trial - f) / e_unit;
                if let Some(radius) = trust.as_mut()
                    && !judge_step(radius, norm, predicted, actual, self)
                {
                    continue;
                }
                let gq_trial = anc.project(g_trial.view(), to_au_grad);
                let y: Vec<f64> = gq_trial.iter().zip(&gq).map(|(a, b)| a - b).collect();
                match self.update {
                    HessianUpdate::Bfgs => bfgs(&mut hq, &dq, &y, nvar, self.hlow),
                    HessianUpdate::Bofill => bofill(&mut hq, &dq, &y, nvar),
                }
                q = q_trial;
                x = x_trial;
                f = f_trial;
                g = g_trial;
                gq = gq_trial;
                if f < best.0 {
                    best = (f, x.clone());
                }
                if self.converged(g.view()) {
                    return (f, x, evals);
                }
                if !watch(steps - 1, f) {
                    return (f, x, evals);
                }
            }
            micro = next_micro(micro, base_micro);
        }
    }

    fn converged(&self, g: ArrayView1<f64>) -> bool {
        let mut largest = 0.0_f64;
        for (i, v) in g.iter().enumerate() {
            if self.is_frozen(i / 3) {
                continue;
            }
            largest = largest.max(v.abs());
        }
        largest < self.gtol
    }

    fn is_frozen(&self, atom: usize) -> bool {
        self.frozen
            .as_ref()
            .is_some_and(|mask| mask.get(atom).copied().unwrap_or(false))
    }
}

/// Trust-radius verdict on a step of length `norm` bohr whose energy change
/// was `actual` hartree against the model's `predicted`; `true` keeps it.
///
/// A rejected step sets the radius to a quarter of its length. A kept one
/// sets it to half its length on a ratio below 0.25, and doubles it, up to
/// `trust_max`, on a ratio above 0.75 when the step reached the radius. A
/// model that predicted no descent gives no ratio, and the radius stays.
fn judge_step(radius: &mut f64, norm: f64, predicted: f64, actual: f64, opt: &AncOpt) -> bool {
    if actual > opt.energy_rise {
        *radius = (0.25 * norm).max(1e-4);
        return false;
    }
    if predicted < 0.0 {
        let ratio = actual / predicted;
        if ratio < 0.25 {
            *radius = (0.5 * norm).max(1e-4);
        } else if ratio > 0.75 && norm > 0.8 * *radius {
            *radius = (2.0 * *radius).min(opt.trust_max);
        }
    }
    true
}

fn next_micro(micro: usize, base: usize) -> usize {
    ((micro as f64 * 1.1) as usize).max(micro + 1).min(2 * base)
}

/// One set of approximate normal coordinates.
struct Anc {
    /// Coordinates, column `k` of a `3n x nvar` row-major matrix.
    basis: Vec<f64>,
    eigenvalues: Vec<f64>,
    n3: usize,
    nvar: usize,
}

impl Anc {
    fn build(opt: &AncOpt, x: ArrayView1<f64>) -> Self {
        let n = opt.species.len();
        let n3 = 3 * n;
        let x_au: Vec<f64> = x.iter().map(|v| v / opt.units.length_per_bohr).collect();
        let h = model_hessian(&x_au, &opt.species, opt.model);
        let q = complement_basis(opt, &x_au);
        let nvar = q.len() / n3.max(1);
        if nvar == 0 {
            return Self {
                basis: Vec::new(),
                eigenvalues: Vec::new(),
                n3,
                nvar: 0,
            };
        }
        // Hs = Q^T H Q, through HQ.
        let mut hq = vec![0.0; n3 * nvar];
        for r in 0..n3 {
            let row = &h[r * n3..(r + 1) * n3];
            for c in 0..nvar {
                let mut s = 0.0;
                for k in 0..n3 {
                    s += row[k] * q[k * nvar + c];
                }
                hq[r * nvar + c] = s;
            }
        }
        let mut hs = vec![0.0; nvar * nvar];
        for a in 0..nvar {
            for b in a..nvar {
                let mut s = 0.0;
                for k in 0..n3 {
                    s += q[k * nvar + a] * hq[k * nvar + b];
                }
                hs[a * nvar + b] = s;
                hs[b * nvar + a] = s;
            }
        }
        let (mut values, vectors) = symmetric_eigen(&hs, nvar);
        let lowest = values.iter().copied().fold(f64::INFINITY, f64::min);
        let shift = (opt.hlow - lowest).max(0.0);
        for v in values.iter_mut() {
            *v = (*v + shift).clamp(opt.hlow, opt.hmax);
        }
        let mut basis = vec![0.0; n3 * nvar];
        for r in 0..n3 {
            for c in 0..nvar {
                let mut s = 0.0;
                for k in 0..nvar {
                    s += q[r * nvar + k] * vectors[k * nvar + c];
                }
                basis[r * nvar + c] = s;
            }
        }
        Self {
            basis,
            eigenvalues: values,
            n3,
            nvar,
        }
    }

    /// Coordinate gradient in hartree per bohr.
    fn project(&self, g: ArrayView1<f64>, to_au: f64) -> Vec<f64> {
        let mut out = vec![0.0; self.nvar];
        for r in 0..self.n3 {
            let gr = g[r] * to_au;
            if gr == 0.0 {
                continue;
            }
            for c in 0..self.nvar {
                out[c] += self.basis[r * self.nvar + c] * gr;
            }
        }
        out
    }

    fn cartesian(&self, x_ref: ArrayView1<f64>, q: &[f64], length_per_bohr: f64) -> Array1<f64> {
        let mut x = x_ref.to_owned();
        for r in 0..self.n3 {
            let mut s = 0.0;
            for c in 0..self.nvar {
                s += self.basis[r * self.nvar + c] * q[c];
            }
            x[r] += s * length_per_bohr;
        }
        x
    }
}

/// Orthonormal basis, `3n x nvar` row-major, of the coordinates that move.
///
/// Without frozen atoms that is the complement of the rigid translations and
/// rotations about the centroid; five rigid modes for a linear molecule. With
/// frozen atoms it is the mobile Cartesian coordinates themselves.
fn complement_basis(opt: &AncOpt, x_au: &[f64]) -> Vec<f64> {
    let n = opt.species.len();
    let n3 = 3 * n;
    if opt.frozen.as_ref().is_some_and(|m| m.iter().any(|&f| f)) {
        let mobile: Vec<usize> = (0..n3).filter(|&r| !opt.is_frozen(r / 3)).collect();
        let nvar = mobile.len();
        let mut q = vec![0.0; n3 * nvar];
        for (c, &r) in mobile.iter().enumerate() {
            q[r * nvar + c] = 1.0;
        }
        return q;
    }
    let mut centre = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            centre[k] += x_au[3 * i + k] / n as f64;
        }
    }
    let mut rigid: Vec<Vec<f64>> = Vec::with_capacity(6);
    for k in 0..3 {
        let mut t = vec![0.0; n3];
        for i in 0..n {
            t[3 * i + k] = 1.0;
        }
        rigid.push(t);
    }
    for axis in 0..3 {
        let mut e = [0.0; 3];
        e[axis] = 1.0;
        let mut t = vec![0.0; n3];
        for i in 0..n {
            let r = [
                x_au[3 * i] - centre[0],
                x_au[3 * i + 1] - centre[1],
                x_au[3 * i + 2] - centre[2],
            ];
            let c = cross(e, r);
            t[3 * i..3 * i + 3].copy_from_slice(&c);
        }
        rigid.push(t);
    }
    let mut kept: Vec<Vec<f64>> = Vec::with_capacity(n3);
    let scale = (n as f64).sqrt();
    for v in rigid {
        if let Some(u) = orthonormalise(v, &kept, 1e-6 * scale) {
            kept.push(u);
        }
    }
    let n_rigid = kept.len();
    // Unit vectors in order of how much of them lies outside the span so far,
    // so the complement is taken from the best-conditioned candidates.
    let mut candidates: Vec<(f64, usize)> = (0..n3)
        .map(|r| {
            let overlap: f64 = kept.iter().map(|u| u[r] * u[r]).sum();
            (overlap, r)
        })
        .collect();
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for (_, r) in candidates {
        if kept.len() == n3 {
            break;
        }
        let mut v = vec![0.0; n3];
        v[r] = 1.0;
        if let Some(u) = orthonormalise(v, &kept, 1e-3) {
            kept.push(u);
        }
    }
    let nvar = kept.len() - n_rigid;
    let mut q = vec![0.0; n3 * nvar];
    for (c, u) in kept[n_rigid..].iter().enumerate() {
        for r in 0..n3 {
            q[r * nvar + c] = u[r];
        }
    }
    q
}

/// Twice-applied Gram-Schmidt against `basis`; `None` when the residual is
/// shorter than `tol`.
fn orthonormalise(mut v: Vec<f64>, basis: &[Vec<f64>], tol: f64) -> Option<Vec<f64>> {
    for _ in 0..2 {
        for u in basis {
            let p = dot(&v, u);
            for (a, b) in v.iter_mut().zip(u) {
                *a -= p * b;
            }
        }
    }
    let norm = dot(&v, &v).sqrt();
    if norm < tol {
        return None;
    }
    for a in v.iter_mut() {
        *a /= norm;
    }
    Some(v)
}

/// Rational function step for coordinate Hessian `h` and gradient `g`.
fn rfo_step(h: &[f64], g: &[f64], n: usize, hlow: f64) -> Vec<f64> {
    let m = n + 1;
    let mut a = vec![0.0; m * m];
    for i in 0..n {
        a[i * m..i * m + n].copy_from_slice(&h[i * n..(i + 1) * n]);
        a[i * m + n] = g[i];
        a[n * m + i] = g[i];
    }
    let (_, vectors) = symmetric_eigen(&a, m);
    let tail = vectors[n * m];
    if tail.abs() < 1e-10 {
        return g.iter().map(|v| -v / hlow).collect();
    }
    (0..n).map(|i| vectors[i * m] / tail).collect()
}

fn bfgs(h: &mut [f64], s: &[f64], y: &[f64], n: usize, floor: f64) {
    let hs = matvec(h, s, n);
    let shs = dot(s, &hs);
    let sy = dot(s, y);
    if sy > 1e-12 && shs > 1e-12 {
        for i in 0..n {
            for j in 0..n {
                h[i * n + j] += y[i] * y[j] / sy - hs[i] * hs[j] / shs;
            }
        }
    }
    for i in 0..n {
        if h[i * n + i].abs() < floor {
            h[i * n + i] = floor;
        }
    }
}

fn bofill(h: &mut [f64], s: &[f64], y: &[f64], n: usize) {
    let hs = matvec(h, s, n);
    let xi: Vec<f64> = y.iter().zip(&hs).map(|(a, b)| a - b).collect();
    let ss = dot(s, s);
    let xs = dot(&xi, s);
    let xx = dot(&xi, &xi);
    if ss < 1e-14 || xx < 1e-28 {
        return;
    }
    let phi = if xs.abs() < 1e-300 {
        0.0
    } else {
        xs * xs / (xx * ss)
    };
    for i in 0..n {
        for j in 0..n {
            let psb = (xi[i] * s[j] + s[i] * xi[j]) / ss - xs * s[i] * s[j] / (ss * ss);
            let sr1 = if xs.abs() > 1e-12 {
                xi[i] * xi[j] / xs
            } else {
                0.0
            };
            h[i * n + j] += phi * sr1 + (1.0 - phi) * psb;
        }
    }
}

fn matvec(h: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            h[i * n..(i + 1) * n]
                .iter()
                .zip(v)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

fn quad(h: &[f64], v: &[f64], n: usize) -> f64 {
    dot(v, &matvec(h, v, n))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Row of the periodic table in Lindh's three-class sense.
fn lindh_row(z: u32) -> usize {
    match z {
        0..=2 => 0,
        3..=10 => 1,
        _ => 2,
    }
}

/// Lindh reference distances, bohr, by row pair.
const LINDH_R: [[f64; 3]; 3] = [[1.35, 2.10, 2.53], [2.10, 2.87, 3.40], [2.53, 3.40, 3.40]];
/// Lindh exponents, bohr^-2, by row pair.
const LINDH_A: [[f64; 3]; 3] = [
    [1.0, 0.3949, 0.3949],
    [0.3949, 0.28, 0.28],
    [0.3949, 0.28, 0.28],
];
const K_STRETCH: f64 = 0.40;
const K_BEND: f64 = 0.13;
const K_TORSION: f64 = 0.0075;
/// Squared separation, bohr^2, beyond which a pair joins no bend or torsion.
const CUTOFF2: f64 = 70.0;
/// Scale on the C6 pair term of the stretch.
const S6: f64 = 20.0;
/// Steepness of the C6 damping.
const VDW_STEEPNESS: f64 = 20.0;

/// Grimme D2 C6 coefficients, J nm^6 mol^-1, H to Xe, used unconverted the
/// way the xtb model Hessian uses them.
fn c6(z: u32) -> f64 {
    const C6: [f64; 54] = [
        0.14, 0.08, 1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75, 0.63, 5.71, 5.71, 10.79, 9.23, 7.84,
        5.57, 5.07, 4.61, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8,
        16.99, 17.10, 16.37, 12.64, 12.47, 12.01, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67,
        24.67, 24.67, 24.67, 24.67, 24.67, 37.32, 38.71, 38.44, 31.74, 31.50, 29.99,
    ];
    match z {
        1..=54 => C6[z as usize - 1],
        _ => 50.0,
    }
}

/// Grimme D2 van der Waals radii, angstrom, H to Xe.
fn vdw_radius(z: u32) -> f64 {
    const R: [f64; 54] = [
        0.91, 0.92, 0.75, 1.28, 1.35, 1.32, 1.27, 1.22, 1.17, 1.13, 1.04, 1.24, 1.49, 1.56, 1.55,
        1.53, 1.49, 1.45, 1.35, 1.34, 1.42, 1.42, 1.42, 1.42, 1.42, 1.42, 1.42, 1.42, 1.42, 1.42,
        1.50, 1.57, 1.60, 1.61, 1.59, 1.57, 1.48, 1.46, 1.49, 1.49, 1.49, 1.49, 1.49, 1.49, 1.49,
        1.49, 1.49, 1.49, 1.52, 1.64, 1.71, 1.72, 1.72, 1.71,
    ];
    match z {
        1..=54 => R[z as usize - 1],
        _ => 2.0,
    }
}

/// Weight of a pair in the model: `exp(alpha (r_ref^2 - r^2))` for Lindh,
/// `exp(1 - r / (R_i + R_j))` for Swart.
fn pair_weight(model: ModelHessian, zi: u32, zj: u32, r2: f64) -> f64 {
    match model {
        ModelHessian::Lindh => {
            let (a, b) = (lindh_row(zi), lindh_row(zj));
            let r0 = LINDH_R[a][b];
            LINDH_A[a][b] * (r0 * r0 - r2)
        }
        ModelHessian::Swart => {
            let rc = (crate::methods::cluster_hopping::covalent_radius(zi)
                + crate::methods::cluster_hopping::covalent_radius(zj))
                / BOHR_ANGSTROM;
            let rc = if rc > 0.0 { rc } else { 3.0 };
            1.0 - r2.sqrt() / rc
        }
    }
    .exp()
}

/// Radial derivatives `(V'(r), V''(r))` of the damped attraction
/// `V = -s6 c6 f(r) / r^6`, `f = 1 / (1 + exp(-a (r / R0 - 1)))`.
///
/// The model subtracts the Cartesian Hessian built from these.
fn dispersion_derivatives(c: f64, r0: f64, r: f64) -> (f64, f64) {
    let a = VDW_STEEPNESS / r0;
    let e = (-a * (r - r0)).exp();
    let f = 1.0 / (1.0 + e);
    let f1 = a * e * f * f;
    let f2 = a * a * e * f * f * (2.0 * e * f - 1.0);
    let g = r.powi(-6);
    let g1 = -6.0 * r.powi(-7);
    let g2 = 42.0 * r.powi(-8);
    let v1 = -c * (g1 * f + g * f1);
    let v2 = -c * (g2 * f + 2.0 * g1 * f1 + g * f2);
    (v1, v2)
}

/// Adds `k b b^T` over the atoms `atoms`, `b` holding three entries per atom.
fn add_outer(h: &mut [f64], n3: usize, atoms: &[usize], b: &[[f64; 3]], k: f64) {
    for (ia, &a) in atoms.iter().enumerate() {
        for (ib, &bb) in atoms.iter().enumerate() {
            for p in 0..3 {
                let row = (3 * a + p) * n3 + 3 * bb;
                for q in 0..3 {
                    h[row + q] += k * b[ia][p] * b[ib][q];
                }
            }
        }
    }
}

/// Dense Cartesian model Hessian, hartree per bohr squared, `3n x 3n`
/// row-major, at the geometry `x` in bohr.
pub fn model_hessian(x: &[f64], z: &[u32], model: ModelHessian) -> Vec<f64> {
    let n = z.len();
    let n3 = 3 * n;
    assert_eq!(x.len(), n3, "coordinates do not match the species");
    let mut h = vec![0.0; n3 * n3];
    let pos = |i: usize| [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let norm2 = |v: [f64; 3]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2];

    let mut r2 = vec![0.0; n * n];
    let mut weight = vec![0.0; n * n];
    let mut near: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d2 = norm2(sub(pos(i), pos(j)));
            r2[i * n + j] = d2;
            weight[i * n + j] = pair_weight(model, z[i], z[j], d2);
            if d2 <= CUTOFF2 {
                near[i].push(j);
            }
        }
    }
    let close = |i: usize, j: usize| r2[i * n + j] <= CUTOFF2;

    // Stretch over every pair, less the Hessian of the damped C6 attraction.
    for i in 0..n {
        for j in 0..i {
            let d = sub(pos(i), pos(j));
            let d2 = r2[i * n + j];
            if d2 < 1e-20 {
                continue;
            }
            let r = d2.sqrt();
            let u = [d[0] / r, d[1] / r, d[2] / r];
            let k = K_STRETCH * weight[i * n + j];
            let c = S6 * (c6(z[i]) * c6(z[j])).sqrt();
            let rv = (vdw_radius(z[i]) + vdw_radius(z[j])) / BOHR_ANGSTROM;
            let (v1, v2) = dispersion_derivatives(c, rv, r);
            for p in 0..3 {
                for q in 0..3 {
                    let delta = if p == q { 1.0 } else { 0.0 };
                    let vdw = v2 * u[p] * u[q] + v1 / r * (delta - u[p] * u[q]);
                    let block = k * u[p] * u[q] - vdw;
                    h[(3 * i + p) * n3 + 3 * i + q] += block;
                    h[(3 * j + p) * n3 + 3 * j + q] += block;
                    h[(3 * i + p) * n3 + 3 * j + q] -= block;
                    h[(3 * j + p) * n3 + 3 * i + q] -= block;
                }
            }
        }
    }

    // Bends i-m-j about every centre m.
    for m in 0..n {
        for &i in &near[m] {
            for &j in &near[m] {
                if j >= i || !close(i, j) {
                    continue;
                }
                let a = sub(pos(i), pos(m));
                let b = sub(pos(j), pos(m));
                let ra = r2[i * n + m].sqrt();
                let rb = r2[j * n + m].sqrt();
                if ra < 1e-10 || rb < 1e-10 {
                    continue;
                }
                let k = K_BEND * weight[m * n + i] * weight[m * n + j];
                let ea = [a[0] / ra, a[1] / ra, a[2] / ra];
                let eb = [b[0] / rb, b[1] / rb, b[2] / rb];
                let cos = (ea[0] * eb[0] + ea[1] * eb[1] + ea[2] * eb[2]).clamp(-1.0, 1.0);
                let sin = norm2(cross(ea, eb)).sqrt();
                if sin > 1e-10 {
                    let mut bi = [0.0; 3];
                    let mut bj = [0.0; 3];
                    for p in 0..3 {
                        bi[p] = (ea[p] * cos - eb[p]) / (ra * sin);
                        bj[p] = (eb[p] * cos - ea[p]) / (rb * sin);
                    }
                    let bm = [-bi[0] - bj[0], -bi[1] - bj[1], -bi[2] - bj[2]];
                    add_outer(&mut h, n3, &[i, m, j], &[bi, bm, bj], k);
                } else if cos < 0.0 {
                    // Linear: two bends in the planes perpendicular to the axis.
                    let helper = if ea[0].abs() < 0.9 {
                        [1.0, 0.0, 0.0]
                    } else {
                        [0.0, 1.0, 0.0]
                    };
                    let w1 = cross(ea, helper);
                    let n1 = norm2(w1).sqrt();
                    let w1 = [w1[0] / n1, w1[1] / n1, w1[2] / n1];
                    let w2 = cross(ea, w1);
                    for w in [w1, w2] {
                        let bi = [-w[0] / ra, -w[1] / ra, -w[2] / ra];
                        let bj = [-w[0] / rb, -w[1] / rb, -w[2] / rb];
                        let bm = [-bi[0] - bj[0], -bi[1] - bj[1], -bi[2] - bj[2]];
                        add_outer(&mut h, n3, &[i, m, j], &[bi, bm, bj], k);
                    }
                }
            }
        }
    }

    // Torsions i-j-k-l about every axis j-k, each counted once.
    let cos_max = (35.0_f64).to_radians().cos();
    for j in 0..n {
        for &k in &near[j] {
            for &i in &near[j] {
                if i == k || !close(i, k) {
                    continue;
                }
                for &l in &near[k] {
                    if l == i || l == j || !close(l, i) || !close(l, j) {
                        continue;
                    }
                    if n * j + i <= n * k + l {
                        continue;
                    }
                    let f = sub(pos(i), pos(j));
                    let g = sub(pos(j), pos(k));
                    let hv = sub(pos(l), pos(k));
                    let rf2 = norm2(f);
                    let rg2 = norm2(g);
                    let rh2 = norm2(hv);
                    let c2 = (f[0] * g[0] + f[1] * g[1] + f[2] * g[2]) / (rf2 * rg2).sqrt();
                    let c3 = (hv[0] * g[0] + hv[1] * g[1] + hv[2] * g[2]) / (rh2 * rg2).sqrt();
                    if c2.abs() > cos_max || c3.abs() > cos_max {
                        continue;
                    }
                    let t = K_TORSION * weight[i * n + j] * weight[j * n + k] * weight[k * n + l];
                    let a = cross(f, g);
                    let b = cross(hv, g);
                    let a2 = norm2(a);
                    let b2 = norm2(b);
                    if a2 < 1e-20 || b2 < 1e-20 {
                        continue;
                    }
                    let rg = rg2.sqrt();
                    let fg = (f[0] * g[0] + f[1] * g[1] + f[2] * g[2]) / (a2 * rg);
                    let hg = (hv[0] * g[0] + hv[1] * g[1] + hv[2] * g[2]) / (b2 * rg);
                    let mut bi = [0.0; 3];
                    let mut bj = [0.0; 3];
                    let mut bk = [0.0; 3];
                    let mut bl = [0.0; 3];
                    for p in 0..3 {
                        bi[p] = -rg / a2 * a[p];
                        bl[p] = rg / b2 * b[p];
                        bj[p] = rg / a2 * a[p] + fg * a[p] - hg * b[p];
                        bk[p] = -rg / b2 * b[p] - fg * a[p] + hg * b[p];
                    }
                    add_outer(&mut h, n3, &[i, j, k, l], &[bi, bj, bk, bl], t);
                }
            }
        }
    }
    h
}

/// Eigenvalues ascending and eigenvectors (column `k` of a row-major `n x n`
/// matrix) of the symmetric `a`.
///
/// Householder reduction to tridiagonal form with the transformations
/// accumulated, then the implicitly shifted QL iteration (Wilkinson and
/// Reinsch, Handbook for Automatic Computation II, contributions II/2 and
/// II/3, 1971), in the arrangement of the public-domain JAMA `tred2` and
/// `tql2`. A relaxation step solves one of these at the coordinate count, and
/// cyclic Jacobi ([`crate::spectral::symmetric_eigen`]) is several times
/// slower there.
pub fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), n * n, "symmetric_eigen needs an n x n matrix");
    let mut v = a.to_vec();
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    if n == 0 {
        return (d, v);
    }
    if n == 1 {
        return (vec![a[0]], vec![1.0]);
    }
    let at = |v: &Vec<f64>, i: usize, j: usize| v[i * n + j];

    for j in 0..n {
        d[j] = at(&v, n - 1, j);
    }
    for i in (1..n).rev() {
        let mut scale = 0.0;
        let mut h = 0.0;
        for k in 0..i {
            scale += d[k].abs();
        }
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = at(&v, i - 1, j);
                v[i * n + j] = 0.0;
                v[j * n + i] = 0.0;
            }
        } else {
            for k in 0..i {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for item in e.iter_mut().take(i) {
                *item = 0.0;
            }
            for j in 0..i {
                f = d[j];
                v[j * n + i] = f;
                g = e[j] + at(&v, j, j) * f;
                for k in (j + 1)..i {
                    g += at(&v, k, j) * d[k];
                    e[k] += at(&v, k, j) * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..i {
                    v[k * n + j] -= f * e[k] + g * d[k];
                }
                d[j] = at(&v, i - 1, j);
                v[i * n + j] = 0.0;
            }
        }
        d[i] = h;
    }
    for i in 0..(n - 1) {
        v[(n - 1) * n + i] = at(&v, i, i);
        v[i * n + i] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = at(&v, k, i + 1) / h;
            }
            for j in 0..=i {
                let mut g = 0.0;
                for k in 0..=i {
                    g += at(&v, k, i + 1) * at(&v, k, j);
                }
                for k in 0..=i {
                    v[k * n + j] -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            v[k * n + i + 1] = 0.0;
        }
    }
    for j in 0..n {
        d[j] = at(&v, n - 1, j);
        v[(n - 1) * n + j] = 0.0;
    }
    v[(n - 1) * n + n - 1] = 1.0;
    e[0] = 0.0;

    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    let mut f = 0.0;
    let mut tst1 = 0.0_f64;
    let eps = f64::EPSILON;
    for l in 0..n {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n - 1 {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }
        if m > l {
            let mut iter = 0;
            loop {
                iter += 1;
                let mut g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for item in d.iter_mut().skip(l + 2) {
                    *item -= h;
                }
                f += h;
                p = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for k in 0..n {
                        h = v[k * n + i + 1];
                        v[k * n + i + 1] = s * v[k * n + i] + c * h;
                        v[k * n + i] = c * v[k * n + i] - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= eps * tst1 || iter >= 64 || !e[l].is_finite() {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
    for i in 0..(n - 1) {
        let mut k = i;
        let mut p = d[i];
        for (j, &dj) in d.iter().enumerate().skip(i + 1) {
            if dj < p {
                k = j;
                p = dj;
            }
        }
        if k != i {
            d[k] = d[i];
            d[i] = p;
            for j in 0..n {
                v.swap(j * n + i, j * n + k);
            }
        }
    }
    (d, v)
}

/// Small valence force fields in electronvolt and angstrom, for tests.
#[cfg(test)]
pub(crate) mod test_ff {
    use ndarray::{Array1, ArrayView1};

    /// Harmonic bonds and angles, cosine torsions, and Lennard-Jones plus
    /// Coulomb between atoms more than two bonds apart.
    pub(crate) struct ValenceFf {
        pub species: Vec<u32>,
        pub bonds: Vec<(usize, usize, f64, f64)>,
        pub angles: Vec<(usize, usize, usize, f64, f64)>,
        pub torsions: Vec<(usize, usize, usize, usize, f64, f64)>,
        pub charges: Vec<f64>,
        pub lj: Vec<(f64, f64)>,
        excluded: Vec<bool>,
    }

    const COULOMB: f64 = 14.399_645;

    impl ValenceFf {
        fn finish(mut self) -> Self {
            let n = self.species.len();
            let mut excluded = vec![false; n * n];
            let mut adj = vec![Vec::new(); n];
            for &(a, b, _, _) in &self.bonds {
                adj[a].push(b);
                adj[b].push(a);
            }
            for a in 0..n {
                excluded[a * n + a] = true;
                for &b in &adj[a] {
                    excluded[a * n + b] = true;
                    for &c in &adj[b] {
                        excluded[a * n + c] = true;
                    }
                }
            }
            self.excluded = excluded;
            self
        }

        /// One flexible water per molecule, `n_mol` of them.
        pub(crate) fn waters(n_mol: usize) -> Self {
            let mut ff = Self {
                species: Vec::new(),
                bonds: Vec::new(),
                angles: Vec::new(),
                torsions: Vec::new(),
                charges: Vec::new(),
                lj: Vec::new(),
                excluded: Vec::new(),
            };
            for m in 0..n_mol {
                let o = 3 * m;
                ff.species.extend([8, 1, 1]);
                ff.charges.extend([-0.82, 0.41, 0.41]);
                ff.lj.extend([(0.006_74, 3.166), (0.0, 1.0), (0.0, 1.0)]);
                ff.bonds.push((o, o + 1, 45.9, 1.0));
                ff.bonds.push((o, o + 2, 45.9, 1.0));
                ff.angles
                    .push((o + 1, o, o + 2, 3.29, 109.47_f64.to_radians()));
            }
            ff.finish()
        }

        /// All-atom n-pentane, `CH3(CH2)3CH3`, 17 atoms.
        pub(crate) fn pentane() -> Self {
            let mut species = vec![6; 5];
            species.extend(vec![1; 12]);
            let mut bonds = Vec::new();
            for c in 0..4 {
                bonds.push((c, c + 1, 26.0, 1.53));
            }
            let mut h = 5;
            let mut hydrogens = vec![Vec::new(); 5];
            for (c, count) in [(0, 3), (1, 2), (2, 2), (3, 2), (4, 3)] {
                for _ in 0..count {
                    bonds.push((c, h, 29.0, 1.09));
                    hydrogens[c].push(h);
                    h += 1;
                }
            }
            let mut adj = vec![Vec::new(); 17];
            for &(a, b, _, _) in &bonds {
                adj[a].push(b);
                adj[b].push(a);
            }
            let mut angles = Vec::new();
            for centre in 0..17 {
                let nb = &adj[centre];
                for x in 0..nb.len() {
                    for y in (x + 1)..nb.len() {
                        angles.push((nb[x], centre, nb[y], 3.4, 109.5_f64.to_radians()));
                    }
                }
            }
            let mut torsions = Vec::new();
            for c in 0..4 {
                let b = c + 1;
                for &a in adj[c].iter().filter(|&&a| a != b) {
                    for &d in adj[b].iter().filter(|&&d| d != c) {
                        let k = if a < 5 && d < 5 { 0.06 } else { 0.007 };
                        torsions.push((a, c, b, d, k, 0.0));
                    }
                }
            }
            let mut charges = vec![0.0; 17];
            for (c, hs) in hydrogens.iter().enumerate() {
                charges[c] = -0.1 * hs.len() as f64;
                for &hh in hs {
                    charges[hh] = 0.1;
                }
            }
            let lj = species
                .iter()
                .map(|&z| if z == 6 { (0.0028, 3.4) } else { (0.0013, 2.6) })
                .collect();
            Self {
                species,
                bonds,
                angles,
                torsions,
                charges,
                lj,
                excluded: Vec::new(),
            }
            .finish()
        }

        pub(crate) fn energy_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
            let n = self.species.len();
            let mut e = 0.0;
            let mut g = Array1::<f64>::zeros(3 * n);
            let p = |i: usize| [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
            let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
            let dotp = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            for &(a, b, k, r0) in &self.bonds {
                let d = sub(p(a), p(b));
                let r = dotp(d, d).sqrt();
                e += 0.5 * k * (r - r0) * (r - r0);
                let c = k * (r - r0) / r;
                for q in 0..3 {
                    g[3 * a + q] += c * d[q];
                    g[3 * b + q] -= c * d[q];
                }
            }
            for &(a, m, b, k, t0) in &self.angles {
                let u = sub(p(a), p(m));
                let v = sub(p(b), p(m));
                let ru = dotp(u, u).sqrt();
                let rv = dotp(v, v).sqrt();
                let cos = (dotp(u, v) / (ru * rv)).clamp(-1.0, 1.0);
                let theta = cos.acos();
                let sin = (1.0 - cos * cos).sqrt().max(1e-8);
                e += 0.5 * k * (theta - t0) * (theta - t0);
                let de = k * (theta - t0);
                for q in 0..3 {
                    let da = (u[q] / ru * cos - v[q] / rv) / (ru * sin);
                    let db = (v[q] / rv * cos - u[q] / ru) / (rv * sin);
                    g[3 * a + q] += de * da;
                    g[3 * b + q] += de * db;
                    g[3 * m + q] -= de * (da + db);
                }
            }
            for &(i, j, k, l, kt, phase) in &self.torsions {
                // V = kt (1 + cos(3 phi - phase)), Blondel-Karplus gradient.
                let f = sub(p(i), p(j));
                let gg = sub(p(j), p(k));
                let h = sub(p(l), p(k));
                let a = super::cross(f, gg);
                let b = super::cross(h, gg);
                let a2 = dotp(a, a);
                let b2 = dotp(b, b);
                let rg = dotp(gg, gg).sqrt();
                let cos = dotp(a, b) / (a2 * b2).sqrt();
                let sin = dotp(super::cross(b, a), gg) / ((a2 * b2).sqrt() * rg);
                let phi = sin.atan2(cos);
                e += kt * (1.0 + (3.0 * phi - phase).cos());
                let de = -3.0 * kt * (3.0 * phi - phase).sin();
                let fg = dotp(f, gg) / (a2 * rg);
                let hg = dotp(h, gg) / (b2 * rg);
                for q in 0..3 {
                    let di = -rg / a2 * a[q];
                    let dl = rg / b2 * b[q];
                    let dj = rg / a2 * a[q] + fg * a[q] - hg * b[q];
                    let dk = -rg / b2 * b[q] - fg * a[q] + hg * b[q];
                    g[3 * i + q] += de * di;
                    g[3 * j + q] += de * dj;
                    g[3 * k + q] += de * dk;
                    g[3 * l + q] += de * dl;
                }
            }
            for a in 0..n {
                for b in (a + 1)..n {
                    if self.excluded[a * n + b] {
                        continue;
                    }
                    let d = sub(p(a), p(b));
                    let r2 = dotp(d, d);
                    let r = r2.sqrt();
                    let (ea, sa) = self.lj[a];
                    let (eb, sb) = self.lj[b];
                    let eps = (ea * eb).sqrt();
                    let sig = 0.5 * (sa + sb);
                    let sr6 = (sig * sig / r2).powi(3);
                    let qq = COULOMB * self.charges[a] * self.charges[b];
                    e += 4.0 * eps * (sr6 * sr6 - sr6) + qq / r;
                    let dedr = 4.0 * eps * (-12.0 * sr6 * sr6 + 6.0 * sr6) / r - qq / r2;
                    for q in 0..3 {
                        g[3 * a + q] += dedr * d[q] / r;
                        g[3 * b + q] -= dedr * d[q] / r;
                    }
                }
            }
            (e, g)
        }
    }

    /// Zig-zag pentane with every hydrogen pushed off its tetrahedral site.
    pub(crate) fn pentane_start(seed: u64) -> Array1<f64> {
        let mut x = vec![0.0; 51];
        for c in 0..5 {
            x[3 * c] = 1.27 * c as f64;
            x[3 * c + 1] = if c % 2 == 0 { 0.0 } else { 0.85 };
        }
        let mut h = 5;
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for (c, count) in [(0usize, 3usize), (1, 2), (2, 2), (3, 2), (4, 3)] {
            for k in 0..count {
                let angle = k as f64 * 2.1 + c as f64;
                x[3 * h] = x[3 * c] + 0.3 * next();
                x[3 * h + 1] = x[3 * c + 1] + if c % 2 == 0 { -0.6 } else { 0.6 } + 0.3 * next();
                x[3 * h + 2] = 0.95 * angle.cos() + 0.3 * next();
                h += 1;
            }
        }
        Array1::from(x)
    }

    /// Water molecules on a loose ring, each tilted differently.
    pub(crate) fn water_ring(n_mol: usize, seed: u64) -> Array1<f64> {
        let mut x = Vec::with_capacity(9 * n_mol);
        let mut state = seed.wrapping_mul(2_862_933_555_777_941_757).wrapping_add(3);
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let radius = 2.9 / (2.0 * (std::f64::consts::PI / n_mol as f64).sin());
        for m in 0..n_mol {
            let t = 2.0 * std::f64::consts::PI * m as f64 / n_mol as f64;
            let o = [radius * t.cos(), radius * t.sin(), 0.3 * next()];
            let a = next() * 3.0;
            let b = next() * 3.0;
            let h1 = [
                o[0] + 0.95 * a.cos(),
                o[1] + 0.95 * a.sin(),
                o[2] + 0.2 * next(),
            ];
            let h2 = [o[0] + 0.95 * b.cos() * 0.5, o[1] + 0.3 * next(), o[2] + 0.9];
            x.extend(o);
            x.extend(h1);
            x.extend(h2);
        }
        Array1::from(x)
    }
}

#[cfg(test)]
mod tests {
    use super::test_ff::{ValenceFf, pentane_start, water_ring};
    use super::*;
    use crate::methods::warm_lbfgs::WarmLbfgs;

    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(7);
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }
    }

    #[test]
    fn the_eigensolver_reconstructs_a_dense_symmetric_matrix() {
        let mut next = rng(3);
        for n in [1usize, 2, 7, 40] {
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..=i {
                    let v = next();
                    a[i * n + j] = v;
                    a[j * n + i] = v;
                }
            }
            let (d, v) = symmetric_eigen(&a, n);
            for w in d.windows(2) {
                assert!(w[0] <= w[1], "not ascending: {d:?}");
            }
            for i in 0..n {
                for j in 0..n {
                    let mut back = 0.0;
                    let mut ortho = 0.0;
                    for k in 0..n {
                        back += v[i * n + k] * d[k] * v[j * n + k];
                        ortho += v[k * n + i] * v[k * n + j];
                    }
                    assert!((back - a[i * n + j]).abs() < 1e-10, "n={n} reconstruction");
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((ortho - want).abs() < 1e-10, "n={n} orthonormality");
                }
            }
        }
    }

    #[test]
    fn the_damped_attraction_derivatives_match_finite_differences() {
        let c = 20.0 * 0.7;
        let r0 = 2.44 / BOHR_ANGSTROM;
        let v = |r: f64| {
            let f = 1.0 / (1.0 + (-VDW_STEEPNESS * (r / r0 - 1.0)).exp());
            -c * f / r.powi(6)
        };
        for r in [3.0, 4.6, 6.0, 9.0] {
            let (v1, v2) = dispersion_derivatives(c, r0, r);
            let h = 1e-4;
            let fd1 = (v(r + h) - v(r - h)) / (2.0 * h);
            let fd2 = (v(r + h) - 2.0 * v(r) + v(r - h)) / (h * h);
            assert!(
                (v1 - fd1).abs() < 1e-6 * (1.0 + fd1.abs()),
                "r={r}: {v1} vs {fd1}"
            );
            assert!(
                (v2 - fd2).abs() < 1e-4 * (1.0 + fd2.abs()),
                "r={r}: {v2} vs {fd2}"
            );
        }
    }

    /// xtb 6.7.1 `getvdwxx` (`src/model_hessian.f90`), transcribed: the
    /// second derivative along `x` of `-s6 c66 f(r) / r^6`.
    fn xtb_getvdwxx(rx: f64, ry: f64, rz: f64, c66: f64, s6: f64, r0: f64) -> f64 {
        let avdw = 20.0;
        let t1 = s6 * c66;
        let t2 = rx * rx;
        let t5 = t2 + ry * ry + rz * rz;
        let t6 = t5 * t5;
        let t7 = t6 * t6;
        let t10 = t5.sqrt();
        let t11 = 1.0 / r0;
        let t15 = (-avdw * (t10 * t11 - 1.0)).exp();
        let t16 = 1.0 + t15;
        let t17 = 1.0 / t16;
        let t24 = t16 * t16;
        let t25 = 1.0 / t24;
        let t29 = t11 * t15;
        let t33 = 1.0 / t7;
        let t44 = avdw * avdw / (r0 * r0);
        let t45 = t15 * t15;
        -48.0 * t1 / t7 / t5 * t17 * t2
            + 13.0 * t1 / t10 / t7 * t25 * t2 * avdw * t29
            + 6.0 * t1 * t33 * t17
            - 2.0 * t1 * t33 / t24 / t16 * t44 * t2 * t45
            - t1 / t10 / t6 / t5 * t25 * avdw * t29
            + t1 * t33 * t25 * t44 * t2 * t15
    }

    /// xtb 6.7.1 `getvdwxy`, transcribed: the mixed `x`, `y` derivative.
    fn xtb_getvdwxy(rx: f64, ry: f64, rz: f64, c66: f64, s6: f64, r0: f64) -> f64 {
        let avdw = 20.0;
        let t1 = s6 * c66;
        let t5 = rx * rx + ry * ry + rz * rz;
        let t6 = t5 * t5;
        let t7 = t6 * t6;
        let t11 = t5.sqrt();
        let t12 = 1.0 / r0;
        let t16 = (-avdw * (t11 * t12 - 1.0)).exp();
        let t17 = 1.0 + t16;
        let t25 = t17 * t17;
        let t26 = 1.0 / t25;
        let t35 = 1.0 / t7;
        let t43 = avdw * avdw / (r0 * r0);
        let t44 = t16 * t16;
        -48.0 * t1 / t7 / t5 / t17 * rx * ry
            + 13.0 * t1 / t11 / t7 * t26 * rx * avdw * t12 * ry * t16
            - 2.0 * t1 * t35 / t25 / t17 * t43 * rx * t44 * ry
            + t1 * t35 * t26 * t43 * rx * ry * t16
    }

    /// The assembled pair block is `gmm u u^T - vdw`, as `ddvopt` forms it
    /// (`Hxx = gmm*xkl*xkl/rkl2-vdw(1,1)`), and not `gmm u u^T + vdw`.
    #[test]
    fn the_c6_tensor_is_subtracted_from_the_stretch_block_as_in_xtb() {
        let z = [7u32, 8];
        let dir = [0.48, -0.36, 0.8];
        let c66 = (c6(z[0]) * c6(z[1])).sqrt();
        let rv = (vdw_radius(z[0]) + vdw_radius(z[1])) / BOHR_ANGSTROM;
        for r in [4.0, 6.0, 8.0, 12.0] {
            let d = [r * dir[0], r * dir[1], r * dir[2]];
            let x = [d[0], d[1], d[2], 0.0, 0.0, 0.0];
            let h = model_hessian(&x, &z, ModelHessian::Lindh);
            let gmm = K_STRETCH * pair_weight(ModelHessian::Lindh, z[0], z[1], r * r);
            // vdw(p, q) with the argument order ddvopt passes.
            let vdw = |p: usize, q: usize| {
                let (a, b, c) = match (p.min(q), p.max(q)) {
                    (0, 0) => return xtb_getvdwxx(d[0], d[1], d[2], c66, S6, rv),
                    (1, 1) => return xtb_getvdwxx(d[1], d[0], d[2], c66, S6, rv),
                    (2, 2) => return xtb_getvdwxx(d[2], d[0], d[1], c66, S6, rv),
                    (0, 1) => (d[0], d[1], d[2]),
                    (0, 2) => (d[0], d[2], d[1]),
                    _ => (d[1], d[2], d[0]),
                };
                xtb_getvdwxy(a, b, c, c66, S6, rv)
            };
            let mut radial = 0.0;
            for p in 0..3 {
                for q in 0..3 {
                    let want = gmm * d[p] * d[q] / (r * r) - vdw(p, q);
                    let got = h[p * 6 + q];
                    assert!(
                        (got - want).abs() < 1e-9 * want.abs().max(1e-12),
                        "r={r} ({p},{q}): {got} against xtb {want}"
                    );
                    assert!((h[p * 6 + 3 + q] + want).abs() < 1e-9 * want.abs().max(1e-12));
                    radial += dir[p] * got * dir[q];
                }
            }
            // Beyond the damping radius the attraction curves down, so its
            // subtraction stiffens the pair.
            if r > 1.5 * rv {
                assert!(
                    radial > gmm,
                    "r={r}: radial curvature {radial} not above the bare stretch {gmm}"
                );
            }
        }
    }

    /// With the trust radius on, a step whose energy rises is rejected before
    /// it reaches the update. The next step is then the same RFO step on the
    /// same Hessian, cut to the smaller radius, so it points the same way.
    #[test]
    fn a_rejected_trust_step_leaves_the_coordinate_hessian_alone() {
        let ff = ValenceFf::pentane();
        let x0 = pentane_start(1);
        for update in [HessianUpdate::Bfgs, HessianUpdate::Bofill] {
            let mut opt = AncOpt::new(ff.species.clone(), AncUnits::EV_ANGSTROM);
            opt.gtol = 5e-3;
            opt.update = update;
            opt.trust_radius = Some(0.3);
            let mut points: Vec<Array1<f64>> = Vec::new();
            opt.minimize(x0.view(), 2, |v| {
                points.push(v.to_owned());
                let (e, g) = ff.energy_gradient(v);
                // The first trial is reported 100 eV higher than it is.
                Some(if points.len() == 2 {
                    (e + 100.0, g)
                } else {
                    (e, g)
                })
            });
            assert_eq!(points.len(), 3, "{update:?}");
            let s1 = &points[1] - &x0;
            let s2 = &points[2] - &x0;
            let n1 = s1.dot(&s1).sqrt();
            let n2 = s2.dot(&s2).sqrt();
            assert!(
                (n1 - 0.3 * BOHR_ANGSTROM).abs() < 1e-9,
                "{update:?}: first step {n1} A did not reach the radius"
            );
            assert!(
                (n2 / n1 - 0.25).abs() < 1e-9,
                "{update:?}: radius after the rejection {n2} A against {n1} A"
            );
            let cos = s1.dot(&s2) / (n1 * n2);
            assert!(
                cos > 1.0 - 1e-9,
                "{update:?}: the retry turned (cos {cos}), so the Hessian changed"
            );
        }
    }

    #[test]
    fn a_model_that_predicted_no_descent_does_not_grow_the_radius() {
        let opt = AncOpt::new(vec![1, 1], AncUnits::ATOMIC);
        // A kept step that reached the radius and matched the model doubles it.
        let mut radius = 0.3;
        assert!(judge_step(&mut radius, 0.3, -1e-3, -1e-3, &opt));
        assert_eq!(radius, 0.6);
        for predicted in [0.0, 1e-4] {
            let mut radius = 0.3;
            assert!(judge_step(&mut radius, 0.3, predicted, -1e-3, &opt));
            assert_eq!(radius, 0.3, "predicted {predicted}");
        }
        let mut radius = 0.3;
        assert!(!judge_step(&mut radius, 0.3, -1e-3, 1e-3, &opt));
        assert_eq!(radius, 0.075);
    }

    #[test]
    fn the_model_hessian_is_symmetric_and_blind_to_rigid_motion() {
        let x = pentane_start(1);
        let z = ValenceFf::pentane().species;
        let x_au: Vec<f64> = x.iter().map(|v| v / BOHR_ANGSTROM).collect();
        for model in [ModelHessian::Lindh, ModelHessian::Swart] {
            let h = model_hessian(&x_au, &z, model);
            let n3 = x_au.len();
            let scale = h.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            for i in 0..n3 {
                for j in 0..n3 {
                    assert!((h[i * n3 + j] - h[j * n3 + i]).abs() < 1e-12 * scale);
                }
            }
            // Every term is a function of internal coordinates, so a rigid
            // translation has no curvature.
            for axis in 0..3 {
                let t: Vec<f64> = (0..n3)
                    .map(|r| if r % 3 == axis { 1.0 } else { 0.0 })
                    .collect();
                let ht = matvec(&h, &t, n3);
                assert!(
                    dot(&ht, &ht).sqrt() < 1e-10 * scale,
                    "{model:?} translation"
                );
            }
        }
    }

    #[test]
    fn a_stretch_is_stiffer_than_a_torsion_in_the_model() {
        let ff = ValenceFf::pentane();
        let x = pentane_start(2);
        let mut opt = AncOpt::new(ff.species.clone(), AncUnits::EV_ANGSTROM);
        opt.gtol = 1e-4;
        let (_, xr, _) = opt.minimize(x.view(), 500, |v| Some(ff.energy_gradient(v)));
        let x_au: Vec<f64> = xr.iter().map(|v| v / BOHR_ANGSTROM).collect();
        let anc = Anc::build(&opt, xr.view());
        assert_eq!(anc.nvar, 3 * 17 - 6);
        let soft = anc
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let stiff = anc.eigenvalues.iter().copied().fold(0.0, f64::max);
        assert!(stiff > 20.0 * soft, "model spectrum {soft}..{stiff}");
        assert_eq!(x_au.len(), 51);
    }

    fn lbfgs_calls(ff: &ValenceFf, x: ArrayView1<f64>, gtol: f64) -> (f64, usize) {
        let mut opt = WarmLbfgs::default();
        opt.gtol = gtol;
        let (f, xr, evals) = opt.minimize(x, 5000, |v| Some(ff.energy_gradient(v)));
        let g = ff.energy_gradient(xr.view()).1;
        assert!(g.iter().all(|v| v.abs() < gtol), "L-BFGS did not converge");
        (f, evals)
    }

    fn anc_calls(opt: &AncOpt, ff: &ValenceFf, x: ArrayView1<f64>) -> (f64, usize) {
        let (f, xr, evals) = opt.minimize(x, 5000, |v| Some(ff.energy_gradient(v)));
        let g = ff.energy_gradient(xr.view()).1;
        let worst = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(worst < opt.gtol, "ANC did not converge: max |g| {worst}");
        (f, evals)
    }

    #[test]
    fn it_converges_on_pentane_in_fewer_calls_than_lbfgs() {
        let ff = ValenceFf::pentane();
        let gtol = 5e-3;
        let mut total = (0usize, 0usize);
        let mut same_basin = 0;
        for seed in 0..4 {
            let x = pentane_start(seed);
            let start = ff.energy_gradient(x.view()).0;
            let (fl, cl) = lbfgs_calls(&ff, x.view(), gtol);
            for update in [HessianUpdate::Bfgs, HessianUpdate::Bofill] {
                for trust in [None, Some(0.3)] {
                    let mut opt = AncOpt::new(ff.species.clone(), AncUnits::EV_ANGSTROM);
                    opt.gtol = gtol;
                    opt.update = update;
                    opt.trust_radius = trust;
                    let (fa, ca) = anc_calls(&opt, &ff, x.view());
                    // Both land on a stationary point, but a start can drain
                    // into the gauche rather than the all-anti conformer.
                    assert!(fa < start, "seed {seed}: ANC ended at {fa} from {start}");
                    if update == HessianUpdate::Bfgs && trust.is_none() {
                        total.0 += ca;
                        total.1 += cl;
                        same_basin += usize::from((fa - fl).abs() < 1e-3);
                    }
                }
            }
        }
        assert!(
            same_basin >= 3,
            "ANC and L-BFGS agree on {same_basin} of 4 starts"
        );
        assert!(
            2 * total.0 < 3 * total.1 / 2,
            "ANC {} calls against L-BFGS {}",
            total.0,
            total.1
        );
    }

    #[test]
    fn it_converges_on_a_water_ring_in_fewer_calls_than_lbfgs() {
        let ff = ValenceFf::waters(4);
        let gtol = 5e-3;
        let mut total = (0usize, 0usize);
        for seed in 0..4 {
            let x = water_ring(4, seed);
            let (fl, cl) = lbfgs_calls(&ff, x.view(), gtol);
            for model in [ModelHessian::Lindh, ModelHessian::Swart] {
                let mut opt = AncOpt::new(ff.species.clone(), AncUnits::EV_ANGSTROM);
                opt.gtol = gtol;
                opt.model = model;
                let (fa, ca) = anc_calls(&opt, &ff, x.view());
                assert!(fa < fl + 0.5, "seed {seed}: ANC {fa} against L-BFGS {fl}");
                if model == ModelHessian::Lindh {
                    total.0 += ca;
                    total.1 += cl;
                }
            }
        }
        assert!(
            total.0 < total.1,
            "ANC {} calls against L-BFGS {}",
            total.0,
            total.1
        );
    }

    #[test]
    fn a_frozen_atom_stays_where_it_was() {
        let ff = ValenceFf::waters(2);
        let x = water_ring(2, 5);
        let mut opt = AncOpt::new(ff.species.clone(), AncUnits::EV_ANGSTROM);
        opt.gtol = 5e-3;
        let frozen = vec![false, false, false, true, true, true];
        opt.frozen = Some(frozen.clone());
        let pinned = |v: ArrayView1<f64>| {
            let (e, mut g) = ff.energy_gradient(v);
            for i in 3..6 {
                for k in 0..3 {
                    g[3 * i + k] = 0.0;
                }
            }
            Some((e, g))
        };
        let (_, xr, _) = opt.minimize(x.view(), 500, pinned);
        for i in 9..18 {
            assert_eq!(xr[i], x[i], "frozen coordinate {i} moved");
        }
        assert!(xr.iter().zip(x.iter()).take(9).any(|(a, b)| a != b));
    }

    #[test]
    fn it_stops_when_the_budget_ends_and_returns_the_lowest_point() {
        let ff = ValenceFf::pentane();
        let x = pentane_start(3);
        let opt = AncOpt::new(ff.species.clone(), AncUnits::EV_ANGSTROM);
        let mut left = 4;
        let mut seen = Vec::new();
        let (f, xr, evals) = opt.minimize(x.view(), 500, |v| {
            if left == 0 {
                return None;
            }
            left -= 1;
            let out = ff.energy_gradient(v);
            seen.push(out.0);
            Some(out)
        });
        assert_eq!(evals, 4);
        let lowest = seen.iter().copied().fold(f64::INFINITY, f64::min);
        assert_eq!(f, lowest);
        assert_eq!(ff.energy_gradient(xr.view()).0, f);
    }

    #[test]
    fn a_watch_that_refuses_ends_the_relaxation() {
        let ff = ValenceFf::pentane();
        let x = pentane_start(4);
        let opt = AncOpt::new(ff.species.clone(), AncUnits::EV_ANGSTROM);
        let mut calls = 0;
        let (_, _, evals) = opt.minimize_watched(
            x.view(),
            500,
            |v| {
                calls += 1;
                Some(ff.energy_gradient(v))
            },
            |step, _| step < 2,
        );
        assert_eq!(evals, calls);
        assert_eq!(evals, 4, "initial point plus three steps");
    }

    #[test]
    fn atomic_units_and_electronvolt_angstrom_take_the_same_path() {
        let ff = ValenceFf::waters(3);
        let x = water_ring(3, 2);
        let mut ev = AncOpt::new(ff.species.clone(), AncUnits::EV_ANGSTROM);
        ev.gtol = 5e-3;
        // Rounding differs between the two unit systems, so compare a short
        // trajectory rather than a full relaxation.
        let (fe, xe, ce) = ev.minimize(x.view(), 15, |v| Some(ff.energy_gradient(v)));
        let mut au = AncOpt::new(ff.species.clone(), AncUnits::ATOMIC);
        au.gtol = 5e-3 * BOHR_ANGSTROM / HARTREE_EV;
        let x_au = x.mapv(|v| v / BOHR_ANGSTROM);
        let (fa, xa, ca) = au.minimize(x_au.view(), 15, |v| {
            let (e, g) = ff.energy_gradient(v.mapv(|c| c * BOHR_ANGSTROM).view());
            Some((e / HARTREE_EV, g * (BOHR_ANGSTROM / HARTREE_EV)))
        });
        assert_eq!(ce, ca);
        assert!((fe - fa * HARTREE_EV).abs() < 1e-8);
        for (a, b) in xe.iter().zip(xa.iter()) {
            assert!((a - b * BOHR_ANGSTROM).abs() < 1e-8);
        }
    }
}
