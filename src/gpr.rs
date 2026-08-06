//! The gradient-enhanced Gaussian process of `gpr_optim`, over its C API.
//!
//! Basin hopping walks `E~(x) = E(Q(x))` with `Q` a local minimisation, which
//! is a black-box objective costing tens of force evaluations per query.
//! Bayesian optimisation is the machinery for that object, and this crate's
//! [`crate::funnel_bo`] is a strawman version of it: one hand-picked morphology
//! descriptor, energies only, and a squared-exponential kernel written here.
//!
//! The real thing already exists and is not rewritten here. `gpr_optim`
//! (<https://github.com/TheochemUI/gpr_optim>) carries a gradient-enhanced
//! Gaussian process built for GPR-NEB and GP dimer searches, with the
//! Koistinen, Dagbjartsdottir, Asgeirsson, Vehtari and Jonsson inverse-distance
//! kernel and an HS-SVD stable-basis posterior solve. This module is the
//! binding, nothing more.
//!
//! # What the model is conditioned on
//!
//! Not a SOAP vector. The kernel is a squared exponential over inverse
//! interatomic distances,
//! `d(C, C')^2 = sum_ij [(1/r_ij - 1/r_ij') / l_ij]^2`, with one length scale
//! per pair type. Inverse distances rather than distances because the potential
//! diverges as points approach and a representation that is linear in `r` gives
//! the kernel no way to express that; the same choice appears in the GPR-NEB
//! and GP dimer literature for the same reason.
//!
//! Every observation carries its gradient. `gpr_model_train` takes energies and
//! gradients together and the posterior weight vector is
//! `n_obs * (1 + n_coords)` long, so the joint system is the standard one:
//! `cov(f, f) = k`, `cov(f, df/dx') = dk/dx'`, `cov(df/dx, df/dx') = d2k/dx dx'`.
//! That is the whole reason to use this rather than a value-only model here:
//! the search already computes a gradient at every relaxation step and throws
//! all of them away.
//!
//! # Measured cost
//!
//! On rg.terra, a 7-point cluster at 21 coordinates with 12 observations, so
//! 264 rows in the joint system: training 2379 ms, prediction with variance 6.1
//! ms. Training is dominated by the scaled-conjugate-gradient hyperparameter
//! search rather than by the solve, and [`GpConfig::max_iter`] is the knob. A
//! Lennard-Jones gradient on 38 points costs about 3 microseconds, so one
//! retrain at those settings costs what 800000 force evaluations cost and a
//! whole run's budget is 400000. The model has to be retrained on a schedule
//! measured in hundreds of hops, not per hop, and that is a property of the
//! problem rather than of this binding.
//!
//! # Build
//!
//! Behind the `gpr` feature, which links `libgpr_optim` from the directory in
//! `GPR_OPTIM_LIB_DIR`. Same contract as the `ira` feature.

use std::os::raw::{c_int, c_void};
use std::sync::{Mutex, MutexGuard};

/// Serialises every call into the library.
///
/// Not defensive programming. `libgpr_optim` is normally built against
/// ScaLAPACK with MPI, and it initialises MPI lazily on first use, so two
/// threads entering the library at once abort the process with "An error
/// occurred in MPI_Init on a NULL communicator". That is what happens when the
/// test binary runs these tests in parallel, and it would happen to any caller
/// holding one surrogate per thread.
///
/// The lock is process-wide rather than per surrogate because the state being
/// protected is the library's, not the handle's. It costs nothing here: the
/// model is retrained on a schedule of hundreds of hops and each seed of a
/// campaign runs in its own process.
static LIBRARY: Mutex<()> = Mutex::new(());

/// Takes the library lock, ignoring poisoning.
///
/// A panic in one caller says nothing about whether the library's own state is
/// consistent, and refusing every later call because an unrelated test failed
/// turns one failure into a cascade.
fn lock() -> MutexGuard<'static, ()> {
    LIBRARY.lock().unwrap_or_else(|e| e.into_inner())
}

/// Configuration of the process, mirroring `gpr_model_config_t`.
///
/// Field for field the C struct, because a Rust-side struct that reorders or
/// renames anything is a silent memory error rather than a compile failure.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpConfig {
    /// Observation noise variance.
    pub sigma2: f64,
    /// Diagonal jitter.
    pub jitter_sigma2: f64,
    /// Signal variance; zero asks the library to pick one.
    pub magn_sigma2: f64,
    /// Constant covariance term.
    pub const_sigma2: f64,
    /// Prior mean.
    pub prior_mu: f64,
    /// Prior degrees of freedom.
    pub prior_nu: f64,
    /// Prior scale.
    pub prior_s2: f64,
    /// Hyperparameter optimiser; 0 is scaled conjugate gradient.
    pub optim_alg: c_int,
    /// Optimiser iteration cap. The dominant term in the training cost.
    pub max_iter: c_int,
    /// Function tolerance.
    pub tol_func: f64,
    /// Solution tolerance.
    pub tol_sol: f64,
    /// 0 silent, 1 summary, 2 verbose.
    pub report_level: c_int,
    /// Student-t inverse-gamma shape.
    pub tp_hyperprior_a: f64,
    /// Student-t inverse-gamma scale.
    pub tp_hyperprior_b: f64,
    /// Whether to use a residual screened-Coulomb prior mean.
    pub use_zbl_prior_mean: c_int,
    /// Solver backend: 1 for the HS-SVD stable basis, 0 for Cholesky.
    ///
    /// See [`Solve`] for what the two are and why both are reachable.
    pub use_hssvd_backend: c_int,
}

/// Which linear solve the posterior is computed through.
///
/// # Why both are here
///
/// The gradient-enhanced covariance is severely ill-conditioned. A Cholesky
/// factorisation of the raw kernel plus a jitter ladder produces posterior
/// weights that move for numerical rather than physical reasons, and a
/// surrogate whose predictions move that way is indistinguishable, from the
/// outside, from a surrogate that models the wrong thing. This crate has the
/// failure on record: the delayed-acceptance surrogate died on a design matrix
/// at condition 1.6e71 with numerical rank 2 of 11, and read as a modelling
/// failure until the conditioning was looked at.
///
/// [`Solve::StableBasis`] rewrites the kernel as `K = Psi diag(S^2) Psi^T` with
/// `Psi^T Psi = I` and applies the noisy inverse analytically in that basis,
/// `Psi diag(1/(S^2 + s2)) Psi^T y + (1/s2)(y - Psi Psi^T y)`, so the working
/// basis has condition number 1 whatever the nugget. It is `gpr_optim`'s
/// default and the construction is Fasshauer and McCourt's stable Gaussian RBF
/// basis (doi:10.1137/110824784) specialised to the gradient-enhanced dual
/// kernel.
///
/// Running only the stable solve would make a null result unreadable: it could
/// not be told from a conditioning artefact. Running only the Cholesky solve
/// would measure the artefact. Both arms are the measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Solve {
    /// HS-SVD eigenbasis, the conditioned route.
    #[default]
    StableBasis,
    /// Cholesky of the raw kernel with a jitter ladder, the baseline.
    Cholesky,
}

impl Solve {
    /// The flag the C API wants.
    fn flag(self) -> c_int {
        match self {
            Solve::StableBasis => 1,
            Solve::Cholesky => 0,
        }
    }
}

unsafe extern "C" {
    fn gpr_version_components(major: *mut c_int, minor: *mut c_int, patch: *mut c_int);
    fn gpr_atoms_create_simple(n_atoms: c_int) -> *mut c_void;
    fn gpr_atoms_destroy(atoms: *mut c_void);
    fn gpr_model_create(model_type: c_int) -> *mut c_void;
    fn gpr_model_destroy(model: *mut c_void);
    fn gpr_model_config_defaults() -> GpConfig;
    fn gpr_model_init(model: *mut c_void, config: *const GpConfig, atoms: *mut c_void) -> c_int;
    fn gpr_model_train(
        model: *mut c_void,
        r: *const f64,
        e: *const f64,
        g: *const f64,
        n_obs: c_int,
        n_coords: c_int,
    ) -> c_int;
    fn gpr_model_predict_full(
        model: *mut c_void,
        r_query: *const f64,
        e_out: *mut f64,
        g_out: *mut f64,
        var_e_out: *mut f64,
        var_g_out: *mut f64,
        n_coords: c_int,
    ) -> c_int;
    fn gpr_model_n_observations(model: *mut c_void) -> c_int;
    fn gpr_model_set_hssvd_backend(model: *mut c_void, enabled: c_int) -> c_int;
    fn gpr_model_get_hssvd_backend(model: *mut c_void, out: *mut c_int) -> c_int;
}

/// Version of the linked `gpr_optim`, as `(major, minor, patch)`.
pub fn version() -> (i32, i32, i32) {
    let _guard = lock();
    let (mut a, mut b, mut c) = (0, 0, 0);
    unsafe { gpr_version_components(&mut a, &mut b, &mut c) };
    (a, b, c)
}

impl GpConfig {
    /// The library's defaults.
    pub fn defaults() -> Self {
        let _guard = lock();
        unsafe { gpr_model_config_defaults() }
    }
}

/// A trained surrogate over a fixed number of points.
///
/// Owns the two C handles and frees them on drop. Retraining replaces the
/// posterior in place; the observations are held here rather than in the
/// library so a caller can decide what to keep.
pub struct GpSurrogate {
    model: *mut c_void,
    atoms: *mut c_void,
    n_coords: usize,
    /// Positions, `n_obs` rows of `n_coords`.
    positions: Vec<f64>,
    /// Energies, one per row.
    energies: Vec<f64>,
    /// Gradients, `n_obs` rows of `n_coords`.
    gradients: Vec<f64>,
    /// Observations retained. The joint system is `capacity * (1 + n_coords)`
    /// square and the solve is cubic in that, so this is the cost knob.
    capacity: usize,
    trained: bool,
}

/// The C handles are owned exclusively by one `GpSurrogate` and the library
/// takes `&mut`-equivalent pointers for every mutating call, so moving one
/// between threads is sound. It is deliberately not `Sync`: two threads
/// predicting on one model would race on the library's internal scratch.
unsafe impl Send for GpSurrogate {}

impl GpSurrogate {
    /// A surrogate over `n_atoms` moving points, holding at most `capacity`
    /// observations.
    ///
    /// Returns `None` if the library refuses the configuration, which is what
    /// happens when the atoms handle and the coordinate count disagree.
    pub fn new(n_atoms: usize, capacity: usize, config: &GpConfig) -> Option<Self> {
        assert!(capacity > 0, "a surrogate with no room holds nothing");
        let _guard = lock();
        let atoms = unsafe { gpr_atoms_create_simple(n_atoms as c_int) };
        if atoms.is_null() {
            return None;
        }
        // 0 is the Gaussian process; 1 is the Student-t process, which is a
        // different model and not what the cost figures above describe.
        let model = unsafe { gpr_model_create(0) };
        if model.is_null() {
            unsafe { gpr_atoms_destroy(atoms) };
            return None;
        }
        if unsafe { gpr_model_init(model, config, atoms) } != 0 {
            unsafe {
                gpr_model_destroy(model);
                gpr_atoms_destroy(atoms);
            }
            return None;
        }
        Some(Self {
            model,
            atoms,
            n_coords: 3 * n_atoms,
            positions: Vec::new(),
            energies: Vec::new(),
            gradients: Vec::new(),
            capacity,
            trained: false,
        })
    }

    /// Coordinates per observation.
    pub fn n_coords(&self) -> usize {
        self.n_coords
    }

    /// Observations held.
    pub fn len(&self) -> usize {
        self.energies.len()
    }

    /// Whether nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.energies.is_empty()
    }

    /// Whether a posterior exists.
    pub fn trained(&self) -> bool {
        self.trained
    }

    /// Rows in the joint system, which is what the solve is cubic in.
    pub fn rows(&self) -> usize {
        self.len() * (1 + self.n_coords)
    }

    /// Records a structure with its energy and gradient.
    ///
    /// The gradient is not optional. A value-only observation is representable
    /// in the library only by lying about the gradient, and a fabricated zero
    /// gradient tells the process the structure is a stationary point, which
    /// bends the posterior everywhere nearby.
    pub fn record(&mut self, x: &[f64], energy: f64, gradient: &[f64]) {
        if x.len() != self.n_coords || gradient.len() != self.n_coords {
            return;
        }
        if !energy.is_finite()
            || x.iter().any(|v| !v.is_finite())
            || gradient.iter().any(|v| !v.is_finite())
        {
            return;
        }
        self.positions.extend_from_slice(x);
        self.energies.push(energy);
        self.gradients.extend_from_slice(gradient);
        // Oldest first. A surrogate consulted near the incumbent is describing
        // a region the search has moved into, and the observations it has moved
        // away from cost rows without buying accuracy there.
        while self.energies.len() > self.capacity {
            self.positions.drain(0..self.n_coords);
            self.gradients.drain(0..self.n_coords);
            self.energies.remove(0);
        }
    }

    /// Refits the posterior on everything recorded.
    ///
    /// Returns the library's status code, zero on success. Expensive: see the
    /// module docs for the measured figure and why it belongs on a schedule
    /// rather than on every hop.
    pub fn train(&mut self) -> i32 {
        if self.energies.is_empty() {
            return -1;
        }
        let _guard = lock();
        let rc = unsafe {
            gpr_model_train(
                self.model,
                self.positions.as_ptr(),
                self.energies.as_ptr(),
                self.gradients.as_ptr(),
                self.energies.len() as c_int,
                self.n_coords as c_int,
            )
        };
        self.trained = rc == 0;
        rc
    }

    /// Posterior energy, gradient and energy variance at a structure.
    ///
    /// `None` before the first successful training, rather than a prior-mean
    /// guess: a caller screening proposals on an untrained surrogate would be
    /// screening on a constant, and returning a number would hide that.
    pub fn predict(&mut self, x: &[f64]) -> Option<(f64, Vec<f64>, f64)> {
        if !self.trained || x.len() != self.n_coords {
            return None;
        }
        let _guard = lock();
        let mut e = 0.0;
        let mut g = vec![0.0; self.n_coords];
        let mut var_e = 0.0;
        let mut var_g = vec![0.0; self.n_coords];
        let rc = unsafe {
            gpr_model_predict_full(
                self.model,
                x.as_ptr(),
                &mut e,
                g.as_mut_ptr(),
                &mut var_e,
                var_g.as_mut_ptr(),
                self.n_coords as c_int,
            )
        };
        if rc != 0 || !e.is_finite() {
            return None;
        }
        Some((e, g, var_e.max(0.0)))
    }

    /// Observations the library itself reports holding, which is a check that
    /// the training call did what the record count says it should have.
    pub fn library_observations(&self) -> usize {
        let _guard = lock();
        unsafe { gpr_model_n_observations(self.model) }.max(0) as usize
    }

    /// Switches the posterior solve. Takes effect on the next [`Self::train`].
    ///
    /// Returns the library's status code, zero on success.
    pub fn set_solve(&mut self, solve: Solve) -> i32 {
        let _guard = lock();
        let rc = unsafe { gpr_model_set_hssvd_backend(self.model, solve.flag()) };
        if rc == 0 {
            self.trained = false;
        }
        rc
    }

    /// Which solve the library reports it will use.
    ///
    /// Read back from the library rather than remembered here, because a
    /// two-arm measurement whose arm label comes from the caller's own
    /// bookkeeping proves nothing about which code ran.
    pub fn solve(&self) -> Option<Solve> {
        let _guard = lock();
        let mut v: c_int = -1;
        if unsafe { gpr_model_get_hssvd_backend(self.model, &mut v) } != 0 {
            return None;
        }
        Some(if v != 0 {
            Solve::StableBasis
        } else {
            Solve::Cholesky
        })
    }
}

impl Drop for GpSurrogate {
    fn drop(&mut self) {
        let _guard = lock();
        unsafe {
            gpr_model_destroy(self.model);
            gpr_atoms_destroy(self.atoms);
        }
    }
}

/// Lower confidence bound as a score to be maximised, `-(mean - kappa * sd)`.
///
/// Negated so every acquisition here has the same sense and swapping one for
/// another does not also flip a comparison.
///
/// # Why this rather than expected improvement
///
/// Expected improvement measures how much better than the incumbent a point is
/// likely to be, which is the right question when the surrogate is trustworthy.
/// It is not the question here. The failure this is aimed at is a search that
/// never reaches the face-centred-cubic funnel of LJ38 at all, so the region
/// that matters is one the surrogate has no observations in, where the mean is
/// the prior and the improvement term contributes almost nothing. The
/// confidence bound weights the standard deviation directly and with a
/// coefficient the caller sets, so how much of the budget goes to unexplored
/// structure is a number in the configuration rather than a consequence of the
/// prior. It is also what GOFEE uses for the same reason.
pub fn lower_confidence_bound(mean: f64, variance: f64, kappa: f64) -> f64 {
    -(mean - kappa * variance.max(0.0).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small cluster near its minimum separation, so the energies are on the
    /// bound side of the potential rather than in the steep repulsive wall
    /// where a surrogate is being asked about a different problem.
    fn cluster(n: usize, jitter: f64, seed: u64) -> Vec<f64> {
        let mut s = seed | 1;
        let mut rnd = || {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            ((s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        };
        // Points on a shell of radius 1.2, which puts neighbours near the
        // Lennard-Jones minimum at 1.122.
        let mut v = Vec::with_capacity(3 * n);
        for i in 0..n {
            let t = i as f64 * 2.399_963;
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            v.push(1.2 * r * t.cos() + jitter * rnd());
            v.push(1.2 * r * t.sin() + jitter * rnd());
            v.push(1.2 * z + jitter * rnd());
        }
        v
    }

    /// Lennard-Jones in reduced units, the same form the driver charges for.
    fn lj(x: &[f64], n: usize) -> (f64, Vec<f64>) {
        let mut e = 0.0;
        let mut g = vec![0.0; 3 * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = [
                    x[3 * i] - x[3 * j],
                    x[3 * i + 1] - x[3 * j + 1],
                    x[3 * i + 2] - x[3 * j + 2],
                ];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let s6 = 1.0 / (r2 * r2 * r2);
                e += 4.0 * (s6 * s6 - s6);
                let f = 24.0 * (2.0 * s6 * s6 - s6) / r2;
                for k in 0..3 {
                    g[3 * i + k] -= f * d[k];
                    g[3 * j + k] += f * d[k];
                }
            }
        }
        (e, g)
    }

    #[test]
    fn the_library_is_linked_and_reports_a_version() {
        let (a, b, c) = version();
        assert!(a > 0 || b > 0 || c > 0, "gpr_optim reported {a}.{b}.{c}");
    }

    #[test]
    fn it_trains_on_energies_with_gradients_and_predicts_a_held_out_structure() {
        // The claim the whole arm rests on: a process conditioned on the
        // energies and gradients the quench already computed predicts the
        // energy of a structure it was not shown, to better than the spread of
        // the training set. Anything worse and screening proposals on it is
        // screening on noise.
        let n = 7;
        let base = cluster(n, 0.0, 7);
        let mut cfg = GpConfig::defaults();
        cfg.report_level = 0;
        cfg.max_iter = 50;
        let mut gp = GpSurrogate::new(n, 32, &cfg).expect("model creation failed");

        let mut train_e = Vec::new();
        for k in 0..12u64 {
            let x: Vec<f64> = cluster(n, 0.06, 100 + k * 17);
            let x: Vec<f64> = base.iter().zip(x.iter()).map(|(a, b)| a + 0.3 * b).collect();
            let (e, g) = lj(&x, n);
            gp.record(&x, e, &g);
            train_e.push(e);
        }
        assert_eq!(gp.len(), 12);
        assert_eq!(gp.rows(), 12 * (1 + 21));
        let rc = gp.train();
        assert_eq!(rc, 0, "training returned {rc}");
        assert_eq!(gp.library_observations(), 12);

        let mean_e = train_e.iter().sum::<f64>() / train_e.len() as f64;
        let spread = (train_e
            .iter()
            .map(|v| (v - mean_e) * (v - mean_e))
            .sum::<f64>()
            / train_e.len() as f64)
            .sqrt();

        let mut worst: f64 = 0.0;
        for k in 0..6u64 {
            let x: Vec<f64> = cluster(n, 0.06, 9000 + k * 31);
            let x: Vec<f64> = base.iter().zip(x.iter()).map(|(a, b)| a + 0.3 * b).collect();
            let (e_true, _) = lj(&x, n);
            let (e, _, var) = gp.predict(&x).expect("prediction failed after training");
            assert!(var >= 0.0, "negative variance {var}");
            worst = worst.max((e - e_true).abs());
        }
        assert!(
            worst < spread,
            "held-out error {worst:.4} against a training spread of {spread:.4}"
        );
    }

    #[test]
    fn an_untrained_surrogate_refuses_rather_than_guessing() {
        let cfg = GpConfig::defaults();
        let mut gp = GpSurrogate::new(5, 8, &cfg).expect("model creation failed");
        assert!(gp.predict(&cluster(5, 0.0, 3)).is_none());
        assert!(!gp.trained());
    }

    #[test]
    fn it_drops_the_oldest_observation_past_capacity() {
        let cfg = GpConfig::defaults();
        let mut gp = GpSurrogate::new(4, 3, &cfg).expect("model creation failed");
        for k in 0..6u64 {
            let x = cluster(4, 0.05, k + 1);
            let (e, g) = lj(&x, 4);
            gp.record(&x, e, &g);
        }
        assert_eq!(gp.len(), 3, "capacity was not enforced");
        assert_eq!(gp.rows(), 3 * (1 + 12));
    }

    #[test]
    fn a_non_finite_observation_is_refused_rather_than_poisoning_the_solve() {
        let cfg = GpConfig::defaults();
        let mut gp = GpSurrogate::new(4, 8, &cfg).expect("model creation failed");
        let x = cluster(4, 0.05, 11);
        gp.record(&x, f64::NAN, &vec![0.0; 12]);
        gp.record(&x, -1.0, &vec![f64::INFINITY; 12]);
        assert!(gp.is_empty(), "a non-finite observation was accepted");
    }

    #[test]
    fn the_default_solve_is_the_conditioned_one_and_both_are_reachable() {
        // The arm label has to come from the library, not from this crate's
        // own bookkeeping: a two-solve comparison whose labels are asserted
        // rather than read back measures nothing.
        let cfg = GpConfig::defaults();
        assert_eq!(cfg.use_hssvd_backend, 1, "the C default is not HS-SVD");
        let mut gp = GpSurrogate::new(5, 8, &cfg).expect("model creation failed");
        assert_eq!(gp.solve(), Some(Solve::StableBasis));
        assert_eq!(gp.set_solve(Solve::Cholesky), 0);
        assert_eq!(gp.solve(), Some(Solve::Cholesky));
        assert_eq!(gp.set_solve(Solve::StableBasis), 0);
        assert_eq!(gp.solve(), Some(Solve::StableBasis));
    }

    #[test]
    fn both_solves_train_and_predict_on_the_same_observations() {
        // Not that they agree, which on an ill-conditioned covariance is
        // exactly the open question, but that both arms run to completion on
        // identical data so a difference between them is a difference in the
        // solve rather than in what each was shown.
        let n = 7;
        let base = cluster(n, 0.0, 21);
        let mut cfg = GpConfig::defaults();
        cfg.report_level = 0;
        cfg.max_iter = 20;

        let mut out = Vec::new();
        for solve in [Solve::StableBasis, Solve::Cholesky] {
            let mut gp = GpSurrogate::new(n, 32, &cfg).expect("model creation failed");
            assert_eq!(gp.set_solve(solve), 0);
            for k in 0..10u64 {
                let j = cluster(n, 0.06, 500 + k * 13);
                let x: Vec<f64> = base.iter().zip(j.iter()).map(|(a, b)| a + 0.3 * b).collect();
                let (e, g) = lj(&x, n);
                gp.record(&x, e, &g);
            }
            let rc = gp.train();
            assert_eq!(rc, 0, "{solve:?} training returned {rc}");
            assert_eq!(gp.solve(), Some(solve), "the library changed arms");
            let j = cluster(n, 0.06, 77_777);
            let x: Vec<f64> = base.iter().zip(j.iter()).map(|(a, b)| a + 0.3 * b).collect();
            let (e, _, var) = gp.predict(&x).expect("prediction failed");
            assert!(e.is_finite() && var.is_finite(), "{solve:?} gave {e} {var}");
            out.push((solve, e, var));
        }
        // Reported rather than asserted equal: what the two solves do to each
        // other on this covariance is the measurement, not a precondition.
        println!("solve comparison: {out:?}");
    }

    #[test]
    fn the_confidence_bound_prefers_uncertainty_at_equal_means() {
        let certain = lower_confidence_bound(-400.0, 0.01, 2.0);
        let uncertain = lower_confidence_bound(-400.0, 25.0, 2.0);
        assert!(
            uncertain > certain,
            "the bound scored the uncertain structure {uncertain} against {certain}"
        );
        // And at zero kappa it is the mean, so a caller can turn exploration
        // off and get a pure surrogate descent.
        assert!((lower_confidence_bound(-400.0, 25.0, 0.0) - 400.0).abs() < 1e-12);
    }
}
