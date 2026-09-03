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
//! # What it is not invariant to
//!
//! Permutation, and this is worth stating plainly because the library contains
//! machinery that looks like it handles the problem and does not.
//! `auxiliary/Distance` carries `get_canonical_configuration`, a Hungarian
//! assignment in `solve_assignment_problem`, and an aligned RMSD. None of them
//! is on the kernel path: `SexpatCF` calls only `dist_at_cached_sq` and
//! `dist_at_vec`, which are the inverse-distance metric above. The canonical
//! ordering and the assignment live in `dist_rmsd` and `dist_emd`, which are
//! trust-radius metrics selected by `gpr_distance_metric_t` and consumed by the
//! dimer and NEB drivers, a separate axis from the covariance.
//!
//! So the kernel compares pair `(i, j)` of one structure with pair `(i, j)` of
//! another, and relabelling the points changes the distance. Inside a
//! basin-hopping trajectory that is usually harmless, because a structure and
//! its own relaxed image carry the same labels. Between two independently
//! generated structures it is not harmless at all, and it is the reason a
//! descriptor kernel with permutation invariance built in is worth measuring
//! against this one rather than assuming either is better.
//!
//! Every observation carries its gradient. `gpr_model_train` takes energies and
//! gradients together and the posterior weight vector is
//! `n_obs * (1 + n_coords)` long, so the joint system is the standard one:
//! `cov(f, f) = k`, `cov(f, df/dx') = dk/dx'`, `cov(df/dx, df/dx') = d2k/dx dx'`.
//! That is the whole reason to use this rather than a value-only model here:
//! the search already computes a gradient at every relaxation step and throws
//! all of them away.
//!
//! # Measured cost, and why this model does not fit LJ38
//!
//! All figures on rg.terra, one core, from
//! `what_the_model_costs_at_the_size_the_campaign_runs`, which is `#[ignore]`d
//! because it takes 96 seconds.
//!
//! At 38 points with 20 observations, so 2300 rows in the joint system:
//! training 31.7 s, prediction with variance 3116 ms, prediction of the mean
//! alone 3.46 ms, and a Lennard-Jones gradient 3.19 us. At 60 observations,
//! 6900 rows: training 214 s and prediction with variance 5650 ms.
//!
//! Put in the currency the ledger charges, one posterior variance at 20
//! observations costs what 976000 force evaluations cost, and a whole run's
//! budget is 400000. One query is two and a half runs. At 60 observations it is
//! 1780000, four and a half runs. The posterior *mean* costs 1083 gradients and
//! is not the problem: the split is 3.46 ms against 3116 ms, so 99.9 percent of
//! a prediction is `calculateVarianceDispatched`. `gpr_model_predict_full` runs
//! both dispatches unconditionally and passing null outputs saves only a
//! memcpy, so there is no cheap way out through the current C API.
//!
//! Every acquisition function needs the variance. Without it there is no
//! Bayesian optimisation, only greedy descent on a surrogate mean. So the
//! inverse-distance model is not usable as a per-hop acquisition surface at
//! this cluster size, and the limit is the variance dispatch rather than
//! anything about permutation or conditioning.
//!
//! Where it *is* usable: as a rarely-consulted model. Training on a schedule of
//! hundreds of hops and predicting the mean only is affordable at 1083 gradients
//! per query, which buys a surrogate-screened proposal filter of the kind
//! [`crate::delayed`] already implements, not a Bayesian one.
//!
//! # Permutation, measured
//!
//! Relabelling the points of a cluster changes nothing physical: same energy to
//! 1e-12, same structure. Through this kernel at 13 points it changes a great
//! deal. As labelled, the posterior mean sits 2.6e-4 from the truth with a
//! standard deviation of 1.42e-4. Relabelled, it sits 3.1e-2 from the truth
//! with a standard deviation of 9.74e-1. The two predictions differ by 218
//! standard deviations of the first, and the reported uncertainty rises by a
//! factor of 6900: the model treats a relabelled copy of a structure it has
//! effectively been shown as a structure it has never seen.
//!
//! Canonicalising with [`crate::shape::CanonicalOrder`] fixes it exactly. A
//! distorted structure and a relabelled copy of it canonicalise to the same
//! coordinates at 2.9e-16 root-mean-square, at 38 points, and it costs 2.4 ms
//! per structure warm. That is 750 gradients, which is real but is one part in
//! 1300 of the variance dispatch it would sit next to. Permutation is a solved
//! problem here and it is not what makes the arm unaffordable.
//!
//! # Build
//!
//! Behind the `gpr` feature, which links `libgpr_optim` from the directory in
//! `GPR_OPTIM_LIB_DIR`. Same contract as the `ira` feature.

use std::os::raw::{c_int, c_void};
use std::sync::{Condvar, Mutex, MutexGuard};

/// Serialises every call into the library.
///
/// Not defensive programming. `libgpr_optim` is normally built against
/// ScaLAPACK with MPI and initialises MPI lazily on first use, so two threads
/// entering the library at once abort the process with "An error occurred in
/// MPI_Init on a NULL communicator". The test binary running these tests in
/// parallel is enough to trigger it, and so is any caller holding one surrogate
/// per thread.
static LIBRARY: Mutex<()> = Mutex::new(());

/// Whether a surrogate is currently alive, with the condition variable callers
/// wait on.
///
/// # Why one model at a time, and not merely one call at a time
///
/// Serialising individual calls is not enough, and this is measured rather than
/// assumed. Running the tests in this module in parallel, with every C call
/// already under [`LIBRARY`], moved the posterior standard deviation at a
/// structure inside the training data from 1.703e-5 to 4.237e-1, the same value
/// it takes far outside the data. In other words the model stopped
/// distinguishing what it had seen from what it had not. The same tests run
/// serially reproduce 1.703e-5 exactly, run after run.
///
/// So the library keeps state that belongs to a model but does not live in the
/// model handle, and interleaving two models' train-and-predict sequences
/// corrupts both. The constraint is one live model per process, and a caller
/// that constructs a second one blocks here until the first is dropped rather
/// than silently getting a model that reports confidence it does not have.
///
/// A guard object would express this better than a flag, but a `MutexGuard`
/// held inside the struct would make [`GpSurrogate`] `!Send`, and a campaign
/// wants to move one between threads even though it will only ever use it from
/// one at a time.
static LIVE: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());

/// Takes the library lock, ignoring poisoning.
///
/// A panic in one caller says nothing about whether the library's own state is
/// consistent, and refusing every later call because an unrelated test failed
/// turns one failure into a cascade.
fn lock() -> MutexGuard<'static, ()> {
    LIBRARY.lock().unwrap_or_else(|e| e.into_inner())
}

/// Blocks until no surrogate is alive, then claims the slot.
fn claim_slot() {
    let (m, cv) = &LIVE;
    let mut live = m.lock().unwrap_or_else(|e| e.into_inner());
    while *live {
        live = cv.wait(live).unwrap_or_else(|e| e.into_inner());
    }
    *live = true;
}

/// Releases the slot and wakes one waiter.
fn release_slot() {
    let (m, cv) = &LIVE;
    let mut live = m.lock().unwrap_or_else(|e| e.into_inner());
    *live = false;
    cv.notify_one();
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
    fn gpr_model_predict(
        model: *mut c_void,
        r_query: *const f64,
        e_out: *mut f64,
        g_out: *mut f64,
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
        // Blocks while another surrogate is alive. See LIVE for the measured
        // reason this is a whole-lifetime claim rather than a per-call one.
        claim_slot();
        let _guard = lock();
        let atoms = unsafe { gpr_atoms_create_simple(n_atoms as c_int) };
        if atoms.is_null() {
            release_slot();
            return None;
        }
        // 0 is the Gaussian process; 1 is the Student-t process, which is a
        // different model and not what the cost figures above describe.
        let model = unsafe { gpr_model_create(0) };
        if model.is_null() {
            unsafe { gpr_atoms_destroy(atoms) };
            release_slot();
            return None;
        }
        if unsafe { gpr_model_init(model, config, atoms) } != 0 {
            unsafe {
                gpr_model_destroy(model);
                gpr_atoms_destroy(atoms);
            }
            release_slot();
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

    /// Posterior energy and gradient, without the variance.
    ///
    /// Separate from [`Self::predict`] because the two library dispatches cost
    /// very differently and an acquisition that needs only the mean should not
    /// pay for the other. `gpr_model_predict_full` runs both unconditionally;
    /// passing null outputs saves a memcpy and nothing else.
    pub fn predict_mean(&mut self, x: &[f64]) -> Option<(f64, Vec<f64>)> {
        if !self.trained || x.len() != self.n_coords {
            return None;
        }
        let _guard = lock();
        let mut e = 0.0;
        let mut g = vec![0.0; self.n_coords];
        let rc = unsafe {
            gpr_model_predict(
                self.model,
                x.as_ptr(),
                &mut e,
                g.as_mut_ptr(),
                self.n_coords as c_int,
            )
        };
        if rc != 0 || !e.is_finite() {
            return None;
        }
        Some((e, g))
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
        {
            let _guard = lock();
            unsafe {
                gpr_model_destroy(self.model);
                gpr_atoms_destroy(self.atoms);
            }
        }
        release_slot();
    }
}

/// Which rule picks the next structure out of a candidate set.
///
/// # Where these come from, and why they are not taken verbatim
///
/// `gpr_optim` already implements this family, in `gpr/neb/AcquisitionStrategy`
/// with `AcquisitionType::{MaxVariance, MaxForce, UCB, ThompsonSampling,
/// IMSPE, ExpectedImprovement, PriorityCascade, TPE, PSBAX}`. Its selection
/// machinery is the right shape, an argmax of a marginal score over a discrete
/// set, and the rules here are its rules: UCB is its `f_norm + kappa *
/// force_sigma`, Thompson is its "sample from the posterior, take the best
/// draw", expected improvement is its `(mu - mu_best) Phi(z) + sigma phi(z)`.
///
/// What does not transfer is the quantity scored and the direction. That
/// selector scores the NEB force magnitude at a path image, with the
/// perpendicular force variance, and takes an argmax because it is hunting a
/// saddle. This scores the energy at a candidate structure, with the energy
/// variance, and hunts a minimum, so every expression is mirrored. Calling
/// `selectNextImage` would mean handing it NEB forces and path tangents that do
/// not exist in a basin-hopping candidate set. The arithmetic below is five
/// lines over the posterior the library computes; the model, which is the part
/// worth reusing, is entirely the library's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Acquisition {
    /// Lower confidence bound, `mean - kappa * sd`.
    ///
    /// The default, and the one both published Gaussian-process structure
    /// searches converged on independently. Bisbo and Hammer, *Efficient global
    /// structure optimization with a machine-learned surrogate model*, Phys Rev
    /// Lett 124, 086102 (2020), doi:10.1103/PhysRevLett.124.086102, use it at
    /// kappa = 2 and publish the sweep that justifies the value: kappa = 1
    /// over-exploits and stalls, kappa = 4 over-explores. Kaappa, del Rio and
    /// Jacobsen, *Global optimization of atomic structures with
    /// gradient-enhanced Gaussian process regression*, Phys Rev B 103, 174114
    /// (2021), doi:10.1103/PhysRevB.103.174114, use the same value, and cap
    /// training at 100 structures rather than sparsifying, which is the same
    /// answer [`GpSurrogate::capacity`] gives to the same problem.
    #[default]
    ConfidenceBound,
    /// Expected improvement over the incumbent.
    ///
    /// The Jones, Schonlau and Welch baseline, *Efficient global optimization
    /// of expensive black-box functions*, J Global Optim 13, 455 (1998),
    /// doi:10.1023/A:1008306431147. Kept
    /// because it is what everyone reaches for first, and because the argument
    /// against it here is a claim that should be measured rather than
    /// asserted: the region that matters for LJ38 is the face-centred-cubic
    /// funnel the search never reaches, where the surrogate has no
    /// observations, the mean sits at the prior and the improvement term is
    /// small. The confidence bound weights the standard deviation with a
    /// coefficient the caller sets instead.
    ExpectedImprovement,
    /// One draw from the posterior at each candidate, then the lowest draw.
    ///
    /// Marginal rather than joint: the C API returns a marginal variance per
    /// query and no candidate-to-candidate covariance, so this samples each
    /// candidate independently. That is the same approximation the library's
    /// own `ThompsonSampling` makes over path images, and it is worth naming,
    /// because a joint draw is what would stop a cluster of near-duplicate
    /// candidates from being sampled as if it were a cluster of independent
    /// chances.
    Thompson,
}

/// Two citations that are easy to get wrong, recorded so they are not.
///
/// Max-value entropy search, Wang and Jegelka, *Max-value entropy search for
/// efficient Bayesian optimization*, ICML 2017, arXiv:1703.01968, targets the
/// entropy of the minimum *value* rather than of the minimiser location, which
/// is a one-dimensional quantity and is why it is cheaper than predictive
/// entropy search. Its robustness claim is to the number of sampled `y*`
/// values, where performance is reported as insensitive down to a single
/// sample. It is *not* a claim of robustness to model misspecification. For
/// that, cite Berkenkamp, Schoellig and Krause, *No-regret Bayesian
/// optimization with unknown hyperparameters*, JMLR 20(50) 2019, or Bogunovic
/// and Krause, *Misspecified Gaussian process bandit optimization*, NeurIPS
/// 2021.
///
/// The SOAP structure kernel `k(A, B) = (pA . pB)^zeta` on normalised power
/// spectra, with normalisation before the exponent, is Bartok, Kondor and
/// Csanyi, *On representing chemical environments*, Phys Rev B 87, 184115
/// (2013), doi:10.1103/PhysRevB.87.184115, equation 36, which uses zeta = 4.
/// De, Bartok, Csanyi and Ceriotti, *Comparing molecules and solids across
/// structural and alchemical space*, Phys Chem Chem Phys 18, 13754 (2016),
/// doi:10.1039/C6CP00415F, is the reference for the structure-level average
/// and REMatch kernels; it contains no zeta and does not assert zeta = 2.
pub mod citations {}

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

/// Expected improvement below `best`, for a minimisation.
///
/// `(best - mean) Phi(z) + sd phi(z)` with `z = (best - mean) / sd`. Never
/// negative, which is what makes it comparable across candidates.
pub fn expected_improvement(mean: f64, variance: f64, best: f64) -> f64 {
    let sd = variance.max(0.0).sqrt();
    if sd < 1e-12 {
        return (best - mean).max(0.0);
    }
    let z = (best - mean) / sd;
    (best - mean) * crate::funnel_bo::normal_cdf(z) + sd * crate::funnel_bo::normal_pdf(z)
}

impl GpSurrogate {
    /// Scores every candidate and returns the index of the best, with its
    /// score.
    ///
    /// `None` when the model is untrained or the candidate list is empty. An
    /// untrained model scores every candidate identically, so returning an
    /// index would let a caller believe a choice had been made.
    ///
    /// `kappa` is read only by [`Acquisition::ConfidenceBound`]; `rng` only by
    /// [`Acquisition::Thompson`], and should return standard normal draws.
    pub fn select(
        &mut self,
        candidates: &[Vec<f64>],
        acquisition: Acquisition,
        kappa: f64,
        rng: &mut impl FnMut() -> f64,
    ) -> Option<(usize, f64)> {
        if !self.trained || candidates.is_empty() {
            return None;
        }
        let best = self
            .energies
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let mut pick = None;
        for (i, c) in candidates.iter().enumerate() {
            let Some((mean, _, var)) = self.predict(c) else {
                continue;
            };
            let score = match acquisition {
                Acquisition::ConfidenceBound => lower_confidence_bound(mean, var, kappa),
                Acquisition::ExpectedImprovement => expected_improvement(mean, var, best),
                // Negated so every rule is maximised, as with the bound.
                Acquisition::Thompson => -(mean + var.max(0.0).sqrt() * rng()),
            };
            if !score.is_finite() {
                continue;
            }
            match pick {
                Some((_, s)) if score <= s => {}
                _ => pick = Some((i, score)),
            }
        }
        pick
    }
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
    fn the_acquisitions_disagree_about_which_candidate_to_take() {
        // Three rules that always agreed would be one rule with three names,
        // and comparing them would measure nothing. The setup is the LJ38
        // shape in miniature: candidates close to the observed structures,
        // where the surrogate is confident, against one far from all of them,
        // where it is not.
        let n = 7;
        let base = cluster(n, 0.0, 33);
        let mut cfg = GpConfig::defaults();
        cfg.report_level = 0;
        cfg.max_iter = 20;
        let mut gp = GpSurrogate::new(n, 32, &cfg).expect("model creation failed");
        for k in 0..10u64 {
            let j = cluster(n, 0.05, 4000 + k * 7);
            let x: Vec<f64> = base.iter().zip(j.iter()).map(|(a, b)| a + 0.2 * b).collect();
            let (e, g) = lj(&x, n);
            gp.record(&x, e, &g);
        }
        assert_eq!(gp.train(), 0);

        let mut cands: Vec<Vec<f64>> = Vec::new();
        for k in 0..6u64 {
            let j = cluster(n, 0.05, 6000 + k * 7);
            cands.push(base.iter().zip(j.iter()).map(|(a, b)| a + 0.2 * b).collect());
        }
        // The far candidate, where the model has nothing to say.
        let j = cluster(n, 0.9, 12_345);
        cands.push(base.iter().zip(j.iter()).map(|(a, b)| a + b).collect());

        for (i, c) in cands.iter().enumerate() {
            let (m, _, v) = gp.predict(c).expect("prediction failed");
            println!("  cand {i}: mean {m:.6} sd {:.3e} true {:.6}", v.sqrt(), lj(c, n).0);
        }
        let mut zero = || 0.0;
        let (ei_pick, _) = gp
            .select(&cands, Acquisition::ExpectedImprovement, 2.0, &mut zero)
            .expect("no expected-improvement pick");
        let (lcb_pick, _) = gp
            .select(&cands, Acquisition::ConfidenceBound, 2.0, &mut zero)
            .expect("no confidence-bound pick");
        // At kappa zero the bound is the posterior mean, so it must be a pure
        // exploit and cannot be the far candidate unless the mean says so.
        let (greedy_pick, _) = gp
            .select(&cands, Acquisition::ConfidenceBound, 0.0, &mut zero)
            .expect("no greedy pick");
        let far = cands.len() - 1;
        println!(
            "picks: ei {ei_pick}, lcb kappa 2 {lcb_pick}, lcb kappa 0 {greedy_pick}, \
             far candidate is {far}"
        );
        // Exploration and exploitation land in different places, which is the
        // only thing that makes the choice of rule a choice.
        assert_eq!(lcb_pick, far, "the confidence bound did not explore");
        assert_eq!(ei_pick, far, "expected improvement did not explore");
        assert_ne!(greedy_pick, far, "kappa zero explored, so it is not greedy");
    }

    #[test]
    fn the_posterior_is_overconfident_where_it_extrapolates() {
        // Worth pinning as a property of this surrogate rather than a
        // footnote. Inside the data the posterior is exact to 1e-5 with a
        // standard deviation to match. One structure away from it, the mean is
        // wrong by more than a whole unit of energy while the standard
        // deviation is 0.35, so the truth sits several standard deviations
        // outside the posterior. Any acquisition that trusts the variance as a
        // calibrated error bar is trusting the wrong number out there, and
        // that is the misspecification risk a stationary kernel carries into a
        // multi-funnel landscape.
        let n = 7;
        let base = cluster(n, 0.0, 33);
        let mut cfg = GpConfig::defaults();
        cfg.report_level = 0;
        cfg.max_iter = 20;
        let mut gp = GpSurrogate::new(n, 32, &cfg).expect("model creation failed");
        for k in 0..10u64 {
            let j = cluster(n, 0.05, 4000 + k * 7);
            let x: Vec<f64> = base.iter().zip(j.iter()).map(|(a, b)| a + 0.2 * b).collect();
            let (e, g) = lj(&x, n);
            gp.record(&x, e, &g);
        }
        assert_eq!(gp.train(), 0);

        let j = cluster(n, 0.05, 6100);
        let near: Vec<f64> = base.iter().zip(j.iter()).map(|(a, b)| a + 0.2 * b).collect();
        let (mn, _, vn) = gp.predict(&near).expect("prediction failed");
        let near_z = (mn - lj(&near, n).0).abs() / vn.sqrt().max(1e-300);

        let j = cluster(n, 0.9, 12_345);
        let far: Vec<f64> = base.iter().zip(j.iter()).map(|(a, b)| a + b).collect();
        let (mf, _, vf) = gp.predict(&far).expect("prediction failed");
        let far_err = (mf - lj(&far, n).0).abs();
        println!(
            "near: error {:.3e} sd {:.3e} ({near_z:.2} sd). far: error {far_err:.3e} sd {:.3e} \
             ({:.2} sd)",
            (mn - lj(&near, n).0).abs(),
            vn.sqrt(),
            vf.sqrt(),
            far_err / vf.sqrt()
        );
        assert!(
            vf.sqrt() > 1000.0 * vn.sqrt(),
            "the model is no less certain away from its data: {:.3e} against {:.3e}",
            vf.sqrt(),
            vn.sqrt()
        );
        assert!(
            far_err > 2.0 * vf.sqrt(),
            "the extrapolation error {far_err:.3e} sits inside the posterior's own \
             {:.3e}, so this test is not measuring overconfidence",
            vf.sqrt()
        );
    }

    #[test]
    fn thompson_moves_with_its_draws_and_the_others_do_not() {
        // The property that makes Thompson a different rule rather than a
        // noisier confidence bound: the same posterior, different draws,
        // different picks. The deterministic rules must be unmoved by the
        // stream they are handed.
        let n = 6;
        let base = cluster(n, 0.0, 91);
        let mut cfg = GpConfig::defaults();
        cfg.report_level = 0;
        cfg.max_iter = 20;
        let mut gp = GpSurrogate::new(n, 32, &cfg).expect("model creation failed");
        for k in 0..8u64 {
            let j = cluster(n, 0.05, 200 + k * 11);
            let x: Vec<f64> = base.iter().zip(j.iter()).map(|(a, b)| a + 0.2 * b).collect();
            let (e, g) = lj(&x, n);
            gp.record(&x, e, &g);
        }
        assert_eq!(gp.train(), 0);
        // Four candidates inside the data and four well outside it. Sampling
        // can only move the pick where the posterior has width, and inside the
        // data it has none: at the library's default noise of 1e-8 the
        // standard deviation among near candidates is around 1e-5 against
        // energy gaps of 1e-2, so a draw never flips the order. Thompson
        // sampling degenerates to the posterior mean there, which is worth
        // knowing before setting kappa by analogy with a published value.
        let mut cands: Vec<Vec<f64>> = (0..4u64)
            .map(|k| {
                let j = cluster(n, 0.05, 700 + k * 5);
                base.iter().zip(j.iter()).map(|(a, b)| a + 0.2 * b).collect()
            })
            .collect();
        for k in 0..4u64 {
            let j = cluster(n, 0.8, 3300 + k * 29);
            cands.push(base.iter().zip(j.iter()).map(|(a, b)| a + b).collect());
        }

        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut draw = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let u1 = ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64)
                / (1u64 << 53) as f64;
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let u2 = ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64)
                / (1u64 << 53) as f64;
            (-2.0 * u1.max(1e-12).ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..60 {
            if let Some((i, _)) = gp.select(&cands, Acquisition::Thompson, 2.0, &mut draw) {
                seen.insert(i);
            }
        }
        assert!(
            seen.len() > 1,
            "Thompson picked candidate {seen:?} every time, so it is not sampling"
        );

        let mut zero = || 0.0;
        let a = gp.select(&cands, Acquisition::ConfidenceBound, 2.0, &mut zero);
        let mut noisy = draw;
        let b = gp.select(&cands, Acquisition::ConfidenceBound, 2.0, &mut noisy);
        assert_eq!(a.map(|v| v.0), b.map(|v| v.0), "the bound consumed randomness");
    }

    #[test]
    fn expected_improvement_is_non_negative_and_zero_far_below_the_incumbent() {
        for var in [0.0, 1e-9, 4.0, 100.0] {
            for mean in [-500.0, -400.0, -300.0] {
                let ei = expected_improvement(mean, var, -400.0);
                assert!(ei >= 0.0 && ei.is_finite(), "EI {ei} at mean {mean} var {var}");
            }
        }
        // A candidate the model is certain is worse than the incumbent offers
        // no improvement at all, which is what makes the quantity comparable.
        assert!(expected_improvement(-300.0, 0.0, -400.0) < 1e-12);
    }

    #[test]
    fn an_untrained_surrogate_selects_nothing() {
        let cfg = GpConfig::defaults();
        let mut gp = GpSurrogate::new(4, 8, &cfg).expect("model creation failed");
        let cands = vec![cluster(4, 0.0, 1), cluster(4, 0.1, 2)];
        let mut zero = || 0.0;
        assert!(
            gp.select(&cands, Acquisition::ConfidenceBound, 2.0, &mut zero)
                .is_none(),
            "an untrained model scores every candidate alike and must not pretend to choose"
        );
    }

    /// Reorders the points of a flattened `(n, 3)` state.
    fn relabel(x: &[f64], perm: &[usize]) -> Vec<f64> {
        let mut out = vec![0.0; x.len()];
        for (slot, &src) in perm.iter().enumerate() {
            out[3 * slot..3 * slot + 3].copy_from_slice(&x[3 * src..3 * src + 3]);
        }
        out
    }

    /// A fixed derangement of `n` points, so no point keeps its index.
    fn derangement(n: usize) -> Vec<usize> {
        (0..n).map(|i| (i * 7 + 3) % n).collect()
    }

    #[test]
    fn relabelling_a_structure_moves_the_posterior() {
        // The blocking question for this kernel. Relabelling the points of a
        // cluster changes nothing physical: same energy, same multiset of
        // distances, same structure. A model of the potential energy surface
        // that answers differently for the two is wrong about one of them, and
        // no hyperparameter search fixes it, because the kernel compares pair
        // (i, j) of one structure with pair (i, j) of another.
        let n = 13;
        let base = cluster(n, 0.0, 5);
        let mut cfg = GpConfig::defaults();
        cfg.report_level = 0;
        cfg.max_iter = 30;
        let mut gp = GpSurrogate::new(n, 32, &cfg).expect("model creation failed");
        for k in 0..12u64 {
            let j = cluster(n, 0.05, 800 + k * 13);
            let x: Vec<f64> = base.iter().zip(j.iter()).map(|(a, b)| a + 0.2 * b).collect();
            let (e, g) = lj(&x, n);
            gp.record(&x, e, &g);
        }
        assert_eq!(gp.train(), 0);

        let j = cluster(n, 0.05, 999_331);
        let x: Vec<f64> = base.iter().zip(j.iter()).map(|(a, b)| a + 0.2 * b).collect();
        let perm = derangement(n);
        let xp = relabel(&x, &perm);

        // The relabelling really is physical identity.
        let (e_true, _) = lj(&x, n);
        let (e_perm, _) = lj(&xp, n);
        assert!(
            (e_true - e_perm).abs() < 1e-12,
            "the relabelling changed the energy by {}",
            (e_true - e_perm).abs()
        );

        let (m, _, v) = gp.predict(&x).expect("prediction failed");
        let (mp, _, vp) = gp.predict(&xp).expect("prediction failed");
        let gap = (m - mp).abs();
        println!(
            "relabelling: true {e_true:.6}; as labelled {m:.6} sd {:.3e}; \
             relabelled {mp:.6} sd {:.3e}; gap {gap:.4e} = {:.1} sd; \
             error as labelled {:.3e}, relabelled {:.3e}",
            v.sqrt(),
            vp.sqrt(),
            gap / v.sqrt().max(1e-300),
            (m - e_true).abs(),
            (mp - e_true).abs()
        );
        assert!(gap.is_finite());
    }

    /// A measurement harness, not a correctness check: it asserts only that
    /// training succeeds and prints timings. Ignored because it takes 96
    /// seconds, almost all of it inside one variance dispatch. Its numbers are
    /// recorded in the module docs. Run with
    /// `cargo test --release --features gpr what_the_model_costs -- --ignored --nocapture`.
    #[ignore = "96 second measurement harness; figures are in the module docs"]
    #[test]
    fn what_the_model_costs_at_the_size_the_campaign_runs() {
        // The feasibility numbers, at LJ38 with a realistic observation count.
        // The ledger charges force evaluations rather than seconds, so a model
        // that saves quenches wins on the ledger however slow it is in wall
        // clock; what these decide is whether a 24-seed array fits in a queue
        // slot, and how many candidates per hop can be afforded.
        let n = 38;
        let base = cluster(n, 0.0, 3);
        let mut cfg = GpConfig::defaults();
        cfg.report_level = 0;
        cfg.max_iter = 30;
        for obs in [20usize] {
            let mut gp = GpSurrogate::new(n, obs, &cfg).expect("model creation failed");
            for k in 0..obs as u64 {
                let j = cluster(n, 0.05, 60_000 + k * 17);
                let x: Vec<f64> = base.iter().zip(j.iter()).map(|(a, b)| a + 0.2 * b).collect();
                let (e, g) = lj(&x, n);
                gp.record(&x, e, &g);
            }
            let t0 = std::time::Instant::now();
            let rc = gp.train();
            let train_s = t0.elapsed().as_secs_f64();
            assert_eq!(rc, 0, "training returned {rc} at {obs} observations");

            let j = cluster(n, 0.05, 424_242);
            let q: Vec<f64> = base.iter().zip(j.iter()).map(|(a, b)| a + 0.2 * b).collect();
            let _ = gp.predict(&q);
            let reps = 20;
            let t1 = std::time::Instant::now();
            for _ in 0..reps {
                let _ = gp.predict(&q);
            }
            let predict_ms = t1.elapsed().as_secs_f64() * 1e3 / f64::from(reps);

            // Where the cost sits: the mean dispatch or the variance one.
            let _ = gp.predict_mean(&q);
            let t3 = std::time::Instant::now();
            for _ in 0..reps {
                let _ = gp.predict_mean(&q);
            }
            let mean_ms = t3.elapsed().as_secs_f64() * 1e3 / f64::from(reps);

            // The same clock on the thing the ledger actually charges.
            let t2 = std::time::Instant::now();
            let grad_reps = 2000;
            for _ in 0..grad_reps {
                let _ = lj(&q, n);
            }
            let grad_us = t2.elapsed().as_secs_f64() * 1e6 / f64::from(grad_reps);

            println!(
                "n=38, {obs} observations ({} joint rows): train {train_s:.2} s, \
                 predict with variance {predict_ms:.2} ms, mean only {mean_ms:.2} ms, \
                 LJ gradient {grad_us:.2} us; one prediction costs {:.0} gradients, \
                 mean only {:.0}, one retrain {:.0}",
                obs * (1 + 3 * n),
                predict_ms * 1e3 / grad_us,
                mean_ms * 1e3 / grad_us,
                train_s * 1e6 / grad_us
            );
        }
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
