//! Morphology collective variables: descriptors that say what shape a cluster
//! has, rather than which basin it sits in.
//!
//! Every [`crate::bias::Fingerprint`] this crate shipped before keys on basin
//! identity. A sorted distance spectrum, a sorted site-energy spectrum and an
//! IRA canonical order all answer "is this the same structure as one I have
//! already seen". None of them answers "is this icosahedral or face-centred
//! cubic", which is the coordinate the enhanced-sampling literature deposits a
//! well-tempered bias on for Lennard-Jones clusters. The four descriptors here
//! answer the second question, and they implement the same `Fingerprint` trait,
//! so [`crate::bias::BasinBias`] biases them with no other change: the
//! deposition kernel is a ball of the merge radius in whatever space the
//! descriptor lives in, which for these is a coordinate with physical units
//! rather than a structure identifier.
//!
//! Spherical harmonics come from featomic's spherical expansion rather than
//! from a hand-rolled recursion. featomic is the descriptor library in the
//! metatensor ecosystem, it is the same code the reference implementations use,
//! and reimplementing associated Legendre recursions to get a number that is
//! already tabulated is how sign conventions get into a paper.
//!
//! The coordination histogram follows the recipe in
//! `atomistic-cookbook/examples/metatomic-plumed/metatomic-plumed.py`, class
//! `CoordinationHistogram`, which was written to drive metadynamics on this
//! exact cluster problem. Its kernel width and switching-function shape are
//! carried across unchanged; the cutoff radius is not, and [`CoordinationKde`]
//! says why with numbers.
//!
//! # Cost
//!
//! Every descriptor here runs once per hop, against roughly thirty charged
//! gradient evaluations per hop. featomic's spherical expansion on 38 points
//! costs about 40 microseconds and one Lennard-Jones gradient on 38 points
//! costs about 3 microseconds, so a Steinhardt CV spends about 13 percent of a
//! hop. The SOAP power spectrum at `max_radial = 3`, `max_angular = 4` costs
//! about 260 microseconds, near a whole hop's worth of force, which is why the
//! projection is a scalar read off a frozen basis rather than a full
//! fingerprint compared vector-wise.

use ndarray::{Array1, ArrayView1};

use crate::bias::Fingerprint;

/// Published Steinhardt values for a site with its full first shell, used as
/// the correctness bar for [`SteinhardtQ`].
///
/// Ordered `(Q4, Q6)`. From Steinhardt, Nelson and Ronchetti,
/// *Bond-orientational order in liquids and glasses*, Phys Rev B 28, 784
/// (1983), as tabulated at
/// <https://www.pas.rochester.edu/~wangyt/algorithms/bop/>.
pub mod ideal {
    /// Twelve neighbours at the vertices of a regular icosahedron.
    pub const ICOSAHEDRAL: (f64, f64) = (0.0, 0.66332);
    /// Twelve nearest neighbours of a face-centred cubic site.
    pub const FCC: (f64, f64) = (0.19094, 0.57452);
    /// Twelve nearest neighbours of a hexagonal close-packed site.
    pub const HCP: (f64, f64) = (0.09722, 0.48476);
    /// Six neighbours along the cubic axes.
    pub const SIMPLE_CUBIC: (f64, f64) = (0.763763, 0.353553);
}

/// Cubic switching function of the cookbook recipe.
///
/// `f(y) = 1` for `y <= 0`, `0` for `y >= 1`, and `(y - 1)^2 (1 + 2y)` between.
/// The value and the first derivative both vanish at `y = 1`, so a neighbour
/// crossing the cutoff does so without a step in the coordinate or in its
/// gradient.
fn switch(y: f64) -> f64 {
    if y <= 0.0 {
        1.0
    } else if y >= 1.0 {
        0.0
    } else {
        (y - 1.0) * (y - 1.0) * (1.0 + 2.0 * y)
    }
}

/// Kernel density estimate over per-site coordination numbers.
///
/// The coordination number of a site is `c_i = sum_j f((r_ij - r1) / (r0 - r1))`
/// with `f` the cubic switch above, and the descriptor is
/// `h_k = sum_i exp(-(c_i - b_k)^2 / (2 sigma^2))` over the bin centres `b_k`.
///
/// It is a kernel density estimate and not a histogram, and that is the whole
/// point of using it as a bias coordinate. A binned histogram jumps when a site
/// crosses a bin edge, so it is discontinuous in the positions; a well-tempered
/// bias deposited on a discontinuous coordinate is a bias on a step function
/// rather than on the landscape, and the Barducci convergence argument, which
/// assumes the free energy in the CV is a function you can fill, says nothing
/// about it. The estimate above is smooth in the positions, so the deposited
/// hills mean what the derivation says they mean.
///
/// It also carries more than a single Steinhardt number does. Cluster
/// morphologies differ in their coordination distributions: the 38-point
/// face-centred-cubic truncated octahedron has 6 sites at coordination 12
/// against 0 for the icosahedral competitor, and their surface populations
/// differ as well. A distribution-valued coordinate reads that directly, where
/// a scalar bond-order parameter compresses it. That matters here because the
/// fourth-order bond-order parameter separates the two 75-point funnels by
/// 0.023, thinner than any deposition width that is not also thinner than the
/// numerical noise on the coordinate.
///
/// # Provenance
///
/// From `CoordinationHistogram` in
/// `atomistic-cookbook/examples/metatomic-plumed/metatomic-plumed.py`. Carried
/// across unchanged: the cubic switching function, the kernel width
/// `sigma = 0.5` (half the unit spacing between integer coordination numbers,
/// so adjacent counts overlap at `exp(-0.5) = 0.61` and the coordinate stays
/// smooth across a coordination change), and the two bin centres 6 and 8.
///
/// Changed, with the number: the recipe sets the switch to start at `4/5` of
/// the cutoff and leaves the cutoff to the caller. [`CoordinationKde::for_lj`]
/// picks 1.5 length units, because the Lennard-Jones first neighbour shell sits
/// at 1.09 to 1.15 and the face-centred-cubic second shell at 1.59, so 1.5
/// separates them, and `4/5` of it is 1.2, above the whole first shell. A
/// cutoff taken from the recipe's own system would put the switch inside the
/// first shell and make the coordination number of a bulk site depend on the
/// compression of the cluster rather than on its coordination.
pub struct CoordinationKde {
    /// Points per state; the state length must be `3 * n_points`.
    pub n_points: usize,
    /// Distance at which the switching function reaches zero.
    pub r0: f64,
    /// Distance below which the switching function is one.
    pub r1: f64,
    /// Coordination numbers the kernels are centred on.
    pub bins: Vec<f64>,
    /// Kernel width in coordination-number units.
    pub sigma: f64,
}

impl CoordinationKde {
    /// The recipe's settings at a Lennard-Jones length scale of `scale`.
    ///
    /// `scale` is the potential's `r_min` divided by `2^(1/6)`, which is 1 for
    /// Lennard-Jones and rescales the shells for Morse.
    pub fn for_lj(n_points: usize, scale: f64) -> Self {
        let r0 = 1.5 * scale;
        Self {
            n_points,
            r0,
            r1: 0.8 * r0,
            bins: vec![6.0, 8.0],
            sigma: 0.5,
        }
    }

    /// Per-site coordination numbers, in the order the points are stored.
    pub fn coordination(&self, x: ArrayView1<f64>) -> Vec<f64> {
        let n = self.n_points;
        let mut c = vec![0.0_f64; n];
        let span = self.r0 - self.r1;
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = x[3 * i] - x[3 * j];
                let dy = x[3 * i + 1] - x[3 * j + 1];
                let dz = x[3 * i + 2] - x[3 * j + 2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r >= self.r0 {
                    continue;
                }
                let w = switch((r - self.r1) / span);
                c[i] += w;
                c[j] += w;
            }
        }
        c
    }
}

impl Fingerprint for CoordinationKde {
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let c = self.coordination(x);
        let two_s2 = 2.0 * self.sigma * self.sigma;
        let mut h = Array1::zeros(self.bins.len());
        for (k, b) in self.bins.iter().enumerate() {
            let mut acc = 0.0;
            for ci in &c {
                let d = ci - b;
                acc += (-d * d / two_s2).exp();
            }
            h[k] = acc;
        }
        h
    }
}

#[cfg(feature = "featomic")]
mod featomic_cv {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::RwLock;

    use featomic::systems::UnitCell;
    use featomic::types::Vector3D;
    use featomic::{CalculationOptions, Calculator, SimpleSystem, System};
    use ndarray::{Array1, ArrayView1};

    use crate::bias::Fingerprint;

    thread_local! {
        /// One calculator per thread per hyperparameter set.
        ///
        /// featomic's `Calculator` needs `&mut self` to compute and is neither
        /// `Send` nor `Sync`, while `Fingerprint` is both. A thread-local cache
        /// keyed on the hyperparameter JSON gives every thread its own
        /// calculator without a lock and without reparsing the JSON per hop.
        static CALCULATORS: RefCell<HashMap<String, Calculator>> =
            RefCell::new(HashMap::new());
    }

    /// Runs `f` on the calculator for `(name, json)`, building it once per
    /// thread.
    fn with_calculator<T>(
        name: &str,
        json: &str,
        f: impl FnOnce(&mut Calculator) -> T,
    ) -> T {
        CALCULATORS.with(|cache| {
            let mut cache = cache.borrow_mut();
            let calc = cache.entry(json.to_string()).or_insert_with(|| {
                Calculator::new(name, json.to_string())
                    .expect("featomic rejected the hyperparameters")
            });
            f(calc)
        })
    }

    /// A flattened `(n, 3)` point set as a single-species featomic system.
    fn system_of(x: ArrayView1<f64>, n: usize) -> Vec<Box<dyn System>> {
        let mut sys = SimpleSystem::new(UnitCell::infinite());
        for i in 0..n {
            sys.add_atom(1, Vector3D::new(x[3 * i], x[3 * i + 1], x[3 * i + 2]));
        }
        vec![Box::new(sys) as Box<dyn System>]
    }

    /// Steinhardt bond-order parameters from featomic's spherical expansion.
    ///
    /// `Q_l = sqrt(4 pi / (2l + 1) * sum_m |q_lm|^2)` with `q_lm` the
    /// neighbour-averaged spherical harmonic coefficients. featomic supplies
    /// the harmonics; the averaging and the normalisation are here.
    ///
    /// # How the textbook parameter falls out of a density expansion
    ///
    /// featomic expands a neighbour density on radial and angular functions and
    /// returns `c_nlm = sum_j f_cut(r_ij) R_nl(r_ij) Y_lm(r_ij)`. Steinhardt's
    /// `q_lm` is the plain neighbour average of `Y_lm`, so two things have to
    /// hold for the expansion to reproduce it. The radial weight must not
    /// depend on `l`, or the ratio `R_nl / R_n0` survives the normalisation and
    /// every `Q_l` comes out scaled by an arbitrary factor; and the central
    /// atom must not contribute to its own density, or the `l = 0` channel that
    /// normalises everything is wrong by one neighbour.
    ///
    /// Both are settings. A Dirac-delta density makes the radial integral the
    /// radial basis function itself, with no `l` dependence, and a tabulated
    /// basis holding the constant 1 makes that function unity. Setting
    /// `center_atom_weight` to zero drops the self contribution. What is left
    /// is `c_lm = sum_j f_cut(r_ij) Y_lm(r_ij)`, from which
    /// `W = sum_j f_cut = sqrt(4 pi) c_00` and
    /// `Q_l = ||c_l|| / (sqrt(2l + 1) c_00)`.
    ///
    /// Measured against the published table, on 12 neighbours at unit distance:
    /// icosahedral 0.000000 and 0.663325 against 0 and 0.66332, face-centred
    /// cubic 0.190941 and 0.574524 against 0.19094 and 0.57452, hexagonal
    /// close-packed 0.097222 and 0.484762 against 0.09722 and 0.48476, simple
    /// cubic 0.763763 and 0.353553 against the same.
    ///
    /// # Cutoff
    ///
    /// The switching function is featomic's shifted cosine rather than the hard
    /// step of the 1983 definition, for the reason [`super::CoordinationKde`]
    /// gives: a hard cutoff makes the coordinate discontinuous in the positions
    /// and a bias on a discontinuous coordinate is not a bias on the landscape.
    /// The smoothing costs nothing on the ideal geometries above, where every
    /// neighbour sits on the flat part of the switch, which is why those values
    /// come out at the published figures to six places.
    ///
    /// # What it does not distinguish
    ///
    /// The descriptor is the whole-cluster bond average, not a per-site value.
    /// It is a coordinate of a few numbers, which is what a well-tempered bias
    /// needs, and it is correspondingly blind: two structures with the same
    /// bond-orientational average are one point to it.
    pub struct SteinhardtQ {
        /// Points per state; the state length must be `3 * n_points`.
        pub n_points: usize,
        /// Angular channels to report, in order. `[4]` for Q4, `[4, 6]` for the
        /// pair.
        pub channels: Vec<usize>,
        /// Distance at which the shifted cosine reaches zero.
        pub cutoff: f64,
        /// Width of the shifted cosine; it is flat below `cutoff - width`.
        pub width: f64,
        /// featomic hyperparameters, built once at construction.
        hypers: String,
    }

    impl SteinhardtQ {
        /// A Steinhardt CV over `channels` at a length scale of `scale`.
        ///
        /// `scale` is the potential's `r_min` divided by `2^(1/6)`, so the
        /// cutoff tracks the potential rather than assuming Lennard-Jones. The
        /// cutoff is 1.5 and the switch runs from 1.2, the same two distances
        /// [`super::CoordinationKde::for_lj`] uses and for the same reason: the
        /// first neighbour shell sits at 1.09 to 1.15 and the second at 1.59.
        pub fn new(n_points: usize, channels: Vec<usize>, scale: f64) -> Self {
            let cutoff = 1.5 * scale;
            let width = 0.3 * scale;
            let hypers = Self::hypers(cutoff, width, &channels);
            Self {
                n_points,
                channels,
                cutoff,
                width,
                hypers,
            }
        }

        /// Q4 alone.
        pub fn q4(n_points: usize, scale: f64) -> Self {
            Self::new(n_points, vec![4], scale)
        }

        /// Q4 and Q6 as a two-component coordinate.
        pub fn q4q6(n_points: usize, scale: f64) -> Self {
            Self::new(n_points, vec![4, 6], scale)
        }

        /// The spherical expansion hyperparameters, as featomic's JSON.
        ///
        /// The `l = 0` channel is always requested because it is the
        /// normalisation.
        fn hypers(cutoff: f64, width: f64, channels: &[usize]) -> String {
            let points = format!(
                "[{{\"position\": 0.0, \"values\": [1.0], \"derivatives\": [0.0]}}, \
                  {{\"position\": {cutoff}, \"values\": [1.0], \"derivatives\": [0.0]}}]"
            );
            let mut ls: Vec<usize> = channels.to_vec();
            ls.push(0);
            ls.sort_unstable();
            ls.dedup();
            let by_angular: Vec<String> = ls
                .iter()
                .map(|l| format!("\"{l}\": {{\"type\": \"Tabulated\", \"points\": {points}}}"))
                .collect();
            format!(
                "{{\"cutoff\": {{\"radius\": {cutoff}, \
                   \"smoothing\": {{\"type\": \"ShiftedCosine\", \"width\": {width}}}}}, \
                   \"density\": {{\"type\": \"DiracDelta\", \"center_atom_weight\": 0.0}}, \
                   \"basis\": {{\"type\": \"Explicit\", \"by_angular\": {{{}}}, \
                   \"spline_accuracy\": null}}}}",
                by_angular.join(", ")
            )
        }

        /// Per-site `Q_l` for every point and every requested channel, indexed
        /// `[site][channel]`.
        ///
        /// The local parameter, which is the one the published ideal-geometry
        /// table reports: the value at a site with its own first shell.
        pub fn site_q(&self, x: ArrayView1<f64>) -> Vec<Vec<f64>> {
            let n = self.n_points;
            let mut systems = system_of(x, n);
            let out = with_calculator("spherical_expansion", &self.hypers, |calc| {
                calc.compute(&mut systems, CalculationOptions::default())
                    .expect("spherical expansion failed")
            });

            let norms = |l: usize| -> Vec<f64> {
                let Some(idx) = out
                    .keys()
                    .position(&[l.into(), 1.into(), 1.into(), 1.into()])
                else {
                    return vec![0.0; n];
                };
                let block = out.block_by_id(idx);
                let array = block.values().to_array();
                let shape = array.shape().to_vec();
                let flat: Vec<f64> = array.iter().copied().collect();
                let (n_m, n_p) = (shape[1], shape[2]);
                let mut acc = vec![0.0_f64; n];
                for site in 0..n {
                    let Some(row) = block.samples().position(&[0.into(), site.into()]) else {
                        continue;
                    };
                    let mut s = 0.0;
                    for m in 0..n_m {
                        let v = flat[row * n_m * n_p + m * n_p];
                        s += v * v;
                    }
                    acc[site] = s.sqrt();
                }
                acc
            };

            let c00 = norms(0);
            let per_channel: Vec<Vec<f64>> = self.channels.iter().map(|&l| norms(l)).collect();
            (0..n)
                .map(|site| {
                    self.channels
                        .iter()
                        .enumerate()
                        .map(|(k, &l)| {
                            let w = c00[site];
                            if w <= 1e-12 {
                                0.0
                            } else {
                                per_channel[k][site] / (w * ((2 * l + 1) as f64).sqrt())
                            }
                        })
                        .collect()
                })
                .collect()
        }
    }

    impl Fingerprint for SteinhardtQ {
        /// Whole-cluster `Q_l`, one entry per requested channel.
        ///
        /// The bond average runs over every bond in the cluster, not over the
        /// per-site values: the coefficients are summed across sites before the
        /// norm is taken, which is the global parameter of the 1983 paper. The
        /// average of per-site `Q_l` is a different quantity and does not take
        /// the tabulated ideal values on a finite cluster, because the surface
        /// sites of a 13-point icosahedron do not have icosahedral
        /// environments.
        fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let n = self.n_points;
            let mut systems = system_of(x, n);
            let out = with_calculator("spherical_expansion", &self.hypers, |calc| {
                calc.compute(&mut systems, CalculationOptions::default())
                    .expect("spherical expansion failed")
            });

            // Summed over sites first, then the norm: the global parameter.
            let summed_norm = |l: usize| -> f64 {
                let Some(idx) = out
                    .keys()
                    .position(&[l.into(), 1.into(), 1.into(), 1.into()])
                else {
                    return 0.0;
                };
                let block = out.block_by_id(idx);
                let array = block.values().to_array();
                let shape = array.shape().to_vec();
                let flat: Vec<f64> = array.iter().copied().collect();
                let (n_s, n_m, n_p) = (shape[0], shape[1], shape[2]);
                let mut totals = vec![0.0_f64; n_m];
                for row in 0..n_s {
                    for m in 0..n_m {
                        totals[m] += flat[row * n_m * n_p + m * n_p];
                    }
                }
                totals.iter().map(|v| v * v).sum::<f64>().sqrt()
            };

            let c00 = summed_norm(0);
            let mut q = Array1::zeros(self.channels.len());
            if c00 <= 1e-12 {
                return q;
            }
            for (k, &l) in self.channels.iter().enumerate() {
                q[k] = summed_norm(l) / (c00 * ((2 * l + 1) as f64).sqrt());
            }
            q
        }
    }

    /// A structure-level SOAP descriptor with its derivative in the
    /// coordinates.
    ///
    /// The pair a gradient-enhanced Gaussian process over structures needs: the
    /// descriptor that goes in the kernel, and the Jacobian that carries a
    /// Cartesian gradient into descriptor space by the chain rule.
    pub struct SoapDescriptor {
        /// Normalised structure descriptor, length `D`.
        pub p: Vec<f64>,
        /// `d p / d x`, row-major `(D, 3N)`, present only when asked for.
        ///
        /// Of the *normalised* descriptor: the projection
        /// `(I - p p^T) / ||p||` is already applied, so a caller does not have
        /// to remember to.
        pub jacobian: Option<Vec<f64>>,
    }

    /// SOAP power spectrum with position derivatives, for a kernel over
    /// structures.
    ///
    /// # Why this sits next to [`SoapProjection`]
    ///
    /// [`SoapProjection`] compresses the power spectrum to one number for use
    /// as a bias coordinate. This keeps the whole vector, because a kernel
    /// `k(A, B) = (pA . pB)^zeta` needs it, and adds the derivative, because
    /// conditioning a process on the gradients the quench already computes
    /// needs the chain rule from coordinates into descriptor space.
    ///
    /// # Why a descriptor kernel rather than the coordinate one
    ///
    /// Two measurements, both in [`crate::gpr`].
    ///
    /// The inverse-distance kernel there is not invariant to relabelling the
    /// points. On 13 points a relabelling that changes nothing physical moved
    /// the posterior mean by 218 standard deviations and raised the reported
    /// standard deviation from 1.42e-4 to 9.74e-1. SOAP is invariant to
    /// permutation, rotation and translation by construction, so the question
    /// does not arise and the 2.4 ms per structure that canonicalisation costs
    /// is not paid.
    ///
    /// And the variance. That library spends 3116 ms of a 3116 ms prediction
    /// inside its variance dispatch at 38 points, which is 976000 force
    /// evaluations against a whole run budget of 400000. A kernel written here
    /// computes `k(x, x) - ||L^-1 k*||^2` in time quadratic in the row count,
    /// microseconds at these sizes.
    pub struct SoapFeatures {
        /// Points per state; the state length must be `3 * n_points`.
        pub n_points: usize,
        /// featomic hyperparameters, built once at construction.
        hypers: String,
    }

    impl SoapFeatures {
        /// A power spectrum at a length scale of `scale`, matching the cutoff
        /// and density of [`SoapProjection::new`] so the two describe the same
        /// neighbourhoods.
        pub fn new(n_points: usize, scale: f64) -> Self {
            let cutoff = 1.5 * scale;
            let width = 0.3 * scale;
            let sigma = 0.3 * scale;
            let hypers = format!(
                "{{\"cutoff\": {{\"radius\": {cutoff}, \
                   \"smoothing\": {{\"type\": \"ShiftedCosine\", \"width\": {width}}}}}, \
                   \"density\": {{\"type\": \"Gaussian\", \"width\": {sigma}, \
                   \"center_atom_weight\": 0.0}}, \
                   \"basis\": {{\"type\": \"TensorProduct\", \"max_angular\": 4, \
                   \"radial\": {{\"type\": \"Gto\", \"max_radial\": 3}}}}}}"
            );
            Self { n_points, hypers }
        }

        /// The descriptor, and its Jacobian when `with_jacobian`.
        ///
        /// The unnormalised structure descriptor is the per-site power spectrum
        /// averaged over sites; the returned one is that divided by its length.
        /// The Jacobian of the normalisation is
        /// `d(p / ||p||) / dx = (I - p p^T) / ||p|| dp/dx`, applied here so the
        /// caller receives the derivative of what the kernel consumes.
        pub fn describe(&self, x: ArrayView1<f64>, with_jacobian: bool) -> SoapDescriptor {
            let n = self.n_points;
            let m = 3 * n;
            let mut systems = system_of(x, n);
            let grads: &[&str] = if with_jacobian { &["positions"] } else { &[] };
            let options = CalculationOptions {
                gradients: grads,
                ..Default::default()
            };
            let out = with_calculator("soap_power_spectrum", &self.hypers, |calc| {
                calc.compute(&mut systems, options)
                    .expect("SOAP power spectrum failed")
            });

            let mut raw: Vec<f64> = Vec::new();
            let mut jac: Vec<f64> = Vec::new();
            for idx in 0..out.keys().count() {
                let block = out.block_by_id(idx);
                let array = block.values().to_array();
                let shape = array.shape().to_vec();
                let flat: Vec<f64> = array.iter().copied().collect();
                let (n_s, n_f) = (shape[0], shape[shape.len() - 1]);
                let base = raw.len();
                raw.resize(base + n_f, 0.0);
                for row in 0..n_s {
                    for f in 0..n_f {
                        raw[base + f] += flat[row * n_f + f];
                    }
                }
                if n > 0 {
                    for f in 0..n_f {
                        raw[base + f] /= n as f64;
                    }
                }

                if !with_jacobian {
                    continue;
                }
                jac.resize(raw.len() * m, 0.0);
                let Some(g) = block.gradient("positions") else {
                    continue;
                };
                let garray = g.values().to_array();
                let gshape = garray.shape().to_vec();
                let gflat: Vec<f64> = garray.iter().copied().collect();
                // Gradient samples are ("sample", "system", "atom") with
                // components [xyz], so the layout is (rows, 3, n_f) and "atom"
                // is the point being moved rather than the centre of the
                // environment.
                let (g_rows, g_dirs, g_f) = (gshape[0], gshape[1], gshape[2]);
                let samples = g.samples();
                for row in 0..g_rows {
                    let atom = samples[row][2].usize();
                    if atom >= n {
                        continue;
                    }
                    for d in 0..g_dirs {
                        for f in 0..g_f {
                            let v = gflat[(row * g_dirs + d) * g_f + f];
                            jac[(base + f) * m + 3 * atom + d] += v / n as f64;
                        }
                    }
                }
            }

            let norm = raw.iter().map(|a| a * a).sum::<f64>().sqrt();
            if norm <= 1e-12 {
                return SoapDescriptor {
                    p: raw,
                    jacobian: with_jacobian.then(Vec::new),
                };
            }
            let p: Vec<f64> = raw.iter().map(|a| a / norm).collect();
            if !with_jacobian {
                return SoapDescriptor { p, jacobian: None };
            }
            // (I - p p^T) J / ||p||, column by column.
            let d = p.len();
            let mut out_j = vec![0.0_f64; d * m];
            for k in 0..m {
                let mut dot = 0.0;
                for f in 0..d {
                    dot += p[f] * jac[f * m + k];
                }
                for f in 0..d {
                    out_j[f * m + k] = (jac[f * m + k] - p[f] * dot) / norm;
                }
            }
            SoapDescriptor {
                p,
                jacobian: Some(out_j),
            }
        }
    }

    /// State of the projection basis, fitted online.
    #[derive(Default)]
    struct Projector {
        /// SOAP vectors of the structures seen so far, until the fit.
        sample: Vec<Vec<f64>>,
        /// Mean of the fitting sample, once fitted.
        mean: Vec<f64>,
        /// Leading principal direction of the fitting sample, once fitted.
        axis: Vec<f64>,
        /// Standard deviation of the fitting sample along `axis`.
        sd: f64,
        /// Whether the fit has happened.
        fitted: bool,
    }

    /// Leading principal component of the SOAP power spectrum, learned from the
    /// structures the search has already visited.
    ///
    /// # What "learned" means, and what it does not
    ///
    /// The basis carries no outside information. It is not a pretrained model,
    /// not a fit to the Cambridge Cluster Database, and not a projection onto
    /// any known minimum. It is the first principal direction of the SOAP
    /// vectors of the first `warmup` quenched structures this run reached, and
    /// nothing else enters it. A different seed fits a different axis, because
    /// it visits different structures. That is deliberate: a projection fitted
    /// against known answers would be a search that has been told where to go,
    /// and the number it produced would not mean anything.
    ///
    /// # The warm-up
    ///
    /// Before the fit the coordinate is 0 for every structure, so the whole
    /// warm-up is one point and the bias deposited during it accumulates in one
    /// place. That place is not arbitrary: the fit centres the coordinate on
    /// the mean of the fitting sample, so 0 after the fit is the centre of the
    /// distribution the chain was in while depositing. At the default 200
    /// structures against roughly 13000 hops in a 4e5-evaluation run, the
    /// warm-up is about 1.5 percent of the run, and the well-tempered weight
    /// makes the bias accumulated in one basin grow logarithmically rather than
    /// linearly, so it does not empty that basin before the coordinate exists.
    ///
    /// # Scale
    ///
    /// The coordinate is divided by the standard deviation of the fitting
    /// sample along the axis, so a merge radius is in units of "how spread out
    /// the structures this run has seen are" rather than in units of an
    /// arbitrary SOAP norm. Without it there is no way to set a deposition
    /// width that transfers between cluster sizes or potentials.
    pub struct SoapProjection {
        /// Points per state; the state length must be `3 * n_points`.
        pub n_points: usize,
        /// Structures collected before the projection is fitted and frozen.
        pub warmup: usize,
        /// featomic hyperparameters, built once at construction.
        hypers: String,
        /// The fitted basis.
        state: RwLock<Projector>,
    }

    impl SoapProjection {
        /// A projection over the SOAP power spectrum at a length scale of
        /// `scale`.
        ///
        /// The cutoff and switching width match [`SteinhardtQ::new`]. The
        /// density width of 0.3 is a third of the first-shell spacing, which
        /// keeps neighbouring shells from smearing into each other; the basis
        /// is `max_radial = 3`, `max_angular = 4`, giving 80 features per site
        /// before the average over sites.
        pub fn new(n_points: usize, scale: f64) -> Self {
            let cutoff = 1.5 * scale;
            let width = 0.3 * scale;
            let sigma = 0.3 * scale;
            let hypers = format!(
                "{{\"cutoff\": {{\"radius\": {cutoff}, \
                   \"smoothing\": {{\"type\": \"ShiftedCosine\", \"width\": {width}}}}}, \
                   \"density\": {{\"type\": \"Gaussian\", \"width\": {sigma}, \
                   \"center_atom_weight\": 0.0}}, \
                   \"basis\": {{\"type\": \"TensorProduct\", \"max_angular\": 4, \
                   \"radial\": {{\"type\": \"Gto\", \"max_radial\": 3}}}}}}"
            );
            Self {
                n_points,
                warmup: 200,
                hypers,
                state: RwLock::new(Projector::default()),
            }
        }

        /// Whether the projection has been fitted and frozen.
        pub fn fitted(&self) -> bool {
            self.state.read().expect("projector lock").fitted
        }

        /// The structure-level SOAP vector: the power spectrum averaged over
        /// sites and normalised to unit length.
        ///
        /// The unit-length step is the usual SOAP normalisation. Without it the
        /// leading principal component is dominated by how many neighbours are
        /// inside the cutoff, which is the coordination coordinate
        /// [`super::CoordinationKde`] already measures directly and better.
        pub fn soap_vector(&self, x: ArrayView1<f64>) -> Vec<f64> {
            let n = self.n_points;
            let mut systems = system_of(x, n);
            let out = with_calculator("soap_power_spectrum", &self.hypers, |calc| {
                calc.compute(&mut systems, CalculationOptions::default())
                    .expect("SOAP power spectrum failed")
            });

            let mut v: Vec<f64> = Vec::new();
            for idx in 0..out.keys().count() {
                let block = out.block_by_id(idx);
                let array = block.values().to_array();
                let shape = array.shape().to_vec();
                let flat: Vec<f64> = array.iter().copied().collect();
                let (n_s, n_f) = (shape[0], shape[shape.len() - 1]);
                let base = v.len();
                v.resize(base + n_f, 0.0);
                for row in 0..n_s {
                    for f in 0..n_f {
                        v[base + f] += flat[row * n_f + f];
                    }
                }
                if n_s > 0 {
                    for f in 0..n_f {
                        v[base + f] /= n_s as f64;
                    }
                }
            }
            let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for a in &mut v {
                    *a /= norm;
                }
            }
            v
        }

        /// Fits the mean and the leading principal direction of `sample`.
        ///
        /// Power iteration on the covariance, run as repeated
        /// `C v = X^T (X v)` products so the covariance is never formed: at 80
        /// features that is a convenience, at a larger basis it is the
        /// difference between a fit and an allocation.
        fn fit(state: &mut Projector) {
            let m = state.sample.len();
            if m < 2 {
                return;
            }
            let d = state.sample[0].len();
            let mut mean = vec![0.0_f64; d];
            for row in &state.sample {
                for (k, a) in row.iter().enumerate() {
                    mean[k] += a;
                }
            }
            for a in &mut mean {
                *a /= m as f64;
            }

            let centred: Vec<Vec<f64>> = state
                .sample
                .iter()
                .map(|row| row.iter().zip(&mean).map(|(a, b)| a - b).collect())
                .collect();

            // Deterministic start, so the fit depends on the visited
            // structures and on nothing else.
            let mut v = vec![0.0_f64; d];
            v[0] = 1.0;
            for _ in 0..128 {
                let mut w = vec![0.0_f64; d];
                for row in &centred {
                    let dot: f64 = row.iter().zip(&v).map(|(a, b)| a * b).sum();
                    for (k, a) in row.iter().enumerate() {
                        w[k] += dot * a;
                    }
                }
                let norm = w.iter().map(|a| a * a).sum::<f64>().sqrt();
                if norm <= 1e-14 {
                    break;
                }
                for a in &mut w {
                    *a /= norm;
                }
                v = w;
            }

            let scores: Vec<f64> = centred
                .iter()
                .map(|row| row.iter().zip(&v).map(|(a, b)| a * b).sum())
                .collect();
            let mu: f64 = scores.iter().sum::<f64>() / m as f64;
            let var: f64 = scores.iter().map(|s| (s - mu) * (s - mu)).sum::<f64>() / m as f64;
            state.sd = var.sqrt().max(1e-9);
            state.mean = mean;
            state.axis = v;
            state.fitted = true;
            state.sample = Vec::new();
        }
    }

    impl Fingerprint for SoapProjection {
        fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
            let v = self.soap_vector(x);
            {
                let state = self.state.read().expect("projector lock");
                if state.fitted {
                    let s: f64 = v
                        .iter()
                        .zip(&state.mean)
                        .map(|(a, b)| a - b)
                        .zip(&state.axis)
                        .map(|(a, b)| a * b)
                        .sum();
                    return Array1::from(vec![s / state.sd]);
                }
            }
            let mut state = self.state.write().expect("projector lock");
            if state.fitted {
                return Array1::from(vec![0.0]);
            }
            state.sample.push(v);
            if state.sample.len() >= self.warmup {
                SoapProjection::fit(&mut state);
            }
            Array1::from(vec![0.0])
        }
    }
}

#[cfg(feature = "featomic")]
pub use featomic_cv::{SoapDescriptor, SoapFeatures, SoapProjection, SteinhardtQ};

#[cfg(test)]
mod tests {
    use super::*;

    /// Twelve neighbours at unit distance, at the vertices of a regular
    /// icosahedron, plus a centre.
    pub(crate) fn icosahedron() -> Vec<f64> {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let s = (1.0 + phi * phi).sqrt();
        let mut v = vec![0.0, 0.0, 0.0];
        for &a in &[1.0_f64, -1.0] {
            for &b in &[1.0_f64, -1.0] {
                v.extend_from_slice(&[0.0, a / s, b * phi / s]);
                v.extend_from_slice(&[a / s, b * phi / s, 0.0]);
                v.extend_from_slice(&[b * phi / s, 0.0, a / s]);
            }
        }
        v
    }

    /// The twelve nearest neighbours of a face-centred cubic site, plus the
    /// site.
    pub(crate) fn fcc_shell() -> Vec<f64> {
        let r = 2.0_f64.sqrt();
        let mut v = vec![0.0, 0.0, 0.0];
        for &a in &[1.0_f64, -1.0] {
            for &b in &[1.0_f64, -1.0] {
                v.extend_from_slice(&[a / r, b / r, 0.0]);
                v.extend_from_slice(&[0.0, a / r, b / r]);
                v.extend_from_slice(&[a / r, 0.0, b / r]);
            }
        }
        v
    }

    /// The twelve nearest neighbours of a hexagonal close-packed site, plus the
    /// site. The two triangles are mirror images across the basal plane, which
    /// is what separates hcp from fcc.
    pub(crate) fn hcp_shell() -> Vec<f64> {
        let mut v = vec![0.0, 0.0, 0.0];
        for k in 0..6 {
            let t = std::f64::consts::PI / 3.0 * f64::from(k);
            v.extend_from_slice(&[t.cos(), t.sin(), 0.0]);
        }
        let h = (2.0_f64 / 3.0).sqrt();
        let rr = 1.0 / 3.0_f64.sqrt();
        for k in 0..3 {
            let t = std::f64::consts::PI / 2.0
                + 2.0 * std::f64::consts::PI / 3.0 * f64::from(k);
            v.extend_from_slice(&[rr * t.cos(), rr * t.sin(), h]);
            v.extend_from_slice(&[rr * t.cos(), rr * t.sin(), -h]);
        }
        v
    }

    /// Six neighbours along the cubic axes, plus the site.
    pub(crate) fn simple_cubic_shell() -> Vec<f64> {
        vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, -1.0,
        ]
    }

    #[test]
    fn coordination_of_a_full_shell_is_the_neighbour_count() {
        // Twelve neighbours at 1.0, well inside the 1.2 flat region, so the
        // switching function is exactly one on every bond and the centre's
        // coordination is exactly 12.
        let cv = CoordinationKde::for_lj(13, 1.0);
        let c = cv.coordination(Array1::from(icosahedron()).view());
        assert!(
            (c[0] - 12.0).abs() < 1e-12,
            "centre coordination {} is not 12",
            c[0]
        );
    }

    #[test]
    fn the_coordination_estimate_is_continuous_across_the_cutoff() {
        // A pair walked out through the switching region must not step. The
        // bound is the largest change the smooth switch can make over a step
        // of 1e-4 in the separation, which is where a binned histogram would
        // instead jump by a whole unit of kernel weight.
        let cv = CoordinationKde::for_lj(2, 1.0);
        let mut prev: Option<Array1<f64>> = None;
        let mut r = 1.15_f64;
        while r < 1.55 {
            let x = Array1::from(vec![0.0, 0.0, 0.0, r, 0.0, 0.0]);
            let h = cv.describe(x.view());
            if let Some(p) = prev {
                let step = h
                    .iter()
                    .zip(p.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                assert!(step < 1e-3, "coordination KDE stepped by {step} at r = {r}");
            }
            prev = Some(h);
            r += 1e-4;
        }
    }

    #[test]
    fn the_coordination_estimate_ignores_labelling_and_orientation() {
        let cv = CoordinationKde::for_lj(13, 1.0);
        let x = Array1::from(icosahedron());
        let base = cv.describe(x.view());
        let rotated = Array1::from(rotate(&icosahedron(), 0.7, 1.3, 2.1));
        let permuted = Array1::from(permute(&icosahedron()));
        for (name, other) in [("rotation", rotated), ("permutation", permuted)] {
            let d = base
                .iter()
                .zip(cv.describe(other.view()).iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(d < 1e-9, "coordination KDE moved by {d} under {name}");
        }
    }

    /// Rotates a flattened point set by the three Euler angles.
    pub(crate) fn rotate(v: &[f64], a: f64, b: f64, c: f64) -> Vec<f64> {
        let (ca, sa) = (a.cos(), a.sin());
        let (cb, sb) = (b.cos(), b.sin());
        let (cc, sc) = (c.cos(), c.sin());
        let mut out = Vec::with_capacity(v.len());
        for p in v.chunks(3) {
            let (x, y, z) = (p[0], p[1], p[2]);
            let (x, y) = (ca * x - sa * y, sa * x + ca * y);
            let (y, z) = (cb * y - sb * z, sb * y + cb * z);
            let (x, y) = (cc * x - sc * y, sc * x + cc * y);
            out.extend_from_slice(&[x, y, z]);
        }
        out
    }

    /// Reverses the order of the points, leaving the point set itself alone.
    pub(crate) fn permute(v: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(v.len());
        for p in v.chunks(3).rev() {
            out.extend_from_slice(p);
        }
        out
    }

    /// Displaces every coordinate by a fixed pattern of the given size.
    pub(crate) fn jitter(v: &[f64], amp: f64) -> Vec<f64> {
        v.iter()
            .enumerate()
            .map(|(i, a)| a + amp * ((i as f64 * 12.9898).sin()))
            .collect()
    }
}

#[cfg(all(test, feature = "featomic"))]
mod featomic_tests {
    use super::tests::{
        fcc_shell, hcp_shell, icosahedron, jitter, permute, rotate, simple_cubic_shell,
    };
    use super::{ideal, SoapFeatures, SteinhardtQ};
    use crate::bias::Fingerprint;
    use ndarray::Array1;

    /// `Q4` and `Q6` at the centre of each ideal shell, in the local sense the
    /// published table reports.
    fn site_q4q6(points: &[f64]) -> (f64, f64) {
        let n = points.len() / 3;
        let cv = SteinhardtQ::q4q6(n, 1.0);
        let q = cv.site_q(Array1::from(points.to_vec()).view());
        (q[0][0], q[0][1])
    }

    #[test]
    fn the_ideal_geometries_take_their_published_values() {
        // Steinhardt, Nelson and Ronchetti, Phys Rev B 28, 784 (1983), as
        // tabulated at pas.rochester.edu/~wangyt/algorithms/bop/. The tolerance
        // is the precision the table is quoted to, not a fitted number.
        for (name, points, want) in [
            ("icosahedral", icosahedron(), ideal::ICOSAHEDRAL),
            ("fcc", fcc_shell(), ideal::FCC),
            ("hcp", hcp_shell(), ideal::HCP),
            ("simple cubic", simple_cubic_shell(), ideal::SIMPLE_CUBIC),
        ] {
            let (q4, q6) = site_q4q6(&points);
            assert!(
                (q4 - want.0).abs() < 1e-5,
                "{name}: Q4 is {q4:.6}, published {:.6}",
                want.0
            );
            assert!(
                (q6 - want.1).abs() < 1e-5,
                "{name}: Q6 is {q6:.6}, published {:.6}",
                want.1
            );
        }
    }

    #[test]
    fn the_bond_order_coordinate_ignores_labelling_and_orientation() {
        // Rotation invariance holds to 1e-9, which is spherical-harmonic
        // round-off, not a tolerance chosen to pass. Permutation invariance is
        // exact up to summation order.
        let cv = SteinhardtQ::q4q6(13, 1.0);
        let base = cv.describe(Array1::from(icosahedron()).view());
        for (name, other) in [
            ("rotation", rotate(&icosahedron(), 0.7, 1.3, 2.1)),
            ("permutation", permute(&icosahedron())),
        ] {
            let moved = cv.describe(Array1::from(other).view());
            let d = base
                .iter()
                .zip(moved.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(d < 1e-9, "Q4/Q6 moved by {d} under {name}");
        }
    }

    #[test]
    fn icosahedral_and_fcc_separate_further_than_either_moves_under_jitter() {
        // The claim a bias coordinate has to satisfy: the gap between the two
        // morphologies is wider than the smear a small distortion puts on
        // either of them, or a deposition width that resolves the gap also
        // resolves noise.
        let cv = SteinhardtQ::q4q6(13, 1.0);
        let dist = |a: &[f64], b: &[f64]| -> f64 {
            let qa = cv.describe(Array1::from(a.to_vec()).view());
            let qb = cv.describe(Array1::from(b.to_vec()).view());
            qa.iter()
                .zip(qb.iter())
                .map(|(p, q)| (p - q) * (p - q))
                .sum::<f64>()
                .sqrt()
        };
        let ico = icosahedron();
        let fcc = fcc_shell();
        let between = dist(&ico, &fcc);
        let ico_smear = dist(&ico, &jitter(&ico, 0.02));
        let fcc_smear = dist(&fcc, &jitter(&fcc, 0.02));
        assert!(
            between > 4.0 * ico_smear.max(fcc_smear),
            "morphologies are {between:.4} apart, jitter smears them by \
             {ico_smear:.4} and {fcc_smear:.4}"
        );
    }
    #[test]
    fn the_soap_jacobian_matches_a_finite_difference() {
        // The part most likely to be silently wrong. featomic returns gradient
        // rows keyed by which point is being moved rather than by which
        // environment is being described, and the structure descriptor is an
        // average over environments, so a wrong axis or a missed accumulation
        // still produces a plausible-looking matrix. A central difference on
        // the normalised descriptor catches all of it.
        let n = 8;
        let f = SoapFeatures::new(n, 1.0);
        let base = icosahedron();
        // Eight points from the thirteen, jittered off the ideal so no
        // symmetry accidentally zeroes a component.
        let mut x: Vec<f64> = base[..3 * n].to_vec();
        for (i, v) in x.iter_mut().enumerate() {
            *v += 0.03 * ((i as f64 * 7.3).sin());
        }
        let xa = Array1::from(x.clone());
        let d = f.describe(xa.view(), true);
        let jac = d.jacobian.as_ref().expect("no jacobian");
        let dim = d.p.len();
        assert_eq!(jac.len(), dim * 3 * n, "jacobian shape");

        let h = 1e-6;
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for k in 0..(3 * n) {
            let mut xp = x.clone();
            xp[k] += h;
            let mut xm = x.clone();
            xm[k] -= h;
            let pp = f.describe(Array1::from(xp).view(), false).p;
            let pm = f.describe(Array1::from(xm).view(), false).p;
            for fi in 0..dim {
                let numeric = (pp[fi] - pm[fi]) / (2.0 * h);
                worst = worst.max((numeric - jac[fi * 3 * n + k]).abs());
                scale = scale.max(numeric.abs());
            }
        }
        assert!(
            worst < 1e-6 * scale.max(1.0),
            "jacobian disagrees with a central difference by {worst:.3e} \
             on a scale of {scale:.3e}"
        );
    }

}
