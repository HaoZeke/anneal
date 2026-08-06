//! Posterior over the lower endpoint of the quenched-energy distribution.
//!
//! A basin-hopping run pays for one local minimisation per hop and reads a
//! single number off each: the quenched energy. The run then throws the number
//! away and keeps only the running minimum. That discards a sample from the
//! density of states of minima in whatever region of configuration space the
//! chain is currently in, and that sample answers a question the running
//! minimum cannot: *can this region reach the target at all*.
//!
//! # The model
//!
//! Write the quenched energies as `e_1, ..., e_m` and negate them, `y_i =
//! -e_i`, so the deepest minimum is the largest `y`. Let `F` be the
//! distribution of `Y` and `y_F = sup{y : F(y) < 1}` its upper endpoint.
//!
//! Pickands' theorem, and independently Balkema and de Haan's, says `F` lies in
//! the maximum domain of attraction of an extreme value law if and only if the
//! conditional excess distribution
//!
//! ```text
//! F_u(y) = P(Y - u <= y | Y > u)
//! ```
//!
//! converges uniformly, as `u -> y_F`, to a generalised Pareto distribution
//!
//! ```text
//! G(y; sigma, xi) = 1 - (1 + xi y / sigma)^(-1/xi),   xi != 0
//! G(y; sigma, 0)  = 1 - exp(-y / sigma)
//! ```
//!
//! supported on `y >= 0` when `xi >= 0` and on `0 <= y <= sigma / (-xi)` when
//! `xi < 0`. See Pickands, *Statistical inference using extreme order
//! statistics*, Ann. Statist. 3 (1975) 119-131,
//! [doi:10.1214/aos/1176343003](https://doi.org/10.1214/aos/1176343003), and
//! Balkema and de Haan, *Residual life time at great age*, Ann. Probab. 2
//! (1974) 792-804, [doi:10.1214/aop/1176996548](https://doi.org/10.1214/aop/1176996548).
//!
//! The shape `xi` is the whole of the question. With `xi < 0` the fitted law
//! has a finite upper endpoint
//!
//! ```text
//! theta_Y = u + sigma / (-xi),
//! ```
//!
//! which on the energy scale is a *lower* bound,
//!
//! ```text
//! theta_E = e_u - sigma / (-xi),      e_u = -u,
//! ```
//!
//! the deepest minimum the sampled region contains. With `xi >= 0` the fitted
//! law has no finite endpoint and the sample says nothing about a floor. Both
//! outcomes are reported; nothing here forces `xi` negative.
//!
//! # Why a posterior rather than a maximum likelihood point
//!
//! `theta_E` is a ratio of the two parameters and diverges as `xi -> 0-`, so
//! its sampling distribution is skewed however large the sample is, and a
//! plug-in `sigma_hat / (-xi_hat)` inherits none of that skew. Smith,
//! *Maximum likelihood estimation in a class of nonregular cases*, Biometrika
//! 72 (1985) 67-90, [doi:10.1093/biomet/72.1.67](https://doi.org/10.1093/biomet/72.1.67),
//! shows the likelihood theory itself fails here: the maximum likelihood
//! estimator is asymptotically normal only for `xi > -1/2`, has a non-normal
//! limit on `-1 < xi <= -1/2`, and does not exist at all for `xi <= -1`, which
//! is precisely the range where the endpoint is sharpest. The likelihood
//! remains a perfectly good likelihood throughout; it is the point summary
//! that stops meaning anything. The posterior is computed by deterministic
//! quadrature, so there is no sampler to diagnose.
//!
//! # The prior
//!
//! In `(log sigma, xi)` coordinates,
//!
//! ```text
//! pi(log sigma, xi)  propto  exp(-xi^2 / (2 tau^2)) . 1[xi_lo < xi < xi_hi]
//! ```
//!
//! with `tau = 0.5`, `xi_lo = -1`, `xi_hi = 2` by default. Two separate
//! choices:
//!
//! * `pi(sigma) propto 1/sigma`, flat in `log sigma`. This is the right-Haar
//!   prior for a scale parameter and the only one under which the endpoint
//!   posterior is equivariant: reporting in units of the pair well depth or in
//!   kJ/mol gives the same answer rescaled, rather than two different answers.
//! * A normal on the shape, centred at zero. Zero is the boundary between a
//!   region with a floor and a region without one, so centring there commits to
//!   neither. `tau = 0.5` puts 95 per cent of the prior mass on `|xi| < 0.98`,
//!   which covers the whole range shape estimates are reported in. The lower
//!   truncation at `-1` is where the likelihood stops being bounded (Smith
//!   1985) and discards prior mass 2.275e-2;
//!   [`TailPosterior::p_xi_floor`] reports how much posterior mass ends up
//!   against it, so a fit the truncation is driving is visible as such. The
//!   upper truncation discards 3.17e-5 and exists only to bound the
//!   quadrature box.
//!
//! The prior is proper, so the posterior is proper for any `k >= 1`; nothing
//! rests on the propriety arguments needed for the improper reference priors of
//! Northrop and Attalides, *Posterior propriety in Bayesian extreme value
//! analyses using reference priors*, Statist. Sinica 26 (2016) 721-743,
//! [doi:10.5705/ss.2014.034](https://doi.org/10.5705/ss.2014.034).
//!
//! # Threshold choice
//!
//! Under the model the endpoint does not depend on the threshold. If
//! `Y - u_0 | Y > u_0` is generalised Pareto with `(sigma_0, xi)`, then for
//! `u_1 > u_0` the excesses over `u_1` are generalised Pareto with the same
//! shape and scale `sigma_1 = sigma_0 + xi (u_1 - u_0)`, and
//!
//! ```text
//! u_1 + sigma_1 / (-xi) = u_0 + sigma_0 / (-xi).
//! ```
//!
//! So the endpoint posterior computed at every threshold in a ladder must agree
//! wherever the approximation has taken hold, and disagree below it. That is
//! the diagnostic [`ladder`] reports and the rule [`select_threshold`]
//! applies, in place of a percentile picked by convention.

use std::f64::consts::PI;

/// Two-parameter generalised Pareto law for excesses over a threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gpd {
    /// Scale, strictly positive.
    pub sigma: f64,
    /// Shape. Negative gives a finite upper endpoint.
    pub xi: f64,
}

/// Below this the shape is treated as zero and the exponential limit is used.
///
/// The series `(1 + 1/xi) ln(1 + xi z) = z + xi (z - z^2/2) + O(xi^2 z^3)` is
/// kept to first order, so the truncation costs `O(1e-12 k z^3)` in the log
/// likelihood, against `O(1/xi)` cancellation if the closed form were used.
const XI_SMALL: f64 = 1e-6;

impl Gpd {
    /// A law with the given scale and shape.
    pub fn new(sigma: f64, xi: f64) -> Self {
        Self { sigma, xi }
    }

    /// Upper endpoint of the support: `sigma / (-xi)` when `xi < 0`.
    pub fn support_upper(&self) -> f64 {
        if self.xi < 0.0 {
            self.sigma / (-self.xi)
        } else {
            f64::INFINITY
        }
    }

    /// Log density at an excess `y >= 0`.
    pub fn log_pdf(&self, y: f64) -> f64 {
        if !(self.sigma > 0.0) || y < 0.0 || y > self.support_upper() {
            return f64::NEG_INFINITY;
        }
        let z = y / self.sigma;
        if self.xi.abs() < XI_SMALL {
            -self.sigma.ln() - (z + self.xi * (z - 0.5 * z * z))
        } else {
            let t = self.xi * z;
            if t <= -1.0 {
                return f64::NEG_INFINITY;
            }
            -self.sigma.ln() - (1.0 + 1.0 / self.xi) * t.ln_1p()
        }
    }

    /// Distribution function at an excess `y`.
    pub fn cdf(&self, y: f64) -> f64 {
        if y <= 0.0 {
            return 0.0;
        }
        if y >= self.support_upper() {
            return 1.0;
        }
        let z = y / self.sigma;
        if self.xi.abs() < XI_SMALL {
            -(-z).exp_m1()
        } else {
            let t = self.xi * z;
            -(-t.ln_1p() / self.xi).exp_m1()
        }
    }

    /// Quantile of the excess distribution at probability `p` in `[0, 1)`.
    pub fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        if p >= 1.0 {
            return self.support_upper();
        }
        let l = -(1.0 - p).ln();
        if self.xi.abs() < XI_SMALL {
            // sigma/xi ((1-p)^-xi - 1) = sigma/xi (exp(xi l) - 1)
            //                          = sigma (l + xi l^2 / 2 + ...)
            self.sigma * (l + 0.5 * self.xi * l * l)
        } else {
            self.sigma * (self.xi * l).exp_m1() / self.xi
        }
    }

    /// Mean of the excess, finite only for `xi < 1`.
    pub fn mean(&self) -> Option<f64> {
        (self.xi < 1.0).then(|| self.sigma / (1.0 - self.xi))
    }

    /// The same law seen from a higher threshold, `shift` above this one.
    ///
    /// Returns `None` when the shift lands at or beyond the upper endpoint,
    /// where there is nothing left to condition on.
    pub fn raise_threshold(&self, shift: f64) -> Option<Gpd> {
        if shift < 0.0 || shift >= self.support_upper() {
            return None;
        }
        Some(Gpd::new(self.sigma + self.xi * shift, self.xi))
    }

    /// One draw, by inverting the distribution function at `u` in `(0, 1)`.
    pub fn sample_from_uniform(&self, u: f64) -> f64 {
        self.quantile(u)
    }
}

/// Log likelihood of generalised Pareto excesses.
///
/// Returns negative infinity outside the support, which for `xi < 0` is the
/// constraint `sigma > (-xi) max_i y_i`: the fitted endpoint can never sit
/// below an excess already observed.
pub fn log_likelihood(excess: &[f64], sigma: f64, xi: f64) -> f64 {
    if !(sigma > 0.0) || !sigma.is_finite() || excess.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut acc = 0.0f64;
    if xi.abs() < XI_SMALL {
        for &y in excess {
            if y < 0.0 {
                return f64::NEG_INFINITY;
            }
            let z = y / sigma;
            acc += z + xi * (z - 0.5 * z * z);
        }
    } else {
        let c = 1.0 + 1.0 / xi;
        for &y in excess {
            if y < 0.0 {
                return f64::NEG_INFINITY;
            }
            let t = xi * y / sigma;
            if t <= -1.0 {
                return f64::NEG_INFINITY;
            }
            acc += c * t.ln_1p();
        }
    }
    if !acc.is_finite() {
        return f64::NEG_INFINITY;
    }
    -(excess.len() as f64) * sigma.ln() - acc
}

/// Prior on `(log sigma, xi)`, flat in `log sigma` and truncated normal in `xi`.
#[derive(Debug, Clone, Copy)]
pub struct Prior {
    /// Standard deviation of the untruncated normal on the shape.
    pub xi_scale: f64,
    /// Lower truncation of the shape.
    pub xi_lo: f64,
    /// Upper truncation of the shape.
    pub xi_hi: f64,
}

impl Default for Prior {
    fn default() -> Self {
        Self {
            xi_scale: 0.5,
            xi_lo: -1.0,
            xi_hi: 2.0,
        }
    }
}

impl Prior {
    /// Log density in `(log sigma, xi)` coordinates, up to the normalising
    /// constant of the truncation, which cancels in every quantity reported.
    pub fn log_density(&self, xi: f64) -> f64 {
        if xi <= self.xi_lo || xi >= self.xi_hi {
            return f64::NEG_INFINITY;
        }
        -0.5 * (xi / self.xi_scale).powi(2)
    }

    /// Normalising constant of the truncated shape prior, for reporting the
    /// prior mass the truncation discards.
    pub fn shape_mass(&self) -> f64 {
        let phi = |x: f64| 0.5 * (1.0 + erf(x / 2.0_f64.sqrt()));
        phi(self.xi_hi / self.xi_scale) - phi(self.xi_lo / self.xi_scale)
    }
}

/// Abramowitz and Stegun 7.1.26, accurate to 1.5e-7 absolute, which is well
/// inside what a prior-mass report needs.
fn erf(x: f64) -> f64 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    s * y
}

/// Quadrature resolution for the two-dimensional posterior.
#[derive(Debug, Clone, Copy)]
pub struct GridSpec {
    /// Shape nodes on the refined grid.
    pub n_xi: usize,
    /// Log-scale nodes on the refined grid.
    pub n_log_sigma: usize,
    /// Drop in log posterior defining the refined box.
    pub log_drop: f64,
}

impl Default for GridSpec {
    fn default() -> Self {
        Self {
            n_xi: 161,
            n_log_sigma: 161,
            log_drop: 30.0,
        }
    }
}

/// One quadrature node with its normalised posterior weight.
#[derive(Debug, Clone, Copy)]
pub struct Node {
    /// Shape at the node.
    pub xi: f64,
    /// Scale at the node.
    pub sigma: f64,
    /// Normalised weight; the weights sum to one.
    pub weight: f64,
    /// Endpoint on the energy scale, negative infinity when `xi >= 0`.
    pub endpoint_energy: f64,
}

/// Posterior over `(sigma, xi)` and the endpoint it induces.
#[derive(Debug, Clone)]
pub struct TailPosterior {
    /// Threshold on the energy scale; exceedances are energies below it.
    pub energy_threshold: f64,
    /// Number of exceedances the fit used.
    pub n_exceedances: usize,
    /// Largest excess, a hard upper bound on the endpoint's depth.
    pub max_excess: f64,
    /// Quadrature nodes, weights normalised.
    pub nodes: Vec<Node>,
    /// Posterior probability that the shape is non-negative, so that the
    /// sampled region has no floor the sample can see.
    pub p_unbounded: f64,
    /// Posterior mass in the lowest shape cell of the refined grid.
    ///
    /// Large values mean the lower truncation of the prior is carrying the fit
    /// rather than the data.
    pub p_xi_floor: f64,
    /// Coarse-grid posterior mass falling outside the refined box.
    ///
    /// The quadrature's containment error; small by construction, reported so
    /// that it is checkable rather than assumed.
    pub box_leak: f64,
}

/// Excesses of the negated energies over a threshold set on the energy scale.
///
/// An energy `e` below `energy_threshold` contributes the excess
/// `energy_threshold - e`.
pub fn exceedances(energies: &[f64], energy_threshold: f64) -> Vec<f64> {
    energies
        .iter()
        .filter(|&&e| e < energy_threshold)
        .map(|&e| energy_threshold - e)
        .collect()
}

/// Runs declustering: within each run of exceedances separated by fewer than
/// `gap` non-exceedances, keep only the deepest.
///
/// A Metropolis chain on quenched energies revisits a basin several times
/// before it leaves, so consecutive exceedances are not independent draws and
/// the raw count `k` overstates the information the tail carries. Keeping one
/// value per excursion is the standard remedy; see Smith and Weissman,
/// *Estimating the extremal index*, J. R. Statist. Soc. B 56 (1994) 515-528,
/// <https://www.jstor.org/stable/2346107>.
pub fn decluster(energies: &[f64], energy_threshold: f64, gap: usize) -> Vec<f64> {
    let mut out = Vec::new();
    let mut best: Option<f64> = None;
    let mut since = 0usize;
    for &e in energies {
        if e < energy_threshold {
            since = 0;
            best = Some(best.map_or(e, |b: f64| b.min(e)));
        } else {
            since += 1;
            if since >= gap
                && let Some(b) = best.take()
            {
                out.push(energy_threshold - b);
            }
        }
    }
    if let Some(b) = best {
        out.push(energy_threshold - b);
    }
    out
}

/// Empirical quantile of a slice by the nearest-rank rule, on a copy.
pub fn quantile_of(values: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        return f64::NAN;
    }
    let i = ((q * (v.len() - 1) as f64).round() as usize).min(v.len() - 1);
    v[i]
}

/// Fits the posterior to excesses over a threshold, by grid quadrature.
///
/// Returns `None` when fewer than two exceedances are supplied or the excesses
/// are degenerate.
pub fn fit(
    excess: &[f64],
    energy_threshold: f64,
    prior: &Prior,
    grid: &GridSpec,
) -> Option<TailPosterior> {
    let k = excess.len();
    if k < 2 {
        return None;
    }
    let ymax = excess.iter().copied().fold(0.0_f64, f64::max);
    let mean = excess.iter().sum::<f64>() / k as f64;
    if !(ymax > 0.0) || !(mean > 0.0) {
        return None;
    }

    // Coarse pass, wide enough to hold the mode wherever it is. The scale must
    // be able to reach (-xi) ymax, which is the support constraint's floor, and
    // the mean, which is where the exponential fit sits.
    let lo_ls = mean.min(ymax).ln() - 6.0;
    let hi_ls = mean.max(ymax).ln() + 6.0;
    let (nx_c, ns_c) = (97usize, 121usize);
    let xi_lo = prior.xi_lo + 1e-6;
    let xi_hi = prior.xi_hi - 1e-6;
    let coarse = evaluate(excess, prior, xi_lo, xi_hi, nx_c, lo_ls, hi_ls, ns_c);
    let peak = coarse.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !peak.is_finite() {
        return None;
    }

    // Refined box: the smallest coarse cell rectangle holding everything within
    // `log_drop` of the peak, widened by one cell each way.
    let dxi = (xi_hi - xi_lo) / (nx_c - 1) as f64;
    let dls = (hi_ls - lo_ls) / (ns_c - 1) as f64;
    let (mut ia, mut ib, mut ja, mut jb) = (usize::MAX, 0usize, usize::MAX, 0usize);
    let mut inside = 0.0f64;
    let mut total = 0.0f64;
    for i in 0..nx_c {
        for j in 0..ns_c {
            let lp = coarse[i * ns_c + j];
            if !lp.is_finite() {
                continue;
            }
            let w = (lp - peak).exp();
            total += w;
            if lp > peak - grid.log_drop {
                inside += w;
                ia = ia.min(i);
                ib = ib.max(i);
                ja = ja.min(j);
                jb = jb.max(j);
            }
        }
    }
    if ia == usize::MAX {
        return None;
    }
    let box_leak = if total > 0.0 {
        (1.0 - inside / total).max(0.0)
    } else {
        1.0
    };
    let bx_lo = (xi_lo + (ia.saturating_sub(1)) as f64 * dxi).max(xi_lo);
    let bx_hi = (xi_lo + (ib + 1).min(nx_c - 1) as f64 * dxi).min(xi_hi);
    let bs_lo = lo_ls + (ja.saturating_sub(1)) as f64 * dls;
    let bs_hi = lo_ls + (jb + 1).min(ns_c - 1) as f64 * dls;

    let fine = evaluate(
        excess,
        prior,
        bx_lo,
        bx_hi,
        grid.n_xi,
        bs_lo,
        bs_hi,
        grid.n_log_sigma,
    );
    let fpeak = fine.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !fpeak.is_finite() {
        return None;
    }
    let mut nodes = Vec::with_capacity(grid.n_xi * grid.n_log_sigma);
    let mut norm = 0.0f64;
    for i in 0..grid.n_xi {
        let xi = lerp(bx_lo, bx_hi, i, grid.n_xi);
        for j in 0..grid.n_log_sigma {
            let lp = fine[i * grid.n_log_sigma + j];
            if !lp.is_finite() {
                continue;
            }
            let w = (lp - fpeak).exp();
            if w <= 0.0 {
                continue;
            }
            norm += w;
            let sigma = lerp(bs_lo, bs_hi, j, grid.n_log_sigma).exp();
            let endpoint_energy = if xi < 0.0 {
                energy_threshold + sigma / xi
            } else {
                f64::NEG_INFINITY
            };
            nodes.push(Node {
                xi,
                sigma,
                weight: w,
                endpoint_energy,
            });
        }
    }
    if !(norm > 0.0) {
        return None;
    }
    for n in &mut nodes {
        n.weight /= norm;
    }
    let p_unbounded = nodes
        .iter()
        .filter(|n| n.xi >= 0.0)
        .map(|n| n.weight)
        .sum();
    let floor_xi = lerp(bx_lo, bx_hi, 0, grid.n_xi);
    let p_xi_floor = if (floor_xi - xi_lo).abs() < 1.5 * dxi {
        nodes
            .iter()
            .filter(|n| n.xi <= lerp(bx_lo, bx_hi, 1, grid.n_xi))
            .map(|n| n.weight)
            .sum()
    } else {
        0.0
    };

    Some(TailPosterior {
        energy_threshold,
        n_exceedances: k,
        max_excess: ymax,
        nodes,
        p_unbounded,
        p_xi_floor,
        box_leak,
    })
}

fn lerp(lo: f64, hi: f64, i: usize, n: usize) -> f64 {
    if n <= 1 {
        return 0.5 * (lo + hi);
    }
    lo + (hi - lo) * i as f64 / (n - 1) as f64
}

#[allow(clippy::too_many_arguments)]
fn evaluate(
    excess: &[f64],
    prior: &Prior,
    xi_lo: f64,
    xi_hi: f64,
    n_xi: usize,
    ls_lo: f64,
    ls_hi: f64,
    n_ls: usize,
) -> Vec<f64> {
    let mut out = vec![f64::NEG_INFINITY; n_xi * n_ls];
    for i in 0..n_xi {
        let xi = lerp(xi_lo, xi_hi, i, n_xi);
        let lpr = prior.log_density(xi);
        if !lpr.is_finite() {
            continue;
        }
        for j in 0..n_ls {
            let sigma = lerp(ls_lo, ls_hi, j, n_ls).exp();
            out[i * n_ls + j] = log_likelihood(excess, sigma, xi) + lpr;
        }
    }
    out
}

impl TailPosterior {
    /// Posterior probability that the endpoint lies above `energy`, that is,
    /// that the sampled region contains no minimum as deep as `energy`.
    ///
    /// Nodes with `xi >= 0` carry an endpoint of negative infinity and never
    /// contribute: an unbounded tail excludes nothing.
    pub fn prob_endpoint_above(&self, energy: f64) -> f64 {
        self.nodes
            .iter()
            .filter(|n| n.endpoint_energy > energy)
            .map(|n| n.weight)
            .sum()
    }

    /// Equal-tailed quantile of the endpoint on the energy scale.
    ///
    /// Returns negative infinity when the requested quantile falls inside the
    /// unbounded-shape mass, which is the honest answer: no floor is implied.
    pub fn endpoint_quantile(&self, q: f64) -> f64 {
        let mut v: Vec<(f64, f64)> = self
            .nodes
            .iter()
            .map(|n| (n.endpoint_energy, n.weight))
            .collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut acc = 0.0;
        for (e, w) in v {
            acc += w;
            if acc >= q {
                return e;
            }
        }
        f64::INFINITY
    }

    /// Posterior mean of the shape.
    pub fn xi_mean(&self) -> f64 {
        self.nodes.iter().map(|n| n.weight * n.xi).sum()
    }

    /// Posterior mean of the scale.
    pub fn sigma_mean(&self) -> f64 {
        self.nodes.iter().map(|n| n.weight * n.sigma).sum()
    }

    /// Posterior mean of a function of the parameters.
    pub fn expect<F: Fn(f64, f64) -> f64>(&self, f: F) -> f64 {
        self.nodes.iter().map(|n| n.weight * f(n.sigma, n.xi)).sum()
    }

    /// Endpoint mass binned on `bins` equal cells spanning `[lo, hi]`.
    ///
    /// The first returned entry is the mass below `lo`, including the
    /// unbounded-shape atom, and the last is the mass above `hi`.
    pub fn endpoint_histogram(&self, lo: f64, hi: f64, bins: usize) -> Vec<f64> {
        let mut h = vec![0.0; bins + 2];
        let w = (hi - lo) / bins as f64;
        for n in &self.nodes {
            let e = n.endpoint_energy;
            if e < lo {
                h[0] += n.weight;
            } else if e >= hi {
                h[bins + 1] += n.weight;
            } else {
                let b = (((e - lo) / w) as usize).min(bins - 1);
                h[1 + b] += n.weight;
            }
        }
        h
    }
}

/// Overlap coefficient between two endpoint posteriors.
///
/// The common-support integral `int min(p, q)`, which is one for identical
/// posteriors and zero for disjoint ones. Used to say whether two thresholds
/// give the same answer, which under the model they must.
pub fn endpoint_overlap(a: &TailPosterior, b: &TailPosterior, bins: usize) -> f64 {
    let lo = a
        .endpoint_quantile(0.005)
        .min(b.endpoint_quantile(0.005))
        .max(-1e12);
    let hi = a
        .endpoint_quantile(0.995)
        .max(b.endpoint_quantile(0.995))
        .min(1e12);
    if !(hi > lo) {
        return 1.0;
    }
    let pad = 0.05 * (hi - lo);
    let (lo, hi) = (lo - pad, hi + pad);
    let ha = a.endpoint_histogram(lo, hi, bins);
    let hb = b.endpoint_histogram(lo, hi, bins);
    ha.iter().zip(hb.iter()).map(|(x, y)| x.min(*y)).sum()
}

/// One threshold in the stability ladder, with the fit it produced.
#[derive(Debug, Clone)]
pub struct Rung {
    /// Sample quantile the threshold was set at.
    pub quantile: f64,
    /// The fit at that threshold.
    pub posterior: TailPosterior,
    /// Overlap of this rung's endpoint posterior with the top rung's.
    pub overlap_top: f64,
}

/// Fits at a ladder of thresholds, for the stability diagnostic.
///
/// `quantiles` are lower-tail quantiles of the energy sample, so 0.2 means the
/// deepest fifth of the quenched energies. Rungs with fewer than `k_min`
/// exceedances are dropped.
pub fn ladder(
    energies: &[f64],
    quantiles: &[f64],
    k_min: usize,
    gap: usize,
    prior: &Prior,
    grid: &GridSpec,
) -> Vec<Rung> {
    let mut fits: Vec<(f64, TailPosterior)> = Vec::new();
    for &q in quantiles {
        let u = quantile_of(energies, q);
        let ex = if gap > 0 {
            decluster(energies, u, gap)
        } else {
            exceedances(energies, u)
        };
        if ex.len() < k_min {
            continue;
        }
        if let Some(p) = fit(&ex, u, prior, grid) {
            fits.push((q, p));
        }
    }
    if fits.is_empty() {
        return Vec::new();
    }
    let top = fits.len() - 1;
    let mut out = Vec::with_capacity(fits.len());
    for (i, (q, p)) in fits.iter().enumerate() {
        let ov = if i == top {
            1.0
        } else {
            endpoint_overlap(p, &fits[top].1, 256)
        };
        out.push(Rung {
            quantile: *q,
            posterior: p.clone(),
            overlap_top: ov,
        });
    }
    out
}

/// Picks the lowest threshold whose endpoint posterior agrees with every
/// higher one.
///
/// The bias-variance trade-off of threshold choice, made a stated rule rather
/// than an eyeballed plot: raising the threshold reduces the approximation
/// error of the generalised Pareto limit and increases the variance of the fit,
/// so the top rung is the least biased and the most diffuse. Under the model
/// the endpoint is threshold-invariant, so a lower rung is admissible exactly
/// while it still agrees with everything above it. Returns `None` when no rung
/// meets the bar.
pub fn select_threshold(rungs: &[Rung], min_overlap: f64) -> Option<usize> {
    if rungs.is_empty() {
        return None;
    }
    let top = rungs.len() - 1;
    for i in 0..rungs.len() {
        let ok = (i..=top).all(|l| {
            l == i || endpoint_overlap(&rungs[i].posterior, &rungs[l].posterior, 256) >= min_overlap
        });
        if ok {
            return Some(i);
        }
    }
    Some(top)
}

/// Normal density, for the prior report only.
pub fn normal_pdf(x: f64, sd: f64) -> f64 {
    (-0.5 * (x / sd).powi(2)).exp() / (sd * (2.0 * PI).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn draws(g: Gpd, k: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..k)
            .map(|_| g.sample_from_uniform(rng.random_range(1e-12..1.0 - 1e-12)))
            .collect()
    }

    /// The density integrates to one and the quantile inverts the cdf.
    #[test]
    fn gpd_is_a_distribution() {
        for &xi in &[-0.9, -0.5, -0.2, 0.0, 0.3, 0.8] {
            let g = Gpd::new(1.3, xi);
            // Trapezoid of the density against the distribution function on a
            // bounded window, which tests the pair against each other rather
            // than either against a constant. The window is the 0.999 quantile
            // so the heavy-tailed shapes are integrated where they have mass
            // instead of over an interval the step size cannot resolve.
            let hi = g.quantile(0.999);
            let n = 200_001usize;
            let h = hi / (n - 1) as f64;
            let mut s = 0.0f64;
            for i in 0..n {
                let y = i as f64 * h;
                let w = if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
                let d = g.log_pdf(y).exp();
                if d.is_finite() {
                    s += w * d;
                }
            }
            s *= h;
            assert!(
                (s - 0.999).abs() < 2e-5,
                "xi {xi}: density integrates to {s:.8} where the cdf says 0.999"
            );
            for &p in &[0.01, 0.25, 0.5, 0.9, 0.999] {
                let y = g.quantile(p);
                assert!(
                    (g.cdf(y) - p).abs() < 1e-10,
                    "xi {xi} p {p}: cdf(quantile) = {}",
                    g.cdf(y)
                );
            }
        }
    }

    /// The endpoint, the mean and the support constraint match their closed
    /// forms.
    #[test]
    fn gpd_closed_forms() {
        let g = Gpd::new(2.0, -0.4);
        assert!((g.support_upper() - 5.0).abs() < 1e-12);
        assert!((g.cdf(5.0) - 1.0).abs() < 1e-12);
        assert!((g.mean().unwrap() - 2.0 / 1.4).abs() < 1e-12);
        assert!(Gpd::new(1.0, 1.5).mean().is_none());
        // Exponential limit.
        let e = Gpd::new(0.7, 0.0);
        assert!((e.cdf(1.4) - (1.0 - (-2.0f64).exp())).abs() < 1e-12);
        assert!((e.quantile(0.5) - 0.7 * 2.0f64.ln()).abs() < 1e-12);
    }

    /// Raising the threshold changes the scale by `xi . shift` and leaves the
    /// endpoint where it was.
    #[test]
    fn threshold_invariance_of_the_endpoint() {
        let g = Gpd::new(2.0, -0.4);
        let u0 = 10.0;
        let theta = u0 + g.support_upper();
        for shift in [0.5, 1.5, 3.0, 4.9] {
            let h = g.raise_threshold(shift).unwrap();
            assert!((h.xi - g.xi).abs() < 1e-15);
            assert!((h.sigma - (g.sigma + g.xi * shift)).abs() < 1e-14);
            let theta_shift = u0 + shift + h.support_upper();
            assert!(
                (theta_shift - theta).abs() < 1e-12,
                "shift {shift}: endpoint moved to {theta_shift} from {theta}"
            );
            // And the conditional excess law is the one `raise_threshold` names.
            for y in [0.1, 0.7, 2.0] {
                if y >= h.support_upper() {
                    continue;
                }
                let cond = (g.cdf(shift + y) - g.cdf(shift)) / (1.0 - g.cdf(shift));
                assert!(
                    (cond - h.cdf(y)).abs() < 1e-12,
                    "shift {shift} y {y}: {cond} against {}",
                    h.cdf(y)
                );
            }
        }
    }

    /// With the shape pinned at zero the posterior for the scale is inverse
    /// gamma, and the quadrature must reproduce its moments.
    #[test]
    fn exponential_slice_is_inverse_gamma() {
        let g = Gpd::new(1.7, 0.0);
        let y = draws(g, 400, 11);
        let k = y.len() as f64;
        let s: f64 = y.iter().sum();
        // p(sigma) propto sigma^-(k+1) exp(-S/sigma): InvGamma(k, S).
        let (mean, var) = (s / (k - 1.0), s * s / ((k - 1.0).powi(2) * (k - 2.0)));
        // One-dimensional quadrature in log sigma with the flat prior there.
        let (lo, hi, n) = ((s / k).ln() - 4.0, (s / k).ln() + 4.0, 40_001usize);
        let mut z = 0.0f64;
        let mut m1 = 0.0;
        let mut m2 = 0.0;
        let peak = -k * (s / k).ln() - k;
        for i in 0..n {
            let ls = lerp(lo, hi, i, n);
            let sig = ls.exp();
            let w = (log_likelihood(&y, sig, 0.0) - peak).exp();
            z += w;
            m1 += w * sig;
            m2 += w * sig * sig;
        }
        let (qm, qv) = (m1 / z, m2 / z - (m1 / z).powi(2));
        assert!(
            (qm / mean - 1.0).abs() < 1e-6,
            "quadrature mean {qm:.9} against {mean:.9}"
        );
        assert!(
            (qv / var - 1.0).abs() < 1e-4,
            "quadrature variance {qv:.9} against {var:.9}"
        );
    }

    /// With the shape pinned at minus one the law is uniform and the endpoint
    /// posterior is Pareto with closed-form quantiles.
    #[test]
    fn uniform_slice_gives_a_pareto_endpoint() {
        let g = Gpd::new(3.0, -1.0);
        let y = draws(g, 250, 5);
        let k = y.len() as f64;
        let ymax = y.iter().copied().fold(0.0f64, f64::max);
        // Log likelihood must be exactly -k log sigma above ymax and -inf below.
        assert!(log_likelihood(&y, ymax * 0.999, -1.0).is_infinite());
        let l = log_likelihood(&y, ymax * 1.5, -1.0);
        assert!(
            (l + k * (ymax * 1.5).ln()).abs() < 1e-9,
            "uniform log likelihood {l}"
        );
        // p(sigma) propto sigma^-(k+1) on sigma > ymax, so
        // P(sigma <= s) = 1 - (ymax/s)^k and the q-quantile is
        // ymax (1-q)^(-1/k).
        let (lo, hi, n) = (ymax.ln(), ymax.ln() + 3.0, 400_001usize);
        let mut z = 0.0f64;
        let mut acc = vec![0.0; n];
        for i in 0..n {
            let sig = lerp(lo, hi, i, n).exp();
            let w = if sig > ymax {
                (-k * (sig / ymax).ln()).exp()
            } else {
                0.0
            };
            z += w;
            acc[i] = z;
        }
        for &q in &[0.1f64, 0.5, 0.9, 0.99] {
            let want = ymax * (1.0 - q).powf(-1.0 / k);
            let i = acc.partition_point(|&a| a < q * z).min(n - 1);
            let got = lerp(lo, hi, i, n).exp();
            assert!(
                (got / want - 1.0).abs() < 1e-4,
                "q {q}: quadrature {got:.9} against Pareto {want:.9}"
            );
        }
    }

    /// The general log likelihood agrees with the exponential and uniform
    /// special cases evaluated directly.
    #[test]
    fn log_likelihood_matches_special_cases() {
        let y = draws(Gpd::new(1.0, -0.3), 60, 3);
        let sigma = 1.4f64;
        let direct: f64 = y.iter().map(|v| -sigma.ln() - v / sigma).sum();
        assert!((log_likelihood(&y, sigma, 0.0) - direct).abs() < 1e-9);
        // Continuity in xi through zero.
        let a = log_likelihood(&y, sigma, -1e-9);
        let b = log_likelihood(&y, sigma, 1e-9);
        assert!(
            (a - b).abs() < 1e-6,
            "log likelihood jumps across xi = 0: {a} against {b}"
        );
        let c = log_likelihood(&y, sigma, 1e-3);
        let d: f64 = y.iter().map(|v| Gpd::new(sigma, 1e-3).log_pdf(*v)).sum();
        assert!((c - d).abs() < 1e-9);
    }

    /// The fitted posterior recovers a known endpoint from a large sample.
    #[test]
    fn recovers_a_known_endpoint() {
        let g = Gpd::new(1.0, -0.35);
        let truth_excess = g.support_upper(); // 1/0.35
        let y = draws(g, 4000, 21);
        let p = fit(&y, 0.0, &Prior::default(), &GridSpec::default()).unwrap();
        // On the energy scale with u = 0 the endpoint is -support_upper.
        let med = -p.endpoint_quantile(0.5);
        assert!(
            (med / truth_excess - 1.0).abs() < 0.05,
            "median endpoint {med:.4} against {truth_excess:.4}"
        );
        let (lo, hi) = (-p.endpoint_quantile(0.975), -p.endpoint_quantile(0.025));
        assert!(
            lo < truth_excess && truth_excess < hi,
            "95 per cent interval [{lo:.4}, {hi:.4}] misses {truth_excess:.4}"
        );
        assert!(
            p.p_unbounded < 1e-3,
            "a clearly bounded sample reported P(unbounded) = {:.4}",
            p.p_unbounded
        );
        assert!(p.box_leak < 1e-6, "quadrature box leaks {:.3e}", p.box_leak);
        assert!(
            p.p_xi_floor < 1e-6,
            "posterior piled against the shape truncation: {:.3e}",
            p.p_xi_floor
        );
    }

    /// An unbounded sample is not given a floor. The false-positive control:
    /// exponential data have no endpoint, and the posterior must say so.
    #[test]
    fn refuses_an_endpoint_when_there_is_none() {
        let y = draws(Gpd::new(1.0, 0.0), 2000, 33);
        let p = fit(&y, 0.0, &Prior::default(), &GridSpec::default()).unwrap();
        assert!(
            p.p_unbounded > 0.2,
            "exponential data reported P(unbounded) = {:.4}",
            p.p_unbounded
        );
        assert!(
            p.xi_mean().abs() < 0.1,
            "exponential data gave shape mean {:.4}",
            p.xi_mean()
        );
    }

    /// The fitted endpoint does not move with the threshold when the data are
    /// generalised Pareto, which is what the stability diagnostic rests on.
    #[test]
    fn fitted_endpoint_is_threshold_stable() {
        let g = Gpd::new(1.0, -0.4);
        let y = draws(g, 6000, 44);
        // Energies: e = -y, so the endpoint on the energy scale is
        // -support_upper = -2.5.
        let e: Vec<f64> = y.iter().map(|v| -v).collect();
        let mut meds = Vec::new();
        for &q in &[0.5, 0.3, 0.15, 0.07] {
            let u = quantile_of(&e, q);
            let ex = exceedances(&e, u);
            let p = fit(&ex, u, &Prior::default(), &GridSpec::default()).unwrap();
            meds.push(p.endpoint_quantile(0.5));
        }
        let truth = -g.support_upper();
        for m in &meds {
            assert!(
                (m - truth).abs() < 0.12,
                "threshold ladder gave {m:.4} against {truth:.4}; ladder {meds:?}"
            );
        }
    }

    /// Credible intervals for the endpoint cover at their stated rate.
    ///
    /// The one test that checks the whole construction rather than a piece of
    /// it: prior, likelihood, quadrature and the nonlinear map to the endpoint.
    #[test]
    fn endpoint_intervals_cover() {
        let g = Gpd::new(1.0, -0.3);
        let truth = -g.support_upper();
        let grid = GridSpec {
            n_xi: 81,
            n_log_sigma: 81,
            log_drop: 25.0,
        };
        let (mut hit, mut n) = (0usize, 0usize);
        for s in 0..120u64 {
            let y = draws(g, 400, 900 + s);
            let Some(p) = fit(&y, 0.0, &Prior::default(), &grid) else {
                continue;
            };
            let (lo, hi) = (p.endpoint_quantile(0.05), p.endpoint_quantile(0.95));
            if lo <= truth && truth <= hi {
                hit += 1;
            }
            n += 1;
        }
        let rate = hit as f64 / n as f64;
        // Nominal 0.90 with 120 replicates: a binomial standard error of 0.027,
        // so a three-error band either side.
        assert!(
            (0.80..=0.98).contains(&rate),
            "90 per cent intervals covered {hit}/{n} = {rate:.3}"
        );
    }

    /// Declustering keeps one value per excursion.
    #[test]
    fn declustering_keeps_one_per_excursion() {
        // Two excursions below -1, separated by three values above it.
        let e = [0.0, -1.5, -1.2, -1.8, 0.5, 0.5, 0.5, -1.4, -1.9, 0.2];
        let d = decluster(&e, -1.0, 3);
        assert_eq!(d.len(), 2, "declustered to {d:?}");
        assert!((d[0] - 0.8).abs() < 1e-12, "{d:?}"); // -1.0 - (-1.8)
        assert!((d[1] - 0.9).abs() < 1e-12, "{d:?}"); // -1.0 - (-1.9)
        // Raw exceedances keep all five.
        assert_eq!(exceedances(&e, -1.0).len(), 5);
    }

    /// The overlap coefficient is one against itself and small against a
    /// posterior somewhere else.
    #[test]
    fn overlap_separates_disagreeing_fits() {
        let a = fit(
            &draws(Gpd::new(1.0, -0.4), 1500, 61),
            0.0,
            &Prior::default(),
            &GridSpec::default(),
        )
        .unwrap();
        let b = fit(
            &draws(Gpd::new(4.0, -0.4), 1500, 62),
            0.0,
            &Prior::default(),
            &GridSpec::default(),
        )
        .unwrap();
        assert!((endpoint_overlap(&a, &a, 256) - 1.0).abs() < 1e-9);
        assert!(
            endpoint_overlap(&a, &b, 256) < 0.05,
            "endpoints 2.5 and 10 overlapped {:.4}",
            endpoint_overlap(&a, &b, 256)
        );
    }

    /// The support constraint bars an endpoint shallower than an observed
    /// minimum, at every node the quadrature keeps.
    #[test]
    fn endpoint_never_sits_above_the_deepest_sample() {
        let y = draws(Gpd::new(1.0, -0.5), 800, 77);
        let ymax = y.iter().copied().fold(0.0f64, f64::max);
        let p = fit(&y, 0.0, &Prior::default(), &GridSpec::default()).unwrap();
        for n in &p.nodes {
            if n.xi < 0.0 {
                assert!(
                    n.endpoint_energy <= -ymax + 1e-9,
                    "node xi {:.4} sigma {:.4} put the endpoint at {:.6}, above the \
                     deepest sample -{ymax:.6}",
                    n.xi,
                    n.sigma,
                    n.endpoint_energy
                );
            }
        }
        assert!(p.prob_endpoint_above(-ymax) < 1e-12);
    }

    /// The prior integrates and its truncation discards the mass claimed.
    #[test]
    fn prior_truncation_is_what_it_says() {
        let p = Prior::default();
        assert!(p.log_density(-1.0).is_infinite());
        assert!(p.log_density(2.0).is_infinite());
        assert!((p.log_density(0.5) + 0.5).abs() < 1e-12);
        // Mass outside (-1, 2) under N(0, 0.5^2).
        // Phi(-2) below the cut plus 1 - Phi(4) above it.
        let outside = 1.0 - p.shape_mass();
        assert!(
            (outside - 2.278e-2).abs() < 2e-4,
            "truncation discards {outside:.5}"
        );
        // Of which the part above the upper cut.
        let above = 0.5 * (1.0 - erf(2.0 / 0.5 / 2.0_f64.sqrt()));
        assert!(
            (above - 3.17e-5).abs() < 1e-6,
            "upper truncation discards {above:.3e}"
        );
    }

    /// Grid refinement changes the reported endpoint by less than the number
    /// is quoted to.
    #[test]
    fn quadrature_is_converged() {
        let y = draws(Gpd::new(1.0, -0.35), 2000, 88);
        let coarse = fit(
            &y,
            0.0,
            &Prior::default(),
            &GridSpec {
                n_xi: 81,
                n_log_sigma: 81,
                log_drop: 30.0,
            },
        )
        .unwrap();
        let fine = fit(
            &y,
            0.0,
            &Prior::default(),
            &GridSpec {
                n_xi: 321,
                n_log_sigma: 321,
                log_drop: 30.0,
            },
        )
        .unwrap();
        for q in [0.05, 0.5, 0.95] {
            let (a, b) = (coarse.endpoint_quantile(q), fine.endpoint_quantile(q));
            assert!(
                (a - b).abs() < 5e-3,
                "q {q}: 81 nodes gave {a:.6}, 321 nodes gave {b:.6}"
            );
        }
        assert!((coarse.xi_mean() - fine.xi_mean()).abs() < 2e-3);
    }
}
