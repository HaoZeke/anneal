"""Dimension-collapse + Chebyshev surrogate for the CUTEst drivers.

This is the Python side of the typed objective transform. A high-dimensional
objective is collapsed onto a low-dimensional active
subspace and, after a pilot phase, replaced by a cheap total-degree
Chebyshev model with an analytic gradient.

The point of doing it at the objective slot is the SA / MCMC unification: a
single reduced objective drops into the shared `CutestProblem.fn`/`.grad`
chokepoint, so classical SA, the MCMC variants, parallel tempering, the
Bayesian mixer, and the HMC point all run in the reduced space unmodified.
The same collapse that makes Chebyshev approximation tractable in high
dimension is therefore inherited by every point of the algebra at once.

The Chebyshev evaluation here mirrors the Rust `ChebyshevSurrogate`
(`to_unit`, the first-kind recurrence, total-degree truncation) so the two
implementations agree numerically.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


def total_degree_terms(k: int, degree: int) -> list[tuple[int, ...]]:
    """Multi-indices over `k` dimensions with total degree at most `degree`.

    Total-degree truncation keeps the basis size polynomial in `k` rather
    than the `(degree+1)**k` of a tensor-product grid, which is what makes a
    Chebyshev model usable after the active subspace has collapsed the search
    to a few coordinates.
    """
    terms = []
    for idx in product(range(degree + 1), repeat=k):
        if sum(idx) <= degree:
            terms.append(idx)
    return terms


def _cheb_tables(t: np.ndarray, max_deg: int) -> tuple[np.ndarray, np.ndarray]:
    """First-kind Chebyshev values and `d/dt` derivatives up to `max_deg`.

    `t` has shape (N,); returns two (N, max_deg+1) arrays via the recurrences
    `T_{m+1} = 2 t T_m - T_{m-1}` and `T'_{m+1} = 2 T_m + 2 t T'_m - T'_{m-1}`.
    """
    n = t.shape[0]
    vals = np.zeros((n, max_deg + 1))
    ders = np.zeros((n, max_deg + 1))
    vals[:, 0] = 1.0
    if max_deg >= 1:
        vals[:, 1] = t
        ders[:, 1] = 1.0
    for m in range(1, max_deg):
        vals[:, m + 1] = 2.0 * t * vals[:, m] - vals[:, m - 1]
        ders[:, m + 1] = 2.0 * vals[:, m] + 2.0 * t * ders[:, m] - ders[:, m - 1]
    return vals, ders


@dataclass
class ChebyshevSurrogate:
    """Total-degree Chebyshev model on a box `[low, high]` in R^k."""

    low: np.ndarray
    high: np.ndarray
    terms: list[tuple[int, ...]]
    coeffs: np.ndarray

    @property
    def k(self) -> int:
        return self.low.shape[0]

    def _to_unit(self, x: np.ndarray) -> np.ndarray:
        """Maps box coordinates to [-1, 1] columnwise; degenerate spans -> 0."""
        span = self.high - self.low
        safe = np.where(np.abs(span) <= np.finfo(float).eps, 1.0, span)
        t = 2.0 * (x - self.low) / safe - 1.0
        t = np.where(np.abs(span) <= np.finfo(float).eps, 0.0, t)
        return np.clip(t, -1.0, 1.0)

    def _design(self, x2: np.ndarray):
        """Design matrix (and per-dim tables) for points `x2` of shape (N, k)."""
        k = self.k
        max_deg = [max((term[d] for term in self.terms), default=0) for d in range(k)]
        t = self._to_unit(x2)
        tabs = [_cheb_tables(t[:, d], max_deg[d]) for d in range(k)]
        vals = [v for v, _ in tabs]
        ders = [d for _, d in tabs]
        n = x2.shape[0]
        design = np.ones((n, len(self.terms)))
        for j, term in enumerate(self.terms):
            for d in range(k):
                design[:, j] *= vals[d][:, term[d]]
        return design, vals, ders

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, low, high, degree: int) -> "ChebyshevSurrogate":
        """Least-squares fit on reduced samples `x` (N, k) and values `y` (N,)."""
        low = np.asarray(low, dtype=float).reshape(-1)
        high = np.asarray(high, dtype=float).reshape(-1)
        terms = total_degree_terms(low.shape[0], degree)
        self = cls(low=low, high=high, terms=terms, coeffs=np.zeros(len(terms)))
        design, _, _ = self._design(np.atleast_2d(x))
        coeffs, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=float).reshape(-1), rcond=None)
        self.coeffs = coeffs
        return self

    def eval(self, x: np.ndarray) -> float:
        """Surrogate value at a single reduced point `x` of shape (k,)."""
        x2 = np.atleast_2d(np.asarray(x, dtype=float).reshape(-1))
        design, _, _ = self._design(x2)
        return float((design @ self.coeffs).ravel()[0])

    def grad(self, x: np.ndarray) -> np.ndarray:
        """Analytic gradient at a single reduced point `x` of shape (k,)."""
        k = self.k
        x2 = np.atleast_2d(np.asarray(x, dtype=float).reshape(-1))
        _, vals, ders = self._design(x2)
        span = self.high - self.low
        g = np.zeros(k)
        for term, c in zip(self.terms, self.coeffs):
            for j in range(k):
                if abs(span[j]) <= np.finfo(float).eps:
                    continue
                prod = ders[j][0, term[j]] * (2.0 / span[j])
                for d in range(k):
                    if d != j:
                        prod *= vals[d][0, term[d]]
                g[j] += c * prod
        return g


@dataclass
class ActiveSubspaceEncoder:
    """Affine encode/decode between full R^n and a reduced R^k box.

    `decode(r) = clip(origin + basis @ r, full_low, full_high)` and
    `encode(x) = basis.T @ (x - origin)`. The basis columns span the retained
    subspace: the dominant eigenvectors of the pilot gradient covariance
    (active subspace) when gradients are available, otherwise the leading PCA
    directions of the pilot points.
    """

    origin: np.ndarray
    basis: np.ndarray
    full_low: np.ndarray
    full_high: np.ndarray

    def lift(self, r: np.ndarray) -> np.ndarray:
        """Affine lift `origin + basis @ r` into the full space, without box clipping."""
        return self.origin + self.basis @ np.asarray(r, dtype=float).reshape(-1)

    def decode(self, r: np.ndarray) -> np.ndarray:
        return np.clip(self.lift(r), self.full_low, self.full_high)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.basis.T @ (np.asarray(x, dtype=float).reshape(-1) - self.origin)

    @classmethod
    def from_gradients(cls, origin, grads, low, high, k):
        """Active subspace from a stack of pilot gradients `grads` (N, n)."""
        grads = np.atleast_2d(np.asarray(grads, dtype=float))
        finite = np.all(np.isfinite(grads), axis=1)
        grads = grads[finite]
        if grads.size == 0:
            raise ValueError("active-subspace gradients must contain a finite row")

        row_scale = np.max(np.abs(grads), axis=1)
        usable = row_scale > 0.0
        grads = grads[usable]
        row_scale = row_scale[usable]
        if grads.size == 0:
            raise ValueError("active-subspace gradients must contain a nonzero row")

        scaled = grads / row_scale[:, None]
        row_norm = np.linalg.norm(scaled, axis=1)
        usable = row_norm > 0.0
        scaled = scaled[usable] / row_norm[usable, None]
        if scaled.size == 0:
            raise ValueError("active-subspace gradients must define a finite direction")

        cov = scaled.T @ scaled / scaled.shape[0]
        if not np.all(np.isfinite(cov)):
            raise ValueError("active-subspace gradient covariance is not finite")
        evals, evecs = np.linalg.eigh(cov)
        basis = evecs[:, ::-1][:, :k]
        return cls(np.asarray(origin, float), basis,
                   np.asarray(low, float), np.asarray(high, float))

    @classmethod
    def from_samples(cls, samples, low, high, k):
        """PCA fallback: leading directions of the pilot points `samples`."""
        samples = np.atleast_2d(samples)
        origin = samples.mean(axis=0)
        centered = samples - origin
        cov = centered.T @ centered / max(1, samples.shape[0] - 1)
        evals, evecs = np.linalg.eigh(cov)
        basis = evecs[:, ::-1][:, :k]
        return cls(origin, basis, np.asarray(low, float), np.asarray(high, float))


@dataclass
class ReducedFit:
    """A fitted reduction: the encoder, the surrogate, and the pilot cost."""

    encoder: ActiveSubspaceEncoder
    surrogate: ChebyshevSurrogate
    reduced_low: np.ndarray
    reduced_high: np.ndarray
    pilot_work_units: float


def fit_reduced_surrogate(fn, grad, low, high, dim, *, k=3, degree=6,
                          n_pilot=None, pad=0.1, rng=None) -> ReducedFit:
    """Collapse `(fn, grad, low, high)` onto `k` dims and fit a degree-`degree`
    Chebyshev surrogate from a uniform pilot sample.

    Returns a `ReducedFit`. `pilot_work_units` is the objective-equivalent
    cost of the pilot in the common work unit (one true eval = one unit; a
    gradient costs one unit when native, else `dim + 1`).
    """
    rng = np.random.default_rng(rng)
    low = np.asarray(low, dtype=float).reshape(-1)
    high = np.asarray(high, dtype=float).reshape(-1)
    k = min(k, dim)
    if n_pilot is None:
        n_pilot = max(64, 8 * len(total_degree_terms(k, degree)))

    samples = rng.uniform(low, high, size=(n_pilot, dim))
    values = np.array([fn(x) for x in samples], dtype=float)
    work = float(n_pilot)

    finite_values = np.isfinite(values)
    if not np.any(finite_values):
        raise ValueError("surrogate pilot must produce at least one finite objective value")
    fit_samples = samples[finite_values]
    fit_values = values[finite_values]

    origin = 0.5 * (low + high)
    if grad is not None:
        grads = np.array([grad(x) for x in samples], dtype=float)
        try:
            encoder = ActiveSubspaceEncoder.from_gradients(origin, grads, low, high, k)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            encoder = ActiveSubspaceEncoder.from_samples(fit_samples, low, high, k)
        work += float(n_pilot)  # native gradient: one unit each
    else:
        encoder = ActiveSubspaceEncoder.from_samples(fit_samples, low, high, k)

    reduced = np.array([encoder.encode(x) for x in fit_samples])
    r_low = reduced.min(axis=0)
    r_high = reduced.max(axis=0)
    spread = np.where(r_high > r_low, r_high - r_low, 1.0)
    r_low = r_low - pad * spread
    r_high = r_high + pad * spread

    surrogate = ChebyshevSurrogate.fit(reduced, fit_values, r_low, r_high, degree)
    return ReducedFit(encoder, surrogate, r_low, r_high, work)


def _self_test():
    """Collapse a 10-D ridge function (2 active directions) and check the fit.

    Active-subspace reduction targets objectives with low-dimensional
    structure; a ridge `f(x) = g(W^T x)` is the canonical case. (High-rank
    objectives such as Rosenbrock degrade gracefully: the surrogate becomes a
    coarse search guide and the reported best is still a true objective
    evaluation at the decoded point.)
    """
    dim = 10
    rng_w = np.random.default_rng(7)
    w = rng_w.standard_normal((dim, 2))
    w, _ = np.linalg.qr(w)  # two orthonormal active directions

    def ridge(x):
        a = w.T @ np.asarray(x, float)
        return float(a[0] ** 2 - 0.3 * a[1] ** 2 + 0.5 * a[0] * a[1])

    def ridge_grad(x):
        a = w.T @ np.asarray(x, float)
        da = np.array([2.0 * a[0] + 0.5 * a[1], -0.6 * a[1] + 0.5 * a[0]])
        return w @ da

    low = np.full(dim, -2.0)
    high = np.full(dim, 2.0)
    fit = fit_reduced_surrogate(ridge, ridge_grad, low, high, dim,
                                k=2, degree=4, n_pilot=800, rng=0)
    rng = np.random.default_rng(1)
    # Held-out fit quality on the active subspace (unclipped lift; the clip in
    # decode() is a feasibility safeguard for the driver, tested separately).
    test = rng.uniform(fit.reduced_low, fit.reduced_high, size=(400, fit.surrogate.k))
    true = np.array([ridge(fit.encoder.lift(r)) for r in test])
    pred = np.array([fit.surrogate.eval(r) for r in test])
    rel = np.linalg.norm(pred - true) / np.linalg.norm(true)
    # Analytic gradient vs finite difference at a probe point.
    r0 = test[0]
    g = fit.surrogate.grad(r0)
    h = 1e-5
    fd = np.zeros_like(g)
    for j in range(len(r0)):
        rp, rm = r0.copy(), r0.copy()
        rp[j] += h
        rm[j] -= h
        fd[j] = (fit.surrogate.eval(rp) - fit.surrogate.eval(rm)) / (2.0 * h)
    gerr = np.linalg.norm(g - fd) / (np.linalg.norm(fd) + 1e-12)
    print(f"reduced k={fit.surrogate.k} terms={len(fit.surrogate.terms)} "
          f"pilot_work={fit.pilot_work_units:.0f}")
    print(f"held-out relative surrogate error = {rel:.4f}")
    print(f"gradient relative error vs FD     = {gerr:.2e}")
    assert rel < 0.02, f"surrogate fit too poor on a ridge: {rel}"
    assert gerr < 1e-4, f"gradient mismatch: {gerr}"
    print("self-test OK")


if __name__ == "__main__":
    _self_test()
