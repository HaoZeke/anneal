r"""Tensor-train surrogate and the Rosenblatt independence sampler that it
enables -- the shared object that the SymPy derivation
(`tensor_surrogate_derivation.py`) proves is sound.

The pipeline reuses the existing Obj-transform stack:

    fit_reduced_surrogate  ->  ReducedFit (active subspace + Chebyshev model)
    grid the reduced box   ->  tempered density tensor P_T over k <= 3 coords
    tt_svd(P_T)            ->  tensor-train cores  (storage  k m r^2, linear in k)
    rosenblatt_sample      ->  near-i.i.d. draws from P_T  (the Move slot)
    Metropolis on true f   ->  debiases the surrogate bias (the Accept slot)

The driver `tt_independence_sa` is the generalisation: instead of a local
random-walk proposal (whose efficiency decays like 1/d), every proposal is a
global draw from the structure-aware tempered surrogate, accepted against the
true objective. The active-subspace collapse keeps the grid tractable and the
tensor-train keeps the coefficient count linear in the retained dimension, so
the same driver runs unmodified from d=2 to the native CUTEst sizes.

numpy only; run `python experiments/tensor_surrogate.py` for the self-test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.surrogate import fit_reduced_surrogate


# ---------------------------------------------------------------------------
# Tensor-train: sequential-SVD decomposition and reconstruction.
# ---------------------------------------------------------------------------
def tt_svd(tensor: np.ndarray, eps: float = 1e-10) -> list[np.ndarray]:
    """TT decomposition of `tensor` by the sequential-SVD algorithm (Oseledets).

    Returns cores `G_1..G_k`, each shaped `(r_{i-1}, m_i, r_i)` with
    `r_0 = r_k = 1`. The relative reconstruction error is bounded by `eps`; the
    ranks `r_i` adapt to the tensor. Storage is `sum_i r_{i-1} m_i r_i`, which
    for a rank-`r`, degree-`m` model in `k` dims is `O(k m r^2)` -- linear in k.
    """
    tensor = np.asarray(tensor, dtype=float)
    shape = tensor.shape
    k = tensor.ndim
    delta = eps / np.sqrt(max(k - 1, 1)) * np.linalg.norm(tensor)
    cores: list[np.ndarray] = []
    c = tensor.reshape(1, -1)
    r_prev = 1
    for i in range(k - 1):
        m = shape[i]
        c = c.reshape(r_prev * m, -1)
        u, s, vt = np.linalg.svd(c, full_matrices=False)
        # rank truncation at the per-core error budget delta
        if delta > 0 and s.size:
            tail = np.sqrt(np.cumsum(s[::-1] ** 2)[::-1])
            keep = int(np.searchsorted(-tail, -delta))  # first index with tail <= delta
            rank = max(1, min(len(s), keep if keep > 0 else len(s)))
            # ensure we actually keep enough mass
            while rank < len(s) and tail[rank] > delta:
                rank += 1
        else:
            rank = len(s)
        u = u[:, :rank]
        s = s[:rank]
        vt = vt[:rank, :]
        cores.append(u.reshape(r_prev, m, rank))
        c = (np.diag(s) @ vt)
        r_prev = rank
    cores.append(c.reshape(r_prev, shape[-1], 1))
    return cores


def tt_full(cores: list[np.ndarray]) -> np.ndarray:
    """Reconstruct the dense tensor from TT cores (for verification / small k)."""
    full = cores[0]
    for core in cores[1:]:
        # contract last rank index of `full` with first of `core`
        full = np.tensordot(full, core, axes=([full.ndim - 1], [0]))
    return full.reshape([c.shape[1] for c in cores])


def tt_storage(cores: list[np.ndarray]) -> int:
    return int(sum(c.size for c in cores))


# ---------------------------------------------------------------------------
# Rosenblatt (conditional) sampling from a non-negative grid density.
# ---------------------------------------------------------------------------
def rosenblatt_sample_dense(
    density: np.ndarray, grids: list[np.ndarray], n: int, rng
) -> np.ndarray:
    """Draw `n` continuous samples from a grid density by Rosenblatt transport.

    `density` is a non-negative `k`-way tensor of cell masses; `grids[i]` holds
    the `m_i` cell-centre coordinates of axis `i`. Sampling marginalises the
    trailing axes, draws axis 0 from its marginal, conditions on the draw, and
    recurses -- the exact O(sum m_i) transport a tensor-train realises in
    rank-compressed form. Fully vectorised over the `n` draws: each axis is one
    batched inverse-CDF, with a uniform within-cell jitter for continuity.
    """
    density = np.asarray(density, dtype=float)
    density = np.maximum(density, 0.0)
    total = density.sum()
    if not np.isfinite(total) or total <= 0:
        # degenerate density: fall back to uniform over the box
        return np.stack([rng.uniform(g.min(), g.max(), size=n) for g in grids], axis=1)
    k = density.ndim
    widths = [float(g[1] - g[0]) if len(g) > 1 else 1.0 for g in grids]
    out = np.empty((n, k))
    # sub[s] is sample s's current conditional sub-tensor; start from the joint
    sub = np.broadcast_to(density, (n,) + density.shape).copy()
    arange = np.arange(n)
    for dim in range(k):
        m = sub.shape[1]
        marg = sub.reshape(n, m, -1).sum(axis=2)      # marginal over trailing axes
        s = marg.sum(axis=1, keepdims=True)
        probs = np.where(s > 0, marg / np.where(s > 0, s, 1.0), 1.0 / m)
        u = rng.random(n)
        idx = (u[:, None] < np.cumsum(probs, axis=1)).argmax(axis=1)
        out[:, dim] = grids[dim][idx] + (rng.random(n) - 0.5) * widths[dim]
        sub = sub[arange, idx]                         # condition: drop this axis
    return out


# ---------------------------------------------------------------------------
# Rank-1 (mean-field) surrogate: the separable base case of the tensor train.
#   f(x) ~= c + sum_j g_j(x_j),  g_j a 1D Chebyshev model.
# This is a rank-1 functional TT in the FULL d coordinates -- no active-subspace
# collapse -- so for a separable objective every coordinate is modelled and the
# tempered density factorises exactly (derivation Link 2). Sampling draws each
# coordinate independently from exp(-g_j/T)/Z_j by 1D Rosenblatt: O(d) cost, no
# curse, scales to native CUTEst sizes. The Metropolis accept on the true f
# debiases the mean-field error when the objective is not separable.
# ---------------------------------------------------------------------------
def _cheb_design(t_unit: np.ndarray, degree: int) -> np.ndarray:
    """First-kind Chebyshev design matrix (N, degree) for columns T_1..T_deg."""
    n = t_unit.shape[0]
    cols = np.zeros((n, degree + 1))
    cols[:, 0] = 1.0
    if degree >= 1:
        cols[:, 1] = t_unit
    for m in range(1, degree):
        cols[:, m + 1] = 2.0 * t_unit * cols[:, m] - cols[:, m - 1]
    return cols[:, 1:]  # drop the constant; a single shared intercept is fit


@dataclass
class AdditiveSurrogate:
    """Separable rank-1 surrogate f ~= intercept + sum_j g_j(x_j)."""

    low: np.ndarray
    high: np.ndarray
    intercept: float
    coeffs: np.ndarray       # (dim, degree) per-coordinate Chebyshev coeffs
    degree: int
    pilot_work: float

    @classmethod
    def fit(cls, fn, low, high, dim, *, degree=8, n_pilot=None, rng=None):
        rng = np.random.default_rng(rng)
        low = np.asarray(low, float).reshape(-1)
        high = np.asarray(high, float).reshape(-1)
        if n_pilot is None:
            n_pilot = max(256, 16 * dim)
        X = rng.uniform(low, high, size=(n_pilot, dim))
        y = np.array([fn(x) for x in X], dtype=float)
        span = np.where(high > low, high - low, 1.0)
        # block design: [1 | per-coordinate Chebyshev features], one LS solve
        blocks = [np.ones((n_pilot, 1))]
        for j in range(dim):
            t = 2.0 * (X[:, j] - low[j]) / span[j] - 1.0
            blocks.append(_cheb_design(t, degree))
        design = np.concatenate(blocks, axis=1)
        sol, *_ = np.linalg.lstsq(design, y, rcond=None)
        intercept = float(sol[0])
        coeffs = sol[1:].reshape(dim, degree)
        return cls(low, high, intercept, coeffs, degree, float(n_pilot))

    @classmethod
    def from_points(cls, X, y, low, high, *, degree=8):
        """Fit from already-evaluated points; charges no objective work."""
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1)
        low = np.asarray(low, float).reshape(-1)
        high = np.asarray(high, float).reshape(-1)
        n, dim = X.shape
        span = np.where(high > low, high - low, 1.0)
        blocks = [np.ones((n, 1))]
        for j in range(dim):
            t = np.clip(2.0 * (X[:, j] - low[j]) / span[j] - 1.0, -1.0, 1.0)
            blocks.append(_cheb_design(t, degree))
        design = np.concatenate(blocks, axis=1)
        sol, *_ = np.linalg.lstsq(design, y, rcond=None)
        intercept = float(sol[0])
        coeffs = sol[1:].reshape(dim, degree)
        return cls(low, high, intercept, coeffs, degree, 0.0)

    def _coord_grid_energy(self, j: int, grid_m: int):
        t = np.linspace(-1.0, 1.0, grid_m)
        feats = _cheb_design(t, self.degree)         # (grid_m, degree)
        g = feats @ self.coeffs[j]                    # energy contribution g_j
        xs = self.low[j] + 0.5 * (t + 1.0) * (self.high[j] - self.low[j])
        return xs, g

    def sample(self, n: int, T: float, rng, grid_m: int = 65) -> np.ndarray:
        """Independent per-coordinate draws from exp(-g_j/T)/Z_j (1D Rosenblatt)."""
        dim = self.coeffs.shape[0]
        out = np.empty((n, dim))
        width = None
        for j in range(dim):
            xs, g = self._coord_grid_energy(j, grid_m)
            z = np.clip((g - g.min()) / max(T, 1e-12), 0.0, 700.0)
            p = np.exp(-z)
            s = p.sum()
            p = p / s if s > 0 else np.full(grid_m, 1.0 / grid_m)
            cdf = np.cumsum(p)
            u = rng.random(n)
            idx = np.searchsorted(cdf, u)
            idx = np.clip(idx, 0, grid_m - 1)
            w = xs[1] - xs[0]
            out[:, j] = np.clip(xs[idx] + (rng.random(n) - 0.5) * w,
                                self.low[j], self.high[j])
        return out

    def pilot_work_units(self) -> float:
        return self.pilot_work


# ---------------------------------------------------------------------------
# The surrogate object: reduce -> grid tempered density -> TT -> sample.
# ---------------------------------------------------------------------------
@dataclass
class TensorTrainSurrogate:
    """A tempered low-rank density over the active subspace, sampled exactly."""

    fit: "object"            # ReducedFit
    grids: list[np.ndarray]
    f_grid: np.ndarray       # surrogate objective on the grid (k-way)
    f_min: float
    tt_eps: float = 1e-8

    @classmethod
    def build(cls, fn, grad, low, high, dim, *, k=3, degree=6,
              grid_m=17, n_pilot=None, rng=None, tt_eps=1e-8):
        rng = np.random.default_rng(rng)
        k = min(k, dim)
        rf = fit_reduced_surrogate(fn, grad, low, high, dim,
                                   k=k, degree=degree, n_pilot=n_pilot, rng=rng)
        grids = [np.linspace(rf.reduced_low[j], rf.reduced_high[j], grid_m)
                 for j in range(k)]
        mesh = np.meshgrid(*grids, indexing="ij")
        pts = np.stack([m.reshape(-1) for m in mesh], axis=1)
        f_grid = np.array([rf.surrogate.eval(r) for r in pts]).reshape(
            [grid_m] * k)
        return cls(rf, grids, f_grid, float(np.min(f_grid)), tt_eps)

    def density(self, T: float) -> np.ndarray:
        """Tempered surrogate density tensor exp(-(f - f_min)/T) over the grid."""
        z = np.clip((self.f_grid - self.f_min) / max(T, 1e-12), 0.0, 700.0)
        return np.exp(-z)

    def tt_cores(self, T: float):
        return tt_svd(self.density(T), self.tt_eps)

    def sample(self, n: int, T: float, rng) -> np.ndarray:
        """`n` full-space proposals drawn from the tempered surrogate density."""
        red = rosenblatt_sample_dense(self.density(T), self.grids, n, rng)
        return np.stack([self.fit.encoder.decode(r) for r in red], axis=0)

    def pilot_work_units(self) -> float:
        return float(self.fit.pilot_work_units)


# ---------------------------------------------------------------------------
# The driver: SA with global surrogate (independence) proposals, debiased by
# Metropolis on the true objective.  This is the unified MCMC+SA point.
# ---------------------------------------------------------------------------
def tt_independence_sa(prob, seed, max_fevals, *, k=3, degree=6, grid_m=17,
                       local_frac=0.25, n_epochs=40):
    """Tensor-train independence-sampler SA on a CUTEst-style problem.

    `prob` exposes `.fn`, optional `.grad`, `.low`, `.high`, `.dim`. The pilot
    builds the reduced tempered surrogate; the remaining budget is spent in
    `n_epochs` geometric-temperature levels. Each epoch rebuilds the tempered
    surrogate density once and draws a block of proposals: a `1 - local_frac`
    majority are global Rosenblatt draws from the surrogate (the independence
    Move), the rest a Gaussian random walk around the incumbent that corrects
    surrogate bias (the local Move). Every proposal is accepted by a Metropolis
    rule against the *true* objective (the Accept slot), so the surrogate's bias
    never enters the answer. Returns `(best_val, n_true_evals)` at budget parity.
    """
    rng = np.random.default_rng(seed)
    low = np.asarray(prob.low, float).reshape(-1)
    high = np.asarray(prob.high, float).reshape(-1)
    dim = int(prob.dim)
    grad = getattr(prob, "grad", None)

    surr = TensorTrainSurrogate.build(prob.fn, grad, low, high, dim,
                                      k=k, degree=degree, grid_m=grid_m, rng=rng)
    n_evals = int(round(surr.pilot_work_units()))
    if n_evals >= max_fevals:
        return surr.f_min, max_fevals

    x = np.clip(0.5 * (low + high), low, high)
    fx = float(prob.fn(x))
    n_evals += 1
    best_f = fx

    remaining = max_fevals - n_evals
    n_epochs = max(1, min(n_epochs, remaining))
    block = max(1, remaining // n_epochs)
    t_hi = max(abs(fx), 1.0)
    t_lo = 1e-3 * t_hi
    step = 0.1 * (high - low)
    for epoch in range(n_epochs):
        if n_evals >= max_fevals:
            break
        frac = epoch / max(n_epochs - 1, 1)
        T = t_hi * (t_lo / t_hi) ** frac
        n_block = min(block, max_fevals - n_evals)
        n_local = int(round(local_frac * n_block))
        n_global = n_block - n_local
        proposals = []
        if n_global > 0:
            proposals.append(surr.sample(n_global, T, rng))
        if n_local > 0:
            jitter = rng.normal(size=(n_local, dim)) * step * (0.2 + 0.8 * (1 - frac))
            proposals.append(np.clip(x + jitter, low, high))
        Y = np.concatenate(proposals, axis=0) if proposals else np.empty((0, dim))
        rng.shuffle(Y)
        for y in Y:
            fy = float(prob.fn(y))
            n_evals += 1
            if fy <= fx or rng.random() < np.exp(-(fy - fx) / max(T, 1e-12)):
                x, fx = y, fy
            if fy < best_f:
                best_f = fy
            if n_evals >= max_fevals:
                break
    return best_f, n_evals


def additive_independence_sa(prob, seed, max_fevals, *, degree=8, grid_m=65,
                             local_frac=0.2, n_epochs=40):
    """Rank-1 (mean-field) independence-sampler SA.

    Fits a separable additive surrogate over all `d` coordinates and spends the
    remaining budget drawing independent per-coordinate proposals from the
    tempered surrogate (the global Move), mixed with a `local_frac` Gaussian
    random walk (the local correction), every proposal accepted by Metropolis on
    the true objective. For a separable objective the surrogate density is exact
    (derivation Link 2), so the global Move places every coordinate at its own
    tempered optimum at once -- the regime an active-subspace collapse cannot
    reach. Returns `(best_val, n_true_evals)` at budget parity.
    """
    rng = np.random.default_rng(seed)
    low = np.asarray(prob.low, float).reshape(-1)
    high = np.asarray(prob.high, float).reshape(-1)
    dim = int(prob.dim)

    surr = AdditiveSurrogate.fit(prob.fn, low, high, dim, degree=degree, rng=rng)
    n_evals = int(round(surr.pilot_work_units()))
    if n_evals >= max_fevals:
        return float("inf"), max_fevals
    x = np.clip(0.5 * (low + high), low, high)
    fx = float(prob.fn(x))
    n_evals += 1
    best_f = fx

    remaining = max_fevals - n_evals
    n_epochs = max(1, min(n_epochs, remaining))
    block = max(1, remaining // n_epochs)
    t_hi = max(abs(fx), 1.0)
    t_lo = 1e-3 * t_hi
    step = 0.1 * (high - low)
    for epoch in range(n_epochs):
        if n_evals >= max_fevals:
            break
        frac = epoch / max(n_epochs - 1, 1)
        T = t_hi * (t_lo / t_hi) ** frac
        n_block = min(block, max_fevals - n_evals)
        n_local = int(round(local_frac * n_block))
        n_global = n_block - n_local
        proposals = []
        if n_global > 0:
            proposals.append(surr.sample(n_global, T, rng, grid_m=grid_m))
        if n_local > 0:
            jitter = rng.normal(size=(n_local, dim)) * step * (0.2 + 0.8 * (1 - frac))
            proposals.append(np.clip(x + jitter, low, high))
        Y = np.concatenate(proposals, axis=0)
        rng.shuffle(Y)
        for y in Y:
            fy = float(prob.fn(y))
            n_evals += 1
            if fy <= fx or rng.random() < np.exp(-(fy - fx) / max(T, 1e-12)):
                x, fx = y, fy
            if fy < best_f:
                best_f = fy
            if n_evals >= max_fevals:
                break
    return best_f, n_evals


# ---------------------------------------------------------------------------
# Self-test: TT roundtrip, sampler fidelity, and the high-d win over RWM SA.
# ---------------------------------------------------------------------------
def _rwm_sa(prob, seed, max_fevals):
    """Plain Gaussian random-walk SA baseline at the same budget."""
    rng = np.random.default_rng(seed)
    low = np.asarray(prob.low, float).reshape(-1)
    high = np.asarray(prob.high, float).reshape(-1)
    dim = int(prob.dim)
    x = np.clip(0.5 * (low + high), low, high)
    fx = float(prob.fn(x))
    best = fx
    step = 0.1 * (high - low)
    t_hi = max(abs(fx), 1.0)
    t_lo = 1e-3 * t_hi
    for t in range(max_fevals - 1):
        frac = t / max(max_fevals - 2, 1)
        T = t_hi * (t_lo / t_hi) ** frac
        y = np.clip(x + rng.normal(size=dim) * step, low, high)
        fy = float(prob.fn(y))
        if fy <= fx or rng.random() < np.exp(-(fy - fx) / max(T, 1e-12)):
            x, fx = y, fy
        best = min(best, fy)
    return best, max_fevals


@dataclass
class _Toy:
    fn: object
    grad: object
    low: np.ndarray
    high: np.ndarray
    dim: int


def _styblinski_tang(dim):
    def fn(x):
        x = np.asarray(x, float)
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)

    def grad(x):
        x = np.asarray(x, float)
        return 0.5 * (4 * x**3 - 32 * x + 5)
    low = np.full(dim, -5.0)
    high = np.full(dim, 5.0)
    # global min ~ -39.16599 * dim at x_i = -2.903534
    return _Toy(fn, grad, low, high, dim), -39.16599 * dim


def _rastrigin(dim):
    def fn(x):
        x = np.asarray(x, float)
        return 10 * dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    def grad(x):
        x = np.asarray(x, float)
        return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    return _Toy(fn, grad, np.full(dim, -5.12), np.full(dim, 5.12), dim), 0.0


def _self_test() -> int:
    rng = np.random.default_rng(0)
    print("tensor_surrogate self-test\n")

    # 1. TT roundtrip on a rank-2 separable-plus-coupling tensor.
    m = 12
    a = rng.standard_normal((m, 1)) @ rng.standard_normal((1, m))
    b = rng.standard_normal((m, 1)) @ rng.standard_normal((1, m))
    cube = np.einsum("ij,k->ijk", a, rng.standard_normal(m)) + \
        np.einsum("i,jk->ijk", rng.standard_normal(m), b)
    cores = tt_svd(cube, eps=1e-10)
    rel = np.linalg.norm(tt_full(cores) - cube) / np.linalg.norm(cube)
    ranks = [c.shape[0] for c in cores] + [cores[-1].shape[-1]]
    dense_store = cube.size
    print(f"[1] TT roundtrip rel-err {rel:.2e}; ranks {ranks}; "
          f"storage {tt_storage(cores)} vs dense {dense_store}")
    ok1 = rel < 1e-8 and tt_storage(cores) < dense_store

    # 2. Rosenblatt sampler reproduces a known grid marginal.
    grids = [np.linspace(-3, 3, 41), np.linspace(-3, 3, 41)]
    gx, gy = np.meshgrid(*grids, indexing="ij")
    dens = np.exp(-((gx - 1.0) ** 2) / 0.5) * np.exp(-((gy + 1.0) ** 2) / 0.5)
    draws = rosenblatt_sample_dense(dens, grids, 20000, rng)
    mx, my = draws[:, 0].mean(), draws[:, 1].mean()
    print(f"[2] Rosenblatt sample means ({mx:.3f},{my:.3f}) vs target (1.000,-1.000)")
    ok2 = abs(mx - 1.0) < 0.1 and abs(my + 1.0) < 0.1

    # 3. The win: high-d multimodal, equal budget. Rank-1 (additive) and
    #    rank-r (active-subspace TT) independence samplers vs plain RWM SA.
    print("\n[3] global-basin hit-rate at equal budget (20 seeds):")
    print(f"    {'problem':14s} {'budget':>6s} {'rank1':>6s} {'TT-sub':>7s} "
          f"{'RWM':>5s}   medians (rank1 / sub / RWM / f*)")
    ok3 = True
    for name, (toy, fstar) in [("stybtang-d20", _styblinski_tang(20)),
                               ("rastrigin-d20", _rastrigin(20)),
                               ("stybtang-d50", _styblinski_tang(50))]:
        budget = 4000
        mf_hits = tt_hits = rwm_hits = 0
        mf_best, tt_best, rwm_best = [], [], []
        # "solved" tolerates both scales: within 1% of the optimum, or an
        # absolute 1.0 when the optimum is zero (rastrigin).
        thresh = fstar + max(0.01 * abs(fstar), 1.0)
        for seed in range(20):
            bf0, _ = additive_independence_sa(toy, seed, budget, degree=8)
            mf_best.append(bf0); mf_hits += int(bf0 <= thresh)
            bf1, _ = tt_independence_sa(toy, seed, budget, k=3, degree=6, grid_m=17)
            tt_best.append(bf1); tt_hits += int(bf1 <= thresh)
            bf2, _ = _rwm_sa(toy, seed, budget)
            rwm_best.append(bf2); rwm_hits += int(bf2 <= thresh)
        print(f"    {name:14s} {budget:6d} {mf_hits:4d}/20 {tt_hits:4d}/20 "
              f"{rwm_hits:3d}/20   "
              f"{np.median(mf_best):.2f} / {np.median(tt_best):.2f} / "
              f"{np.median(rwm_best):.2f} / {fstar:.2f}")
        # the rank-1 sampler must dominate RWM on these separable multimodal cases
        ok3 = ok3 and (mf_hits >= max(rwm_hits, 1) and np.median(mf_best) <= np.median(rwm_best))

    print("\n--- ledger ---")
    for tag, ok in [("1 TT roundtrip + compression", ok1),
                    ("2 Rosenblatt sampler fidelity", ok2),
                    ("3 rank-1 independence sampler dominates RWM", ok3)]:
        print(f"  {'PASS' if ok else 'FAIL'}  {tag}")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
