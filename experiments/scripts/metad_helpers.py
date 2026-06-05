"""Well-tempered metadynamics primitives (Barducci/Bussi/Parrinello 2008).

A bias potential V(s) = sum_k w_k * exp(-|s - s_k|^2 / (2 sigma^2))
is grown by depositing Gaussians at the current CV value `s = s(x)`
every `tau` MCMC sweeps. In well-tempered metaD the height
self-attenuates: w_k = w_0 * exp(-V_{k-1}(s_k) / ((gamma - 1) * T)),
so V converges to -((gamma - 1)/gamma) * F(s) and the bias does
not grow without bound.

Used by demo_bgsa.py's bgsa_metad driver to flatten the cost
landscape on multimodal benchmarks. Closes the SA-enhanced-sampling
gap from the design notes Item 2.
"""

from __future__ import annotations

import numpy as np


class WellTemperedBiasOnFeaturizer:
    """Well-tempered metadynamics on a 2D CV produced by an arbitrary
    Featurizer (TICA / PCA / SGOOP from auto_cv.py).

    The black-box sampler does not need user-supplied CVs: it runs a
    short pilot trajectory, fits a Featurizer to discover the slow
    modes, then biases those modes via metadynamics. This is the
    closure of the design notes metadynamics + the user's "no
    user-facing CV shit" constraint.
    """

    def __init__(self, featurizer, cv_low, cv_high, sigma=0.3, w0=0.05,
                 gamma=8.0, grid_n=64):
        self.featurizer = featurizer
        self.low_cv = np.asarray(cv_low, dtype=np.float64)
        self.high_cv = np.asarray(cv_high, dtype=np.float64)
        self.sigma = float(sigma)
        self.w0 = float(w0)
        self.gamma = float(gamma)
        self.grid_n = int(grid_n)
        self.V = np.zeros((grid_n, grid_n), dtype=np.float64)
        gx = np.linspace(self.low_cv[0], self.high_cv[0], grid_n)
        gy = np.linspace(self.low_cv[1], self.high_cv[1], grid_n)
        self._gx, self._gy = np.meshgrid(gx, gy, indexing="ij")

    def cv(self, x):
        return self.featurizer.cv(x)

    def potential(self, s):
        sx = np.clip(s[0], self.low_cv[0], self.high_cv[0])
        sy = np.clip(s[1], self.low_cv[1], self.high_cv[1])
        i_f = (sx - self.low_cv[0]) / (self.high_cv[0] - self.low_cv[0]) * (self.grid_n - 1)
        j_f = (sy - self.low_cv[1]) / (self.high_cv[1] - self.low_cv[1]) * (self.grid_n - 1)
        i0 = int(np.clip(np.floor(i_f), 0, self.grid_n - 2))
        j0 = int(np.clip(np.floor(j_f), 0, self.grid_n - 2))
        di = i_f - i0
        dj = j_f - j0
        v00 = self.V[i0, j0]
        v10 = self.V[i0 + 1, j0]
        v01 = self.V[i0, j0 + 1]
        v11 = self.V[i0 + 1, j0 + 1]
        return float(
            v00 * (1 - di) * (1 - dj)
            + v10 * di * (1 - dj)
            + v01 * (1 - di) * dj
            + v11 * di * dj
        )

    def deposit(self, s, T):
        v_at_s = self.potential(s)
        w = self.w0 * np.exp(-v_at_s / ((self.gamma - 1.0) * T))
        dx2 = (self._gx - s[0]) ** 2 + (self._gy - s[1]) ** 2
        self.V += w * np.exp(-dx2 / (2.0 * self.sigma ** 2))


class WellTemperedBias:
    """Grid-binned well-tempered metadynamics bias on a 2D CV.

    The CV is the first two coordinates of x by default (override via
    `cv_indices`). Bias is stored on a `(grid_n, grid_n)` array; each
    deposit adds a Gaussian centred at the bin nearest to the current
    CV value. Reweighting via `reweight(s)` returns
    `exp(+V(s) / T)` for post-hoc unbiasing.
    """

    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        sigma: float = 0.3,
        w0: float = 0.05,
        gamma: float = 8.0,
        grid_n: int = 64,
        cv_indices: tuple[int, int] = (0, 1),
    ):
        self.low_cv = np.array([low[cv_indices[0]], low[cv_indices[1]]])
        self.high_cv = np.array([high[cv_indices[0]], high[cv_indices[1]]])
        self.sigma = float(sigma)
        self.w0 = float(w0)
        self.gamma = float(gamma)
        self.grid_n = int(grid_n)
        self.cv_indices = cv_indices
        self.V = np.zeros((grid_n, grid_n), dtype=np.float64)
        self._grid_x = np.linspace(self.low_cv[0], self.high_cv[0], grid_n)
        self._grid_y = np.linspace(self.low_cv[1], self.high_cv[1], grid_n)
        # Pre-compute (gx, gy) mesh for efficient Gaussian deposit.
        self._gx, self._gy = np.meshgrid(self._grid_x, self._grid_y, indexing="ij")

    def cv(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[self.cv_indices[0]], x[self.cv_indices[1]]])

    def potential(self, s: np.ndarray) -> float:
        """Bilinear-interpolate V(s) from the grid."""
        # Clip s to grid bounds.
        sx = np.clip(s[0], self.low_cv[0], self.high_cv[0])
        sy = np.clip(s[1], self.low_cv[1], self.high_cv[1])
        # Linear index in the grid.
        i_f = (sx - self.low_cv[0]) / (self.high_cv[0] - self.low_cv[0]) * (self.grid_n - 1)
        j_f = (sy - self.low_cv[1]) / (self.high_cv[1] - self.low_cv[1]) * (self.grid_n - 1)
        i0 = int(np.clip(np.floor(i_f), 0, self.grid_n - 2))
        j0 = int(np.clip(np.floor(j_f), 0, self.grid_n - 2))
        di = i_f - i0
        dj = j_f - j0
        v00 = self.V[i0, j0]
        v10 = self.V[i0 + 1, j0]
        v01 = self.V[i0, j0 + 1]
        v11 = self.V[i0 + 1, j0 + 1]
        return float(
            v00 * (1 - di) * (1 - dj)
            + v10 * di * (1 - dj)
            + v01 * (1 - di) * dj
            + v11 * di * dj
        )

    def deposit(self, s: np.ndarray, T: float) -> None:
        """Deposit a well-tempered Gaussian at CV value s."""
        # Well-tempered height: w = w0 * exp(-V(s) / ((gamma - 1) * T))
        v_at_s = self.potential(s)
        w = self.w0 * np.exp(-v_at_s / ((self.gamma - 1.0) * T))
        # Add Gaussian to the grid.
        dx2 = (self._gx - s[0]) ** 2 + (self._gy - s[1]) ** 2
        self.V += w * np.exp(-dx2 / (2.0 * self.sigma ** 2))

    def reweight(self, s: np.ndarray, T: float) -> float:
        """Returns exp(+V(s) / T) for post-hoc unbiasing of an observable."""
        return float(np.exp(self.potential(s) / T))

    def free_energy_estimate(self) -> np.ndarray:
        """Returns the metadynamics estimate of F(s) on the grid:
        F(s) = -((gamma - 1)/gamma) * V(s) (well-tempered limit)."""
        return -((self.gamma - 1.0) / self.gamma) * self.V
