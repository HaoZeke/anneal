"""Hessian-vector products at one centre, as a minimum-mode search makes them."""

from __future__ import annotations

import numpy as np

import rgpot

PRODUCTS = 16


def _lattice(cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = 3.92
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    grid = np.stack(np.meshgrid(*[np.arange(cells)] * 3, indexing="ij"), -1).reshape(-1, 3)
    sites = (grid[:, None, :] + basis[None, :, :]).reshape(-1, 3) * a
    rng = np.random.default_rng(3)
    positions = np.ascontiguousarray(sites + rng.normal(0.0, 0.05, sites.shape))
    types = np.full(len(sites), 78, dtype=np.int32)
    box = np.ascontiguousarray(np.eye(3) * a * cells)
    return positions, types, box


class HessianVector:
    """Wall time and force calls of 16 products at one Pt geometry.

    ``forward`` reuses the centre forces, so the loop makes 17 force
    calls; ``forward_naive`` drops the centre before every product, the
    loop a force-only consumer writes, and makes 32; ``central`` makes 32
    at second order; ``analytic`` is the Morse kernel's own product.
    """

    params = [3, 5]
    param_names = ["cells"]
    rounds = 5
    repeat = 1
    sample_time = 0.05
    warmup_time = 0.2
    min_run_count = 1
    timeout = 180

    def setup(self, cells: int) -> None:
        self.positions, self.types, self.box = _lattice(cells)
        rng = np.random.default_rng(4)
        self.directions = [
            np.ascontiguousarray(rng.normal(size=self.positions.shape)) for _ in range(PRODUCTS)
        ]
        self.pot = rgpot.MorsePot()
        self.pot(self.positions, self.types, self.box)

    def _loop(self, scheme: str, fresh: bool = False) -> dict:
        self.pot.invalidate_hvp()
        self.pot.reset_hvp_counts()
        for v in self.directions:
            if fresh:
                self.pot.invalidate_hvp()
            self.pot.hvp(self.positions, self.types, self.box, v, scheme=scheme)
        return self.pot.hvp_counts()

    def time_forward(self, cells: int) -> None:
        self._loop("forward")

    def time_forward_naive(self, cells: int) -> None:
        self._loop("forward", fresh=True)

    def time_central(self, cells: int) -> None:
        self._loop("central")

    def time_analytic(self, cells: int) -> None:
        self._loop("analytic")

    def track_forward_force_calls(self, cells: int) -> int:
        return int(self._loop("forward")["force_calls"])

    def track_forward_naive_force_calls(self, cells: int) -> int:
        return int(self._loop("forward", fresh=True)["force_calls"])

    track_forward_force_calls.unit = "calls"  # type: ignore[attr-defined]
    track_forward_naive_force_calls.unit = "calls"  # type: ignore[attr-defined]
