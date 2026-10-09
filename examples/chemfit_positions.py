"""ChemFit review-response pattern: fit per-atom positions with anneal.

This is the corrected form of the ``run_anneal(benchmark_context)`` snippet
from the ChemFit review thread. Three things were wrong there:

1. ``fitter.evaluate`` / ``fitter.step`` are not the ChemFit protocol; the
   user-driven loop is ``init`` -> ``ask`` -> ``tell`` -> ``finish``.
2. ``anneal.run`` used to drive an unconstrained variant, so proposals left
   ``[low, high]``. It now mirror-reflects every proposal into the box, and
   the portfolio reflects before evaluation.
3. There was no way to pass the benchmark's initial parameters; ``x0`` (the
   fitter's ``initial_params`` by default) now seeds the chain / incumbent.

Run with a stub fitter (no ChemFit install needed)::

    python examples/chemfit_positions.py --driver portfolio --budget 2000

Against a real ChemFit checkout, replace ``StubFitter`` with the review
``Fitter`` and keep the ``fit_anneal`` call unchanged.
"""

import argparse
from typing import Any

import numpy as np

from anneal.chemfit import fit_anneal


class StubFitter:
    """Minimal duck-typed stand-in for ``chemfit.Fitter``.

    Harmonic wells pull each atom toward a hidden target; the loss is the
    squared deviation. Implements the user-driven protocol
    (``init``/``ask``/``tell``/``finish``) with ChemFit's best-seen
    tracking semantics.
    """

    def __init__(self, initial_params: dict[str, Any], target: np.ndarray):
        self.initial_parameters = initial_params
        self.bounds: dict[str, Any] = {}
        self.target = np.asarray(target, dtype=np.float64)
        self.contexts: list[dict[str, Any]] = []

    def init(self) -> None:
        self._evals = 0
        self._best_loss: float | None = None
        self._best_params: dict[str, Any] | None = None

    def ask(self, params: dict[str, Any]):
        positions = np.asarray(params["positions"], dtype=np.float64)
        loss = float(np.sum((positions - self.target) ** 2))
        self._evals += 1
        if self._best_loss is None or loss < self._best_loss:
            self._best_loss = loss
            self._best_params = {"positions": positions.copy()}
        return loss

    def tell(self, step: int | None = None) -> None:
        return None

    def finish(self, opt_params: dict[str, Any] | None = None) -> dict[str, Any]:
        if opt_params is None:
            assert self._best_params is not None
            return self._best_params
        return opt_params


def run_anneal(benchmark_context: dict[str, Any], driver: str = "portfolio"):
    """Drop-in corrected review-response driver.

    ``benchmark_context`` provides ``fitter``, ``budget`` and
    ``initial_params`` with ``positions`` of shape ``(n_atoms, 3)``.
    """
    fitter = benchmark_context["fitter"]
    budget = int(benchmark_context["budget"])
    n_atoms = len(benchmark_context["initial_params"]["positions"])
    low = -3.0 * np.ones(3 * n_atoms)
    high = 3.0 * np.ones(3 * n_atoms)
    return fit_anneal(fitter, budget, driver=driver, low=low, high=high, seed=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", default="portfolio")
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--n-atoms", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    target = rng.uniform(-2.0, 2.0, size=(args.n_atoms, 3))
    start = rng.uniform(-3.0, 3.0, size=(args.n_atoms, 3))
    fitter = StubFitter({"positions": start}, target)
    context = {
        "fitter": fitter,
        "budget": args.budget,
        "initial_params": {"positions": start},
    }
    result = run_anneal(context, driver=args.driver)
    got = np.asarray(result["positions"])
    print(f"driver={args.driver} start_loss={np.sum((start - target) ** 2):.6f}")
    print(f"driver={args.driver} final_loss={np.sum((got - target) ** 2):.6f}")
    print(f"driver={args.driver} in_bounds={bool(np.all(got >= -3.0) and np.all(got <= 3.0))}")


if __name__ == "__main__":
    main()
