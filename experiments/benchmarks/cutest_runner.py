"""CUTEst-driven benchmark runner. Wraps pycutest problems as the
shared-runner Problem dataclass so the existing experiments infra
(MCMC-SA, classical SA, sparse skip) drives them without changes.

Bootstrap: run `bash experiments/benchmarks/bootstrap_cutest.sh` once
to clone+build CUTEst into .bench/. Then export the env vars printed
by the bootstrap (or use the cutest_env() helper here).

Selected manifest: 12 unconstrained or bound-constrained CUTEst
problems with `n in [2, 30]`, hand-picked to span the standard
test-set difficulty spectrum (Rosenbrock-like ravines, Powell
singularity, trigonometric multi-modality, etc.). The hand-pick is
the v0.3.x manifest; v0.4 will switch to pycutest.find_problems()
once the SIF classification index is sorted out.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np


# Hand-picked manifest. All are unconstrained (constraints='U') with
# n in [2, 30]. Chosen for diversity of landscape shapes -- ravines
# (ROSENBR), separable polynomials (CUBE), trigonometric multimodality
# (BOX3, MEYER3), singular Hessians (POWELLSG, BROWNDEN), etc.
DEFAULT_MANIFEST = [
    "ROSENBR",   # n=2  Rosenbrock 2D ravine; classical SA hard
    "BEALE",     # n=2  Beale: f* at (3, 0.5)
    "BRKMCC",    # n=2  Brkmcc
    "CUBE",      # n=2  Cubic ravine
    "POWELLSG",  # n=4  Powell singular function
    "BROWNDEN",  # n=4  Brown-Dennis nonlinear LSQ
    "BOX3",      # n=3  Box 3D
    "BIGGS6",   # n=6  Biggs EXP6
    "WATSON",   # n=12 Watson nonlinear LSQ; harder
    "EXTROSNB", # n=10 Extended Rosenbrock 10D
    "GULF",     # n=3  Gulf research and development
    "OSBORNEA", # n=5  Osborne A nonlinear LSQ
]


def cutest_env() -> dict:
    """Returns the env-var dict pycutest expects.

    Reads from the .bench/ directory in the repo root; raises
    RuntimeError with a clear message if the bootstrap hasn't run."""
    root = os.environ.get("PIXI_PROJECT_ROOT", os.getcwd())
    bench = os.path.join(root, ".bench")
    if not os.path.isdir(os.path.join(bench, "SIFDecode", "install", "bin")):
        raise RuntimeError(
            f"CUTEst not bootstrapped. Run 'bash experiments/benchmarks/bootstrap_cutest.sh' "
            f"first (looked in {bench})"
        )
    env = {
        "ARCHDEFS": os.path.join(bench, "ARCHDefs"),
        "SIFDECODE": os.path.join(bench, "SIFDecode", "install"),
        "CUTEST": os.path.join(bench, "CUTEst", "install"),
        "MASTSIF": os.path.join(bench, "sif"),
        "MYARCH": "pc64.lnx.gfo",
        "PYCUTEST_CACHE": os.path.join(bench, "cache"),
    }
    os.makedirs(os.path.join(env["PYCUTEST_CACHE"], "pycutest_cache_holder"), exist_ok=True)
    return env


def setup_cutest_env() -> None:
    """Install the CUTEst env vars into the running process and prepend
    the SIFDecode bin dir to PATH. Idempotent."""
    env = cutest_env()
    for k, v in env.items():
        os.environ[k] = v
    sif_bin = os.path.join(env["SIFDECODE"], "bin")
    path = os.environ.get("PATH", "")
    if sif_bin not in path:
        os.environ["PATH"] = f"{sif_bin}:{path}"


@dataclass(frozen=True)
class CutestProblem:
    """CUTEst problem wrapped to match the shared-runner Problem shape."""

    name: str
    dim: int
    fn: callable
    low: np.ndarray
    high: np.ndarray
    f_star: float | None  # may be None for problems without a stored optimum


def load(name: str, x_box: float = 5.0, f_star: float | None = None) -> CutestProblem:
    """Loads a CUTEst problem and wraps it as a CutestProblem. The SA
    runner needs box bounds; if the underlying problem is unconstrained
    we synthesise a `[-x_box, x_box]^n` box around `x0`. The Problem's
    own bl/bu are used when finite, falling back to the synth box."""
    setup_cutest_env()
    import pycutest
    p = pycutest.import_problem(name)

    bl = np.where(np.isfinite(p.bl), p.bl, p.x0 - x_box).astype(np.float64)
    bu = np.where(np.isfinite(p.bu), p.bu, p.x0 + x_box).astype(np.float64)

    def fn(x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        return float(p.obj(x_arr))

    return CutestProblem(name=name, dim=p.n, fn=fn, low=bl, high=bu, f_star=f_star)


def list_default_manifest() -> list[str]:
    return list(DEFAULT_MANIFEST)


def main():
    """Smoke test: load every problem in the default manifest, evaluate
    f(x0), and print n / f(x0). Use as `pixi run -e verify cutest-smoke`."""
    setup_cutest_env()
    import pycutest
    print(f"Loading {len(DEFAULT_MANIFEST)} CUTEst problems...\n")
    print(f"{'name':<10} {'n':>4} {'f(x0)':>16} {'box low':>10} {'box high':>10}")
    print("-" * 56)
    bad = []
    for name in DEFAULT_MANIFEST:
        try:
            prob = load(name)
            fval = prob.fn(np.where(np.isfinite(prob.low + prob.high) & (prob.low < prob.high),
                                    (prob.low + prob.high) / 2, np.zeros(prob.dim)))
            print(f"{name:<10} {prob.dim:>4} {fval:>16.4g} {prob.low.min():>10.2g} {prob.high.max():>10.2g}")
        except Exception as e:
            print(f"{name:<10} FAIL: {type(e).__name__}: {e}")
            bad.append(name)
    print()
    if bad:
        print(f"FAILED: {len(bad)} problem(s): {bad}", file=sys.stderr)
        return 1
    print(f"OK: {len(DEFAULT_MANIFEST)} CUTEst problems loaded successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
