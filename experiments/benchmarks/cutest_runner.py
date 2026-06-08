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
    # (name, sif_params dict for problem-size capping)
    ("ROSENBR",  None),                # n=2  Rosenbrock ravine
    ("BEALE",    None),                # n=2  Beale: f* at (3, 0.5)
    ("BRKMCC",   None),                # n=2  Brkmcc
    ("CUBE",     None),                # n=2  Cubic ravine
    ("POWELLSG", {"N": 4}),            # n=4  Powell singular (cap from default 5000)
    ("BROWNDEN", None),                # n=4  Brown-Dennis nonlinear LSQ
    ("BOX3",     None),                # n=3  Box 3D
    ("BIGGS6",   None),                # n=6  Biggs EXP6
    ("WATSON",   {"N": 12}),           # n=12 Watson nonlinear LSQ
    ("EXTROSNB", {"N": 10}),           # n=10 Extended Rosenbrock (cap from default 1000)
    ("GULF",     None),                # n=3  Gulf research and development
    ("OSBORNEA", None),                # n=5  Osborne A nonlinear LSQ
]


# IEEE-754 sentinel CUTEst uses for "no bound": 1e+20.
_BOUND_INF = 1e19


def _is_finite_bound(b: np.ndarray) -> np.ndarray:
    """Return mask of bounds that are NOT the CUTEst -1e20 / +1e20 sentinel."""
    return np.isfinite(b) & (np.abs(b) < _BOUND_INF)


def cutest_env() -> dict:
    """Returns the env-var dict pycutest expects.

    Reads from the .bench/ directory in the repo root; raises
    RuntimeError with a clear message if the bootstrap hasn't run."""
    root = os.environ.get("PIXI_PROJECT_ROOT", os.getcwd())
    bench = os.path.join(root, ".bench")
    env = {
        "ARCHDEFS": os.path.join(bench, "ARCHDefs"),
        "SIFDECODE": os.path.join(bench, "SIFDecode", "install"),
        "CUTEST": os.path.join(bench, "CUTEst", "install"),
        "MASTSIF": os.path.join(bench, "sif"),
        "MYARCH": "pc64.lnx.gfo",
        "PYCUTEST_CACHE": os.environ.get(
            "PYCUTEST_CACHE",
            os.path.join(bench, "cache"),
        ),
    }
    required_paths = {
        "SIFDecode decoder": os.path.join(env["SIFDECODE"], "bin", "sifdecoder"),
        "CUTEst single library": os.path.join(env["CUTEST"], "lib", "libcutest_single.a"),
        "CUTEst double library": os.path.join(env["CUTEST"], "lib", "libcutest_double.a"),
        "MASTSIF catalogue": env["MASTSIF"],
    }
    missing = [f"{label} ({path})" for label, path in required_paths.items() if not os.path.exists(path)]
    if missing:
        joined = "; ".join(missing)
        raise RuntimeError(
            "CUTEst bootstrap incomplete. Run "
            "'bash experiments/benchmarks/bootstrap_cutest.sh' first "
            f"(looked in {bench}; missing {joined})"
        )
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
    grad: callable | None
    low: np.ndarray
    high: np.ndarray
    x0: np.ndarray
    design_low: np.ndarray
    design_high: np.ndarray
    has_cutest_bounds: bool
    f_star: float | None  # may be None for problems without a stored optimum
    objective_degree: int | None = None


def effective_design_bounds(
    low: np.ndarray,
    high: np.ndarray,
    anchor: np.ndarray,
    x_box: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a compact QMC design box inside the true feasibility box."""

    low = np.asarray(low, dtype=np.float64).reshape(-1)
    high = np.asarray(high, dtype=np.float64).reshape(-1)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)
    if low.shape != high.shape or low.shape != anchor.shape:
        raise ValueError("low, high, and anchor must have the same shape")
    x_box = float(x_box)
    if not np.isfinite(x_box) or x_box <= 0.0:
        raise ValueError("x_box must be positive and finite")

    x0 = np.clip(anchor, low, high)
    design_low = low.copy()
    design_high = high.copy()
    span = high - low
    shrink = np.isfinite(span) & (span > 2.0 * x_box)
    design_low[shrink] = np.maximum(low[shrink], x0[shrink] - x_box)
    design_high[shrink] = np.minimum(high[shrink], x0[shrink] + x_box)
    same = design_low >= design_high
    design_low[same] = low[same]
    design_high[same] = high[same]
    return design_low, design_high, x0


def load(
    name: str,
    sif_params: dict | None = None,
    x_box: float = 5.0,
    f_star: float | None = None,
) -> CutestProblem:
    """Loads a CUTEst problem and wraps it as a CutestProblem.

    Args:
        name: SIF problem id.
        sif_params: pycutest sifParams dict to fix problem size (e.g.,
            {"N": 10} caps EXTROSNB at n=10 instead of its default 1000).
        x_box: half-width of the synthetic box around `x0` for problems
            whose CUTEst-side bounds are the +-1e20 "no bound" sentinel.
        f_star: known global optimum if available (used by the bench
            "solved" predicate). None means "use a tolerance from x0".
    """
    setup_cutest_env()
    import pycutest
    p = pycutest.import_problem(name, sifParams=sif_params)
    try:
        properties = pycutest.problem_properties(name)
    except Exception:
        properties = {}
    try:
        objective_degree = int(properties["degree"])
    except (KeyError, TypeError, ValueError):
        objective_degree = None

    bl_finite = _is_finite_bound(p.bl)
    bu_finite = _is_finite_bound(p.bu)
    bl = np.where(bl_finite, p.bl, p.x0 - x_box).astype(np.float64)
    bu = np.where(bu_finite, p.bu, p.x0 + x_box).astype(np.float64)
    # Guard pathological cases where bl == bu after substitution.
    same = bl >= bu
    bl[same] -= x_box
    bu[same] += x_box
    design_low, design_high, x0 = effective_design_bounds(
        bl, bu, np.asarray(p.x0, dtype=np.float64).reshape(-1), x_box=x_box
    )

    def fn(x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        return float(p.obj(x_arr))

    grad = None
    if hasattr(p, "grad"):

        def grad(x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
            return np.asarray(p.grad(x_arr), dtype=np.float64).reshape(-1)

    return CutestProblem(
        name=name,
        dim=p.n,
        fn=fn,
        grad=grad,
        low=bl,
        high=bu,
        x0=x0,
        design_low=design_low,
        design_high=design_high,
        has_cutest_bounds=bool(np.any(bl_finite | bu_finite)),
        f_star=f_star,
        objective_degree=objective_degree,
    )


def list_default_manifest() -> list[tuple[str, dict | None]]:
    return list(DEFAULT_MANIFEST)


def load_default_manifest() -> list[CutestProblem]:
    """Load every (name, sif_params) pair from the default manifest."""
    return [load(name, sif_params=params) for name, params in DEFAULT_MANIFEST]


def main():
    """Smoke test: load every problem in the default manifest, evaluate
    f(x0), and print n / f(x0). Use as `pixi run -e verify cutest-smoke`."""
    setup_cutest_env()
    print(f"Loading {len(DEFAULT_MANIFEST)} CUTEst problems...\n")
    print(f"{'name':<10} {'n':>4} {'f(x0)':>16} {'box low':>10} {'box high':>10}")
    print("-" * 56)
    bad = []
    for name, params in DEFAULT_MANIFEST:
        try:
            prob = load(name, sif_params=params)
            mid = (prob.low + prob.high) / 2
            fval = prob.fn(mid)
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
