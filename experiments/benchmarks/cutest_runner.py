"""CUTEst-driven benchmark runner. Wraps pycutest problems as the
shared-runner Problem dataclass so the existing experiments infra
(MCMC-SA, classical SA, sparse skip) drives them without changes.

Bootstrap: run `bash experiments/benchmarks/bootstrap_cutest.sh` once
to clone+build CUTEst into .bench/. Pass the resulting paths through
`CutestConfig` when using a non-default location.

The paper protocol supplies a frozen problem manifest to the campaign driver.
This module also retains a 12-problem diagnostic fallback for direct smoke
runs that do not provide a manifest.
"""

from __future__ import annotations

import ctypes.util
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np


# Diagnostic fallback for direct smoke runs. All are unconstrained
# (constraints='U') with n in [2, 30]. Chosen for diversity -- ravines
# (ROSENBR), separable polynomials (CUBE), trigonometric multimodality
# (BOX3, MEYER3), singular Hessians (POWELLSG, BROWNDEN), etc.
DEFAULT_MANIFEST = [
    # (name, sif_params dict for problem-size capping)
    ("ROSENBR", None),  # n=2  Rosenbrock ravine
    ("BEALE", None),  # n=2  Beale: f* at (3, 0.5)
    ("BRKMCC", None),  # n=2  Brkmcc
    ("CUBE", None),  # n=2  Cubic ravine
    ("POWELLSG", {"N": 4}),  # n=4  Powell singular (cap from default 5000)
    ("BROWNDEN", None),  # n=4  Brown-Dennis nonlinear LSQ
    ("BOX3", None),  # n=3  Box 3D
    ("BIGGS6", None),  # n=6  Biggs EXP6
    ("WATSON", {"N": 12}),  # n=12 Watson nonlinear LSQ
    ("EXTROSNB", {"N": 10}),  # n=10 Extended Rosenbrock (cap from default 1000)
    ("GULF", None),  # n=3  Gulf research and development
    ("OSBORNEA", None),  # n=5  Osborne A nonlinear LSQ
]


# IEEE-754 sentinel CUTEst uses for "no bound": 1e+20.
_BOUND_INF = 1e19


def _is_finite_bound(b: np.ndarray) -> np.ndarray:
    """Return mask of bounds that are NOT the CUTEst -1e20 / +1e20 sentinel."""
    return np.isfinite(b) & (np.abs(b) < _BOUND_INF)


@dataclass(frozen=True)
class CutestConfig:
    """Filesystem configuration for a bootstrapped CUTEst/PyCUTEst stack."""

    bench_dir: Path
    cache_dir: Path
    myarch: str = "pc64.lnx.gfo"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "bench_dir", Path(self.bench_dir).expanduser().resolve()
        )
        object.__setattr__(
            self, "cache_dir", Path(self.cache_dir).expanduser().resolve()
        )

    @classmethod
    def from_root(
        cls,
        root: str | Path | None = None,
        *,
        cache_dir: str | Path | None = None,
        myarch: str = "pc64.lnx.gfo",
    ) -> "CutestConfig":
        project_root = Path.cwd() if root is None else Path(root)
        bench_dir = project_root / ".bench"
        cache = bench_dir / "cache" if cache_dir is None else Path(cache_dir)
        return cls(bench_dir=bench_dir, cache_dir=cache, myarch=myarch)

    @property
    def archdefs_dir(self) -> Path:
        return self.bench_dir / "ARCHDefs"

    @property
    def sifdecode_install(self) -> Path:
        return self.bench_dir / "SIFDecode" / "install"

    @property
    def sifdecoder(self) -> Path:
        return self.sifdecode_install / "bin" / "sifdecoder"

    @property
    def cutest_install(self) -> Path:
        return self.bench_dir / "CUTEst" / "install"

    @property
    def cutest_include_dir(self) -> Path:
        return self.cutest_install / "include"

    @property
    def cutest_header(self) -> Path:
        return self.cutest_include_dir / "cutest.h"

    @property
    def cutest_single_library(self) -> Path:
        return self.cutest_install / "lib" / "libcutest_single.a"

    @property
    def cutest_double_library(self) -> Path:
        return self.cutest_install / "lib" / "libcutest_double.a"

    @property
    def mastsif_dir(self) -> Path:
        return self.bench_dir / "sif"

    @property
    def pycutest_cache_holder(self) -> Path:
        return self.cache_dir / "pycutest_cache_holder"

    def validate(self) -> "CutestConfig":
        """Validate required files and ensure the explicit PyCUTEst cache exists."""

        required_paths = {
            "SIFDecode decoder": self.sifdecoder,
            "CUTEst include header": self.cutest_header,
            "CUTEst single library": self.cutest_single_library,
            "CUTEst double library": self.cutest_double_library,
            "MASTSIF catalogue": self.mastsif_dir,
        }
        missing = [
            f"{label} ({path})"
            for label, path in required_paths.items()
            if not path.exists()
        ]
        if missing:
            joined = "; ".join(missing)
            raise RuntimeError(
                "CUTEst bootstrap incomplete. Run "
                "'bash experiments/benchmarks/bootstrap_cutest.sh' first "
                f"(looked in {self.bench_dir}; missing {joined})"
            )
        self.pycutest_cache_holder.mkdir(parents=True, exist_ok=True)
        return self


def default_cutest_config(
    root: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> CutestConfig:
    return CutestConfig.from_root(root, cache_dir=cache_dir).validate()


def _pycutest_package_dir() -> Path:
    spec = importlib.util.find_spec("pycutest")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("PyCUTEst is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def _pycutest_version(init_path: Path) -> str:
    text = init_path.read_text(encoding="utf-8")
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else "0"


def _load_pycutest_submodule(package_dir: Path, name: str):
    fullname = f"pycutest.{name}"
    if fullname in sys.modules:
        return sys.modules[fullname]
    spec = importlib.util.spec_from_file_location(fullname, package_dir / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load PyCUTEst submodule {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    return module


def _install_pycutest_path_providers(system_paths, config: CutestConfig) -> None:
    system_paths.get_cutest_path = lambda: str(config.cutest_double_library)
    system_paths.get_cutest_include_path = lambda: str(config.cutest_include_dir)
    system_paths.get_sifdecoder_path = lambda: str(config.sifdecoder)
    system_paths.get_mastsif_path = lambda: str(config.mastsif_dir)
    system_paths.get_cache_path = lambda: str(config.cache_dir)


def _configure_pycutest_linux_link_libraries(
    install_scripts,
    *,
    platform: str = sys.platform,
    find_library=ctypes.util.find_library,
) -> None:
    """Link glibc vector math when gfortran emits libmvec symbols."""

    if platform != "linux" or find_library("mvec") is None:
        return
    needle = "libraries=['gfortran']"
    replacement = "libraries=['gfortran','mvec']"
    if replacement not in install_scripts.setupScriptLinux:
        install_scripts.setupScriptLinux = install_scripts.setupScriptLinux.replace(
            needle, replacement, 1
        )


def configured_pycutest(config: CutestConfig | None = None):
    """Return PyCUTEst configured from explicit paths instead of process state."""

    existing = sys.modules.get("pycutest")
    if existing is not None and all(
        hasattr(existing, name)
        for name in (
            "import_problem",
            "clear_cache",
            "problem_properties",
            "find_problems",
        )
    ):
        return existing

    config = default_cutest_config() if config is None else config.validate()
    package_dir = _pycutest_package_dir()
    package = ModuleType("pycutest")
    package.__file__ = str(package_dir / "__init__.py")
    package.__path__ = [str(package_dir)]
    package.__package__ = "pycutest"
    package.__version__ = _pycutest_version(package_dir / "__init__.py")
    package.__all__ = []
    sys.modules["pycutest"] = package

    system_paths = _load_pycutest_submodule(package_dir, "system_paths")
    _install_pycutest_path_providers(system_paths, config)
    for stale in (
        "pycutest.install_scripts",
        "pycutest.build_interface",
        "pycutest.sifdecode_extras",
    ):
        sys.modules.pop(stale, None)

    install_scripts = _load_pycutest_submodule(package_dir, "install_scripts")
    _configure_pycutest_linux_link_libraries(install_scripts)
    build_interface = _load_pycutest_submodule(package_dir, "build_interface")
    sifdecode_extras = _load_pycutest_submodule(package_dir, "sifdecode_extras")
    problem_class = _load_pycutest_submodule(package_dir, "problem_class")

    package.import_problem = build_interface.import_problem
    package.clear_cache = build_interface.clear_cache
    package.all_cached_problems = build_interface.all_cached_problems
    package.print_available_sif_params = sifdecode_extras.print_available_sif_params
    package.problem_properties = sifdecode_extras.problem_properties
    package.find_problems = sifdecode_extras.find_problems
    package.CUTEstProblem = problem_class.CUTEstProblem
    package.__all__ = [
        "import_problem",
        "clear_cache",
        "all_cached_problems",
        "print_available_sif_params",
        "problem_properties",
        "find_problems",
        "CUTEstProblem",
    ]
    cache_path = str(config.cache_dir)
    if cache_path not in sys.path:
        sys.path.append(cache_path)
    return package


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
    config: CutestConfig | None = None,
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
    pycutest = configured_pycutest(config)
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


def load_default_manifest(config: CutestConfig | None = None) -> list[CutestProblem]:
    """Load every (name, sif_params) pair from the default manifest."""
    return [
        load(name, sif_params=params, config=config)
        for name, params in DEFAULT_MANIFEST
    ]


# Scalable unconstrained problems at higher dimension, to test the
# dimension-robustness thesis: population-based and simplex external solvers
# degrade as n grows, while the 1/sqrt(D) proposal scale, the dimension-collapse
# surrogate, and the separable independence sampler are built to hold. Spans
# separable (SROSENBR, QUARTC, ARWHEAD), ill-conditioned (VARDIM, PENALTY1,
# NONDIA), and general coupled (EXTROSNB, WOODS, BROYDN7D) landscapes.
HIGHDIM_MANIFEST = [
    ("SROSENBR", {"N": 100}),  # separable Rosenbrock
    ("EXTROSNB", {"N": 100}),  # extended Rosenbrock (coupled, ill-cond)
    ("ARWHEAD", {"N": 100}),  # almost-separable
    ("NONDIA", {"N": 100}),  # Shanno, ill-conditioned
    ("WOODS", {"N": 100}),  # coupled quartic
    ("POWELLSG", {"N": 100}),  # group-separable singular
    ("ENGVAL1", {"N": 100}),  # sum of squares
    ("VARDIM", {"N": 100}),  # ill-conditioned
    ("PENALTY1", {"N": 100}),  # ill-conditioned penalty
    ("QUARTC", {"N": 100}),  # separable quartic
    ("TOINTGSS", {"N": 100}),  # Toint Gaussian
    ("BROYDN7D", {"N": 100}),  # Broyden tridiagonal
    ("FREUROTH", {"N": 100}),  # Freudenstein-Roth
    ("COSINE", {"N": 100}),  # trigonometric
    ("DIXMAANA", {"M": 30}),  # n = 3M = 90
]


def load_highdim_manifest(config: CutestConfig | None = None) -> list[CutestProblem]:
    """Load the high-dimensional scalable manifest, skipping any problem that
    fails to decode (SIF parameter names vary across problems)."""
    out = []
    for name, params in HIGHDIM_MANIFEST:
        try:
            out.append(load(name, sif_params=params, config=config))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return out


def main():
    """Smoke test: load every problem in the default manifest, evaluate
    f(x0), and print n / f(x0). Use as `pixi run -e verify cutest-smoke`."""
    config = default_cutest_config()
    print(f"Loading {len(DEFAULT_MANIFEST)} CUTEst problems...\n")
    print(f"{'name':<10} {'n':>4} {'f(x0)':>16} {'box low':>10} {'box high':>10}")
    print("-" * 56)
    bad = []
    for name, params in DEFAULT_MANIFEST:
        try:
            prob = load(name, sif_params=params, config=config)
            mid = (prob.low + prob.high) / 2
            fval = prob.fn(mid)
            print(
                f"{name:<10} {prob.dim:>4} {fval:>16.4g} {prob.low.min():>10.2g} {prob.high.max():>10.2g}"
            )
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
