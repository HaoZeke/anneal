"""rgpot: potential energy surfaces for atomistic simulation.

Positions and cells are in Angstrom, energies in eV and forces in eV/Angstrom.
Names that depend on a build option (``D3Pot``, ``D4Pot``, ``ExprPot``,
``XcKernel``, ``PotentialCache``, the Fortran potentials and others) are
``None`` when the build left them out; the ``has_*`` flags say which parts
were compiled.

Metatomic models run through an engine library opened with ``dlopen``.
``default_metatomic_engine_path`` looks for a bundled engine under
``rgpot/lib/torch-X.Y/`` for the installed torch, then under any other
``torch-X.Y`` directory, then in legacy layouts, then for the file named by ``RGPOT_METATOMIC_ENGINE`` or ``METATOMIC_ENGINE``.
Wheels bundle engines for torch 2.7 and newer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import rgpot._core as _core

__version__: str = str(getattr(_core, "__version__", "0"))
LJPot = _core.LJPot
rgpot_lammps_potential = _core.rgpot_lammps_potential
rgpot_gromacs_potential = _core.rgpot_gromacs_potential
rgpot_openmm_potential = _core.rgpot_openmm_potential
LammpsPot = _core.LammpsPot
GromacsPot = _core.GromacsPot
OpenMMPot = _core.OpenMMPot
LJClusterPot = _core.LJClusterPot
MorsePot = _core.MorsePot
NumericalError = _core.NumericalError
evaluate_lj = _core.evaluate_lj
neighbor_list = _core.neighbor_list
has_cache = _core.has_cache
has_dftd3 = _core.has_dftd3
has_dftd4 = _core.has_dftd4
has_expr = _core.has_expr
has_fortran_pots = _core.has_fortran_pots
has_metatomic_dlopen = _core.has_metatomic_dlopen
has_xckernel = _core.has_xckernel
has_ipi = _core.has_ipi

# Absent when the matching meson flag was off. getattr keeps import rgpot
# working for a reduced build and for the full wheel.
evaluate_metatomic_dlopen: Any = getattr(_core, "evaluate_metatomic_dlopen", None)
PotentialCache: Any = getattr(_core, "PotentialCache", None)
D3Pot: Any = getattr(_core, "D3Pot", None)
D4Pot: Any = getattr(_core, "D4Pot", None)
ExprPot: Any = getattr(_core, "ExprPot", None)
XcKernel: Any = getattr(_core, "XcKernel", None)
StillingerWeberSi: Any = getattr(_core, "StillingerWeberSi", None)
TersoffSiC: Any = getattr(_core, "TersoffSiC", None)
SWPot: Any = getattr(_core, "SWPot", None)
EDIPPot: Any = getattr(_core, "EDIPPot", None)
LenoskyPot: Any = getattr(_core, "LenoskyPot", None)
TersoffPot: Any = getattr(_core, "TersoffPot", None)
EAMAlPot: Any = getattr(_core, "EAMAlPot", None)
FeHePot: Any = getattr(_core, "FeHePot", None)
CuH2Pot: Any = getattr(_core, "CuH2Pot", None)
WaterHPot: Any = getattr(_core, "WaterHPot", None)

try:
    from rgpot._core import IPIPot
except ImportError:  # compiled without -DRGPOT_HAS_IPI
    IPIPot = None  # type: ignore[misc, assignment]


def _torch_major() -> str | None:
    """Return installed torch X.Y, or None if torch is not importable."""
    try:
        import torch  # pyright: ignore[reportMissingImports]

        v = torch.__version__.split("+", 1)[0]
        parts = v.split(".")
        return f"{parts[0]}.{parts[1]}"
    except Exception:
        return None


def default_metatomic_engine_path() -> str | None:
    """Return the path of a Metatomic engine library, or None.

    Bundled engines for the installed torch come first, then any bundled
    engine, then legacy layouts, then the file named by
    ``RGPOT_METATOMIC_ENGINE`` or ``METATOMIC_ENGINE`` if it exists.
    """
    here = Path(__file__).resolve().parent
    maj = _torch_major()
    candidates: list[Path] = []
    # Multi-ABI layout first (rgpot/lib/torch-X.Y/) — preferred product path
    if maj:
        candidates.append(here / "lib" / f"torch-{maj}" / "librgpot_metatomic_engine.so")
        candidates.append(here / "lib" / f"torch-{maj}" / "libmetatomic_engine.so")
    lib_root = here / "lib"
    if lib_root.is_dir():
        for d in sorted(lib_root.glob("torch-*")):
            candidates.append(d / "librgpot_metatomic_engine.so")
            candidates.append(d / "libmetatomic_engine.so")
    # Legacy single-engine layouts (last resort). The librgpot_ name is the
    # engine. libmetatomic_engine.so remains for one release.
    candidates.extend(
        [
            here / "lib" / "librgpot_metatomic_engine.so",
            here / "lib" / "libmetatomic_engine.so",
            here.parent / ".rgpot.mesonpy.libs" / "librgpot_metatomic_engine.so",
            here.parent / ".rgpot.mesonpy.libs" / "libmetatomic_engine.so",
            here / "librgpot_metatomic_engine.so",
            here / "libmetatomic_engine.so",
        ]
    )

    for c in candidates:
        if c.is_file():
            return str(c)

    env = os.environ.get("RGPOT_METATOMIC_ENGINE") or os.environ.get(
        "METATOMIC_ENGINE"
    )
    if env and Path(env).is_file():
        return env
    return None


def available_metatomic_engine_abis() -> list[str]:
    """List torch-X.Y majors for which a bundled engine is present.

    PyPI wheels ship engines for torch 2.7+ (see package README).
    """
    here = Path(__file__).resolve().parent / "lib"
    out: list[str] = []
    if not here.is_dir():
        return out
    for d in sorted(here.glob("torch-*")):
        if (d / "librgpot_metatomic_engine.so").is_file() or (
            d / "libmetatomic_engine.so"
        ).is_file():
            out.append(d.name.removeprefix("torch-"))
    return out


def __getattr__(name: str) -> Any:
    """Lazy exports. ``ase`` is not imported by ``import rgpot``."""
    if name == "ase":
        import importlib

        module = importlib.import_module("rgpot.ase")
        globals()["ase"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def evaluate_metatomic(
    positions: Any,
    atom_types: Any,
    box: Any,
    *,
    model_path: str,
    engine_path: str | None = None,
    device: str = "cpu",
) -> Any:
    """Evaluate a Metatomic model through the dlopen engine.

    Raises RuntimeError when the build has no Metatomic dlopen frontend.
    """
    # Load torch via the Python extension first so the engine's DT_NEEDED
    # libtorch resolves to the same process image (avoids dual-load races).
    try:
        import torch  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        pass
    if evaluate_metatomic_dlopen is None:
        raise RuntimeError("this build has no MetatomicDlopen frontend")
    eng = engine_path or default_metatomic_engine_path() or ""
    return evaluate_metatomic_dlopen(
        positions, atom_types, box, model_path, eng, device
    )


__all__ = [
    "LJPot",
    "LJClusterPot",
    "MorsePot",
    "LammpsPot",
    "GromacsPot",
    "OpenMMPot",
    "rgpot_lammps_potential",
    "rgpot_gromacs_potential",
    "rgpot_openmm_potential",
    "NumericalError",
    "StillingerWeberSi",
    "TersoffSiC",
    "D3Pot",
    "D4Pot",
    "ExprPot",
    "IPIPot",
    "XcKernel",
    "PotentialCache",
    "SWPot",
    "EDIPPot",
    "LenoskyPot",
    "TersoffPot",
    "EAMAlPot",
    "FeHePot",
    "CuH2Pot",
    "WaterHPot",
    "evaluate_lj",
    "neighbor_list",
    "evaluate_metatomic",
    "evaluate_metatomic_dlopen",
    "default_metatomic_engine_path",
    "available_metatomic_engine_abis",
    "has_metatomic_dlopen",
    "has_cache",
    "has_expr",
    "has_ipi",
    "has_dftd3",
    "has_dftd4",
    "has_xckernel",
    "has_fortran_pots",
    "ase",
    "__version__",
]
