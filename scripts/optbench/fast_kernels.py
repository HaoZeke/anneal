"""ctypes front for ``kernels.c``, checked against optbench's Python kernels.

The library is compiled on first use with optbench's own native flags
(``optbench.fastpot``): no fused multiply-add, no fast math. ``check``
compares a native kernel with its Python twin at a point; the runner calls
it before a row starts and again at every hit.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_SOURCE = Path(__file__).resolve().parent / "kernels.c"
_FLAGS = ["-std=c11", "-O3", "-ffp-contract=off", "-fno-fast-math", "-fno-builtin", "-fPIC", "-shared"]
_LIB: ctypes.CDLL | None = None
_D = ctypes.POINTER(ctypes.c_double)
_L = ctypes.POINTER(ctypes.c_long)

LJ_CUTOFF = 40.0
BINARY_CUTOFF = 20.0


def library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    stamp = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
    root = Path(os.environ.get("ANNEAL_KERNEL_DIR", Path(tempfile.gettempdir()) / "anneal-kernels"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"kernels-{stamp}.so"
    if not path.is_file():
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run(["gcc", *_FLAGS, "-o", str(temporary), str(_SOURCE), "-lm"], check=True)
        temporary.replace(path)
    lib = ctypes.CDLL(str(path))
    lib.anneal_lj.argtypes = [_D, ctypes.c_int, ctypes.c_double, _D, _D]
    lib.anneal_binary_lj.argtypes = [_D, _L, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, _D, _D]
    lib.anneal_tip4p.argtypes = [_D, ctypes.c_int, _D, _D, _D, _D]
    _LIB = lib
    return lib


def _ptr(array: np.ndarray):
    return array.ctypes.data_as(_D)


def lennard_jones():
    lib = library()

    def evaluate(x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        grad = np.empty_like(x)
        energy = ctypes.c_double()
        lib.anneal_lj(_ptr(x), x.size // 3, LJ_CUTOFF, ctypes.byref(energy), _ptr(grad))
        return energy.value, grad

    return evaluate


def binary_lennard_jones(species):
    lib = library()
    labels = np.ascontiguousarray(species, dtype=np.int64)

    def evaluate(x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        grad = np.empty_like(x)
        energy = ctypes.c_double()
        lib.anneal_binary_lj(
            _ptr(x), labels.ctypes.data_as(_L), x.size // 3, BINARY_CUTOFF, 1.0, 1.3, ctypes.byref(energy), _ptr(grad)
        )
        return energy.value, grad

    return evaluate


def chart_to_blocks(x: np.ndarray) -> np.ndarray:
    """GMIN's 2N chart (centres, then angle-axis vectors) to (N, 6) blocks."""
    x = np.asarray(x, dtype=np.float64).reshape(2, -1, 3)
    return np.concatenate([x[0], x[1]], axis=1)


def blocks_to_chart(blocks: np.ndarray) -> np.ndarray:
    blocks = np.asarray(blocks, dtype=np.float64).reshape(-1, 6)
    return np.concatenate([blocks[:, :3].reshape(-1), blocks[:, 3:].reshape(-1)])


def tip4p():
    from optbench.tip4p import reference_sites

    lib = library()
    ref = np.ascontiguousarray(reference_sites(), dtype=np.float64).reshape(-1)

    def evaluate(x):
        blocks = np.ascontiguousarray(chart_to_blocks(x)).reshape(-1)
        molecules = blocks.size // 6
        work = np.empty(60 * molecules, dtype=np.float64)
        grad = np.empty_like(blocks)
        energy = ctypes.c_double()
        lib.anneal_tip4p(_ptr(blocks), molecules, _ptr(ref), _ptr(work), ctypes.byref(energy), _ptr(grad))
        return energy.value, blocks_to_chart(grad)

    return evaluate


def reference(kind: str, species=None):
    """optbench's own Python kernel, on the point set anneal sees."""
    if kind == "lj":
        from optbench.potentials import lennard_jones as python_lj

        return python_lj
    if kind == "binary":
        from optbench.potentials import binary_lennard_jones as python_binary

        labels = np.asarray(species, dtype=np.int64)
        return lambda x: python_binary(x, labels)
    if kind == "tip4p":
        from optbench.tip4p import tip4p as python_tip4p

        def evaluate(x):
            energy, grad = python_tip4p(chart_to_blocks(x))
            return energy, blocks_to_chart(grad)

        return evaluate
    raise ValueError(f"no potential {kind}")


def native(kind: str, species=None):
    if kind == "lj":
        return lennard_jones()
    if kind == "binary":
        return binary_lennard_jones(species)
    if kind == "tip4p":
        return tip4p()
    raise ValueError(f"no potential {kind}")


def check(kind: str, x, species=None, rtol: float = 1e-10) -> float:
    """Largest disagreement of the native kernel with optbench's, relative."""
    e_fast, g_fast = native(kind, species)(x)
    e_ref, g_ref = reference(kind, species)(x)
    scale = max(1.0, abs(e_ref), float(np.max(np.abs(g_ref))))
    worst = max(abs(e_fast - e_ref), float(np.max(np.abs(g_fast - g_ref)))) / scale
    if not worst <= rtol:
        raise AssertionError(f"native {kind} kernel disagrees with optbench's by {worst:.3e}")
    return worst
