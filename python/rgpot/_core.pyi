"""
rgpot Python bindings (nanobind). Classes that need a build option, such as ExprPot, D3Pot, D4Pot, XcKernel and the Fortran kernels, are absent when the build left them out. Evaluations release the interpreter while the kernel runs.
"""

from typing import Annotated

import numpy
from numpy.typing import NDArray


class NumericalError(ArithmeticError):
    pass

has_metatomic_dlopen: bool = True

has_cache: bool = False

has_expr: bool = True

has_dftd3: bool = True

has_dftd4: bool = True

has_xckernel: bool = True

has_fortran_pots: bool = True

has_eam_table: bool = True

has_ipi: bool = True

def echo_f64(values: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> NDArray[numpy.float64]: ...

def evaluate_lj(positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

def evaluate_metatomic_dlopen(positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], model_path: str, engine_path: str = '', device: str = 'cpu') -> tuple:
    """
    Evaluate forces through a Metatomic engine opened with dlopen. With an empty engine_path the search is RGPOT_METATOMIC_ENGINE, METATOMIC_ENGINE, libmetatomic_engine.so on the loader path, then the directories in EON_POTENTIALS_PATH and RGPOT_ENGINE_PATH. rgpot.evaluate_metatomic also tries the engines bundled with the package. Each call builds its own frontend.
    """

class LJPot:
    def __init__(self, u0: float = 1.0, cutoff: float = 15.0, psi: float = 1.0, switch_width: float = 0.0) -> None:
        """
        Shifted 12-6 Lennard-Jones. u0 is the well depth (eV), cutoff and psi are Angstrom. A positive switch_width (Angstrom) replaces the shift with the C^2 quintic switch over [cutoff - switch_width, cutoff]; it must not exceed cutoff.
        """

    def config(self) -> dict: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_per_atom(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        evaluate() plus energies, shape (N,), eV, summing to energy, and virials, shape (N, 6), eV, Voigt xx yy zz yz xz xy, summing to -volume * stress. A pair term goes half to each atom. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class LJClusterPot:
    def __init__(self, u0: float = 1.0, cutoff: float = 15.0, psi: float = 1.0, switch_width: float = 0.0) -> None:
        """
        Free-boundary shifted 12-6 Lennard-Jones. u0 is the well depth (eV), cutoff and psi are Angstrom. An input cell does not fold pair vectors. A positive switch_width (Angstrom) replaces the shift with the C^2 quintic switch over [cutoff - switch_width, cutoff]; it must not exceed cutoff.
        """

    def config(self) -> dict: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_per_atom(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        evaluate() plus energies, shape (N,), eV, summing to energy, and virials, shape (N, 6), eV, Voigt xx yy zz yz xz xy, summing to -volume * stress. A pair term goes half to each atom. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class MorsePot:
    def __init__(self, De: float = 0.7102, a: float = 1.6047, re: float = 2.897, cutoff: float = 9.5, switch_width: float = 0.0) -> None:
        """
        Shifted Morse pair potential. De is the well depth (eV), a is 1/Angstrom, re and cutoff are Angstrom. Defaults are the platinum parameters. A positive switch_width (Angstrom) replaces the shift with the C^2 quintic switch over [cutoff - switch_width, cutoff]; it must not exceed cutoff.
        """

    def config(self) -> dict: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_per_atom(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        evaluate() plus energies, shape (N,), eV, summing to energy, and virials, shape (N, 6), eV, Voigt xx yy zz yz xz xy, summing to -volume * stress. A pair term goes half to each atom. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class EAMTablePot:
    def __init__(self, file: str, format: str, periodic: bool = True, stress: bool = False, skin: float = 1.0) -> None:
        """
        Tabulated EAM or Finnis-Sinclair potential. format is funcfl, setfl, or fs. Energies and lengths use the table's units.
        """

    @property
    def cutoff(self) -> float:
        """Radial cutoff read from the table."""

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_per_atom(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        evaluate() plus energies, shape (N,), eV, summing to energy, and virials, shape (N, 6), eV, Voigt xx yy zz yz xz xy, summing to -volume * stress. A pair term goes half to each atom. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class SWPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class EDIPPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class LenoskyPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class TersoffPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class EAMAlPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class FeHePot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class CuH2Pot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class WaterHPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class StillingerWeberSi:
    def __init__(self) -> None:
        """
        Stillinger-Weber silicon, Stillinger and Weber, Phys. Rev. B 31, 5262 (1985). Epsilon is the published 2.16723920 eV, not eOn's 2.3149945 eV (ratio 1.068176741337). Energy is eV, lengths are Angstrom. The gradient is dE/dx.
        """

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        Return (energy, forces, variance) for one configuration. Forces are -dE/dx.
        """

    def caps(self) -> dict: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and the Cauchy stress, shape (3, 3), in eV/Angstrom^3. The kernel runs detached.
        """

    def energy_gradient(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        Energy and dE/dx. positions is (n_atoms, 3) or (n_config, n_atoms, 3). A batch shares atom_types and box, and crosses into the library once.
        """

class TersoffSiC:
    def __init__(self) -> None:
        """
        Tersoff silicon-carbon kernel, Tersoff, Phys. Rev. B 39, 5566 (1989). The 1990 erratum sets carbon B to 346.74 eV and d to 4.3484. Carbon beta is dimensionless. Atomic numbers 14 and 6. This class always evaluates the 1989 kernel. The engine name tersoff uses the Fortran silicon gather when every atom is silicon. Energy is eV, lengths are Angstrom. The gradient is dE/dx.
        """

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        Return (energy, forces, variance) for one configuration. Forces are -dE/dx.
        """

    def energy_gradient(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        Energy and dE/dx. positions is (n_atoms, 3) or (n_config, n_atoms, 3). A batch shares atom_types and box, and crosses into the library once.
        """

class D3Pot:
    def __init__(self, damping: str = 'bj', functional: str = 'pbe', atm: bool = True) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class D4Pot:
    def __init__(self, functional: str = 'pbe', charge: float = 0.0, atm: bool = True) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class SumPot:
    def __init__(self, terms: list) -> None:
        """
        Weighted sum of child potentials. terms is a list of kind strings or dicts with kind, name (term<i>), weight (1.0, sign allowed), enabled (True) and the kind's fields as for ExprPot. A disabled term is never built or evaluated.
        """

    def components(self) -> list:
        """
        Per-term record of this thread's last call: name, weight, enabled, evaluated, energy (eV, before the weight), forces (N, 3), variance and stress (3, 3, or None).
        """

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class ExprPot:
    def __init__(self, expression: str, terms: dict) -> None:
        """
        expression is a Lepton string; terms maps names to 'lj'/'morse'/'d3'/'d4' or a dict with kind plus that kind's fields: u0, cutoff, psi, switch_width (lj); De, a, re, cutoff, switch_width (morse); the D3/D4 fields.
        """

    @property
    def expression(self) -> str: ...

    def d_energy_d_term(self, name: str) -> float: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. The kernel runs with the interpreter released.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

    def hvp(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], direction: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], step: float = 0.0, scheme: str = 'auto') -> Annotated[NDArray[numpy.float64], dict(order='C')]:
        """
        Hessian-vector product H @ direction, shape (N, 3). 'forward' reuses the forces of the last geometry this object evaluated, so k products at one geometry cost k + 1 force calls; 'central' costs 2 a product and is second order in step; 'analytic' uses the kernel's own product; 'auto' is analytic where the kernel has one, otherwise forward. step 0 is 1e-3 Angstrom. The kernel runs with the interpreter released.
        """

    def hvp_centre(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        (energy, forces, variance) at the geometry, from the centre that hvp() reuses. A search that needs the centre forces asks here so the forward products that follow make one force call each.
        """

    def invalidate_hvp(self) -> None:
        """Drop the centre hvp() reuses."""

    def hvp_counts(self) -> dict:
        """
        Products, force calls, centre reuses and analytic products of hvp() and hvp_centre() so far.
        """

    def reset_hvp_counts(self) -> None:
        """Zero hvp_counts(). The centre stays."""

    def has_analytic_hvp(self) -> bool:
        """Whether hvp(scheme='auto') uses the kernel's own product."""

class IPIPot:
    def __init__(self, address: str, timeout_s: float = 30.0) -> None:
        """
        i-PI socket server. address is unix:/absolute/path, unix:<name> (binds /tmp/ipi_<name>), tcp:<host>:<port>, or tcp:<port>. The driver connects and returns forces. Wire units are bohr and hartree. timeout_s covers connecting and one force exchange.
        """

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def stress(self) -> object:
        """
        Cauchy stress from the last call, shape (3, 3), in eV/Angstrom^3. None before a successful call.
        """

    @property
    def endpoint(self) -> str: ...

class XcKernel:
    def __init__(self, name: str) -> None: ...

    @staticmethod
    def catalog() -> list[str]: ...

    @property
    def name(self) -> str: ...

    def scal_names(self) -> list[str]: ...

    def contract(self, chi: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], dchi: object | None = None, scal: dict | None = None, lapl_chi: object | None = None, hess_chi: object | None = None) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @staticmethod
    def fields_from_density(chi: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], dchi: object, density: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], lapl_chi: object | None = None, hess_chi: object | None = None) -> dict: ...

def neighbor_list(positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], cell: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], cutoff: float, quantities: str = 'ijd', full: bool = False, offsets: bool = False) -> object:
    """
    Neighbour arrays selected by quantities. i and j are indices, d is the distance, D is r_j - r_i with shape (npairs, 3), S is the integer shift with that shape. offsets=True appends a CSR row pointer of length natoms + 1 after sorting pairs by i, then j. `i, j, d = rgpot.neighbor_list(positions, cell, cutoff)`.
    """

class LammpsPot:
    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """Energy, forces and variance. Positions are Angstrom, energy is eV."""

class GromacsPot:
    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """Energy, forces and variance. Positions are Angstrom, energy is eV."""

    def native_u0(self) -> float:
        """Well depth stored for GROMACS, kJ/mol."""

    def native_cutoff(self) -> float:
        """Cutoff stored for GROMACS, nanometres."""

    def native_psi(self) -> float:
        """Zero crossing stored for GROMACS, nanometres."""

class OpenMMPot:
    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """Energy, forces and variance. Positions are Angstrom, energy is eV."""

    def native_u0(self) -> float:
        """Well depth stored for OpenMM, kJ/mol."""

    def native_cutoff(self) -> float:
        """Cutoff stored for OpenMM, nanometres."""

    def native_psi(self) -> float:
        """Zero crossing stored for OpenMM, nanometres."""

def rgpot_lammps_potential(u0: float = 1.0, cutoff: float = 8.0, psi: float = 1.0, engine_path: str = '') -> LammpsPot:
    """
    Shifted 12-6 Lennard-Jones. The constructor dlopens an engine plugin, which stays the backend. engine_path empty searches RGPOT_LAMMPS_ENGINE, LAMMPS_ENGINE, liblammps_engine.so, then liblj_engine.so. u0 is eV; cutoff and psi are Angstrom.
    """

def rgpot_gromacs_potential(u0: float = 1.0, cutoff: float = 8.0, psi: float = 1.0, engine_path: str = '') -> GromacsPot:
    """
    Shifted 12-6 Lennard-Jones. Nanometres and kJ/mol are stored once. An empty engine_path evaluates in process. A path is a library whose rgpot_md_native_force uses those units.
    """

def rgpot_openmm_potential(u0: float = 1.0, cutoff: float = 8.0, psi: float = 1.0, engine_path: str = '') -> OpenMMPot:
    """
    Shifted 12-6 Lennard-Jones. Nanometres and kJ/mol are stored once. An empty engine_path evaluates in process. A path is a library whose rgpot_md_native_force uses those units, such as libopenmm_engine.so.
    """
