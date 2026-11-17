"""Run a Python rgpot potential through LAMMPS ``fix external``.

The module imports neither LAMMPS nor mpi4py.  :class:`FixExternal` accepts an
already-created ``lammps.lammps`` object and imports mpi4py only when that
object reports more than one rank.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from rgpot import atomic_number, unit_conversion_factor

__all__ = ["UNIT_STYLES", "FixExternal", "box_rows", "type_numbers", "unit_factors"]

# LAMMPS unit style -> (LAMMPS length units per Angstrom,
# LAMMPS energy units per eV).  Conversion comes from rgpot's unit parser.
UNIT_STYLES: dict[str, tuple[float, float]] = {
    "metal": (
        unit_conversion_factor("angstrom", "angstrom"),
        unit_conversion_factor("eV", "eV"),
    ),
    "real": (
        unit_conversion_factor("angstrom", "angstrom"),
        unit_conversion_factor("eV", "kcal/mol"),
    ),
    "electron": (
        unit_conversion_factor("angstrom", "bohr"),
        unit_conversion_factor("eV", "hartree"),
    ),
}


def unit_factors(style: str) -> tuple[float, float]:
    """Return LAMMPS length units per Angstrom and energy units per eV."""
    try:
        return UNIT_STYLES[style]
    except KeyError:
        raise ValueError(
            "rgpot supports LAMMPS units metal, real, and electron; "
            f"this input uses units {style!r}"
        ) from None


def box_rows(
    boxlo: Sequence[float],
    boxhi: Sequence[float],
    xy: float,
    yz: float,
    xz: float,
) -> np.ndarray:
    """LAMMPS' restricted triclinic box, one lattice vector per row."""
    lx, ly, lz = (float(boxhi[k]) - float(boxlo[k]) for k in range(3))
    return np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]], dtype=float)


def type_numbers(elements: str | Sequence[str | int], n_types: int) -> np.ndarray:
    """Map one element symbol or atomic number per LAMMPS atom type."""
    items = elements.split() if isinstance(elements, str) else list(elements)
    numbers: list[int] = []
    for item in items:
        if isinstance(item, (int, np.integer)) or (
            isinstance(item, str) and item.isdigit()
        ):
            number = int(item)
            if not 1 <= number <= 118:
                raise ValueError(f"atomic number {number} is outside 1..118")
        else:
            number = atomic_number(str(item))
        numbers.append(number)
    if len(numbers) != n_types:
        raise ValueError(
            f"elements names {len(numbers)} atom types, "
            f"but the LAMMPS box has {n_types}"
        )
    return np.asarray(numbers, dtype=np.int32)


def _mpi_world(comm: Any, size: int) -> Any:
    if comm is None:
        try:
            from mpi4py import MPI
        except ImportError as exc:
            raise RuntimeError(
                f"LAMMPS runs on {size} MPI ranks; rgpot.lammps needs mpi4py "
                "to gather the atoms on rank 0"
            ) from exc
        comm = MPI.COMM_WORLD
    if comm.Get_size() != size:
        raise RuntimeError(
            f"comm has {comm.Get_size()} ranks, but LAMMPS runs on {size}; "
            "pass the communicator the lammps object was created with"
        )
    return comm


class FixExternal:
    """Evaluate any Python rgpot potential from ``fix external pf/callback``.

    ``potential`` follows rgpot's Python potential protocol: either
    ``evaluate(positions, atomic_numbers, cell)`` returns a result dictionary,
    or the object is callable and returns ``(energy, forces, variance)``.
    Native rgpot units are Angstrom, eV, and eV/Angstrom.

    ``virial="auto"`` uses an analytic stress when the potential reports one
    and otherwise takes twelve finite-difference energy evaluations.  Use
    ``"finite-difference"`` to force that route or ``"none"`` for zero virial.
    """

    _STRAIN = 1.0e-5

    def __init__(
        self,
        lmp: Any,
        potential: Any,
        elements: str | Sequence[str | int],
        *,
        fix_id: str = "rgpot",
        create: bool = True,
        comm: Any = None,
        virial: str = "auto",
    ) -> None:
        if not callable(potential) and not callable(getattr(potential, "evaluate", None)):
            raise TypeError("potential must be a Python rgpot potential object")
        if virial not in {"auto", "finite-difference", "none"}:
            raise ValueError("virial must be 'auto', 'finite-difference', or 'none'")

        self._lmp = lmp
        self.potential = potential
        self.fix_id = fix_id
        self.virial = virial
        self.units = str(lmp.extract_global("units"))
        self.length_per_angstrom, self.energy_per_ev = unit_factors(self.units)
        self.type_numbers = type_numbers(elements, int(lmp.extract_global("ntypes")))

        periodicity = list(lmp.extract_box()[5])
        if any(periodicity) and not all(periodicity):
            names = [axis for axis, flag in zip("xyz", periodicity) if not flag]
            raise ValueError(
                "rgpot takes a fully periodic LAMMPS box or none; "
                "non-periodic axes: " + ", ".join(names)
            )
        self.periodic = all(periodicity)

        size = int(lmp.extract_setting("world_size"))
        self._comm = _mpi_world(comm, size) if size > 1 else None
        self._root = self._comm is None or self._comm.Get_rank() == 0
        self._imgmask = int(lmp.extract_setting("IMGMASK"))
        self._imgmax = int(lmp.extract_setting("IMGMAX"))
        self._imgbits = int(lmp.extract_setting("IMGBITS"))
        self._img2bits = int(lmp.extract_setting("IMG2BITS"))
        self.error: BaseException | None = None
        self.energy = 0.0
        self.evaluations = 0
        self._flag = f"{fix_id}_failed"
        self._created = create

        if create:
            lmp.command(f"fix {fix_id} all external pf/callback 1 1")
            lmp.command(f"fix_modify {fix_id} energy yes virial yes")
        lmp.command(f"variable {self._flag} internal 0")
        lmp.command(
            f"fix {fix_id}_halt all halt 1 v_{self._flag} "
            "> 0.5 error hard message yes"
        )
        lmp.set_fix_external_callback(fix_id, self._callback, None)

    def _callback(
        self, _caller: Any, _step: int, nlocal: int, tag: Any, x: Any, f: Any
    ) -> None:
        try:
            self._step(int(nlocal), tag, x, f)
        except Exception as exc:
            if self.error is None:
                self.error = exc
            f[:nlocal] = 0.0
            self._lmp.set_internal_variable(self._flag, 1.0)

    def _unwrapped(self, nlocal: int, x: Any, rows: np.ndarray) -> np.ndarray:
        xu = np.array(x[:nlocal], dtype=float).reshape(nlocal, 3)
        if self.periodic and nlocal:
            image = np.asarray(
                self._lmp.numpy.extract_atom("image")[:nlocal], dtype=np.int64
            )
            shifts = np.empty((nlocal, 3), dtype=float)
            shifts[:, 0] = (image & self._imgmask) - self._imgmax
            shifts[:, 1] = (
                (image >> self._imgbits) & self._imgmask
            ) - self._imgmax
            shifts[:, 2] = (image >> self._img2bits) - self._imgmax
            xu += shifts @ rows
        return xu

    def _raw(
        self, positions: np.ndarray, numbers: np.ndarray, rows: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        evaluate = getattr(self.potential, "evaluate", None)
        if callable(evaluate):
            result = evaluate(positions, numbers, rows)
            energy = float(result["energy"])
            forces = np.asarray(result["forces"], dtype=float)
            stress = result.get("stress")
        else:
            energy, forces, _variance = self.potential(positions, numbers, rows)
            energy = float(energy)
            forces = np.asarray(forces, dtype=float)
            stress_method = getattr(self.potential, "stress", None)
            stress = stress_method() if callable(stress_method) else None
        if forces.shape != positions.shape:
            raise ValueError(
                f"potential returned forces with shape {forces.shape}, "
                f"expected {positions.shape}"
            )
        tensor = (
            None
            if stress is None
            else np.asarray(stress, dtype=float).reshape(3, 3)
        )
        return energy, np.ascontiguousarray(forces), tensor

    def _finite_difference_virial(
        self, positions: np.ndarray, numbers: np.ndarray, rows: np.ndarray
    ) -> np.ndarray:
        virial = np.zeros((3, 3))
        for i, j in ((0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1)):
            energies = []
            for sign in (1.0, -1.0):
                deform = np.eye(3)
                deform[i, j] += sign * self._STRAIN
                energy, _forces, _stress = self._raw(
                    np.ascontiguousarray(positions @ deform),
                    numbers,
                    np.ascontiguousarray(rows @ deform),
                )
                energies.append(energy)
            value = -(energies[0] - energies[1]) / (2.0 * self._STRAIN)
            virial[i, j] = virial[j, i] = value
        return virial

    def _evaluate(
        self, tags: np.ndarray, xu: np.ndarray, numbers: np.ndarray, rows: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Energy, local-order forces, and virial in LAMMPS units."""
        natoms = tags.size
        if not np.array_equal(np.sort(tags), np.arange(1, natoms + 1)):
            low = tags.min() if natoms else "-"
            high = tags.max() if natoms else "-"
            raise RuntimeError(
                f"rgpot needs atom IDs 1..N; LAMMPS holds {natoms} atoms "
                f"with IDs {low}..{high}"
            )
        index = tags - 1
        positions = np.empty((natoms, 3), dtype=float)
        positions[index] = xu / self.length_per_angstrom
        ordered = np.empty(natoms, dtype=np.int32)
        ordered[index] = numbers
        native_rows = np.ascontiguousarray(rows / self.length_per_angstrom)

        energy, forces, stress = self._raw(positions, ordered, native_rows)
        if not self.periodic or self.virial == "none":
            virial = np.zeros((3, 3))
        elif stress is not None and self.virial == "auto":
            virial = -abs(float(np.linalg.det(native_rows))) * stress
        else:
            virial = self._finite_difference_virial(
                positions, ordered, native_rows
            )

        force_factor = self.energy_per_ev / self.length_per_angstrom
        return (
            energy * self.energy_per_ev,
            forces[index] * force_factor,
            virial * self.energy_per_ev,
        )

    def _step(self, nlocal: int, tag: Any, x: Any, f: Any) -> None:
        lmp = self._lmp
        boxlo, boxhi, xy, yz, xz = lmp.extract_box()[:5]
        rows = box_rows(boxlo, boxhi, xy, yz, xz)
        tags = np.asarray(tag[:nlocal], dtype=np.int64)
        types = np.asarray(lmp.numpy.extract_atom("type")[:nlocal], dtype=np.int64)
        numbers = self.type_numbers[types - 1]
        xu = self._unwrapped(nlocal, x, rows)

        if self._comm is None:
            energy, forces, virial = self._evaluate(tags, xu, numbers, rows)
        else:
            comm = self._comm
            parts = comm.gather((tags, xu, numbers), root=0)
            outcome: tuple[Any, ...]
            shares: list[Any] | None = None
            caught: BaseException | None = None
            if self._root:
                try:
                    all_tags = np.concatenate([part[0] for part in parts])
                    energy, ordered, virial = self._evaluate(
                        all_tags,
                        np.concatenate([part[1] for part in parts]),
                        np.concatenate([part[2] for part in parts]),
                        rows,
                    )
                    outcome = ("", energy, virial)
                    shares = []
                    start = 0
                    for part in parts:
                        stop = start + part[0].size
                        shares.append(ordered[start:stop])
                        start = stop
                except Exception as exc:
                    caught = exc
                    outcome = (
                        str(exc) or type(exc).__name__,
                        0.0,
                        np.zeros((3, 3)),
                    )
                    shares = [np.zeros((part[0].size, 3)) for part in parts]
            problem, energy, virial = comm.bcast(
                outcome if self._root else None, root=0
            )
            forces = comm.scatter(shares, root=0)
            if caught is not None:
                raise caught
            if problem:
                raise RuntimeError(problem)

        f[:nlocal] = forces
        sym = 0.5 * (virial + virial.T)
        lmp.fix_external_set_energy_global(self.fix_id, float(energy))
        lmp.fix_external_set_virial_global(
            self.fix_id,
            [
                sym[0, 0],
                sym[1, 1],
                sym[2, 2],
                sym[0, 1],
                sym[0, 2],
                sym[1, 2],
            ],
        )
        self.energy = float(energy)
        self.evaluations += 1

    def run(self, steps: int, *keywords: str) -> None:
        """Issue ``run`` and re-raise a callback failure in Python."""
        command = " ".join(["run", str(int(steps)), *keywords])
        try:
            self._lmp.command(command)
        except Exception as exc:
            if self.error is not None:
                raise self._take_error() from exc
            raise
        if self.error is not None:
            raise self._take_error()

    def _take_error(self) -> BaseException:
        error, self.error = self.error, None
        self._lmp.set_internal_variable(self._flag, 0.0)
        assert error is not None
        return error

    def close(self) -> None:
        """Remove the halt fix and a ``fix external`` created here."""
        lmp, self._lmp = self._lmp, None
        if lmp is None:
            return
        try:
            lmp.command(f"unfix {self.fix_id}_halt")
            if self._created:
                lmp.command(f"unfix {self.fix_id}")
        except Exception:
            return

    def __enter__(self) -> FixExternal:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
