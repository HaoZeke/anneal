"""In-process ASE calculator over an rgpot potential.

Units are ASE units: energy in eV, forces in eV/Angstrom, stress in
eV/Angstrom^3. Stress is Voigt order ``(xx, yy, zz, yz, xz, xy)`` with
ASE's sign, ``sigma = (1/V) dE/dstrain``. Per-atom ``stresses`` are the
per-atom virials over the same volume and sum to ``stress``.

``free_energy`` equals ``energy``. The classical potentials have no
electronic entropy, so the two results are the same value.

The ASE calculator contract is that of
Hjorth Larsen et al., J. Phys.: Condens. Matter 29, 273002 (2017),
doi:10.1088/1361-648X/aa680e.

``protocol_reentrant`` is True for :meth:`RgpotCalculator.__rg_evaluate__`
only (F5). :meth:`calculate` and ``get_property`` mutate ``self.atoms``
and ``self.results`` and must not run on two threads at once. The
potential object underneath keeps one workspace per calling thread (F2).

The bound kernels are fully periodic and take the minimum image of each
pair. The calculator turns any ASE geometry into a kernel call that
reproduces the infinite system:

- A cell whose perpendicular width along a periodic axis is not more
  than twice the kernel cutoff is tiled into a supercell that is. The
  energy is divided by the number of copies, and the forces (and
  per-atom values) of the first copy are returned. Stress is intensive
  and needs no correction.
- Every periodic flag false is a cluster: the kernel receives a cubic box
  whose side is at least 100 Angstrom and at least the cluster extent
  plus twice the kernel cutoff, so a periodic image cannot enter the
  cutoff.
- A mixed periodic flag (a slab or a wire) keeps the periodic cell
  vectors and replaces each non-periodic one with a vector orthogonal to
  them, padded the same way as the cluster box.

Energy and forces of a cluster or a slab do not depend on the caller's
cell along a non-periodic axis. Stress does: it is the virial divided by
the caller's cell volume, which is what ``calculate_numerical_stress``
differentiates. A caller with no volume keeps the padded box's stress.
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Iterable, Sequence

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator

_ASE_KEYS = {
    "restart",
    "ignore_bad_restart_file",
    "label",
    "atoms",
    "directory",
}

# Names that are not derived from a class name. Every bound potential
# class is also reachable by its own name, lower case, with or without
# the "Pot" suffix and with "_", "-", "." and spaces ignored.
_ALIASES = {
    "lennardjones": "LJPot",
}

# Neighbour cutoffs of the bound kernels, in Angstrom, for kernels whose
# binding has no config()["cutoff"]. WaterH has no cutoff, so an open
# cell for it uses the 100 Angstrom floor and a periodic cell is not
# tiled.
_KERNEL_CUTOFF_ANGSTROM = {
    "SWPot": 1.8 * 2.0951,
    "StillingerWeberSi": 1.8 * 2.0951,
    "EDIPPot": 3.1213820,
    "LenoskyPot": 4.5,
    "TersoffPot": 3.0,
    "TersoffSiC": 3.0,
    "EAMAlPot": 7.1,
    "FeHePot": 5.4,
    "CuH2Pot": 6.1,
    "WaterHPot": 0.0,
}

_OPEN_BOX_FLOOR_ANGSTROM = 100.0


def _normalise(name: str) -> str:
    return re.sub(r"[\s_\-.]", "", name).lower()


def _potential_classes() -> dict[str, type]:
    """Kind string -> bound potential class, from the compiled module."""
    import rgpot

    found: dict[str, type] = {}
    for name in dir(rgpot):
        cls = getattr(rgpot, name, None)
        if not isinstance(cls, type) or issubclass(cls, BaseException):
            continue
        if not str(getattr(cls, "__module__", "")).startswith("rgpot"):
            continue
        if "__call__" not in vars(cls):
            continue
        key = _normalise(name)
        found.setdefault(key, cls)
        if key.endswith("pot") and len(key) > 3:
            found.setdefault(key[:-3], cls)
    for alias, target in _ALIASES.items():
        cls = getattr(rgpot, target, None)
        if isinstance(cls, type):
            found.setdefault(alias, cls)
    return found


def available_kinds() -> list[str]:
    """Kind strings :class:`RgpotCalculator` accepts in this build."""
    return sorted(_potential_classes())


def _init_parameters(cls: type) -> list[str] | None:
    """Keyword names of the bound constructor, None when unknown."""
    sigs = getattr(getattr(cls, "__init__", None), "__nb_signature__", None)
    if not sigs:
        return None
    names: set[str] = set()
    for sig in sigs:
        text = str(sig[0])
        inner = text[text.find("(") + 1 : text.rfind(")")]
        for part in inner.split(","):
            match = re.match(r"\s*(\w+)\s*:", part)
            if match:
                names.add(match.group(1))
    return sorted(names)


def _build(cls: type, params: dict[str, Any]) -> Any:
    accepted = _init_parameters(cls)
    if accepted is not None:
        unknown = sorted(set(params) - set(accepted))
        if unknown:
            raise ValueError(
                f"unknown {cls.__name__} parameters: {unknown}; "
                f"accepted: {accepted}"
            )
    try:
        return cls(**params)
    except TypeError as err:
        raise ValueError(
            f"{cls.__name__} rejected parameters {sorted(params)}: {err}"
        ) from err


def _resolve_potential(
    potential: Any, params: dict[str, Any]
) -> tuple[Any, tuple[str, dict[str, Any]] | None]:
    """The potential object, and the (class name, parameters) that rebuild it."""
    if not isinstance(potential, str):
        if params:
            raise ValueError(
                "potential parameters are accepted only with a kind string"
            )
        if not (hasattr(potential, "evaluate") or callable(potential)):
            raise TypeError(
                "potential must be an rgpot potential or a kind string"
            )
        return potential, _spec_of(potential)
    classes = _potential_classes()
    cls = classes.get(_normalise(potential))
    if cls is None:
        raise ValueError(
            f"unknown potential kind {potential!r}; this build has "
            f"{sorted(classes)}"
        )
    return _build(cls, dict(params)), (cls.__name__, dict(params))


def _spec_of(potential: Any) -> tuple[str, dict[str, Any]] | None:
    """How to rebuild a bound potential object, None for anything else."""
    import rgpot

    cls = type(potential)
    if getattr(rgpot, cls.__name__, None) is not cls:
        return None
    config = getattr(potential, "config", None)
    if callable(config):
        cfg = config()
        if isinstance(cfg, dict):
            return cls.__name__, dict(cfg)
    if _init_parameters(cls) == []:
        return cls.__name__, {}
    return None


def _kernel_cutoff(potential: Any) -> float:
    """Cutoff the open box and the supercell have to clear, in Angstrom."""
    cutoff = getattr(potential, "cutoff", None)
    if cutoff is not None:
        return float(cutoff)
    config = getattr(potential, "config", None)
    if callable(config):
        cfg = config()
        if isinstance(cfg, dict) and "cutoff" in cfg:
            return float(cfg["cutoff"])
    return float(_KERNEL_CUTOFF_ANGSTROM.get(type(potential).__name__, 0.0))


def _caps(potential: Any) -> dict[str, Any]:
    caps = getattr(potential, "caps", None)
    result = caps() if callable(caps) else None
    return dict(result) if isinstance(result, dict) else {}


def _pbc_flags(pbc: Any) -> np.ndarray:
    flags = np.asarray(pbc, dtype=bool).reshape(-1)
    if flags.size == 1:
        flags = np.repeat(flags, 3)
    if flags.size != 3:
        raise ValueError(f"pbc must be one flag or three, got {int(flags.size)}")
    return flags


def _cluster_extent(positions: Any) -> float:
    """Longest cartesian span of the cluster, in Angstrom."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if pos.shape[0] == 0:
        return 0.0
    span = np.max(pos, axis=0) - np.min(pos, axis=0)
    return float(np.max(span))


def _open_side(extent: float, cutoff: float) -> float:
    """Box length along a non-periodic direction.

    At least 100 Angstrom and at least the extent plus twice the kernel
    cutoff. It is also longer than twice the extent, so a minimum-image
    kernel does not fold the structure.
    """
    side = max(_OPEN_BOX_FLOOR_ANGSTROM, extent + 2.0 * float(cutoff))
    if side <= 2.0 * extent:
        side = 2.0 * extent + 2.0 * float(cutoff)
    return side


def _open_box(positions: Any, cutoff: float) -> np.ndarray:
    """Cubic cell for a non-periodic cluster."""
    side = _open_side(_cluster_extent(positions), cutoff)
    return np.eye(3, dtype=np.float64) * side


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _complement(periodic: np.ndarray, seeds: np.ndarray) -> list[np.ndarray]:
    """Orthonormal directions orthogonal to the periodic cell vectors."""
    if len(periodic) == 2:
        normal = np.cross(periodic[0], periodic[1])
        if np.linalg.norm(normal) <= 1e-12 * np.prod(
            np.linalg.norm(periodic, axis=1)
        ):
            raise ValueError("periodic cell vectors are parallel")
        return [_unit(normal)]
    axis = _unit(periodic[0])
    candidates = list(seeds) + list(np.eye(3))
    for seed in candidates:
        ortho = seed - np.dot(seed, axis) * axis
        if np.linalg.norm(ortho) > 1e-6 * max(np.linalg.norm(seed), 1.0):
            first = _unit(ortho)
            return [first, np.cross(axis, first)]
    raise ValueError("cannot build directions orthogonal to the periodic axis")


def _mixed_box(
    positions: np.ndarray, cell: np.ndarray, flags: np.ndarray, cutoff: float
) -> np.ndarray:
    """Kernel cell for a slab or a wire.

    The periodic rows are the caller's cell vectors. Each non-periodic row
    is orthogonal to them, so the kernel cell has the same lattice in the
    periodic directions, and long enough that no image comes within the
    cutoff.
    """
    periodic_idx = [i for i in range(3) if flags[i]]
    open_idx = [i for i in range(3) if not flags[i]]
    periodic = cell[periodic_idx]
    if np.any(np.linalg.norm(periodic, axis=1) == 0.0):
        names = [("x", "y", "z")[i] for i in periodic_idx]
        raise ValueError(
            f"periodic axes {', '.join(names)} need non-zero cell vectors"
        )
    directions = _complement(periodic, cell[open_idx])
    box = np.zeros((3, 3), dtype=np.float64)
    box[periodic_idx] = periodic
    for row, direction in zip(open_idx, directions):
        projected = positions @ direction
        extent = float(np.ptp(projected)) if projected.size else 0.0
        box[row] = _open_side(extent, cutoff) * direction
    if np.linalg.det(box) < 0.0:
        box[open_idx[-1]] *= -1.0
    return box


def _kernel_box(
    positions: np.ndarray, cell: np.ndarray, flags: np.ndarray, cutoff: float
) -> np.ndarray:
    """Row-major cell passed to a fully periodic kernel."""
    if bool(flags.all()):
        return np.ascontiguousarray(cell, dtype=np.float64)
    if not bool(flags.any()):
        return _open_box(positions, cutoff)
    return _mixed_box(positions, cell, flags, cutoff)


def _replicas(box: np.ndarray, flags: np.ndarray, cutoff: float) -> tuple[int, int, int]:
    """Copies along each periodic axis so every width exceeds twice the cutoff.

    A minimum-image kernel sees every neighbour within the cutoff exactly
    once when the perpendicular width of the cell along each axis is more
    than twice the cutoff.
    """
    if cutoff <= 0.0 or not bool(flags.any()):
        return (1, 1, 1)
    volume = abs(float(np.linalg.det(box)))
    if volume == 0.0:
        raise ValueError("periodic cell has zero volume")
    reps = [1, 1, 1]
    for i in range(3):
        if not flags[i]:
            continue
        j, k = (i + 1) % 3, (i + 2) % 3
        width = volume / float(np.linalg.norm(np.cross(box[j], box[k])))
        if width <= 2.0 * cutoff:
            reps[i] = int(np.floor(2.0 * cutoff / width)) + 1
    return (reps[0], reps[1], reps[2])


def _tile(
    positions: np.ndarray, numbers: np.ndarray, box: np.ndarray, reps: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Supercell whose first len(positions) atoms are the originals."""
    shifts = np.array(
        list(itertools.product(range(reps[0]), range(reps[1]), range(reps[2]))),
        dtype=np.float64,
    )
    offsets = shifts @ box
    tiled = (positions[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
    tiled_numbers = np.tile(numbers, len(shifts))
    super_box = box * np.asarray(reps, dtype=np.float64)[:, None]
    return (
        np.ascontiguousarray(tiled),
        np.ascontiguousarray(tiled_numbers, dtype=np.int32),
        np.ascontiguousarray(super_box),
    )


def _cell_volume(cell: Any) -> float:
    """Absolute volume of a 3x3 cell. A singular cell has volume 0."""
    matrix = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    return float(abs(np.linalg.det(matrix)))


def _voigt(stress: np.ndarray) -> np.ndarray:
    """Row-major 3x3 Cauchy stress to ASE Voigt (xx, yy, zz, yz, xz, xy)."""
    s = np.asarray(stress, dtype=np.float64).reshape(3, 3)
    return np.ascontiguousarray(
        [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]],
        dtype=np.float64,
    )


def _as_f64(array: Any, what: str) -> np.ndarray:
    out = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    if out.dtype != np.float64:
        raise TypeError(f"{what} must be float64")
    return out


class _Geometry:
    """One ASE structure as the kernel sees it."""

    __slots__ = (
        "positions",
        "numbers",
        "box",
        "n_atoms",
        "copies",
        "volume",
        "_stress_scale",
    )

    def __init__(
        self,
        positions: Any,
        numbers: Any,
        cell: Any,
        pbc: Any,
        cutoff: float,
        kernel_periodic: bool,
        name: str,
    ) -> None:
        pos = _as_f64(positions, "positions").reshape(-1, 3)
        num = np.ascontiguousarray(np.asarray(numbers, dtype=np.int32))
        user_cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
        flags = _pbc_flags(pbc)
        if not kernel_periodic and bool(flags.any()):
            raise ValueError(f"{name} has no periodic boundaries; set pbc=False")
        box = _kernel_box(pos, user_cell, flags, cutoff)
        user_volume = _cell_volume(user_cell)
        self.volume = user_volume if user_volume > 0.0 else _cell_volume(box)
        self.n_atoms = int(pos.shape[0])
        reps = _replicas(box, flags, cutoff) if kernel_periodic else (1, 1, 1)
        self.copies = reps[0] * reps[1] * reps[2]
        if self.copies > 1:
            pos, num, box = _tile(pos, num, box, reps)
        self.positions = pos
        self.numbers = num
        self.box = np.ascontiguousarray(box)
        # Kernel stress is -W / V(kernel cell); ASE's is -W / V(caller).
        # A periodic kernel cell is copies times the caller's cell.
        if bool(flags.all()):
            self._stress_scale = 1.0
        else:
            self._stress_scale = _cell_volume(self.box) / (self.copies * self.volume)

    def finish(self, raw: dict[str, Any]) -> dict[str, Any]:
        n = self.n_atoms
        energy = float(raw["energy"])
        forces = np.asarray(raw["forces"], dtype=np.float64)
        if self.copies > 1:
            energy /= self.copies
            forces = forces[:n]
        forces = np.ascontiguousarray(forces)
        out: dict[str, Any] = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
        }
        stress = raw.get("stress")
        if stress is not None:
            voigt = _voigt(stress)
            if self._stress_scale != 1.0:
                voigt = voigt * self._stress_scale
            out["stress"] = voigt
        energies = raw.get("energies")
        if energies is not None:
            out["energies"] = np.ascontiguousarray(
                np.asarray(energies, dtype=np.float64)[:n]
            )
        virials = raw.get("virials")
        if virials is not None:
            out["stresses"] = np.ascontiguousarray(
                -np.asarray(virials, dtype=np.float64)[:n] / self.volume
            )
        return out


def _call(potential: Any, positions: np.ndarray, numbers: np.ndarray, box: np.ndarray) -> dict[str, Any]:
    evaluate = getattr(potential, "evaluate", None)
    if evaluate is not None:
        return evaluate(positions, numbers, box)
    energy, forces, _variance = potential(positions, numbers, box)
    return {"energy": energy, "forces": forces, "stress": None}


class RgpotCalculator(Calculator):
    """ASE calculator that evaluates an rgpot potential in process.

    Parameters
    ----------
    potential:
        An rgpot potential object, or a kind string naming any potential
        class of this build (``"lj"``, ``"morse"``, ``"eam_al"``, ``"sw"``,
        ...; :func:`available_kinds` lists them). Keyword arguments that
        are not ASE calculator arguments are the constructor arguments of
        that class (``u0``, ``cutoff``, ``psi``, ``switch_width`` for LJ;
        ``De``, ``a``, ``re``, ``cutoff``, ``switch_width`` for Morse).
    numerical_stress:
        For a potential whose kernel writes no stress, add ``stress``
        from a central difference of the energy under six cell strains
        of ``stress_step``. Off by default: each stress then costs twelve
        energy calls.
    """

    implemented_properties = ["energy", "free_energy", "forces"]
    # The pure protocol may run concurrently. calculate() writes
    # self.atoms and self.results, so this name is not "calculate is
    # reentrant".
    protocol_reentrant = True

    def __init__(
        self,
        potential: Any,
        *,
        numerical_stress: bool = False,
        stress_step: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        pot_kw = {k: v for k, v in kwargs.items() if k not in _ASE_KEYS}
        ase_kw = {k: v for k, v in kwargs.items() if k in _ASE_KEYS}
        self._potential, self._spec = _resolve_potential(potential, pot_kw)
        self._numerical_stress = bool(numerical_stress)
        self._stress_step = float(stress_step)
        Calculator.__init__(self, **ase_kw)
        if self._spec is not None and self.parameters is not None:
            self.parameters.update(self._spec[1])
            self.parameters["potential"] = self._spec[0]
        self._configure()

    def _configure(self) -> None:
        pot = self._potential
        self._cutoff = _kernel_cutoff(pot)
        self._caps = _caps(pot)
        self._kernel_periodic = bool(self._caps.get("periodic", True))
        self._name = type(pot).__name__
        self._analytic_stress = bool(self._caps.get("stress"))
        self._per_atom = callable(getattr(pot, "evaluate_per_atom", None))
        props = ["energy", "free_energy", "forces"]
        if self._analytic_stress or self._numerical_stress:
            props.append("stress")
        if self._per_atom:
            props.append("energies")
            if self._analytic_stress:
                props.append("stresses")
        self.implemented_properties = props

    @property
    def potential(self) -> Any:
        """The rgpot potential this calculator evaluates."""
        return self._potential

    def set(self, **kwargs: Any) -> dict[str, Any]:
        """ASE ``set``. A changed potential parameter rebuilds the potential."""
        pot_kw = {k: v for k, v in kwargs.items() if k not in _ASE_KEYS}
        if pot_kw and self._spec is None:
            raise ValueError(
                "potential parameters can be set only on a calculator built "
                "from a kind string or a bound potential"
            )
        if "potential" in pot_kw:
            raise ValueError("build a new calculator to change the potential kind")
        changed = Calculator.set(self, **kwargs)
        if pot_kw and any(k in changed for k in pot_kw):
            assert self._spec is not None
            name, params = self._spec
            params = {**params, **pot_kw}
            import rgpot

            self._potential = _build(getattr(rgpot, name), params)
            self._spec = (name, params)
            self._configure()
            self.reset()
        return changed

    def __getstate__(self) -> dict[str, Any]:
        """Pickle and deepcopy rebuild the potential from its parameters."""
        if self._spec is None:
            raise TypeError(
                f"cannot pickle a calculator over {self._name}: build it from "
                "a kind string or a bound rgpot potential"
            )
        state = self.__dict__.copy()
        del state["_potential"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        import rgpot

        vars(self).update(state)
        assert self._spec is not None
        name, params = self._spec
        self._potential = _build(getattr(rgpot, name), params)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] = ("energy",),
        system_changes: Any = all_changes,
    ) -> None:
        """Fill ``self.results``. ``free_energy`` equals ``energy``."""
        Calculator.calculate(self, atoms, properties, system_changes)
        assert self.atoms is not None
        wanted = set(properties)
        per_atom = self._per_atom and bool(wanted & {"energies", "stresses"})
        evaluated = self.__rg_evaluate__(
            self.atoms.positions,
            self.atoms.numbers,
            self.atoms.cell.array,
            self.atoms.pbc,
            per_atom=per_atom,
        )
        self.results.update(evaluated)
        if "stress" not in evaluated and self._numerical_stress and "stress" in wanted:
            self.results["stress"] = self._finite_difference_stress(self.atoms)

    def __rg_evaluate__(
        self,
        positions: Any,
        numbers: Any,
        cell: Any,
        pbc: Any,
        *,
        per_atom: bool = False,
    ) -> dict[str, Any]:
        """Pure evaluation. Does not read or write ``self`` state (F5).

        Returns energy (eV), free_energy (equal to energy), forces
        (eV/Angstrom, shape (N, 3)), and stress in ASE Voigt order when
        the potential reports it. ``per_atom=True`` on a potential with
        ``evaluate_per_atom`` adds ``energies`` (N,) and ``stresses``
        (N, 6).
        """
        geom = self._geometry(positions, numbers, cell, pbc)
        if per_atom and self._per_atom:
            raw = self._potential.evaluate_per_atom(geom.positions, geom.numbers, geom.box)
        else:
            raw = _call(self._potential, geom.positions, geom.numbers, geom.box)
        return geom.finish(raw)

    def _geometry(self, positions: Any, numbers: Any, cell: Any, pbc: Any) -> _Geometry:
        return _Geometry(
            positions,
            numbers,
            cell,
            pbc,
            self._cutoff,
            self._kernel_periodic,
            self._name,
        )

    def _finite_difference_stress(self, atoms: Atoms) -> np.ndarray:
        """Central difference of the energy under cell strain, ASE Voigt."""
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        volume = _cell_volume(cell)
        if volume == 0.0:
            raise ValueError("numerical stress needs a cell with non-zero volume")
        positions = np.asarray(atoms.positions, dtype=np.float64)
        numbers, pbc = atoms.numbers, atoms.pbc
        step = self._stress_step

        def energy(strain: np.ndarray) -> float:
            deform = np.eye(3) + strain
            row = self.__rg_evaluate__(positions @ deform, numbers, cell @ deform, pbc)
            return float(row["energy"])

        # A symmetric shear strain d puts d in both (i, j) and (j, i), so
        # it moves the energy by 2 V sigma_ij d, against V sigma_ii d on
        # the diagonal.
        stress = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                strain = np.zeros((3, 3))
                strain[i, j] = strain[j, i] = step
                denominator = (2.0 if i == j else 4.0) * step * volume
                value = (energy(strain) - energy(-strain)) / denominator
                stress[i, j] = stress[j, i] = value
        return _voigt(stress)

    def get_hessian(self, atoms: Atoms) -> np.ndarray:
        """Analytic Hessian, shape (3N, 3N).

        Raises NotImplementedError. No bound potential reports an analytic
        Hessian.
        """
        del atoms
        raise NotImplementedError(f"{self._name} has no analytic Hessian")

    def calculate_many(
        self,
        frames: Iterable[Atoms],
        properties: Sequence[str] = ("energy", "forces"),
        *,
        attach: bool = False,
        threads: int = 0,
    ) -> list[dict[str, Any]]:
        """Evaluate many structures in one detached batch.

        Results match N single :meth:`calculate` calls, bit for bit, for
        energy and forces. Inputs are not mutated. ``attach=True`` writes
        a :class:`~ase.calculators.singlepoint.SinglePointCalculator` onto
        each :class:`~ase.Atoms`.

        ``threads`` is the BatchEvaluator worker count. 0 selects the
        hardware concurrency. The worker count is not part of the result.
        """
        wanted = tuple(properties)
        frames_list = list(frames)
        # Read the arrays ASE stores. get_positions copies, and that copy
        # was comparable to an EAM evaluation of Pt7. The C++ batch copies
        # into BatchInput before it detaches, so the Atoms objects are not
        # mutated and are not read again.
        geoms = [
            self._geometry(atoms.positions, atoms.numbers, atoms.cell.array, atoms.pbc)
            for atoms in frames_list
        ]
        batch = getattr(self._potential, "evaluate_batch", None)
        if batch is not None:
            raw_rows = batch(
                [g.positions for g in geoms],
                [g.numbers for g in geoms],
                [g.box for g in geoms],
                int(threads),
            )
        else:
            raw_rows = [
                _call(self._potential, g.positions, g.numbers, g.box) for g in geoms
            ]
        rows: list[dict[str, Any]] = []
        for atoms, geom, raw in zip(frames_list, geoms, raw_rows):
            done = geom.finish({"energy": raw["energy"], "forces": raw["forces"]})
            row: dict[str, Any] = {}
            if "energy" in wanted or "free_energy" in wanted:
                row["energy"] = done["energy"]
                row["free_energy"] = done["energy"]
            if "forces" in wanted:
                row["forces"] = done["forces"]
            if "stress" in wanted and self._analytic_stress:
                # BatchEvaluator returns energy and forces. Stress comes
                # from the same kernel on the copied geometry.
                stress_eval = _call(self._potential, geom.positions, geom.numbers, geom.box)
                stressed = geom.finish(stress_eval)
                if "stress" in stressed:
                    row["stress"] = stressed["stress"]
            rows.append(row)
            if attach:
                spc_kw = {
                    key: row[key] for key in ("energy", "forces", "stress") if key in row
                }
                atoms.calc = SinglePointCalculator(atoms, **spc_kw)
        return rows
