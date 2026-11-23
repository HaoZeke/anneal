"""rgpot.EMTPot: the Asap effective-medium potential from Python."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import rgpot
from ase.build import bulk
from rgpot.ase import RgpotCalculator

CLUSTER = np.array(
    [
        [8.1925, 8.1925, 8.1925],
        [10.0, 10.0, 8.1925],
        [10.0, 8.1925, 10.0],
        [8.1925, 10.0, 10.0],
    ]
)
CUBE = np.eye(3) * 20.0
COPPER = [29] * 4


def isolated(body: str) -> subprocess.CompletedProcess:
    """Run body in a fresh interpreter, so an abort fails one test only."""
    prelude = f"import sys; sys.path[:] = {sys.path!r}\n"
    program = prelude + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=120
    )


def test_unsupported_element_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="EMT"):
        rgpot.EMTPot()(CLUSTER, [6, 6, 6, 6], CUBE)


def test_unsupported_element_twice_does_not_crash() -> None:
    run = isolated(
        """
        import numpy as np
        import rgpot

        xyz = np.array([[8.1925, 8.1925, 8.1925], [10.0, 10.0, 8.1925],
                        [10.0, 8.1925, 10.0], [8.1925, 10.0, 10.0]])
        pot = rgpot.EMTPot()
        for _ in range(3):
            try:
                pot(xyz, [6, 6, 6, 6], np.eye(3) * 20.0)
            except ValueError:
                pass
            else:
                raise SystemExit("no error")
        """
    )
    assert run.returncode == 0, run.stderr


def test_a_failed_call_leaves_no_state_behind() -> None:
    pot = rgpot.EMTPot()
    with pytest.raises(ValueError):
        pot(CLUSTER, [6, 6, 6, 6], CUBE)
    energy, forces, _ = pot(CLUSTER, COPPER, CUBE)
    fresh_energy, fresh_forces, _ = rgpot.EMTPot()(CLUSTER, COPPER, CUBE)
    assert energy == fresh_energy
    np.testing.assert_array_equal(forces, fresh_forces)


def test_a_refused_cell_leaves_no_state_behind() -> None:
    pot = rgpot.EMTPot()
    energy, _, _ = pot(CLUSTER, COPPER, CUBE)
    thin = np.diag([6.0, 20.0, 20.0])
    with pytest.raises(ValueError, match="cell"):
        pot(CLUSTER, COPPER, thin)
    assert pot(CLUSTER, COPPER, CUBE)[0] == energy


def test_zero_atoms_is_a_zero_result() -> None:
    run = isolated(
        """
        import numpy as np
        import rgpot

        energy, forces, variance = rgpot.EMTPot()(np.empty((0, 3)), [], np.eye(3) * 20.0)
        print(energy, forces.shape, variance)
        """
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == ["0.0", "(0,", "3)", "0.0"]


@pytest.mark.parametrize(
    "box",
    [
        np.zeros((3, 3)),
        np.array([[20.0, 0.0, 0.0], [40.0, 0.0, 0.0], [0.0, 0.0, 20.0]]),
        np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [20.0, 20.0, 0.0]]),
        np.array([[20.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 20.0]]),
    ],
    ids=["zero", "parallel-rows", "coplanar", "nan"],
)
def test_a_singular_box_is_a_value_error_not_an_abort(box: np.ndarray) -> None:
    run = isolated(
        f"""
        import numpy as np
        import rgpot

        nan = float("nan")
        xyz = np.array({CLUSTER.tolist()!r})
        box = np.array({box.tolist()!r})
        try:
            rgpot.EMTPot()(xyz, [29] * 4, box)
        except ValueError as error:
            print(error)
        else:
            raise SystemExit("no error")
        """
    )
    assert run.returncode == 0, run.stderr
    assert "cell" in run.stdout


def test_non_finite_positions_are_a_value_error_not_an_abort() -> None:
    run = isolated(
        f"""
        import numpy as np
        import rgpot

        xyz = np.array({CLUSTER.tolist()!r})
        xyz[2, 1] = np.inf
        try:
            rgpot.EMTPot()(xyz, [29] * 4, np.eye(3) * 20.0)
        except ValueError as error:
            print(error)
        else:
            raise SystemExit("no error")
        """
    )
    assert run.returncode == 0, run.stderr
    assert "positions" in run.stdout


def _coincident() -> np.ndarray:
    xyz = CLUSTER.copy()
    xyz[1] = xyz[0]
    return xyz


@pytest.mark.parametrize("how", ["call", "evaluate", "evaluate_batch", "hvp_centre"])
def test_coincident_atoms_raise_a_numerical_error(how: str) -> None:
    pot = rgpot.EMTPot()
    xyz = _coincident()
    with pytest.raises(rgpot.NumericalError):
        if how == "call":
            pot(xyz, COPPER, CUBE)
        elif how == "evaluate":
            pot.evaluate(xyz, COPPER, CUBE)
        elif how == "evaluate_batch":
            pot.evaluate_batch([xyz], [COPPER], [CUBE])
        else:
            pot.hvp_centre(xyz, COPPER, CUBE)


def test_a_numerical_error_leaves_the_instance_usable() -> None:
    pot = rgpot.EMTPot()
    with pytest.raises(rgpot.NumericalError):
        pot(_coincident(), COPPER, CUBE)
    energy, forces, _ = pot(CLUSTER, COPPER, CUBE)
    fresh_energy, fresh_forces, _ = rgpot.EMTPot()(CLUSTER, COPPER, CUBE)
    assert energy == fresh_energy
    np.testing.assert_array_equal(forces, fresh_forces)


def test_calculator_tiles_the_primitive_copper_cube() -> None:
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.calc = RgpotCalculator("emt")
    energy = atoms.get_potential_energy()
    large = atoms.repeat(4)
    reference, _, _ = rgpot.EMTPot()(large.positions, [29] * len(large), large.cell.array)
    assert energy == pytest.approx(reference / 64, rel=1e-12)
    np.testing.assert_allclose(atoms.get_forces(), 0.0, atol=1e-10)


@pytest.mark.parametrize(
    ("rasmussen", "numbers"),
    [(False, [1, 12, 13, 28, 29, 46, 47, 78, 79]), (True, [13, 29, 47, 79])],
    ids=["default", "rasmussen"],
)
def test_every_covered_element_tiles_to_a_cell_asap_accepts(
    rasmussen: bool, numbers: list[int]
) -> None:
    for number in numbers:
        atoms = bulk("Cu", "sc", a=3.2, cubic=True)
        atoms.numbers[:] = number
        atoms.calc = RgpotCalculator(rgpot.EMTPot(rasmussen=rasmussen))
        assert np.isfinite(atoms.get_potential_energy()), number
