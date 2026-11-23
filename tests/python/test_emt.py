"""rgpot.EMTPot: the Asap effective-medium potential from Python."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import rgpot

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
