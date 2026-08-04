"""Tests for D15–D19 landscape / force-ledger math."""

from proofs.d15_staircase_transform import WITNESS as W15
from proofs.d16_minima_graph_force import WITNESS as W16
from proofs.d16_minima_graph_force import expected_force, symbolic_linear_system_two_transient
from proofs.d17_superbasin_and_compression import WITNESS as W17
from proofs.d18_force_lower_bounds import (
    L3_composition,
    L5_expected_force,
    L5_per_hop_p,
    WITNESS as W18,
)
from proofs.d19_disconnectivity_depth import WITNESS as W19, funnel_depth
import numpy as np


def test_d15():
    assert W15


def test_d16():
    assert W16


def test_d16_two_state_symbolic():
    assert symbolic_linear_system_two_transient()


def test_d17():
    assert W17


def test_d18():
    assert W18


def test_d18_wales_inversion():
    p = L5_per_hop_p(0.04, 5000)
    assert abs(1 - (1 - p) ** 5000 - 0.04) < 1e-12
    assert L5_expected_force(1.0, 0.04, 5000) > 5000


def test_d18_L3():
    assert abs(L3_composition(0.25, 8.0, 4.0, 0.2) - (32.0 + 20.0)) < 1e-12


def test_d19():
    assert W19


def test_d19_depth_gm():
    E = np.array([2.0, 0.0, 1.0])
    W = np.array([[0.0, 3.5, 4.0], [3.5, 0.0, 2.5], [4.0, 2.5, 0.0]])
    assert abs(funnel_depth(E, W, 1, 1)) < 1e-15
