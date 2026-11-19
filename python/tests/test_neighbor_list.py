# MIT License
# Copyright 2023--present rgpot developers

"""Neighbour lists in the shape of ASE's neighbor_list."""

import numpy as np
import pytest

rgpot = pytest.importorskip("rgpot")


def test_three_line_half_list():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float64)
    cell = np.diag([8.0, 8.0, 8.0])
    i, j, d = rgpot.neighbor_list(positions, cell, 2.0)
    assert list(i) == [0]
    assert list(j) == [1]
    assert d[0] == pytest.approx(1.2)
    assert i.base is not None
    assert j.base is not None
    assert d.base is not None


def test_quantity_selector_returns_one_array():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float64)
    cell = np.diag([8.0, 8.0, 8.0])
    d = rgpot.neighbor_list(positions, cell, 2.0, quantities="d")
    assert d.shape == (1,)
    assert d[0] == pytest.approx(1.2)


def test_displacement_and_shift():
    positions = np.array([[0.0, 0.0, 0.0], [7.5, 0.0, 0.0]], dtype=np.float64)
    cell = np.diag([8.0, 8.0, 8.0])
    i, j, D, S = rgpot.neighbor_list(positions, cell, 2.0, quantities="ijDS")
    assert list(i) == [0]
    assert list(j) == [1]
    assert D.shape == (1, 3)
    assert D[0, 0] == pytest.approx(-0.5)
    assert S.shape == (1, 3)
    assert list(S[0]) == [-1, 0, 0]


def test_full_list_offsets_are_sorted_csr():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.2, 1.1, 0.0]], dtype=np.float64)
    cell = np.diag([10.0, 10.0, 10.0])
    i, j, offsets = rgpot.neighbor_list(positions, cell, 2.0, quantities="ij", full=True, offsets=True)
    assert offsets.shape == (4,)
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == len(i)
    for atom in range(3):
        block = slice(int(offsets[atom]), int(offsets[atom + 1]))
        assert list(i[block]) == [atom] * (int(offsets[atom + 1]) - int(offsets[atom]))
        assert list(j[block]) == sorted(j[block])
    assert len(i) == 6


def test_unknown_quantity_is_rejected():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float64)
    cell = np.diag([8.0, 8.0, 8.0])
    with pytest.raises(ValueError):
        rgpot.neighbor_list(positions, cell, 2.0, quantities="ijx")


def test_arrays_view_the_table_after_the_wrapper_is_deleted():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float64)
    cell = np.diag([8.0, 8.0, 8.0])
    wrapped = rgpot.neighbor_list(positions, cell, 2.0)
    i, j, d = wrapped
    address = i.ctypes.data
    assert i.flags.owndata is False
    assert i.base is not None
    assert j.base is not None
    assert int(i[0]) == 0
    assert int(j[0]) == 1
    assert d[0] == pytest.approx(1.2)
    del wrapped, j, d
    assert i.ctypes.data == address
    assert int(i[0]) == 0
    i[0] = 4
    assert int(i[0]) == 4


def _keyed(i, j, d):
    order = np.lexsort((d, j, i))
    return i[order].astype(np.int64), j[order].astype(np.int64), d[order]


def _same_pairs(i_a, j_a, d_a, i_b, j_b, d_b):
    i_a, j_a, d_a = _keyed(i_a, j_a, d_a)
    i_b, j_b, d_b = _keyed(i_b, j_b, d_b)
    assert len(i_a) == len(i_b)
    assert np.array_equal(i_a, i_b)
    assert np.array_equal(j_a, j_b)
    assert np.allclose(d_a, d_b, rtol=0.0, atol=1e-12)


def _unordered(i, j, d):
    """ASE's half list keeps each pair once, but not with vesin's i < j rule."""
    i = np.asarray(i)
    j = np.asarray(j)
    swap = i > j
    return np.where(swap, j, i), np.where(swap, i, j), np.asarray(d)


def _triclinic():
    cell = np.array([[6.0, 0.0, 0.0], [1.2, 5.4, 0.0], [0.3, 0.7, 5.8]], dtype=np.float64)
    frac = np.array(
        [[(a + 0.2) / 4.0, (b + 0.2) / 4.0, (c + 0.2) / 4.0] for a in range(4) for b in range(4) for c in range(4)],
        dtype=np.float64,
    )
    return np.ascontiguousarray(frac @ cell), cell


def _fcc(ncell):
    lattice = 3.615
    basis = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    idx = np.stack(np.meshgrid(np.arange(ncell), np.arange(ncell), np.arange(ncell), indexing="ij"), axis=-1)
    idx = idx.reshape(-1, 3)
    pos = ((idx[:, None, :] + basis[None, :, :]) * lattice).reshape(-1, 3)
    return np.ascontiguousarray(pos), np.diag([ncell * lattice] * 3)


def _vesin(pos, cell, cutoff, full):
    vesin = pytest.importorskip("vesin")

    calc = vesin.NeighborList(cutoff=cutoff, full_list=full, skin=0.0, n_threads=1)
    i, j, d = calc.compute(pos, cell, True, quantities="ijd")
    return np.asarray(i), np.asarray(j), np.asarray(d)


def _ase(pos, cell, cutoff, full):
    from ase import Atoms
    from ase.neighborlist import NeighborList, neighbor_list

    atoms = Atoms(positions=pos, cell=cell, pbc=True)
    if full:
        i, j, d = neighbor_list("ijd", atoms, cutoff)
        return np.asarray(i), np.asarray(j), np.asarray(d)
    nlist = NeighborList(
        [cutoff / 2.0] * len(atoms),
        skin=0.0,
        sorted=False,
        self_interaction=False,
        bothways=False,
    )
    nlist.update(atoms)
    first = []
    second = []
    dist = []
    for i, (indices, offsets) in enumerate(zip(nlist.nl.neighbors, nlist.nl.displacements)):
        for j, shift in zip(indices, offsets):
            vector = pos[j] - pos[i] + shift @ cell
            first.append(i)
            second.append(j)
            dist.append(float(np.linalg.norm(vector)))
    return np.asarray(first), np.asarray(second), np.asarray(dist)


@pytest.mark.parametrize("system", ["tri", "cu108"])
@pytest.mark.parametrize("full", [False, True])
def test_pairs_match_vesin_and_ase(system, full):
    pos, cell = _triclinic() if system == "tri" else _fcc(30)
    cutoff = 5.0
    i, j, d = rgpot.neighbor_list(pos, cell, cutoff, quantities="ijd", full=full)
    vi, vj, vd = _vesin(pos, cell, cutoff, full)
    ai, aj, ad = _ase(pos, cell, cutoff, full)
    _same_pairs(i, j, d, vi, vj, vd)
    if full:
        _same_pairs(i, j, d, ai, aj, ad)
    else:
        _same_pairs(*_unordered(i, j, d), *_unordered(ai, aj, ad))
