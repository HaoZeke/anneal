# Copyright (c) anneal developers
"""A conformer and its mirror image are one species for a molecule without handedness."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "openconf_h2h"))

from h2h_score import ENERGY_TIE, Molecule  # noqa: E402
from rdkit.Chem import rdMolTransforms  # noqa: E402


def gauche(m: Molecule, angle: float):
    import h2h_common as hc

    hc.embed(m.mol, 3)
    conf = m.mol.GetConformer()
    atoms = [a.GetIdx() for a in m.mol.GetAtoms() if a.GetAtomicNum() > 1]
    fl = [i for i in atoms if m.mol.GetAtomWithIdx(i).GetAtomicNum() == 9]
    cs = [i for i in atoms if m.mol.GetAtomWithIdx(i).GetAtomicNum() == 6]
    f1, f2 = fl
    c1 = [c for c in cs if m.mol.GetBondBetweenAtoms(c, f1)][0]
    c2 = [c for c in cs if m.mol.GetBondBetweenAtoms(c, f2)][0]
    rdMolTransforms.SetDihedralDeg(conf, f1, c1, c2, f2, angle)
    x = np.asarray(conf.GetPositions())
    px, pe, valid = m.polish(x)
    assert valid
    return px, float(pe)


def test_the_mirror_image_of_a_conformer_matches_only_when_asked():
    m = Molecule({"smiles": "FCCF"})
    left, right, anti = gauche(m, 62.0), gauche(m, -62.0), gauche(m, 180.0)
    assert abs(left[1] - right[1]) < 1e-3
    assert not m.same(left, right)
    assert m.same_or_mirror(left, right)
    assert m.same_or_mirror(left, left)
    assert not m.same_or_mirror(left, anti)


def turned(smiles: str, quad: tuple[int, int, int, int], angle: float):
    import h2h_common as hc

    m = Molecule({"smiles": smiles})
    hc.embed(m.mol, 7)
    conf = m.mol.GetConformer()
    rdMolTransforms.SetDihedralDeg(conf, *quad, angle)
    px, pe, valid = m.polish(np.asarray(conf.GetPositions()))
    assert valid
    return m, (px, float(pe))


def reflected(conformer):
    return conformer[0] * np.array([1.0, 1.0, -1.0]), conformer[1]


def test_a_mirror_match_of_a_chiral_molecule_is_a_miss():
    # Ethanol-1-d is chiral by its deuterium alone, which the potential and the heavy-atom fit
    # cannot see. The two gauche conformers of one enantiomer differ in the heavy-atom fit, and
    # the reflection of one is the other with the hydrogen and the deuterium exchanged.
    smiles, quad = "C[C@H]([2H])CO", (0, 1, 3, 4)
    m, left = turned(smiles, quad, 62.0)
    _, right = turned(smiles, quad, -62.0)
    assert abs(left[1] - right[1]) < ENERGY_TIE
    assert not m.same(left, right)
    assert m.rmsd(left[0], reflected(right)[0]) < 0.05
    assert m.perceived_key(reflected(right)[0]) != m.reference
    assert not m.same_or_mirror(left, right)
    assert m.same_or_mirror(left, left)


def test_a_mirror_match_of_a_meso_molecule_is_a_hit():
    m, left = turned("C[C@H](F)[C@@H](C)F", (0, 1, 3, 4), 62.0)
    _, right = turned("C[C@H](F)[C@@H](C)F", (0, 1, 3, 4), -62.0)
    assert not m.same(left, right)
    assert m.perceived_key(reflected(right)[0]) == m.reference
    assert m.same_or_mirror(left, right)


def test_mirror_hits_stay_blank_for_rows_written_before_the_column(tmp_path):
    pytest.importorskip("scipy")
    import csv

    from paired_score import existing_fields, mirror_hits

    old = tmp_path / "seeds.csv"
    old.write_text("molecule,arm,seed,hit\nz34,anneal,2048,1\n")
    assert existing_fields(old) == ["molecule", "arm", "seed", "hit"]
    rows = list(csv.DictReader(old.open()))
    assert mirror_hits(rows) == ""
    assert mirror_hits([{"hit_mirror": "1"}, {"hit_mirror": "0"}, {"hit_mirror": "1"}]) == "2/3"
