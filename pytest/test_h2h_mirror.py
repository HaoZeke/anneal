# Copyright (c) anneal developers
"""A conformer and its mirror image are one species for a molecule without handedness."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "openconf_h2h"))

from h2h_score import Molecule  # noqa: E402
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
