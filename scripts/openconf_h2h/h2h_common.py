# Copyright (c) anneal developers
"""Molecules, the MMFF94 potential and the force-call counter shared by the arms."""

from __future__ import annotations

import csv
import ctypes
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

KCAL_PER_EV = 23.060547830619026
ROOT = Path(__file__).resolve().parents[2]
MOLECULES = ROOT / "bench" / "openconf-h2h" / "molecules.csv"

# Lipinski's strict rotatable bond, without the acyclic condition. The ring
# condition is applied separately so a macrocycle bond can count.
_ROTOR = Chem.MolFromSmarts(
    "[!$([NH]!@C(=O))&!D1&!$(*#*)]-[!$([NH]!@C(=O))&!D1&!$(*#*)]"
)
SMALL_RING = 7


def molecules(path: Path = MOLECULES) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["name"]: row for row in csv.DictReader(handle)}


def build_mol(smiles: str) -> Chem.Mol:
    """Hydrogen-complete molecule in a fixed atom order."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES {smiles}")
    return Chem.AddHs(mol)


def reference_smiles(mol: Chem.Mol) -> str:
    """Isomeric SMILES of the input graph, the stereo every conformer keeps."""
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def rigid_groups(mol: Chem.Mol) -> list[list[int]]:
    """Atoms between rotatable bonds, each with its hydrogens.

    A bond is cut when it matches the strict rotatable pattern and lies in
    no ring of seven or fewer atoms. That is Lipinski's strict count on an
    acyclic molecule, and it also cuts the single bonds of a macrocycle.
    """
    ring_info = mol.GetRingInfo()
    cut = []
    for begin, end in mol.GetSubstructMatches(_ROTOR):
        bond = mol.GetBondBetweenAtoms(begin, end)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        idx = bond.GetIdx()
        small = any(
            idx in ring and len(ring) <= SMALL_RING for ring in ring_info.BondRings()
        )
        if not small:
            cut.append(idx)
    if not cut:
        return [list(range(mol.GetNumAtoms()))]
    pieces = Chem.FragmentOnBonds(mol, sorted(set(cut)), addDummies=False)
    groups = [sorted(group) for group in Chem.GetMolFrags(pieces)]
    return sorted(groups)


def mmff94(mol: Chem.Mol, conf_id: int = -1):
    """RDKit MMFF94 at a constant dielectric of one.

    The non-bonded list is RDKit's default, every pair within 100 Å of the
    geometry the field is built on, the same list openconf builds.
    """
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    if props is None:
        raise ValueError("MMFF94 cannot type this molecule")
    props.SetMMFFDielectricConstant(1.0)
    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    if ff is None:
        raise ValueError("no MMFF94 force field")
    return ff


def embed(mol: Chem.Mol, seed: int) -> np.ndarray:
    """One ETKDGv3 geometry with no force-field cleanup, centred."""
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    with phase(1):
        cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        raise RuntimeError(f"ETKDG failed for seed {seed}")
    x = mol.GetConformer(cid).GetPositions()
    return x - x.mean(axis=0)


class _Counter:
    """Reads the LD_PRELOAD counter in scripts/openconf_h2h/rdff_count.c."""

    def __init__(self) -> None:
        lib = ctypes.CDLL(None)
        try:
            self._read = lib.rdff_read
            self._reset = lib.rdff_reset
            self._phase = lib.rdff_set_phase
        except AttributeError as err:
            raise RuntimeError("librdff_count.so is not preloaded") from err
        self._phase.argtypes = [ctypes.c_int]

    def reset(self) -> None:
        self._reset()

    def set_phase(self, value: int) -> None:
        self._phase(int(value))

    def read(self) -> dict[str, float]:
        buf = (ctypes.c_uint64 * 6)()
        self._read(buf)
        return {
            "energy_calls": int(buf[0]),
            "grad_calls": int(buf[1]),
            "pot_s": buf[2] * 1e-9,
            "embed_energy_calls": int(buf[3]),
            "embed_grad_calls": int(buf[4]),
            "embed_s": buf[5] * 1e-9,
        }


_COUNTER: _Counter | None = None


def counter() -> _Counter:
    global _COUNTER
    if _COUNTER is None:
        _COUNTER = _Counter()
    return _COUNTER


class phase:
    """Attribute force-field calls inside the block to ETKDG embedding."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __enter__(self):
        try:
            counter().set_phase(self.value)
        except RuntimeError:
            pass
        return self

    def __exit__(self, *exc):
        try:
            counter().set_phase(0)
        except RuntimeError:
            pass
        return False


def write_xyz(path: Path, symbols: list[str], frames: list[tuple[np.ndarray, float]]) -> None:
    lines = []
    for x, energy in frames:
        lines.append(str(len(symbols)))
        lines.append(f"{energy:.10f}")
        for sym, (a, b, c) in zip(symbols, np.asarray(x).reshape(-1, 3), strict=True):
            lines.append(f"{sym:2s} {a:16.10f} {b:16.10f} {c:16.10f}")
    path.write_text("\n".join(lines) + "\n")


def read_xyz(path: Path) -> list[tuple[list[str], np.ndarray, str]]:
    frames = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        comment = lines[i + 1]
        symbols, coords = [], []
        for line in lines[i + 2 : i + 2 + n]:
            parts = line.split()
            symbols.append(parts[0])
            coords.append([float(v) for v in parts[1:4]])
        frames.append((symbols, np.asarray(coords), comment))
        i += 2 + n
    return frames


def symbols(mol: Chem.Mol) -> list[str]:
    return [atom.GetSymbol() for atom in mol.GetAtoms()]
