#!/usr/bin/env python3
"""Leftover RMS of featomic calculators on ico75, water, and a slab."""

from __future__ import annotations

import sys

import numpy as np


def load_xyz(path: str) -> np.ndarray:
    vals = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    vals.extend(float(parts[k]) for k in (1, 2, 3))
                except ValueError:
                    continue
            elif len(parts) == 3:
                try:
                    vals.extend(float(p) for p in parts)
                except ValueError:
                    continue
    x = np.asarray(vals, dtype=np.float64)
    assert x.size % 3 == 0 and x.size > 0, path
    return x.reshape(-1, 3)


def leftover_rms(name: str, hypers: dict, positions: np.ndarray, numbers) -> float:
    import ase
    import featomic

    calc = getattr(featomic, name)(**hypers)
    atoms = ase.Atoms(numbers=numbers, positions=positions, pbc=False)
    d = calc.compute(atoms)
    # collapse sparse keys so one block remains
    for key in list(d.keys.names):
        if key.endswith("_type") or key == "o3_lambda" or key == "o3_sigma":
            try:
                d = d.keys_to_properties(key) if key.startswith("neighbor") or key.startswith("o3") else d.keys_to_samples(key)
            except Exception:
                try:
                    d = d.keys_to_properties(key)
                except Exception:
                    d = d.keys_to_samples(key)
    block = d.block()
    raw = np.asarray(block.values, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    # leftover: per-centre minus species-conditioned mean
    mu = raw.mean(axis=0)
    delta = raw - mu
    return float(np.sqrt(np.mean(delta * delta)))


SOAP = {
    "cutoff": {"radius": 3.5, "smoothing": {"type": "ShiftedCosine", "width": 0.5}},
    "density": {"type": "Gaussian", "width": 0.35},
    "basis": {
        "type": "TensorProduct",
        "max_angular": 6,
        "radial": {"type": "Gto", "max_radial": 3},
    },
}
RADIAL = {
    "cutoff": {"radius": 3.5, "smoothing": {"type": "ShiftedCosine", "width": 0.5}},
    "density": {"type": "Gaussian", "width": 0.35},
    "basis": {"type": "TensorProduct", "max_angular": 0, "radial": {"type": "Gto", "max_radial": 6}},
}
LODE = {
    "cutoff": {"radius": 3.5, "smoothing": {"type": "ShiftedCosine", "width": 0.5}},
    "density": {"type": "SmearedPowerLaw", "smearing": 1.0},
    "basis": {
        "type": "TensorProduct",
        "max_angular": 3,
        "radial": {"type": "Gto", "max_radial": 3},
    },
}


def report(label: str, pos: np.ndarray, numbers) -> None:
    print(f"=== {label} n={len(pos)} ===")
    for name, hyp in (
        ("SoapPowerSpectrum", SOAP),
        ("SoapRadialSpectrum", RADIAL),
        ("SphericalExpansion", SOAP),
    ):
        try:
            rms = leftover_rms(name, hyp, pos, numbers)
            print(f"{name:24s} leftover_rms {rms:.6e}")
        except Exception as e:
            print(f"{name:24s} FAIL {type(e).__name__}: {e}")
    try:
        rms = leftover_rms("LodeSphericalExpansion", LODE, pos, numbers)
        print(f"{'LodeSphericalExpansion':24s} leftover_rms {rms:.6e}")
    except Exception as e:
        print(f"{'LodeSphericalExpansion':24s} FAIL {type(e).__name__}: {e}")


def water4():
    proto = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    pos = []
    z = []
    origins = [(0, 0, 0), (2.3, 0.1, 0.0), (0.1, 2.3, 0.0), (2.2, 2.2, 0.2)]
    for o in origins:
        for p in proto:
            pos.append(p + o)
        z.extend([8, 1, 1])
    return np.asarray(pos), z


def main():
    ico = sys.argv[1] if len(sys.argv) > 1 else "ico75.xyz"
    x = load_xyz(ico)
    report("ico75", x, [18] * len(x))
    wpos, wz = water4()
    report("water4", wpos, wz)


if __name__ == "__main__":
    main()
