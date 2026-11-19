#!/usr/bin/env python3
"""Check the sign convention of the buffer a GFN2 engine library hands back.

The buffer is compared with a central finite difference of the energy on a
slightly displaced geometry. A buffer that is the force, minus dE/dx, gives a
ratio of -1; a buffer that is dE/dx gives +1.

To check the engine shim, pass the real engine and the buffer mode the shim
should produce; they are given to the shim before it is loaded.
"""
import argparse
import ctypes
import os

import numpy as np


class XtbConfig(ctypes.Structure):
    _fields_ = [
        ("method", ctypes.c_int),
        ("accuracy", ctypes.c_double),
        ("electronic_temperature", ctypes.c_double),
        ("max_iterations", ctypes.c_int),
        ("charge", ctypes.c_double),
        ("uhf", ctypes.c_int),
    ]


def read_xyz(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    count = int(lines[0].split()[0])
    symbols, coordinates = [], []
    for line in lines[2 : 2 + count]:
        fields = line.split()
        symbols.append(fields[0])
        coordinates.append([float(value) for value in fields[1:4]])
    return symbols, np.array(coordinates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="engine or shim library to load")
    parser.add_argument("--geometry", required=True, help="xyz of a minimum, angstrom")
    parser.add_argument("--real-engine", help="engine behind the shim")
    parser.add_argument("--buffer", choices=("force", "gradient"), help="buffer mode of the shim")
    args = parser.parse_args()
    if args.real_engine:
        os.environ["XTB_SHIM_REAL"] = args.real_engine
        os.environ["XTB_SHIM_BUFFER"] = args.buffer or "force"

    lib = ctypes.CDLL(args.library)
    double_p = ctypes.POINTER(ctypes.c_double)
    lib.rgpot_xtb_create.restype = ctypes.c_void_p
    lib.rgpot_xtb_create.argtypes = [ctypes.POINTER(XtbConfig), ctypes.c_char_p, ctypes.c_size_t]
    lib.rgpot_xtb_force.restype = ctypes.c_int
    lib.rgpot_xtb_force.argtypes = [
        ctypes.c_void_p, ctypes.c_long, double_p, ctypes.POINTER(ctypes.c_int),
        double_p, double_p, double_p, double_p,
    ]
    config = XtbConfig(3, 0.01, 300.0, 500, 0.0, 0)
    error = ctypes.create_string_buffer(256)
    handle = lib.rgpot_xtb_create(ctypes.byref(config), error, 256)
    if not handle:
        raise SystemExit(error.value.decode())

    symbols, start = read_xyz(args.geometry)
    start = start + 0.05 * np.random.default_rng(1).standard_normal(start.shape)
    numbers = np.array([8 if s[0].upper() == "O" else 1 for s in symbols], dtype=np.int32)
    box = np.diag([60.0, 60.0, 60.0]).ravel().copy()

    def evaluate(flat):
        flat = np.ascontiguousarray(flat)
        buffer = np.zeros_like(flat)
        energy, variance = ctypes.c_double(), ctypes.c_double()
        status = lib.rgpot_xtb_force(
            handle, len(numbers), flat.ctypes.data_as(double_p),
            numbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            buffer.ctypes.data_as(double_p), ctypes.byref(energy), ctypes.byref(variance),
            box.ctypes.data_as(double_p),
        )
        if status != 0:
            raise SystemExit(f"engine returned {status}")
        return energy.value, buffer

    flat = start.reshape(-1).copy()
    energy, buffer = evaluate(flat)
    print(f"energy_eV: {energy:.9f}")
    step = 1e-4
    for index in (0, 4, 7, 13, 30):
        plus, minus = flat.copy(), flat.copy()
        plus[index] += step
        minus[index] -= step
        slope = (evaluate(plus)[0] - evaluate(minus)[0]) / (2 * step)
        print(f"coordinate {index:2d}: dE/dx finite difference {slope:+.6f}  buffer {buffer[index]:+.6f}"
              f"  buffer/difference {buffer[index] / slope:+.4f}")


if __name__ == "__main__":
    main()
