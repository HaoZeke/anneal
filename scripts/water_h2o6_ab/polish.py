#!/usr/bin/env python3
"""Polish one (H2O)6 geometry on the GFN2 engine with L-BFGS.

Every engine evaluation is counted. The polish ends at the evaluation budget,
when the largest gradient component is below the threshold, or when the line
search can make no further progress. The lowest-energy geometry evaluated is
written, so the polished energy is never above the starting energy.

The engine buffer is the force, minus dE/dx, in eV per angstrom; the objective
gradient is its negative.
"""
import argparse
import ctypes
import sys
import time

import numpy as np
from scipy.optimize import minimize

class XtbConfig(ctypes.Structure):
    _fields_ = [
        ("method", ctypes.c_int),
        ("accuracy", ctypes.c_double),
        ("electronic_temperature", ctypes.c_double),
        ("max_iterations", ctypes.c_int),
        ("charge", ctypes.c_double),
        ("uhf", ctypes.c_int),
    ]


class BudgetExhausted(Exception):
    pass


class EngineFailure(Exception):
    pass


def read_xyz(path):
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    count = int(lines[0].split()[0])
    symbols, coordinates = [], []
    for line in lines[2 : 2 + count]:
        fields = line.split()
        symbols.append(fields[0])
        coordinates.append([float(value) for value in fields[1:4]])
    return symbols, np.array(coordinates, dtype=np.float64)


def atomic_numbers(symbols):
    table = {"H": 1, "O": 8}
    return np.array([table[symbol[0].upper()] for symbol in symbols], dtype=np.int32)


def write_xyz(path, symbols, coordinates, comment):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{len(symbols)}\n{comment}\n")
        for symbol, row in zip(symbols, coordinates):
            handle.write(f"{symbol} {row[0]:.10f} {row[1]:.10f} {row[2]:.10f}\n")


class Engine:
    def __init__(self, library_path):
        self.library = ctypes.CDLL(library_path)
        double_p = ctypes.POINTER(ctypes.c_double)
        self.library.rgpot_xtb_create.restype = ctypes.c_void_p
        self.library.rgpot_xtb_create.argtypes = [
            ctypes.POINTER(XtbConfig),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self.library.rgpot_xtb_destroy.argtypes = [ctypes.c_void_p]
        self.library.rgpot_xtb_force.restype = ctypes.c_int
        self.library.rgpot_xtb_force.argtypes = [
            ctypes.c_void_p,
            ctypes.c_long,
            double_p,
            ctypes.POINTER(ctypes.c_int),
            double_p,
            double_p,
            double_p,
            double_p,
        ]
        config = XtbConfig(3, 0.01, 300.0, 500, 0.0, 0)
        error = (ctypes.c_uint8 * 256)()
        self.handle = self.library.rgpot_xtb_create(ctypes.byref(config), error, 256)
        if not self.handle:
            raise RuntimeError(bytes(error).split(b"\0")[0].decode())
        self.box = np.diag([60.0, 60.0, 60.0]).astype(np.float64)

    def force(self, positions, numbers):
        count = len(numbers)
        buffer = np.zeros(3 * count, dtype=np.float64)
        energy = ctypes.c_double(0.0)
        variance = ctypes.c_double(0.0)
        status = self.library.rgpot_xtb_force(
            self.handle,
            count,
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            numbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(energy),
            ctypes.byref(variance),
            self.box.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return status, energy.value, buffer

    def close(self):
        self.library.rgpot_xtb_destroy(self.handle)


def polish(engine_path, symbols, start, budget, gradient_tolerance, history):
    numbers = atomic_numbers(symbols)
    state = {"calls": 0, "best_energy": np.inf, "best_x": None, "best_grad": None}
    first_energy = [None]

    def objective(flat):
        if state["calls"] >= budget:
            raise BudgetExhausted
        state["calls"] += 1
        positions = np.ascontiguousarray(flat.reshape(-1, 3), dtype=np.float64)
        status, energy, buffer = engine.force(positions, numbers)
        if status != 0 or not np.isfinite(energy) or not np.all(np.isfinite(buffer)):
            raise EngineFailure
        if first_energy[0] is None:
            first_energy[0] = energy
        gradient = -buffer
        if energy < state["best_energy"]:
            state["best_energy"] = energy
            state["best_x"] = positions.copy()
            state["best_grad"] = gradient.copy()
        return energy, gradient

    outcome = "budget"
    begin = time.perf_counter()
    engine = Engine(engine_path)
    try:
        result = minimize(
            objective,
            start.reshape(-1).copy(),
            jac=True,
            method="L-BFGS-B",
            options={
                "maxcor": history,
                "gtol": gradient_tolerance,
                "ftol": 1e-15,
                "maxiter": 10**7,
                "maxfun": 10**7,
                "maxls": 20,
            },
        )
        outcome = "converged" if result.success else "stalled"
    except BudgetExhausted:
        outcome = "budget"
    except EngineFailure:
        outcome = "engine_failure"
    finally:
        wall = time.perf_counter() - begin
        engine.close()
    return state, first_energy[0], outcome, wall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="engine or shim library")
    parser.add_argument("--geometry", required=True, help="starting xyz, angstrom")
    parser.add_argument("--out", required=True, help="polished xyz")
    parser.add_argument("--report", required=True, help="key: value report")
    parser.add_argument("--budget", type=int, required=True, help="engine evaluations")
    parser.add_argument("--gtol", type=float, required=True, help="eV per angstrom")
    parser.add_argument("--history", type=int, required=True, help="L-BFGS corrections")
    args = parser.parse_args()

    symbols, start = read_xyz(args.geometry)
    state, first_energy, outcome, wall = polish(
        args.engine, symbols, start, args.budget, args.gtol, args.history
    )
    lines = [
        f"outcome: {outcome}",
        f"calls: {state['calls']}",
        f"wall_s: {wall:.6f}",
    ]
    if state["best_x"] is not None:
        write_xyz(
            args.out,
            symbols,
            state["best_x"],
            f"polished energy {state['best_energy']:.10f}",
        )
        lines += [
            f"energy_start_eV: {first_energy:.10f}",
            f"energy_final_eV: {state['best_energy']:.10f}",
            f"max_gradient_final_eV_A: {np.max(np.abs(state['best_grad'])):.6e}",
        ]
    with open(args.report, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return 0 if state["best_x"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
