#!/usr/bin/env python3
"""Exercise the checker exits that a passing run never takes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eeq_kernel_check
import d4_periodic_check
import mpmath as mp


def raised_system_exit(fn):
    try:
        fn()
    except SystemExit:
        return True
    return False


def main():
    if not raised_system_exit(lambda: d4_periodic_check.judge(float("nan"), 0.0)):
        raise SystemExit("expected SystemExit")
    if not raised_system_exit(lambda: d4_periodic_check.judge(1.0, 0.0)):
        raise SystemExit("expected SystemExit")
    if not raised_system_exit(lambda: eeq_kernel_check.judge(mp.mpf("1e-6"))):
        raise SystemExit("expected SystemExit")
    if raised_system_exit(lambda: d4_periodic_check.judge(0.0, 0.0)):
        raise SystemExit("a passing comparison must not exit")
    if d4_periodic_check.version_of(object()) != "dftd4-python":
        raise SystemExit("version fallback did not run")
    text = "\nprovenance head=x\n\nenergy_eV 1.0\nforce_eV_A 0.1 0.0 0.0\n"
    provenance, energy, forces = d4_periodic_check.parse_rgpot(text)
    if not provenance.startswith("provenance ") or energy is None or forces.size != 3:
        raise SystemExit("parser missed a line")
    print("failure exits ok")


if __name__ == "__main__":
    main()
