#!/usr/bin/env python3
"""mpmath check of erf(gamma r)/r - erf(alpha r)/r and its radial derivative."""

import mpmath as mp

mp.mp.dps = 50


def kernel(gamma, alpha, radius):
    return mp.erf(gamma * radius) / radius - mp.erf(alpha * radius) / radius


def dtmp(gam, radius):
    """(d/dr (erf(gam r)/r)) / r, the Cartesian factor used by the solver."""
    r2 = radius * radius
    arg = gam * gam * r2
    return 2 * gam * mp.e ** (-arg) / (mp.sqrt(mp.pi) * r2) - mp.erf(mp.sqrt(arg)) / (r2 * radius)


def judge(worst):
    if worst > mp.mpf("1e-20"):
        raise SystemExit("kernel gradient does not match mpmath")


def main():
    print("command eeq_kernel_check.py")
    worst = mp.mpf(0)
    where = ""
    for gamma in (mp.mpf("0.4"), mp.mpf("0.8"), mp.mpf("1.5")):
        for alpha in (mp.mpf("0.1"), mp.mpf("0.25")):
            for radius in (mp.mpf("0.5"), mp.mpf("1.2"), mp.mpf("3.5"), mp.mpf("8")):
                analytic = dtmp(gamma, radius) - dtmp(alpha, radius)
                numeric = mp.diff(lambda r: kernel(gamma, alpha, r), radius) / radius
                err = abs(analytic - numeric)
                if err > worst:
                    worst = err
                    where = f"gamma={gamma} alpha={alpha} r={radius}"
    print(f"max_abs_dtmp_error {worst}")
    print(f"at {where}")
    judge(worst)
    print("eeq kernel gradient ok")
    print("exit_code=0")


if __name__ == "__main__":
    main()
