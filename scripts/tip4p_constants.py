#!/usr/bin/env python3
"""Jorgensen TIP4P pair constants and the anneal rounding."""

A = 2510400.0
C = 2552.24
epsilon = C * C / (4.0 * A)
sigma = (A / C) ** (1.0 / 6.0)
anneal_epsilon = 0.6485
anneal_sigma = 3.15365
anneal_c6 = 4.0 * anneal_epsilon * anneal_sigma**6
anneal_c12 = 4.0 * anneal_epsilon * anneal_sigma**12
gmin_c6 = 2552.24
gmin_c12 = 2510.4e3

print(f"Jorgensen A = {A:.4f} kJ A^12/mol, C = {C:.6f} kJ A^6/mol")
print(f"exact epsilon = {epsilon:.7f} kJ/mol, sigma = {sigma:.6f} A")
print(
    f"anneal epsilon = {anneal_epsilon}, sigma = {anneal_sigma}: "
    f"C6 = {anneal_c6:.4f}, C12 = {anneal_c12:.1f}"
)
print(
    f"GMIN C6 = {gmin_c6}, C12 = 2510.4e3: "
    f"equal to Jorgensen C and A: {gmin_c6 == C and gmin_c12 == A}"
)
print(
    f"relative epsilon rounding (anneal/exact - 1) = {anneal_epsilon / epsilon - 1:.3e}; "
    f"sigma (anneal/exact - 1) = {anneal_sigma / sigma - 1:.3e}"
)
