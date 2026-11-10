# Copyright (c) anneal developers
A = 2510400.0
C = 2552.24
sig6 = A / C
sigma = sig6 ** (1 / 6)
eps = C / (4 * sig6)
ea, sa = 0.6485, 3.15365
c6 = 4 * ea * sa ** 6
c12 = 4 * ea * sa ** 12
g6, g12 = 2552.24, 2510.4e3
print(f"Jorgensen A = {A:.4f} kJ A^12/mol, C = {C:.6f} kJ A^6/mol")
print(f"exact epsilon = {eps:.7f} kJ/mol, sigma = {sigma:.6f} A")
print(f"anneal epsilon = {ea}, sigma = {sa}: C6 = {c6:.4f}, C12 = {c12:.1f}")
print(
    f"GMIN C6 = {g6}, C12 = {g12:.4e}: equal to Jorgensen C and A: "
    f"{abs(g6 - C) < 1e-9 and abs(g12 - A) < 1e-3}"
)
print(
    f"relative epsilon rounding (anneal/exact - 1) = {ea / eps - 1:.3e}; "
    f"sigma (anneal/exact - 1) = {sa / sigma - 1:.3e}"
)
