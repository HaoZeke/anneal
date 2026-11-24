The covalent geometry audit now rejects a preserved final geometry whose
digest is missing, and the covalent driver checks non-contiguous coordinate
views instead of treating them as empty geometries. Regression coverage keeps
an explicit absolute call floor active when element-aware floors are enabled.
