The covalent geometry audit now rejects a preserved final geometry whose
digest is missing, and the covalent driver checks non-contiguous coordinate
views instead of treating them as empty geometries. Element-aware call floors
retain the absolute fallback whenever either element has no covalent radius.
