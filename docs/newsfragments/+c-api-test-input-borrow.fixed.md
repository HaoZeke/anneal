The C-ABI potential tests in `rgpot-core` keep the position, atomic number and cell arrays on the heap, so the borrowed force input still points at live storage when it is evaluated.
