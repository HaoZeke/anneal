`ZBLPot` on a stored neighbour list keeps the energy, forces, stress and per-atom terms of one thread on every `OMP_NUM_THREADS`, including on arm64.
The row walk rounds `fscale * d` and then adds that product, as the one-thread walk does.
`ZBLPot.cc` is built with contraction off, as the other classical kernels are, so a compiler that fuses across statements rounds both walks the same way.
