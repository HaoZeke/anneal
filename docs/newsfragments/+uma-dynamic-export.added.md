`scripts/export_uma_aoti.py` exports a dynamic UMA package by default: the atom count (2 to `--max-atoms`, default 1024) and the edge count (1 to `--max-edges`, default 300 per atom) are symbolic, so one package keeps working when an NEB image or a dimer step changes the neighbour list, and, with the MoLE experts kept in the graph, for other molecules of the same task head.
The package is kept only after it matches `FAIRChemCalculator` on the input, on the input with one atom moved past the cutoff, and on a second molecule.
The checkpoint can come from `--checkpoint` or `RGPOT_UMA_CHECKPOINT`.
`UmaPot` refuses an atomic number above the package's `z_max` before the compiled graph runs.
