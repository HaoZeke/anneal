The PLUMED action `RGPOT` evaluates any potential `pair_style rgpot` accepts: `POTENTIAL=Name OPTIONS={key value ...} TYPES=El,...`, with the names and options of `pair_coeff` (now listed once in the potential names reference).
Positions, the cell, the value and its derivatives follow PLUMED's units, converted with its `Units` object.
Periodic cells, including triclinic GROMACS boxes, go to the potential, and the box derivatives come from its stress, so `BIASVALUE` under `gmx mdrun -plumed` gives the right virial and pressure in NPT runs.
A potential without a stress stops on a periodic cell; `NOPBC` evaluates an isolated cluster.
New meson tests check the forces and box derivatives against `plumed driver --debug-forces`, the units, and the C and Python interfaces, and `gromacs_mdrun` now also checks the bias, virial and pressure of a short Sutton-Chen NPT run.
