`RgpotCalculator` now tiles a short periodic cell by the cutoff of the
potential it was given. An `ExprPot` or `SumPot` built from kind strings or
terms without a `cutoff` key, and a GROMACS or OpenMM engine object, resolved
to a cutoff of 0 and gave the wrong energy and forces in a cell narrower than
twice the cutoff. A potential whose cutoff cannot be determined now raises
`ValueError` when the calculator is built.
