`rgpot-eval LJ`, `Morse` and `SuttonChen` without a `cutoff` option use the potential's own default cutoff (15, 9.5 and 40 Angstrom) instead of 0.
`pair_style rgpot` still passes its cutoff.
