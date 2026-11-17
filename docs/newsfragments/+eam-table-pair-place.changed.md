`EAMTablePot` locates each pair distance on its radial grid once per pass.
It located it once per table before: twice per pair in the density pass and three times in the force pass.
It also keeps the three radial tables of each element pair together instead of indexing nested vectors for every pair.
The arithmetic is the same, so the results do not change.
On one rank with `mode domain`, a 32000-atom CuNi step of `pair_style rgpot` went from 54.13 to 47.03 ms, against 43.87 ms for `eam/alloy`.
