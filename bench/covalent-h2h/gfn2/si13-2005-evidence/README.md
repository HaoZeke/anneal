# Si13 seed 2005 evidence

The Si13 row names driver `c1c02d16` and engine artifact SHA-256
`ccce2a80...`. The panel record maps that artifact to rgpot
`04aceaa79e2336060269bc4ee988397e102f74f1`. That commit is an ancestor
of rgpot `origin/main`. Its tree is
`c27e227c1d802d0fd76cceee31f4414d8ff23ea1`. The release XTB engine
target builds. The full default tree does not: its locked Rust 1.88
environment cannot build `rgpot_fastnlist`, which requires Rust 1.89.

The archived engine artifact was not available. A release rebuild of the
cited tree has SHA-256 `1be58772...`. With the recorded driver, seed,
target, budget, electronic temperature, and one-thread settings, it
stops at call 2948 with -617.68988566 eV. It does not reproduce the old
call 2481 value of -617.18016907 eV. The old row is therefore not
reproducible from the cited source revisions; bit-level reproduction
would additionally require its archived engine binary.

`replay-stop.xyz` is the call-2948 geometry. It is disconnected at a
2.7 Angstrom neighbour cutoff: twelve atoms form one component and one
atom is isolated. A fresh xtb 6.7.1 single point gives
-22.699684612803 Eh, or -617.68985851 eV.

A fresh xtb 6.7.1 `--opt tight` did not converge. Its 43 recorded
frames end at `tight-relax-last.xyz`; a fresh single point on that frame
also fails to converge its SCC in 500 iterations. Thus this replay is
evidence of a disconnected, unstable result, not a lower Si13 cluster
minimum.

The original call-2481 geometry is neither in this repository nor in the
955 paths at Hugging Face dataset `HaoZeke/anneal-paper-data` revision
`7d72d0c8`. It cannot be rescored from the retained evidence.

The panel reference should still be lower than its current
-599.34546694 eV value, but not because of seed 2005. The independently
retained, connected seed-2003 ring in `../minima/si13-ring.xyz` has a
fresh score of -624.70443593 eV in `../minima.csv`.

`replay-summary.txt` records the replay and commands.
`fresh-energies.csv` records the fresh xtb outcomes. No campaign row or
minima ledger is changed.
