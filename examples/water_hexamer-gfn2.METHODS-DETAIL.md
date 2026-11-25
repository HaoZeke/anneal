# GFN2-xTB water hexamer

Engine kind `gfn2-xtb-rgpot-v1`. The handle is `libxtb_engine.so`
(`RGPOT_XTB_ENGINE` or `./libxtb_engine.so`).

Pinned handle, from `src/catalog/molecular.rs`: method 3, accuracy 0.01,
electronic temperature 300 K, max iterations 500, charge 0, uhf 0,
vacuum box 60 angstrom. rgpot-core 3.7.0 git rev
`3aa8f9caf87dce22a5e1d74f4a06c9779d0c2212`. eindir-core 0.6.0 checksum
`e330e3abad9c1a597b144d40eb4b07ab3511d79f7d2cdbbcb37a44662f974db0`. The xTB library version is
the one that engine was linked against and is recorded from the loaded
file, not guessed here.

Budget 4000. Seeds 0-31. Arms and the smoke command are in
`water_hexamer.METHODS-DETAIL.md`. Use selector `gfn2`.

A missing `libxtb_engine.so` is an engine-load failure. The smoke file
says `engine_loaded: false` and does not invent an energy.
