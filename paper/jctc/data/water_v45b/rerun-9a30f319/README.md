# Water-hexamer 192-seed rerun

These are the unedited per-seed `status.json` receipts from a fresh rerun of
the committed v4.5b workflow at
`56160f2d100fc2c9f7365f4ee6d1d86653c8b31f` (tree
`149899211dbec73aa88877cb7af0ebdf9b99f39e`).  Each arm ran seeds 0--191
with a 4000-call budget and the committed `h2o6` configuration.

The rerun produced 121/192 hits for anneal and 94/192 hits for plain CREST.
The corresponding median walls were 16.9731480 s and 13.8815665 s.
The repository record revision is
`9a30f3190b3bb7040c675a904f8388eb68224151`, whose tree is
`4cf101da6fe8f4d1069f5cb472413bd7e232d69c`.

No standalone xtb executable or libxtb package was used, so this rerun has
no xtb 6.x software version.  It evaluated GFN2-xTB method 3 through tblite
0.7.0 (tag `v0.7.0`, commit
`fa8a4416e8fe093d0075bc10ac875494c2a449a9`).  CREST 3.0.2 was linked to
that same tblite library.

The workflow used these inputs:

- `config/config.yaml`: `a79a8621f88d9492d76a3bead231427e9dca8a79240a87f1c9c28d6474230ad8`
- `config/benchmarks.yaml`: `04338d1c7deacb2837ec26e06d2dba39da1f4bb0e0d5eb10e08b27c4b16f2eb5`
- `config/lj-run.yaml`: `8c6844e595657d8d5c6b3bf43487417b1f643bb33eac2360576f60693503a49b`
- `workflow/molecular/water-v10.xyz`: `faaba2431950230f1c57c5bfed88889e2636ed05d5f8cd1ad142ccec8838c99a`
- anneal `water_hexamer`: `130b9d8056b552ecea6052afa2bbfba5fd194c82f9b0faca10d9cf9f600d4fde`
- CREST 3.0.2: `c8854768a10c748ed230b2b8b6790e4afc6375aa8fc4a9b9bb3aa6a3bfeec36e`

The source pins are anneal
`36086e988c5a71c156ffcbde152bf321a6f5ef25`, CREST
`af7eb9927e2b36e24b14055f9eba3bea5be0014e`, and tblite
`fa8a4416e8fe093d0075bc10ac875494c2a449a9`.  The status receipts record
the binary, start-structure, and hardware hashes used by each seed.
`SHA256SUMS` covers all 384 receipts.
