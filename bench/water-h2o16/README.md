# GFN2 (H2O)16 seed 15005

The earlier shim run was dropped: CREST's optimiser was a 160-step FIRE loop, weaker than anneal's 200-step L-BFGS, and the one retained optimisation had hit that cap. This run is native xtb 6.7.1, `crest start.xyz --gfn2 -T 1 --legacy -xnam /work/xtb-gfn2`, so `--opt`, `--md` and `--sp` go to real xtb. An xtb distance restraint is not the one-sided O-H wall, so CREST has no wall, and a structure counts only when every oxygen has exactly two hydrogens within 1.2 Å. Both arms then take one `xtb --gfn 2 --opt vtight` energy; CREST keeps the best of its five lowest ensemble members. A hit is at or below −81.396292669509 Eh by 1e-6.

xtb 6.7.1 (edcfbbe), sha256 271dd6cdd5da127a8e39919e7e32d924790fb346f0057b4183769ba8b3411887.
CREST 3.0.2, commit 6914a25, sha256 16c5de4578000dfd8ea5758da9f5c497778be0dd7a6a1c1e8933030479d0e56e.

| Seed | Arm | Energy (Eh) | Wall (s) | Intact | Hit |
| --- | --- | --- | --- | --- | --- |
| 15005 | CREST | −81.391048653341 | 12838.357 | yes | no |
| 15005 | anneal | −81.389551248650 | 8191.873653 | yes | no |
