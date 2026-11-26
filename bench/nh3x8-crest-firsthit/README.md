# CREST first hit on (NH3)8

Reference −35.470769324714 Eh. Tolerance 1e-6 Eh. Threshold −35.470768324714 Eh. Budget 332675 calls. A miss is stored as 332676.

A counted call is a finished MD step or one optimization cycle, the same accounting as `bench/h2o8-crest-firsthit/run.py`. A finished MD step is `MD time /ps` times 1000 divided by `dt /fs`, and only a block that later says finished is counted. An optimization cycle is one `* total energy` line. The cycle energy is the full-precision value from the optimization log. The geometry after the final `:: total energy` line is a later point and is not a counted call. Intact means every nitrogen has three hydrogens within 1.2 Å. `first_hit_call` is the index of the first counted call whose intact geometry is at or below the threshold. The search stops at the budget.

The invocation is `crest start.xyz --gfn2 -T 1 --legacy` with the GFN2 evaluator behind `-xnam` and one thread. The start geometry is the rigid repack from `gfn2_mol_h2h nh3-8 4000 <seed>`, the same generator as `bench/nh3-octamer`. CREST is 3.0.2, revision `6914a25`. The evaluator is xtb 6.7.1, revision `edcfbbe`.

Seeds 17000–17005 have full-run call totals in `bench/nh3-octamer/results.csv`. Metadynamics velocities are not pinned, so a rerun need not repeat those totals.
