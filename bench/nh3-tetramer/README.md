# GFN2 (NH3)4

A second CREST comparison, on ammonia rather than water. The search is
`Config::recommended_molecular` with no added wall and no constant changed
for this molecule. The budget is the catalog GFN2 molecular budget, 4000
charged evaluations, the same number as `GFN2_HEXAMER_BUDGET`.

The published minimum used as the reference is the cyclic hydrogen-bonded
tetramer. Beu and Buck, J. Chem. Phys. 114, 7848 (2001), report a nearly
planar ring with an N–N distance of 3.226 Å and the nitrogens displaced by
about 0.4 Å to either side of the plane. That motif was built from the
GFN2-xTB 6.7.1 ammonia monomer and relaxed with

```
xtb in.xyz --gfn 2 --opt vtight --acc 0.01 --iterations 500
```

The relaxed energy is −17.729616186942 Eh. Four separated monomers at the
same settings sum to −17.704976147576 Eh, so the binding is 15.46 kcal/mol.
That sits next to the counterpoise-corrected MP2 binding of about 15.5
kcal/mol for the cyclic tetramer reported by Janeiro-Barral.

A hit is a vtight energy at or below −17.729616186942 Eh by 1e-6. A
structure counts only when every nitrogen has exactly three hydrogens
within 1.2 Å, the same neighbour cutoff the water scorer uses. CREST is
`crest start.xyz --gfn2 -T 1 --legacy -xnam` real xtb 6.7.1, with no wall.
Both arms share one rigid repack. Seeds 17000–17005 run serially. The arm
order is one random byte per seed. Every timed command is

```
sudo unshare --uts sh -c 'hostname bench; . /work/bench-env.sh; exec <cmd>'
```

Anneal's `pot_s` is time inside `value_and_gradient`. CREST's inside time
is the sum of the real xtb process walls. A CREST call is an MD step from
the log plus one `* total energy` line from each `--opt` invocation.
`outside_s` is the wall minus the inside time.
