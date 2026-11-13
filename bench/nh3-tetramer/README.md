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

Seeds 17001 and 17002 are the anneal arm after the covalent call wall.
Their first anneal rows are kept and are not the rows below. Every CREST
row is the original arm. All twelve geometries are intact. All twelve hit.

| seed | anneal E / Eh | anneal wall / s | anneal inside / call | anneal outside / call | CREST E / Eh | CREST wall / s | CREST inside / call | CREST outside / call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17000 | −17.729688698035 | 10.476 | 2.608 ms | 0.012 ms | −17.729688627818 | 460.043 | 2.703 ms | 0.465 ms |
| 17001 | −17.729672551197 | 10.439 | 2.600 ms | 0.010 ms | −17.729688563018 | 436.973 | 2.449 ms | 0.418 ms |
| 17002 | −17.729685512974 | 10.482 | 2.610 ms | 0.010 ms | −17.729688648400 | 434.564 | 2.512 ms | 0.452 ms |
| 17003 | −17.729688700714 | 10.821 | 2.694 ms | 0.011 ms | −17.729688643040 | 435.635 | 2.544 ms | 0.420 ms |
| 17004 | −17.729688671575 | 10.593 | 2.637 ms | 0.011 ms | −17.729688630353 | 446.662 | 2.479 ms | 0.418 ms |
| 17005 | −17.729688660029 | 10.506 | 2.615 ms | 0.011 ms | −17.729688660593 | 414.636 | 2.446 ms | 0.433 ms |

Anneal is 4000 charged calls on every seed. CREST calls are 145202, 152380,
146611, 146979, 154197 and 144033.
