# nm3 gate

Arm A is commit `ef1e1d07ad214d545cfa0bcb0c2e15b8be3bcbcb`, a test-only
change. Arm B is that commit plus the two motif-removal patches. B's
tree is `e54c8935d5627dc140b0ef1417aacdc44173cb07`. Neither arm's
sources are edited for this gate. The search is the recommended
preset at the paper budgets, one fresh process per seed, core 0, one
thread. Even seeds run A then B. Odd seeds run B then A. `rows.csv`
is append-only and carries `code_revision` and `host` on every row.

## Sample size

n = 12, seeds 2000–2011, for every supported system. This was fixed
from walls already stored under `bench/` before any row here existed.
A stored full Au98 run is about 470 s, so 24 runs are about 3.1 h.
M80 at rho 10 is about 2.1 h for 24 runs. LJ38, LJ98 and LJ104 are
n = 12 in any case. No supported system was above 6 h on that record,
so none was reduced to n = 6.

## Systems

Lennard-Jones LJ38, LJ55, LJ75, LJ98, LJ103 and LJ104 use
`h2h_gmin_proof`. Morse M80 at rho 3, 6, 10 and 14 uses `h2h_morse`,
the Morse panel's existing ranges. Au38, Au55, Au75 and Au98 use
`h2h_metal`. Au75 and Au98 keep that example's unset reference, so a
hit is an energy at or below negative infinity and a recorded miss
spends the budget. TIP4P (H2O)8, 12, 15, 16, 20 and 21 use `h2h_seed`
at the recommended water preset and 2_000_000 calls, stopping at the
Cambridge header plus 0.001 kJ/mol.

## Unsupported

Binary Lennard-Jones (`globalopt/lj100-binary`, 42 and 58 atoms) reads
a directory of `coords.N` starts. No such archive is in the tree or
under `bench/`. Morse M75 is not a system in `h2h_morse` or the Morse
panel. Sutton-Chen loads `librgpot.so` at run time. That library is
not vendored, and the build host has no Fortran compiler. The CuH2
slab fixtures are present, and the slab scripts require
`engines/librgpot_cuh2.so`, which is not vendored. Those scripts also
state no reference energy, so a hit is undefined. (H2O)25 is in the
TIP4P size list and has no Cambridge reference.

## Statistics

A hit is the example's own hit. A miss is stored at the budget.
Fisher's exact test on hits is two-sided. The call comparison is the
two-sided Mann-Whitney test, normal approximation, with the tie
correction and the continuity correction. Medians are Hyndman-Fan
type 7. Time inside the potential is `pot_s`. Time outside it is
`outside_s`. Bit-identity is an exact match of hit, calls, charged
count and the printed best energy. B loses if any system has fewer
hits at Fisher p < 0.05, or a higher median call count at
Mann-Whitney p < 0.05.

## Results

<!-- results: start -->
PASS
hit drops:
lj38 hits 11/12 against 12/12 Fisher p 1
lj103 hits 5/12 against 6/12 Fisher p 1
call increases:
lj38 median calls 162770 against 127840 Mann-Whitney p 0.885234
lj55 median calls 7533 against 5722 Mann-Whitney p 0.340569
lj103 median calls 4e+06 against 3.90507e+06 Mann-Whitney p 0.549957
m80r6 median calls 97451 against 62677 Mann-Whitney p 0.885234

system hits_A hits_B fisher median_calls_A median_calls_B mannwhitney pot_A pot_B outside_A outside_B identical
lj38 12/12 11/12 1 127839.5 162769.5 0.8852339145 0.317938 0.4004815 0.370086 0.463768 0/12
lj55 12/12 12/12 1 5722 7533 0.3405685881 0.034468 0.0405685 0.0501005 0.064881 2/12
lj75 5/12 7/12 0.6842726529 4000000 3799817.5 0.6215980188 34.987187 33.1827225 26.0535525 23.1436585 4/12
lj98 7/12 11/12 0.1549525989 3166811.5 1998509 0.2689826348 45.794995 28.996995 29.733658 18.5350985 0/12
lj103 6/12 5/12 1 3905072.5 4000000 0.5499565726 62.19313 64.2946755 37.5153345 39.012598 0/12
lj104 5/12 8/12 0.4136492125 4000000 1083653 0.09500471015 65.2195035 19.7045405 38.8248675 13.1920475 1/12
m80r3 12/12 12/12 1 916054 752990 0.6236048844 27.347892 22.348862 10.7080435 8.917045 0/12
m80r6 12/12 12/12 1 62677 97451 0.8852339145 1.7280745 2.7566695 1.0020095 0.958243 0/12
m80r10 6/12 6/12 1 3539041.5 3380138 0.8774158698 96.547717 89.672214 34.464902 34.5670985 1/12
m80r14 6/12 9/12 0.4003230583 3901429 1034077.5 0.04696196261 103.9786875 28.0254485 36.6384955 9.5606825 0/12
au38 8/12 11/12 0.3167701863 186585 123073.5 0.2124944963 2.36546 1.5612695 0.817339 0.5234825 0/12
au55 4/12 5/12 1 1000000 1000000 0.8681988792 26.148656 26.119434 9.0527835 9.123223 1/12
au75 0/12 0/12 1 4000000 4000000 1 193.78207 193.900307 65.4025195 67.7719445 12/12
au98 0/12 0/12 1 4000000 4000000 1 328.6616155 328.650628 105.12296 105.68223 4/12
tip4p8 12/12 12/12 1 6339 6339 1 0.011949 0.01193 0.007823 0.007874 12/12
tip4p12 12/12 12/12 1 41418.5 41418.5 1 0.1490185 0.1499665 0.069948 0.069944 12/12
tip4p15 7/12 7/12 1 1481619.5 1481619.5 1 7.74677 7.75174 2.6918595 2.6774235 12/12
tip4p16 2/12 2/12 1 2000000 2000000 1 11.725242 11.721142 4.110629 4.1068235 12/12
tip4p20 6/12 6/12 1 1686249.5 1686249.5 1 14.6372655 14.630532 4.032662 4.018594 12/12
tip4p21 0/12 0/12 1 2000000 2000000 1 18.9661775 18.9634865 5.836384 5.8320545 12/12
<!-- results: end -->
