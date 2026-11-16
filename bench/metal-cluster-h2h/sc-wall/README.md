# Sutton-Chen wall, anneal against GMIN

A negative record. GMIN's median wall beats anneal's on sc126-55 and
sc96-55 because each anneal call costs more than three GMIN calls, and
almost all of that is inside rgpot's Sutton-Chen entry,
`rgpot_sutton_chen_energy_forces`. The potential is rgpot's, so anneal is
not changed.

`scripts/metal_h2h/sc_wall.sh` builds that entry alone at the rgpot
revision of `Cargo.lock`, with the neighbour-list and SIMD sources it
needs. It runs anneal (`h2h_sc` of the harness revision, the search code of
develop) on seeds 2000-2011 at 1000000 calls, one job at a time on core 0
in a UTS namespace with hostname `bench`. `seeds.csv` holds those rows. The
tables below are the rows of rgpot 8efdd246; `seeds.csv` and `micro.csv`
also hold the rows of rgpot 14680ddb and efd0750f, which `../sc-wall2/README.md`
reads. The
GMIN rows are the corrected-file reruns in `../sc-n12/seeds.csv`. Both
codes time only the potential call: anneal around the C entry, GMIN around
its `SC` routine (`../gmin-sc-time.patch`). Outside time is the rest of the
wall. Per-call times are totals over the twelve seeds divided by charged
calls. Harnesses e7ee1e86 and 91c8dbf0 were local commits, pushed as
3cae4806 and 0e11e9ae with the same patches; `../../revision-aliases.csv`
maps each to its pushed commit.

| System | Anneal hits | GMIN hits | GMIN harness | Anneal calls | GMIN calls | Mann-Whitney p calls | Anneal wall (s) | GMIN wall (s) | Mann-Whitney p wall |
|---|---|---|---|---|---|---|---|---|---|
| sc126-55 | 12/12 | 12/12 | a50f4b02 | 50803 | 30565.5 | 0.285 | 3.414 | 0.576 | 4.78e-04 |
| sc96-55 | 12/12 | 12/12 | a50f4b02 | 28767 | 32261 | 0.507 | 1.917 | 0.652 | 0.0141 |

| System | Code | Inside (µs/call) | Outside (µs/call) | Wall (µs/call) |
|---|---|---|---|---|
| sc126-55 | Anneal | 62.69 | 5.31 | 68.00 |
| sc126-55 | GMIN | 16.72 | 2.12 | 18.84 |
| sc96-55 | Anneal | 62.01 | 5.20 | 67.21 |
| sc96-55 | GMIN | 17.99 | 2.15 | 20.14 |

On sc126-55 anneal's wall per call is 3.61 times GMIN's and its median calls
are 1.66 times GMIN's, at Mann-Whitney p 0.285. Of the 49.2 µs per call
between them, 46.0 µs is inside the potential. On sc96-55 anneal's wall per
call is 3.34 times GMIN's and its median calls are 0.89 times GMIN's, at
Mann-Whitney p 0.507. Of the 47.1 µs per call between them, 44.0 µs is
inside the potential.

`sc_wall_micro` times one call on the seed-2000 start and 63 noisy copies
of it, 20000 calls, three times; `micro.csv` holds each repeat and the
table gives the medians in µs. Entry is the C entry. Entry with dlsym and
allocation adds the symbol lookup, the force buffer and the sign flip of
`h2h_sc`. Kept builds one `SuttonChenPot` and calls it again and again, with
its default neighbour mode and with the streaming walk. Plain is a
half-pair O(N²) loop that evaluates the two powers of a pair once and keeps
them for the force pass. On one of the 64 geometries, Entry and Plain give
the same energy and forces, bit for bit.

| System | Entry | Entry with dlsym and allocation | Kept | Kept, streaming | Plain | max abs dE | max abs dF |
|---|---|---|---|---|---|---|---|
| sc126-55 | 58.19 | 58.44 | 30.95 | 23.80 | 9.86 | 0 | 0 |
| sc96-55 | 59.78 | 60.12 | 32.54 | 25.23 | 10.85 | 0 | 0 |

The entry builds a new `SuttonChenPot` on every call, so its neighbour list
is built from nothing each time, with a cutoff of 1e6 in place of none, and
it evaluates both powers of every pair in the density pass and again in the
force pass. On sc126-55 the entry takes 48.3 µs per call more than the plain
loop: 27.2 µs to build the potential and its neighbour list again, 7.2 µs
for the stored list over the streaming walk, and 13.9 µs for the walk and
the second evaluation of both powers. On sc96-55 the entry takes 48.9 µs per
call more than the plain loop: 27.2 µs to build the potential and its
neighbour list again, 7.3 µs for the stored list over the streaming walk,
and 14.4 µs for the walk and the second evaluation of both powers.

On sc126-55, 50803 calls at 9.9 + 5.3 µs would take 0.771 s, against GMIN's
median 0.576 s. On sc96-55, 28767 calls at 10.8 + 5.2 µs would take 0.462 s,
against GMIN's median 0.652 s. That is arithmetic on medians, not a run.

What anneal's side adds is the symbol lookup, the force buffer and the sign
flip, 0.25 µs per call on sc126-55 and 0.34 µs on sc96-55, and the time
outside the potential, 3.2 µs per call more than GMIN's on sc126-55 and 3.0
µs on sc96-55.