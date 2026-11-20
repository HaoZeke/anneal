Copyright (c) anneal developers

# The lattice ranks Morse on the Lennard-Jones range

Negative record. Nothing here is proposed for develop.

Pair scale stops on two shelves that recommended leaves. On
m55 rho=14 seed 2006 both record -220.49847976, recommended
at calls 286543 and 336869 and pair scale at call 221283.
Recommended hits 55C at call 519192. On m80 rho=14 seed 2010
both record -340.38733557, recommended at call 1514022 and
pair scale at call 2087416. Recommended hits 80G at call
2973511. Pair scale's best is that shelf on both seeds.

## From the first shared-shelf call

Counts are hops that reach the acceptance test while the
best is still the shelf. A same-basin accept lands within
1e-4 of the chain energy. Displacement is the root sum of
squares over the coordinates.

m55 seed 2006, recommended, calls 286572 to 518627. 3889
hops, 1420 accepted (36.5%): 103 same basin, 592 uphill,
725 downhill. The chain stands on the shelf energy for 53
of those hops. Mean proposal displacement 2.583, mean
accepted quench move 2.577. The stall trail fires 0 times.
The climb fires 45 times, 7 return to the same energy, and
the lowest landing is the shelf, -220.49847976. The walk
leaves at call 518627 on the lattice arm: the chain is at
-216.08347920, the quench is -220.63363359, and the
proposal displacement is 2.023. The hit is the next record,
call 519192.

m55 seed 2006, pair scale, from call 221283 through the
budget. 6075 hops, 2245 accepted (37.0%): 119 same, 966
up, 1160 down. The chain stands on the shelf for 29 hops.
Mean proposal displacement 2.633. Trail 0. Climb 148, 32
return, lowest landing -220.49001260, above the shelf.
The best stays -220.49847976.

m80 seed 2010, recommended, from call 1514022. 22554 hops,
7955 accepted (35.3%): 291 same, 3796 up, 3868 down. The
chain stands on the shelf for 8 hops. Mean proposal
displacement 3.183. Trail 0. Climb 286, 72 return, lowest
landing -339.83568870. The hit at call 2973511 is the
lattice arm: the chain is at -339.81119326 and the quench
is -340.81137066. The climb never beats the shelf.

m80 seed 2010, pair scale, from call 2087416 through the
budget. 15561 hops, 6109 accepted (39.3%): 335 same, 3351
up, 2423 down. The chain stands on the shelf for 3 hops.
Mean proposal displacement 2.874. Trail 0. Climb 372, 108
return, lowest landing -340.35713133. The best stays
-340.38733557.

The acceptance rates sit together, and the stall climb
does not beat the shelf on either arm. Both leaves are
lattice exchanges. On m55 that arm accepts 857 of 1165
draws under recommended and 1307 of 1754 under pair scale.

## Step size against the Morse range

At the pair minimum, Morse rho=14 has second derivative
2 rho^2 = 392, so the unit-energy width 1/sqrt(V'') is
0.050508. Lennard-Jones at its minimum has second
derivative 57.146438 and width 0.132283. The ratio of the
widths is 0.381814.

The preset single-point step is 1. Pair scale multiplies
every length by r_min/2^(1/6) = 0.8908987181403393, so the
step is 0.890899 and the escape amplitude is 0.222725
against 0.250000. In curvature widths the step is 19.80
for recommended, 17.64 for pair scale, and 7.56 for
Lennard-Jones. The temperature stays 0.8, because the well
depth is 1.

The displacements drawn while the shelf is the best are
about 2.6 on both arms. That is a jump from an occupied
site to a vacant site. It is the lattice arm, and it is
the arm that leaves the m55 shelf.

## The length that does not follow the pair

The lattice arm builds the Lennard-Jones lattice. Vacant
sites are then scaled onto the structure's own median
bond, so their positions follow the cluster in hand. The
Boltzmann weight is the pair energy after that shift, and
the pair in the default lattice is Lennard-Jones. Its well
is 2.62 times wider than Morse rho=14. A contact that
Morse has already dropped is still scored as bound.

The pair's own lattice reads the minimum from r_min and
the energy from that pair, so the range is the curvature
at the minimum. For Lennard-Jones that lattice matches the
historical default. recommended() keeps the default.
`recommended_for_pair` is the opt-in, selected in the
harness by H2H_PAIR_SCALE=1. Unset, the harness calls
recommended().

recommended() on this commit still hits m55 seed 2006 at
call 519192 and m80 seed 2010 at call 2973511, the same
calls as develop. The exchange it draws matches
Lattice::default() on the same seed.

`pair_rank` is that opt-in: the preset lengths stay on the
pair minimum, and the lattice scores an exchange with the
pair. No added constant. Its code_revision is
446cd701452b98d0d54dc6f8a7d8cc4cd19635c7.

## The gate

Seeds 2000-2011, serial h2h_morse, develop 61fa67cf, paper
budgets (400000 at 38, 1000000 at 55, 4000000 at 80). A
hit is a quenched energy at or below the Cambridge
reference plus 0.001. Lennard-Jones length_scale() is 1
and its lattice is the historical default, so LJ38, LJ75
and LJ98 were not searched again. Their outside ratio is 1.

The outside ratio is the median of outside_s/charged on
pair_rank divided by the same median on recommended.

| system | recommended | pair_rank | outside ratio |
| --- | --- | --- | --- |
| m38r6 | 12/12 | 12/12 | 0.824 |
| m38r14 | 4/12 | 10/12 | 0.829 |
| m55r14 | 9/12 | 2/12 | 0.694 |
| m80r6 | 12/12 | 12/12 | 0.701 |
| m80r14 | 6/12 | 6/12 | 0.659 |

m55 loses recommended hits 2000, 2001, 2003, 2006, 2007,
2008 and 2010, and gains none. Seed 2006 ends at 55D,
-220.51853338. Seed 2000 ends on -220.49847976, the shelf
pair scale already stops on. The two hits are 2004 and
2011.

m80r14 loses recommended hits 2000 and 2003. Seed 2000
ends on -340.38733557. Seed 2010 does hit, at call
1614791. The count stays 6/12 because 2002 and 2009 are
gained.

m38r6 and m80r6 stay 12/12. m38r14 keeps 2001, 2008, 2009
and 2011, and adds 2002, 2003, 2005, 2006, 2007 and 2010.

Every outside ratio is below 1.10. pair_rank still drops
recommended hits, so it is not the default. recommended
and pair_scale rows are the weak14 measurements, and their
code_revision is
61fa67cf380e46876b569da45e7cc091647a775b.
pair_rank rows name the opt-in commit above. The other
columns are unchanged. Rows are rows.csv.

## Audit on this clone

paper_audit reports 5797 checks and 15 drifts in a clone
that also holds the local ASE branches, and 0 drifts on a
checkout of 61fa67cf that does not. Five revisions listed
as unpushed are objects here, on local ASE branches, and
not ancestors of develop or origin/develop: 33bed75f,
36548663, 5a7f629d, 90b88235 and b49341e3. The audit records
those five unpushed checks as resolved. The ten provenance
rows that expect the text "not in this repository" for
those revisions no longer match. 1ce3ab64 is absent, so two
of those rows still name it. This CSV is not one of the 68
files the audit reads.
