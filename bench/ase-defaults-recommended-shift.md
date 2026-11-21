# Recommended-arm shift

The published recommended rows and the rerun at
`5eb79cdcb0a986984dc7b487e5a3cc93224246f9` use the same seeds
(2000-2011), the same paper budgets, and the same hit rule (best
energy at or below the reference plus 0.001). The budgets are LJ38
400000, LJ55 1000000, LJ75 and LJ98 and LJ104 and LJ150 4000000, and
Au55 1000000. `Config::recommended` is still Lennard-Jones
`recommended_with_scales` at the reduced length and energy. Nothing
in that preset's remaining switches was retuned.

The revision named in the published rows,
`b49341e308e6e98e39a2ea7bcc7fbf743f6c45ac`, is not in this clone. The
parent of the row commit `70c0ad6c` is
`1c40f464c82f12916929b9e2e6ff005ef6c2e1aa`. A release `ase_panel` at
that tree, and at its descendant
`628773d6837ea2d959a27256ff4163b21162fd4d` (the parent of the template
removal), reproduces the published LJ38 recommended first-hit calls
on every seed. The next commit,
`1c1b12384683485cf820d3a244151b48973dc0f7` (`fix: remove structure
templates from cluster search`), reproduces the
`5eb79cdc` rerun on every seed.

LJ38 recommended, n=12, budget 400000, no wall cap. First-hit calls:

| seed | paper, `1c40f464`, and `628773d6` | `1c1b1238` and the `5eb79cdc` rerun |
| --- | ---: | ---: |
| 2000 | 16377 | 124155 |
| 2001 | 11698 | 111509 |
| 2002 | 14433 | 293837 |
| 2003 | 11833 | 174442 |
| 2004 | 6890 | 131524 |
| 2005 | 16305 | 69728 |
| 2006 | 10065 | 99994 |
| 2007 | 9371 | 110660 |
| 2008 | 4700 | 325364 |
| 2009 | 11166 | 317864 |
| 2010 | 7454 | 231706 |
| 2011 | 9412 | 49500 |

Both sides hit 12/12 at best energy -173.92842659. The type-7 median
is 10615.5 on the published side and 127839.5 after the removal.
That is a trajectory change. The tolerance and the budget are the
same, and every seed still stops on the same minimum.

What the commit removes is the structure-template path. At
`1c40f464`, `recommended` sets a continuous-symmetry measure with a
five-hop reseed and leaves `prior_reseed` and `prior_growth` at
their default, on. The walk reseeds an icosahedral plateau, and an
escape leave grows a non-icosahedral one. `for_cluster` does not
set `escape_on_stall` and does not set a reseed hop count, so that
path does not run on the anneal arm. The anneal arm's first-hit
calls matched the published rows seed for seed; this split matches
that.

`5499be31` later deletes the `prior_*` keys and says they had been
inert since `1c1b1238`. Commits after `1c1b1238` add
`branch_on_stall`, `retry_noop_stall_leave`,
`align_directed_stall_source`, and `plateau_heat`. Each defaults
off, and `recommended` does not set them. `recommended_for_pair`
leaves Lennard-Jones on `recommended`. The twelve LJ38 seeds are
identical at `1c1b1238` and at `5eb79cdc`, so those later edits did
not move this cell.

LJ98 (10 hits to 7), LJ104 (9 to 5), and LJ150 (5 to 8) were not
re-run at the boundary. They use this same `recommended` preset and
the same deleted template path, and their budgets did not change.
The measured cell pins the shift on `1c1b1238`.

Verdict: code change, commit `1c1b12384683485cf820d3a244151b48973dc0f7`.
Not a settings change and not a budget change.
