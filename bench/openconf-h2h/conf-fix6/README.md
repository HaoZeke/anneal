Copyright (c) anneal developers

Why anneal misses the lowest conformer of z34, amprenavir and lopinavir,
on development seeds 3000-3059 (not the evaluation seeds 2000-2023), and
the one general fix tried. The fix does not win, so nothing here changes
anneal: this directory is the diagnosis, its rows and the gate.

- develop: anneal 8f91a35e4076963c866082f60b06e669f64f7a3b, the
  recommended molecular preset.
- MMFF94 through RDKit 2025.09.6 at a dielectric of one, the start
  embedding of `h2h_job.py`, and one budget per molecule within 2% of
  the mean paired budget of the evaluation sessions (`BUDGET` in the
  script).

## Reference and rows

`reference/<molecule>.xyz` is the lowest valid polished conformer of
every session on disk (energy in kcal/mol on the comment line), so a dev
run is scored against the global lowest rather than a session's three
arms. `dev-rows.csv` has one row per (molecule, arm, seed), written by

    scripts/openconf_h2h/dev_diagnose.py run --molecules z34 --seeds 3000-3059 --out DIR
    scripts/openconf_h2h/dev_diagnose.py analyse --runs DIR --molecules z34 --rows dev-rows.csv --arm develop

`run` searches as `h2h_job.py` does and keeps every minimum the harvest
records (a geometry whose largest gradient component is under 0.05 eV/Å,
whatever the search does with it afterwards) and every evaluated geometry
within 4 kcal/mol of the reference. `analyse` polishes the minima within
3 kcal/mol as `h2h_score.py` does. The columns are described in the
script. A hit is a valid polished minimum within 0.1 kcal/mol of the
reference. The dev runs ran seven at a time, so their times are not
compared; the gate's are.

## Where the misses go

Develop's misses on the 60 dev seeds:

| molecule | hits | misses | invalid at the reference energy | misses never within 1 Å | misses with a minimum within 0.5 Å | median nearest minimum (Å) | median gap (kcal/mol) | misses with no minimum within 3 kcal/mol |
|---|---|---|---|---|---|---|---|---|
| rq3 | 60 | 0 | 0 | | | | | |
| 915 | 45 | 15 | 0 | 0 | 6 | 0.51 | 1.46 | 1 |
| xld | 51 | 9 | 0 | 3 | 0 | 0.99 | 0.61 | 1 |
| fluvastatin | 24 | 36 | 0 | 1 | 7 | 0.71 | 0.22 | 0 |
| z34 | 3 | 57 | 0 | 56 | 0 | 2.41 | 0.98 | 3 |
| amprenavir | 5 | 55 | 0 | 41 | 3 | 1.70 | 0.47 | 16 |
| lopinavir | 0 | 60 | 0 | 59 | 0 | 2.97 | 2.11 | 55 |
| lorlatinib | 34 | 26 | 0 | 15 | 8 | 0.84 | 2.14 | 7 |

"Never within 1 Å" means that neither a recorded minimum within
8 kcal/mol nor a sampled low evaluation comes within 1 Å heavy-atom RMSD
of the reference. No run polishes a minimum at the reference energy to
an invalid conformer, so the stereo hold loses nothing. The harvest
records minima before the search deduplicates them, so dedup cannot hide
one either. The weak molecules' lowest conformer is **never visited**:
56 of z34's 57 misses, 59 of lopinavir's 60 and 41 of amprenavir's 55
never come near it. Amprenavir's other 14 record a minimum 0.35 to
1.0 Å from it (three within 0.5 Å) that polishes 0.15 kcal/mol or more
up, into a neighbouring basin: these runs reach the region but not the
basin.

How each miss's lowest valid conformer differs from the reference, atoms
mapped by the best automorphism (means over the misses with a minimum
within 3 kcal/mol):

| molecule | rotatable torsions over 45° apart | amide torsions | rings with a pucker over 30° apart | reference H-bonds | lost | new |
|---|---|---|---|---|---|---|
| 915 | 0.9 | 0 | 1.00 | 2 | 0.0 | 0.0 |
| xld | 2.0 | 0 | 0.50 | 0 | 0.0 | 0.0 |
| fluvastatin | 3.9 | 0 | 0.00 | 1 | 1.0 | 1.9 |
| z34 | 7.4 | 0 | 0.44 | 3 | 2.9 | 2.4 |
| amprenavir | 3.7 | 0 | 0.15 | 1 | 0.3 | 0.3 |
| lopinavir | 4.2 | 0 | 0.00 | 3 | 1.6 | 1.6 |
| lorlatinib | 0.0 | 0 | 0.58 | 0 | 0.0 | 0.0 |

No miss flips an amide. z34's misses are a different fold: seven
torsions apart, and almost none of the reference's three intramolecular
hydrogen bonds, replaced by about two others. Lopinavir's are four
torsions apart and keep about half of its three. Amprenavir's keep its
one hydrogen bond and differ in about four torsions. Fluvastatin's form
two hydrogen bonds its reference lacks, and lorlatinib's and 915's
differ only in a ring pucker. For z34 and lopinavir, the missing
hydrogen-bond network spans several torsions. A move that turns one to
three random bonds by random angles almost never closes one of these
bonds, and a quench from a partial turn does not finish it.

## The fix tried: hydrogen-bond torsion draws

Half of the torsion draws that turn no ring crankshaft instead close or
open one hydrogen bond between a polar hydrogen and a nitrogen or oxygen
at least two rotatable bonds from it. Only the bonds between the two
atoms turn: either by descent on the H...A distance (target 2.0 Å) with
the clash penalty, or by random turns that take a bonded pair past
2.5 Å. Clash repair and the stereo check are those of the torsion move.
Nothing in the move depends on the molecule. Three placements, each with
the same 60 dev seeds:

| molecule | develop | separate kernel | in the torsion kernel, close only | in the torsion kernel, close or open |
|---|---|---|---|---|
| rq3 | 60 | 60 | 60 | 60 |
| 915 | 45 | 38 | 40 | 33 |
| xld | 51 | 40 | 46 | 46 |
| fluvastatin | 24 | 8 | 4 | 12 |
| z34 | 3 | 12 | 8 | 8 |
| amprenavir | 5 | 8 | 4 | 3 |
| lopinavir | 0 | 4 | 3 | 2 |
| lorlatinib | 34 | 5 | 32 | 32 |
| total | 222 | 175 | 197 | 196 |

The draws do reach the hydrogen-bonded folds: z34 rises from 3 to 8-12,
and lopinavir from 0 to 2-4. As a separate kernel, the allocator moves
draws off the crankshaft and lorlatinib falls from 34 to 5. Folded into
the torsion kernel, lorlatinib holds, but fluvastatin, whose lowest
conformer has fewer hydrogen bonds than its near misses, falls from 24
to 4-12. 915 loses 5 to 12, xld 5. With close or open, 915 drops from 45
to 33 (Fisher p 0.035) and fluvastatin from 24 to 12 (p 0.028).

## Gate, seeds 2000-2023

The close-or-open variant was enabled in `recommended_molecular` and run
as the `anneal` arm against same-session develop (8f91a35e) and openconf
e32c5fa7b6e6aac944bda31c851be42e14340235. One serial session under
`sudo unshare --uts`, hostname bench, every job on core 0,
`scripts/openconf_h2h/enter_bench.sh paired` with
`OPENCONF_H2H_BASELINE=8f91a35e` and then `paired-score` (`gate/`).
Hits are scored against the lowest valid conformer of the session's
three arms, as in `../conf-fix`.

| molecule | hits fix | hits develop | hits openconf | Fisher p vs develop | Fisher p vs openconf | n_3kcal fix | n_3kcal develop | n_3kcal openconf | ΔE_min fix | ΔE_min develop | ΔE_min openconf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 24/24 | 24/24 | 20/24 | 1.0000 | 0.1092 | 43.5 | 45 | 22.5 | 0.00 | 0.00 | 0.00 |
| 915 | 15/24 | 15/24 | 10/24 | 1.0000 | 0.2476 | 3 | 4 | 2 | 0.00 | 0.00 | 0.01 |
| xld | 20/24 | 16/24 | 2/24 | 0.3177 | 0.0000 | 34 | 33 | 4 | 0.00 | 0.00 | 2.74 |
| fluvastatin | 11/24 | 10/24 | 8/24 | 1.0000 | 0.5556 | 36 | 39.5 | 7 | 0.05 | 0.05 | 0.48 |
| z34 | 3/24 | 0/24 | 0/24 | 0.2340 | 0.2340 | 16.5 | 24 | 1.5 | 0.53 | 0.53 | 2.38 |
| amprenavir | 3/24 | 3/24 | 0/24 | 1.0000 | 0.2340 | 2.5 | 3 | 0 | 1.48 | 2.08 | 4.19 |
| lopinavir | 0/24 | 2/24 | 0/24 | 0.4894 | 1.0000 | 0 | 0 | 0 | 5.78 | 5.89 | 9.29 |
| lorlatinib | 12/24 | 12/24 | 5/24 | 1.0000 | 0.0687 | 2.5 | 3 | 2 | 0.60 | 0.60 | 2.68 |

The fix hits 88 of 192 runs, develop 82 and openconf 59, with no
significant change on any molecule and lorlatinib at 12 against 12. No
arm has an invalid candidate or an error. The gate alone would pass it,
but the six hits come from z34, xld and lopinavir moving in opposite
directions, none significant. On 2.5 times as many dev seeds, the same
variant loses 26 hits, significantly on 915 and fluvastatin. It stays
off and is not delivered.

| molecule | calls to hit fix | calls to hit develop | calls to hit openconf | Mann-Whitney p vs develop | Mann-Whitney p vs openconf |
|---|---|---|---|---|---|
| rq3 | 7242 | 12613 | 4165 | 0.2882 | 0.0415 |
| 915 | 16248 | 18883 | 4339 | 0.9669 | 0.0025 |
| xld | 23651 | 36271.5 | 10363 | 0.2721 | 0.3636 |
| fluvastatin | 28693 | 13292 | 3523.5 | 0.0845 | 0.0003 |
| z34 | 36178 |  |  |  |  |
| amprenavir | 42974 | 56050 |  | 0.1000 |  |
| lopinavir |  | 48892.5 |  |  |  |
| lorlatinib | 39760 | 55483.5 | 7971 | 0.0690 | 0.0268 |

Median wall time per run, and each anneal arm's median time inside and
outside the potential:

| molecule | wall fix (s) | wall develop (s) | wall openconf (s) | fix inside (s) | fix outside (s) | develop inside (s) | develop outside (s) |
|---|---|---|---|---|---|---|---|
| rq3 | 4.14 | 3.88 | 3.85 | 2.25 | 1.89 | 2.26 | 1.62 |
| 915 | 1.64 | 1.62 | 1.97 | 0.92 | 0.71 | 0.93 | 0.70 |
| xld | 4.33 | 4.33 | 5.01 | 2.56 | 1.79 | 2.59 | 1.75 |
| fluvastatin | 4.08 | 4.03 | 5.00 | 2.42 | 1.66 | 2.42 | 1.60 |
| z34 | 5.19 | 5.08 | 6.08 | 3.08 | 2.11 | 3.06 | 2.01 |
| amprenavir | 5.33 | 5.16 | 7.74 | 3.17 | 2.15 | 3.13 | 2.03 |
| lopinavir | 10.07 | 10.02 | 21.35 | 6.11 | 3.96 | 6.17 | 3.86 |
| lorlatinib | 3.55 | 3.55 | 7.23 | 2.04 | 1.49 | 2.06 | 1.49 |

The fix adds 0.00 to 0.27 s outside the potential per median run, most
on rq3, and changes the time inside it by at most 0.06 s.

## What this leaves

The weak molecules need a hydrogen-bond network that spans several
torsions, and fluvastatin, 915 and lorlatinib are hurt by any draw spent
building one. A fix has to decide per molecule from what the search
measures, for example by giving the hydrogen-bond draws their own arm in
the depth allocator while keeping the crankshaft share fixed, rather
than a fixed half of the torsion draws.
