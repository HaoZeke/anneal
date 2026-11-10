# Prose receipt

We left the two cohesion hits for Papers.
The manuscript is the v1.16 file plus one patch.
The patch file is manuscript-v116-to-patch.diff.
No manuscript sentence outside that patch was rewritten.
The data-availability section is the v1.16 text.

## Patch

Hit: az-spine at base line 67.
The line break now falls after "made that".
It no longer falls before "method for atomic clusters".
No word changes.

Hit: subject-verb-distance at base line 282.
The LJ75 figure now follows the systems figure.
It no longer precedes the information-theory section.
No word of the caption changes.

Hit: echo-skeleton at base line 493.
Old: Seed 1 reaches both 1e-7 values above that line.
New: Seed 1 exceeds that line, at both values.
Both numbers, both ledgers and Eh stay.
The threshold stays the line named just before.

Hit: subject-verb-distance at base line 545.
The Newcombe and McNemar cite moves to before the period.
A comment mark keeps the cite keys on two lines.

Hit: subject-verb-distance at base line 551.
Old: three program names precede "are not compared here".
New: "This comparison leaves out" those three programs.
The names stay.

Hit: az-spine at base line 555.
Old: the sentence names 97 of 192 seeds.
New: that sentence opens with "In contrast," before the same count.

## Left for Papers

paragraph-cohesion at patched line 565.
The sentence is "Author contributions follow the CRediT taxonomy."
paragraph-cohesion at patched line 607.
The sentence says the source is the repository under the licence.
An opener from the paragraph above would add no fact.
Both hits stay.

## Fixed text

Each record in the block is a file, a line and the exact string.
The header is the copyright line the ruling keeps.
The 2023 notice is the restored quotation.
The copyright strings in the block were not rewritten.
The affiliation lines are the author decision below.

```

paper/jctc/CONSTANTS.tsv:1
Copyright (c) anneal developers

paper/jctc/NUMBERS_TODO.md:1
Copyright (c) anneal developers

paper/jctc/cover-letter.tex:1
Copyright (c) anneal developers

paper/jctc/cover-letter.tex:8
SURF B.V., Amsterdam

paper/jctc/macros.tex:1
Copyright (c) anneal developers

paper/jctc/manuscript.tex:1
Copyright (c) anneal developers

paper/jctc/manuscript.tex:14
SURF B.V., Amsterdam

paper/jctc/refs.bib:1
Copyright (c) anneal developers

paper/jctc/scripts/cargo_build_receipt.sh:2
Copyright (c) anneal developers

paper/jctc/scripts/check_message_newlines.sh:2
Copyright (c) anneal developers

paper/jctc/scripts/emit_sealed_stats.py:2
Copyright (c) anneal developers

paper/jctc/scripts/extract_water_logs.py:2
Copyright (c) anneal developers

paper/jctc/scripts/fig_algorithm_stack.py:2
Copyright (c) anneal developers

paper/jctc/scripts/fig_disconnectivity.py:2
Copyright (c) anneal developers

paper/jctc/scripts/fig_landfold_maps.py:2
Copyright (c) anneal developers

paper/jctc/scripts/fig_landscapes.py:2
Copyright (c) anneal developers

paper/jctc/scripts/fig_lj75_ecdf.py:2
Copyright (c) anneal developers

paper/jctc/scripts/fig_rec_arms.py:2
Copyright (c) anneal developers

paper/jctc/scripts/fig_si_panels.py:2
Copyright (c) anneal developers

paper/jctc/scripts/fig_si_structures.py:2
Copyright (c) anneal developers

paper/jctc/scripts/fig_si_water.py:2
Copyright (c) anneal developers

paper/jctc/scripts/fig_summary_rates.py:2
Copyright (c) anneal developers

paper/jctc/scripts/fig_systems.py:2
Copyright (c) anneal developers

paper/jctc/scripts/fig_water_crest.py:2
Copyright (c) anneal developers

paper/jctc/scripts/figstyle.py:1
Copyright (c) anneal developers

paper/jctc/scripts/hexamer_numbers.py:2
Copyright (c) anneal developers

paper/jctc/scripts/lj75_gmin_table.py:2
Copyright (c) anneal developers

paper/jctc/scripts/lj75_hardware_capture.sh:2
Copyright (c) anneal developers

paper/jctc/scripts/lj75_rebuild_from_archive.sh:2
Copyright (c) anneal developers

paper/jctc/scripts/lj75_source_verify.py:2
Copyright (c) anneal developers

paper/jctc/scripts/lj_csm_sizes_table.py:2
Copyright (c) anneal developers

paper/jctc/scripts/lockfile_delta.py:2
Copyright (c) anneal developers

paper/jctc/scripts/pack_lj75_source.sh:2
Copyright (c) anneal developers

paper/jctc/scripts/redact_paths.py:2
Copyright (c) anneal developers

paper/jctc/scripts/render_all.py:2
Copyright (c) anneal developers

paper/jctc/scripts/results_jcc.py:2
Copyright (c) anneal developers

paper/jctc/scripts/seal_checks.sh:2
Copyright (c) anneal developers

paper/jctc/scripts/sketchmap_landscapes.py:2
Copyright (c) anneal developers

paper/jctc/scripts/system_visuals.py:2
Copyright (c) anneal developers

paper/jctc/scripts/tip4p_constants.py:1
Copyright (c) anneal developers

paper/jctc/scripts/toc_graphic.py:2
Copyright (c) anneal developers

paper/jctc/scripts/watch_crest_steps.py:2
Copyright (c) anneal developers

paper/jctc/scripts/water_census.py:2
Copyright (c) anneal developers

paper/jctc/scripts/water_crest_figure.py:2
Copyright (c) anneal developers

paper/jctc/scripts/water_h2h.py:2
Copyright (c) anneal developers

paper/jctc/scripts/water_host_relabel.py:2
Copyright (c) anneal developers

paper/jctc/scripts/water_results_table.py:2
Copyright (c) anneal developers

paper/jctc/si.tex:1
Copyright (c) anneal developers

paper/jctc/si.tex:11
SURF B.V., Amsterdam

paper/jctc/si.tex:121
Copyright (c) 2023 anneal core developers

paper/jctc/toc-graphic.tex:4
Copyright (c) anneal developers

```

## Author decision

Each of the three lines reads `SURF B.V., Amsterdam`.
Rohit decided that text in the Anneal Paper chat.
The time was 15:51 on 2026-11-10.
Each line is cited below.

```
manuscript.tex:14.

Rohit decided it at 15:51 CST on 2026-11-10 in the Anneal Paper chat.

si.tex:11.

Rohit decided it at 15:51 CST on 2026-11-10 in the Anneal Paper chat.

cover-letter.tex:8.

Rohit decided it at 15:51 CST on 2026-11-10 in the Anneal Paper chat.
```

A checker may still flag `B.V.`.
That flag stays, because the wording is the author's.
No checker config is changed.

## Non-prose

The class name is non-prose: raw run record.
These lines stay as the run wrote them.
No checker is changed.

`paper/jctc/data/hq_sci_h2o6/h2o6_rec_0.txt`
`paper/jctc/data/hq_sci_h2o6/h2o6_rec_1.txt`
`paper/jctc/data/hq_sci_h2o6/h2o6_rec_2.txt`
`paper/jctc/data/hq_sci_h2o6/h2o6_rec_3.txt`
`paper/jctc/data/hq_sci_h2o6/h2o6_rec_4.txt`
`paper/jctc/data/hq_sci_h2o6/h2o6_rec_5.txt`
`paper/jctc/data/hq_sci_h2o6/h2o6_rec_6.txt`
`paper/jctc/data/hq_sci_h2o6/h2o6_rec_7.txt`
`paper/jctc/data/hq_sci_h2o6/h2o6_rec_7_rerun.txt`

The writer is `scripts/elja_hq_sci_mol.sh`.

That script submits the hexamer and stores each seed log.

`paper/jctc/data/hq_sci_h2o6/TALLY.txt`

The writer is `scripts/hq_sci_tally.py`.

The hexamer block matches the file in the tree.
The copper hit count uses the fixed target.
The set_relative_0.01 line records the 0.01 eV rate.

`paper/jctc/data/hq_sci_cuh2/cuh2_fcc_rec_0.txt`

`paper/jctc/data/hq_sci_cuh2/cuh2_fcc_rec_1.txt`

`paper/jctc/data/hq_sci_cuh2/cuh2_fcc_rec_2.txt`

`paper/jctc/data/hq_sci_cuh2/cuh2_fcc_rec_3.txt`

`paper/jctc/data/hq_sci_cuh2/cuh2_fcc_rec_4.txt`

`paper/jctc/data/hq_sci_cuh2/cuh2_fcc_rec_5.txt`

`paper/jctc/data/hq_sci_cuh2/cuh2_fcc_rec_6.txt`

`paper/jctc/data/hq_sci_cuh2/cuh2_fcc_rec_7.txt`

The writer is `scripts/elja_hq_sci_slab.sh`.

`paper/jctc/logs/cuh2_fcc_rec_0.out`

`paper/jctc/logs/cuh2_fcc_rec_1.out`

`paper/jctc/logs/cuh2_fcc_rec_2.out`

`paper/jctc/logs/cuh2_fcc_rec_3.out`

`paper/jctc/logs/cuh2_fcc_rec_4.out`

`paper/jctc/logs/cuh2_fcc_rec_5.out`

`paper/jctc/logs/cuh2_fcc_rec_6.out`

`paper/jctc/logs/cuh2_fcc_rec_7.out`

`paper/jctc/logs/cuh2-hardware-receipt.txt`

The writer is `scripts/run_cuh2_fcc_rec.sh`.

The figure archive is a different raw history.
logs/archive/figure-provenance-v128.txt is that figure archive.
No committed script wrote it.
Commit b3546986 added it by hand.
Ten of its fifteen hash lines were stale.

`paper/jctc/logs/figure-provenance.txt`

`scripts/rebuild_figures.py` redraws the files.
`scripts/figure_provenance.py` writes these hash lines.
All nineteen figures rebuilt.
Two clean runs matched.

The digest lines of the two provenance records are the same class.

`paper/jctc/data/lj75_gmin/provenance.txt`

`scripts/fig_lj75_ecdf.py` reads that record.

`paper/jctc/data/system_visuals/provenance.txt`

`scripts/fig_systems.py` reads that record.
`scripts/fig_si_structures.py` reads the same file.
`scripts/fig_si_water.py` uses those digests.
`scripts/fig_landscapes.py` checks those digests.
`scripts/fig_disconnectivity.py` loads those digests.
`scripts/fig_si_panels.py` draws from those digests.
