# Report, v1.29

We drew the eighteen figures with the committed generators.
The branch is paper-integration-v1.29.
It starts from fd4e691db9152bea8086c2102af84a1b0ea39444.
The inbox sums file has 359 rows.
Its sha256 follows.

b236bd7dd327aed1db3672126c577bf342db0f87b5d6668298cf82ddf44f078a

Every row checked clean.

## Rohit decisions pending

The cover letter still has the open items.

- cover-letter.tex:11 is the telephone placeholder.
- cover-letter.tex:55 is three placeholders.
- manuscript.tex:586 says the disclosure option is not submitted.
- manuscript.tex:591 is the option line for the tool and the version.
- manuscript.tex:614 is the identifier placeholder.
- manuscript.tex:556 adds "In contrast," before the second count.
- The LJ75 figure now follows the systems figure.
- That later place follows the order of first citation.

## Prose

The manuscript patch is applied exactly.
The data-availability section matches v1.16.
Paragraph cohesion at patched lines 565 and 607 stays open.
PROSE-RECEIPT.md lists each patch change and those two hits.

## Captions

The suppinfo sentence still names the plates.
It names no arm and no offset, so it stays.

The previous LJ38 sketch-map labels an arm "per-basin bias (409)".
The sealed lj38.csv has no such arm.
The new caption says colour is the quenched energy.
The 600 landmarks pool three arms and two references.
The darkest point is the fcc minimum.

The previous occupancy plate puts the recommended best at +9.669.
The sealed LJ38 table puts that arm on the fcc minimum.
Basin hopping stops 2.429 above it.
The plate does not mark those arms.

The previous LJ98 plate puts the recommended best at +6.622.
It puts the basin-hopping best at +6.119.
The sealed table puts the recommended hop 0.022 above the Leary minimum.
Basin hopping stops 0.119 above it.
The plate does not mark those arms.

## Water

The plate prints -197.7805 kJ/mol.
That is the Cambridge Cluster Database value.
It uses the Jorgensen constants C6 = 2552.24 and C12 = 2510400.
anneal uses epsilon 0.6485 kJ/mol and sigma 3.15365 Å.
Its C6 is 2551.8252 and its C12 is 2510336.2.
Those constants give the committed -197.7750 kJ/mol reported here.

## Bundle

The bundle ref is refs/heads/paper-integration-v1.29.
It also carries refs/archive/paper-integration-v1.28.
That ref is abb65a5b47be2de02dad87a082e539525dba648b.
It also carries refs/archive/paper-integration-v1.27.
That ref is 77a9e399660f2a54f2a5fd17e0da184f0c7312e0.
It also carries refs/archive/paper-integration-v1.26.
That ref is 975f6028338172d2edb6810e216ee2f0caf451f4.
It also carries refs/archive/paper-integration-v1.25.
That ref is 9109df21fbfd9470c16cff37a7a7b5228bb809e0.
It also carries refs/archive/paper-integration-v1.24.
That ref is 705fa8a4b4b259023db551a4f20da5efbea361f6.
It also carries refs/archive/paper-integration-v1.20.
That ref is 7d20cca206fb50cce9f2e393078eaa1e03e9c86f.
It also carries refs/archive/paper-integration-v1.19.
That ref is 27279cbb08176d7c045229d5d8ad96ed4fafd399.
It also carries refs/archive/paper-integration-v1.18.
That ref is 31e4535d5416223c1cbf0342348170a5ea002590.
It also carries refs/archive/paper-integration-v1.15.
That ref is 12de8866efaaf801853336302f972c5e0faa7966.

## Generators

`scripts/hq_sci_tally.py` reads the eight h2o6 rec logs.
The hexamer block matches `data/hq_sci_h2o6/TALLY.txt`.
The copper block does not.
No cuh2 rec log is in this tree.
The regenerated tally sha follows.

ca33451735e77296e9a718c2ecbdf03c40af5781e0603a3e4475d75c462bac70

`scripts/rebuild_figures.py` redraws each file in `figs/`.
It pins `SOURCE_DATE_EPOCH` and `PYTHONHASHSEED` to 0.
`figstyle.save` drops CreationDate, ModDate and Creator.
Two clean rebuilds wrote the same bytes.
Three files rebuilt.
`algorithm_stack.pdf`, `rec_arms.pdf` and `toc-graphic.pdf`.
The other generators exit on the committed provenance.
`fig_lj75_ecdf.py` finds no full published commit.
`fig_water_crest.py` reads the rule with a trailing period.
`fig_landfold_maps.py` rejects the lj38 derived run.
`fig_systems.py` misses a digest.
`fig_si_structures.py` misses a digest.
`fig_si_water.py` misses a digest.
`fig_landscapes.py` misses a digest.
`fig_disconnectivity.py` misses a digest.
`fig_si_panels.py` misses a digest.
No figure lacks a committed generator.
The old hash record is `logs/archive/figure-provenance-v128.txt`.
Ten of its fifteen lines were stale.
The new record is `logs/figure-provenance.txt`.
The copper rec logs are not in that record.
They are not in this tree, the branches, or the kits.
`TALLY.txt` still has the copper block.
`scripts/elja_hq_sci_slab.sh` is the committed driver.
It exits before submit.
A rerun needs `examples/slab_adsorption`.
It also needs `engines/librgpot_cuh2.so`.
It also needs `examples/fixtures/cuh2_fcc_slab.con`.
The budget is 2500 and the seeds are 0 through 7.
The stdout names are `cuh2_fcc_rec_0.out` through `cuh2_fcc_rec_7.out`.
HyperQueue has to accept a work directory off the home filer.
