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
It also reads the eight committed cuh2 logs.
The hexamer block matches `data/hq_sci_h2o6/TALLY.txt`.
The copper block matches that file too.
The regenerated tally sha follows.

d1baa5017bcf5047d60656faa7e5405cdac93b7d90e3cb527a853d3f1d57884f

The diff command printed no lines.
`logs/hq-sci-tally-cmp.txt` is that empty diff.

`scripts/rebuild_figures.py` redraws each file in `figs/`.
It pins `SOURCE_DATE_EPOCH` and `PYTHONHASHSEED` to 0.
`figstyle.save` drops CreationDate, ModDate and Creator.
Two clean rebuilds wrote the same bytes.
All nineteen figures rebuilt.
No figure lacks a committed generator.
The old hash record is `logs/archive/figure-provenance-v128.txt`.
Ten of its fifteen lines were stale.
The new record is `logs/figure-provenance.txt`.
The copper rec logs are not in that record.
They are in `data/hq_sci_cuh2/`.
`TALLY.txt` still has the copper block.
`scripts/elja_hq_sci_slab.sh` is the committed driver.
It exits before submit.
A rerun needs `examples/slab_adsorption`.
It also needs `engines/librgpot_cuh2.so`.
It also needs `examples/fixtures/cuh2_fcc_slab.con`.
The budget is 2500 and the seeds are 0 through 7.
The stdout names are `cuh2_fcc_rec_0.out` through `cuh2_fcc_rec_7.out`.
HyperQueue has to accept a work directory off the home filer.
This machine has no gfortran.
The rgpot checkout is not here.
engines/librgpot_cuh2.so is not in the tree.
CppCore/rgpot/fortran/rgpot_cuh2_capi.f90 is not committed here.
`PROVENANCE.txt` names HaoZeke/simmAnnealTeX.
The source commit is 311d9bcc28b5401780a69898a16fff25707901ad.
The source path is rewrite_2026/data/hq_sci_cuh2/.
`scripts/check_cuh2_blobs.py` recomputes each blob id.
A read-only fetch of that repo returned repository not found.
The check has not been run against that commit.
The eight logs do not name an anneal commit.
No run record from that commit is in this tree.
The producing engine is not identified.
