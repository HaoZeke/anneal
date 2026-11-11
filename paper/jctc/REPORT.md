# Report, v1.31

We drew the nineteen figures with the committed generators.
The branch is paper-integration-v1.31.
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
The plate does not print an anneal hexamer energy.

## Bundle

The bundle ref is refs/heads/paper-integration-v1.31.
It also carries refs/heads/paper-integration-v1.30.
That ref is 6bdb3305aaa92ae0780b6b0c04fdfd64f2f38449.
It also carries refs/heads/paper-integration-v1.29.
That ref is df6efccd79358453fe6343c421bd7e6196017a31.
It also carries refs/heads/paper-integration-v1.28.
That name is a branch.

It is not an archive ref.

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
It also reads the eight rerun logs.
The hexamer block matches `data/hq_sci_h2o6/TALLY.txt`.
The copper block matches that file too.
The regenerated tally sha follows.

ee5cc4f9d39bfa03f4936ea551a7e537f67c03fce6d8007628d5b4a42be40029

The new copper energies differ from the archived block.
`logs/hq-sci-tally-cmp.txt` is the raw diff.
The paper uses the new energies.

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
The archived copper logs are in `data/hq_sci_cuh2/`.
The rerun logs are
`logs/cuh2_fcc_rec_0.out` through `logs/cuh2_fcc_rec_7.out`.
The copper numbers come from the slab runs.
Commit aef5cf0b2068c4988509066a881eb68682ad9290 produced them.
rustc -V reports rustc 1.88.0.
That is the Cargo.toml rust-version pin.
The library pins rgpot 25c759da82eececa8897e4634046fab27409c81a.
The library sha256 follows.

d399ef1f675e97c1f5e401a987d6a9df1bc30b3a37ba6586044b66d44be8399d

`scripts/elja_hq_sci_slab.sh` still exits before submit.
`scripts/run_cuh2_fcc_rec.sh` ran the eight seeds.
The budget is 2500.
`PROVENANCE.txt` names HaoZeke/simmAnnealTeX.
The source commit is 311d9bcc28b5401780a69898a16fff25707901ad.
The source path is rewrite_2026/data/hq_sci_cuh2/.
`scripts/check_cuh2_blobs.py` recomputes each blob id.
A read-only fetch of that repo returned repository not found.
The check has not been run against that commit.
The eight archived logs do not name an anneal commit.

## Hit rule

A copper hit is a best below -415.971529 eV plus 0.001 eV.
Line 79 of `scripts/elja_jcc_molslab_ensemble.sh` sets that target.
Line 118 of `scripts/elja_hq_sci_molslab_bank.sh` sets it too.
Commit a8f1092fc8d6937b34e34a18bc50394a559ca9f3 names that energy.
It is the start energy of the slab con file.
The tolerance is `TARGET_TOLERANCE` and `TARGET_TOL` in those scripts.
The tolerance is 0.001 eV.
The script is scripts/elja_hq_sci_molslab_bank.sh.
The lines are 118 and 119.
TARGET_TOL there is 1e-3.
The per-seed gaps run from 8e-6 eV to 3.68e-4 eV above the reference.
No seed reached the reference within 1e-6 eV.
These are not exact hits.
Eight seeds lie inside the 0.001 eV tolerance.
The set_relative_0.01 line records the 0.01 eV rate.
That line is a separate figure.
The hexamer block still uses the 0.01 eV test.

## Rerun energies

The rerun best energies run from -415.971521 eV to -415.971161 eV.
The library pins rgpot 25c759da82eececa8897e4634046fab27409c81a.
The library sha256 follows.

d399ef1f675e97c1f5e401a987d6a9df1bc30b3a37ba6586044b66d44be8399d

## Receipt

`logs/cuh2-hardware-receipt.txt` records the start and the end.
Each side has boot_id, uptime, cpuinfo MHz and lscpu.
The version lines are the raw output of `rustc -V` and `cargo -V`.
They also include `gfortran -v`.
rustc -V reports rustc 1.88.0.
cargo -V reports cargo 1.88.0.
The receipt commit's parent is T.

T is aef5cf0b2068c4988509066a881eb68682ad9290.

## Closed items

The v1.28 we-rate edits stay in range.
The figure hash lines have a writer.
The raw-record class names each path and its writer.
The six-path diff is the text in REVIEW-NOTES.md.
Each affiliation line cites the author decision.
The box prose items from that pass are closed.
This pass did not replay.
The replay count stays seven.

## Gaps

The per-seed gaps to -415.971529 eV follow.
Seed 0 is 0.000040 eV.
Seed 1 is 0.000087 eV.
Seed 2 is 0.000039 eV.
Seed 3 is 0.000008 eV.
Seed 4 is 0.000060 eV.
Seed 5 is 0.000062 eV.
Seed 6 is 0.000065 eV.
Seed 7 is 0.000368 eV.
The largest gap is 0.000368 eV.
The reference count is 8/8.

## Host

The LJ75 writer records the output of hostname.
That receipt names the host bench.
The copper writer records the output of hostname.

## rohit-paper

The profile sha follows.

654d0de1c2632feb93ddfc2f72a08372ad672a8c7cb4a6ad30cca69b2783ddc6

The profile file is not in this repo.
logs/rohit-paper.txt is the raw run.
rgoswami.AbbreviationStyle hits the affiliation abbreviation.
Rohit set that affiliation, so the hits stay.
rgoswami.TricolonAbuse names cited authors.
Those names stay.

## Scope

Rohit ruled that coverage and mutants are out of scope.

This seal follows that ruling.
