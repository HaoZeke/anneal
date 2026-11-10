Two-million evaluation rerun, on the same engine.

Forecast commit 6f74d439 uses central clock 02:55:01 on 2026-11-10.

The arm opened at 02:55:36 the same morning.

The forecast repeated there is the one from commit 88d1cca.

Head 88d1cca.

Tree d16cabd3.

Binary hash
53e08802ead942e22f97d73fc1c015f85b77a57298e43ddd2511ce9e3b0efbeb.

The receipt wrote that digest before the first seed.

Each seed ran alone on processor 0, and the receipt field pins reads 0.

uname -n reads bench.

REPRODUCE.md records the namespace.

At 100000 evaluations the global-minimum tally was 0 of 100.

Scaling 100000 up to 2000000 multiplies by 20, so the tally stays 0.

One icosahedral, Cs result showed up in those 100 short runs.

Multiplied by 20 that rate is 0.2 per seed, and six seeds give 1.2.

Forecast global-minimum total on seeds 2000 through 2005: 0.

Forecast icosahedral, Cs total: 1 of 6.

Global-minimum line: -582.086641 or lower.

Icosahedral, Cs acceptance uses cutoff -582.038428.

Global-minimum outcomes on the six seeds: none.

No seed finished in the icosahedral, Cs basin.

The icosahedral, Cs forecast missed, while the global-minimum forecast
held.

Temperature and step stay as they were.

```
seed start end energy gm_dist cs_dist
2000 02:55:36 02:58:29 -580.42126629 0.520497 0.515938
2001 02:58:29 03:01:12 -580.42126629 0.520497 0.515938
2002 03:01:12 03:04:06 -578.60157826 0.520577 0.513892
2003 03:04:06 03:06:51 -580.42126629 0.520497 0.515938
2004 03:06:51 03:09:29 -578.60157826 0.526995 0.511773
2005 03:09:29 03:12:08 -581.80180238 0.500292 0.523080
```
Dates in that block are 2026-11-10.

Energies and distances match the earlier six-seed table.

No curve crossed either line.

Seed 2017, quenched here, ends at -582.0384290623 after 98 evaluations.

Gradient infinity norm 4.263e-6.

Match to icosahedral, Cs coordinates: 0.000000.

Match to global-minimum coordinates: 0.523902.

The point is the icosahedral, Cs minimum, far from the global minimum.

Reference logs list energy.

They also list charged evaluations.

Coordinates are absent.

Nothing remains for a re-quench.

Seeds 2003, 2004, and 2005 print best= and hit=1.

The gaps are 6.4e-5, 2.4e-5, and 6.2e-5.

The energies sit above -582.086642.

Seeds 2000 and 2002 sit about 1.6e-5 over the icosahedral, Cs energy.

Those gaps are unpolished stops, and the logs lack both clocks.

The command names processor 0 while the header recorded pins 0-7.

A 1e-6 call would need the missing coordinates.

Hop 321 of trace a accepts infinite energy on the stall-climb arm.

Commit 87a1c409 adopts that quench at cluster_hopping.rs:6137 with no
finite test.

Commit 65af3318 refuses a non-finite climb at cluster_hopping.rs:6128.

The trail guard sits at cluster_hopping.rs:5891.

The new test fails on 87a1c409 once the tip test file is copied onto a
clean tree, naming stall-climb with energy inf.

It passes on 88d1cca.

Failure log logs/regress/inf-before.log.

Pass log logs/regress/regress-tip.log.

Hold coverage is missing on commit 299ce4ae, tree 826c4ba6.

The failing run starts there and copies tests/stall_growth_hold.rs from
88d1cca.

Status and diff: logs/regress/overlay-old-status.txt and
logs/regress/overlay-old.diff.

Compilation then reports 0 of 5 all-points acceptances.

Failure binary hash
ca2aee187f2558a84af7c50b3d5fd9c88ae223ffd181a3edd438f529ba388f68.

Passing binary hash
78879a6a9200b39de3ac14cc0444cba714371078256cf467b1c4343fcb448bf7.

Both filenames use cargo suffix d256e840, and the content hashes differ.

Traced logs stay under logs/traced.

Seeds run from 2004 to 2103.

Every line is retained.

The traced panel is carried over unchanged from the v4.2 kit.

Its global-minimum count is 0 of 100.

Its icosahedral, Cs count is 1 of 100, on seed 2017.

The carried rows of logs/sha-map.tsv keep both earlier digests.

The two-million receipts are raw, so their two digest columns match.

Those logs name head ed5849f6, stored here as 87a1c409, and both commits
share tree b8c8d0bc.

Rerun binary, earlier trace, and the hundred-seed check share hash
afcd1454e42f3f7fceb250ff6044ce1c62cd3ac2cb151c333ab333082da7cc94.

End geometries are data/traced-2004.xyz through data/traced-2103.xyz.

Aside from seed 2017, nearest icosahedral, Cs separation is 0.389543 on
seed 2057.

Seed 2054 is the closest to the global minimum, at separation 0.394381.

Seed 2004's earlier trace contains 16974 accept lines under
logs/prefix-trace.

Header tree 826c4ba6 belongs to commit 299ce4ae.

All-points chain source examples/lj_hold_chain.rs hashes to
138acb71b966124cfcfe1133e36195eaa62ecdf705f743a6440333347cb6bf85.

Binary for that check hashes to
911e16082a0adf1a84b6efa7dd1a8ff23068355b07536a4957a7b94d73092128.

Head-to-head example source hashes to
d52e7d219b8b9526ba819b02232271c32d34df3c59930a9c1fb4683fb805ec76.

```
299ce4ae replaces 20e03ec6 and keeps tree 826c4ba6
5fe9cb78 replaces 266de4a9 and adds the second trace emitter at cluster_hopping.rs:2774, the packing test at cluster_hopping.rs:5944, and the tighter tests
92069932 replaces f6fc3bb5 and mentions the two dump assertions
e8f4274f replaces b5fd49dc and only records the receipt script
87a1c409 replaces ed5849f6 and keeps tree b8c8d0bc
299ce4ae cluster_hopping.rs:5935 budget window, :6008 ih_dominated, :2890 empty hold, :4225 all-points replacement
87a1c409 clears the hold at cluster_hopping.rs:5994 and :6080
config.rs:1377 budget window false, preset.rs:8 temperature 0.8, preset.rs:11 step 0.38, on both commits
```
