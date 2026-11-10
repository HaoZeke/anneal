Copyright © anneal developers

We read every commit after 91b15a02 through the tip.
The tip identifier follows.
3d0838e0e71bd209031dff87dff0a71ff963c335.
Subjects and paths come from git.
An added path is a diff-tree addition.
A modified path is a diff-tree modification.
The listing below is one code block.
Each subject is copied whole.

```
Commit 149aa73d5cf4e1afb0546ad86fb401d524d66244
Subject: lj104-prereg: record predictions and McNemar table
Adds 3 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/mcnemar_threshold.py
docs/prereg/lj104-sota/mcnemar_threshold.tsv

Commit 6f07484e2bfc413202596a640dd68995f48bad9e
Subject: lj104-prereg: add the serial runner and its tests
Adds 2 paths.
docs/prereg/lj104-sota/lj104_sota_runner.py
docs/prereg/lj104-sota/test_lj104_sota_runner.py

Commit 20313a8794da66e85ec9fbe95a2d41c02bd43390
Subject: lj104-prereg: name panel seeds 6000-6005
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/PREDICTIONS.md

Commit d3b8cd189afea533a93d5790c030cf322d2be487
Subject: lj104-prereg: bind lj_requench for the LJ104 minimum
Adds no path.
Modifies 2 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
src/methods/lattice_search.rs

Commit ed7b340da4fd3c87161a26fe78fd86cd27438be2
Subject: lj104-prereg: set the plain DLS prediction to 20/30
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/PREDICTIONS.md

Commit c85cdcf46eb58e71e9194bfceec28e5c0261ddbf
Subject: lj104-prereg: run the default panel from the runner
Adds no path.
Modifies 3 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/lj104_sota_runner.py
docs/prereg/lj104-sota/test_lj104_sota_runner.py

Commit ca034201ec3d2ab08f482428c6f3eb4abebd0644
Subject: lj104-prereg: restore the LJ104 occupation test
Adds no path.
Modifies 1 path.
src/methods/lattice_search.rs

Commit a338054c7afabbaf76ab4a273cadfccb2755877a
Subject: lj104-prereg: commit raw panel traces and receipts
Adds 100 paths.
docs/prereg/lj104-sota/panel/01-lj38-6000.span
docs/prereg/lj104-sota/panel/01-lj38-6000.stderr
docs/prereg/lj104-sota/panel/01-lj38-6000.trace.csv
docs/prereg/lj104-sota/panel/01-lj38-6000.xyz
docs/prereg/lj104-sota/panel/02-lj38-6001.span
docs/prereg/lj104-sota/panel/02-lj38-6001.stderr
docs/prereg/lj104-sota/panel/02-lj38-6001.trace.csv
docs/prereg/lj104-sota/panel/02-lj38-6001.xyz
docs/prereg/lj104-sota/panel/03-lj38-6002.span
docs/prereg/lj104-sota/panel/03-lj38-6002.stderr
docs/prereg/lj104-sota/panel/03-lj38-6002.trace.csv
docs/prereg/lj104-sota/panel/03-lj38-6002.xyz
docs/prereg/lj104-sota/panel/04-lj38-6003.span
docs/prereg/lj104-sota/panel/04-lj38-6003.stderr
docs/prereg/lj104-sota/panel/04-lj38-6003.trace.csv
docs/prereg/lj104-sota/panel/04-lj38-6003.xyz
docs/prereg/lj104-sota/panel/05-lj38-6004.span
docs/prereg/lj104-sota/panel/05-lj38-6004.stderr
docs/prereg/lj104-sota/panel/05-lj38-6004.trace.csv
docs/prereg/lj104-sota/panel/05-lj38-6004.xyz
docs/prereg/lj104-sota/panel/06-lj38-6005.span
docs/prereg/lj104-sota/panel/06-lj38-6005.stderr
docs/prereg/lj104-sota/panel/06-lj38-6005.trace.csv
docs/prereg/lj104-sota/panel/06-lj38-6005.xyz
docs/prereg/lj104-sota/panel/07-lj55-6000.span
docs/prereg/lj104-sota/panel/07-lj55-6000.stderr
docs/prereg/lj104-sota/panel/07-lj55-6000.trace.csv
docs/prereg/lj104-sota/panel/07-lj55-6000.xyz
docs/prereg/lj104-sota/panel/08-lj55-6001.span
docs/prereg/lj104-sota/panel/08-lj55-6001.stderr
docs/prereg/lj104-sota/panel/08-lj55-6001.trace.csv
docs/prereg/lj104-sota/panel/08-lj55-6001.xyz
docs/prereg/lj104-sota/panel/09-lj55-6002.span
docs/prereg/lj104-sota/panel/09-lj55-6002.stderr
docs/prereg/lj104-sota/panel/09-lj55-6002.trace.csv
docs/prereg/lj104-sota/panel/09-lj55-6002.xyz
docs/prereg/lj104-sota/panel/10-lj55-6003.span
docs/prereg/lj104-sota/panel/10-lj55-6003.stderr
docs/prereg/lj104-sota/panel/10-lj55-6003.trace.csv
docs/prereg/lj104-sota/panel/10-lj55-6003.xyz
docs/prereg/lj104-sota/panel/11-lj55-6004.span
docs/prereg/lj104-sota/panel/11-lj55-6004.stderr
docs/prereg/lj104-sota/panel/11-lj55-6004.trace.csv
docs/prereg/lj104-sota/panel/11-lj55-6004.xyz
docs/prereg/lj104-sota/panel/12-lj55-6005.span
docs/prereg/lj104-sota/panel/12-lj55-6005.stderr
docs/prereg/lj104-sota/panel/12-lj55-6005.trace.csv
docs/prereg/lj104-sota/panel/12-lj55-6005.xyz
docs/prereg/lj104-sota/panel/13-lj75-6000.span
docs/prereg/lj104-sota/panel/13-lj75-6000.stderr
docs/prereg/lj104-sota/panel/13-lj75-6000.trace.csv
docs/prereg/lj104-sota/panel/13-lj75-6000.xyz
docs/prereg/lj104-sota/panel/14-lj75-6001.span
docs/prereg/lj104-sota/panel/14-lj75-6001.stderr
docs/prereg/lj104-sota/panel/14-lj75-6001.trace.csv
docs/prereg/lj104-sota/panel/14-lj75-6001.xyz
docs/prereg/lj104-sota/panel/15-lj75-6002.span
docs/prereg/lj104-sota/panel/15-lj75-6002.stderr
docs/prereg/lj104-sota/panel/15-lj75-6002.trace.csv
docs/prereg/lj104-sota/panel/15-lj75-6002.xyz
docs/prereg/lj104-sota/panel/16-lj75-6003.span
docs/prereg/lj104-sota/panel/16-lj75-6003.stderr
docs/prereg/lj104-sota/panel/16-lj75-6003.trace.csv
docs/prereg/lj104-sota/panel/16-lj75-6003.xyz
docs/prereg/lj104-sota/panel/17-lj75-6004.span
docs/prereg/lj104-sota/panel/17-lj75-6004.stderr
docs/prereg/lj104-sota/panel/17-lj75-6004.trace.csv
docs/prereg/lj104-sota/panel/17-lj75-6004.xyz
docs/prereg/lj104-sota/panel/18-lj75-6005.span
docs/prereg/lj104-sota/panel/18-lj75-6005.stderr
docs/prereg/lj104-sota/panel/18-lj75-6005.trace.csv
docs/prereg/lj104-sota/panel/18-lj75-6005.xyz
docs/prereg/lj104-sota/panel/19-lj98-6000.span
docs/prereg/lj104-sota/panel/19-lj98-6000.stderr
docs/prereg/lj104-sota/panel/19-lj98-6000.trace.csv
docs/prereg/lj104-sota/panel/19-lj98-6000.xyz
docs/prereg/lj104-sota/panel/20-lj98-6001.span
docs/prereg/lj104-sota/panel/20-lj98-6001.stderr
docs/prereg/lj104-sota/panel/20-lj98-6001.trace.csv
docs/prereg/lj104-sota/panel/20-lj98-6001.xyz
docs/prereg/lj104-sota/panel/21-lj98-6002.span
docs/prereg/lj104-sota/panel/21-lj98-6002.stderr
docs/prereg/lj104-sota/panel/21-lj98-6002.trace.csv
docs/prereg/lj104-sota/panel/21-lj98-6002.xyz
docs/prereg/lj104-sota/panel/22-lj98-6003.span
docs/prereg/lj104-sota/panel/22-lj98-6003.stderr
docs/prereg/lj104-sota/panel/22-lj98-6003.trace.csv
docs/prereg/lj104-sota/panel/22-lj98-6003.xyz
docs/prereg/lj104-sota/panel/23-lj98-6004.span
docs/prereg/lj104-sota/panel/23-lj98-6004.stderr
docs/prereg/lj104-sota/panel/23-lj98-6004.trace.csv
docs/prereg/lj104-sota/panel/23-lj98-6004.xyz
docs/prereg/lj104-sota/panel/24-lj98-6005.span
docs/prereg/lj104-sota/panel/24-lj98-6005.stderr
docs/prereg/lj104-sota/panel/24-lj98-6005.trace.csv
docs/prereg/lj104-sota/panel/24-lj98-6005.xyz
docs/prereg/lj104-sota/panel/end.receipt
docs/prereg/lj104-sota/panel/rows.tsv
docs/prereg/lj104-sota/panel/session.txt
docs/prereg/lj104-sota/panel/start.receipt

Commit 09116a9c1929359881eb9ef0e2bd996b95cbaad0
Subject: lj104-prereg: add the untouched panel stdout logs
Adds 24 paths.
docs/prereg/lj104-sota/panel/01-lj38-6000.log
docs/prereg/lj104-sota/panel/02-lj38-6001.log
docs/prereg/lj104-sota/panel/03-lj38-6002.log
docs/prereg/lj104-sota/panel/04-lj38-6003.log
docs/prereg/lj104-sota/panel/05-lj38-6004.log
docs/prereg/lj104-sota/panel/06-lj38-6005.log
docs/prereg/lj104-sota/panel/07-lj55-6000.log
docs/prereg/lj104-sota/panel/08-lj55-6001.log
docs/prereg/lj104-sota/panel/09-lj55-6002.log
docs/prereg/lj104-sota/panel/10-lj55-6003.log
docs/prereg/lj104-sota/panel/11-lj55-6004.log
docs/prereg/lj104-sota/panel/12-lj55-6005.log
docs/prereg/lj104-sota/panel/13-lj75-6000.log
docs/prereg/lj104-sota/panel/14-lj75-6001.log
docs/prereg/lj104-sota/panel/15-lj75-6002.log
docs/prereg/lj104-sota/panel/16-lj75-6003.log
docs/prereg/lj104-sota/panel/17-lj75-6004.log
docs/prereg/lj104-sota/panel/18-lj75-6005.log
docs/prereg/lj104-sota/panel/19-lj98-6000.log
docs/prereg/lj104-sota/panel/20-lj98-6001.log
docs/prereg/lj104-sota/panel/21-lj98-6002.log
docs/prereg/lj104-sota/panel/22-lj98-6003.log
docs/prereg/lj104-sota/panel/23-lj98-6004.log
docs/prereg/lj104-sota/panel/24-lj98-6005.log

Commit 3aaed17c83d2120ea33aecf1d21d1acbce80eac8
Subject: lj104-prereg: write floors from the raw panel logs
Adds 2 paths.
docs/prereg/lj104-sota/floors.md
docs/prereg/lj104-sota/write_floors.py
Modifies 1 path.
docs/prereg/lj104-sota/PREDICTIONS.md

Commit aaf3bae7114d52a5f835636ff3d01b2f7fbad642
Subject: lj104-prereg: score hits by re-quench at 1e-6
Adds 3 paths.
docs/prereg/lj104-sota/e_known.tsv
docs/prereg/lj104-sota/requench_panel.py
examples/lj_requench_xyz.rs
Modifies 2 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
src/methods/lattice_search.rs

Commit 1a761592355acbe0213361953d4b6e2216ab71d4
Subject: lj104-prereg: record re-quenched energies and calls
Adds 2 paths.
docs/prereg/lj104-sota/requench.md
docs/prereg/lj104-sota/requench.tsv
Modifies 2 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/requench_panel.py

Commit b7effec3901d59cf179aecbbeba27817949a72b0
Subject: lj104-prereg: list commit subjects and added paths
Adds 2 paths.
docs/prereg/lj104-sota/SCOPE.md
docs/prereg/lj104-sota/scope_paths.py

Commit c41f33f17c15924fed4fa0c02cd7727fae2933e8
Subject: lj104-prereg: pin the GMIN digest and charge line
Adds 1 paths.
docs/prereg/lj104-sota/gmin_best.keywords
Modifies 2 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/requench_panel.py

Commit 5519f1a4fa559fb59c83512db41898c8178eda73
Subject: lj104-prereg: record replay notes and map
Adds 8 paths.
docs/prereg/lj104-sota/REPLAY.txt
docs/prereg/lj104-sota/TREE_DIFF_STAT.txt
docs/prereg/lj104-sota/TREE_EQUAL.txt
docs/prereg/lj104-sota/prose/proseguard-prose-human.txt
docs/prereg/lj104-sota/prose/proseguard-rohit-paper.txt
docs/prereg/lj104-sota/prose/snapper-fmt.txt
docs/prereg/lj104-sota/prose/snapper.txt
docs/prereg/lj104-sota/replay-map.tsv

Commit 4bc4fdd924e2a6f567f35998ed4a96e2f4f58528
Subject: lj104-prereg: record the untimed replay pass
Adds 1 paths.
docs/prereg/lj104-sota/replay_commit_tree.py
Modifies 3 paths.
docs/prereg/lj104-sota/REPLAY.txt
docs/prereg/lj104-sota/TREE_EQUAL.txt
docs/prereg/lj104-sota/replay-map.tsv

Commit eb467d8d34796744168c255692098abdd350db39
Subject: lj104-prereg: store the replay raw log
Adds 1 paths.
docs/prereg/lj104-sota/replay-commit-tree.log

Commit 846709f303cc4afb2aa2b0e351fc854781de515b
Subject: lj104-prereg: fix the scope script syntax
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/scope_paths.py

Commit 3420b91b54dc704977c8463b684f99897f592b02
Subject: lj104-prereg: rewrap the registration prose
Adds no path.
Modifies 4 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/TREE_DIFF_STAT.txt
docs/prereg/lj104-sota/floors.md
docs/prereg/lj104-sota/requench.md

Commit 87153f7d6735a853f730516a5894027e75478b52
Subject: lj104-prereg: record the push plan
Adds 1 paths.
docs/prereg/lj104-sota/PUSH.txt

Commit 927fa5e397988f936d0f4ed0e312da3927e0005a
Subject: lj104-prereg: record the GMIN digest check
Adds 2 paths.
docs/prereg/lj104-sota/gmin-best-verify.log
docs/prereg/lj104-sota/verify_gmin_best.py

Commit c493107e0300ac9164593463a4582034ecf695e6
Subject: lj104-prereg: write the scope as sentences
Adds no path.
Modifies 2 paths.
docs/prereg/lj104-sota/SCOPE.md
docs/prereg/lj104-sota/scope_paths.py

Commit 0f6f85e1e0c999d5d70e25e1c26f7de81d92a56b
Subject: lj104-prereg: record the prose checker output
Adds 2 paths.
docs/prereg/lj104-sota/prose/proseguard-commit-msg.txt
docs/prereg/lj104-sota/prose/snapper-fmt.md
Modifies 4 paths.
docs/prereg/lj104-sota/prose/proseguard-prose-human.txt
docs/prereg/lj104-sota/prose/proseguard-rohit-paper.txt
docs/prereg/lj104-sota/prose/snapper-fmt.txt
docs/prereg/lj104-sota/prose/snapper.txt

Commit 9cd85c55fca6268a132d1a91e57a66550af1b921
Subject: lj104-prereg: record the diff against v1.2
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/TREE_DIFF_STAT.txt

Commit 79a3b2dcd3ad5b2b0747338788b968089402afd9
Subject: lj104-prereg: record the GMIN digest status
Adds 3 paths.
docs/prereg/lj104-sota/NOTES.md
docs/prereg/lj104-sota/build_gmin_best.py
docs/prereg/lj104-sota/gmin-best-build.log
Modifies 3 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/gmin-best-verify.log
docs/prereg/lj104-sota/verify_gmin_best.py

Commit 31ac8719696e342fe71e0cc8c0340c33f6a40c4e
Subject: lj104-prereg: record the proseguard search
Adds no path.
Modifies 3 paths.
docs/prereg/lj104-sota/prose/proseguard-commit-msg.txt
docs/prereg/lj104-sota/prose/proseguard-prose-human.txt
docs/prereg/lj104-sota/prose/proseguard-rohit-paper.txt

Commit e2002273e12109c45ae02c187263cfe5d67c3269
Subject: lj104-prereg: record the diff against v1.2
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/TREE_DIFF_STAT.txt

Commit c74219ae28a5b1dd69bb1441320cb2fa48707892
Subject: lj104-prereg: record the full-history bundle verify
Adds 1 paths.
docs/prereg/lj104-sota/bundle-verify.txt

Commit 2f18d4c8bf2fe74cd2e008e709085b271bf3dbf0
Subject: lj104-prereg: keep the bundle verify within 72 columns
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/bundle-verify.txt

Commit d278b120473d2481c3225dc1b3eff9829bbe968b
Subject: lj104-prereg: keep the verify note one sentence wide
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/bundle-verify.txt

Commit 3f8c57f8300b99b2b85ab35526d0bfcd86286af1
Subject: lj104-prereg: clear the registration prose hits
Adds 1 paths.
docs/prereg/lj104-sota/REPORT.md
Modifies 6 paths.
docs/prereg/lj104-sota/NOTES.md
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/SCOPE.md
docs/prereg/lj104-sota/floors.md
docs/prereg/lj104-sota/prose/snapper-fmt.md
docs/prereg/lj104-sota/requench.md

Commit fe4e9d5bdc63f31034180fae07f5a20dadb194a2
Subject: lj104-prereg: record the replay and the GMIN attempt
Adds 1 paths.
docs/prereg/lj104-sota/replay-v134.log
Modifies 4 paths.
docs/prereg/lj104-sota/REPLAY.txt
docs/prereg/lj104-sota/build_gmin_best.py
docs/prereg/lj104-sota/gmin-best-build.log
docs/prereg/lj104-sota/replay-map.tsv

Commit 3d0838e0e71bd209031dff87dff0a71ff963c335
Subject: lj104-prereg: record the diff against v1.2
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/TREE_DIFF_STAT.txt

```
