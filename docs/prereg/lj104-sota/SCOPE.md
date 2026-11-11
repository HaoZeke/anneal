Copyright (c) anneal developers

We read every commit after 91b15a02 through the tip.
The tip identifier follows.
3b17550cae3bdf7d8140419f95258b60a0bc186e.
Git supplies each subject and each path.
diff-tree uses A for an addition.
The same command uses M for a modification.
The listing is one code block.
Each subject is copied whole.

```
Commit 75e82762d4ce8e73b7054ff0848ee22bc36896f6
Subject: lj104-prereg: record predictions and McNemar table
Adds 3 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/mcnemar_threshold.py
docs/prereg/lj104-sota/mcnemar_threshold.tsv

Commit 9348d8da7c382aac038041ff84d5c2ce3c531691
Subject: lj104-prereg: add the serial runner and its tests
Adds 2 paths.
docs/prereg/lj104-sota/lj104_sota_runner.py
docs/prereg/lj104-sota/test_lj104_sota_runner.py

Commit e414b7f83e7dcb7430ec031a638a330ecf8091a1
Subject: lj104-prereg: name panel seeds 6000-6005
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/PREDICTIONS.md

Commit 637d06b69bd961387de54027b4a82b5bf6f4a022
Subject: lj104-prereg: bind lj_requench for the LJ104 minimum
Adds no path.
Modifies 2 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
src/methods/lattice_search.rs

Commit c1c8e3eeaa62717acd246dcfc6acb4811889a3ae
Subject: lj104-prereg: set the plain DLS prediction to 20/30
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/PREDICTIONS.md

Commit b37387552bb5e46ca05d775f28813e2969e88afe
Subject: lj104-prereg: run the default panel from the runner
Adds no path.
Modifies 3 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/lj104_sota_runner.py
docs/prereg/lj104-sota/test_lj104_sota_runner.py

Commit 35967fde5d247d2b46fa1ecd34fac89c69fcce9e
Subject: lj104-prereg: restore the LJ104 occupation test
Adds no path.
Modifies 1 path.
src/methods/lattice_search.rs

Commit 77cfc366be173a95c3bced26bc3e05091d8d8fd2
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

Commit d257fd491b8e1d4c7efaa1302f30e967433e4039
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

Commit 28c48717906a2459e1e7ea171f71e054f9b3eca0
Subject: lj104-prereg: write floors from the raw panel logs
Adds 2 paths.
docs/prereg/lj104-sota/floors.md
docs/prereg/lj104-sota/write_floors.py
Modifies 1 path.
docs/prereg/lj104-sota/PREDICTIONS.md

Commit a661a9abaa13f3b92b88d5814eac6a5bb13c2998
Subject: lj104-prereg: score hits by re-quench at 1e-6
Adds 3 paths.
docs/prereg/lj104-sota/e_known.tsv
docs/prereg/lj104-sota/requench_panel.py
examples/lj_requench_xyz.rs
Modifies 2 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
src/methods/lattice_search.rs

Commit 5e65b208778083aebda0e80c5dc097ee131f1f1a
Subject: lj104-prereg: record re-quenched energies and calls
Adds 2 paths.
docs/prereg/lj104-sota/requench.md
docs/prereg/lj104-sota/requench.tsv
Modifies 2 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/requench_panel.py

Commit ed5dcf5a4eabd4f64fe9feb8eac097eceaf9791b
Subject: lj104-prereg: list commit subjects and added paths
Adds 2 paths.
docs/prereg/lj104-sota/SCOPE.md
docs/prereg/lj104-sota/scope_paths.py

Commit a4f9ccbdf1664ca1edaec0f008b972b46f5b7928
Subject: lj104-prereg: pin the GMIN digest and charge line
Adds 1 paths.
docs/prereg/lj104-sota/gmin_best.keywords
Modifies 2 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/requench_panel.py

Commit 297483c56ff6829109cdbc37c4e79b5d3866ec66
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

Commit f063e65f021bf9c66e7ca08000b890189e5e31b3
Subject: lj104-prereg: record the untimed replay pass
Adds 1 paths.
docs/prereg/lj104-sota/replay_commit_tree.py
Modifies 3 paths.
docs/prereg/lj104-sota/REPLAY.txt
docs/prereg/lj104-sota/TREE_EQUAL.txt
docs/prereg/lj104-sota/replay-map.tsv

Commit b5f4955a7c514a626a059d4292ae63f5edbc4b37
Subject: lj104-prereg: store the replay raw log
Adds 1 paths.
docs/prereg/lj104-sota/replay-commit-tree.log

Commit a5325208fb2afc0f509b742bedc8099ddf52b5f7
Subject: lj104-prereg: fix the scope script syntax
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/scope_paths.py

Commit 6c0b46ce139b18d330a63c34ca659bca5d5f77f5
Subject: lj104-prereg: rewrap the registration prose
Adds no path.
Modifies 4 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/TREE_DIFF_STAT.txt
docs/prereg/lj104-sota/floors.md
docs/prereg/lj104-sota/requench.md

Commit fe9005496356627e37883302674f216be4583a16
Subject: lj104-prereg: record the push plan
Adds 1 paths.
docs/prereg/lj104-sota/PUSH.txt

Commit 5b1f3b3368b81aeb46f6c72d4dd2dd505885ceb6
Subject: lj104-prereg: record the GMIN digest check
Adds 2 paths.
docs/prereg/lj104-sota/gmin-best-verify.log
docs/prereg/lj104-sota/verify_gmin_best.py

Commit ed5da69f3553586b413aa37742d61743d2c12024
Subject: lj104-prereg: write the scope as sentences
Adds no path.
Modifies 2 paths.
docs/prereg/lj104-sota/SCOPE.md
docs/prereg/lj104-sota/scope_paths.py

Commit e320c2625d9f9d26db6e9dd63aaf1920631c31b2
Subject: lj104-prereg: record the prose checker output
Adds 2 paths.
docs/prereg/lj104-sota/prose/proseguard-commit-msg.txt
docs/prereg/lj104-sota/prose/snapper-fmt.md
Modifies 4 paths.
docs/prereg/lj104-sota/prose/proseguard-prose-human.txt
docs/prereg/lj104-sota/prose/proseguard-rohit-paper.txt
docs/prereg/lj104-sota/prose/snapper-fmt.txt
docs/prereg/lj104-sota/prose/snapper.txt

Commit 88d3bb751f1151ee04f0d820c0eedcc9d69ccd3c
Subject: lj104-prereg: record the diff against v1.2
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/TREE_DIFF_STAT.txt

Commit a22d077f996d508a3bd74bc2b9e2c8742f1eb18e
Subject: lj104-prereg: record the GMIN digest status
Adds 3 paths.
docs/prereg/lj104-sota/NOTES.md
docs/prereg/lj104-sota/build_gmin_best.py
docs/prereg/lj104-sota/gmin-best-build.log
Modifies 3 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/gmin-best-verify.log
docs/prereg/lj104-sota/verify_gmin_best.py

Commit be0194fe4602c721c8318e7cc7f6ae80a853adfe
Subject: lj104-prereg: record the proseguard search
Adds no path.
Modifies 3 paths.
docs/prereg/lj104-sota/prose/proseguard-commit-msg.txt
docs/prereg/lj104-sota/prose/proseguard-prose-human.txt
docs/prereg/lj104-sota/prose/proseguard-rohit-paper.txt

Commit a0d86c09f4b98cbaa68d43ff3087c9877e4a9484
Subject: lj104-prereg: record the diff against v1.2
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/TREE_DIFF_STAT.txt

Commit 40d97f9dcce813efc540c6b9a4d4d2f093dbd052
Subject: lj104-prereg: record the full-history bundle verify
Adds 1 paths.
docs/prereg/lj104-sota/bundle-verify.txt

Commit 2aae8742fdb5149cda78a1f768f5875a17fc7f8c
Subject: lj104-prereg: keep the bundle verify within 72 columns
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/bundle-verify.txt

Commit 9db4c36462b537ae15216047f461278b0c3651d0
Subject: lj104-prereg: keep the verify note one sentence wide
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/bundle-verify.txt

Commit f8f73a92b4eb33c6c915933692bdcb9d573c7219
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

Commit 595dcb9b2cfbddc7d007c43e8052ded826860fdb
Subject: lj104-prereg: record the replay and the GMIN attempt
Adds 1 paths.
docs/prereg/lj104-sota/replay-v134.log
Modifies 4 paths.
docs/prereg/lj104-sota/REPLAY.txt
docs/prereg/lj104-sota/build_gmin_best.py
docs/prereg/lj104-sota/gmin-best-build.log
docs/prereg/lj104-sota/replay-map.tsv

Commit 093d0e11e5a9e8f38eb8c74f1c536c69941ec9e6
Subject: lj104-prereg: record the diff against v1.2
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/TREE_DIFF_STAT.txt

Commit 4fe893201c77940278ca4b47ac1fffd1e3bb4a76
Subject: lj104-prereg: close the v1.3.3 receipt gaps
Adds 1 paths.
docs/prereg/lj104-sota/replay-v135.log
Modifies 16 paths.
docs/prereg/lj104-sota/NOTES.md
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/PUSH.txt
docs/prereg/lj104-sota/REPLAY.txt
docs/prereg/lj104-sota/REPORT.md
docs/prereg/lj104-sota/SCOPE.md
docs/prereg/lj104-sota/TREE_EQUAL.txt
docs/prereg/lj104-sota/bundle-verify.txt
docs/prereg/lj104-sota/floors.md
docs/prereg/lj104-sota/prose/proseguard-commit-msg.txt
docs/prereg/lj104-sota/prose/proseguard-prose-human.txt
docs/prereg/lj104-sota/prose/proseguard-rohit-paper.txt
docs/prereg/lj104-sota/replay-map.tsv
docs/prereg/lj104-sota/requench_panel.py
docs/prereg/lj104-sota/scope_paths.py
docs/prereg/lj104-sota/write_floors.py

Commit 46b35839d057c1ef2aadd6473f4f11879e677796
Subject: lj104-prereg: record the prose extract and the tip hashes
Adds 1 paths.
docs/prereg/lj104-sota/PROSE_EXTRACT.txt
Modifies 3 paths.
docs/prereg/lj104-sota/TREE_DIFF_STAT.txt
docs/prereg/lj104-sota/prose/snapper-fmt.txt
docs/prereg/lj104-sota/prose/snapper.txt

Commit a5f5b10cee828c60cd7f54c092dc48bfb3e8f17f
Subject: lj104-prereg: record the replay map and the tip hashes
Adds no path.
Modifies 8 paths.
docs/prereg/lj104-sota/PROSE_EXTRACT.txt
docs/prereg/lj104-sota/REPLAY.txt
docs/prereg/lj104-sota/TREE_DIFF_STAT.txt
docs/prereg/lj104-sota/TREE_EQUAL.txt
docs/prereg/lj104-sota/bundle-verify.txt
docs/prereg/lj104-sota/prose/snapper-fmt.txt
docs/prereg/lj104-sota/prose/snapper.txt
docs/prereg/lj104-sota/replay-map.tsv

Commit 6cdcd860da1f3258a4e51e7d81f3699520a99636
Subject: lj104-prereg: state the cap and archive the replay
Adds 129 paths.
docs/prereg/lj104-sota/LAND_HOLD.txt
docs/prereg/lj104-sota/discarded/REASON.txt
docs/prereg/lj104-sota/discarded/floors.md
docs/prereg/lj104-sota/discarded/panel/01-lj38-6000.log
docs/prereg/lj104-sota/discarded/panel/01-lj38-6000.span
docs/prereg/lj104-sota/discarded/panel/01-lj38-6000.stderr
docs/prereg/lj104-sota/discarded/panel/01-lj38-6000.trace.csv
docs/prereg/lj104-sota/discarded/panel/01-lj38-6000.xyz
docs/prereg/lj104-sota/discarded/panel/02-lj38-6001.log
docs/prereg/lj104-sota/discarded/panel/02-lj38-6001.span
docs/prereg/lj104-sota/discarded/panel/02-lj38-6001.stderr
docs/prereg/lj104-sota/discarded/panel/02-lj38-6001.trace.csv
docs/prereg/lj104-sota/discarded/panel/02-lj38-6001.xyz
docs/prereg/lj104-sota/discarded/panel/03-lj38-6002.log
docs/prereg/lj104-sota/discarded/panel/03-lj38-6002.span
docs/prereg/lj104-sota/discarded/panel/03-lj38-6002.stderr
docs/prereg/lj104-sota/discarded/panel/03-lj38-6002.trace.csv
docs/prereg/lj104-sota/discarded/panel/03-lj38-6002.xyz
docs/prereg/lj104-sota/discarded/panel/04-lj38-6003.log
docs/prereg/lj104-sota/discarded/panel/04-lj38-6003.span
docs/prereg/lj104-sota/discarded/panel/04-lj38-6003.stderr
docs/prereg/lj104-sota/discarded/panel/04-lj38-6003.trace.csv
docs/prereg/lj104-sota/discarded/panel/04-lj38-6003.xyz
docs/prereg/lj104-sota/discarded/panel/05-lj38-6004.log
docs/prereg/lj104-sota/discarded/panel/05-lj38-6004.span
docs/prereg/lj104-sota/discarded/panel/05-lj38-6004.stderr
docs/prereg/lj104-sota/discarded/panel/05-lj38-6004.trace.csv
docs/prereg/lj104-sota/discarded/panel/05-lj38-6004.xyz
docs/prereg/lj104-sota/discarded/panel/06-lj38-6005.log
docs/prereg/lj104-sota/discarded/panel/06-lj38-6005.span
docs/prereg/lj104-sota/discarded/panel/06-lj38-6005.stderr
docs/prereg/lj104-sota/discarded/panel/06-lj38-6005.trace.csv
docs/prereg/lj104-sota/discarded/panel/06-lj38-6005.xyz
docs/prereg/lj104-sota/discarded/panel/07-lj55-6000.log
docs/prereg/lj104-sota/discarded/panel/07-lj55-6000.span
docs/prereg/lj104-sota/discarded/panel/07-lj55-6000.stderr
docs/prereg/lj104-sota/discarded/panel/07-lj55-6000.trace.csv
docs/prereg/lj104-sota/discarded/panel/07-lj55-6000.xyz
docs/prereg/lj104-sota/discarded/panel/08-lj55-6001.log
docs/prereg/lj104-sota/discarded/panel/08-lj55-6001.span
docs/prereg/lj104-sota/discarded/panel/08-lj55-6001.stderr
docs/prereg/lj104-sota/discarded/panel/08-lj55-6001.trace.csv
docs/prereg/lj104-sota/discarded/panel/08-lj55-6001.xyz
docs/prereg/lj104-sota/discarded/panel/09-lj55-6002.log
docs/prereg/lj104-sota/discarded/panel/09-lj55-6002.span
docs/prereg/lj104-sota/discarded/panel/09-lj55-6002.stderr
docs/prereg/lj104-sota/discarded/panel/09-lj55-6002.trace.csv
docs/prereg/lj104-sota/discarded/panel/09-lj55-6002.xyz
docs/prereg/lj104-sota/discarded/panel/10-lj55-6003.log
docs/prereg/lj104-sota/discarded/panel/10-lj55-6003.span
docs/prereg/lj104-sota/discarded/panel/10-lj55-6003.stderr
docs/prereg/lj104-sota/discarded/panel/10-lj55-6003.trace.csv
docs/prereg/lj104-sota/discarded/panel/10-lj55-6003.xyz
docs/prereg/lj104-sota/discarded/panel/11-lj55-6004.log
docs/prereg/lj104-sota/discarded/panel/11-lj55-6004.span
docs/prereg/lj104-sota/discarded/panel/11-lj55-6004.stderr
docs/prereg/lj104-sota/discarded/panel/11-lj55-6004.trace.csv
docs/prereg/lj104-sota/discarded/panel/11-lj55-6004.xyz
docs/prereg/lj104-sota/discarded/panel/12-lj55-6005.log
docs/prereg/lj104-sota/discarded/panel/12-lj55-6005.span
docs/prereg/lj104-sota/discarded/panel/12-lj55-6005.stderr
docs/prereg/lj104-sota/discarded/panel/12-lj55-6005.trace.csv
docs/prereg/lj104-sota/discarded/panel/12-lj55-6005.xyz
docs/prereg/lj104-sota/discarded/panel/13-lj75-6000.log
docs/prereg/lj104-sota/discarded/panel/13-lj75-6000.span
docs/prereg/lj104-sota/discarded/panel/13-lj75-6000.stderr
docs/prereg/lj104-sota/discarded/panel/13-lj75-6000.trace.csv
docs/prereg/lj104-sota/discarded/panel/13-lj75-6000.xyz
docs/prereg/lj104-sota/discarded/panel/14-lj75-6001.log
docs/prereg/lj104-sota/discarded/panel/14-lj75-6001.span
docs/prereg/lj104-sota/discarded/panel/14-lj75-6001.stderr
docs/prereg/lj104-sota/discarded/panel/14-lj75-6001.trace.csv
docs/prereg/lj104-sota/discarded/panel/14-lj75-6001.xyz
docs/prereg/lj104-sota/discarded/panel/15-lj75-6002.log
docs/prereg/lj104-sota/discarded/panel/15-lj75-6002.span
docs/prereg/lj104-sota/discarded/panel/15-lj75-6002.stderr
docs/prereg/lj104-sota/discarded/panel/15-lj75-6002.trace.csv
docs/prereg/lj104-sota/discarded/panel/15-lj75-6002.xyz
docs/prereg/lj104-sota/discarded/panel/16-lj75-6003.log
docs/prereg/lj104-sota/discarded/panel/16-lj75-6003.span
docs/prereg/lj104-sota/discarded/panel/16-lj75-6003.stderr
docs/prereg/lj104-sota/discarded/panel/16-lj75-6003.trace.csv
docs/prereg/lj104-sota/discarded/panel/16-lj75-6003.xyz
docs/prereg/lj104-sota/discarded/panel/17-lj75-6004.log
docs/prereg/lj104-sota/discarded/panel/17-lj75-6004.span
docs/prereg/lj104-sota/discarded/panel/17-lj75-6004.stderr
docs/prereg/lj104-sota/discarded/panel/17-lj75-6004.trace.csv
docs/prereg/lj104-sota/discarded/panel/17-lj75-6004.xyz
docs/prereg/lj104-sota/discarded/panel/18-lj75-6005.log
docs/prereg/lj104-sota/discarded/panel/18-lj75-6005.span
docs/prereg/lj104-sota/discarded/panel/18-lj75-6005.stderr
docs/prereg/lj104-sota/discarded/panel/18-lj75-6005.trace.csv
docs/prereg/lj104-sota/discarded/panel/18-lj75-6005.xyz
docs/prereg/lj104-sota/discarded/panel/19-lj98-6000.log
docs/prereg/lj104-sota/discarded/panel/19-lj98-6000.span
docs/prereg/lj104-sota/discarded/panel/19-lj98-6000.stderr
docs/prereg/lj104-sota/discarded/panel/19-lj98-6000.trace.csv
docs/prereg/lj104-sota/discarded/panel/19-lj98-6000.xyz
docs/prereg/lj104-sota/discarded/panel/20-lj98-6001.log
docs/prereg/lj104-sota/discarded/panel/20-lj98-6001.span
docs/prereg/lj104-sota/discarded/panel/20-lj98-6001.stderr
docs/prereg/lj104-sota/discarded/panel/20-lj98-6001.trace.csv
docs/prereg/lj104-sota/discarded/panel/20-lj98-6001.xyz
docs/prereg/lj104-sota/discarded/panel/21-lj98-6002.log
docs/prereg/lj104-sota/discarded/panel/21-lj98-6002.span
docs/prereg/lj104-sota/discarded/panel/21-lj98-6002.stderr
docs/prereg/lj104-sota/discarded/panel/21-lj98-6002.trace.csv
docs/prereg/lj104-sota/discarded/panel/21-lj98-6002.xyz
docs/prereg/lj104-sota/discarded/panel/22-lj98-6003.log
docs/prereg/lj104-sota/discarded/panel/22-lj98-6003.span
docs/prereg/lj104-sota/discarded/panel/22-lj98-6003.stderr
docs/prereg/lj104-sota/discarded/panel/22-lj98-6003.trace.csv
docs/prereg/lj104-sota/discarded/panel/22-lj98-6003.xyz
docs/prereg/lj104-sota/discarded/panel/23-lj98-6004.log
docs/prereg/lj104-sota/discarded/panel/23-lj98-6004.span
docs/prereg/lj104-sota/discarded/panel/23-lj98-6004.stderr
docs/prereg/lj104-sota/discarded/panel/23-lj98-6004.trace.csv
docs/prereg/lj104-sota/discarded/panel/23-lj98-6004.xyz
docs/prereg/lj104-sota/discarded/panel/24-lj98-6005.log
docs/prereg/lj104-sota/discarded/panel/24-lj98-6005.span
docs/prereg/lj104-sota/discarded/panel/24-lj98-6005.stderr
docs/prereg/lj104-sota/discarded/panel/24-lj98-6005.trace.csv
docs/prereg/lj104-sota/discarded/panel/24-lj98-6005.xyz
docs/prereg/lj104-sota/discarded/panel/end.receipt
docs/prereg/lj104-sota/discarded/panel/rows.tsv
docs/prereg/lj104-sota/discarded/panel/session.txt
docs/prereg/lj104-sota/discarded/panel/start.receipt
docs/prereg/lj104-sota/write_bundle_verify.py
docs/prereg/lj104-sota/write_tree_diff_stat.py
Modifies 23 paths.
docs/prereg/lj104-sota/NOTES.md
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/PROSE_EXTRACT.txt
docs/prereg/lj104-sota/PUSH.txt
docs/prereg/lj104-sota/REPLAY.txt
docs/prereg/lj104-sota/REPORT.md
docs/prereg/lj104-sota/SCOPE.md
docs/prereg/lj104-sota/TREE_DIFF_STAT.txt
docs/prereg/lj104-sota/TREE_EQUAL.txt
docs/prereg/lj104-sota/floors.md
docs/prereg/lj104-sota/lj104_sota_runner.py
docs/prereg/lj104-sota/prose/proseguard-commit-msg.txt
docs/prereg/lj104-sota/prose/proseguard-prose-human.txt
docs/prereg/lj104-sota/prose/proseguard-rohit-paper.txt
docs/prereg/lj104-sota/prose/snapper-fmt.md
docs/prereg/lj104-sota/prose/snapper-fmt.txt
docs/prereg/lj104-sota/prose/snapper.txt
docs/prereg/lj104-sota/replay-map.tsv
docs/prereg/lj104-sota/requench.md
docs/prereg/lj104-sota/requench_panel.py
docs/prereg/lj104-sota/scope_paths.py
docs/prereg/lj104-sota/write_floors.py
src/methods/lattice_search.rs

Commit c0723143a462f3a6e51b6081956c666a509d58d1
Subject: lj104-prereg: rerun the panel on rustc 1.88.0
Adds 2 paths.
docs/prereg/lj104-sota/T.txt
docs/prereg/lj104-sota/writers.log
Modifies 51 paths.
docs/prereg/lj104-sota/floors.md
docs/prereg/lj104-sota/panel/01-lj38-6000.log
docs/prereg/lj104-sota/panel/01-lj38-6000.span
docs/prereg/lj104-sota/panel/02-lj38-6001.log
docs/prereg/lj104-sota/panel/02-lj38-6001.span
docs/prereg/lj104-sota/panel/03-lj38-6002.span
docs/prereg/lj104-sota/panel/04-lj38-6003.span
docs/prereg/lj104-sota/panel/05-lj38-6004.log
docs/prereg/lj104-sota/panel/05-lj38-6004.span
docs/prereg/lj104-sota/panel/06-lj38-6005.log
docs/prereg/lj104-sota/panel/06-lj38-6005.span
docs/prereg/lj104-sota/panel/07-lj55-6000.log
docs/prereg/lj104-sota/panel/07-lj55-6000.span
docs/prereg/lj104-sota/panel/08-lj55-6001.log
docs/prereg/lj104-sota/panel/08-lj55-6001.span
docs/prereg/lj104-sota/panel/09-lj55-6002.span
docs/prereg/lj104-sota/panel/10-lj55-6003.log
docs/prereg/lj104-sota/panel/10-lj55-6003.span
docs/prereg/lj104-sota/panel/11-lj55-6004.log
docs/prereg/lj104-sota/panel/11-lj55-6004.span
docs/prereg/lj104-sota/panel/12-lj55-6005.log
docs/prereg/lj104-sota/panel/12-lj55-6005.span
docs/prereg/lj104-sota/panel/13-lj75-6000.log
docs/prereg/lj104-sota/panel/13-lj75-6000.span
docs/prereg/lj104-sota/panel/14-lj75-6001.log
docs/prereg/lj104-sota/panel/14-lj75-6001.span
docs/prereg/lj104-sota/panel/15-lj75-6002.log
docs/prereg/lj104-sota/panel/15-lj75-6002.span
docs/prereg/lj104-sota/panel/16-lj75-6003.log
docs/prereg/lj104-sota/panel/16-lj75-6003.span
docs/prereg/lj104-sota/panel/17-lj75-6004.log
docs/prereg/lj104-sota/panel/17-lj75-6004.span
docs/prereg/lj104-sota/panel/18-lj75-6005.log
docs/prereg/lj104-sota/panel/18-lj75-6005.span
docs/prereg/lj104-sota/panel/19-lj98-6000.log
docs/prereg/lj104-sota/panel/19-lj98-6000.span
docs/prereg/lj104-sota/panel/20-lj98-6001.log
docs/prereg/lj104-sota/panel/20-lj98-6001.span
docs/prereg/lj104-sota/panel/21-lj98-6002.log
docs/prereg/lj104-sota/panel/21-lj98-6002.span
docs/prereg/lj104-sota/panel/22-lj98-6003.log
docs/prereg/lj104-sota/panel/22-lj98-6003.span
docs/prereg/lj104-sota/panel/23-lj98-6004.log
docs/prereg/lj104-sota/panel/23-lj98-6004.span
docs/prereg/lj104-sota/panel/24-lj98-6005.log
docs/prereg/lj104-sota/panel/24-lj98-6005.span
docs/prereg/lj104-sota/panel/end.receipt
docs/prereg/lj104-sota/panel/rows.tsv
docs/prereg/lj104-sota/panel/session.txt
docs/prereg/lj104-sota/panel/start.receipt
docs/prereg/lj104-sota/requench.md

Commit 32a05d20ae4b5090adcec8dfd095ebdb86834d07
Subject: lj104-prereg: record the scope listing and the tip hashes
Adds 2 paths.
docs/prereg/lj104-sota/.scored-run.lock
docs/prereg/lj104-sota/PROSE-RECEIPT.txt
Modifies 10 paths.
docs/prereg/lj104-sota/PROSE_EXTRACT.txt
docs/prereg/lj104-sota/SCOPE.md
docs/prereg/lj104-sota/TREE_DIFF_STAT.txt
docs/prereg/lj104-sota/bundle-verify.txt
docs/prereg/lj104-sota/prose/snapper-fmt.txt
docs/prereg/lj104-sota/prose/snapper.txt
docs/prereg/lj104-sota/scope_paths.py
docs/prereg/lj104-sota/write_bundle_verify.py
docs/prereg/lj104-sota/write_tree_diff_stat.py
docs/prereg/lj104-sota/writers.log

Commit a89c0e7d922cbde17ff24740def1ba674fada32b
Subject: lj104-prereg: drop the empty scored-run lock
Adds no path.

Commit 8d22df40f0910592a45ca28cd65768053f1e8461
Subject: lj104-prereg: record the real replay and the seal writer
Adds 8 paths.
docs/prereg/lj104-sota/discarded/README.txt
docs/prereg/lj104-sota/note_tip.py
docs/prereg/lj104-sota/prose_rate.py
docs/prereg/lj104-sota/replay-pass-command.txt
docs/prereg/lj104-sota/run_coverage.py
docs/prereg/lj104-sota/write_prose_receipt.py
docs/prereg/lj104-sota/write_replay_log.py
docs/prereg/lj104-sota/write_seal.py
Modifies 14 paths.
docs/prereg/lj104-sota/LAND_HOLD.txt
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/PROSE-RECEIPT.txt
docs/prereg/lj104-sota/REPLAY.txt
docs/prereg/lj104-sota/T.txt
docs/prereg/lj104-sota/TREE_DIFF_STAT.txt
docs/prereg/lj104-sota/discarded/REASON.txt
docs/prereg/lj104-sota/prose/snapper.txt
docs/prereg/lj104-sota/replay-commit-tree.log
docs/prereg/lj104-sota/scope_paths.py
docs/prereg/lj104-sota/write_bundle_verify.py
docs/prereg/lj104-sota/write_tree_diff_stat.py
examples/lj_requench_xyz.rs
src/methods/lattice_search.rs

Commit 1b831de7aca86509448542ae61140c53e6818337
Subject: lj104-prereg: record line and branch coverage
Adds 1 paths.
docs/prereg/lj104-sota/COVERAGE.txt
Modifies 1 path.
docs/prereg/lj104-sota/run_coverage.py

Commit 3b17550cae3bdf7d8140419f95258b60a0bc186e
Subject: lj104-prereg: keep the coverage note within 72 columns
Adds no path.
Modifies 2 paths.
docs/prereg/lj104-sota/COVERAGE.txt
docs/prereg/lj104-sota/run_coverage.py

```
