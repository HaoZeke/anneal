Copyright (c) anneal developers

Scope of the twelve commits after 91b15a02 through 5fb7368d.
Subjects and paths are taken from git. An added path is a
diff-tree addition. A modified path is a diff-tree modification.

Commit fdc321918537e44ea452b887ee4d818289e95884
Subject: Register the LJ104 held-out predictions, the one-sided McNemar
thresholds, and the unfilled six-seed floors.
Adds 3 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/mcnemar_threshold.py
docs/prereg/lj104-sota/mcnemar_threshold.tsv

Commit 8004d790392e4f8614cc0e9e64a1aead7d0f3ce7
Subject: Add the serial LJ104 runner and tests that it refuses a second
scored start, writes its blob, HEAD, and tree, and leaves seeds
5000-5029 unrun.
Adds 2 paths.
docs/prereg/lj104-sota/lj104_sota_runner.py
docs/prereg/lj104-sota/test_lj104_sota_runner.py

Commit c9bac09f0d3328f35e767d4c5059eebe737f1729
Subject: Name the no-regression panel seeds 6000-6005 and record that
they were not used for tuning.
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/PREDICTIONS.md

Commit 45139a21e18d0c911c7c60149d62e4c87e5428e0
Subject: Bind lj_requench to WarmLbfgs::minimize and test it on the
known LJ104 minimum.
Adds no path.
Modifies 2 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
src/methods/lattice_search.rs

Commit 54818da1e54c6dc4e3bfce7a8c6a129f2af289d4
Subject: Record the plain DLS prediction of 20 of 30 from the lattice v2
recompute on seeds 3000-3029.
Adds no path.
Modifies 1 path.
docs/prereg/lj104-sota/PREDICTIONS.md

Commit d77d8eb557e4288c9eeee1ecd553e4de2525599a
Subject: Define the default panel as recommended at the paper budgets,
and teach the serial runner to execute those 24 seeds one at a time.
Adds no path.
Modifies 3 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/lj104_sota_runner.py
docs/prereg/lj104-sota/test_lj104_sota_runner.py

Commit cc0587fa24b9bd72c4e0399b942bd3a5c3c8cff5
Subject: Restore the LJ104 occupation test attribute that the requench
test displaced.
Adds no path.
Modifies 1 path.
src/methods/lattice_search.rs

Commit 380ceedd7fcd3db9bd0184fd987216822de684d3
Subject: Commit the untouched default-panel logs, geometries, traces,
spans, scored rows, and the start and end receipts.
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

Commit 3c1069acdc90306feda50f770489d892bd8ade85
Subject: Add the untouched default-panel stdout logs that the ignore
rule had left out.
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

Commit 9c072e5faa7f28d815b917c3d330f91190480829
Subject: Write the four default-panel floors from the raw logs, with the
per-seed outcomes beside them.
Adds 2 paths.
docs/prereg/lj104-sota/floors.md
docs/prereg/lj104-sota/write_floors.py
Modifies 1 path.
docs/prereg/lj104-sota/PREDICTIONS.md

Commit 6a813084d36962c1d61154eeafd1ea377d150390
Subject: Score every code, GMIN included, by lj_requench at E_known plus
1e-6, and count those calls apart from the search.
Adds 3 paths.
docs/prereg/lj104-sota/e_known.tsv
docs/prereg/lj104-sota/requench_panel.py
examples/lj_requench_xyz.rs
Modifies 2 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
src/methods/lattice_search.rs

Commit 5fb7368d5ea15cdf517c8bc48e2f56286ccb8b33
Subject: Re-quench the 24 saved structures and set the floors from those
energies and their separate call counts.
Adds 2 paths.
docs/prereg/lj104-sota/requench.md
docs/prereg/lj104-sota/requench.tsv
Modifies 2 paths.
docs/prereg/lj104-sota/PREDICTIONS.md
docs/prereg/lj104-sota/requench_panel.py
