# Shard a

Cells lj38, lj55, lj75, and m80r6 at the 2026-11-19 GMIN snapshot. Seeds 2000-2011. Order abba: 6 anneal-first seeds and 6 GMIN-first seeds in each cell. Budgets are the harness paper budgets, lj38 400000, lj55 1000000, lj75 4000000, and m80r6 4000000. Each run is a fresh process, one thread, pinned to core 0.

SVNREV 9cbec1a446352bc9ef064bd2b1a5767194655919. Tarball sha256 ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776. Anneal eeec8e2c216ae1506768bca8e29846c1c039ca3b. Harness eeec8e2c216ae1506768bca8e29846c1c039ca3b. machine_id_sha256 3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873.

h2h_gmin_proof sha256 235e2409818e0e36f6ae8ab12647d5cae32c8600c7ec608352e10f356bf2c257. h2h_metal sha256 8d0bd481a1c481e09d11feca2c5a73049faccbf5eb2ef3834cdb656a1831788b. h2h_morse sha256 f53276984c6f34dccbce87268f809b24c9fe12b294f87576f9f8f43d7f2ea5ce. These three match the expected prefixes. GMIN sha256 24939393456c1dbe9c20d515b981f33c86d43b79b01de369db231edc8dbe5e68, against the expected prefix dcd3ce8c. The binary records its compilation time, Thu Nov 19 02:56:20 PM UTC 2026.

A hit is a quenched energy at or below the reference plus 0.001. Fisher p is the two-sided exact test. Median calls and median wall are Hyndman-Fan type 7, and a miss is entered at the budget. Mann-Whitney uses the harness two-sided normal approximation with the tie and continuity corrections on calls to the minimum, ranking a miss above every hit. U is the anneal sample. An arm reaches the reference energy when a seed is at or below that reference; the hit window is the separate 0.001 rule.

| Cell | Reference | Anneal hits | GMIN hits | Fisher p | Anneal median calls | GMIN median calls | MW U | MW p | Anneal median wall s | GMIN median wall s | Anneal at or below reference | GMIN at or below reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lj38 | -173.928427 | 12/12 | 7/12 | 6/161 | 127839.5 | 389399.0 | 36 | 0.039536094115483866 | 0.6400495 | 3.4269500 | 0/12 | 0/12 |
| lj55 | -279.248470 | 12/12 | 12/12 | 1 | 5722.0 | 12764.5 | 33 | 0.026229387834857038 | 0.0517635 | 0.2093755 | 12/12 | 0/12 |
| lj75 | -397.492331 | 5/12 | 0/12 | 6/161 | 4000000.0 | 4000000.0 | 42 | 0.016473155805582677 | 61.0768230 | 117.7440000 | 0/12 | 0/12 |
| m80r6 | -378.333471 | 12/12 | 12/12 | 1 | 62677.0 | 268934.0 | 52 | 0.260236202617189 | 2.4563560 | 9.9766800 | 6/12 | 0/12 |

lj38 anneal is inside the hit window on 12/12 and above the reference on 12/12. lj38 GMIN is inside the window on 7/12 and above the reference on 12/12. lj55 anneal is at or below the reference on 12/12. lj55 GMIN is inside the window on 12/12 and above the reference on 12/12. lj75 anneal is inside the window on 5/12 and above the reference on 12/12; both medians sit on the budget because at least half the seeds miss, while the rank test still places those five hits below the misses. lj75 GMIN is outside the window on 12/12, lowest -396.282243226 against -397.492331. m80r6 anneal is at or below the reference on 6/12 and inside the window on 12/12. m80r6 GMIN is inside the window on 12/12 and above the reference on 12/12.
