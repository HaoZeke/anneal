# LJ98 and LJ102

Session `20261119T150112Z`. Seeds 2000-2011 (n=12), order abba, budget 4000000 calls. GMIN snapshot 2026-11-19, SVNREV `9cbec1a446352bc9ef064bd2b1a5767194655919`, tarball `ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776`. Anneal and harness `eeec8e2c216ae1506768bca8e29846c1c039ca3b`. Machine `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`.

Each cell has 6 anneal-first seeds and 6 GMIN-first seeds.

Release binaries: `h2h_gmin_proof` `235e2409818e0e36f6ae8ab12647d5cae32c8600c7ec608352e10f356bf2c257`, `h2h_metal` `8d0bd481a1c481e09d11feca2c5a73049faccbf5eb2ef3834cdb656a1831788b`, `h2h_morse` `f53276984c6f34dccbce87268f809b24c9fe12b294f87576f9f8f43d7f2ea5ce`. GMIN `d3542eca295d7e1343860cd20a780943fcd269bb5293344f5371cf81d68fa629`, which does not match the expected prefix `dcd3ce8c`.

A hit is a quenched energy at or below the reference plus 0.001. Fisher p is the two-sided exact test. Median calls are Hyndman-Fan type 7 of the calls to the minimum, with a miss entered at the budget. Mann-Whitney p is the two-sided normal approximation with the tie and continuity corrections on those calls, ranking a miss above every hit. Median wall is type 7 of the run wall. An arm reaches the reference when its lowest energy is a hit.

## LJ98

Reference -543.665361.

| Arm | Hits | Median calls | Median wall (s) | Reaches reference |
| --- | --- | --- | --- | --- |
| anneal | 7/12 | 3166811.5 | 82.51394 | yes |
| GMIN | 0/12 | 4000000 | 177.2575 | no |

Fisher p = 2/437 = 0.004576659038901602. Mann-Whitney p on calls = 0.002855585486096581 (U = 30).

## LJ102

Reference -569.363652.

| Arm | Hits | Median calls | Median wall (s) | Reaches reference |
| --- | --- | --- | --- | --- |
| anneal | 6/12 | 3256052.5 | 83.4353175 | yes |
| GMIN | 5/12 | 4000000 | 185.1875 | yes |

Fisher p = 1. Mann-Whitney p on calls = 0.9748991155119652 (U = 73).
