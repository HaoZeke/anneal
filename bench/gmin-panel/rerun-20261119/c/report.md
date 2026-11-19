# LJ103 and LJ104

GMIN snapshot 2026-11-19, SVNREV 9cbec1a446352bc9ef064bd2b1a5767194655919, tarball sha256 ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776. Anneal eeec8e2c216ae1506768bca8e29846c1c039ca3b. Harness eeec8e2c216ae1506768bca8e29846c1c039ca3b. Machine 3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873.

Seeds 2000-2011 (n=12). The run order is abba: six anneal-first seeds (2000, 2002, 2004, 2006, 2008, 2010) and six GMIN-first seeds (2001, 2003, 2005, 2007, 2009, 2011) on each cell. The call budget is 4000000. A hit is a quenched energy at or below the reference plus 0.001. Median calls and median wall time are Hyndman-Fan type 7. Fisher p is the two-sided exact test. Mann-Whitney p is the two-sided normal approximation with the tie and continuity corrections, on calls to the minimum, with a full-budget miss ranked above every hit.

Release binary sha256: GMIN b2d448da772879c57dfcde5002a4cf991e4c4903e684e562daa65e3bdbf6216e h2h_gmin_proof 235e2409818e0e36f6ae8ab12647d5cae32c8600c7ec608352e10f356bf2c257 h2h_metal 8d0bd481a1c481e09d11feca2c5a73049faccbf5eb2ef3834cdb656a1831788b h2h_morse f53276984c6f34dccbce87268f809b24c9fe12b294f87576f9f8f43d7f2ea5ce.
The GMIN binary sha256 does not start with dcd3ce8c. h2h_gmin_proof, h2h_metal and h2h_morse match 235e2409, 8d0bd481 and f5327698.

## LJ103

Reference -575.766131. anneal 6/12, GMIN 3/12, Fisher p = 2974/7429. Median calls 3905072.5 and 4000000, Mann-Whitney p = 0.118784. Median wall time 103.0235545 s and 191.4055 s. anneal reaches the reference (best -575.76613087). GMIN reaches the reference (best -575.766095794).

## LJ104

Reference -582.086642. anneal 5/12, GMIN 6/12, Fisher p = 1. Median calls 4000000 and 3696118, Mann-Whitney p = 0.874991. Median wall time 107.0749545 s and 183.704 s. anneal reaches the reference (best -582.08664207). GMIN reaches the reference (best -582.086601194).
