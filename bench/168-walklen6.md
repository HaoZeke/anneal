# 168-walklen6

Negative. No measured pairing of the plain quench and the diameter-penalty quench kept every develop hit. This commit does not change the search.

Develop binary sha256 `43780d327a1933251afb7902b7d69d47f511555bc483358ca61274eef9c34554`, built from `4feb0699d4e61c9531283e82196fcb60baf66932`.
Candidate binary sha256 `db9b2ff5fbb24e4786f353d41e1037dfcaed11657d66da268c01f61dabb4ddb3`. It is a local build and is not this commit.
Machine: Intel(R) Xeon(R) Processor, 8 cores.

The candidate keeps the diameter penalty as a Thompson arm at every size. An opening restraint quench saves the proposal. If the next quench does not hit, the opening plain state is restored and the plain arm is held. A later restraint block is allowed the preceding block's call count, then the same restore runs.

Stops: LJ98 `-543.664361`, LJ75 `-397.491331`, LJ38 `-173.927427`. A hit is the first call at or below the stop. `miss` is the best energy at the budget.

| system | seed | budget | A develop | B candidate |
|---|---:|---:|---|---|
| lj75 | 2120 | 41010 | miss -396.2822488154 | not run |
| lj75 | 2121 | 41010 | miss -396.2822487902 | not run |
| lj75 | 2122 | 41010 | 21272 | not run |
| lj75 | 2123 | 41010 | 11077 | not run |
| lj75 | 2124 | 41010 | 39847 | not run |
| lj75 | 2125 | 41010 | 7126 | not run |
| lj75 | 2126 | 41010 | 10399 | not run |
| lj75 | 2127 | 41010 | 16898 | not run |
| lj75 | 2128 | 41010 | 14576 | not run |
| lj75 | 2129 | 41010 | 12351 | not run |
| lj75 | 2130 | 41010 | 40427 | not run |
| lj75 | 2131 | 41010 | 16771 | not run |
| lj38 | 2120 | 400000 | 18715 | not run |
| lj38 | 2121 | 400000 | 10646 | not run |
| lj38 | 2122 | 400000 | 4227 | not run |
| lj38 | 2123 | 400000 | 1515 | not run |
| lj38 | 2124 | 400000 | 11684 | not run |
| lj38 | 2125 | 400000 | 346 | not run |
| lj38 | 2126 | 400000 | 2316 | not run |
| lj38 | 2127 | 400000 | 10822 | not run |
| lj38 | 2128 | 400000 | 11501 | not run |
| lj38 | 2129 | 400000 | 10654 | not run |
| lj38 | 2130 | 400000 | 11786 | not run |
| lj38 | 2131 | 400000 | 8735 | not run |
| lj98 | 2048 | 4000000 | 1545274 | not run |
| lj98 | 2049 | 4000000 | 200585 | not run |
| lj98 | 2050 | 4000000 | 1262637 | not run |
| lj98 | 2051 | 4000000 | miss -543.6429572250 | not run |
| lj98 | 2052 | 4000000 | 2088500 | not run |
| lj98 | 2053 | 4000000 | miss -543.6429572246 | not run |
| lj98 | 2054 | 4000000 | 69295 | not run |
| lj98 | 2055 | 4000000 | 2022010 | not run |
| lj98 | 2056 | 4000000 | 251900 | not run |
| lj98 | 2057 | 4000000 | 40472 | not run |
| lj98 | 2058 | 4000000 | 2083419 | not run |
| lj98 | 2059 | 4000000 | 397135 | not run |
| lj98 | 2060 | 4000000 | miss -543.6429572158 | not run |
| lj98 | 2061 | 4000000 | 80532 | not run |
| lj98 | 2062 | 4000000 | 618288 | not run |
| lj98 | 2063 | 4000000 | 66798 | not run |
| lj98 | 2064 | 4000000 | 551152 | not run |
| lj98 | 2065 | 4000000 | 1309139 | not run |
| lj98 | 2066 | 4000000 | 1492126 | not run |
| lj98 | 2067 | 4000000 | 633607 | not run |
| lj98 | 2068 | 4000000 | 2644844 | not run |
| lj98 | 2069 | 4000000 | 1431261 | not run |
| lj98 | 2070 | 4000000 | 3436187 | not run |
| lj98 | 2071 | 4000000 | 1878185 | not run |
| lj98 | 2072 | 4000000 | 1331690 | not run |
| lj98 | 2073 | 4000000 | 2298280 | not run |
| lj98 | 2074 | 4000000 | 3239284 | not run |
| lj98 | 2075 | 4000000 | 3720507 | not run |
| lj98 | 2076 | 4000000 | 480256 | not run |
| lj98 | 2077 | 4000000 | 3183 | not run |
| lj98 | 2078 | 4000000 | 2050092 | not run |
| lj98 | 2079 | 4000000 | 983774 | not run |
| lj98 | 2080 | 4000000 | 123911 | not run |
| lj98 | 2081 | 4000000 | 393810 | not run |
| lj98 | 2082 | 4000000 | 247323 | not run |
| lj98 | 2083 | 4000000 | 78953 | not run |
| lj75 | 2120 | 50000 | 40859 | 40859 |
| lj75 | 2121 | 50000 | 41010 | 41010 |
| lj38 | 2000 | 8000 | 287 | 287 |
| lj38 | 1 | 50000 | 17204 | 17204 |
| lj98 | 2077 | 25000 | 3183 | 4102 |
| lj98 | 2083 | 100000 | 78870 | 79836 |
| lj98 | 2054 | 120000 | 69255 | miss -540.8015244054 |
| lj98 | 2079 | 1100000 | 983774 | miss -543.6429572121 |
| lj98 | 2078 | 2200000 | 2050092 | miss -543.6429572170 |

The five LJ98 rows below the 4000000 block were run on both binaries at the budget in that row. The 4000000 develop calls are not the same run: seed 2054 hits at 69295 there and at 69255 at budget 120000; seed 2083 hits at 78953 there and at 78870 at budget 100000.

Lost against develop at the same budget: LJ98 2054 (develop 69255, candidate miss, best -540.8015244054), LJ98 2078 (develop 2050092, candidate miss, best -543.6429572170), LJ98 2079 (develop 983774, candidate miss, best -543.6429572121).

LJ75 seeds 2120 and 2121 miss on develop when the run budget is 41010 (best -396.2822488154 and -396.2822487902). The same binary at budget 50000 hits at 40859 and 41010. The candidate matches those two calls.

Seeds on the acceptance list that were not run on the candidate: LJ98 2048-2053, 2055-2076, 2080-2082; LJ75 2122-2131; LJ38 2120-2131. The three losses above already fail the acceptance set.

## Why walk-length changes stop here

Walklen4, walklen5 and this trial all lose develop hits on LJ98.
Walklen4's 4e6 comparison loses 2054, 2055, 2056, 2066, 2067, 2078,
2079 and 2083. Walklen5 found no size-free change that keeps those
LJ98 hits and the LJ38 seed 2000 call. This trial still loses 2054,
2078 and 2079.

Those three open on the plain surface. Through the first 2000 calls
every full quench on 2054, 2078 and 2079 is surface 0. The first
restraint block starts at call 8911, 8654 and 9267. Seed 2077, which
this candidate still hits, opens on the restraint: surface 1 at call
336. Returning to the plain opening after the later restraint block
does not reach the hit develop records. 2078 and 2079 then end at
-543.642957, the energy develop also reaches on 2051, 2053 and 2060,
which develop misses inside 4000000 calls. 2054 ends higher, at
-540.8015244054.

LJ75 2120 and 2121 are a budget effect. With budget 50000, develop
hits at 40859 and 41010, and the candidate matches those calls. With
budget 41010 the same develop binary misses, best -396.2822488154
and -396.2822487902. The shorter budget changes the walk. It does
not cut off a hit that has already been reached.
