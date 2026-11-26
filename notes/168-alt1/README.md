# Quiet surface walk, serial windows

Parent is develop 4feb0699d4e61c9531283e82196fcb60baf66932.
The tried diff is walk.diff. It is not applied in this commit.
Host label: cursor. One core, one BLAS thread, one Rayon thread.
Driver: h2h_seed, arm recommended.
Stops are the published minimum plus 0.001.
LJ38 budget 400000, seeds 2120-2131, stop -173.927427.
LJ75 budget 4000000, seeds 2120-2131, stop -397.491331.
LJ98 budget 4000000, seeds 2048-2083, stop -543.664361.

The walk kept the portfolio block until the chain had survived
a longer quiet stretch. A later block then ran for that many hops,
and ended sooner only when the current quiet passed that record.
A side ledger that reported a zero pair did not shorten the walk.
No potential call was added.

Every seed printed the same hit, calls and charged count on both binaries.

Concurrent outside-time ratio (tried/develop), median 1.0026, max 1.0950, min 0.6418.
The two arms ran at the same time on two cores, so that ratio is noisy.
Serial remeasure of LJ98 seed 2049 on one core: develop calls 200585 charged 203518 outside_s 3.781622; tried calls 200585 charged 203518 outside_s 3.687880; outside per call ratio 0.9752.

Cell summary (develop, and the tried arm matched it):

- LJ38 2120-2131: 12/12 hits, median calls on hits 10650.0, misses none.
- LJ75 2120-2131: 12/12 hits, median calls on hits 16834.5, misses none.
- LJ98 2048-2083: 33/36 hits, median calls on hits 983774.0, misses [2051, 2053, 2060].

LJ98 recommended does not install a surface portfolio, so this walk cannot move it.
The LJ38 and LJ75 hits finished before a later block was lengthened, so their calls did not move either.
The rule does not beat develop.

Per seed, calls to hit (budget if missed):

LJ38 2120:18715 2121:10646 2122:4227 2123:1515 2124:11684 2125:346 2126:2316 2127:10822 2128:11501 2129:10654 2130:11786 2131:8735

LJ75 2120:40859 2121:41010 2122:21272 2123:11077 2124:39847 2125:7126 2126:10399 2127:16898 2128:14576 2129:12351 2130:40427 2131:16771

LJ98 2048:1545274 2049:200585 2050:1262637 2051:4000000:miss 2052:2088500 2053:4000000:miss 2054:69295 2055:2022010 2056:251900 2057:40472 2058:2083419 2059:397135 2060:4000000:miss 2061:80532 2062:618288 2063:66798 2064:551152 2065:1309139 2066:1492126 2067:633607 2068:2644844 2069:1431261 2070:3436187 2071:1878185 2072:1331690 2073:2298280 2074:3239284 2075:3720507 2076:480256 2077:3183 2078:2050092 2079:983774 2080:123911 2081:393810 2082:247323 2083:78953

