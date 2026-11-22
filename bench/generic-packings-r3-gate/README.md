# generic-packings-r3 full-panel gate

The r2 rows and verdict remain unchanged in their original directory.
They are superseded because r2 polled RssAnon externally, used only
the same-layout A/A band, and obtained packing counters by replay.

Verdict: **INCONCLUSIVE**.
Rows: 680. Sessions: 20261121T203243Z.
App-keyed machine hash: 712f047aa784a423bba86c25a46cd2c686922b98b1dbc63b335a5ddfef4647a6.

RssAnon is sampled by a thread inside each timed process. The gate
uses its per-cell median. G passes when G/D is no greater than the
largest ratio from five D layouts and D2, or when the median absolute
difference is at most 8 kB. VmHWM, ru_maxrss, and external polling
are report-only. A G row with offered=0 invalidates its cell.

| cell | hits D/D2/G/P | outside G/D (band) | RssAnon G/D (band; delta kB) | offered/hit | status |
| --- | --- | ---: | ---: | ---: | --- |
| lj38 | 12/12/12/12 | 0.937988 (1.002419) | 0.130372 (1.000000; -3602.0) | 12/12 | PASS |
| lj55 | 12/12/12/12 | 0.608471 (1.017916) | 0.328537 (1.007194; -1120.0) | 12/12 | PASS |
| lj75 | 5/5/5/12 | 0.993791 (1.001961) | 0.999466 (1.000067; -64.0) | 12/0 | PASS |
| lj98 | 7/7/7/10 | 0.991542 (1.001884) | 1.000335 (1.001792; 40.0) | 12/0 | PASS |
| lj102 | 6/6/6/9 | 0.995268 (1.000052) | 0.997755 (1.000000; -316.0) | 12/0 | PASS |
| lj103 | 6/6/6/8 | 0.989956 (1.000249) | 0.999959 (1.000571; -6.0) | 12/0 | PASS |
| lj104 | 5/5/5/9 | 0.995228 (1.000183) | 1.000119 (1.000901; 18.0) | 12/0 | PASS |
| lj150 | incomplete | - | - | - | INCONCLUSIVE |
| au55 | incomplete | - | - | - | INCONCLUSIVE |
| m55r6 | incomplete | - | - | - | INCONCLUSIVE |
| m55r14 | incomplete | - | - | - | INCONCLUSIVE |
| m80r6 | incomplete | - | - | - | INCONCLUSIVE |
| m80r14 | incomplete | - | - | - | INCONCLUSIVE |
| m38r14 | incomplete | - | - | - | INCONCLUSIVE |
| m38r6 | incomplete | - | - | - | INCONCLUSIVE |
| sc126-38 | incomplete | - | - | - | INCONCLUSIVE |
| sc126-55 | incomplete | - | - | - | INCONCLUSIVE |
| sc126-80 | incomplete | - | - | - | INCONCLUSIVE |
| sc108-38 | incomplete | - | - | - | INCONCLUSIVE |
| sc108-55 | incomplete | - | - | - | INCONCLUSIVE |
| sc108-80 | incomplete | - | - | - | INCONCLUSIVE |
| sc96-38 | incomplete | - | - | - | INCONCLUSIVE |
| sc96-55 | incomplete | - | - | - | INCONCLUSIVE |
| sc96-80 | incomplete | - | - | - | INCONCLUSIVE |
