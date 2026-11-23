# generic-packings-r3 full-panel gate

The r2 rows and verdict remain unchanged in their original directory.
They are superseded because r2 polled RssAnon externally, used only
the same-layout A/A band, and obtained packing counters by replay.

Verdict: **PASS**.
Rows: 2304. Sessions: 20261121T203243Z.
App-keyed machine hash: 712f047aa784a423bba86c25a46cd2c686922b98b1dbc63b335a5ddfef4647a6.

RssAnon is sampled by a thread inside each timed process. The gate
uses its per-cell median. G passes when G/D is no greater than the
largest ratio from five D layouts and D2, or when the median absolute
difference is at most 8 kB. VmHWM, ru_maxrss, and external polling
are report-only. A G row with offered=0 invalidates its cell.

The counter is offered/hit. Offered counts timed G runs that entered
the new generic-packing path; hit counts those whose packing probe
itself reached the cell reference. Thus 12/0 means the new path ran
in all 12 replicates but none of its initial candidates hit directly.

P is historical revision 628773d6, before the regression attribution.
It is not D with only placement changed. L2 through L5 are the
placement-only controls; D is the first layout and D2 its independent
same-layout build. P can therefore have different hit trajectories.

| cell | hits D/D2/G/P | outside G/D (band) | RssAnon G/D (band; delta kB) | offered/hit | status |
| --- | --- | ---: | ---: | ---: | --- |
| lj38 | 12/12/12/12 | 0.937988 (1.002419) | 0.130372 (1.000000; -3602.0) | 12/12 | PASS |
| lj55 | 12/12/12/12 | 0.608471 (1.017916) | 0.328537 (1.007194; -1120.0) | 12/12 | PASS |
| lj75 | 5/5/5/12 | 0.993791 (1.001961) | 0.999466 (1.000067; -64.0) | 12/0 | PASS |
| lj98 | 7/7/7/10 | 0.991542 (1.001884) | 1.000335 (1.001792; 40.0) | 12/0 | PASS |
| lj102 | 6/6/6/9 | 0.995268 (1.000052) | 0.997755 (1.000000; -316.0) | 12/0 | PASS |
| lj103 | 6/6/6/8 | 0.989956 (1.000249) | 0.999959 (1.000571; -6.0) | 12/0 | PASS |
| lj104 | 5/5/5/9 | 0.995228 (1.000183) | 1.000119 (1.000901; 18.0) | 12/0 | PASS |
| lj150 | 8/8/8/5 | 0.999022 (1.002414) | 1.000086 (1.000471; 8.0) | 12/0 | PASS |
| au55 | 4/4/4/6 | 1.004403 (1.000020) | 1.000000 (1.000000; 0.0) | 12/0 | PASS |
| m55r6 | 12/12/12/12 | 0.545817 (1.003107) | 0.250454 (1.000907; -1652.0) | 12/12 | PASS |
| m55r14 | 9/9/9/7 | 0.997103 (1.009920) | 0.998667 (1.000148; -18.0) | 12/0 | PASS |
| m80r6 | 12/12/12/2 | 0.928075 (1.011888) | 0.997508 (1.000000; -12.0) | 12/0 | PASS |
| m80r14 | 6/6/6/9 | 1.008842 (1.003616) | 0.999737 (1.000452; -50.0) | 12/0 | PASS |
| m38r14 | 4/4/12/12 | 0.793663 (1.000962) | 0.072464 (1.000000; -6912.0) | 12/12 | PASS |
| m38r6 | 12/12/12/12 | 0.508413 (1.009625) | 0.237555 (1.001747; -1746.0) | 12/12 | PASS |
| sc126-38 | 12/12/12/12 | 1.070708 (1.007959) | 0.312096 (1.000923; -1490.0) | 12/12 | PASS |
| sc126-55 | 12/12/12/12 | 0.523505 (1.001519) | 0.394077 (1.002278; -1064.0) | 12/12 | PASS |
| sc126-80 | 5/5/5/6 | 0.992475 (1.009051) | 0.995547 (1.000061; -146.0) | 12/0 | PASS |
| sc108-38 | 7/7/12/12 | 0.783758 (1.005649) | 0.101843 (1.000604; -5944.0) | 12/12 | PASS |
| sc108-55 | 5/5/5/6 | 0.959444 (1.010966) | 0.998317 (1.000073; -46.0) | 12/0 | PASS |
| sc108-80 | 3/3/3/0 | 0.981455 (1.000005) | 0.996259 (1.000052; -144.0) | 12/0 | PASS |
| sc96-38 | 12/12/12/12 | 0.858225 (1.003049) | 0.409201 (1.001211; -976.0) | 12/12 | PASS |
| sc96-55 | 12/12/12/12 | 0.557653 (1.022861) | 0.355236 (1.000000; -1256.0) | 12/12 | PASS |
| sc96-80 | 7/7/7/7 | 0.987354 (1.002804) | 0.998578 (1.000043; -66.0) | 12/0 | PASS |
