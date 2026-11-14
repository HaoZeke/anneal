# Mean-SOAP collisions

Twelve quenched minima per system, each shaken and quenched again. A pair
counts when its energy gap exceeds both 0.1 and 100 times the re-quench
energy width. The taboo calls two mean-SOAP spectra one minimum at
distance 0.02. Rows are in `rows.csv`.

| system | pairs | mean SOAP under 0.02 | inside the re-quench width |
|---|---:|---:|---:|
| Au55 | 55 | 55 | 0 |
| LJ38 | 15 | 0 | 0 |
| LJ75 | 3 | 0 | 0 |
| TIP4P15 | 54 | 21 | 0 |

On Au55 every one of the 28 pairs more than 0.5 eV apart is still inside
0.02. The re-quench width of that spectrum is 2.32e-4, and the closest
distinct pair is at 2.06e-3. A radial spectrum, a pair-distance histogram
and sorted per-atom SOAP, each compared inside its own re-quench width,
call none of those pairs the same minimum.

## What that does to the Au55 search

The radial spectrum with a cutoff of twice the widest re-quench distance
measured in the run was tried as the taboo, the stored population and the
repeated-minimum stall. On Au55 Gupta at 1e6 calls, hostname bench, the
four seeds that hit on the paper preset all miss:

| seed | paper preset | finer fingerprint | best |
|---:|---|---|---:|
| 2003 | hit at 196741 | miss | −190.98001732 |
| 2004 | hit at 85523 | miss | −190.98001732 |
| 2005 | hit at 838920 | miss | −190.98001732 |
| 2006 | hit at 167719 | miss | −190.98001732 |
| 2000 | miss | miss | −191.00616498 |

Separating minima that the mean spectrum merges, and refusing a
non-improving return to a stored one, turns those hits into the
decahedron. The coarse taboo's merging is doing useful work on this
landscape. The paper preset leaves the finer test off.
