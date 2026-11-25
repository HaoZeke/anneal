# Chain ensemble results

Seeds 2000-2011. A miss counts as budget + 1 in the calls-to-hit
median and in the Mann-Whitney test. Bootstrap intervals are 95%
percentiles from 10000 resamples, seed 20261123.
Fisher and Mann-Whitney compare comm against indep, then comm
against single. Both tests are two-sided.

| system | arm | hits | hit rate | rate CI | median calls | calls CI | ledger sum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tip4p20 | comm | 0/12 | 0.0000 | [0.0000, 0.0000] | 2000001.0 | [2000001.0, 2000001.0] | 24000000 |
| tip4p20 | indep | 0/12 | 0.0000 | [0.0000, 0.0000] | 2000001.0 | [2000001.0, 2000001.0] | 24000000 |
| tip4p20 | single | 0/12 | 0.0000 | [0.0000, 0.0000] | 2000001.0 | [2000001.0, 2000001.0] | 24000000 |
| au55 | comm | 0/12 | 0.0000 | [0.0000, 0.0000] | 1000001.0 | [1000001.0, 1000001.0] | 12000000 |
| au55 | indep | 0/12 | 0.0000 | [0.0000, 0.0000] | 1000001.0 | [1000001.0, 1000001.0] | 12000000 |
| au55 | single | 3/12 | 0.2500 | [0.0000, 0.5000] | 1000001.0 | [988506.5, 1000001.0] | 12000000 |

## Tests

- tip4p20 comm vs indep: Fisher p 1.0000, Mann-Whitney p 1.0000
- tip4p20 comm vs single: Fisher p 1.0000, Mann-Whitney p 1.0000
- au55 comm vs indep: Fisher p 1.0000, Mann-Whitney p 1.0000
- au55 comm vs single: Fisher p 0.2174, Mann-Whitney p 0.0788

The openconf ligand is dropped from this panel.
The openconf ligand (rq3) was not run: rgpot has no MMFF potential, and run_ensemble needs an energy and gradient.
