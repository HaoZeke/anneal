# Au55 worker scaling

Shared communicating ensemble, budget 1000000 oracle calls,
seeds 2000-2011. nproc 8. CPU model Intel(R) Xeon(R) Processor.
Revision c4d891ec00314f06d226ca8de2b1b915a9c6cd74.

Wall time to hit is the median over hits. Calls to hit enter a
miss as budget + 1. Efficiency is T1/(p Tp) from the wall-to-hit
medians. It is undefined when a rung has no hit, and it is omitted
when oversubscribed is true. Intervals are bootstrap 95%
percentiles, 10000 resamples, seed 20261123. A single hit makes
that interval a point. Fisher and Mann-Whitney compare each rung
with 1 worker.

| workers | hits | hit CI | calls median | calls CI | wall median s | wall CI | efficiency | efficiency CI | Fisher p | MWU p | oversubscribed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1/12 | [0, 3] | 1000001 | [1000001, 1000001] | 53.336 | [53.336, 53.336] | 1 |  |  |  | no |
| 2 | 0/12 | [0, 0] | 1000001 | [1000001, 1000001] |  |  | undefined |  | 1 | 0.3593 | no |
| 4 | 1/12 | [0, 3] | 1000001 | [1000001, 1000001] | 47.269 | [47.269, 47.269] | 0.2821 | [0.2821, 0.2821] | 1 | 1 | no |
| 8 | 2/12 | [0, 5] | 1000001 | [1000001, 1000001] | 115.439 | [106.252, 124.626] | 0.0578 | [0.0535, 0.0627] | 1 | 0.6513 | no |
| 16 | 0/12 | [0, 0] | 1000001 | [1000001, 1000001] |  |  | omitted |  | 1 | 0.3593 | yes |

Pins read back from the affinity mask:

- 1 workers: cores 0, wall-to-hit samples 1
- 2 workers: cores 0-1, wall-to-hit samples 0
- 4 workers: cores 0-3, wall-to-hit samples 1
- 8 workers: cores 0-7, wall-to-hit samples 2
- 16 workers: cores 0-7, wall-to-hit samples 0
