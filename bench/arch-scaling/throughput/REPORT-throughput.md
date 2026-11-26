# Throughput scaling

Shared communicating ensemble, one cell at a time, budget
1000000 oracle calls spent to exhaustion. Replicas are operating-system
threads from `std::thread::scope`, not a rayon pool. Bootstrap intervals
use 10000 resamples from `random.Random(20261123)`.

Revision `7af60bec62655f1cc558cbaceee3eec7bd186ab3`.
nproc 8. CPU Intel(R) Xeon(R) Processor.

| system | workers | n | wall median s | wall median CI | wall geomean s | wall geomean CI | calls/s median | calls/s median CI | calls/s geomean | calls/s geomean CI | efficiency | efficiency CI | hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| au55 | 1 | 12 | 94.640 | [87.590, 104.240] | 95.022 | [89.413, 101.284] | 10566.5 | [9599.8, 11435.5] | 10523.9 | [9886.2, 11170.9] | 1 |  | 1/12 |
| au55 | 2 | 12 | 58.149 | [54.911, 63.502] | 59.735 | [56.052, 64.374] | 17197.4 | [15747.5, 18238.5] | 16740.5 | [15609.1, 17869.0] | 0.8138 | [0.7310, 0.8724] | 0/12 |
| au55 | 4 | 12 | 60.721 | [58.339, 68.474] | 65.494 | [59.197, 74.811] | 16473.6 | [14604.8, 17147.1] | 15268.7 | [13363.2, 16904.8] | 0.3896 | [0.3448, 0.4109] | 1/12 |
| au55 | 8 | 12 | 104.499 | [98.896, 111.171] | 109.069 | [100.849, 120.907] | 9569.6 | [9023.5, 10114.1] | 9168.5 | [8297.1, 9916.0] | 0.1132 | [0.1036, 0.1252] | 2/12 |
| lj75 | 1 | 12 | 255.706 | [230.766, 296.714] | 254.389 | [228.712, 280.087] | 3912.3 | [3370.7, 4346.6] | 3931.0 | [3578.3, 4373.2] | 1 |  | 3/12 |
| lj75 | 2 | 12 | 145.217 | [135.279, 157.797] | 147.209 | [138.114, 157.481] | 6887.4 | [6342.9, 7392.8] | 6793.1 | [6345.0, 7234.8] | 0.8804 | [0.7988, 0.9930] | 2/12 |
| lj75 | 4 | 12 | 93.222 | [87.177, 103.084] | 94.263 | [88.193, 100.743] | 10729.0 | [9710.9, 11473.5] | 10608.6 | [9916.6, 11330.8] | 0.6857 | [0.6253, 0.7912] | 3/12 |
| lj75 | 8 | 12 | 70.075 | [61.369, 75.301] | 68.542 | [64.133, 73.085] | 14270.7 | [13280.1, 16301.8] | 14589.7 | [13679.9, 15608.9] | 0.4561 | [0.4103, 0.5610] | 1/12 |

## Deviations

Every stored cell spent the budget. The three ledger totals agree on each row.

## Pins and hits

- au55 p=1: pin `0`, hits 1/12, oversubscribed no
- au55 p=2: pin `0-1`, hits 0/12, oversubscribed no
- au55 p=4: pin `0-3`, hits 1/12, oversubscribed no
- au55 p=8: pin `0-7`, hits 2/12, oversubscribed no
- lj75 p=1: pin `0`, hits 3/12, oversubscribed no
- lj75 p=2: pin `0-1`, hits 2/12, oversubscribed no
- lj75 p=4: pin `0-3`, hits 3/12, oversubscribed no
- lj75 p=8: pin `0-7`, hits 1/12, oversubscribed no

Hits use the reference plus `1e-3` and do not enter the intervals.
Efficiency is `T1 / (p Tp)` from median wall times of paired seeds.
A cell with `p` above nproc is omitted from efficiency.
