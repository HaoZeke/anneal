Copyright (c) anneal developers

# Held stream, then a forked adaptive escape

Motif-free `recommended` against opt-in `recommended_adaptive` on the same
binary. The adaptive arm keeps the incumbent on the single-chain proposal
stream, then gives a fork of the unused stream the calls that remain.
Seeds 2000-2011, host `bench`, serial `taskset -c 0`, paper call budgets,
no wall cap.
code_revision 9ac91a41695838b7e43c922315834cc4d1d12344.
Misses are censored at their call budget.

| system | budget | base hits | fix hits | base med calls | fix med calls | base wall s | fix wall s | Fisher p | calls MW p | wall MW p | wall ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m38r10 | 400,000 | 10/12 | 10/12 | 50144.0 | 50144.0 | 0.436 | 0.434 | 1 | 1 | 0.930987 | 0.997 |
| m38r14 | 400,000 | 8/12 | 8/12 | 156127.5 | 156127.5 | 1.264 | 1.264 | 1 | 1 | 0.885234 | 1.000 |
| m55r10 | 1,000,000 | 12/12 | 12/12 | 298800.5 | 298800.5 | 4.954 | 4.895 | 1 | 1 | 0.885234 | 0.988 |
| m55r14 | 1,000,000 | 8/12 | 8/12 | 694593.5 | 898109.5 | 11.311 | 14.577 | 1 | 0.701965 | 0.402504 | 1.289 |
| m75r6 | 4,000,000 | 10/12 | 10/12 | 1047864.0 | 897955.0 | 32.717 | 28.451 | 1 | 1 | 0.83986 | 0.870 |
| m75r14 | 4,000,000 | 7/12 | 9/12 | 2102774.5 | 1405811.5 | 64.153 | 42.858 | 0.666846 | 0.317128 | 0.312321 | 0.668 |
| m80r6 | 4,000,000 | 12/12 | 12/12 | 65487.0 | 65487.0 | 2.430 | 2.431 | 1 | 1 | 0.930987 | 1.000 |
| m80r14 | 4,000,000 | 8/12 | 11/12 | 2201178.5 | 2095247.5 | 74.110 | 71.624 | 0.31677 | 0.727838 | 0.795012 | 0.966 |
| lj38 | 400,000 | 12/12 | 12/12 | 30621.5 | 30621.5 | 0.158 | 0.157 | 1 | 1 | 1 | 0.990 |
| lj55 | 1,000,000 | 12/12 | 12/12 | 5722.0 | 5722.0 | 0.064 | 0.064 | 1 | 1 | 1 | 1.001 |
| lj75 | 4,000,000 | 9/12 | 9/12 | 2500360.0 | 2272729.0 | 37.186 | 35.124 | 1 | 0.748925 | 1 | 0.945 |
| lj98 | 4,000,000 | 9/12 | 12/12 | 2919346.0 | 1467510.0 | 69.592 | 37.995 | 0.217391 | 0.193349 | 0.260236 | 0.546 |
| lj104 | 4,000,000 | 8/12 | 10/12 | 1945461.5 | 2026917.0 | 53.209 | 55.944 | 0.640405 | 0.884306 | 0.97697 | 1.051 |

Gate result: **passed**.
A hit-count decrease rejects even when Fisher is not individually significant.
Calls and wall use two-sided, tie- and continuity-corrected Mann-Whitney tests.
