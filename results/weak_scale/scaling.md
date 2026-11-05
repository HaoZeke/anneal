# Gustafson and Karp-Flatt

Host CPU: Intel Xeon Processor, 8 physical cores.
GPU: none. CUDA: none.
Worker counts are 1, 2, 4, and 8, each at most the physical core count.
The walls are the medians in summary.csv.
eta(p) is the weak efficiency T(1) / T(p).
Gustafson scaled speedup is S_G(p) = p * eta(p).
Gustafson serial fraction is s_G(p) = p * (1 - eta(p)) / (p - 1) for p > 1.
The Karp-Flatt serial fraction is e(p) = (1 / S_G(p) - 1 / p) / (1 - 1 / p).
The fixed-work speedup that uses the Karp-Flatt fraction is
S_A(p) = 1 / (e(p) + (1 - e(p)) / p).

## pt

Host CPU: Intel Xeon Processor, 8 physical cores.

| family | workers | cores | eta | Gustafson S_G | Gustafson s_G | Karp-Flatt | fixed-work S |
|---|---:|---:|---:|---:|---:|---:|---:|
| pt | 1 | 8 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| pt | 2 | 8 | 0.938857 | 1.877714 | 0.122286 | 0.065125 | 1.877714 |
| pt | 4 | 8 | 0.980135 | 3.920542 | 0.026486 | 0.006756 | 3.920542 |
| pt | 8 | 8 | 0.895303 | 7.162420 | 0.119654 | 0.016706 | 7.162420 |

## population

Host CPU: Intel Xeon Processor, 8 physical cores.

| family | workers | cores | eta | Gustafson S_G | Gustafson s_G | Karp-Flatt | fixed-work S |
|---|---:|---:|---:|---:|---:|---:|---:|
| population | 1 | 8 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| population | 2 | 8 | 0.976261 | 1.952522 | 0.047478 | 0.024316 | 1.952522 |
| population | 4 | 8 | 0.963501 | 3.854006 | 0.048665 | 0.012627 | 3.854006 |
| population | 8 | 8 | 0.901383 | 7.211060 | 0.112706 | 0.015630 | 7.211060 |

## dmc

Host CPU: Intel Xeon Processor, 8 physical cores.

| family | workers | cores | eta | Gustafson S_G | Gustafson s_G | Karp-Flatt | fixed-work S |
|---|---:|---:|---:|---:|---:|---:|---:|
| dmc | 1 | 8 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| dmc | 2 | 8 | 0.996993 | 1.993986 | 0.006014 | 0.003016 | 1.993986 |
| dmc | 4 | 8 | 0.958504 | 3.834016 | 0.055328 | 0.014431 | 3.834016 |
| dmc | 8 | 8 | 0.836283 | 6.690266 | 0.187105 | 0.027967 | 6.690266 |

