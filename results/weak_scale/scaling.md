# Amdahl and Gustafson

Host CPU: Intel Xeon Processor, 8 physical cores.
GPU: none. CUDA: none.
Worker counts are 1, 2, 4, and 8, each at most the physical core count.
The walls are the medians in summary.csv.
eta(p) is the weak efficiency T(1) / T(p).
Gustafson scaled speedup is S_G(p) = p * eta(p).
Gustafson serial fraction is s_G(p) = p * (1 - eta(p)) / (p - 1) for p > 1.
Karp-Flatt serial fraction is e(p) = (1 / S_G(p) - 1 / p) / (1 - 1 / p).
Amdahl speedup uses that e(p) as a fixed-work serial fraction:
S_A(p) = 1 / (e(p) + (1 - e(p)) / p).

## pt

Host CPU: Intel Xeon Processor, 8 physical cores.

| family | workers | cores | eta | Gustafson S_G | Gustafson s_G | Karp-Flatt e | Amdahl S_A |
|---|---:|---:|---:|---:|---:|---:|---:|
| pt | 1 | 8 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| pt | 2 | 8 | 0.984059 | 1.968118 | 0.031882 | 0.016199 | 1.968118 |
| pt | 4 | 8 | 0.959024 | 3.836097 | 0.054634 | 0.014242 | 3.836097 |
| pt | 8 | 8 | 0.925217 | 7.401735 | 0.085466 | 0.011547 | 7.401735 |

## population

Host CPU: Intel Xeon Processor, 8 physical cores.

| family | workers | cores | eta | Gustafson S_G | Gustafson s_G | Karp-Flatt e | Amdahl S_A |
|---|---:|---:|---:|---:|---:|---:|---:|
| population | 1 | 8 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| population | 2 | 8 | 0.976802 | 1.953604 | 0.046396 | 0.023749 | 1.953604 |
| population | 4 | 8 | 0.969620 | 3.878480 | 0.040507 | 0.010444 | 3.878480 |
| population | 8 | 8 | 0.961163 | 7.689303 | 0.044385 | 0.005772 | 7.689303 |

## dmc

Host CPU: Intel Xeon Processor, 8 physical cores.

| family | workers | cores | eta | Gustafson S_G | Gustafson s_G | Karp-Flatt e | Amdahl S_A |
|---|---:|---:|---:|---:|---:|---:|---:|
| dmc | 1 | 8 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| dmc | 2 | 8 | 0.983940 | 1.967879 | 0.032121 | 0.016323 | 1.967879 |
| dmc | 4 | 8 | 0.957309 | 3.829235 | 0.056922 | 0.014865 | 3.829235 |
| dmc | 8 | 8 | 0.840216 | 6.721728 | 0.182610 | 0.027167 | 6.721728 |

