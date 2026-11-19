# nmpins5 default-on gate

A commit `dd054749d4703923f4cf4c34f50589c574ee1276` tree `f1705d6f02f7f257f50e831c16057740283efea5`.
B commit `82a07ec3723fae37827e51eeeb411f1feccfabcd` tree `5cd688757a896be8cd1a5c2705c32c8787243437`.
A2 is a second build of A, same commit and tree, run in the same session.
Machine id sha256 `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`. Core 0 on every row.

Hits use `scripts/target_tol.py` `energy_hits` with tolerance 1e-3.
Lennard-Jones references are the Cambridge energies in `scripts/gmin_panel/run_panel.py`.
Morse references are the Cambridge rows in `examples/fixtures/morse_ccd.csv`.
TIP4P references for n <= 21, including n = 6, are the Cambridge headers in `data/tip4p_ccd.csv`.
(H2O)25 uses the pooled A/B minimum, as `scripts/tip4p_h2h/run_h2h.py` does when the header is absent.
Call medians and the Mann-Whitney test censor a miss at that system's budget.
The Mann-Whitney p is the exact two-sided midrank test (twice the smaller tail, capped at 1).
Fisher p is the exact two-sided test on the hit counts.
Medians are Hyndman-Fan type 7.
B loses if it has fewer hits, or its calls are worse at p < 0.05, or its median outside_s exceeds A by more than the A/A band.
The band is the absolute median outside_s gap of A versus A2 on the anchor: LJ75 for Lennard-Jones, M80 rho 6 for Morse, TIP4P6 for the hexamer, and TIP4P15 for the other TIP4P cells.
A non-anchor limit scales that anchor gap by this system's median A outside_s over the anchor's median A outside_s.
Best identity compares the recorded best token of A and B on each seed.

| system | hits A/B | fisher p | median calls A/B | mw p calls | median wall A/B | median pot A/B | median outside A/B | mw p outside | delta s | limit s | best identical | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lj38 | 12/12 | 1.00000000 | 397275.500000/397275.500000 | 1.00000000 | 2.468867/2.490169 | 0.902894/0.899847 | 1.573793/1.591782 | 0.71253508 | 0.017990 | 0.015355 | 12/12 | LOSES |
| lj75 | 12/12 | 1.00000000 | 3969648.500000/3969648.500000 | 1.00000000 | 76.649524/76.635861 | 33.817400/33.740757 | 42.764055/42.883058 | 0.93230050 | 0.119003 | 0.417223 | 12/12 | TIE |
| lj98 | 10/10 | 1.00000000 | 3962661/3962661 | 1.00000000 | 127.534333/130.065962 | 57.512963/57.485261 | 70.220614/70.955388 | 0.84283599 | 0.734775 | 0.685100 | 12/12 | LOSES |
| lj150 | 9/9 | 1.00000000 | 3958244/3958244 | 1.00000000 | 276.910128/277.632962 | 133.683733/133.714320 | 143.480333/144.247080 | 0.97740219 | 0.766747 | 1.399851 | 12/12 | TIE |
| m55r14 | 3/3 | 1.00000000 | 1000000/1000000 | 1.00000000 | 18.296596/18.325693 | 12.657539/12.644949 | 5.693682/5.696269 | 0.97740219 | 0.002587 | 0.001253 | 12/12 | LOSES |
| m80r6 | 12/12 | 1.00000000 | 3928574/3928574 | 1.00000000 | 158.639915/158.803731 | 106.520163/105.938144 | 52.497485/52.916670 | 0.84283599 | 0.419185 | 0.011555 | 12/12 | LOSES |
| m38r10 | 12/12 | 1.00000000 | 393748.500000/393748.500000 | 1.00000000 | 4.057369/4.065262 | 2.428536/2.428812 | 1.639509/1.646182 | 0.88738593 | 0.006673 | 0.000361 | 12/12 | LOSES |
| tip4p8 | 12/12 | 1.00000000 | 2000000/2000000 | 1.00000000 | 5.928774/6.005415 | 3.768664/3.770486 | 2.166616/2.217148 | 0.11350233 | 0.050531 | 0.002593 | 12/12 | LOSES |
| tip4p15 | 7/10 | 0.37070938 | 2000000/2000000 | 1.00000000 | 14.614351/14.519761 | 10.639204/10.627107 | 3.964127/3.902410 | 0.67065953 | -0.061717 | 0.004744 | 2/12 | WINS |
| tip4p20 | 6/2 | 0.19302733 | 2000000/2000000 | 1.00000000 | 22.970140/23.214548 | 17.656154/17.611275 | 5.454085/5.500835 | 0.84283599 | 0.046750 | 0.006527 | 2/12 | LOSES |
| tip4p21 | 0/0 | 1.00000000 | 2000000/2000000 | 1.00000000 | 25.521185/25.579601 | 19.284275/19.282587 | 6.136456/6.201340 | 0.71253508 | 0.064884 | 0.007344 | 1/12 | LOSES |
| tip4p25 | 1/1 | 1.00000000 | 2000000/2000000 | 1.00000000 | 35.038691/36.203172 | 26.849964/27.172764 | 8.304709/8.810359 | 0.12768790 | 0.505650 | 0.009939 | 1/12 | LOSES |
| tip4p6 | 12/12 | 1.00000000 | 2000000/2000000 | 1.00000000 | 4.300434/4.369846 | 2.512192/2.545887 | 1.788242/1.819613 | 0.51372258 | 0.031371 | 0.017116 | 12/12 | LOSES |

## A/A anchors

| anchor | median outside A | median outside A2 | band s |
|---|---:|---:|---:|
| lj75 | 42.764055 | 43.181278 | 0.417223 |
| m80r6 | 52.497485 | 52.509040 | 0.011555 |
| tip4p15 | 3.964127 | 3.968871 | 0.004744 |
| tip4p6 | 1.788242 | 1.805357 | 0.017116 |

Overall: default-on NO

Losing systems: lj38, lj98, m55r14, m80r6, m38r10, tip4p8, tip4p20, tip4p21, tip4p25, tip4p6.
