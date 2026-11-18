# branch_on_stall opt-in gate against develop

Base develop revision: `0ae5bfe0efb874a718f8e9632d18ff5fbc29f782`.
Candidate revision: `0670fcd69a8ede1e5a678ff855e0d8159bd47de3`.

`branch_on_stall` remains **off by default**. These rows are the opt-in arm
(`H2H_BRANCH=1`) against develop in one serial session on core 0 under
`sudo unshare --uts` with hostname `bench`.

- `rows-optin-2000.csv`: full 13-system gate, seeds 2000-2011.
- `compare-optin-2000.csv`: Fisher on hits and tie-corrected two-sided
  Mann-Whitney on calls, plus inside and outside potential medians and p-values.
- `rows-optin-3000.csv`: held-out systems (LJ150, LJ75, LJ104, Au55, M80 rho 14),
  seeds 3000-3011.
- `compare-optin-3000.csv`: same tests for held-out.

## Full 13-system gate (seeds 2000-2011)

`branch_hits/develop_hits`, Fisher `p`, Mann-Whitney calls `p`,
inside medians and `p`, outside medians and `p`.

- LJ38: 12/12, Fisher 1, calls p=0.015, inside 0.038/0.026 (p=0.0166), outside 0.056/0.0419 (p=0.0464)
- LJ55: 12/12, Fisher 1, calls p=1, inside 0.063/0.055 (p=0.885), outside 0.092/0.0732 (p=0.751)
- LJ75: 11/12, Fisher 1, calls p=0.507, inside 6.80/11.68 (p=0.507), outside 6.89/11.50 (p=0.507)
- LJ98: 12/10, Fisher 0.478, calls p=0.507, inside 20.35/21.34 (p=0.507), outside 14.88/15.77 (p=0.544)
- LJ102: 8/9, Fisher 1, calls p=0.661, inside 14.98/13.71 (p=0.665), outside 11.42/10.54 (p=0.885)
- LJ103: 9/8, Fisher 1, calls p=0.930, inside 35.39/35.81 (p=0.977), outside 25.23/23.81 (p=1)
- LJ104: 10/9, Fisher 1, calls p=0.817, inside 39.87/37.08 (p=0.885), outside 30.18/28.18 (p=0.885)
- LJ150: 9/5, Fisher 0.214, calls p=0.0993, inside 60.98/131.88 (p=0.0999), outside 42.54/86.71 (p=0.126)
- Au55: 6/6, Fisher 1, calls p=0.600, inside 26.22/22.27 (p=0.0999), outside 7.85/7.42 (p=0.840)
- M55 rho 6: 12/12, Fisher 1, calls p=1, inside 0.00712/0.00726 (p=0.977), outside 0.00104/0.00101 (p=0.707)
- M55 rho 14: 7/7, Fisher 1, calls p=0.590, inside 8.26/9.09 (p=0.751), outside 3.48/3.65 (p=0.885)
- M80 rho 6: 3/2, Fisher 1, calls p=0.655, inside 104.08/103.97 (p=0.583), outside 51.53/52.01 (p=0.665)
- M80 rho 14: 8/9, Fisher 1, calls p=0.619, inside 44.15/35.34 (p=0.707), outside 14.95/11.02 (p=0.583)

Result: losses remain (e.g. LJ75, LJ102, M80 rho 14), so the feature stays opt-in.

## Held-out (seeds 3000-3011) on LJ150 + loss systems

- LJ75: 12/12, Fisher 1, calls p=0.729, inside 10.65/9.01 (p=0.751), outside 10.95/8.86 (p=0.751)
- LJ104: 9/7, Fisher 0.667, calls p=0.953, inside 15.39/18.52 (p=0.840), outside 11.97/14.66 (p=0.840)
- LJ150: 9/6, Fisher 0.400, calls p=0.146, inside 60.37/126.06 (p=0.0999), outside 44.26/80.56 (p=0.157)
- Au55: 6/7, Fisher 1, calls p=0.448, inside 26.95/23.45 (p=0.0783), outside 8.48/7.34 (p=0.977)
- M80 rho 14: 7/8, Fisher 1, calls p=0.678, inside 80.79/64.46 (p=0.371), outside 30.90/23.47 (p=0.624)

Held-out also keeps M80 rho 14 as a loss, so default-on is still not met.
