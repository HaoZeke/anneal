# Trail-arm opt-in gate

Verdict: **FIX** — gate failure in lj13-reach.

Redraw totals: A=0, B=1806, A2=0.

| cell | n | hits A/B/A2 | median calls A/B/A2 | redraws A/B/A2 | B/A time (95% CI) | A2/A time (95% CI) | B/A RSS (95% CI) | gates |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| lj13-panel | 12 | 12/12/12 | 212.5/212.5/212.5 | 0/1101/0 | 1.01902 [0.965144, 1.07444] | 0.994645 [0.892665, 1.10843] | 0.972995 [0.958497, 0.988146] | strict PASS; noise PASS; memory PASS |
| lj13-reach | 3 | 3/3/3 | 856/856/856 | 0/280/0 | 1.09626 [1.00637, 1.27461] | 0.974784 [0.963955, 0.980644] | 0.948188 [0.938462, 0.954853] | strict FAIL; noise FAIL; memory PASS |
| lj38-panel | 12 | 10/10/10 | 12139.5/11766/12139.5 | 0/425/0 | 0.968378 [0.902905, 1.01784] | 1.00969 [0.910372, 1.14133] | 0.99657 [0.954827, 1.04162] | strict PASS; noise PASS; memory PASS |

Fisher and Mann-Whitney p-values are in `summary.csv`.
B trajectories differ from A when `trajectory_differences` is nonzero.
A calls-to-hit benefit is present when B's median is lower; its two-sided p-value is `mwu_b_a`.

Early-stop per-call artifact flag: 0 A/B seed pairs had unequal charged calls.
Aggregate charged calls: A=2160000, B=2160000, A2=2160000.
Aggregate outside wall seconds: A=11.103141, B=10.944421, A2=11.185601.
