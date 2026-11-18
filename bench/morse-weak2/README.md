Copyright (c) anneal developers

# Rejected adaptive-default gate

Same-session current `Config::recommended` against the proposed adaptive default.
Seeds 2000-2011, host `bench`, serial `taskset -c 0`, paper call budgets, no wall cap.
code_revision 59d09eba90d4e973c766731076d39b56b6ff46d5, bcba0727e4f3c689ac7a84390799acb9e12330dc.
Misses are censored at their call budget.
The gate stopped after the first exact hit-count regressions; later systems are therefore
intentionally incomplete append-only evidence, not an asserted full-panel result.
The adaptive machinery remains opt-in.

| system | budget | base hits | fix hits | base med calls | fix med calls | base wall s | fix wall s | Fisher p | U | MW p | wall ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m38r3 | 400,000 | 12/12 | 12/12 | 735.5 | 558.5 | 0.008 | 0.005 | 1 | 67.5 | 0.817166 | 0.607 |
| m38r6 | 400,000 | 12/12 | 12/12 | 28925.0 | 33338.5 | 0.260 | 0.291 | 1 | 71 | 0.97697 | 1.119 |
| m38r10 | 400,000 | 12/12 | 8/12 | 11856.5 | 179350.5 | 0.118 | 1.461 | 0.0931677 | 119 | 0.00713377 | 12.378 |
| m38r14 | 400,000 | 12/12 | 5/12 | 19529.5 | 400000.0 | 0.186 | 3.270 | 0.00457666 | 142 | 4.85825e-05 | 17.610 |
| m55r3 | 1,000,000 | 12/12 | 12/12 | 2040.5 | 3706.0 | 0.037 | 0.066 | 1 | 90 | 0.312321 | 1.772 |
| m55r6 | 1,000,000 | 12/12 | 12/12 | 12285.0 | 8459.5 | 0.247 | 0.159 | 1 | 62 | 0.58336 | 0.643 |
| m55r10 | 1,000,000 | 12/12 | 12/12 | 104281.0 | 115213.0 | 1.678 | 1.845 | 1 | 71 | 0.97697 | 1.099 |
| m55r14 | 1,000,000 | 5/12 | 6/12 | 1000000.0 | 962472.5 | 15.946 | 15.646 | 1 | 63 | 0.592722 | 0.981 |

Medians are Hyndman-Fan type 7. Fisher is two-sided exact on hits.
Mann-Whitney is two-sided asymptotic on calls with tie and continuity corrections.
