# PR 171 r-next2 gate

Overall verdict: **PASS**.

| cell | hits D/D2/G | Fisher p | MWU p | time G/D (95% CI) | RSS G/D (95% CI) | probe offered/hit/calls | best splits D/G | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| m38r14d | 11/11/12 | 1 | 0.659785 | 0.842676 (0.695171, 0.98135) | 1 (1, 1) | 11/0/328 | 1/0 | PASS |
| lj75 | 5/5/5 | 1 | 0.446061 | 1.00749 (0.992414, 1.02719) | 1.05119 (1.0412, 1.05696) | 187/0/30231 | 0/0 | PASS |
| lj98 | 7/7/7 | 1 | 0.425639 | 0.983269 (0.971091, 0.995676) | 1.07083 (1.0559, 1.08088) | 125/0/20744 | 0/0 | PASS |
| tip4p6 | 24/24/24 | 1 | 0.508154 | 0.992097 (0.972092, 1.01159) | 1 (1, 1) | 0/0/0 | 0/0 | PASS |

The protocol, arm definitions, thresholds, and statistical rules are
in `PROTOCOL.md`. Rerun rows are in `rows-r2.csv`; expanded results are
in `report-r2.csv`.
