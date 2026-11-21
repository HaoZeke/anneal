# scan-safe6 against scan-safe4

Bar 3 gates. The commits on `7d5bd692..7965b187` claim the scan
matches the scalar walk, so a difference in final energy, calls
or the trajectory digest is a failure and not only a report.

- Hits: PASS
- Time outside the potential per call: FAIL
- Bit identity: PASS
- Peak RssAnon: PASS

`ru_maxrss` and `VmHWM` ratios are in the table and do not gate.

| cell | hits A | hits B | Fisher p | MWU p | hits | B/A outside | outside 95% | A2/A 95% | time | bits | B/A RssAnon | RssAnon 95% | A2/A RssAnon | rss | ru_maxrss B/A | VmHWM B/A |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: |
| lj38 | 6 | 6 | 1 | 1 | pass | 1.034 | 0.982 .. 1.078 | 0.9461 .. 1.035 | pass | 6/6 pass | 1.005 | 0.8893 .. 1.123 | 0.8896 .. 1.124 | pass | 1 | 1.005 |
| lj55 | 6 | 6 | 1 | 1 | pass | 1.026 | 0.5604 .. 2.057 | 0.5403 .. 1.964 | pass | 6/6 pass | 1.041 | 0.5191 .. 2.5 | 0.4926 .. 2.728 | pass | 1 | 1.002 |
| lj75 | 3 | 3 | 1 | 1 | pass | 1.078 | 1.032 .. 1.128 | 0.9281 .. 1.014 | pass | 6/6 pass | 1 | 0.6509 .. 1.379 | 0.6509 .. 1.378 | pass | 0.9993 | 0.9997 |
| lj98 | 4 | 4 | 1 | 1 | pass | 1.092 | 0.9694 .. 1.192 | 0.8736 .. 1.074 | pass | 6/6 pass | 1 | 0.3649 .. 2.078 | 0.3641 .. 2.068 | pass | 0.9994 | 0.9996 |
| lj102 | 2 | 2 | 1 | 1 | pass | 1.12 | 0.931 .. 1.359 | 0.8529 .. 1.189 | pass | 6/6 pass | 1 | 0.6973 .. 1.582 | 0.6975 .. 1.581 | pass | 1 | 1 |
| lj103 | 1 | 1 | 1 | 1 | pass | 1.108 | 1.013 .. 1.154 | 0.9263 .. 1.097 | pass | 6/6 pass | 1 | 0.71 .. 1.289 | 0.7108 .. 1.289 | pass | 1 | 0.9999 |
| lj104 | 3 | 3 | 1 | 1 | pass | 1.105 | 1.027 .. 1.189 | 0.9602 .. 1.053 | pass | 6/6 pass | 0.9991 | 0.4155 .. 1.907 | 0.4163 .. 1.965 | pass | 1 | 1 |
| lj150 | 5 | 5 | 1 | 1 | pass | 1.074 | 0.9799 .. 1.259 | 0.9148 .. 1.118 | pass | 6/6 pass | 0.9991 | 0.3105 .. 2.525 | 0.3105 .. 2.754 | pass | 0.9997 | 0.9994 |
| au55 | 1 | 1 | 1 | 1 | pass | 0.9697 | 0.9291 .. 1.006 | 0.9409 .. 1.014 | pass | 6/6 pass | 1 | 0.9434 .. 1.08 | 0.9433 .. 1.08 | pass | 0.9995 | 0.9997 |
| m80r6 | 6 | 6 | 1 | 1 | pass | 1.104 | 0.6945 .. 1.799 | 0.6206 .. 1.594 | pass | 6/6 pass | 1 | 0.04201 .. 38.04 | 0.05114 .. 38.04 | pass | 1.003 | 1.006 |
| m80r14 | 2 | 2 | 1 | 1 | pass | 1.127 | 1.09 .. 1.165 | 0.96 .. 1.081 | fail | 6/6 pass | 1 | 0.763 .. 1.154 | 0.763 .. 1.168 | pass | 0.9998 | 0.9998 |

## Journal

The orchestrator was not restarted.
Every started run has an end line.
