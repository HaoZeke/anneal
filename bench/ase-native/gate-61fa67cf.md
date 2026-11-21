Native-unit ASE gate on 61fa67cf

This comparison is budget only. There is no wall-clock cap.
A hit is a finite best energy at or below the reference plus 0.001.
That hit column is the accept count: a run accepts the reference or it does not.
Mann-Whitney uses evaluations to the hit. A miss is ranked strictly worse than every hit,
and misses tie with each other.
Fisher is the two-sided exact test of anneal against that ASE arm.

| system | arm | hits | n | Fisher p | MW p evals | median wall s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| lj38 | anneal | 10 | 12 |  |  | 0.968968 |
| lj38 | ase_bh | 0 | 12 | 6.73038e-05 | 6.73038e-05 | 45.979 |
| lj38 | ase_mh | 4 | 12 | 0.0360748 | 0.0733841 | 45.0127 |
| lj55 | anneal | 12 | 12 |  |  | 0.793851 |
| lj55 | ase_bh | 0 | 12 | 7.39602e-07 | 7.39602e-07 | 121.551 |
| lj55 | ase_mh | 12 | 12 | 1 | 0.842836 | 11.8036 |
| lj98 | anneal | 4 | 12 |  |  | 81.1422 |
| lj98 | ase_bh | 0 | 12 | 0.0931677 | 0.0931677 | 538.717 |
| lj98 | ase_mh | 0 | 12 | 0.0931677 | 0.0931677 | 625.021 |
| lj104 | anneal | 0 | 12 |  |  | 91.0986 |
| lj104 | ase_bh | 0 | 12 | 1 | 1 | 547.704 |
| lj104 | ase_mh | 0 | 12 | 1 | 1 | 636.914 |
| lj150 | anneal | 0 | 12 |  |  | 185.384 |
| lj150 | ase_bh | 0 | 12 | 1 | 1 | 642.592 |
| lj150 | ase_mh | 0 | 12 | 1 | 1 | 768.822 |
| au55 | anneal | 0 | 12 |  |  | 33.4437 |
| au55 | ase_bh | 0 | 12 | 1 | 1 | 160.577 |
| au55 | ase_mh | 0 | 12 | 1 | 1 | 136.533 |
| m38r6 | anneal | 5 | 12 |  |  | 3.1515 |
| m38r6 | ase_bh | 0 | 12 | 0.0372671 | 0.0372671 | 51.0884 |
| m38r6 | ase_mh | 2 | 12 | 0.370709 | 0.241255 | 48.8979 |
| m38r14 | anneal | 0 | 12 |  |  | 3.14717 |
| m38r14 | ase_bh | 0 | 12 | 1 | 1 | 47.2603 |
| m38r14 | ase_mh | 0 | 12 | 1 | 1 | 48.1209 |
| m55r6 | anneal | 12 | 12 |  |  | 1.23883 |
| m55r6 | ase_bh | 0 | 12 | 7.39602e-07 | 7.39602e-07 | 134.829 |
| m55r6 | ase_mh | 12 | 12 | 1 | 0.442833 | 14.6889 |
| m55r14 | anneal | 0 | 12 |  |  | 15.4844 |
| m55r14 | ase_bh | 0 | 12 | 1 | 1 | 128.836 |
| m55r14 | ase_mh | 0 | 12 | 1 | 1 | 129.749 |

Diff versus 5eb79cdc

The earlier rows file is unchanged. This re-session supersedes it,
including the rows recorded before the minima-comparison prune.
A cell changed when its hit count, which is also its accept count, differs.

| system | arm | earlier hits | new hits | changed | pre-prune seeds | pre-prune hits | new on those seeds | pre-prune changed |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| lj38 | ase_bh | 0/12 | 0/12 | no | 12 | 0 | 0 | no |
| lj38 | ase_mh | 4/12 | 4/12 | no | 12 | 4 | 4 | no |
| lj55 | ase_bh | 0/12 | 0/12 | no | 12 | 0 | 0 | no |
| lj55 | ase_mh | 12/12 | 12/12 | no | 12 | 12 | 12 | no |
| lj98 | ase_bh | 0/12 | 0/12 | no | 1 | 0 | 0 | no |
| lj98 | ase_mh | 0/12 | 0/12 | no | 0 | 0 | 0 | none |
| lj104 | ase_bh | 0/12 | 0/12 | no | 0 | 0 | 0 | none |
| lj104 | ase_mh | 0/12 | 0/12 | no | 0 | 0 | 0 | none |
| lj150 | ase_bh | 0/12 | 0/12 | no | 0 | 0 | 0 | none |
| lj150 | ase_mh | 0/12 | 0/12 | no | 0 | 0 | 0 | none |
| au55 | ase_bh | 0/12 | 0/12 | no | 0 | 0 | 0 | none |
| au55 | ase_mh | 0/12 | 0/12 | no | 0 | 0 | 0 | none |
| m38r6 | ase_bh | 0/12 | 0/12 | no | 0 | 0 | 0 | none |
| m38r6 | ase_mh | 2/12 | 2/12 | no | 0 | 0 | 0 | none |
| m38r14 | ase_bh | 0/12 | 0/12 | no | 0 | 0 | 0 | none |
| m38r14 | ase_mh | 0/12 | 0/12 | no | 0 | 0 | 0 | none |
| m55r6 | ase_bh | 0/12 | 0/12 | no | 0 | 0 | 0 | none |
| m55r6 | ase_mh | 12/12 | 12/12 | no | 0 | 0 | 0 | none |
| m55r14 | ase_bh | 0/12 | 0/12 | no | 0 | 0 | 0 | none |
| m55r14 | ase_mh | 0/12 | 0/12 | no | 0 | 0 | 0 | none |
