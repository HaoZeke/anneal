Copyright (c) anneal developers

Au55, seeds 2000-2011, maximum budget 1,000,000, host bench, one core.
The reference minimum is -191.03085280; a hit is at most 0.001 above it.
Anneal uses the cluster preset. ASE basin hopping and minima hopping use
the ASE 3.29 constructor defaults.
Rows carry code_revision 33bed75ff4e49af3628e7156e31c6e53e2fde32c.
The ASE wall cap is 96.650229 s. A wall-stopped row did not receive
the full call budget, so its call count is right-censored at `charged`.

| arm | hits | Fisher p | Mann-Whitney U | Mann-Whitney p | median best | gap to reference | median wall s | over anneal | anneal over | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anneal | 0/12 | 1 | 72 | 1 | -190.81083051 | 0.22002228 | 31.896040 | 1.000000 | 1.000000 | 26.788756 | 5.087176 |
| basin hopping | 0/12 | 1 | not estimated | not estimated | -189.78747254 | 1.24338026 | 96.650629 | 3.030176 | 0.330014 | 14.617083 | 82.033499 |
| minima hopping | 0/12 | 1 | not estimated | not estimated | -190.29395447 | 0.73689833 | 98.391769 | 3.084764 | 0.324174 | 2.126042 | 96.272051 |

Basin hopping stopped at the wall cap on 12/12 seeds; those rows charged 542,502-544,722 calls.
Minima hopping stopped at the wall cap on 12/12 seeds; those rows charged 67,491-86,896 calls.

Fisher is the two-sided exact test on hits under the recorded stopping rule.
Mann-Whitney is two-sided with tie and continuity corrections. It is not
estimated for an arm with wall-censored misses; imputing those misses at the
larger paper budget would manufacture observations that were never run.
Medians are type 7. Wall ratios are ratios of arm medians. The Rust and
NumPy random generators do not make a shared starting geometry from the
same numeric seed, so the report does not treat seed labels as pairs.
