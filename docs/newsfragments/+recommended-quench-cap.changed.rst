``Config::recommended`` lets a full quench run 1000 steps instead of 200.
At 200 steps, at least 27 per cent of the full quenches of LJ98 runs ended above the record gradient.
Those quenches could not be recorded, and the chain accepted on their energies.
On LJ98 seeds 0 to 63 at 2e6 force calls, 61 of 64 runs reach the Leary tetrahedron against 53 of 64.
LJ55 gets slower: all 64 runs still hit, but the median calls to hit [interquartile range] rise from 6603 [3344, 16285.25] to 10200.5 [4292.5, 19646.75].
The Bayes threshold of ``Config::recommended`` keeps its 200-step value, ``7/15``, and ``Config::derived`` recomputes it to ``39/79``.
The resolved digests of both presets change, and so do their search results.
