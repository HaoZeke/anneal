`tensor_id`: basin identity from a three-body tensor invariant.
`SortedPairs` identifies a basin by the multiset of pairwise distances, which
fixes `tr A^2` of the Gaussian kernel `A` and leaves `tr A^3`, the weighted
triangle sum, free to differ; Bloom's homometric pair `{0,1,4,10,12,17}` and
`{0,1,8,11,13,17}` share the multiset and are not congruent. `TripletSpectrum`
appends the spectra of `A` and of the mode-3 contraction
`M = A .* (A A)` of the triangle tensor `T_ijk = A_ij A_jk A_ik`, both exactly
invariant to relabelling and to rigid motions, to the sorted distances it
leaves unchanged. Worst-case separation over all pairs of distinct quenched
Lennard-Jones minima, in units of the descriptor's own response to a 0.02
jitter, goes from 0.99 to 1.27 at 38 points and from 0.98 to 1.32 at 55, and
the homometric pair from exactly 0 to 10.5. The cost is 80.6 us at 38 points
and 1.00 ms at 98, against 2.27 us and 15.3 us for one Lennard-Jones gradient
of which a hop spends thirty-one, and no potential evaluations at all. The
Tucker core is not a descriptor: it is invariant only up to a block-orthogonal
gauge, and the full `N^4` mode spectrum separates no better than the `N^3`
contraction.
