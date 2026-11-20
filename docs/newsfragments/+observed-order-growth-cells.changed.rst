Observed-order candidates (``lattice::candidate`` with
``Source::Observed`` and ``lattice::candidate_keeping``) are built with
cell lists instead of all-pairs scans, and the monoatomic stall escape
reuses the growth while the chain stays on one structure. The
candidates are unchanged bit for bit; at 80 and 98 points a fresh one
takes about half the time it did. The new ``lattice::candidate_reusing``
and ``lattice::Growth`` expose the reuse.
