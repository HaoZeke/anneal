``examples/sc_h2h.rs`` records each chain's validated minima in a readcon-db corpus when ``ANNEAL_MINIMA_DB`` names its root.
Those minima are read back by ``examples/sc_minima.rs``.
The reader lists the distinct energies, how many chains reached each, and their common-neighbour and coordination counts.
