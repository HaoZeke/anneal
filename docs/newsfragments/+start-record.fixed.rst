A molecular search now always has a structure to record and polish. Under
``Config::keep_bond_graph`` a run can refuse every trial, so its opening quench
may be the only structure it can record. When that quench keeps the graph but
fails its validation gradient, the search now continues it, ``relax_steps`` at
a time and at most eight more times, as ``Config::finish_improving`` continues
an improving trial. A continuation is kept only when it is sane and does not
rise. ``Outcome::start_continuations`` counts them. Under
``Config::record_polish`` a search that recorded nothing hands the polish the
state its chain holds, or the start when that state is not sane, so the
reserve is spent and the lowest recordable iterate is recorded. Searches
without species or without the graph invariant, Lennard-Jones among them, are
unchanged.
