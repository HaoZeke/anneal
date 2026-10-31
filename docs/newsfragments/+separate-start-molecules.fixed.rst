A graph-keeping search no longer holds a crushed start for its whole budget.
``Config::for_molecular`` sets ``Config::separate_start_molecules``: when two
atoms of different declared groups bond under the graph rule, before the graph
is bound and before any call, every group moves rigidly away from the mean of
the group centroids by the smallest factor 1.05^k, k at most 60, that leaves no
such bond, and the search starts from that structure. An atom in no group is a
group of its own. It acts only under ``Config::keep_bond_graph``, with species,
nothing frozen and at least two declared groups. ``Outcome::start_separation``
reports k and the factor, or that no factor worked, and ``separate_start``
returns the separated structure. A start without such a contact is unchanged.
