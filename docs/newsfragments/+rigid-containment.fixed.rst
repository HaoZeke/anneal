The molecular preset no longer crushes a molecule into its container. Each
trial was recentred and every atom beyond ``Config::container`` was pulled
radially onto the sphere. On a single molecule wider than the sphere, such as
trofinetide at 4.79 Å, that compressed the molecule and formed bonds before
any quench, and since rotations leave the radii unchanged, every trial of such
a run changed the bond graph and was refused. Under the new
``Config::rigid_containment``, set by ``Config::for_molecular``, a covalent
fragment of the chain's state that is wider than the sphere, with an atom
farther from its centroid than the radius, moves rigidly onto the centre
instead. Every other atom is contained as before, so a search whose molecules
all fit the sphere is unchanged. ``Outcome::graph_causes`` counts the trials
the graph invariant refused by where each left the graph: the proposal,
containment, or the quench from a contained trial that kept the graph.
``Outcome::graph_first`` gives the first with the bonds it broke and formed.
``organic_bench`` prints both, as ``GRAPH_CAUSES`` and ``GRAPH_FIRST``, and
its ``crush`` arm is the preset with atom-by-atom containment. Searches
without species, Lennard-Jones among them, are unchanged.
