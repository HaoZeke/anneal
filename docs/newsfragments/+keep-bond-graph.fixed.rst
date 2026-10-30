A molecular search no longer keeps reacted products. ``Config::for_molecular``
sets ``Config::keep_bond_graph``: the covalent graph of the start (the bond
matrix over the Cordero radii at ``bond_tolerance``, pair by pair over atom
indices) is the reference, and ``quench_is_sane`` fails any quench whose graph
differs. Such a quench is a rejected step at every site that judges one, so it
is never recorded, adopted or reported as the best. When the start's own quench
changes the graph, the chain begins on the start geometry, priced by one
zero-step relaxation and never recorded. ``Outcome::graph_rejections`` and
``Outcome::start_graph_rejected`` report both. ``archive_search`` binds the
reference from its own start, so its nested walks judge against it. The
``pt111_heptamer`` and ``slab_adsorption`` examples turn the setting off.
Searches without species, Lennard-Jones among them, are unchanged.
