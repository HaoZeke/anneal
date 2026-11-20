``gfn2_covalent_h2h`` takes ``si13``, ``si20`` and ``c20`` beside ``si10`` and ``c10``.
It declares ``call_floor`` and ``call_floor_elements`` for its element, starts every call from a fresh self-consistent-charge (SCC) solution, and honours ``H2H_WALL_S``, a wall-time cap in seconds after which the ledger closes and the search returns its best so far.
``gfn2_water_h2h`` reports ``objective_calls`` and ``first_hit_call`` in ``run.txt``.
``gfn2_water_crest_driver`` serves the same (H2O)m objective, with its O-H wall, to ``xtb --driver`` through a Unix socket, so CREST and anneal run on one objective.
The new examples need the ``rgpot-ex`` feature.
