``paper_controls`` runs every arm but ``population`` through
``cluster_search``. The control arms had their own closure: a bare L-BFGS
to a 1e-6 gradient on the plain surface, a start in the preset sphere, and
no stop at the reference. ``recommended`` on LJ75 therefore ignored the
preset's surfaces and spent far more calls than ``cluster_search::search``
on the same seed. A system with a reference now stops at the reference
plus ``HIT``, and ``force_to_hit`` is the charged count at that stop. The
example's unit tests run under ``cargo test``, and one checks that the
``recommended`` and ``derived`` arms match ``cluster_search::search`` on
LJ75 seeds 2000 to 2003 in best energy bits and charged calls.
