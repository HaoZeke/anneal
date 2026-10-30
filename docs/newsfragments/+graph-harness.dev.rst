``organic_bench`` prints ``keep_bond_graph`` and
``GRAPH_REJECTIONS n start true|false``. Under ``WATER_GAP_ARM``,
``water_hexamer`` prints ``keep_bond_graph``, ``graph_rejections`` and
``start_graph_rejected`` per seed. With ``WATER_GEOMETRY_DIR`` set, it writes
each seed's start and best structure there as XYZ. ``experiments/graph_rerun.py``
rescores the stored GFN2 results by covalent-graph identity next to the SMILES
rule, freezes the graph-rule references, runs the preset and scores it.
``experiments/graph_rerun.sh`` runs the campaign's jobs once the references are
frozen.
