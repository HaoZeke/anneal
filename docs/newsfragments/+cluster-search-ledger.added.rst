``cluster_search`` returns a ``ledger`` dict that says where the search's budget went.
It splits the charged evaluations into ``spent``, ``screen``, ``full`` and ``check``, and counts the relaxations that ``converged`` or were ``capped``, the trials the screen dropped (``screened_out``) or the return test sent home (``returned``), the line-search geometries refused for overlap (``overlap_rejected``) and the ``basins`` registered.
The openconf pairing job (``scripts/openconf_h2h/h2h_job.py``) writes the ledger of each run into its ``job.json``.
