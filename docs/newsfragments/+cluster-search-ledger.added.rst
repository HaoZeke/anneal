``cluster_search`` returns a ``ledger`` dict that says where the search's budget went.
One charge is one oracle evaluation. An energy and a gradient returned together are one charge, and a separate energy call and a separate gradient call are two.
``spent`` equals the sum of ``screen``, ``full``, ``check``, ``closure``, ``quench``, ``settle``, ``lattice``, ``reoccupy`` and ``probe``, in a release build as well as a debug build.
``callbacks`` counts raw Python callbacks and ``split_extras`` counts the together-returns, each on its own. ``callbacks`` plus the charges that never entered Python equals ``spent`` plus ``split_extras``.
The dict also counts the relaxations that ``converged`` or were ``capped``, the trials the screen dropped (``screened_out``) or the return test sent home (``returned``), the fresh-start branches run (``branch_draws``), their charged evaluations (``branch_spent``), the line-search geometries refused for overlap (``overlap_rejected``) and the ``basins`` registered.
On archive searches, ``charged`` and ``ledger.spent`` are that same charge total.
The Python entry point accepts ``branch_on_stall=True`` to run and report the fresh-start branch path.
The openconf pairing job (``scripts/openconf_h2h/h2h_job.py``) writes the ledger of each run into its ``job.json``.
Head-to-head scoring uses this one unit for both codes. Openconf requests energy and gradient separately, so its count remains ``energy_calls + grad_calls``. An anneal run records ``oracle_calls``.
