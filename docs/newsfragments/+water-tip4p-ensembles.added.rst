``water_tip4p`` runs thread replicas of the rigid hop when ``HISTORY_REPLICAS`` is set.
The replicas read the same ensemble variables as ``lj_cluster_search``, such as ``HISTORY`` and ``HISTORY_POLICY``.
Their shared history judges identity on the lab-frame O and H sites, by IRA when the ``ira`` feature is built.
