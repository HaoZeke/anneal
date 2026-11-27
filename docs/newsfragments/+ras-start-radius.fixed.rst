``cluster_search(ras=True)`` draws its start where the normal search does,
through ``cluster_hopping::search_start``: a point cluster at liquid
density. It drew from the container sphere before, so one seed gave the
two searches different starts. ``Config.search_start(seed)`` and
``Config.liquid_start_radius(n)`` expose the rule in Python.
