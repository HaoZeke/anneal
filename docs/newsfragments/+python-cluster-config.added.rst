``anneal.Config`` exposes the full ``cluster_hopping::Config`` preset,
builder, digest, and field surface in Python, and ``cluster_search``
accepts ``energy_scale`` and a user-configured ``config`` instance while
matching ``Config::recommended_with_scales`` and
``Config::for_cluster_with_scales`` exactly on scaled runs.
