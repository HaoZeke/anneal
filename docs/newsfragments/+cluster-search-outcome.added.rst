``cluster_search(full=True)`` returns every field of the Rust ``Outcome``,
including ``final_state``, ``accepted_transitions``, ``proposal_counts``,
the superbasin and energy-bias reports, the HMC diagnostics and ``stop``,
the code for why the run ended. The default result keeps its short shape.
``pytest/test_cluster_search_config.py`` fails when a field of the struct
has no key.
