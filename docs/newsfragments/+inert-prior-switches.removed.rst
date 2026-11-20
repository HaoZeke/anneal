Removed the inert ``Config`` fields ``prior_reseed``, ``prior_growth``,
``prior_core``, ``prior_handover``, and ``prior_motif``, along with
``ContinuousSymmetry::Measure::reseed_hops`` and the ``H2H_HANDOVER`` /
``PRIOR_*`` environment switches, and added ``Config::validate_json_keys``,
``Config::validate_toml_keys``, ``Config::check_removed_env``, and Python
``Config.validate_json`` / ``Config.validate_toml`` to reject stale keys.
