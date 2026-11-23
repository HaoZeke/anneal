The inert ``prior_reseed``, ``prior_growth``, ``prior_core``,
``prior_handover`` and ``prior_motif`` switches are gone. A TOML, JSON
or environment configuration that still names one is rejected.
``Config::validate_json`` reports that error to Python.
