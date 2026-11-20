``Config::prior_reseed``, ``prior_growth``, ``prior_core``, ``prior_handover`` and ``prior_motif`` are removed, and so is ``reseed_hops`` of ``ContinuousSymmetry::Measure``, which is now ``Measure { interval }``.
Only ``prior_soap`` remains.
A JSON or TOML configuration, a campaign file or a Python ``Config`` payload that sets a removed key is refused with a ``RemovedConfigKeyError`` that names the key, and so is a ``reseed_hops`` in a serialised ``ContinuousSymmetry``.
``Config::validate_json_keys`` and ``Config::validate_toml_keys`` run that check on a text, and Python has ``Config.validate_json`` and ``Config.validate_toml``.
The ``H2H_PRIORS`` names ``reseed``, ``growth``, ``core``, ``handover`` and ``motif`` are refused, and so are ``H2H_HANDOVER`` and the ``PRIOR_RESEED``, ``PRIOR_GROWTH``, ``PRIOR_CORE``, ``PRIOR_HANDOVER`` and ``PRIOR_MOTIF`` variables.
``H2H_PRIORS`` accepts ``soap``, ``0``, ``none`` and ``1``.
``Config::parse_prior_switches``, ``check_removed_env_vars`` and ``check_removed_env`` do the parsing.
