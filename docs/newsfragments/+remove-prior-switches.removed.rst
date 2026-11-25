Breaking removal: the inert structure-prior switches are gone from
``Config`` and ``ContinuousSymmetry``. The fields had no effect once
the structure templates were removed, and the fields themselves are
now deleted. v0.12.0 still has them.
The public fields removed from ``anneal_core::methods::cluster_hopping::Config``
are ``prior_reseed``, ``prior_growth``, ``prior_core``, ``prior_handover``
and ``prior_motif``. The public field removed from the variant
``ContinuousSymmetry::Measure`` is ``reseed_hops``, so it is now
``ContinuousSymmetry::Measure { interval }``. ``Config::prior_soap`` is
the one prior switch that remains.
Callers should delete every assignment, struct-literal field and
``Measure`` pattern that names these fields, since Rust code that still
names them no longer compiles. Set ``prior_soap`` to ``false`` or use
``Knockouts`` with ``Config::apply_knockouts`` to turn the SOAP proposal off.
A saved JSON or TOML configuration must drop the keys ``prior_reseed``,
``prior_growth``, ``prior_core``, ``prior_handover``, ``prior_motif`` and
``reseed_hops`` (also as ``continuous_symmetry.reseed_hops``). Deserialising
``ContinuousSymmetry::Measure`` with ``reseed_hops`` now fails with a
``RemovedConfigKeyError``. ``Config::validate_json_keys``,
``Config::validate_toml_keys`` and the Python
``Config.validate_json`` / ``Config.validate_toml`` reject the removed keys
with an explicit error. The list is ``REMOVED_PRIOR_KEYS``.
The ``H2H_HANDOVER`` environment toggle of ``h2h_metal`` is gone, and
setting it to a non-empty value now fails, so unset it. ``H2H_PRIORS``
accepts only ``soap``, ``0``, ``none`` or ``1``. Use
``Config::parse_prior_switches`` and ``Config::check_removed_env`` to run
the same checks in your own driver.
