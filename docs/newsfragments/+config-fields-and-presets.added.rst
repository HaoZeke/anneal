``Config`` now exposes every setting and every generic preset of the Rust
``cluster_hopping::Config``. ``Config.field_names()``, ``get``, ``to_dict``,
``replace``, ``resolved_json`` and ``resolved_sha256`` read and change the
settings, and the presets ``with_scales``, ``recommended_with_scales``,
``for_cluster_with_scales``, ``recommended_for_pair``,
``for_cluster_dual_core``, ``packing_superbasin``, ``for_tip4p``,
``for_molecular`` and ``recommended_molecular`` join ``with_dual_orbit_core``,
``with_species`` and ``with_flexible_molecules``. A preset argument that
Rust would answer with a panic raises ``InvalidArgumentError``.
``pytest/test_config_fields.py`` fails when a field of the Rust struct has no
entry.
