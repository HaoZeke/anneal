``Config`` reports ``monoatomic``, ``packing_cna_applies``, ``start_radius``
and ``proposal_arms`` as properties and ``packing_surface()`` as a dict, and
``Config.with_rigid_bodies(sites, species)`` returns the copy whose state is
rigid bodies. Geometry that Rust refuses raises ``InvalidArgumentError``.
``pytest/test_config_methods.py`` fails when a public method of the Rust
``Config`` is neither exposed nor named as plumbing for the Rust binaries.
