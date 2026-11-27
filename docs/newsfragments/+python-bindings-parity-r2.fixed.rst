Scaled cluster searches now scale liquid-density start spheres in the Rust
core and preserve ``length_scale`` and ``energy_scale`` on stock
``species`` and ``rigid`` runs without panicking across Python.
``anneal.Config`` additionally exposes ``rigid_body_geometry``,
``two_phase``, ``surfaces``, ``reoccupy``, ``hmc``, and
``with_rigid_bodies``, PyO3 extension classes are marked ``@final`` in
``_core.pyi`` and verified by ``mypy.stubtest``, and the
``cluster_search`` docstring documents the full result surface.
