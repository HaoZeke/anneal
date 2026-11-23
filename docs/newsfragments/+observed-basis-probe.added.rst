``lattice::observed_basis_candidate`` is new public API that builds a
compact lattice candidate from a well-conditioned Gabriel tetrahedron
observed in the current geometry. On a directed stall, cluster hopping
runs this observed-basis probe once per basin on a side ledger; it is on
by default under ``Config::recommended`` (and skipped for rigid-body,
molecular, frozen, and active-region runs). Charging the side quench
against the run budget moved the pinned preset-walk and heated
fingerprints in ``metropolis_heat_follows_the_plateau`` and
``lattice_exchange_follows_the_plateau`` in ``tests/plateau_heat.rs``.
