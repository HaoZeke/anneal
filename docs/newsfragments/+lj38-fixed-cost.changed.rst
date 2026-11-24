Cut fixed per-run overhead in ``cluster_hopping``, ``csm``,
``assignment``, and ``lattice::candidate_from_basis`` while keeping
trajectories bit-identical. ``cluster_hopping`` no longer computes a
write-only ``PackingBank`` on every accepted hop and hoists environment
flag reads out of the acceptance loop. ``candidate_from_basis`` grows and
peels lattice clouds with deterministic integer-hashed indexed maps and
binary heaps instead of quadratic frontier scans. ``csm`` caches group
tables, precomputes class blocks, hashes cached operations without per-op
bit-vector allocations, and certifies duals in place. ``assignment``
reuses scratch buffers and keeps the AVX2 target region across the
shortest-paths, uniqueness, and Kuhn-Munkres loops.
