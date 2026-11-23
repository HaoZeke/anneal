Added a ``catalog`` Cargo feature (kept in ``default``) gating the
``readcon-core`` and ``readcon-db`` dependencies along with
``minima_db``, ``pes_db``, ``PesNetwork::from_records``, and the
corpus/slab examples and tests. Downstream Rust crates that link another
version of ``readcon-db`` can pass ``default-features = false`` to
exclude ``readcon-core``, ``readcon-db``, and their unmangled
``rkr_*`` / ``rkrdb_*`` C symbols without affecting default consumers.
