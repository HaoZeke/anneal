Record a salted physical-machine identifier, CPU model, and GMIN build key
across all paper and GMIN benchmark runners while retaining the compatibility
hostname, stage ``machine_provenance.py`` into isolated work directories,
record all build-key inputs (``cap=``, ``patch=``, ``fc=``, ``cc=``, ``cxx=``,
``fcflags=``, ``cppflags=``) and ``binary_sha256=`` in ``gmin-source.txt``,
reuse saved GMIN rows only when both ``code_revision`` and ``gmin_build_key``
match (rejecting mismatches naming the field), mark missing provenance on
reused legacy ``metal_h2h`` rows as ``unknown`` in summaries, and refuse
legacy schemas or mixed sample provenance unless panel runners target an
alternate ``--out`` directory or explicitly opt in with ``--overwrite-legacy``.
