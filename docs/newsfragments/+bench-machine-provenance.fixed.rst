Record a salted physical-machine identifier, CPU model, and GMIN build key
across all paper and GMIN benchmark runners while retaining the compatibility
hostname, stage ``machine_provenance.py`` into isolated work directories,
reuse saved GMIN rows only when both ``code_revision`` and ``gmin_build_key``
match (rejecting mismatches naming the field), and refuse legacy schemas or
mixed sample provenance.
