Coordinate-length checks on the GMIN TIP4P helpers use
``is_multiple_of``, which is what current clippy asks for. The C
header and the Python bindings are unchanged.
