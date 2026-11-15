`rgpot_engine_last_error` in the in-tree Lennard-Jones engine plugin writes the
terminating NUL into a one-byte buffer, as `engine_c_abi.h` promises. Before,
it left that buffer untouched.
