The Rust RPC server rejects a `calculate` request whose positions do not hold
three values per atom or whose box does not hold 9 values, and a callback
result whose forces tensor has the wrong size. Before, the callback read past
the end of a short request and the server read past the end of short forces.
