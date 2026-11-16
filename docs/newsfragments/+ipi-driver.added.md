`rgpot::IPIDriver` and the `rgpot-ipi-driver` program are the client side
of the i-PI socket protocol (built with `with_ipi`). They connect to i-PI,
ASE's `SocketIOCalculator` or `IPIPot` over a UNIX socket or TCP and answer
`STATUS`, `INIT`, `POSDATA`, `GETFORCE` and `EXIT` with any potential
`engines/pot_spec.hpp` builds, returning hartree, hartree/bohr and the virial
`-sigma V`. Open clusters get a padded box, thin periodic cells are tiled to
the cutoff, and `--stress-step` gives a finite-difference virial for kernels
without stress. The flags follow `i-pi-py_driver`. `rgpot-ipi-bench` times the
socket round trip against a direct call; `docs/orgmode/howto/ipi.org` now
covers both directions and lists the timings.
