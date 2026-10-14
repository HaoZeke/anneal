``run_qmc`` gives each start the point of the Halton sequence at an index
derived from the seed plus the start's number, and added the two with an
addition that overflows past ``2**64 - 1``. At a seed from
``2**64 - n_starts`` to ``2**64 - 2``, a build with overflow checks, such as
one that ``maturin develop`` makes, so raised a ``PanicException`` when it
reached the start whose index passes ``2**64 - 1``, where a release wheel
wraps the index to 0. The index now wraps in every build, so such a seed
runs in every build as it runs from a release wheel, and no run from a
release wheel changes.
