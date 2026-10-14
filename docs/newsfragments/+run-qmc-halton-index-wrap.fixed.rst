``run_qmc`` begins each start, but the first when given ``x0``, at the point
of the Halton sequence at an index derived from the seed plus the start's
number, and added the two with an addition that overflows past
``2**64 - 1``. At a seed from ``2**64 - n_starts`` to ``2**64 - 2`` that sum
reaches ``2**64`` at one of the starts, and as ``run_qmc`` laid out the
points of all its starts before its first call, a build with overflow
checks, such as one that ``maturin develop`` makes, raised a
``PanicException`` before any call, with or without ``max_evals``, where a
release wheel wraps the index to 0. The index now wraps in every build, so
such a seed runs in every build as it runs from a release wheel, and
wrapping it changes no run from a release wheel.
