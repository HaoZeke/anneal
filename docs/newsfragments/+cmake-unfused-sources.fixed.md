A CMake build compiles the core library sources and the reduced pair,
Lennard-Jones, Sutton-Chen and Tersoff potentials without floating-point
contraction, as meson does.
On arm64, where the compiler fuses `a*b+c` into one multiply-add by default,
the CMake library's results differed from meson's in the last bits, and
`ReducedPairTest` and the skewed-cell case of `FastNlistTest` failed on macOS.
