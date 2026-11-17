`EAMCellVerletTest` requires a NaN wherever the direct pair loop gives one,
whatever its sign bit, and every other value bit for bit.
IEEE 754 does not specify the sign of a NaN result, and optimised Clang builds
flip it for the forces between coincident atoms, so the test failed in the
macOS CMake builds and passed in the unoptimised meson builds.
