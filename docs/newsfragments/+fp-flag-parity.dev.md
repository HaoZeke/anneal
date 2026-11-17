`tools/check_test_parity.sh`, and so the `test_parity` CI job, also reads the
`compile_commands.json` meson and CMake write when configured, and fails when a
source both compile gets different contraction or fast-math flags from each.
`tools/check_fp_parity.py` does the comparison.
