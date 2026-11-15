`D3Pot` and the `"d3"` term of `ExprPot` accept `damping` as `"bj"`, `"BJ"`,
`"zero"` or `"Zero"` and raise `ValueError` for any other string. Before, an
unknown or misspelled name silently selected Becke-Johnson damping.
