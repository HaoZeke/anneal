`EAMPot` throws `std::invalid_argument` for an atomic number it has no
parameters for. Before, it threw the bare `int` 14324, which callers catching
`std::exception` could not handle.
