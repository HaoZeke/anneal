<!-- Copyright 2023--present rgpot developers -->
The Muller-Brown analytic surface now has configurable parameters and C++, C, Rust, Python, RPC, and generic-engine interfaces.
An empty parameter list selects that parameter's published default in the Python, RPC, and generic-engine interfaces.
A PotentialConfig arm that does not belong on that builtin is refused with an error that names the arm.
MullerBrownPot::paramsKey follows the six parameter lists, so two surfaces do not share a result-cache entry.
