<!-- Copyright 2023--present rgpot developers -->
The Muller-Brown analytic surface now has configurable parameters and C++, C, Rust, Python, RPC, and generic-engine interfaces.
Each configure call describes the whole surface, and an empty parameter list takes the published default; the PotentialConfig arms reference has the rule.
A PotentialConfig arm that does not belong on that builtin is refused with an error that names the arm.
MullerBrownPot::paramsKey follows the six parameter lists, so two surfaces do not share a result-cache entry.
