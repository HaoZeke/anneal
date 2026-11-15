`StillingerWeberSi`, `TersoffSiC` and `IPIPot` raise `rgpot.NumericalError`
from `__call__` for a non-finite energy, and `StillingerWeberSi` and
`TersoffSiC` release the interpreter during the call, like the other
potentials. Before, they returned the energy unchecked, and the two kernels
held the interpreter.
