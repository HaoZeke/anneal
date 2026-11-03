Common-neighbour analysis now reports a third index of 0 for a set with more than ``TRAIL_BOND_LIMIT = 12`` bonds.
It no longer searches the longest trail of such a set.
The exhaustive search grows exponentially with the bonds.
A Morse search at rho = 3, N = 35 was still in it at 410 s and had not finished when a 420 s limit stopped it.
Every signature the crate reads has at most five bonds, so no fraction changes.
