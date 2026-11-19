Periodic `EAMTablePot` evaluation sums every image inside the cutoff, so a cell whose face spacing is at most twice the cutoff is valid.
An atom can interact with its own lattice translations.
Evaluation still throws `std::invalid_argument` when one axis would search past an image extent of 1024, or when the three axes together would visit more than 262144 lattice shifts for one pair.
A fractional separation that is not finite, or whose nearest integer plus the image extent does not fit in `int`, throws `std::invalid_argument` as well.
