<!-- Copyright 2023--present rgpot developers -->

`rgpot::nlist::build` and `buildByAtom` now leave the previous table in place when they reject a periodic coordinate that is not finite or lies more than 2^29 cells from the origin, as the header promises for a rejected geometry. Both cleared the table before that check and left it empty.
