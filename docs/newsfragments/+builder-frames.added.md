<!-- Copyright 2023--present rgpot developers -->

`rgpot::nlist::Builder::frames()` counts the times an update folded the positions into the cell. Each accepted update folds once, whether it keeps the table or searches again, so `frames()` equals the number of accepted updates and `builds()` the number of searches among them. The state stays behind the builder's pointer, so the layout of `Builder` and the soversion of `librgpot` do not change.
