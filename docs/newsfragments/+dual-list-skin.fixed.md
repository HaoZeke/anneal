<!-- Copyright 2023--present rgpot developers -->

`rgpot::lr::DualList` rejects an infinite or NaN skin with `std::invalid_argument` ("skin must be finite and non-negative"), as it already did for a negative skin.
