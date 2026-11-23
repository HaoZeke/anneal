<!-- Copyright 2023--present rgpot developers -->

`rgpot_so_exports` is in the test registry, under the feature that is on
only when that pin is registered. The tabulated EAM and vendored vesin
pytest modules read their fixtures from the checkout named by
`RGPOT_PROJECT` or from pytest's rootdir when that directory is the
checkout, and they skip only when no checkout is present. A file missing
from a checkout that is present fails the test.
