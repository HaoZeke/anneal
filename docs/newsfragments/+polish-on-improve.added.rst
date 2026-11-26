``Config::polish_on_improve`` polishes the incumbent when a hop
improves it, at the component gradient tolerance, and charges every
evaluation on that hop's ledger. A polished point at the stop target
ends the search on that evaluation. The point replaces the incumbent
only when it is lower. The option is off on every preset, so a
resolved preset is unchanged. ``H2H_POLISH_ON_IMPROVE=1`` turns it on
for the molecular H2H driver.
