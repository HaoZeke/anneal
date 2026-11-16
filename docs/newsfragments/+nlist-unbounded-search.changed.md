An infinite cutoff or skin in `rgpot::nlist::Neighbours` walks every pair straight from the coordinates, with no bins and no stored list, `i` ascending and then `j` ascending, and keeps the update for any move while the atom count, the cell and the spec stay the same; `allPairs()` reports it.
That walk takes a copy of its callback, so a caller reads back only the state it captured by reference.
`SuttonChenPot` with `cutoff <= 0` searches that way.
It used to stand in a 1e6 search, which binned a cluster into one cell, read `MemAvailable` twice and stored and sorted every pair on each call.
Energies and forces keep their bits.
