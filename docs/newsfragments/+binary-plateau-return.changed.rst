A chain with two or more species labels now takes part in the call-plateau leave of ``Config::general_collective``.
When the chain has gone ``restart_patience`` calls without a lower minimum, its leave first moves it onto the lowest stored minimum that lies below it and whose fingerprint is outside the merge radius of its own, and does not return to one of its last few destinations.
A run with a single species label, with or without frozen atoms or an active region, keeps the ordinary activation path.
The unlike-pair exchange of a binary chain starts on every fourth hop and then follows its own acceptance.
Acceptance above one half shortens the interval to as little as two hops, and acceptance below one in five lengthens it to at most thirty-two.
