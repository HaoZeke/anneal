``global_optimize`` with ``noise_sigma`` and no ``grad_fn``, with one replica
or several, could run on without end when every value it saw was ``+inf``,
NaN or ``-inf``: on ``[0, 1]`` at a budget of 60, for one, after 38 calls
every arm of the bandit returned without a call, and the loop, whose exits
depend on the budget that remains, went round again. The loop now goes on
to its closing polish once every arm has played since the last call without
making one, as none can then make one again, so such a run ends within its
budget. A run that never reaches that point makes the same calls as before.
