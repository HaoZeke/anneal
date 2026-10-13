``run`` and ``run_qmc`` reserve room for the epochs a start records, after
the start's call. With ``n_epochs=10**12`` a start records every epoch when
``steps_per_epoch=200`` and there is no ``max_evals``, and when
``steps_per_epoch=0`` and ``max_evals`` leaves it more than its own call;
that room is 40 TB, and where the allocator refused it the interpreter
aborted. With ``2**62`` epochs of one step and no ``max_evals`` the room
could not even be sized, and a ``PanicException`` was raised. A run now
checks before any call that the room its histories hold at once can be
allocated together, and raises ``ValueError`` when it cannot: a start's
epochs and stationarity flags, and for ``run_qmc`` those of the first two
starts, as the best start's history is kept while the next runs. The end
of a start no longer copies its values to flag them, so it takes no room
beyond the 41 bytes an epoch reserved at its start.
