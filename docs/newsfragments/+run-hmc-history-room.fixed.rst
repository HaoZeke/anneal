``run_hmc`` reserved room in its history for every epoch once its start was
called, before the first proposal, and nothing checked that room first: with
``n_epochs=10**12`` the 40 TB it asked for could not be allocated and the
interpreter aborted, and with ``2**62`` epochs of one step the room could
not even be sized, and a ``PanicException`` was raised. Before any call,
``run_hmc`` now checks that the room can be allocated, as ``run`` does, and
raises ``ValueError`` when it cannot.
