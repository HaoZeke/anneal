``run`` and ``run_qmc`` reserve room for the epochs a start records, after
the start's call. With ``n_epochs=10**12`` a start records every epoch when
``steps_per_epoch=200`` and there is no ``max_evals``, and when
``steps_per_epoch=0`` and ``max_evals`` leaves it more than its own call;
that room is 40 TB, and where the allocator refused it the interpreter
aborted. With ``2**62`` epochs of one step and no ``max_evals`` the room
could not even be sized, and a ``PanicException`` was raised. A run whose
history cannot be allocated now raises ``ValueError`` before any call.
