Search drivers return as soon as the caller stops evaluation, keeping the
best state found, instead of walking the rest of the budget after the
potential has stopped answering. A ledger slice keeps the caller's stop
flag. ``ANNEAL_STOPPED`` from a C callback ends ``anneal_box_minimize``
the same way and writes the best point. A flag that stays down does not
change the number of force calls.
