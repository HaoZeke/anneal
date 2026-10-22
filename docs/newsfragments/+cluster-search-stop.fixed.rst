``cluster_search`` now ends at its next evaluation when a callback raises a
``BaseException`` that is not an ``Exception``, such as ``KeyboardInterrupt``,
instead of spending the rest of the budget on refused calls before raising
it. ``Ledger::with_stop`` closes a ledger on a caller-owned flag.
