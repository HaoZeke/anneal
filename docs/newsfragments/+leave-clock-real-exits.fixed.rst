When ``Config::retry_noop_stall_leave`` is on, a directed leave
consumes the stall clock only when the stall response produces a new
state, and a no-op leave stays due instead of waiting out
``restart_patience``.
