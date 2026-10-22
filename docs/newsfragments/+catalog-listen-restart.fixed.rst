Stopping a catalog coordinator closes its listen socket before stop
returns, so a new coordinator can bind that same address. The thread
that owned the listener kept running after stop, and the listen fd
was closed only after stop had already returned, so a restart on that
port failed with the address already in use.
