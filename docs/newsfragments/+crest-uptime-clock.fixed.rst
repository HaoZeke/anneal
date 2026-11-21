CREST's xtb wrapper times each call from ``/proc/uptime``, which stops
while the host is suspended. ``date`` had counted that suspended time.
