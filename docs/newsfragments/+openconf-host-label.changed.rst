A row of the openconf pairing benchmark names its host by a label, never by the machine's own name.
The label is ``BENCH_HOST`` when set, ``bench`` inside the bench namespace, and ``unlabelled`` otherwise (``scripts/openconf_h2h/bench_host.py``).
Scoring a run directory whose ``job.json`` predates the label no longer stops with a ``KeyError``; its rows read ``unlabelled``.
