The audit helpers run on macOS and on Python 3.10.
``paper/jctc/scripts/tblite_receipt.py`` reads its metadata with ``tomli`` when ``tomllib`` is absent, and the ``testing`` extra and the CI test install declare ``tomli`` for Python before 3.11.
``scripts/paper_audit.py`` matches the template switches in old revisions in Python after a fixed-string ``git grep``, since the BSD regex library has no ``\b``.
``scripts/session_metadata.py`` records every CPU where the OS has no ``sched_getaffinity``.
``scripts/machine_provenance.py`` reads the hardware UUID with ``ioreg`` and the CPU brand with ``sysctl`` on macOS; a host that offers neither still raises rather than record an invented value.
