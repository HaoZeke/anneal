The Sutton-Chen rerun records the salted machine digest from
``machine_digest()`` instead of a plain SHA-256 of ``/etc/machine-id``.
That helper reads the ``ioreg`` ``IOPlatformUUID`` on macOS when the
Linux files are absent. The paper audit, the Sutton-Chen rerun reader
and its summary, and the ASE panel summary refuse a row file that
contains a carriage return when the header names a revision, host or
machine column. Tables without those columns still load with CR endings.
The benchmark row notes and the gmin-184 record readme name the stored
plain digests, including the ``machine_id`` column of
``record/gmin-184/rows.csv``, and leave those rows as written.
