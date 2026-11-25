Every harness that records a machine identity now takes it from the one
salted digest in ``scripts/machine_provenance.py``, which falls back to
the ``ioreg`` ``IOPlatformUUID`` on macOS. The Sutton-Chen rerun and the
template-free runner no longer hash ``/etc/machine-id`` without the salt,
and the LJ board journal no longer prints the raw machine id. The
documentation marks the stored unsalted digests, including the
``machine_id`` column of ``record/gmin-184/rows.csv``. Benchmark row files
read by the paper audit and the Sutton-Chen rerun are refused when they
hold a carriage return, the same check the ASE panel summary applies.
