Native rigid-potential searches now keep internal evaluations in block
coordinates while translating interleaved Python inputs and outputs.
Structured ``Config.reoccupy`` and ``Config.hmc`` values reject mismatched
point counts, and ``rigid_body_geometry`` is read-only so rigid-body
invariants stay atomic. The ASE adapter rejects unsupported restart files,
honours ``master=False`` for trajectory output, and preserves ASE's
one-shot semantics for nonpositive observer intervals.
