Voided GFN2 Si13 references are no longer treated as valid by the audit,
rescore, or floor-gate tooling. ``filter_rows`` in
``scripts/covalent_h2h/gfn2_crest.py`` applies ``--anneal-revision`` and
``--false-hits`` the same way to summaries and rescores; a named false
hit counts as a miss at its budget, with ``best`` set to its ``xtb_eV``.
