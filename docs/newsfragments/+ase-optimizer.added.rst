Added ``from anneal.ase import Anneal``. Same ``run(fmax=...)`` call site
as ASE ``BFGS`` / ``LBFGS`` / ``FIRE``. Default ``mode="local"`` is a
quench only. ``mode="search"`` hops from the current geometry
(``search_from``, length scale from nearest neighbours) then quenches;
``mode="auto"`` hops when there are four or more free atoms. Not a local-optimizer rename: hops leave the
starting basin. Hop ``seed`` and ``budget`` are explicit. Periodic
cells and cell filters are refused on hop.
