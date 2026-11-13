``Config::local_step = LocalStep::Tric`` quenches a molecular search in translation-rotation internal coordinates (TRIC), with bonds, angles and dihedrals inside each covalent fragment and a translation and rotation per fragment.
``anneal.quench(..., method="tric", species=...)`` and ``cluster_search(..., groups=..., local_step="tric")`` expose it in Python.
It is opt-in and left out of the resolved JSON when unset, so every existing preset resolves and searches as before.
