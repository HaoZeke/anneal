A molecular search now polishes its recorded bests once it ends.
``Config::for_molecular`` sets ``Config::record_polish``: the search runs on its
budget less a reserve of R calls, the dimension of the mobile Cartesians less
the rigid-body modes (capped at a quarter of the budget), and every call left
then relaxes the states ``Ledger::record`` accepted, lowest first. The
optimizer steps in the eigenvectors of the Lindh et al. (1995) model Hessian,
with rational-function steps on a BFGS-updated Hessian, ANCOPT's curvature
clamp, step cap and rebuild period, and ANCOPT's tight stop. Every call is
charged, and a polished structure is recorded only when it passes the same
sanity, graph and gradient tests as a search record. ``Outcome::polish``
reports the reserve, the search's and the polish's calls and each polished
record. ``polish_structure`` polishes one structure alone, and
``model_hessian`` returns the model Hessian. The setting needs species, so
searches without species, Lennard-Jones among them, are unchanged.
