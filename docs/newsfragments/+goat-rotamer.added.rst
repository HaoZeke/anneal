``Config::rotamer_test`` counts two minima as one basin when GOAT's sorting calls them the same structure (de Souza 2025).
The energies agree within CREST's 0.05 kcal/mol, read in the run's unit through ``Config::kcal_per_mol``.
The rotational constants agree within 1 percent for a spherical top and 2.5 percent for an anisotropic one.
Those ends are GOAT's limit and the top of CREST's range (Pracht et al. 2024), and the straight line between them is anneal's.
The test acts on one chain's own store: its bias key and, under minima hopping, its controller's basin identity.
The shared history of a thread ensemble keeps its exact witness and does not consult the test.
A run that names no species uses unit masses, so on Lennard-Jones clusters the moments are those of equal points.
``Outcome::rotamers`` counts the quenched trials it placed in a kept basin.
``goat::Rotor`` computes the principal moments of inertia from standard atomic weights, and ``goat::RotamerArchive`` keeps the minima.
``lj_cluster_search`` and ``water_tip4p`` take ``rotamer``.
The flag is off by default, so seeded searches follow the same trajectories as before.
