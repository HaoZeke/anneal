``archive_search`` separates its start as a search does. Under
``Config::separate_start_molecules`` it moves molecules that start in covalent
contact apart before it binds the covalent graph and before any call, and every
hop, residual start and residual origin it derives begins from that structure.
``ArchiveOutcome::start_separation`` reports what the separation did. A start
without such a contact, and a search without species, is unchanged.
