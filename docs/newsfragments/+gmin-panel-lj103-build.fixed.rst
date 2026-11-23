The GMIN panel applies the LJ103 table patch only on revisions that do
not already name LJ103. It honours ``GMIN_PANEL_OUT`` and runs an
isolated asv seed against the checked-out commit instead of overwriting
the committed source notes.
