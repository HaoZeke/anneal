The ASE objective bridge refuses a request with non-finite coordinates
before any engine sees it, and rebuilds the tblite calculator after a
refused SCF. tblite 0.7.0 bound once to non-finite positions failed every
later evaluation in the process, so a GFN2 ``water_hexamer`` board stopped
at the next seed with ``NoFiniteStart``.
