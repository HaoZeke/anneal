# PBE/6-31G* water hexamer

The DFT code is NWChem. LIMITATIONS item 7 names it for the PBE/6-31G*
rerun, and the rgpot profile id is `nwchemc`.

Settings: basis `6-31g*`, theory `dft`, exchange-correlation `pbe`
(`scfType`), charge 0, multiplicity 1, task `gradient`. No dispersion
correction is named in FIXES or in the NWChem helper. The recorded
choice is `none named`. No cell is passed. The embed library is
`libnwchemc.so` (`POTENTIAL_LIBRARY` or `RGPOT_NWCHEMC_ENGINE`) plus a
PotentialConfig message (`POTENTIAL_CONFIG` or `NWCHEMC_CONFIG`). The
conda NWChem binary is not that library. Building the embed needs
`NWCHEM_TOP`.

Budget 4000, not the 1200 named in FIXES R6. Seeds 0-31. Selector `pbe`.
See `water_hexamer.METHODS-DETAIL.md`.

This build of rgpot-core does not enable its profile client, so the
driver records the settings and does not call NWChem. A missing library
or config is part of that load failure. The smoke file does not invent
an energy.
