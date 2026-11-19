- Added `rgpot.EAMTablePot(file, format, periodic=True, stress=False, skin=1.0)`
  and `rgpot.has_eam_table`. The class reads a funcfl, setfl or fs table, has a
  `cutoff` property, and works with `RgpotCalculator("eam_table", ...)`. The
  wheel build now enables `with_eam_table`.
