`rgpot.ase.RgpotCalculator` gives the right energy, forces and stress for a
periodic cell narrower than twice the potential's cutoff. The kernels take
the minimum image of each pair, so before this a 32-atom argon cell with an
8.5 Angstrom LJ cutoff came out 12% too high and a 4-atom EAM-Al cell gave
-2.996 instead of -3.356 eV/atom. The calculator now tiles such a cell into
a supercell that is wide enough and returns the energy per copy and the
forces of the first copy. Structures with periodic and non-periodic axes
(slabs, wires) are evaluated instead of raising `ValueError`.
