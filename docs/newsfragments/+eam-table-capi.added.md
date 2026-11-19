- Added `rgpot_potential_new_eam_table(path, format, periodic)` to the C ABI
  of `rgpot-core`, with `rgpot_core::eam_table::EamTablePotential` as its safe
  Rust wrapper and `rgpot::PotentialHandle::eam_table` for C++. A host reads a
  LAMMPS funcfl, setfl or Finnis-Sinclair table in process, with no C++ kernel
  and no engine plugin. The evaluation matches LAMMPS `pair_style eam` on
  `Cu_u3.eam` for a copper cell with a vacancy to 3e-12 eV and 3e-14 eV per
  Angstrom.
