- Added `rgpot::PotentialHandle::eam_table(path, format, periodic)`, the C++
  face of `rgpot_potential_new_eam_table`. It throws `rgpot::Error` where the C
  constructor returns NULL.
