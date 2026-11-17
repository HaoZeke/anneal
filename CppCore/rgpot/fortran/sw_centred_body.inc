! MIT License
! Copyright 2023--present rgpot developers
!
! Body of the centred-triplet loop in atom_contribution (rgpot_sw.f90),
! included by its force-only and strain loops.
rik = table%dist(sk)
u_ik = table%vec(:, sk)/rik
e_i = e_i + triplet_energy(par, u_ij, rij, u_ik, rik)
f_j = end_atom_force(par, u_ij, rij, u_ik, rik)
f_k = end_atom_force(par, u_ik, rik, u_ij, rij)
f_i = f_i - f_j - f_k
