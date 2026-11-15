`pair_style rgpot` works as a native LAMMPS pair style. It fills the
global virial from the potential's stress, or from a strain difference
when there is none, so pressure, NPT and `fix box/relax` work in
orthogonal and triclinic boxes. `pair_coeff * * Name [key value ...] El ...`
selects an rgpot potential by name and maps types to elements. MPI runs
gather the box on rank 0. Other unit styles stop with an error.
`rgpot-eval` evaluates a named potential on one geometry.
