A CPMD profile session accepts the Potentials file id without the
0x prefix. When libcpmdc.so is present, one water gradient is
compared with a central difference. The deck sets ``CONVERGENCE
ORBITALS`` to ``1e-7`` a.u. (the orbital-gradient threshold ``GNMAX``,
not a Hartree total-energy tolerance), and the finite-difference noise
floor uses one ulp of the returned energy. Four components are zero by
symmetry and are not counted.
