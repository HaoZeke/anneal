``Plan`` can select the published lattice-ensemble choices one at a
time. ``retire = 0`` never frees a bank slot. ``Resemblance::LeeCsa`` is
eq. (2) of Lee, Lee and Lee, Phys. Rev. Lett. 91, 080201 (2003).
``CutoffSchedule::LeeFixedRatio`` takes the merge distance from
D_ave/2 to D_ave/5 by a fixed ratio over 10000 local minimizations,
then holds it. ``banked = false`` is Shao, Cheng and Cai's lattice
move with a fresh restart and no splice. Each default is the previous
behaviour. The private arm of ``lj_lattice_ensemble`` uses two slots
unless ``SLOTS`` is set, so the bank has a donor. The default seed is
the registered block, 3100, or 3000 for LJ75 and LJ98.
``lattice_ensemble_ablation`` scores those arms.
