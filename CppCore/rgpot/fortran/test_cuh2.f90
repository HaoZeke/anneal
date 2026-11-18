! MIT License
! Copyright 2023--present rgpot developers

!> Physics checks for the CuH2 embedded-atom gather kernel, run from pure
!! Fortran so a failure points at the kernel rather than the bindings.
!!
!! Checked: translational invariance of the energy, vanishing net force,
!! invariance under a relabelling of the atoms, and agreement between the
!! analytic forces and a central difference of the energy.
!!
!! The finite-difference check is the acceptance gate for the gather
!! rearrangement. An embedding force attributed to the wrong end of a
!! pair leaves the energy exactly right, keeps the net force at zero, and
!! shows up only here.
!!
!! The reordering check is the gate for the species dispatch: energies
!! and forces are compared against a run with the atom list reversed, so
!! H no longer occupies the trailing indices.
program test_cuh2
   use rgpot_kinds, only: wp, ip
   use rgpot_neighbors, only: neighbor_table_t
   use rgpot_cuh2, only: cuh2_params_t, cuh2_energy_forces, cuh2_z_cu, cuh2_z_h
   implicit none

   integer, parameter :: n_side = 2
   integer, parameter :: n_cu = 4*n_side**3
   integer, parameter :: natoms = n_cu + 2
   real(wp), parameter :: lattice = 3.615_wp
   real(wp), parameter :: vacuum = 8.0_wp
   !> Height of the H2 centre above the topmost Cu layer.
   real(wp), parameter :: h2_height = 2.08_wp
   !> H-H bond length.
   real(wp), parameter :: h2_bond = 0.74_wp

   !> Central-difference step. The Cu embedding polynomial sums terms of
   !! order 10^2 down to a result of order 1, so the energy carries about
   !! 10^-12 of absolute noise; a step this size keeps the quotient's
   !! rounding contribution near 10^-7 while the truncation term stays
   !! below that.
   real(wp), parameter :: delta = 1.0e-5_wp
   real(wp), parameter :: tol_force = 1.0e-5_wp
   real(wp), parameter :: tol_sum = 1.0e-9_wp
   real(wp), parameter :: tol_energy = 1.0e-8_wp

   !> Atoms sampled by the finite-difference check, spanning both species.
   integer, parameter :: probe(9) = [1, 5, 9, 14, 20, 27, 32, 33, 34]

   type(cuh2_params_t) :: par
   type(neighbor_table_t) :: table
   real(wp) :: cell(3, 3), energy, shifted_energy, reversed_energy
   real(wp), allocatable :: positions(:, :), forces(:, :), f0(:, :)
   real(wp), allocatable :: moved(:, :), reversed(:, :), f_reversed(:, :)
   integer(ip), allocatable :: znum(:), znum_reversed(:), znum_foreign(:)
   real(wp) :: net(3), e_plus, e_minus, numeric, worst, worst_perm, e_foreign
   integer :: status, k, atom, comp, i
   character(len=:), allocatable :: errmsg
   logical :: failed

   failed = .false.

   call cu_slab_with_h2(n_side, lattice, vacuum, h2_height, h2_bond, &
                        positions, znum, cell)
   allocate (forces(3, natoms), f0(3, natoms), moved(3, natoms))
   allocate (reversed(3, natoms), f_reversed(3, natoms))
   allocate (znum_reversed(natoms), znum_foreign(natoms))

   if (maxval(probe) > natoms) then
      error stop "test_cuh2: probe list outruns the atom count"
   end if

   call cuh2_energy_forces(positions, znum, cell, par, table, energy, f0, &
                           status, errmsg)
   call require(status == 0, "energy/force evaluation failed", failed)

   ! Net force on a periodic cell vanishes.
   net = sum(f0, dim=2)
   call require(maxval(abs(net)) < tol_sum, "net force is not zero", failed)

   ! Rigid translation leaves the energy untouched.
   moved = positions + 0.37_wp
   call cuh2_energy_forces(moved, znum, cell, par, table, shifted_energy, &
                           forces, status, errmsg)
   call require(status == 0, "translated evaluation failed", failed)
   call require(abs(shifted_energy - energy) < tol_energy, &
                "energy is not translation invariant", failed)

   ! Reversing the atom list puts H at the front. Species dispatch reads
   ! atomic numbers, so the energy holds and the forces follow the
   ! permutation.
   do i = 1, natoms
      reversed(:, i) = positions(:, natoms + 1 - i)
      znum_reversed(i) = znum(natoms + 1 - i)
   end do
   call cuh2_energy_forces(reversed, znum_reversed, cell, par, table, &
                           reversed_energy, f_reversed, status, errmsg)
   call require(status == 0, "reordered evaluation failed", failed)
   call require(abs(reversed_energy - energy) < tol_energy, &
                "energy depends on the atom ordering", failed)

   worst_perm = 0.0_wp
   do i = 1, natoms
      worst_perm = max(worst_perm, &
                       maxval(abs(f_reversed(:, i) - f0(:, natoms + 1 - i))))
   end do
   call require(worst_perm < tol_force, &
                "forces depend on the atom ordering", failed)

   ! Analytic forces agree with a central difference of the energy.
   worst = 0.0_wp
   do k = 1, size(probe)
      atom = probe(k)
      do comp = 1, 3
         moved = positions
         moved(comp, atom) = moved(comp, atom) + delta
         call cuh2_energy_forces(moved, znum, cell, par, table, e_plus, &
                                 forces, status, errmsg)
         call require(status == 0, "displaced evaluation failed", failed)

         moved(comp, atom) = moved(comp, atom) - 2.0_wp*delta
         call cuh2_energy_forces(moved, znum, cell, par, table, e_minus, &
                                 forces, status, errmsg)
         call require(status == 0, "displaced evaluation failed", failed)

         numeric = -(e_plus - e_minus)/(2.0_wp*delta)
         worst = max(worst, abs(numeric - f0(comp, atom)))
      end do
   end do
   call require(worst < tol_force, "forces disagree with finite differences", &
                failed)

   ! An atomic number outside {29, 1} is refused rather than guessed at.
   znum_foreign = znum
   znum_foreign(3) = 26_ip
   call cuh2_energy_forces(positions, znum_foreign, cell, par, table, &
                           e_foreign, forces, status, errmsg)
   call require(status /= 0, "a foreign species was accepted", failed)
   call require(len(errmsg) > 0, "a foreign species carried no message", failed)

   call table%release()
   deallocate (positions, forces, f0, moved, reversed, f_reversed)
   deallocate (znum, znum_reversed, znum_foreign)
   if (allocated(errmsg)) deallocate (errmsg)

   if (failed) then
      error stop "test_cuh2: failures above"
   end if
   print *, "test_cuh2: ok, energy ", energy
   print *, "test_cuh2: worst force deviation ", worst
   print *, "test_cuh2: worst reordering deviation ", worst_perm

contains

   !> FCC Cu slab of `cells` conventional cells per direction, with a
   !! vacuum gap stacked on top and an H2 molecule inside it.
   !!
   !! The gap exceeds the 6.1 Angstrom cutoff, so the slab does not see
   !! its own periodic image through the vacuum.
   subroutine cu_slab_with_h2(cells, a, gap, height, bond, pos, z, box)
      integer, intent(in) :: cells
      real(wp), intent(in) :: a, gap, height, bond
      real(wp), allocatable, intent(out) :: pos(:, :)
      integer(ip), allocatable, intent(out) :: z(:)
      real(wp), intent(out) :: box(3, 3)

      real(wp), parameter :: basis(3, 4) = reshape([ &
                                           0.00_wp, 0.00_wp, 0.00_wp, &
                                           0.00_wp, 0.50_wp, 0.50_wp, &
                                           0.50_wp, 0.00_wp, 0.50_wp, &
                                           0.50_wp, 0.50_wp, 0.00_wp], [3, 4])
      integer :: ix, iy, iz, b, m, ncu
      real(wp) :: span, top, centre

      ncu = 4*cells**3
      allocate (pos(3, ncu + 2), z(ncu + 2))

      span = a*real(cells, wp)
      box = 0.0_wp
      box(1, 1) = span
      box(2, 2) = span
      box(3, 3) = span + gap

      m = 0
      do ix = 0, cells - 1
         do iy = 0, cells - 1
            do iz = 0, cells - 1
               do b = 1, 4
                  m = m + 1
                  pos(1, m) = a*(real(ix, wp) + basis(1, b))
                  pos(2, m) = a*(real(iy, wp) + basis(2, b))
                  pos(3, m) = a*(real(iz, wp) + basis(3, b))
                  z(m) = cuh2_z_cu
               end do
            end do
         end do
      end do

      ! Topmost Cu layer, then the molecule above it.
      top = span - 0.5_wp*a
      centre = 0.5_wp*span

      pos(:, ncu + 1) = [centre - 0.5_wp*bond, centre, top + height]
      pos(:, ncu + 2) = [centre + 0.5_wp*bond, centre, top + height]
      z(ncu + 1) = cuh2_z_h
      z(ncu + 2) = cuh2_z_h
   end subroutine cu_slab_with_h2

   subroutine require(condition, what, failed_flag)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: what
      logical, intent(inout) :: failed_flag

      if (.not. condition) then
         print *, "FAIL: ", what
         failed_flag = .true.
      end if
   end subroutine require

end program test_cuh2
