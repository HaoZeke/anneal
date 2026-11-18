! MIT License
! Copyright 2023--present rgpot developers

!> Physics checks for the embedded-atom aluminium gather kernel, run from
!! pure Fortran so a failure points at the kernel rather than the bindings.
!!
!! Checked: the force vanishes on the ideal FCC lattice, the net force
!! vanishes on a rattled periodic cell, the energy is translation
!! invariant, and the analytic forces agree with a central difference of
!! the energy. The finite-difference check is the gate on the gather
!! rearrangement: dropping a neighbour's embedding derivative from `f_i`
!! leaves the energy exactly right and the force wrong by order one.
program test_eam_al
   use rgpot_kinds, only: wp
   use rgpot_neighbors, only: neighbor_table_t
   use rgpot_eam_al, only: eam_al_params_t, eam_al_energy_forces
   implicit none

   integer, parameter :: n_side = 2
   integer, parameter :: natoms = 4*n_side**3
   real(wp), parameter :: lattice = 4.05_wp

   !> Per-component displacement off the ideal lattice. Small enough that
   !! no pair lands within `delta` of the kernel's inner truncation radius,
   !! where a pair term steps to zero and a central difference would
   !! straddle the step: the outermost FCC shell inside the cutoff sits at
   !! 7.015 Angstrom and the truncation at 7.0996, a gap of 0.085, while a
   !! rattle of 0.02 moves a pair separation by at most 0.07.
   real(wp), parameter :: rattle = 0.02_wp

   !> Finite-difference step and the tolerance it can support. The
   !! embedding polynomial reaches terms of order 3e3 that cancel down to
   !! order 1e1, so the energy carries roughly 1e-12 eV of cancellation
   !! noise; dividing that by `2 * delta` puts the achievable floor near
   !! 1e-7 eV/Angstrom.
   real(wp), parameter :: delta = 5.0e-5_wp
   real(wp), parameter :: tol_force = 1.0e-5_wp

   real(wp), parameter :: tol_symmetry = 1.0e-9_wp
   real(wp), parameter :: tol_sum = 1.0e-9_wp
   real(wp), parameter :: tol_energy = 1.0e-8_wp

   type(eam_al_params_t) :: par
   type(neighbor_table_t) :: table
   real(wp) :: cell(3, 3), energy, shifted_energy
   real(wp), allocatable :: positions(:, :), forces(:, :), moved(:, :)
   real(wp), allocatable :: base_forces(:, :)
   real(wp) :: net(3), e_plus, e_minus, numeric, worst
   integer :: status, k, atom, comp
   character(len=:), allocatable :: errmsg
   logical :: failed

   failed = .false.

   call fcc_lattice(n_side, lattice, positions, cell)
   allocate (forces(3, natoms), moved(3, natoms), base_forces(3, natoms))

   ! Every site of the ideal lattice is a centre of inversion, so the force
   ! vanishes atom by atom and not merely in the sum.
   call eam_al_energy_forces(positions, cell, par, table, energy, forces, &
                             status, errmsg)
   call require(status == 0, "ideal lattice evaluation failed", failed)
   call require(maxval(abs(forces)) < tol_symmetry, &
                "forces do not vanish on the ideal lattice", failed)
   print *, "test_eam_al: ideal FCC energy per atom ", energy/real(natoms, wp)

   ! Everything below runs on a rattled cell, where the forces are non-zero
   ! and the finite-difference check has something to bite on.
   call rattle_positions(positions, rattle)

   call eam_al_energy_forces(positions, cell, par, table, energy, &
                             base_forces, status, errmsg)
   call require(status == 0, "rattled evaluation failed", failed)

   ! Net force on a periodic cell vanishes.
   net = sum(base_forces, dim=2)
   call require(maxval(abs(net)) < tol_sum, "net force is not zero", failed)

   ! Rigid translation leaves the energy untouched.
   moved = positions + 0.37_wp
   call eam_al_energy_forces(moved, cell, par, table, shifted_energy, forces, &
                             status, errmsg)
   call require(status == 0, "translated evaluation failed", failed)
   call require(abs(shifted_energy - energy) < tol_energy, &
                "energy is not translation invariant", failed)

   ! Analytic forces agree with a central difference of the energy.
   worst = 0.0_wp
   do k = 1, 12
      atom = 1 + mod(7*k, natoms)
      comp = 1 + mod(k, 3)

      moved = positions
      moved(comp, atom) = moved(comp, atom) + delta
      call eam_al_energy_forces(moved, cell, par, table, e_plus, forces, &
                                status, errmsg)
      call require(status == 0, "forward difference evaluation failed", failed)

      moved(comp, atom) = moved(comp, atom) - 2.0_wp*delta
      call eam_al_energy_forces(moved, cell, par, table, e_minus, forces, &
                                status, errmsg)
      call require(status == 0, "backward difference evaluation failed", failed)

      numeric = -(e_plus - e_minus)/(2.0_wp*delta)
      worst = max(worst, abs(numeric - base_forces(comp, atom)))
   end do
   call require(worst < tol_force, "forces disagree with finite differences", &
                failed)

   call table%release()
   deallocate (positions, forces, moved, base_forces)
   if (allocated(errmsg)) deallocate (errmsg)

   if (failed) then
      error stop "test_eam_al: failures above"
   end if
   print *, "test_eam_al: ok, worst force deviation ", worst

contains

   !> Face-centred cubic lattice, `cells` conventional cells per direction.
   subroutine fcc_lattice(cells, a, pos, box)
      integer, intent(in) :: cells
      real(wp), intent(in) :: a
      real(wp), allocatable, intent(out) :: pos(:, :)
      real(wp), intent(out) :: box(3, 3)

      real(wp), parameter :: basis(3, 4) = reshape([ &
                                           0.00_wp, 0.00_wp, 0.00_wp, &
                                           0.00_wp, 0.50_wp, 0.50_wp, &
                                           0.50_wp, 0.00_wp, 0.50_wp, &
                                           0.50_wp, 0.50_wp, 0.00_wp], [3, 4])
      integer :: ix, iy, iz, b, m

      allocate (pos(3, 4*cells**3))
      box = 0.0_wp
      box(1, 1) = a*real(cells, wp)
      box(2, 2) = a*real(cells, wp)
      box(3, 3) = a*real(cells, wp)

      m = 0
      do ix = 0, cells - 1
         do iy = 0, cells - 1
            do iz = 0, cells - 1
               do b = 1, 4
                  m = m + 1
                  pos(1, m) = a*(real(ix, wp) + basis(1, b))
                  pos(2, m) = a*(real(iy, wp) + basis(2, b))
                  pos(3, m) = a*(real(iz, wp) + basis(3, b))
               end do
            end do
         end do
      end do
   end subroutine fcc_lattice

   !> Deterministic displacement off the lattice sites. `sin` of successive
   !! integers spreads well enough for the purpose and reproduces across
   !! compilers, which a seeded generator would not.
   subroutine rattle_positions(pos, amplitude)
      real(wp), intent(inout) :: pos(:, :)
      real(wp), intent(in) :: amplitude

      integer :: m, c

      do m = 1, size(pos, 2)
         do c = 1, 3
            pos(c, m) = pos(c, m) + amplitude*sin(real(7*m + 3*c, wp))
         end do
      end do
   end subroutine rattle_positions

   subroutine require(condition, what, failed_flag)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: what
      logical, intent(inout) :: failed_flag

      if (.not. condition) then
         print *, "FAIL: ", what
         failed_flag = .true.
      end if
   end subroutine require

end program test_eam_al
