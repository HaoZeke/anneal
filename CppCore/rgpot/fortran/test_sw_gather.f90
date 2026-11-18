! MIT License
! Copyright 2023--present rgpot developers

!> Physics checks for the Stillinger-Weber gather kernel, run from pure
!! Fortran so a failure points at the kernel rather than the bindings.
!!
!! Checked: translational invariance of the energy, vanishing net force,
!! and agreement between the analytic forces and a central difference of
!! the energy. Together these pin the gather rearrangement: a mismatched
!! end-atom term would break the finite-difference check even where the
!! energy stayed right.
program test_sw_gather
   use rgpot_kinds, only: wp
   use rgpot_neighbors, only: neighbor_table_t
   use rgpot_sw, only: sw_params_t, sw_energy_forces
   implicit none

   integer, parameter :: n_side = 2
   integer, parameter :: natoms = 8*n_side**3
   real(wp), parameter :: lattice = 5.431_wp
   real(wp), parameter :: tol_force = 1.0e-6_wp
   real(wp), parameter :: tol_sum = 1.0e-9_wp
   real(wp), parameter :: delta = 1.0e-5_wp

   type(sw_params_t) :: par
   type(neighbor_table_t) :: table
   real(wp) :: cell(3, 3), energy, shifted_energy
   real(wp), allocatable :: positions(:, :), forces(:, :), moved(:, :)
   real(wp) :: net(3), e_plus, e_minus, numeric, worst
   integer :: status, k, atom, comp
   character(len=:), allocatable :: errmsg
   logical :: failed

   failed = .false.

   call diamond_lattice(n_side, lattice, positions, cell)
   allocate (forces(3, natoms), moved(3, natoms))

   call sw_energy_forces(positions, cell, par, table, energy, forces, status, &
                         errmsg)
   call require(status == 0, "energy/force evaluation failed", failed)

   ! Net force on a periodic cell vanishes.
   net = sum(forces, dim=2)
   call require(maxval(abs(net)) < tol_sum, "net force is not zero", failed)

   ! Rigid translation leaves the energy untouched.
   moved = positions + 0.37_wp
   call sw_energy_forces(moved, cell, par, table, shifted_energy, forces, &
                         status, errmsg)
   call require(status == 0, "translated evaluation failed", failed)
   call require(abs(shifted_energy - energy) < 1.0e-9_wp, &
                "energy is not translation invariant", failed)

   ! Analytic forces agree with a central difference of the energy.
   worst = 0.0_wp
   do k = 1, 12
      atom = 1 + mod(7*k, natoms)
      comp = 1 + mod(k, 3)

      moved = positions
      moved(comp, atom) = moved(comp, atom) + delta
      call sw_energy_forces(moved, cell, par, table, e_plus, forces, status, &
                            errmsg)
      moved(comp, atom) = moved(comp, atom) - 2.0_wp*delta
      call sw_energy_forces(moved, cell, par, table, e_minus, forces, status, &
                            errmsg)

      numeric = -(e_plus - e_minus)/(2.0_wp*delta)

      call sw_energy_forces(positions, cell, par, table, energy, forces, &
                            status, errmsg)
      worst = max(worst, abs(numeric - forces(comp, atom)))
   end do
   call require(worst < tol_force, "forces disagree with finite differences", &
                failed)

   call table%release()
   deallocate (positions, forces, moved)
   if (allocated(errmsg)) deallocate (errmsg)

   if (failed) then
      error stop "test_sw_gather: failures above"
   end if
   print *, "test_sw_gather: ok, worst force deviation ", worst

contains

   !> Diamond-cubic lattice, `n_side` conventional cells per direction.
   subroutine diamond_lattice(cells, a, pos, box)
      integer, intent(in) :: cells
      real(wp), intent(in) :: a
      real(wp), allocatable, intent(out) :: pos(:, :)
      real(wp), intent(out) :: box(3, 3)

      real(wp), parameter :: basis(3, 8) = reshape([ &
                                           0.00_wp, 0.00_wp, 0.00_wp, &
                                           0.00_wp, 0.50_wp, 0.50_wp, &
                                           0.50_wp, 0.00_wp, 0.50_wp, &
                                           0.50_wp, 0.50_wp, 0.00_wp, &
                                           0.25_wp, 0.25_wp, 0.25_wp, &
                                           0.25_wp, 0.75_wp, 0.75_wp, &
                                           0.75_wp, 0.25_wp, 0.75_wp, &
                                           0.75_wp, 0.75_wp, 0.25_wp], [3, 8])
      integer :: ix, iy, iz, b, m

      allocate (pos(3, 8*cells**3))
      box = 0.0_wp
      box(1, 1) = a*real(cells, wp)
      box(2, 2) = a*real(cells, wp)
      box(3, 3) = a*real(cells, wp)

      m = 0
      do ix = 0, cells - 1
         do iy = 0, cells - 1
            do iz = 0, cells - 1
               do b = 1, 8
                  m = m + 1
                  pos(1, m) = a*(real(ix, wp) + basis(1, b))
                  pos(2, m) = a*(real(iy, wp) + basis(2, b))
                  pos(3, m) = a*(real(iz, wp) + basis(3, b))
               end do
            end do
         end do
      end do
   end subroutine diamond_lattice

   subroutine require(condition, what, failed_flag)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: what
      logical, intent(inout) :: failed_flag

      if (.not. condition) then
         print *, "FAIL: ", what
         failed_flag = .true.
      end if
   end subroutine require

end program test_sw_gather
