! MIT License
! Copyright 2023--present rgpot developers

!> Physics checks for the H-in-water star gather, run from pure Fortran so
!! a failure points at the kernel rather than the bindings.
!!
!! Checked: translational invariance of the energy, vanishing net force,
!! agreement between the analytic forces and a central difference of the
!! energy, independence of the energy from the order the water sites
!! arrive in, and rejection of atom lists that are not a water plus H
!! system.
!!
!! The finite-difference check is the one that can fail. Newton's third
!! law holds by construction here (the lone H is handed the negated sum
!! of the legs), so a vanishing net force says nothing about whether the
!! radial derivatives are right; the central difference does.
program test_water_h
   use rgpot_kinds, only: wp, ip
   use rgpot_water_h, only: water_h_params_t, water_h_energy_forces
   implicit none

   integer, parameter :: n_molecules = 2
   integer, parameter :: natoms = 3*n_molecules + 1
   real(wp), parameter :: box = 12.0_wp
   real(wp), parameter :: r_oh = 0.9572_wp
   real(wp), parameter :: theta_hoh = 104.52_wp
   real(wp), parameter :: tol_force = 1.0e-8_wp
   real(wp), parameter :: tol_sum = 1.0e-12_wp
   real(wp), parameter :: tol_energy = 1.0e-12_wp
   real(wp), parameter :: delta = 1.0e-5_wp

   type(water_h_params_t) :: par
   real(wp) :: cell(3, 3), energy, shifted_energy, permuted_energy
   real(wp) :: positions(3, natoms), forces(3, natoms), moved(3, natoms)
   real(wp) :: permuted(3, natoms)
   real(wp) :: net(3), e_plus, e_minus, numeric, worst
   integer(ip) :: numbers(natoms), permuted_numbers(natoms), broken(natoms)
   integer :: status, atom, comp, k
   character(len=:), allocatable :: errmsg
   logical :: failed

   ! eOn hands the sites over as H1, H2 of every molecule, then all the
   ! oxygens, then the lone H last. The order of everything but that last
   ! atom is immaterial to this term, which the permutation check below
   ! asserts, but the system is built in eOn's order anyway so the test
   ! exercises the layout the entry point is given in practice.
   integer, parameter :: shuffle(natoms) = [5, 1, 2, 6, 3, 4, 7]

   failed = .false.

   call water_plus_h(positions, numbers, cell)

   call water_h_energy_forces(positions, numbers, cell, par, energy, forces, &
                              status, errmsg)
   call require(status == 0, "energy/force evaluation failed", failed)

   ! Every leg puts equal and opposite force on its site and on the lone H.
   net = sum(forces, dim=2)
   call require(maxval(abs(net)) < tol_sum, "net force is not zero", failed)

   ! Rigid translation leaves the separations, and so the energy, alone.
   moved = positions + 0.37_wp
   call water_h_energy_forces(moved, numbers, cell, par, shifted_energy, &
                              forces, status, errmsg)
   call require(status == 0, "translated evaluation failed", failed)
   call require(abs(shifted_energy - energy) < tol_energy, &
                "energy is not translation invariant", failed)

   ! The two hydrogen legs share one parameter set, so regrouping the
   ! water sites cannot move the energy.
   do k = 1, natoms
      permuted(:, k) = positions(:, shuffle(k))
      permuted_numbers(k) = numbers(shuffle(k))
   end do
   call water_h_energy_forces(permuted, permuted_numbers, cell, par, &
                              permuted_energy, forces, status, errmsg)
   call require(status == 0, "permuted evaluation failed", failed)
   call require(abs(permuted_energy - energy) < tol_energy, &
                "energy depends on the order of the water sites", failed)

   ! Analytic forces agree with a central difference of the energy.
   worst = 0.0_wp
   do atom = 1, natoms
      do comp = 1, 3
         moved = positions
         moved(comp, atom) = moved(comp, atom) + delta
         call water_h_energy_forces(moved, numbers, cell, par, e_plus, forces, &
                                    status, errmsg)
         call require(status == 0, "forward difference evaluation failed", failed)

         moved(comp, atom) = moved(comp, atom) - 2.0_wp*delta
         call water_h_energy_forces(moved, numbers, cell, par, e_minus, forces, &
                                    status, errmsg)
         call require(status == 0, "backward difference evaluation failed", failed)

         numeric = -(e_plus - e_minus)/(2.0_wp*delta)

         call water_h_energy_forces(positions, numbers, cell, par, energy, &
                                    forces, status, errmsg)
         worst = max(worst, abs(numeric - forces(comp, atom)))
      end do
   end do
   call require(worst < tol_force, "forces disagree with finite differences", &
                failed)

   ! An atom list that is not water plus one H carries no lone H to place,
   ! and is refused rather than guessed at.
   broken = numbers
   broken(natoms) = 8_ip
   call water_h_energy_forces(positions, broken, cell, par, energy, forces, &
                              status, errmsg)
   call require(status /= 0, "an oxygen in the lone H slot was accepted", failed)

   broken = numbers
   broken(1) = 8_ip
   call water_h_energy_forces(positions, broken, cell, par, energy, forces, &
                              status, errmsg)
   call require(status /= 0, "a non-water composition was accepted", failed)

   broken = numbers
   broken(1) = 6_ip
   call water_h_energy_forces(positions, broken, cell, par, energy, forces, &
                              status, errmsg)
   call require(status /= 0, "a carbon atom was accepted", failed)

   if (allocated(errmsg)) deallocate (errmsg)

   if (failed) then
      error stop "test_water_h: failures above"
   end if
   print *, "test_water_h: ok, worst force deviation ", worst

contains

   !> Two water molecules and one extra H in a cubic cell, laid out in the
   !! order the entry point expects: the hydrogens of every molecule, then
   !! the oxygens, then the lone H.
   subroutine water_plus_h(pos, atomic_numbers, cellv)
      real(wp), intent(out) :: pos(3, natoms)
      integer(ip), intent(out) :: atomic_numbers(natoms)
      real(wp), intent(out) :: cellv(3, 3)

      real(wp), parameter :: oxygens(3, n_molecules) = reshape([ &
                                                       3.0_wp, 3.0_wp, 3.0_wp, &
                                                       3.0_wp, 7.0_wp, 3.0_wp], &
                                                       [3, n_molecules])
      real(wp) :: half_angle, arm(3)
      integer :: m

      half_angle = 0.5_wp*theta_hoh*acos(-1.0_wp)/180.0_wp
      arm = [r_oh*sin(half_angle), r_oh*cos(half_angle), 0.0_wp]

      cellv = 0.0_wp
      cellv(1, 1) = box
      cellv(2, 2) = box
      cellv(3, 3) = box

      do m = 1, n_molecules
         pos(:, 2*m - 1) = oxygens(:, m) + arm
         pos(:, 2*m) = oxygens(:, m) + [-arm(1), arm(2), arm(3)]
         pos(:, 2*n_molecules + m) = oxygens(:, m)
         atomic_numbers(2*m - 1) = 1_ip
         atomic_numbers(2*m) = 1_ip
         atomic_numbers(2*n_molecules + m) = 8_ip
      end do

      ! The lone H sits off the plane of both molecules so that all three
      ! force components carry a signal.
      pos(:, natoms) = [5.0_wp, 5.0_wp, 3.4_wp]
      atomic_numbers(natoms) = 1_ip
   end subroutine water_plus_h

   subroutine require(condition, what, failed_flag)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: what
      logical, intent(inout) :: failed_flag

      if (.not. condition) then
         print *, "FAIL: ", what
         failed_flag = .true.
      end if
   end subroutine require

end program test_water_h
