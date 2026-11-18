! MIT License
! Copyright 2023--present rgpot developers

!> Physics checks for the Fe-He gather kernel, run from pure Fortran so a
!! failure points at the kernel rather than the bindings.
!!
!! Checked: translational invariance of the energy, vanishing net force,
!! agreement between the analytic forces and a central difference of the
!! energy, and rejection of an atomic number the potential does not cover.
!! The finite-difference check is the gate: an embedded-atom force splits
!! into a pair part and two embedding parts weighted by `dF/drho` at both
!! ends of the pair, and dropping either end leaves the energy correct.
!!
!! The cell is body-centred iron with two substitutional helium atoms on a
!! nearest-neighbour <111>/2 bond, which puts force into all three pair
!! channels and both density channels at once. Four cells a side keeps the
!! largest cutoff (5.4 A, He-He) below half the box edge (5.71 A), so each
!! pair enters the neighbour table exactly once per direction and the check
!! measures the potential rather than vesin's multiple-image path.
program test_fehe
   use rgpot_kinds, only: wp, ip
   use rgpot_neighbors, only: neighbor_table_t
   use rgpot_fehe, only: fehe_params_t, fehe_energy_forces
   implicit none

   integer, parameter :: n_side = 4
   integer, parameter :: natoms = 2*n_side**3
   real(wp), parameter :: lattice = 2.8553_wp

   !> Central differences of a cell energy near -500 eV carry roughly 1e-7
   !! eV/A of cancellation noise at this step, while a mis-attributed
   !! gather term costs 1e-3 eV/A or more.
   real(wp), parameter :: delta = 1.0e-5_wp
   real(wp), parameter :: tol_force = 5.0e-6_wp
   real(wp), parameter :: tol_sum = 1.0e-9_wp
   real(wp), parameter :: tol_energy = 1.0e-8_wp
   real(wp), parameter :: min_force = 1.0e-3_wp

   integer(ip), parameter :: z_fe = 26_ip
   integer(ip), parameter :: z_he = 2_ip

   !> Helium replaces the two sites of the first conventional cell.
   integer, parameter :: he_sites(2) = [1, 2]

   !> Probes: both helium atoms, four iron atoms inside the helium cutoffs
   !! (indices 3, 9, 33 and 43 are first or second neighbours of one of
   !! them), and three iron atoms that see helium through neither channel.
   integer, parameter :: probe_atom(15) = [ &
                         1, 1, 1, 2, 2, 2, 3, 3, 3, 9, 33, 43, 17, 65, 100]
   integer, parameter :: probe_comp(15) = [ &
                         1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]

   type(fehe_params_t) :: par
   type(neighbor_table_t) :: table
   real(wp) :: cell(3, 3), energy, shifted_energy
   real(wp), allocatable :: positions(:, :), forces(:, :), analytic(:, :)
   real(wp), allocatable :: moved(:, :)
   integer(ip), allocatable :: atomic_numbers(:)
   real(wp) :: net(3), e_plus, e_minus, numeric, worst
   integer :: status, k, atom, comp
   character(len=:), allocatable :: errmsg
   logical :: failed

   failed = .false.

   call bcc_lattice(n_side, lattice, positions, cell)
   allocate (forces(3, natoms), analytic(3, natoms), moved(3, natoms))
   allocate (atomic_numbers(natoms), source=z_fe)
   atomic_numbers(he_sites) = z_he

   call fehe_energy_forces(positions, atomic_numbers, cell, par, table, &
                           energy, analytic, status, errmsg)
   call require(status == 0, "energy/force evaluation failed", failed)

   ! A vacuous finite-difference check would pass on a zero force field.
   call require(maxval(abs(analytic)) > min_force, &
                "reference forces are too small to test", failed)

   ! Net force on a periodic cell vanishes.
   net = sum(analytic, dim=2)
   call require(maxval(abs(net)) < tol_sum, "net force is not zero", failed)

   ! Rigid translation leaves the energy untouched.
   moved = positions + 0.37_wp
   call fehe_energy_forces(moved, atomic_numbers, cell, par, table, &
                           shifted_energy, forces, status, errmsg)
   call require(status == 0, "translated evaluation failed", failed)
   call require(abs(shifted_energy - energy) < tol_energy, &
                "energy is not translation invariant", failed)

   ! Analytic forces agree with a central difference of the energy.
   worst = 0.0_wp
   do k = 1, size(probe_atom)
      atom = probe_atom(k)
      comp = probe_comp(k)

      moved = positions
      moved(comp, atom) = moved(comp, atom) + delta
      call fehe_energy_forces(moved, atomic_numbers, cell, par, table, &
                              e_plus, forces, status, errmsg)
      call require(status == 0, "forward displaced evaluation failed", failed)

      moved(comp, atom) = moved(comp, atom) - 2.0_wp*delta
      call fehe_energy_forces(moved, atomic_numbers, cell, par, table, &
                              e_minus, forces, status, errmsg)
      call require(status == 0, "backward displaced evaluation failed", failed)

      numeric = -(e_plus - e_minus)/(2.0_wp*delta)
      worst = max(worst, abs(numeric - analytic(comp, atom)))
   end do
   call require(worst < tol_force, "forces disagree with finite differences", &
                failed)

   ! An atomic number outside {26, 2} is a caller error, not a crash.
   atomic_numbers(natoms) = 14_ip
   call fehe_energy_forces(positions, atomic_numbers, cell, par, table, &
                           energy, forces, status, errmsg)
   call require(status /= 0, "unsupported atomic number was accepted", failed)
   call require(len(errmsg) > 0, "unsupported atomic number carries no "// &
                "message", failed)

   call table%release()
   deallocate (positions, forces, analytic, moved, atomic_numbers)
   if (allocated(errmsg)) deallocate (errmsg)

   if (failed) then
      error stop "test_fehe: failures above"
   end if
   print *, "test_fehe: ok, worst force deviation ", worst

contains

   !> Body-centred cubic lattice, `cells` conventional cells per direction.
   !! Site 1 is the corner of the first cell and site 2 its body centre, so
   !! substituting the first two atoms puts helium on a nearest-neighbour
   !! bond.
   subroutine bcc_lattice(cells, a, pos, box)
      integer, intent(in) :: cells
      real(wp), intent(in) :: a
      real(wp), allocatable, intent(out) :: pos(:, :)
      real(wp), intent(out) :: box(3, 3)

      real(wp), parameter :: basis(3, 2) = reshape([ &
                                           0.00_wp, 0.00_wp, 0.00_wp, &
                                           0.50_wp, 0.50_wp, 0.50_wp], [3, 2])
      integer :: ix, iy, iz, b, m

      allocate (pos(3, 2*cells**3))
      box = 0.0_wp
      box(1, 1) = a*real(cells, wp)
      box(2, 2) = a*real(cells, wp)
      box(3, 3) = a*real(cells, wp)

      m = 0
      do ix = 0, cells - 1
         do iy = 0, cells - 1
            do iz = 0, cells - 1
               do b = 1, 2
                  m = m + 1
                  pos(1, m) = a*(real(ix, wp) + basis(1, b))
                  pos(2, m) = a*(real(iy, wp) + basis(2, b))
                  pos(3, m) = a*(real(iz, wp) + basis(3, b))
               end do
            end do
         end do
      end do
   end subroutine bcc_lattice

   subroutine require(condition, what, failed_flag)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: what
      logical, intent(inout) :: failed_flag

      if (.not. condition) then
         print *, "FAIL: ", what
         failed_flag = .true.
      end if
   end subroutine require

end program test_fehe
