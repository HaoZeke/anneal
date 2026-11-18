! MIT License
! Copyright 2023--present rgpot developers

!> Physics checks for the Tersoff gather kernel, run from pure Fortran so
!! a failure points at the kernel rather than the bindings.
!!
!! Checked: translational invariance of the energy, vanishing net force,
!! and agreement between the analytic forces and a central difference of
!! the energy. The finite-difference check is the acceptance gate. A
!! bond-order term attributed to the wrong atom leaves the energy exactly
!! right, keeps the net force zero whenever the misattribution is
!! antisymmetric, and shows up only here.
!!
!! Two configurations carry the checks. The diamond lattice at its
!! experimental constant places every bond inside the inner taper radius,
!! where the cutoff function is flat and its derivative drops out; the
!! expanded lattice places bonds in the taper window so the cutoff
!! derivative enters both the pair term and the coordination sum. Both
!! carry a fixed jitter, because a force that vanishes by symmetry tests
!! nothing.
program test_tersoff
   use rgpot_kinds, only: wp
   use rgpot_neighbors, only: neighbor_table_t
   use rgpot_tersoff, only: tersoff_params_t, tersoff_energy_forces
   implicit none

   integer, parameter :: n_side = 2
   integer, parameter :: natoms = 8*n_side**3
   real(wp), parameter :: lattice = 5.431_wp
   !> Nearest neighbours land at 2.771 Angstrom, inside the 2.7 to 3.0
   !! taper window, while the second shell stays beyond the cutoff.
   real(wp), parameter :: taper_lattice = 6.40_wp
   real(wp), parameter :: tol_force = 1.0e-6_wp
   real(wp), parameter :: tol_sum = 1.0e-9_wp
   real(wp), parameter :: tol_energy = 1.0e-9_wp
   !> Balanced against both cells: the step is large enough that the
   !! cancellation inside `g(cos theta)`, which subtracts two terms of
   !! order 1e7 to leave one of order 1e3, stays well below the tolerance,
   !! and small enough that the taper's jump in curvature does too.
   real(wp), parameter :: delta = 1.0e-5_wp

   type(tersoff_params_t) :: par
   type(neighbor_table_t) :: table
   real(wp) :: cell(3, 3), energy, shifted_energy
   real(wp), allocatable :: positions(:, :), forces(:, :), moved(:, :)
   real(wp) :: net(3), worst_inner, worst_taper
   integer :: status, in_window
   character(len=:), allocatable :: errmsg
   logical :: failed

   failed = .false.

   call diamond_lattice(n_side, lattice, positions, cell)
   allocate (forces(3, natoms), moved(3, natoms))

   call tersoff_energy_forces(positions, cell, par, table, energy, forces, &
                              status, errmsg)
   call require(status == 0, "energy/force evaluation failed", failed)

   ! The lattice site symmetry admits no invariant vector, so every force
   ! vanishes here and the sum says little; it is the jittered cells below
   ! that make the check bite.
   net = sum(forces, dim=2)
   call require(maxval(abs(net)) < tol_sum, "net force is not zero", failed)

   ! Rigid translation leaves the energy untouched.
   moved = positions + 0.37_wp
   call tersoff_energy_forces(moved, cell, par, table, shifted_energy, forces, &
                              status, errmsg)
   call require(status == 0, "translated evaluation failed", failed)
   call require(abs(shifted_energy - energy) < tol_energy, &
                "energy is not translation invariant", failed)

   ! Jittered lattice: forces are large and the third law is a real check.
   call jitter(positions, 0.15_wp)
   call tersoff_energy_forces(positions, cell, par, table, energy, forces, &
                              status, errmsg)
   call require(status == 0, "jittered evaluation failed", failed)
   net = sum(forces, dim=2)
   call require(maxval(abs(net)) < tol_sum, &
                "net force on the jittered cell is not zero", failed)

   moved = positions + 0.37_wp
   call tersoff_energy_forces(moved, cell, par, table, shifted_energy, forces, &
                              status, errmsg)
   call require(status == 0, "translated jittered evaluation failed", failed)
   call require(abs(shifted_energy - energy) < tol_energy, &
                "jittered energy is not translation invariant", failed)

   call finite_differences(par, table, positions, cell, worst_inner, status)
   call require(status == 0, "finite differences failed on the inner cell", &
                failed)
   call require(worst_inner < tol_force, &
                "forces disagree with finite differences, inner cell", failed)

   ! Expanded lattice: bonds sit on the cutoff shoulder.
   call diamond_lattice(n_side, taper_lattice, positions, cell)
   call jitter(positions, 0.10_wp)

   in_window = count_in_shell(positions, cell, par%r, par%s)
   call require(in_window > 0, "no bond lies inside the cutoff taper", failed)

   call tersoff_energy_forces(positions, cell, par, table, energy, forces, &
                              status, errmsg)
   call require(status == 0, "expanded evaluation failed", failed)
   net = sum(forces, dim=2)
   call require(maxval(abs(net)) < tol_sum, &
                "net force on the expanded cell is not zero", failed)

   call finite_differences(par, table, positions, cell, worst_taper, status)
   call require(status == 0, "finite differences failed on the taper cell", &
                failed)
   call require(worst_taper < tol_force, &
                "forces disagree with finite differences, taper cell", failed)

   call table%release()
   deallocate (positions, forces, moved)
   if (allocated(errmsg)) deallocate (errmsg)

   if (failed) then
      error stop "test_tersoff: failures above"
   end if
   print *, "test_tersoff: ok, worst force deviation ", &
      max(worst_inner, worst_taper), " over ", in_window, " tapered bonds"

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

   !> Displace every coordinate by a fixed amount drawn from a rapidly
   !! varying sine, which spreads the values over `[-amp, amp]` and
   !! reproduces exactly on every run.
   subroutine jitter(pos, amp)
      real(wp), intent(inout) :: pos(:, :)
      real(wp), intent(in) :: amp

      integer :: atom, comp

      do atom = 1, size(pos, 2)
         do comp = 1, 3
            pos(comp, atom) = pos(comp, atom) &
                              + amp*sin(127.1_wp*real(3*(atom - 1) + comp, wp))
         end do
      end do
   end subroutine jitter

   !> Bonds whose minimum-image length falls between `rlo` and `rhi`, for
   !! an orthorhombic `box`.
   function count_in_shell(pos, box, rlo, rhi) result(cnt)
      real(wp), intent(in) :: pos(:, :)
      real(wp), intent(in) :: box(3, 3)
      real(wp), intent(in) :: rlo, rhi
      integer :: cnt

      integer :: ia, ib, k
      real(wp) :: dr(3), edge(3), rr

      edge = [box(1, 1), box(2, 2), box(3, 3)]
      cnt = 0
      do ia = 1, size(pos, 2) - 1
         do ib = ia + 1, size(pos, 2)
            dr = pos(:, ib) - pos(:, ia)
            do k = 1, 3
               dr(k) = dr(k) - edge(k)*anint(dr(k)/edge(k))
            end do
            rr = norm2(dr)
            if (rr > rlo .and. rr < rhi) cnt = cnt + 1
         end do
      end do
   end function count_in_shell

   !> Largest gap between an analytic force component and a central
   !! difference of the energy, over a spread of atoms and components.
   subroutine finite_differences(par, table, pos, box, worst, status)
      type(tersoff_params_t), intent(in) :: par
      type(neighbor_table_t), intent(inout) :: table
      real(wp), intent(in), contiguous :: pos(:, :)
      real(wp), intent(in) :: box(3, 3)
      real(wp), intent(out) :: worst
      integer, intent(out) :: status

      integer, parameter :: samples = 15
      real(wp), allocatable :: shifted(:, :), scratch(:, :)
      real(wp), allocatable :: analytic(:, :)
      real(wp) :: e_plus, e_minus, e_ref, numeric
      character(len=:), allocatable :: errmsg
      integer :: k, atom, comp, n

      n = size(pos, 2)
      allocate (shifted(3, n), scratch(3, n), analytic(3, n))

      worst = 0.0_wp
      call tersoff_energy_forces(pos, box, par, table, e_ref, analytic, &
                                 status, errmsg)
      if (status /= 0) return

      do k = 1, samples
         atom = 1 + mod(7*k, n)
         comp = 1 + mod(k, 3)

         shifted = pos
         shifted(comp, atom) = shifted(comp, atom) + delta
         call tersoff_energy_forces(shifted, box, par, table, e_plus, scratch, &
                                    status, errmsg)
         if (status /= 0) return

         shifted(comp, atom) = shifted(comp, atom) - 2.0_wp*delta
         call tersoff_energy_forces(shifted, box, par, table, e_minus, &
                                    scratch, status, errmsg)
         if (status /= 0) return

         numeric = -(e_plus - e_minus)/(2.0_wp*delta)
         worst = max(worst, abs(numeric - analytic(comp, atom)))
      end do
   end subroutine finite_differences

   subroutine require(condition, what, failed_flag)
      logical, intent(in) :: condition
      character(len=*), intent(in) :: what
      logical, intent(inout) :: failed_flag

      if (.not. condition) then
         print *, "FAIL: ", what
         failed_flag = .true.
      end if
   end subroutine require

end program test_tersoff
