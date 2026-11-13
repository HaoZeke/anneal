! Copyright (c) anneal developers
! GMIN's USERPOT entry points on rgpot's Tersoff carbon.
!
! This file takes the place of dummy_userpot.f90 in a second GMIN build.
! The energy and the gradient come from rgpot_covalent_energy_gradient in
! librgpot_covalent.so, kind 1, the kernel examples/h2h_covalent and
! ase_bh.py load, in Angstrom and eV. GMIN leaves the atom count and the
! start to a user potential, so both come from the coords file, one atom
! per line as the SW runs read it. The call is timed the way SW.f90 is. GMIN builds
! this file into a library ahead of the H2H_CAP module, so the timer is
! reached by gfortran's module symbol names, which the final link resolves.

subroutine userpot_init
end subroutine

subroutine userpot_get_natoms(num_atoms)
   integer :: num_atoms
   integer :: unit, status
   character(len=256) :: line
   num_atoms = 0
   open (newunit=unit, file='coords', status='old', action='read')
   do
      read (unit, '(a)', iostat=status) line
      if (status /= 0) exit
      if (len_trim(line) > 0) num_atoms = num_atoms + 1
   end do
   close (unit)
end subroutine

subroutine userpot_initialize_gmin(dof, x)
   integer, parameter :: real64 = selected_real_kind(15, 307)
   integer :: dof
   real(kind=real64) :: x(dof)
   integer :: unit
   open (newunit=unit, file='coords', status='old', action='read')
   read (unit, *) x
   close (unit)
end subroutine

subroutine userpot_potential(dof, x, grad, ereal, gradt)
   use, intrinsic :: iso_c_binding, only: c_int, c_long, c_double
   implicit none
   integer, parameter :: real64 = selected_real_kind(15, 307)
   integer :: dof
   real(kind=real64) :: x(dof)
   real(kind=real64) :: grad(dof)
   real(kind=real64) :: ereal
   logical :: gradt
   interface
      subroutine pot_enter(tick) bind(c, name='__h2h_cap_MOD_h2h_pot_enter')
         integer(kind=8), intent(out) :: tick
      end subroutine
      subroutine pot_leave(tick) bind(c, name='__h2h_cap_MOD_h2h_pot_leave')
         integer(kind=8), intent(in) :: tick
      end subroutine
      integer(c_int) function rgpot_covalent_energy_gradient(kind, n_atoms, positions, &
                                                             energy, gradient) &
         bind(c, name='rgpot_covalent_energy_gradient')
         import :: c_int, c_long, c_double
         integer(c_int), value :: kind
         integer(c_long), value :: n_atoms
         real(c_double), intent(in) :: positions(*)
         real(c_double), intent(out) :: energy
         real(c_double), intent(out) :: gradient(*)
      end function
   end interface
   integer(kind=8) :: tick
   integer(c_int) :: status
   real(c_double) :: energy
   real(c_double) :: gradient(dof)

   call pot_enter(tick)
   status = rgpot_covalent_energy_gradient(1_c_int, int(dof/3, c_long), x, energy, gradient)
   call pot_leave(tick)
   if (status /= 0) then
      ereal = 1.0d100
      grad = 0.0d0
      return
   end if
   ereal = energy
   if (gradt) grad = gradient
end subroutine

subroutine userpot_dump_configuration(filename, coords)
   integer, parameter :: real64 = selected_real_kind(15, 307)
   real(kind=real64) :: coords(*)
   character(*) :: filename
end subroutine

subroutine userpot_dump_lowest
end subroutine
