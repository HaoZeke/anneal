! MIT License
! Copyright 2023--present rgpot developers

!> C entry point for the CuH2 embedded-atom potential.
!!
!! Only `bind(c)` names leave this library; the physics module exports no
!! external symbols of its own that C could collide with. The `_ws` entry
!! point keeps the neighbour table in the caller's workspace
!! (rgpot_workspace), so each caller's neighbour table follows its own
!! geometry; the entry point without a handle shares one module-level
!! workspace and is process-serial.
module rgpot_cuh2_capi
   use rgpot_kinds, only: wp, ip, c_double, c_int
   use, intrinsic :: iso_c_binding, only: c_ptr
   use rgpot_ferror, only: set_error, clear_error
   use rgpot_workspace, only: rgpot_workspace_t, workspace_from_c, &
                              ws_set_error, ws_clear_error
   use rgpot_cuh2, only: cuh2_params_t, cuh2_energy_forces, cuh2_hessian_vector
   implicit none
   private

   public :: rgpot_cuh2_force, rgpot_cuh2_force_ws, rgpot_cuh2_hvp_ws
   public :: rgpot_cuh2_ensure_params

   !> Workspace of the entry point without a handle (process-serial).
   type(rgpot_workspace_t), save, target :: legacy
   type(cuh2_params_t), save :: params

   !> F2: set by the C++ wrapper under std::call_once.
   !! After that call the block is read-only. Kernels take it
   !! intent(in). The handle-free entry's legacy workspace, when
   !! this module has one, stays process-serial and is not used
   !! by the Python binding.
   logical, save :: params_ready = .false.

contains

   !> Evaluate CuH2 forces and energy.
   !!
   !! `positions` and `forces` are `3 * natoms` doubles, x/y/z interleaved;
   !! `atomic_numbers` holds one atomic number per atom, 29 for Cu and 1
   !! for H, in any order; `cell` is nine doubles, row-major, one cell
   !! vector per row. Returns zero on success, non-zero on failure with
   !! the message available through `rgpot_fortran_last_error`.
   function rgpot_cuh2_force(natoms, positions, atomic_numbers, cell, forces, &
                             energy) result(status) &
      bind(c, name="rgpot_cuh2_force")
      integer(c_int), value, intent(in) :: natoms
      real(c_double), intent(in) :: positions(3, natoms)
      integer(c_int), intent(in) :: atomic_numbers(natoms)
      real(c_double), intent(in) :: cell(3, 3)
      real(c_double), intent(out) :: forces(3, natoms)
      real(c_double), intent(out) :: energy
      integer(c_int) :: status

      integer :: eval_status
      character(len=:), allocatable :: errmsg

      call clear_error()
      status = 0_c_int
      energy = 0.0_c_double
      forces = 0.0_c_double

      if (natoms < 1_c_int) return

      call cuh2_energy_forces(positions, atomic_numbers, cell, params, legacy%table, &
                              energy, forces, eval_status, errmsg)
      if (eval_status /= 0) then
         call set_error(errmsg)
         status = int(eval_status, c_int)
      end if
   end function rgpot_cuh2_force

   !> Evaluate CuH2 forces and energy in a per-caller
   !! workspace.
   !!
   !! `positions` and `forces` are `3 * natoms` doubles, x/y/z interleaved;
   !! `atomic_numbers` holds one atomic number per atom, 29 for Cu and 1
   !! for H, in any order; `cell` is nine doubles, row-major, one cell
   !! vector per row. Returns zero on success, non-zero on failure with
   !! the message available
   !! through `rgpot_fortran_workspace_error`. `handle` is a workspace from
   !! `rgpot_fortran_workspace_new`; a NULL handle returns -1.
   function rgpot_cuh2_force_ws(handle, natoms, positions, atomic_numbers, cell, forces, &
                             energy) result(status) &
      bind(c, name="rgpot_cuh2_force_ws")
      type(c_ptr), value :: handle
      integer(c_int), value, intent(in) :: natoms
      real(c_double), intent(in) :: positions(3, natoms)
      integer(c_int), intent(in) :: atomic_numbers(natoms)
      real(c_double), intent(in) :: cell(3, 3)
      real(c_double), intent(out) :: forces(3, natoms)
      real(c_double), intent(out) :: energy
      integer(c_int) :: status

      integer :: eval_status
      character(len=:), allocatable :: errmsg
      type(rgpot_workspace_t), pointer :: ws

      ws => workspace_from_c(handle)
      if (.not. associated(ws)) then
         status = -1_c_int
         return
      end if
      call ws_clear_error(ws)
      status = 0_c_int
      energy = 0.0_c_double
      forces = 0.0_c_double

      if (natoms < 1_c_int) return

      call cuh2_energy_forces(positions, atomic_numbers, cell, params, ws%table, &
                              energy, forces, eval_status, errmsg)
      if (eval_status /= 0) then
         call ws_set_error(ws, errmsg)
         status = int(eval_status, c_int)
      end if
   end function rgpot_cuh2_force_ws

   !> Hessian-vector product of the CuH2 energy in a per-caller workspace.
   !!
   !! Takes the arguments of `rgpot_cuh2_force_ws` with `direction` and
   !! `product`, `3 * natoms` doubles each, in place of the forces and
   !! energy. Returns zero on success, non-zero on failure with the message
   !! available through `rgpot_fortran_workspace_error`; a NULL handle
   !! returns -1.
   function rgpot_cuh2_hvp_ws(handle, natoms, positions, atomic_numbers, cell, &
                              direction, product) result(status) &
      bind(c, name="rgpot_cuh2_hvp_ws")
      type(c_ptr), value :: handle
      integer(c_int), value, intent(in) :: natoms
      real(c_double), intent(in) :: positions(3, natoms)
      integer(c_int), intent(in) :: atomic_numbers(natoms)
      real(c_double), intent(in) :: cell(3, 3)
      real(c_double), intent(in) :: direction(3, natoms)
      real(c_double), intent(out) :: product(3, natoms)
      integer(c_int) :: status

      integer :: eval_status
      character(len=:), allocatable :: errmsg
      type(rgpot_workspace_t), pointer :: ws

      ws => workspace_from_c(handle)
      if (.not. associated(ws)) then
         status = -1_c_int
         return
      end if
      call ws_clear_error(ws)
      status = 0_c_int
      product = 0.0_c_double

      if (natoms < 1_c_int) return

      call cuh2_hessian_vector(positions, atomic_numbers, cell, params, ws%table, &
                               direction, product, eval_status, errmsg)
      if (eval_status /= 0) then
         call ws_set_error(ws, errmsg)
         status = int(eval_status, c_int)
      end if
   end function rgpot_cuh2_hvp_ws


   !> F2: default-initialise the saved parameter block once.
   !! The C++ face calls this under std::call_once before any _ws
   !! evaluation. A second call does not write.
   subroutine rgpot_cuh2_ensure_params() bind(c, name="rgpot_cuh2_ensure_params")
      if (.not. params_ready) then
         params = cuh2_params_t()
         params_ready = .true.
      end if
   end subroutine rgpot_cuh2_ensure_params

end module rgpot_cuh2_capi
