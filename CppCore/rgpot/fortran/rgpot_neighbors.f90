! MIT License
! Copyright 2023--present rgpot developers

!> Full neighbour tables in CSR form, built with rgpot's neighbour query.
!!
!! Every ported potential evaluates forces as a *gather*: atom `i` sums
!! the whole force acting on it from its own neighbours, and writes only
!! `f(:, i)`. Iterations of the atom loop then touch disjoint memory, so
!! they run under `do concurrent`. That needs a **full** list (both `i->j`
!! and `j->i`), with the pair vectors and distances already folded, so
!! the kernels never redo minimum image arithmetic.
!!
!! The search is `rgpot_nlist_fortran_build`. A positive skin keeps the
!! candidate list in an `rgpot_nlist_builder` and reuses it while every
!! atom has moved by at most half the skin. Each call still writes
!! vectors for the positions it was given and keeps only pairs strictly
!! inside the cutoff, which is the Verlet filter.
!!
!! Cell convention: C callers hand over nine doubles, row-major, one cell
!! vector per row. Fortran reads that same memory column-major, so a
!! `real(wp) :: cell(3, 3)` dummy sees one cell vector per *column*.
!! `vec(3, npairs)` is that same layout: column `p` is `r_j - r_i`.
module rgpot_neighbors
   use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, c_f_pointer, c_null_ptr, &
                                          c_null_char, c_char, c_int, c_double, &
                                          c_size_t, c_int32_t, c_associated
   use rgpot_kinds, only: wp, ip, lp
   implicit none
   private

   public :: neighbor_table_t

   !> Full neighbour list: for atom `i`, entries `row(i) : row(i+1) - 1`
   !! index into `idx`, `vec`, and `dist`.
   type :: neighbor_table_t
      !> Row starts, `natoms + 1` entries, one-based.
      integer(ip), allocatable :: row(:)
      !> Neighbour atom index for each entry, one-based.
      integer(ip), allocatable :: idx(:)
      !> `r_j - r_i` with the periodic shift folded in, `3 x nentries`.
      real(wp), allocatable :: vec(:, :)
      !> `|r_j - r_i|` for each entry.
      real(wp), allocatable :: dist(:)
      !> Atom count the table was built for.
      integer(ip) :: natoms = 0_ip
      !> Cutoff the table was built at.
      real(wp), private :: cutoff = -1.0_wp
      !> Verlet skin the builder caches with.
      real(wp), private :: skin = -1.0_wp
      !> Builder kept across calls. Reused while atoms stay within half the skin.
      type(c_ptr), private :: builder = c_null_ptr
   contains
      procedure :: build => neighbor_table_build
      procedure :: count_for => neighbor_table_count_for
      procedure :: release => neighbor_table_release
   end type neighbor_table_t

   interface
      function rgpot_nlist_fortran_open() bind(C, name="rgpot_nlist_fortran_open")
         import :: c_ptr
         implicit none
         type(c_ptr) :: rgpot_nlist_fortran_open
      end function rgpot_nlist_fortran_open

      subroutine rgpot_nlist_fortran_close(list) bind(C, name="rgpot_nlist_fortran_close")
         import :: c_ptr
         implicit none
         type(c_ptr), value :: list
      end subroutine rgpot_nlist_fortran_close

      function rgpot_nlist_fortran_build(list, positions, natoms, box9, cutoff, skin, &
                                         periodic, first, second, vector, distance, cap, &
                                         count, reused) &
         bind(C, name="rgpot_nlist_fortran_build")
         import :: c_ptr, c_double, c_size_t, c_int32_t, c_int
         implicit none
         type(c_ptr), value :: list
         real(c_double), intent(in) :: positions(*)
         integer(c_size_t), value :: natoms
         real(c_double), intent(in) :: box9(*)
         real(c_double), value :: cutoff, skin
         integer(c_int32_t), intent(in) :: periodic(*)
         type(c_ptr), value :: first, second, vector, distance
         integer(c_size_t), value :: cap
         integer(c_size_t), intent(out) :: count
         integer(c_int32_t), intent(out) :: reused
         integer(c_int32_t) :: rgpot_nlist_fortran_build
      end function rgpot_nlist_fortran_build

      function rgpot_nlist_strerror(code) bind(C, name="rgpot_nlist_strerror")
         import :: c_ptr, c_int32_t
         implicit none
         integer(c_int32_t), value :: code
         type(c_ptr) :: rgpot_nlist_strerror
      end function rgpot_nlist_strerror
   end interface

contains

   !> Rebuild the table for `positions` (3 x natoms) within `cutoff`.
   !!
   !! `periodic` defaults to fully periodic. `status` is zero on success;
   !! on failure the table is left empty and `errmsg` describes the fault.
   !! `skin` defaults to a tenth of the cutoff. The builder reuses its
   !! candidates while every atom has moved by at most half of that skin.
   subroutine neighbor_table_build(self, positions, cell, cutoff, status, &
                                   errmsg, periodic, skin)
      class(neighbor_table_t), intent(inout) :: self
      real(wp), intent(in), contiguous :: positions(:, :)
      real(wp), intent(in) :: cell(3, 3)
      real(wp), intent(in) :: cutoff
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg
      logical, intent(in), optional :: periodic(3)
      real(wp), intent(in), optional :: skin

      integer(c_int32_t) :: pbc(3), rc, reused
      integer(c_size_t) :: natoms_c, needed, cap
      integer(c_int), allocatable, target :: ifirst(:), isecond(:)
      real(c_double), allocatable, target :: cvec(:, :), cdist(:)
      real(wp) :: use_skin
      integer :: p, np, q, nkept
      integer(ip) :: i, j, natoms
      integer(ip), allocatable :: fill(:), by_j(:), jstart(:)
      type(c_ptr) :: msg

      status = 0
      errmsg = ""
      natoms = int(size(positions, 2), ip)

      if (size(positions, 1) /= 3) then
         status = 1
         errmsg = "rgpot_neighbors: positions must be shaped [3, natoms]"
         return
      end if

      pbc = 1_c_int32_t
      if (present(periodic)) then
         where (.not. periodic) pbc = 0_c_int32_t
      end if

      use_skin = 0.1_wp*cutoff
      if (present(skin)) use_skin = skin

      if (.not. c_associated(self%builder)) then
         self%builder = rgpot_nlist_fortran_open()
         if (.not. c_associated(self%builder)) then
            status = 7
            errmsg = "rgpot_neighbors: the neighbour builder could not be allocated"
            return
         end if
      end if

      natoms_c = int(natoms, c_size_t)
      needed = 0_c_size_t
      rc = rgpot_nlist_fortran_build(self%builder, positions, natoms_c, cell, &
                                     real(cutoff, c_double), real(use_skin, c_double), pbc, &
                                     c_null_ptr, c_null_ptr, c_null_ptr, c_null_ptr, &
                                     0_c_size_t, needed, reused)
      if (rc /= 0_c_int32_t) then
         status = int(rc)
         msg = rgpot_nlist_strerror(rc)
         errmsg = "rgpot_neighbors: "//c_string(msg)
         return
      end if

      np = int(needed)
      self%natoms = natoms
      self%cutoff = cutoff
      self%skin = use_skin

      if (np > 0) then
         allocate (ifirst(np), isecond(np), cvec(3, np), cdist(np))
         cap = int(np, c_size_t)
         rc = rgpot_nlist_fortran_build(self%builder, positions, natoms_c, cell, &
                                        real(cutoff, c_double), real(use_skin, c_double), pbc, &
                                        c_loc(ifirst), c_loc(isecond), c_loc(cvec), c_loc(cdist), &
                                        cap, needed, reused)
         if (rc /= 0_c_int32_t) then
            status = int(rc)
            msg = rgpot_nlist_strerror(rc)
            errmsg = "rgpot_neighbors: "//c_string(msg)
            return
         end if
         np = int(needed)
      end if

      if (allocated(self%row)) then
         if (size(self%row) /= natoms + 1) deallocate (self%row)
      end if
      if (.not. allocated(self%row)) allocate (self%row(natoms + 1))
      allocate (fill(natoms), source=0_ip)

      ! Rows ordered by partner index, so a reused candidate list and a
      ! fresh build sum forces in one order.
      allocate (by_j(max(np, 1)), jstart(natoms + 1))
      jstart = 0_ip
      self%row = 0_ip
      do p = 1, np
         i = int(ifirst(p), ip) + 1_ip
         j = int(isecond(p), ip) + 1_ip
         if (i == j) cycle
         self%row(i) = self%row(i) + 1_ip
         jstart(j) = jstart(j) + 1_ip
      end do

      call prefix_sum(self%row)
      call prefix_sum(jstart)

      nkept = int(self%row(natoms + 1) - 1_ip)
      do p = 1, np
         i = int(ifirst(p), ip) + 1_ip
         j = int(isecond(p), ip) + 1_ip
         if (i == j) cycle
         by_j(jstart(j)) = p
         jstart(j) = jstart(j) + 1_ip
      end do

      if (allocated(self%idx)) then
         if (size(self%idx) < self%row(natoms + 1) - 1_ip) then
            deallocate (self%idx, self%vec, self%dist)
         end if
      end if
      if (.not. allocated(self%idx)) then
         allocate (self%idx(max(int(self%row(natoms + 1)) - 1, 1)))
         allocate (self%vec(3, max(int(self%row(natoms + 1)) - 1, 1)))
         allocate (self%dist(max(int(self%row(natoms + 1)) - 1, 1)))
      end if

      do q = 1, nkept
         p = int(by_j(q))
         i = int(ifirst(p), ip) + 1_ip
         j = int(isecond(p), ip) + 1_ip
         associate (slot => self%row(i) + fill(i))
            self%idx(slot) = j
            self%vec(:, slot) = cvec(:, p)
            self%dist(slot) = cdist(p)
         end associate
         fill(i) = fill(i) + 1_ip
      end do
   end subroutine neighbor_table_build

   !> Free the builder; the table can be built again afterwards.
   subroutine neighbor_table_release(self)
      class(neighbor_table_t), intent(inout) :: self

      if (c_associated(self%builder)) call rgpot_nlist_fortran_close(self%builder)
      self%builder = c_null_ptr
      if (allocated(self%row)) deallocate (self%row)
      if (allocated(self%idx)) deallocate (self%idx)
      if (allocated(self%vec)) deallocate (self%vec)
      if (allocated(self%dist)) deallocate (self%dist)
      self%natoms = 0_ip
      self%cutoff = -1.0_wp
      self%skin = -1.0_wp
   end subroutine neighbor_table_release

   !> Number of neighbours atom `i` owns.
   pure function neighbor_table_count_for(self, i) result(n)
      class(neighbor_table_t), intent(in) :: self
      integer(ip), intent(in) :: i
      integer(ip) :: n

      n = self%row(i + 1_ip) - self%row(i)
   end function neighbor_table_count_for

   !> Turn per-atom counts into one-based row starts in place.
   pure subroutine prefix_sum(row)
      integer(ip), intent(inout) :: row(:)

      integer :: k
      integer(ip) :: running, held

      running = 1_ip
      do k = 1, size(row)
         held = row(k)
         row(k) = running
         running = running + held
      end do
   end subroutine prefix_sum

   function c_string(ptr) result(text)
      type(c_ptr), intent(in) :: ptr
      character(len=:), allocatable :: text
      character(kind=c_char), pointer :: raw(:)
      integer :: n, k

      if (.not. c_associated(ptr)) then
         text = "unknown neighbour error"
         return
      end if
      call c_f_pointer(ptr, raw, [4096])
      n = 0
      do while (raw(n + 1) /= c_null_char .and. n < 4096)
         n = n + 1
      end do
      allocate (character(len=n) :: text)
      do k = 1, n
         text(k:k) = raw(k)
      end do
   end function c_string

end module rgpot_neighbors
