! MIT License
! Copyright 2023--present rgpot developers
!
! Lenosky silicon. The kernel and its spline tables descend from eOn
! (https://github.com/TheochemUI/eOn, client/potentials/Lenosky),
! BSD-3-Clause, copyright the eOn Development Team; the scaffolding around
! them came from S. Goedecker.

!> Lenosky cubic-spline MEAM silicon, force-matched to LDA, in gather form.
!!
!! Atom `i` carries a pair sum and an embedding term evaluated at a local
!! density built from two- and three-body pieces:
!!
!!     dens_i = sum_j rho(r_ij) + sum_{j<k} f(r_ij) f(r_ik) g(cos t_jik),
!!     E_i    = sum_j phi(r_ij)/2 + U(dens_i).
!!
!! Every ingredient is a cubic spline over a tabulated grid, linearly
!! extrapolated outside it.
!!
!! `dU/ddens` is not known until `dens_i` is complete, which is what stops
!! a single-pass gather. Evaluation therefore runs in three passes: the
!! densities, then the embedding derivative per atom, then the force
!! gather, in which atom `a` collects
!!
!!   * its own pair terms,
!!   * `-dU/ddens_a` times the density gradients of the terms it centres,
!!   * `+dU/ddens_m` times the gradients it contributes to each
!!     neighbour `m`'s density, both as `m`'s two-body partner and as an
!!     end of the triplets `m` centres.
!!
!! Each pass writes one slot per atom, so all three run under
!! `do concurrent`.
module rgpot_lenosky
   use rgpot_kinds, only: wp, ip
   use rgpot_neighbors, only: neighbor_table_t
   implicit none
   private

   public :: lenosky_params_t, lenosky_energy_forces

   !> Cubic spline on a uniform grid, with the derived step constants the
   !! tabulated data ships with.
   type :: spline_t
      real(wp) :: tmin    !< First knot.
      real(wp) :: tmax    !< Last knot.
      real(wp) :: hi      !< Reciprocal knot spacing.
      real(wp) :: hsixth  !< Spacing / 6.
      real(wp) :: h2sixth !< Spacing squared / 6.
      integer :: n        !< Knot count.
   end type spline_t

   !> Pair energy spline: 10 knots on [1.5_wp, 4.5_wp].
   real(wp), parameter :: cof_phi(0:9) = [ &
      6.92994_wp, -0.43995_wp, -1.70123_wp, -1.62473_wp, -0.99696_wp, &
      -0.27391_wp, -0.02499_wp, -0.01784_wp, -0.00961_wp, 0.0_wp]
   real(wp), parameter :: dof_phi(0:9) = [ &
      165.33229480429_wp, 39.415410391417_wp, 6.8710036300407_wp, &
      5.3406950884203_wp, 1.5347960162782_wp, -6.3347591535331_wp, &
      -1.7987794021458_wp, 0.47429676211617_wp, &
      -0.040087646318907_wp, -0.23942617684055_wp]
   type(spline_t), parameter :: sp_phi = spline_t( &
      tmin=1.5_wp, tmax=4.5_wp, &
      hi=3.0_wp, hsixth=0.0555555555555556_wp, &
      h2sixth=0.0185185185185185_wp, n=10)

   !> Two-body density spline: 11 knots on [1.5_wp, 3.5_wp].
   real(wp), parameter :: cof_rho(0:10) = [ &
      0.13747_wp, -0.14831_wp, -0.55972_wp, -0.7311_wp, -0.76283_wp, &
      -0.72918_wp, -0.6662_wp, -0.57328_wp, -0.4069_wp, -0.16662_wp, &
      0.0_wp]
   real(wp), parameter :: dof_rho(0:10) = [ &
      -3.2275496741918_wp, -6.4119006516165_wp, 10.030652280658_wp, &
      2.2937915289857_wp, 1.7416816033995_wp, 0.54648205741626_wp, &
      0.47189016693543_wp, 2.056957274842_wp, 2.3192807336964_wp, &
      -0.24908020962757_wp, -12.371959895186_wp]
   type(spline_t), parameter :: sp_rho = spline_t( &
      tmin=1.5_wp, tmax=3.5_wp, &
      hi=5.0_wp, hsixth=0.0333333333333333_wp, &
      h2sixth=0.00666666666666667_wp, n=11)

   !> Three-body radial factor spline: 10 knots on [1.5_wp, 3.5_wp].
   real(wp), parameter :: cof_fff(0:9) = [ &
      1.25031_wp, 0.86821_wp, 0.60846_wp, 0.48756_wp, 0.44163_wp, &
      0.3761_wp, 0.27145_wp, 0.14814_wp, 0.04855_wp, 0.0_wp]
   real(wp), parameter :: dof_fff(0:9) = [ &
      27.904652711432_wp, -4.5230754228635_wp, 5.0531739800222_wp, &
      1.1806545027747_wp, -0.66693699112098_wp, -0.89430653829079_wp, &
      -0.50891685571587_wp, 0.66278396115427_wp, 0.73976101109878_wp, &
      2.5795319944506_wp]
   type(spline_t), parameter :: sp_fff = spline_t( &
      tmin=1.5_wp, tmax=3.5_wp, &
      hi=4.5_wp, hsixth=0.037037037037037_wp, &
      h2sixth=0.00823045267489712_wp, n=10)

   !> Embedding energy vs density spline: 8 knots on [-1.77093_wp, 7.90852_wp].
   real(wp), parameter :: cof_uuu(0:7) = [ &
      -1.07493_wp, -0.20045_wp, 0.41422_wp, 0.87939_wp, 1.26689_wp, &
      1.62998_wp, 1.97738_wp, 2.39618_wp]
   real(wp), parameter :: dof_uuu(0:7) = [ &
      -0.14827125747284_wp, -0.14922155328475_wp, &
      -0.070113224223509_wp, -0.03944902034923_wp, &
      -0.015815242579643_wp, 0.026112640061855_wp, &
      -0.13786974745095_wp, 0.74941595372657_wp]
   type(spline_t), parameter :: sp_uuu = spline_t( &
      tmin=-1.77093_wp, tmax=7.90852_wp, &
      hi=0.723181585730594_wp, hsixth=0.230463095238095_wp, &
      h2sixth=0.31867942960034_wp, n=8)

   !> Three-body angular factor spline: 8 knots on [-1.0_wp, 0.80014_wp].
   real(wp), parameter :: cof_ggg(0:7) = [ &
      5.25416_wp, 2.35915_wp, 1.19595_wp, 1.22995_wp, 2.03565_wp, &
      3.42474_wp, 4.94859_wp, 5.61799_wp]
   real(wp), parameter :: dof_ggg(0:7) = [ &
      15.826876132396_wp, 31.176239377907_wp, 16.589446539683_wp, &
      11.08389250052_wp, 9.088721638386_wp, 5.4902279653967_wp, &
      -18.823313223755_wp, -7.7183416481005_wp]
   type(spline_t), parameter :: sp_ggg = spline_t( &
      tmin=-1.0_wp, tmax=0.80014_wp, &
      hi=3.88858644327663_wp, hsixth=0.0428604761904762_wp, &
      h2sixth=0.0110221225156463_wp, n=8)

   !> Interaction cutoff shared by the pair and density splines.
   real(wp), parameter :: default_cutoff = 4.5_wp

   !> Lenosky parameters. The splines are fixed tabulated data; the type
   !! carries what a caller may reasonably vary.
   type :: lenosky_params_t
      real(wp) :: cutoff = default_cutoff !< Neighbour cutoff (Angstrom).
   end type lenosky_params_t

contains

   !> Cubic spline value and derivative, linear outside the knot range.
   pure subroutine splint(sp, ya, y2a, x, y, yp)
      type(spline_t), intent(in) :: sp
      real(wp), intent(in) :: ya(0:), y2a(0:)
      real(wp), intent(in) :: x
      real(wp), intent(out) :: y, yp

      real(wp) :: tt, a, b, a2, b2, cof1, cof2, cof3, cof4
      integer :: klo, khi

      tt = (x - sp%tmin)*sp%hi

      if (x < sp%tmin) then
         yp = sp%hi*(ya(1) - ya(0)) - (y2a(1) + 2.0_wp*y2a(0))*sp%hsixth
         y = ya(0) + (x - sp%tmin)*yp
      else if (x > sp%tmax) then
         yp = sp%hi*(ya(sp%n - 1) - ya(sp%n - 2)) &
              + (2.0_wp*y2a(sp%n - 1) + y2a(sp%n - 2))*sp%hsixth
         y = ya(sp%n - 1) + (x - sp%tmax)*yp
      else
         ! Hold the interval one below the last knot so that `x == tmax`
         ! reads the final interval rather than one element past the table.
         ! The weights then select the last knot exactly, and the slope
         ! agrees with the upper extrapolation branch.
         klo = min(int(tt), sp%n - 2)
         khi = klo + 1
         b = tt - real(klo, wp)
         a = 1.0_wp - b
         a2 = a*a
         b2 = b*b
         cof1 = a*(a2 - 1.0_wp)
         cof2 = b*(b2 - 1.0_wp)
         cof3 = 3.0_wp*b2 - 1.0_wp
         cof4 = 3.0_wp*a2 - 1.0_wp
         y = a*ya(klo) + b*ya(khi) + (cof1*y2a(klo) + cof2*y2a(khi))*sp%h2sixth
         yp = sp%hi*(ya(khi) - ya(klo)) &
              + (cof3*y2a(khi) - cof4*y2a(klo))*sp%hsixth
      end if
   end subroutine splint

   !> Local density of atom `i`: two-body `rho` plus the three-body
   !! `f f g` sum over unordered neighbour pairs.
   pure function atom_density(table, i) result(dens)
      type(neighbor_table_t), intent(in) :: table
      integer(ip), intent(in) :: i
      real(wp) :: dens

      integer(ip) :: sj, sk
      real(wp) :: rij, rik, rho, rhop, fij, fijp, fik, fikp, gjik, gjikp
      real(wp) :: u_ij(3), u_ik(3)

      dens = 0.0_wp

      do sj = table%row(i), table%row(i + 1_ip) - 1_ip
         rij = table%dist(sj)
         call splint(sp_rho, cof_rho, dof_rho, rij, rho, rhop)
         dens = dens + rho

         call splint(sp_fff, cof_fff, dof_fff, rij, fij, fijp)
         u_ij = table%vec(:, sj)/rij

         do sk = sj + 1_ip, table%row(i + 1_ip) - 1_ip
            rik = table%dist(sk)
            call splint(sp_fff, cof_fff, dof_fff, rik, fik, fikp)
            u_ik = table%vec(:, sk)/rik
            call splint(sp_ggg, cof_ggg, dof_ggg, dot_product(u_ij, u_ik), &
                        gjik, gjikp)
            dens = dens + fij*fik*gjik
         end do
      end do
   end function atom_density

   !> Gradient of the triplet density term with respect to the two end
   !! atoms, given the centre's unit vectors and distances.
   !!
   !! `grad_j` is what the density of the centre changes by when end `j`
   !! moves; the centre itself takes the negated sum of both ends.
   pure subroutine triplet_gradients(u_ij, rij, u_ik, rik, grad_j, grad_k)
      real(wp), intent(in) :: u_ij(3), rij, u_ik(3), rik
      real(wp), intent(out) :: grad_j(3), grad_k(3)

      real(wp) :: fij, fijp, fik, fikp, gjik, gjikp, costheta, tt

      call splint(sp_fff, cof_fff, dof_fff, rij, fij, fijp)
      call splint(sp_fff, cof_fff, dof_fff, rik, fik, fikp)
      costheta = dot_product(u_ij, u_ik)
      call splint(sp_ggg, cof_ggg, dof_ggg, costheta, gjik, gjikp)
      tt = fij*fik

      grad_j = u_ij*(fijp*fik*gjik) &
               + (u_ik - u_ij*costheta)*(tt*gjikp/rij)
      grad_k = u_ik*(fikp*fij*gjik) &
               + (u_ij - u_ik*costheta)*(tt*gjikp/rik)
   end subroutine triplet_gradients

   !> Pair energy owned by `i` and the whole force acting on it.
   !!
   !! Unit vectors follow the tabulated kernel's convention: they point
   !! from the neighbour towards the atom whose density the term feeds,
   !! which is the direction its spline derivatives were fitted against.
   !! The table stores the opposite direction, so each one is negated at
   !! the point of use. Cosines are unaffected, which is why the energy
   !! survives the convention but the forces do not.
   pure subroutine atom_contribution(table, dudens, i, e_i, f_i)
      type(neighbor_table_t), intent(in) :: table
      real(wp), intent(in) :: dudens(:)
      integer(ip), intent(in) :: i
      real(wp), intent(out) :: e_i
      real(wp), intent(out) :: f_i(3)

      integer(ip) :: sj, sk, sm, sn, m, n
      real(wp) :: rij, rik, rim, rmn, e_phi, ep_phi, rho, rhop
      real(wp) :: u_ij(3), u_ik(3), u_mi(3), u_mn(3), grad_j(3), grad_k(3)

      e_i = 0.0_wp
      f_i = 0.0_wp

      do sj = table%row(i), table%row(i + 1_ip) - 1_ip
         rij = table%dist(sj)
         u_ij = -table%vec(:, sj)/rij

         ! Pair term: half the energy, the whole force this pair exerts.
         call splint(sp_phi, cof_phi, dof_phi, rij, e_phi, ep_phi)
         e_i = e_i + 0.5_wp*e_phi
         f_i = f_i - u_ij*ep_phi

         ! Own density, two-body part.
         call splint(sp_rho, cof_rho, dof_rho, rij, rho, rhop)
         f_i = f_i - dudens(i)*u_ij*rhop

         ! Own density, triplets centred here.
         do sk = sj + 1_ip, table%row(i + 1_ip) - 1_ip
            rik = table%dist(sk)
            u_ik = -table%vec(:, sk)/rik
            call triplet_gradients(u_ij, rij, u_ik, rik, grad_j, grad_k)
            f_i = f_i - dudens(i)*(grad_j + grad_k)
         end do
      end do

      ! Terms this atom contributes to its neighbours' densities.
      do sm = table%row(i), table%row(i + 1_ip) - 1_ip
         m = table%idx(sm)
         rim = table%dist(sm)
         u_mi = table%vec(:, sm)/rim

         call splint(sp_rho, cof_rho, dof_rho, rim, rho, rhop)
         f_i = f_i + dudens(m)*u_mi*rhop

         do sn = table%row(m), table%row(m + 1_ip) - 1_ip
            n = table%idx(sn)
            if (n == i) cycle
            rmn = table%dist(sn)
            u_mn = -table%vec(:, sn)/rmn
            call triplet_gradients(u_mi, rim, u_mn, rmn, grad_j, grad_k)
            f_i = f_i + dudens(m)*grad_j
         end do
      end do
   end subroutine atom_contribution

   !> Energy and forces for `positions` (3 x natoms) in `cell`.
   subroutine lenosky_energy_forces(positions, cell, par, table, energy, &
                                    forces, status, errmsg)
      real(wp), intent(in), contiguous :: positions(:, :)
      real(wp), intent(in) :: cell(3, 3)
      type(lenosky_params_t), intent(in) :: par
      type(neighbor_table_t), intent(inout) :: table
      real(wp), intent(out) :: energy
      real(wp), intent(out), contiguous :: forces(:, :)
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg

      integer(ip) :: natoms, i
      real(wp), allocatable :: dens(:), dudens(:), e_embed(:), e_pair(:)

      natoms = int(size(positions, 2), ip)
      energy = 0.0_wp
      forces = 0.0_wp

      call table%build(positions, cell, par%cutoff, status, errmsg)
      if (status /= 0) return

      allocate (dens(natoms), dudens(natoms), e_embed(natoms), e_pair(natoms))

      ! Pass one: local density of every atom.
      do concurrent(i=1:natoms)
         dens(i) = atom_density(table, i)
      end do

      ! Pass two: embedding energy and its derivative. Pass three needs
      ! the derivative at both ends of every pair, which is why the force
      ! cannot fold into pass one.
      do concurrent(i=1:natoms)
         call splint(sp_uuu, cof_uuu, dof_uuu, dens(i), e_embed(i), dudens(i))
      end do

      ! Pass three: pair energy and the force gather.
      do concurrent(i=1:natoms)
         call atom_contribution(table, dudens, i, e_pair(i), forces(:, i))
      end do

      energy = sum(e_pair) + sum(e_embed)
   end subroutine lenosky_energy_forces

end module rgpot_lenosky
