! MIT License
! Copyright 2023--present rgpot developers
!
! Stillinger-Weber silicon. The kernel descends from eOn
! (https://github.com/TheochemUI/eOn, client/potentials/SW), BSD-3-Clause,
! copyright the eOn Development Team.
!
! PotType::SWSi uses eOn's rescaled epsilon, 2.16826 * 1.06767385
! = 2.3149945 eV. The product is copied from
! client/potentials/SW/SWFortran.f90 (file header "Last change:
! AA 3 Apr 2002"), which is the same at tags svn-Dec16_2020 and
! v2.8.0:
!
!   EPSILON = 2.16826D0 * 1.06767385d0, &  !* 1.07d0
!   LAMBDA  = 21.0d0,                   &  !* 1.07d0
!
! Lambda stays 21 and is not multiplied. Neither that file nor the
! rgpot port (7a3b5016) derives 1.06767385. The factor is untraced.
! PotType::SWSi1985 is equation (2.9) of Phys. Rev. B 31, 5262 (1985),
! doi:10.1103/PhysRevB.31.5262: 3.4723e-12 erg. With 1 erg = 1e-7 J
! and the CODATA electron-volt (1.602176634e-19 J) that is
! 2.16723920 eV. The ratio of this epsilon to that published value
! is 1.068176741337.

!> Stillinger-Weber potential in gather form.
!!
!! The published SW expression sums a two-body term over pairs and a
!! three-body term over triplets centred on each atom. Written as a
!! scatter (one triplet updating three atoms) the atom loop carries
!! dependencies; written as a gather each atom accumulates only its own
!! force and the loop runs under `do concurrent`.
!!
!! An atom `i` collects three-body force in two roles:
!!
!!   * as the centre of a triplet `(j, i, k)`, taking `-(f_j + f_k)`;
!!   * as an end of a triplet centred on a neighbour `m`, taking the
!!     end-atom force of `(i, m, k)` for every other neighbour `k` of `m`.
!!
!! The two roles together reproduce the scatter sum exactly, at the cost
!! of walking each neighbour's neighbour list. Energy accrues to the
!! centre atom only, so it is counted once.
!!
!! The strain derivative follows the energy: atom `i` adds d ⊗ dE/dd for
!! the folded bond vectors d of its pair halves and of the triplets it
!! centres (Thompson, Plimpton and Mattson, J. Chem. Phys. 131, 154107
!! (2009), doi:10.1063/1.3245303).
module rgpot_sw
   use rgpot_kinds, only: wp, ip, c_double, c_int
   use rgpot_neighbors, only: neighbor_table_t
   implicit none
   private

   public :: sw_params_t, sw_energy_forces

   !> Stillinger-Weber parameters. Defaults are eOn's silicon set.
   type :: sw_params_t
      real(wp) :: sigma = 2.0951_wp        !< Length scale (Angstrom).
      real(wp) :: alpha = 1.8_wp           !< Cutoff in units of sigma.
      !> eOn rescaled energy scale, 2.3149945 eV. Not the published
      !! 2.16723920 eV. See the module comment.
      real(wp) :: epsilon = 2.16826_wp*1.06767385_wp
      real(wp) :: lambda = 21.0_wp         !< Three-body strength.
      real(wp) :: a = 7.049556277_wp       !< Two-body prefactor.
      real(wp) :: beta = 0.6022245584_wp   !< Two-body repulsion weight.
      real(wp) :: gamma = 1.2_wp           !< Three-body decay.
      integer :: p = 4                     !< Two-body repulsion exponent.
      real(wp) :: s0 = 3.6_wp              !< Binary repulsive range (Angstrom).
      real(wp) :: a0 = 0.0_wp              !< Binary repulsive strength (eV).
   contains
      procedure :: cutoff => sw_cutoff
   end type sw_params_t

   !> Exponential arguments below this underflow to zero; guarding keeps
   !! `exp` out of its denormal range for pairs at the cutoff.
   real(wp), parameter :: exp_floor = -300.0_wp
   real(wp), parameter :: one_third = 1.0_wp/3.0_wp
   real(wp), parameter :: pi = acos(-1.0_wp)

contains

   !> Interaction cutoff, `alpha * sigma`.
   pure function sw_cutoff(self) result(rcut)
      class(sw_params_t), intent(in) :: self
      real(wp) :: rcut

      rcut = self%alpha*self%sigma
   end function sw_cutoff

   !> Two-body energy and the radial derivative that scales `rhat_ij`.
   pure subroutine two_body(par, rij, energy, dedr)
      type(sw_params_t), intent(in) :: par
      real(wp), intent(in) :: rij
      real(wp), intent(out) :: energy, dedr

      real(wp) :: inv_sigma, rho, one_o_a, one_o_a2, expo, r_to_minus_p
      real(wp) :: a_eps

      inv_sigma = 1.0_wp/par%sigma
      a_eps = par%a*par%epsilon
      rho = rij*inv_sigma
      one_o_a = 1.0_wp/(rho - par%alpha)
      one_o_a2 = one_o_a*one_o_a

      expo = 0.0_wp
      if (one_o_a > exp_floor) expo = exp(one_o_a)

      r_to_minus_p = rho**(-par%p)

      energy = a_eps*(par%beta*r_to_minus_p - 1.0_wp)*expo
      dedr = (one_o_a2*energy + a_eps*par%p*par%beta*r_to_minus_p*expo/rho) &
             *inv_sigma

      ! Optional short-range repulsion between unlike species. a0 is a
      ! switch rather than a measurement, so it is compared exactly.
      if (rij <= par%s0 .and. par%a0 /= 0.0_wp) then
         energy = energy + par%a0*(cos(pi*rij/par%s0) + 1.0_wp)
         dedr = dedr + par%a0*pi/par%s0*sin(pi*rij/par%s0)
      end if
   end subroutine two_body

   !> Three-body prefactor `epsilon * lambda * exp(gamma/(rho-alpha))` for
   !! one leg, plus the radial derivative factor that leg contributes.
   pure subroutine leg_factor(par, r, expo, dfac)
      type(sw_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp), intent(out) :: expo, dfac

      real(wp) :: rho, one_o_a, gam_o_a

      rho = r/par%sigma
      one_o_a = 1.0_wp/(rho - par%alpha)
      gam_o_a = par%gamma*one_o_a

      expo = 0.0_wp
      if (gam_o_a > exp_floor) expo = exp(gam_o_a)

      dfac = gam_o_a*one_o_a/par%sigma
   end subroutine leg_factor

   !> Force an end atom of the triplet `(end, centre, other)` receives.
   !!
   !! `u_end` and `u_other` are the unit vectors from the centre towards
   !! each end; `r_end` is the centre-to-end distance.
   pure function end_atom_force(par, u_end, r_end, u_other, r_other) result(f)
      type(sw_params_t), intent(in) :: par
      real(wp), intent(in) :: u_end(3), r_end, u_other(3), r_other
      real(wp) :: f(3)

      real(wp) :: expo_end, expo_other, dfac_end, dfac_other
      real(wp) :: fact, cosine, cos_shift, radial, angular, dcos(3)

      call leg_factor(par, r_end, expo_end, dfac_end)
      call leg_factor(par, r_other, expo_other, dfac_other)

      fact = par%epsilon*par%lambda*expo_end*expo_other
      cosine = dot_product(u_end, u_other)
      cos_shift = cosine + one_third

      radial = fact*dfac_end*cos_shift*cos_shift
      angular = 2.0_wp*fact*cos_shift
      dcos = (u_other - cosine*u_end)/r_end

      f = radial*u_end - angular*dcos
   end function end_atom_force

   !> Three-body energy of the triplet centred on the shared atom.
   pure function triplet_energy(par, u_j, r_j, u_k, r_k) result(e)
      type(sw_params_t), intent(in) :: par
      real(wp), intent(in) :: u_j(3), r_j, u_k(3), r_k
      real(wp) :: e

      real(wp) :: expo_j, expo_k, dfac_j, dfac_k, cos_shift

      call leg_factor(par, r_j, expo_j, dfac_j)
      call leg_factor(par, r_k, expo_k, dfac_k)

      cos_shift = dot_product(u_j, u_k) + one_third
      e = par%epsilon*par%lambda*expo_j*expo_k*cos_shift*cos_shift
   end function triplet_energy

   !> Energy and forces for `positions` (3 x natoms) in `cell`.
   !!
   !! `strain`, when present, receives dE/dε in eV for the right strain
   !! that sends each position and cell vector r to r (I + ε). The order
   !! is xx, yy, zz, yz, xz, xy, and the off-diagonal entries are
   !! d_a dE/dd_b with a < b, as in stress.hpp.
   subroutine sw_energy_forces(positions, cell, par, table, energy, forces, &
                               status, errmsg, strain)
      real(wp), intent(in), contiguous :: positions(:, :)
      real(wp), intent(in) :: cell(3, 3)
      type(sw_params_t), intent(in) :: par
      type(neighbor_table_t), intent(inout) :: table
      real(wp), intent(out) :: energy
      real(wp), intent(out), contiguous :: forces(:, :)
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg
      real(wp), intent(out), optional :: strain(6)

      integer(ip) :: natoms, i
      integer :: nw
      real(wp) :: rcut
      real(wp), allocatable :: e_atom(:), w_atom(:, :)

      natoms = int(size(positions, 2), ip)
      energy = 0.0_wp
      forces = 0.0_wp
      if (present(strain)) strain = 0.0_wp
      rcut = par%cutoff()

      call table%build(positions, cell, rcut, status, errmsg)
      if (status /= 0) return

      allocate (e_atom(natoms), source=0.0_wp)

      ! Without `strain`, w_atom has no rows and atom_contribution skips the
      ! virial. One call site keeps two_body inlined in the pair loop.
      nw = 0
      if (present(strain)) nw = 6
      allocate (w_atom(nw, natoms), source=0.0_wp)

      ! Atom loop writes only its own force column, energy slot and strain
      ! column, so the iterations are independent.
      do concurrent(i=1:natoms)
         call atom_contribution(par, table, i, e_atom(i), forces(:, i), &
                                w_atom(:, i))
      end do

      if (present(strain)) strain = sum(w_atom, dim=2)

      energy = sum(e_atom)
   end subroutine sw_energy_forces

   !> Add d ⊗ g to the six strain entries `w`, upper triangle d_a g_b.
   pure subroutine add_bond_strain(w, d, g)
      real(wp), intent(inout) :: w(6)
      real(wp), intent(in) :: d(3), g(3)

      w(1) = w(1) + d(1)*g(1)
      w(2) = w(2) + d(2)*g(2)
      w(3) = w(3) + d(3)*g(3)
      w(4) = w(4) + d(2)*g(3)
      w(5) = w(5) + d(1)*g(3)
      w(6) = w(6) + d(1)*g(2)
   end subroutine add_bond_strain

   !> Energy owned by atom `i` (its two-body halves plus the triplets it
   !! centres), the total force acting on it, and, when `w_i` has six
   !! entries, the strain derivative of that same energy. A zero-size
   !! `w_i` skips it.
   pure subroutine atom_contribution(par, table, i, e_i, f_i, w_i)
      type(sw_params_t), intent(in) :: par
      type(neighbor_table_t), intent(in) :: table
      integer(ip), intent(in) :: i
      real(wp), intent(out) :: e_i
      real(wp), intent(out) :: f_i(3)
      real(wp), intent(out), contiguous :: w_i(:)

      integer(ip) :: sj, sk, sm, sn, m, n
      real(wp) :: rij, rik, rim, rmn
      real(wp) :: u_ij(3), u_ik(3), u_mi(3), u_mn(3)
      real(wp) :: e_pair, dedr, f_j(3), f_k(3), w(6)
      logical :: with_strain

      e_i = 0.0_wp
      f_i = 0.0_wp
      w = 0.0_wp
      with_strain = size(w_i) > 0

      do sj = table%row(i), table%row(i + 1_ip) - 1_ip
         rij = table%dist(sj)
         u_ij = table%vec(:, sj)/rij

         ! Two-body: half the pair energy, and the whole radial force this
         ! neighbour exerts on i.
         call two_body(par, rij, e_pair, dedr)
         e_i = e_i + 0.5_wp*e_pair
         f_i = f_i - dedr*u_ij
         ! two_body returns dedr = -dE/dr, so the half pair has
         ! dE/d(vec) = -0.5 dedr u_ij.
         if (with_strain) then
            call add_bond_strain(w, table%vec(:, sj), -0.5_wp*dedr*u_ij)
         end if

         ! Three-body, i at the centre: every unordered pair of neighbours.
         ! There are two loops because a strain test inside the force-only
         ! loop slows that loop by several percent. sw_centred_body.inc is
         ! the one copy of the energy and the end-atom forces.
         if (with_strain) then
            do sk = sj + 1_ip, table%row(i + 1_ip) - 1_ip
               include 'sw_centred_body.inc'
               ! An end atom's force is -dE/d(its bond vector from i).
               call add_bond_strain(w, table%vec(:, sj), -f_j)
               call add_bond_strain(w, table%vec(:, sk), -f_k)
            end do
         else
            do sk = sj + 1_ip, table%row(i + 1_ip) - 1_ip
               include 'sw_centred_body.inc'
            end do
         end if
      end do

      ! Three-body, i at an end: triplets centred on each neighbour m.
      do sm = table%row(i), table%row(i + 1_ip) - 1_ip
         m = table%idx(sm)
         rim = table%dist(sm)
         ! Vector m -> i is the reverse of the stored i -> m vector.
         u_mi = -table%vec(:, sm)/rim
         do sn = table%row(m), table%row(m + 1_ip) - 1_ip
            n = table%idx(sn)
            if (n == i) cycle
            rmn = table%dist(sn)
            u_mn = table%vec(:, sn)/rmn
            f_i = f_i + end_atom_force(par, u_mi, rim, u_mn, rmn)
         end do
      end do

      if (with_strain) w_i = w
   end subroutine atom_contribution

end module rgpot_sw
