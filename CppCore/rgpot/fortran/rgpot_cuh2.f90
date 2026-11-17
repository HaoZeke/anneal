! MIT License
! Copyright 2023--present rgpot developers
!
! Copper-hydrogen EAM. The kernel descends from eOn
! (https://github.com/TheochemUI/eOn, client/potentials/CuH2), BSD-3-Clause,
! copyright the eOn Development Team.

!> Embedded-atom potential for Cu/H systems, in gather form.
!!
!! The EAM energy splits into a pair sum and an embedding sum,
!!
!!     U = sum_{i<j} phi_{Z_i Z_j}(r_ij) + sum_i F_{Z_i}(rho_i),
!!     rho_i = sum_{j /= i} rho_{Z_j}(r_ij),
!!
!! so the force on atom `i` needs `dF/drho` at both ends of every pair.
!! That fixes the shape of the evaluation: densities first, embedding
!! derivatives second, forces third. Each of the three passes writes one
!! slot per atom and runs under `do concurrent`.
!!
!! With a full neighbour list the force on `i` closes over its own row,
!!
!!     f_i = sum_j (r_j - r_i) * [ phi'/r + rho'_{Z_j}/r * F'(rho_i)
!!                                        + rho'_{Z_i}/r * F'(rho_j) ],
!!
!! and the bracket is symmetric under i <-> j, so the net force vanishes
!! to rounding without the loop ever touching `f_j`.
!!
!! Species come from atomic numbers rather than index ranges, so a caller
!! may interleave Cu and H in any order.
module rgpot_cuh2
   use rgpot_kinds, only: wp, ip
   use rgpot_neighbors, only: neighbor_table_t
   implicit none
   private

   public :: cuh2_params_t, cuh2_energy_forces, cuh2_hessian_vector
   public :: cuh2_z_cu, cuh2_z_h

   !> Atomic numbers this potential covers.
   integer(ip), parameter :: cuh2_z_cu = 29_ip
   integer(ip), parameter :: cuh2_z_h = 1_ip

   !> Internal species codes, dense so the inner loops branch on 1 or 2.
   integer(ip), parameter :: sp_cu = 1_ip
   integer(ip), parameter :: sp_h = 2_ip

   !> CuH2 embedded-atom parameters.
   !!
   !! Defaults are the eOn `pot.par` set: two exponentials per pair term,
   !! a scaled power-law-times-exponential density per species, and a
   !! polynomial embedding function (degree eight for Cu, five for H).
   !!
   !! The five `*cut_*` components hold the values of the unshifted pair
   !! and density functions at `rcut`. Subtracting them makes `phi` and
   !! `rho` vanish at the cutoff. They default to zero, which is what
   !! evaluating the unshifted functions requires, and `cut_shifted` fills
   !! them from a zeroed copy so it may be applied any number of times.
   type :: cuh2_params_t
      !> Cu-Cu pair term, `DA exp(-alphaA r) + DB exp(-alphaB r)`.
      real(wp) :: da_cucu = 2862.3_wp
      real(wp) :: alpha_a_cucu = 3.51236_wp
      real(wp) :: db_cucu = -109.107_wp
      real(wp) :: alpha_b_cucu = 1.75618_wp
      !> H-H pair term.
      real(wp) :: da_hh = 79.5013_wp
      real(wp) :: alpha_a_hh = 2.47961_wp
      real(wp) :: db_hh = -107.555_wp
      real(wp) :: alpha_b_hh = 2.99918_wp
      !> H-Cu pair term.
      real(wp) :: da_hcu = 86.1495_wp
      real(wp) :: alpha_a_hcu = 4.21154_wp
      real(wp) :: db_hcu = 15355.5_wp
      real(wp) :: alpha_b_hcu = 6.07604_wp
      !> Cu density, `scale r^neta (exp(-betaA r) + gamma exp(-betaB r))`.
      real(wp) :: scale_cu = 0.273072_wp
      real(wp) :: beta_a_cu = 3.69051_wp
      real(wp) :: beta_b_cu = 7.38101_wp
      real(wp) :: gamma_cu = 512.0_wp
      integer :: neta_cu = 6
      !> H density. `gamma_h` zero leaves a single exponential.
      real(wp) :: scale_h = 2.14389_wp
      real(wp) :: beta_a_h = 3.77701_wp
      real(wp) :: beta_b_h = 0.0_wp
      real(wp) :: gamma_h = 0.0_wp
      integer :: neta_h = 0
      !> Cu embedding polynomial coefficients, `f_cu(k)` on `rho**k`.
      real(wp) :: f_cu(8) = [-112.945_wp, 8510.04_wp, -261734.0_wp, &
                             4780090.0_wp, -52341900.0_wp, 339124000.0_wp, &
                             -1201500000.0_wp, 1796190000.0_wp]
      !> H embedding polynomial coefficients, `f_h(k)` on `rho**k`.
      real(wp) :: f_h(5) = [-81.7537_wp, 838.668_wp, -3952.35_wp, &
                            8767.87_wp, -6599.37_wp]
      !> Interaction cutoff (Angstrom).
      real(wp) :: rcut = 6.1_wp
      !> Pair energies at the cutoff, subtracted to make `phi(rcut) = 0`.
      real(wp) :: phicut_cucu = 0.0_wp
      real(wp) :: phicut_hcu = 0.0_wp
      real(wp) :: phicut_hh = 0.0_wp
      !> Densities at the cutoff, subtracted to make `rho(rcut) = 0`.
      real(wp) :: rhocut_cu = 0.0_wp
      real(wp) :: rhocut_h = 0.0_wp
   contains
      procedure :: cutoff => cuh2_cutoff
      procedure :: cut_shifted => cuh2_cut_shifted
   end type cuh2_params_t

contains

   !> Interaction cutoff.
   pure function cuh2_cutoff(self) result(rcut)
      class(cuh2_params_t), intent(in) :: self
      real(wp) :: rcut

      rcut = self%rcut
   end function cuh2_cutoff

   !> Copy of `self` with the five cutoff shifts filled from `rcut`.
   pure function cuh2_cut_shifted(self) result(par)
      class(cuh2_params_t), intent(in) :: self
      type(cuh2_params_t) :: par

      par = self

      par%phicut_cucu = 0.0_wp
      par%phicut_hcu = 0.0_wp
      par%phicut_hh = 0.0_wp
      par%rhocut_cu = 0.0_wp
      par%rhocut_h = 0.0_wp

      par%phicut_cucu = eam_phi_cucu(par, par%rcut)
      par%phicut_hcu = eam_phi_hcu(par, par%rcut)
      par%phicut_hh = eam_phi_hh(par, par%rcut)
      par%rhocut_cu = eam_rho_cu(par, par%rcut)
      par%rhocut_h = eam_rho_h(par, par%rcut)
   end function cuh2_cut_shifted

   ! ------------------------------------------------------------------
   ! Pair terms. Each derivative function returns `dphi/dr / r`, the
   ! factor that multiplies the pair vector `r_j - r_i` directly.
   ! ------------------------------------------------------------------

   !> Cu-Cu pair energy.
   pure elemental function eam_phi_cucu(par, r) result(phi)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: phi

      real(wp) :: term1, term2

      term1 = par%da_cucu*exp(-par%alpha_a_cucu*r)
      term2 = par%db_cucu*exp(-par%alpha_b_cucu*r)
      phi = term1 + term2 - par%phicut_cucu
   end function eam_phi_cucu

   !> Cu-Cu pair energy derivative over `r`.
   pure elemental function eam_dphi_cucu(par, r) result(dphi_r)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: dphi_r

      real(wp) :: term1, term2

      term1 = par%da_cucu*exp(-par%alpha_a_cucu*r)
      term2 = par%db_cucu*exp(-par%alpha_b_cucu*r)
      dphi_r = -(par%alpha_a_cucu*term1 + par%alpha_b_cucu*term2)/r
   end function eam_dphi_cucu

   !> H-Cu pair energy.
   pure elemental function eam_phi_hcu(par, r) result(phi)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: phi

      real(wp) :: term1, term2

      term1 = par%da_hcu*exp(-par%alpha_a_hcu*r)
      term2 = par%db_hcu*exp(-par%alpha_b_hcu*r)
      phi = term1 + term2 - par%phicut_hcu
   end function eam_phi_hcu

   !> H-Cu pair energy derivative over `r`.
   pure elemental function eam_dphi_hcu(par, r) result(dphi_r)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: dphi_r

      real(wp) :: term1, term2

      term1 = par%da_hcu*exp(-par%alpha_a_hcu*r)
      term2 = par%db_hcu*exp(-par%alpha_b_hcu*r)
      dphi_r = -(par%alpha_a_hcu*term1 + par%alpha_b_hcu*term2)/r
   end function eam_dphi_hcu

   !> H-H pair energy.
   pure elemental function eam_phi_hh(par, r) result(phi)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: phi

      real(wp) :: term1, term2

      term1 = par%da_hh*exp(-par%alpha_a_hh*r)
      term2 = par%db_hh*exp(-par%alpha_b_hh*r)
      phi = term1 + term2 - par%phicut_hh
   end function eam_phi_hh

   !> H-H pair energy derivative over `r`.
   pure elemental function eam_dphi_hh(par, r) result(dphi_r)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: dphi_r

      real(wp) :: term1, term2

      term1 = par%da_hh*exp(-par%alpha_a_hh*r)
      term2 = par%db_hh*exp(-par%alpha_b_hh*r)
      dphi_r = -(par%alpha_a_hh*term1 + par%alpha_b_hh*term2)/r
   end function eam_dphi_hh

   ! ------------------------------------------------------------------
   ! Densities. `eam_rho_*` is the electron density a neighbour of that
   ! species deposits at distance `r`; the derivative functions again
   ! return `drho/dr / r`. The shift is a constant, so it enters the
   ! density but not its derivative.
   ! ------------------------------------------------------------------

   !> Density contributed by a Cu neighbour.
   pure elemental function eam_rho_cu(par, r) result(rho)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: rho

      real(wp) :: term1, term2, term3

      term1 = exp(-par%beta_a_cu*r)
      term2 = par%gamma_cu*exp(-par%beta_b_cu*r)
      term3 = par%scale_cu*r**par%neta_cu

      rho = term3*(term1 + term2) - par%rhocut_cu
   end function eam_rho_cu

   !> Cu density derivative over `r`.
   pure elemental function eam_drho_cu(par, r) result(drho_r)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: drho_r

      real(wp) :: term1, term2, term3, rinv, rho_raw

      rinv = 1.0_wp/r
      term1 = exp(-par%beta_a_cu*r)
      term2 = par%gamma_cu*exp(-par%beta_b_cu*r)
      term3 = par%scale_cu*r**par%neta_cu
      rho_raw = term3*(term1 + term2)

      drho_r = (par%neta_cu*rho_raw*rinv &
                - term3*(par%beta_a_cu*term1 + par%beta_b_cu*term2))*rinv
   end function eam_drho_cu

   !> Density contributed by an H neighbour.
   pure elemental function eam_rho_h(par, r) result(rho)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: rho

      real(wp) :: term1, term2, term3

      term1 = exp(-par%beta_a_h*r)
      term2 = par%gamma_h*exp(-par%beta_b_h*r)
      term3 = par%scale_h*r**par%neta_h

      rho = term3*(term1 + term2) - par%rhocut_h
   end function eam_rho_h

   !> H density derivative over `r`.
   pure elemental function eam_drho_h(par, r) result(drho_r)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: drho_r

      real(wp) :: term1, term2, term3, rinv, rho_raw

      rinv = 1.0_wp/r
      term1 = exp(-par%beta_a_h*r)
      term2 = par%gamma_h*exp(-par%beta_b_h*r)
      term3 = par%scale_h*r**par%neta_h
      rho_raw = term3*(term1 + term2)

      drho_r = (par%neta_h*rho_raw*rinv &
                - term3*(par%beta_a_h*term1 + par%beta_b_h*term2))*rinv
   end function eam_drho_h

   ! ------------------------------------------------------------------
   ! Embedding functions of the accumulated density.
   ! ------------------------------------------------------------------

   !> Cu embedding energy.
   pure elemental function eam_f_cu(par, rho) result(embed)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: rho
      real(wp) :: embed

      real(wp) :: rho2, rho3, rho4, rho5, rho6, rho7, rho8

      rho2 = rho*rho
      rho3 = rho2*rho
      rho4 = rho3*rho
      rho5 = rho4*rho
      rho6 = rho5*rho
      rho7 = rho6*rho
      rho8 = rho7*rho

      embed = par%f_cu(1)*rho + par%f_cu(2)*rho2 + par%f_cu(3)*rho3 &
              + par%f_cu(4)*rho4 + par%f_cu(5)*rho5 + par%f_cu(6)*rho6 &
              + par%f_cu(7)*rho7 + par%f_cu(8)*rho8
   end function eam_f_cu

   !> Cu embedding derivative `dF/drho`.
   pure elemental function eam_df_cu(par, rho) result(dfdrho)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: rho
      real(wp) :: dfdrho

      real(wp) :: rho2, rho3, rho4, rho5, rho6, rho7

      rho2 = rho*rho
      rho3 = rho2*rho
      rho4 = rho3*rho
      rho5 = rho4*rho
      rho6 = rho5*rho
      rho7 = rho6*rho

      dfdrho = par%f_cu(1) + 2.0_wp*par%f_cu(2)*rho &
               + 3.0_wp*par%f_cu(3)*rho2 + 4.0_wp*par%f_cu(4)*rho3 &
               + 5.0_wp*par%f_cu(5)*rho4 + 6.0_wp*par%f_cu(6)*rho5 &
               + 7.0_wp*par%f_cu(7)*rho6 + 8.0_wp*par%f_cu(8)*rho7
   end function eam_df_cu

   !> H embedding energy.
   pure elemental function eam_f_h(par, rho) result(embed)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: rho
      real(wp) :: embed

      real(wp) :: rho2, rho3, rho4, rho5

      rho2 = rho*rho
      rho3 = rho2*rho
      rho4 = rho3*rho
      rho5 = rho4*rho

      embed = par%f_h(1)*rho + par%f_h(2)*rho2 + par%f_h(3)*rho3 &
              + par%f_h(4)*rho4 + par%f_h(5)*rho5
   end function eam_f_h

   !> H embedding derivative `dF/drho`.
   pure elemental function eam_df_h(par, rho) result(dfdrho)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: rho
      real(wp) :: dfdrho

      real(wp) :: rho2, rho3, rho4

      rho2 = rho*rho
      rho3 = rho2*rho
      rho4 = rho3*rho

      dfdrho = par%f_h(1) + 2.0_wp*par%f_h(2)*rho + 3.0_wp*par%f_h(3)*rho2 &
               + 4.0_wp*par%f_h(4)*rho3 + 5.0_wp*par%f_h(5)*rho4
   end function eam_df_h

   ! ------------------------------------------------------------------
   ! Second derivatives, for the Hessian-vector product. Unlike the first
   ! derivatives above these are plain `d2/dr2` and `d2/drho2`.
   ! ------------------------------------------------------------------

   !> Second derivative of `A exp(-a r) + B exp(-b r)`.
   pure elemental function two_exp_d2(da, alpha_a, db, alpha_b, r) result(d2)
      real(wp), intent(in) :: da, alpha_a, db, alpha_b, r
      real(wp) :: d2

      d2 = alpha_a*alpha_a*da*exp(-alpha_a*r) + alpha_b*alpha_b*db*exp(-alpha_b*r)
   end function two_exp_d2

   !> Cu-Cu pair energy second derivative.
   pure elemental function eam_d2phi_cucu(par, r) result(d2)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: d2

      d2 = two_exp_d2(par%da_cucu, par%alpha_a_cucu, par%db_cucu, par%alpha_b_cucu, r)
   end function eam_d2phi_cucu

   !> H-Cu pair energy second derivative.
   pure elemental function eam_d2phi_hcu(par, r) result(d2)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: d2

      d2 = two_exp_d2(par%da_hcu, par%alpha_a_hcu, par%db_hcu, par%alpha_b_hcu, r)
   end function eam_d2phi_hcu

   !> H-H pair energy second derivative.
   pure elemental function eam_d2phi_hh(par, r) result(d2)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: d2

      d2 = two_exp_d2(par%da_hh, par%alpha_a_hh, par%db_hh, par%alpha_b_hh, r)
   end function eam_d2phi_hh

   !> Second derivative of `scale r^n (exp(-betaA r) + gamma exp(-betaB r))`.
   !!
   !! With `g` the bracket, `rho'' = scale r^n (n(n-1) g / r^2 + 2 n g' / r
   !! + g'')`, written without `r^(n-2)` so that `n = 0` needs no case.
   pure elemental function power_exp_d2(scale, neta, beta_a, beta_b, gamma, r) &
      result(d2)
      real(wp), intent(in) :: scale, beta_a, beta_b, gamma, r
      integer, intent(in) :: neta
      real(wp) :: d2

      real(wp) :: ea, eb, g, dg, d2g, rinv, n

      rinv = 1.0_wp/r
      n = real(neta, wp)
      ea = exp(-beta_a*r)
      eb = gamma*exp(-beta_b*r)
      g = ea + eb
      dg = -(beta_a*ea + beta_b*eb)
      d2g = beta_a*beta_a*ea + beta_b*beta_b*eb
      d2 = scale*r**neta*(n*(n - 1.0_wp)*g*rinv*rinv + 2.0_wp*n*dg*rinv + d2g)
   end function power_exp_d2

   !> Cu density second derivative.
   pure elemental function eam_d2rho_cu(par, r) result(d2)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: d2

      d2 = power_exp_d2(par%scale_cu, par%neta_cu, par%beta_a_cu, par%beta_b_cu, &
                        par%gamma_cu, r)
   end function eam_d2rho_cu

   !> H density second derivative.
   pure elemental function eam_d2rho_h(par, r) result(d2)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: r
      real(wp) :: d2

      d2 = power_exp_d2(par%scale_h, par%neta_h, par%beta_a_h, par%beta_b_h, &
                        par%gamma_h, r)
   end function eam_d2rho_h

   !> Second derivative of the polynomial `sum_k coef(k) rho**k`.
   pure function polynomial_d2(coef, rho) result(d2)
      real(wp), intent(in) :: coef(:)
      real(wp), intent(in) :: rho
      real(wp) :: d2

      integer :: k

      d2 = 0.0_wp
      do k = size(coef), 2, -1
         d2 = d2*rho + real(k*(k - 1), wp)*coef(k)
      end do
   end function polynomial_d2

   !> Cu embedding second derivative.
   pure elemental function eam_d2f_cu(par, rho) result(d2)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: rho
      real(wp) :: d2

      d2 = polynomial_d2(par%f_cu, rho)
   end function eam_d2f_cu

   !> H embedding second derivative.
   pure elemental function eam_d2f_h(par, rho) result(d2)
      type(cuh2_params_t), intent(in) :: par
      real(wp), intent(in) :: rho
      real(wp) :: d2

      d2 = polynomial_d2(par%f_h, rho)
   end function eam_d2f_h

   ! ------------------------------------------------------------------
   ! Driver
   ! ------------------------------------------------------------------

   !> Energy and forces for `positions` (3 x natoms) in `cell`.
   !!
   !! `atomic_numbers` selects the species per atom and must hold only
   !! `cuh2_z_cu` or `cuh2_z_h`. Non-zero `status` leaves `energy` and
   !! `forces` zeroed and describes the fault in `errmsg`.
   subroutine cuh2_energy_forces(positions, atomic_numbers, cell, par, table, &
                                 energy, forces, status, errmsg)
      real(wp), intent(in), contiguous :: positions(:, :)
      integer(ip), intent(in), contiguous :: atomic_numbers(:)
      real(wp), intent(in) :: cell(3, 3)
      type(cuh2_params_t), intent(in) :: par
      type(neighbor_table_t), intent(inout) :: table
      real(wp), intent(out) :: energy
      real(wp), intent(out), contiguous :: forces(:, :)
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg

      type(cuh2_params_t) :: shifted
      integer(ip) :: natoms, i
      integer(ip), allocatable :: species(:)
      real(wp), allocatable :: rho(:), dfdrho(:), e_pair(:), e_embed(:)

      energy = 0.0_wp
      forces = 0.0_wp
      natoms = int(size(positions, 2), ip)

      call prepare(positions, atomic_numbers, cell, par, table, shifted, species, &
                   status, errmsg)
      if (status /= 0) return

      allocate (rho(natoms), dfdrho(natoms))
      allocate (e_pair(natoms), e_embed(natoms))

      ! Pass one: the density each atom sits in. Reads the table, writes
      ! one slot.
      do concurrent(i=1:natoms)
         rho(i) = atom_density(shifted, table, species, i)
      end do

      ! Pass two: embedding energy and its derivative, per atom. The
      ! derivative feeds pass three at both ends of every pair, which is
      ! why the force cannot be folded into pass one.
      do concurrent(i=1:natoms)
         if (species(i) == sp_cu) then
            e_embed(i) = eam_f_cu(shifted, rho(i))
            dfdrho(i) = eam_df_cu(shifted, rho(i))
         else
            e_embed(i) = eam_f_h(shifted, rho(i))
            dfdrho(i) = eam_df_h(shifted, rho(i))
         end if
      end do

      ! Pass three: pair energy and the whole force on each atom.
      do concurrent(i=1:natoms)
         call atom_force(shifted, table, species, dfdrho, i, e_pair(i), &
                         forces(:, i))
      end do

      energy = sum(e_pair) + sum(e_embed)
   end subroutine cuh2_energy_forces

   !> Hessian-vector product `H v` for `positions` (3 x natoms) in `cell`.
   !!
   !! The second derivatives of the energy of `cuh2_energy_forces`, with
   !! the same species, cutoff shift, neighbour table and faults. With
   !! `u_ij = (r_i - r_j)/r` and `K_f = f'' u u^T + (f'/r)(I - u u^T)` for
   !! a radial function `f`, row `i` of the product is
   !!
   !!     sum_j [K_phi + F'_i K_{rho_Zj} + F'_j K_{rho_Zi}] (v_i - v_j)
   !!     + sum_j u_ij [rho'_Zj F''_i drho_i + rho'_Zi F''_j drho_j],
   !!     drho_i = sum_j rho'_Zj u_ij . (v_i - v_j),
   !!
   !! the second line being the embedding curvature along the density
   !! change `drho = grad rho . v`. Like the force, each row closes over
   !! its own neighbours, after a pass for `drho` that needs every row.
   subroutine cuh2_hessian_vector(positions, atomic_numbers, cell, par, table, &
                                  direction, product, status, errmsg)
      real(wp), intent(in), contiguous :: positions(:, :)
      integer(ip), intent(in), contiguous :: atomic_numbers(:)
      real(wp), intent(in) :: cell(3, 3)
      type(cuh2_params_t), intent(in) :: par
      type(neighbor_table_t), intent(inout) :: table
      real(wp), intent(in), contiguous :: direction(:, :)
      real(wp), intent(out), contiguous :: product(:, :)
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg

      type(cuh2_params_t) :: shifted
      integer(ip) :: natoms, i
      integer(ip), allocatable :: species(:)
      real(wp), allocatable :: rho(:), dfdrho(:), d2fdrho(:), drho(:)

      product = 0.0_wp
      natoms = int(size(positions, 2), ip)

      call prepare(positions, atomic_numbers, cell, par, table, shifted, species, &
                   status, errmsg)
      if (status /= 0) return
      if (any(shape(direction) /= shape(positions))) then
         status = 4
         errmsg = "rgpot_cuh2: the direction does not match the positions"
         return
      end if

      allocate (rho(natoms), dfdrho(natoms), d2fdrho(natoms), drho(natoms))

      do concurrent(i=1:natoms)
         rho(i) = atom_density(shifted, table, species, i)
      end do

      do concurrent(i=1:natoms)
         if (species(i) == sp_cu) then
            dfdrho(i) = eam_df_cu(shifted, rho(i))
            d2fdrho(i) = eam_d2f_cu(shifted, rho(i))
         else
            dfdrho(i) = eam_df_h(shifted, rho(i))
            d2fdrho(i) = eam_d2f_h(shifted, rho(i))
         end if
      end do

      do concurrent(i=1:natoms)
         drho(i) = atom_density_change(shifted, table, species, direction, i)
      end do

      do concurrent(i=1:natoms)
         product(:, i) = atom_hessian_vector(shifted, table, species, dfdrho, &
                                             d2fdrho, drho, direction, i)
      end do
   end subroutine cuh2_hessian_vector

   !> Checks and neighbour table shared by the energy and its derivatives.
   subroutine prepare(positions, atomic_numbers, cell, par, table, shifted, &
                      species, status, errmsg)
      real(wp), intent(in), contiguous :: positions(:, :)
      integer(ip), intent(in), contiguous :: atomic_numbers(:)
      real(wp), intent(in) :: cell(3, 3)
      type(cuh2_params_t), intent(in) :: par
      type(neighbor_table_t), intent(inout) :: table
      type(cuh2_params_t), intent(out) :: shifted
      integer(ip), allocatable, intent(out) :: species(:)
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg

      integer(ip) :: natoms, nentries

      status = 0
      errmsg = ""
      natoms = int(size(positions, 2), ip)

      if (int(size(atomic_numbers), ip) /= natoms) then
         status = 1
         errmsg = "rgpot_cuh2: atomic_numbers holds "// &
                  count_text(int(size(atomic_numbers), ip))//" entries for "// &
                  count_text(natoms)//" atoms"
         return
      end if

      call classify(atomic_numbers, species, status, errmsg)
      if (status /= 0) return

      shifted = par%cut_shifted()

      call table%build(positions, cell, shifted%rcut, status, errmsg)
      if (status /= 0) return

      ! Every radial term divides by the pair distance, so a coincident
      ! pair is reported rather than left to seed a NaN in the forces.
      nentries = table%row(natoms + 1_ip) - 1_ip
      if (nentries > 0_ip) then
         if (minval(table%dist(1:nentries)) <= 0.0_wp) then
            status = 3
            errmsg = "rgpot_cuh2: two atoms occupy the same position"
            return
         end if
      end if
   end subroutine prepare

   !> Map atomic numbers onto the dense species codes.
   subroutine classify(atomic_numbers, species, status, errmsg)
      integer(ip), intent(in), contiguous :: atomic_numbers(:)
      integer(ip), allocatable, intent(out) :: species(:)
      integer, intent(out) :: status
      character(len=:), allocatable, intent(out) :: errmsg

      integer(ip) :: i

      status = 0
      errmsg = ""
      allocate (species(size(atomic_numbers)))

      do i = 1_ip, int(size(atomic_numbers), ip)
         select case (atomic_numbers(i))
         case (cuh2_z_cu)
            species(i) = sp_cu
         case (cuh2_z_h)
            species(i) = sp_h
         case default
            status = 2
            errmsg = "rgpot_cuh2: atom "//count_text(i)//" has atomic number "// &
                     count_text(atomic_numbers(i))// &
                     "; this potential covers Cu (29) and H (1) only"
            return
         end select
      end do
   end subroutine classify

   !> Decimal text of `number`, for error messages.
   function count_text(number) result(text)
      integer(ip), intent(in) :: number
      character(len=:), allocatable :: text

      character(len=12) :: buffer

      write (buffer, '(i0)') number
      text = trim(buffer)
   end function count_text

   !> Electron density at atom `i`, summed over its neighbours.
   !!
   !! Each neighbour deposits the density of *its own* species, so a Cu
   !! atom surrounded by H accumulates `rho_H`.
   pure function atom_density(par, table, species, i) result(rho_i)
      type(cuh2_params_t), intent(in) :: par
      type(neighbor_table_t), intent(in) :: table
      integer(ip), intent(in) :: species(:)
      integer(ip), intent(in) :: i
      real(wp) :: rho_i

      integer(ip) :: s
      real(wp) :: r

      rho_i = 0.0_wp

      do s = table%row(i), table%row(i + 1_ip) - 1_ip
         r = table%dist(s)
         ! A pair sitting exactly at the cutoff carries no shifted energy
         ! or density, but its derivative does not vanish there.
         if (r >= par%rcut) cycle

         if (species(table%idx(s)) == sp_cu) then
            rho_i = rho_i + eam_rho_cu(par, r)
         else
            rho_i = rho_i + eam_rho_h(par, r)
         end if
      end do
   end function atom_density

   !> Pair energy owned by atom `i` and the total force acting on it.
   !!
   !! The pair energy is halved because the full neighbour list visits
   !! each pair from both ends.
   pure subroutine atom_force(par, table, species, dfdrho, i, e_i, f_i)
      type(cuh2_params_t), intent(in) :: par
      type(neighbor_table_t), intent(in) :: table
      integer(ip), intent(in) :: species(:)
      real(wp), intent(in) :: dfdrho(:)
      integer(ip), intent(in) :: i
      real(wp), intent(out) :: e_i
      real(wp), intent(out) :: f_i(3)

      integer(ip) :: s, j, zi, zj
      real(wp) :: r, weight, drho_ij, drho_ji

      e_i = 0.0_wp
      f_i = 0.0_wp
      zi = species(i)

      do s = table%row(i), table%row(i + 1_ip) - 1_ip
         r = table%dist(s)
         if (r >= par%rcut) cycle

         j = table%idx(s)
         zj = species(j)

         if (zi == sp_cu .and. zj == sp_cu) then
            e_i = e_i + 0.5_wp*eam_phi_cucu(par, r)
            weight = eam_dphi_cucu(par, r)
         else if (zi == sp_h .and. zj == sp_h) then
            e_i = e_i + 0.5_wp*eam_phi_hh(par, r)
            weight = eam_dphi_hh(par, r)
         else
            e_i = e_i + 0.5_wp*eam_phi_hcu(par, r)
            weight = eam_dphi_hcu(par, r)
         end if

         ! `drho_ij` is what the neighbour deposits on `i`, `drho_ji` what
         ! `i` deposits on the neighbour; each is weighted by the
         ! embedding derivative of the atom that feels it.
         if (zj == sp_cu) then
            drho_ij = eam_drho_cu(par, r)
         else
            drho_ij = eam_drho_h(par, r)
         end if
         if (zi == sp_cu) then
            drho_ji = eam_drho_cu(par, r)
         else
            drho_ji = eam_drho_h(par, r)
         end if

         weight = weight + drho_ij*dfdrho(i) + drho_ji*dfdrho(j)
         f_i = f_i + weight*table%vec(:, s)
      end do
   end subroutine atom_force

   !> Density derivative `sum_j rho'_Zj u_ij . (v_i - v_j)` at atom `i`
   !! along `direction`. `table%vec` is `r_j - r_i = -r u_ij`.
   pure function atom_density_change(par, table, species, direction, i) &
      result(drho_i)
      type(cuh2_params_t), intent(in) :: par
      type(neighbor_table_t), intent(in) :: table
      integer(ip), intent(in) :: species(:)
      real(wp), intent(in) :: direction(:, :)
      integer(ip), intent(in) :: i
      real(wp) :: drho_i

      integer(ip) :: s, j
      real(wp) :: r, drho_r

      drho_i = 0.0_wp
      do s = table%row(i), table%row(i + 1_ip) - 1_ip
         r = table%dist(s)
         if (r >= par%rcut) cycle
         j = table%idx(s)
         if (species(j) == sp_cu) then
            drho_r = eam_drho_cu(par, r)
         else
            drho_r = eam_drho_h(par, r)
         end if
         drho_i = drho_i + drho_r*dot_product(table%vec(:, s), &
                                              direction(:, j) - direction(:, i))
      end do
   end function atom_density_change

   !> Row `i` of `H v`; see `cuh2_hessian_vector`.
   pure function atom_hessian_vector(par, table, species, dfdrho, d2fdrho, drho, &
                                     direction, i) result(hv_i)
      type(cuh2_params_t), intent(in) :: par
      type(neighbor_table_t), intent(in) :: table
      integer(ip), intent(in) :: species(:)
      real(wp), intent(in) :: dfdrho(:), d2fdrho(:), drho(:)
      real(wp), intent(in) :: direction(:, :)
      integer(ip), intent(in) :: i
      real(wp) :: hv_i(3)

      integer(ip) :: s, j, zi, zj
      real(wp) :: r, w(3), d(3)
      real(wp) :: radial, second, drho_ij, drho_ji, d2rho_ij, d2rho_ji

      hv_i = 0.0_wp
      zi = species(i)
      do s = table%row(i), table%row(i + 1_ip) - 1_ip
         r = table%dist(s)
         if (r >= par%rcut) cycle
         j = table%idx(s)
         zj = species(j)
         d = table%vec(:, s)
         w = direction(:, i) - direction(:, j)

         if (zi == sp_cu .and. zj == sp_cu) then
            radial = eam_dphi_cucu(par, r)
            second = eam_d2phi_cucu(par, r)
         else if (zi == sp_h .and. zj == sp_h) then
            radial = eam_dphi_hh(par, r)
            second = eam_d2phi_hh(par, r)
         else
            radial = eam_dphi_hcu(par, r)
            second = eam_d2phi_hcu(par, r)
         end if

         ! `_ij` is the density the neighbour deposits on `i`, `_ji` what
         ! `i` deposits on the neighbour, as in `atom_force`.
         if (zj == sp_cu) then
            drho_ij = eam_drho_cu(par, r)
            d2rho_ij = eam_d2rho_cu(par, r)
         else
            drho_ij = eam_drho_h(par, r)
            d2rho_ij = eam_d2rho_h(par, r)
         end if
         if (zi == sp_cu) then
            drho_ji = eam_drho_cu(par, r)
            d2rho_ji = eam_d2rho_cu(par, r)
         else
            drho_ji = eam_drho_h(par, r)
            d2rho_ji = eam_d2rho_h(par, r)
         end if

         radial = radial + dfdrho(i)*drho_ij + dfdrho(j)*drho_ji
         second = second + dfdrho(i)*d2rho_ij + dfdrho(j)*d2rho_ji
         hv_i = hv_i + (second - radial)*dot_product(d, w)/(r*r)*d + radial*w &
                - (drho_ij*d2fdrho(i)*drho(i) + drho_ji*d2fdrho(j)*drho(j))*d
      end do
   end function atom_hessian_vector

end module rgpot_cuh2
