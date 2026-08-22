//! Occupancy leave. Not Hyperband successive-halving.
//!
//! ## Packing identity (DECAF)
//!
//! Per-center SOAP class histograms. Two structures share a book *cell* iff
//! \(\|h-h'\|_1 \le\) [`crate::catalog::PACKING_MERGE`] \(= 0.20\). A
//! cell is not a packing: a live LJ75 book holds tens of icosahedral cells.
//!
//! A packing is a single-linkage community of cells at
//! [`crate::catalog::PACKING_LINK`] \(= 0.35\), and a structure belongs to
//! the community it chains to. No radius around one reference can do this
//! job. Measured on 69 quenched LJ75 icosahedral isomers within
//! \(8\varepsilon\) of the ico floor (`examples/decaf_packing_separator`):
//! the shelf reaches L1 \(0.56\) from its own reference while ico-Marks is
//! \(0.4267\), so the shelf spread straddles the gap. Under single linkage
//! the shelf chains into one community and Marks stands alone, and the same
//! radius keeps 153 of 154 LJ38 shelf isomers with ico and Oh on its own.
//!
//! ## Inverted Gelman--Rubin
//!
//! On the family-label series of assigned walks,
//! \(\hat R <\) [`crate::catalog::MIXED_RHAT`] \(= 1.01\)
//! (Vehtari et al. 2021) with
//! \(n_{\mathrm{assigned}} \ge 2\) is collapse: extras Leave. Distinct
//! packing labels are unmixed and do not force Leave.
//!
//! ## Ranking
//!
//! Champion of a family: lowest energy. Keep
//! \(\lfloor n_{\mathrm{extra}} / \eta \rfloor\) extras,
//! \(\eta =\) [`crate::catalog::REDUCTION_FACTOR`] \(= 3\). Surplus
//! Leave. Rank as soon as DECAF assigns a family. This is a keep
//! fraction, not a Li--Jamieson resource schedule.
//!
//! ## Superbasin
//!
//! Chatterjee--Voter AS-KMC: an intra-well hop with bias height
//! \(v_i \ge N_f w_0\) is frequent. Refuse the accept with probability
//! \(1/2\) (\(\alpha = 2\)). That is the well exit. A Feynman--Kac
//! epoch later resamples with the same ranking; it is not the exit.
//!
//! ## Leave start
//!
//! Another *packing community* already on file and packing not saturated:
//! take a catalog representative of the least-occupied community. Cells of
//! one packing are a superbasin; they are not OtherFamily, and a draw that
//! only clears the cell grain hands the extra an isomer of the packing it is
//! leaving. Champion leftover walks those isomers.
//!
//! Extra ArchiveHole is a rung of the Leave ladder
//! ([`crate::known_basin::leave_packing_rung`]): a covering direction of the
//! DECAF feature, pointed away from the packings on file, pulled back
//! through \(J_\mu\). Its size is one rung, not a grain. Wales and Doye
//! put the LJ75 ico-Marks barriers at 8.69 and 7.48 \(\varepsilon\), so a
//! quench from a Cartesian 0.35 cap is a projector onto the packing it
//! started in, whatever direction it took. A rung whose quench lands back in
//! the same packing is not refused into another hole of the same size: the
//! hop loop walks the rest of the ladder
//! ([`crate::known_basin::leave_packing_ladder`]) with the invert armed and
//! reports a refusal only when the ladder is spent. Occupancy extras do not
//! draw a random cluster.
//!
//! ## Modes
//!
//! Serial recommended: independent seeds, each tries to find the
//! published GM. Hit rate (58/72 Oh) is that mode.
//! Occupancy: the ensemble divides the PES. Each DECAF family has one
//! champion (lowest energy) that walks isomers. Extras of that family
//! Leave. Success is find-and-certify the putative GM, not every
//! replica landing on it.
//!
//! ## Stop
//!
//! A mixing certificate names a putative: uniquely deepest, occupant
//! mixed. A mixed competitor is required only when a competitor is
//! on file. A lone mixed floor is Gelman--Rubin on the sampled mode.
//! Unseen funnels are leftover-dwell and EI, not a second R-hat.
//! Packing Good--Turing names completeness of the seen codebook
//! (`packing_saturated`). Leftover-SOAP arrivals stay the hole
//! generator; leftover Good--Turing is not the stop. Hop re-observes
//! of the same well are not draws. A saturated packing census of
//! shallow families is not retire: extras keep Leaving so an unseen
//! funnel can still appear. Replicas retire when a mixing putative is
//! certified and that packing census is saturated and FunnelModel EI
//! on the seen packings is exhausted and the rematched family floor
//! is met. Leftover-SOAP hatches of a seen packing do not walk the
//! remaining force budget once the sparsified book has no holes. After the
//! book exists, single linkage at [`crate::catalog::PACKING_LINK`] folds
//! cells of one packing into one community. Leave continues only while that
//! compacted book still has holes: a community with no well arrivals, or
//! Chao1 incomplete on the merged well counts. A one-community book is
//! answered by the cells it folds, since Good--Turing on a single type has
//! \(n_1=0\) whatever the sample is and certifies nothing. Leftover FES
//! modes of one packing are not a packing hole. The floor is the Fiedler
//! split of the hop graph after DECAF labels the sides, the packing
//! community count of the book, or the book-map FES basin count. A Franzblau primitive-ring floor is
//! reported beside it and does not retire.
//! `CATALOG_MIN_FAMILIES` is an override.
//! Occupant \(\hat R\) uses
//! [`crate::catalog::CERTIFY_MIN_SAMPLES`] traces; two-point quenches
//! on two random-start families are not a certificate.
//! A published energy (Cambridge or otherwise) is a score, not a
//! stop. Leftover-SOAP saturation on one family is collapse, not
//! completeness.

/// Actions that must land off the occupied leftover-SOAP well.
pub fn is_occupancy_leave_action(action: &str) -> bool {
    matches!(
        action,
        "hyperband_reseed" | "catalog_leave" | "population_reseed"
    )
}

/// How an occupancy Leave updates the live chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyLeaveAdopt {
    /// Take the quenched trial as the live minimum.
    Quench,
    /// The quench sat in the occupied packing. Keep the unquenched hole step.
    HoleStep,
    /// Same-packing hole with the ladder spent. Do not adopt; the extra
    /// stays put and the hop loop draws another Leave.
    Refuse,
}

/// Where an occupancy extra goes when it Leaves.
///
/// OtherFamily is a draw from another packing community on the sparsified
/// book. That is how Leave divides the surface once there is something to
/// divide, and the draw has to clear the packing grain: a candidate that
/// only clears the cell grain is an isomer of the packing the extra is
/// leaving. ArchiveHole is a rung of the packing ladder in the DECAF
/// \(\nu=3\) feature (the same `local_nu3_z` rows as packing identity),
/// not SOAP leftover \(p_i-\mu\) and not a named morphology.
///
/// Walk is what a one-packing book asks for. An extra Leaves so the
/// ensemble stops spending two replicas on one funnel, and that trade is
/// only worth making when the Leave has somewhere to go. Measured on the
/// sealed LJ75 icosahedral minimum: a packing-ladder rung, sized by
/// bisection to spend exactly its barrier, quenches back to the floor it
/// started on at every rung from 1.32 to 42.3 \(\varepsilon\), which is
/// five times the ico-Marks barrier in one displacement. So with one
/// community on the book an ArchiveHole is a move with no measured yield,
/// and the replica taking it is not exploring, it is idling. Meanwhile the
/// walk does cross: 3 of 64 independent LJ75 walks at 400k evaluations
/// reached the Marks minimum, each at hop 4160, 6226 and 4411 of about
/// 11000. Coordination earns its keep by keeping walks off each other's
/// basins through the shared bias and the catalog, not by standing 47 of
/// 48 replicas still against a single funnel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyLeaveTarget {
    /// Coordinator has a representative of a different packing community.
    OtherFamily,
    /// First rung of the packing ladder in the DECAF \(\nu=3\) feature.
    ArchiveHole,
    /// Nothing to divide yet: keep walking, and let the shared bias hold
    /// this replica off the basins the others are on.
    Walk,
}

/// Consecutive leftover-sat bits kept for tests of the hatch filter.
/// Live dwell is [`leftover_hatch_stable`]: one more singleton cannot
/// lift \(\hat p_0\) through the unseen-mass ceiling.
pub const LEFTOVER_SAT_DWELL: usize = 5;

/// Bank and CSA turn FunnelModel EI on at three observed morphologies.
pub const OCCUPANCY_EI_MIN_OBS: usize = 3;

/// Jones, Schonlau & Welch (1998), *J. Global Optim.* 13:455-492:
/// \(\mathrm{EI}\to\max(f_{\min}-\mu,0)\) as \(\sigma\to 0\), so
/// \(\sigma=0\) and \(\mu\ge f_{\min}\) give \(\mathrm{EI}=0\) at
/// *observed* sites. Unseen families are leftover-dwell, not a
/// far-field GP probe. Exhausted when the FunnelModel has the bank
/// observation floor and the largest EI at observed packings is at
/// most the model's noise.
pub fn occupancy_ei_exhausted(max_ei: f64, n_obs: usize, noise: f64) -> bool {
    n_obs >= OCCUPANCY_EI_MIN_OBS && max_ei.is_finite() && max_ei <= noise
}

/// Leftover dwell from consecutive leftover-sat bits, newest last.
///
/// The last [`LEFTOVER_SAT_DWELL`] samples must all be saturated.
/// A one-shot nick, or a nick then a hatch, is not a dwell.
pub fn leftover_sat_dwell(consecutive: &[bool]) -> bool {
    consecutive.len() >= LEFTOVER_SAT_DWELL
        && consecutive
            .iter()
            .rev()
            .take(LEFTOVER_SAT_DWELL)
            .all(|&sat| sat)
}

/// Hatch-stable leftover dwell from the Good--Turing increment.
///
/// Good, I. J. (1953), *Biometrika* 40:237-264. A new singleton sends
/// \(n\mapsto n+1\), \(n_1\mapsto n_1+1\) and
/// \(\hat p_0'=(n_1+1)/(n+1)\). Under \(n_1\le n\) the hatch is the
/// larger estimator, so dwell is exactly \(\hat p_0' <\) ceiling
/// (`Hop.hatch_stable_iff_next`). No consecutive-record count.
pub fn leftover_hatch_stable(n: u64, n1: u64, ceiling: f64) -> bool {
    if n == 0 || n1 > n || !ceiling.is_finite() || ceiling <= 0.0 {
        return false;
    }
    (n1 + 1) as f64 / ((n + 1) as f64) < ceiling
}

/// Esty (1983), *Ann. Statist.* 11:905-912, one-sided 95% normal
/// quantile \(\Phi^{-1}(0.95)\).
pub const ESTY_Z95: f64 = 1.6448536269514722;

/// Esty variance of the Good--Turing unseen-mass estimator:
/// \(n_1/n^2 + 2n_2/n^2 - n_1^2/n^3\).
pub fn leftover_esty_var(n: u64, n1: u64, n2: u64) -> Option<f64> {
    if n == 0 {
        return None;
    }
    let nn = n as f64;
    let n1 = n1 as f64;
    let n2 = n2 as f64;
    Some((n1 / nn.powi(2) + 2.0 * n2 / nn.powi(2) - n1 * n1 / nn.powi(3)).max(0.0))
}

/// One-sided Esty upper bound \(\hat p_0 + z_{0.95}\sqrt{\mathrm{Var}}\).
pub fn leftover_esty_upper(n: u64, n1: u64, n2: u64) -> Option<f64> {
    let p0 = (n != 0).then(|| n1 as f64 / n as f64)?;
    let var = leftover_esty_var(n, n1, n2)?;
    Some(p0 + ESTY_Z95 * var.sqrt())
}

/// Leftover dwell: hatch-stable and the Esty upper bound sits under
/// the ceiling. \(n_1=0\) has variance 0, so the bound is 0.
pub fn leftover_esty_stable(n: u64, n1: u64, n2: u64, ceiling: f64) -> bool {
    leftover_hatch_stable(n, n1, ceiling)
        && leftover_esty_upper(n, n1, n2).is_some_and(|upper| upper < ceiling)
}

/// Leave destination. OtherFamily is a draw from another packing
/// community on the sparsified book (`communities >= 2`). Leftover
/// wells of one packing (`communities < 2`) stay ArchiveHole even
/// when DECAF split them. After packing saturation OtherFamily only
/// rematches communities on file. Packing saturation does not
/// disable that draw: ArchiveHole is only for a one-community book.
pub fn occupancy_leave_target(
    other_family_in_catalog: bool,
    packing_saturated: bool,
    packing_communities: usize,
) -> OccupancyLeaveTarget {
    let _ = packing_saturated;
    if packing_communities < 2 {
        // One packing on the book is nothing to divide, and the hole that
        // would be drawn here has no measured yield. Walk.
        return OccupancyLeaveTarget::Walk;
    }
    if other_family_in_catalog {
        OccupancyLeaveTarget::OtherFamily
    } else {
        OccupancyLeaveTarget::ArchiveHole
    }
}

/// Franzblau (1991), *Phys. Rev. B* 44:4925: a new ring class is a
/// packing signal SOAP \(L^1\) merge can miss. Icosahedra are
/// 5-ring rich; octahedra and Marks are not.
pub fn occupancy_ring_class_changed(origin: &[f64], trial: &[f64]) -> bool {
    match (
        occupancy_ring_profile(origin),
        occupancy_ring_profile(trial),
    ) {
        (Some(left), Some(right)) => ring_novelty(left, right) > 0,
        _ => false,
    }
}

/// Leave has found a new class: a different DECAF family, or a
/// different Franzblau ring histogram on the same SOAP merge radius.
///
/// The cell grain. Isomers of one packing clear it, so it names a book cell
/// and is not the Leave accept. [`occupancy_leave_new_packing`] is.
pub fn occupancy_leave_new_class(origin: &[f64], trial: &[f64]) -> bool {
    crate::catalog::different_decaf_family(origin, trial)
        || occupancy_ring_class_changed(origin, trial)
}

/// Leave has installed a packing: the quenched trial chains to no packing on
/// file, the origin included.
///
/// Single linkage at [`crate::catalog::PACKING_LINK`] over a throwaway book
/// of the origin, the published references and the trial. Measured on 69
/// quenched LJ75 icosahedral isomers, [`occupancy_leave_new_class`] adopts 6
/// of them on a ring-count change alone and adopts every one whose cell L1
/// clears the merge grain, which is how a run counts 47 distinct energies and
/// no Marks. Ring counts stay out of the accept: the shelf reaches ring-share
/// L1 0.0789 from its own reference while ico-Marks is 0.0824, and a 0.0035
/// margin does not decide a packing.
pub fn occupancy_leave_new_packing(origin: &[f64], trial: &[f64]) -> bool {
    crate::catalog::different_packing_family(origin, trial)
}

/// Occupancy Leave that walked away from the known packing mean
/// \(\mu_k\) is taken. A catalog_leave that sits at \(\mu_k\) is
/// refused. DECAF isomer grain is not this bit: ico isomers already
/// split at [`super::packing::PACKING_MERGE`] and a raw polish of
/// those walks back onto the occupied packing. Reseeds take the new
/// start even when the invert span did not rise.
pub fn occupancy_leave_adopt(action: &str, walked_off: bool) -> Option<OccupancyLeaveAdopt> {
    if !is_occupancy_leave_action(action) {
        return None;
    }
    if action == "catalog_leave" && !walked_off {
        Some(OccupancyLeaveAdopt::Refuse)
    } else {
        Some(OccupancyLeaveAdopt::Quench)
    }
}

/// Why occupancy may retire a replica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyCertificate {
    /// Occupant chains mixed onto a uniquely deepest attractor.
    /// A competitor, when one exists, must be mixed and less occupied.
    MixingCertified,
    /// Packing Good--Turing unseen mass is small, and the rematched
    /// family count meets the measured Fiedler-and-DECAF floor.
    CatalogSaturated,
}

impl OccupancyCertificate {
    /// Stable token for worker logs (`mixing` or `saturated`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MixingCertified => "mixing",
            Self::CatalogSaturated => "saturated",
        }
    }
}

/// Published energy match. A score for known hurdles, never a stop.
pub fn published_energy_score(best: f64, published: Option<f64>) -> bool {
    published.is_some_and(|target| best < target + 1e-4)
}

/// Default floor on rematched occupied families. `1` is Good--Turing
/// alone: stop when no new packings appear.
pub const DEFAULT_MIN_OCCUPIED_FAMILIES: usize = 1;

/// Bridge referee seam, kept as a named report threshold. It is not
/// a family-floor witness: without \(\lambda_2\) a positive
/// conductance is one community (`Hop.no_spectrum_is_one_community`).
pub const OCCUPANCY_SEAM_CONDUCTANCE: f64 = 0.1;

/// Family floor from the landscape Fiedler split after DECAF labels
/// the sides.
///
/// The hop-graph Fiedler vector names two leftover-SOAP communities.
/// Two when both live sides are nonempty, DECAF labels them as
/// distinct packings, and the cut is a bottleneck: conductance is
/// zero (disconnected) or strictly below algebraic connectivity.
/// On the normalised Laplacian \(\lambda_2\in[0,2]\), \(c<\lambda_2\)
/// implies the Cheeger bound \(c^2<2\lambda_2\)
/// (`Hop.code_cut_is_cheeger`). A missing \(\lambda_2\) is not a
/// substitute cut. A superbasin, a one-sided split, or a well-mixed
/// graph is one community. `CATALOG_MIN_FAMILIES` remains an override.
pub fn occupancy_family_floor(
    conductance: Option<f64>,
    algebraic_connectivity: Option<f64>,
    n_left: usize,
    n_right: usize,
    distinct_packing_sides: bool,
) -> usize {
    if n_left == 0 || n_right == 0 || !distinct_packing_sides {
        return DEFAULT_MIN_OCCUPIED_FAMILIES;
    }
    match (conductance, algebraic_connectivity) {
        (Some(c), _) if c == 0.0 => 2,
        (Some(c), Some(lambda)) if lambda.is_finite() && c < lambda => 2,
        _ => DEFAULT_MIN_OCCUPIED_FAMILIES,
    }
}

/// Secondary family floor from a 2-D folding of packing histograms.
///
/// Torgerson MDS of DECAF L1, then a 2-means split. Two when the sides
/// rematch to distinct packings and the centroids do not overlap.
/// Live rematch is the wrong input: fold the packing book so a family
/// extras have Left still sits on the map. A 2-means split is a
/// bipartition, not a leftover-cloud count; FES maxima stay the
/// landfold figure path. [`occupancy_sparsify_book`] merges leftover
/// wells on the same side and names holes. The hop-graph Fiedler
/// split stays the live floor; this count is the book floor.
pub fn occupancy_map_floor(xy: &[[f64; 2]], family: &[usize]) -> usize {
    occupancy_map_split(xy, family).0
}

/// Map floor and the 2-means side counts.
pub fn occupancy_map_split(xy: &[[f64; 2]], family: &[usize]) -> (usize, usize, usize) {
    if xy.len() < 2 || xy.len() != family.len() {
        return (1, xy.len(), 0);
    }
    let Some((left, right)) = two_means(xy) else {
        return (1, xy.len(), 0);
    };
    if left.is_empty() || right.is_empty() {
        return (1, xy.len(), 0);
    }
    if family.iter().all(|&f| f == family[0]) {
        return (1, left.len(), right.len());
    }
    let Some(left_family) = majority_family(family, &left) else {
        return (1, left.len(), right.len());
    };
    let Some(right_family) = majority_family(family, &right) else {
        return (1, left.len(), right.len());
    };
    if left_family == right_family {
        return (1, left.len(), right.len());
    }
    let (c0, r0) = centroid_radius(xy, &left);
    let (c1, r1) = centroid_radius(xy, &right);
    let gap = ((c0[0] - c1[0]).powi(2) + (c0[1] - c1[1]).powi(2)).sqrt();
    let floor = if gap > r0 + r1 { 2 } else { 1 };
    (floor, left.len(), right.len())
}

/// Torgerson 2-D map of DECAF histograms under L1, then
/// [`occupancy_map_floor`].
pub fn occupancy_landfold_floor(histograms: &[Vec<f64>], family: &[usize]) -> usize {
    occupancy_landfold_split(histograms, family).0
}

/// Landfold floor and 2-means side counts of DECAF histograms.
pub fn occupancy_landfold_split(
    histograms: &[Vec<f64>],
    family: &[usize],
) -> (usize, usize, usize) {
    let Some(xy) = occupancy_map_from_histograms(histograms) else {
        return (1, histograms.len(), 0);
    };
    let (floor, left_n, right_n) = occupancy_map_split(&xy, family);
    if floor == 1 {
        return (1, left_n, right_n);
    }
    let Some((left, right)) = two_means(&xy) else {
        return (1, left_n, right_n);
    };
    let Some(left_family) = majority_family(family, &left) else {
        return (1, left_n, right_n);
    };
    let Some(right_family) = majority_family(family, &right) else {
        return (1, left_n, right_n);
    };
    let Some(left_i) = left.iter().copied().find(|&i| family[i] == left_family) else {
        return (1, left_n, right_n);
    };
    let Some(right_i) = right.iter().copied().find(|&i| family[i] == right_family) else {
        return (1, left_n, right_n);
    };
    let between = super::packing::packing_distance(&histograms[left_i], &histograms[right_i]);
    let within = side_spread(histograms, &left).max(side_spread(histograms, &right));
    if between > within {
        (2, left_n, right_n)
    } else {
        (1, left_n, right_n)
    }
}

/// One occupied book cell on the landfold plane.
#[derive(Clone, Debug, PartialEq)]
pub struct OccupancyLandfoldPoint {
    /// Packing-book family index.
    pub family: usize,
    /// Sparsified landfold community (0 or 1).
    pub community: usize,
    /// Torgerson coordinates after the Ceriotti switch.
    pub xy: [f64; 2],
    /// Leftover-well arrivals credited to this family.
    pub wells: u64,
}

/// Landfold-sparsified occupancy book.
///
/// Leftover DECAF wells that land on the same side of a floor-1 split
/// are one packing. Floor-2 keeps two communities. [`Self::holes`] is
/// whether Leave should continue: a second community on the map with
/// no well arrivals, more FES basins than occupied well-sides, or
/// Chao1 incomplete on the merged well counts.
#[derive(Clone, Debug, PartialEq)]
pub struct OccupancyBookMap {
    /// Occupied book cells in family-index order.
    pub points: Vec<OccupancyLandfoldPoint>,
    /// Landfold floor (1 or 2).
    pub floor: usize,
    /// 2-means left count.
    pub left: usize,
    /// 2-means right count.
    pub right: usize,
    /// Sparsified community count.
    pub communities: usize,
    /// Merged leftover-well counts, one entry per community.
    pub community_wells: Vec<u64>,
    /// Good--Turing sample of the raw DECAF cells the communities fold.
    /// A one-community book is certified against this, not against its own
    /// single merged bin.
    pub cells: super::packing::GoodTuringSample,
    /// Book-map FES basin count ([`occupancy_fes`] on the Torgerson plane).
    pub fes_minima: usize,
    /// \(\Delta F/kT\) between the two deepest book-map FES basins.
    pub fes_delta: Option<f64>,
    /// Continue Leave while this is set.
    pub holes: bool,
}

impl OccupancyBookMap {
    /// Good--Turing sample of the sparsified well counts.
    pub fn sample(&self) -> super::packing::GoodTuringSample {
        super::packing::GoodTuringSample::from_counts(self.community_wells.iter().copied())
    }

    /// Chao1 completeness of the sparsified communities.
    pub fn saturated(&self) -> bool {
        !self.holes
    }
}

/// Whether the sparsified book still has holes extras should Leave into.
///
/// A packing community with no well arrivals is a hole. FES basins reopen a
/// hole only while a second community exists: leftover wells of one packing
/// stay one Chao1 sample even when the leftover cloud has more than one
/// density mode.
///
/// One community is the case that lied. A single merged bin has
/// \(n_1=0\) by construction, so Chao1 called it complete while the raw
/// DECAF cells underneath still carried half their mass in singletons.
/// Good--Turing on one type carries no information about unseen types, so a
/// one-community book falls back to the completeness of the finer codebook
/// it was folded from: while new cells keep arriving, packings may still be
/// arriving too.
pub fn occupancy_book_holes(
    communities: usize,
    community_wells: &[u64],
    fes_minima: usize,
    cells: super::packing::GoodTuringSample,
) -> bool {
    let occupied_sides = community_wells.iter().filter(|&&wells| wells > 0).count();
    if communities >= 2 && occupied_sides < 2 {
        return true;
    }
    if communities >= 2 && fes_minima > occupied_sides {
        return true;
    }
    if communities < 2 {
        return !cells.chao1_complete();
    }
    !super::packing::GoodTuringSample::from_counts(community_wells.iter().copied()).chao1_complete()
}

/// Compact the book into packing communities.
///
/// Single linkage at [`super::packing::PACKING_LINK`] over the DECAF cells,
/// so isomers of one packing chain together and a packing that chains to
/// nothing on file stands alone. The landfold Torgerson plane stays the
/// figure and the reported floor; it is no longer what counts packings, since
/// a forced 2-means bipartition of an isomer cloud has only ever one answer.
/// Leave continues while [`OccupancyBookMap::holes`].
pub fn occupancy_sparsify_book(
    histograms: &[Vec<f64>],
    family: &[usize],
    wells: &[u64],
) -> OccupancyBookMap {
    let n = histograms.len();
    let wells: Vec<u64> = if wells.len() == n {
        wells.to_vec()
    } else {
        vec![0; n]
    };
    let cells = super::packing::GoodTuringSample::from_counts(wells.iter().copied());
    if n == 0 {
        return OccupancyBookMap {
            points: Vec::new(),
            floor: 1,
            left: 0,
            right: 0,
            communities: 0,
            community_wells: Vec::new(),
            cells,
            fes_minima: 0,
            fes_delta: None,
            holes: occupancy_book_holes(0, &[], 0, cells),
        };
    }
    if n == 1 {
        let community_wells = vec![wells[0]];
        return OccupancyBookMap {
            points: vec![OccupancyLandfoldPoint {
                family: family.first().copied().unwrap_or(0),
                community: 0,
                xy: [0.0, 0.0],
                wells: wells[0],
            }],
            floor: 1,
            left: 1,
            right: 0,
            communities: 1,
            holes: occupancy_book_holes(1, &community_wells, 1, cells),
            community_wells,
            cells,
            fes_minima: 1,
            fes_delta: None,
        };
    }
    let (floor, left, right) = occupancy_landfold_split(histograms, family);
    let xy = occupancy_map_from_histograms(histograms).unwrap_or_else(|| vec![[0.0, 0.0]; n]);
    let community = super::packing::packing_communities(histograms);
    let n_communities = community.iter().copied().max().map_or(0, |last| last + 1);
    let mut community_wells = vec![0u64; n_communities];
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let c = community[i].min(n_communities.saturating_sub(1));
        community_wells[c] = community_wells[c].saturating_add(wells[i]);
        points.push(OccupancyLandfoldPoint {
            family: family.get(i).copied().unwrap_or(i),
            community: c,
            xy: xy[i],
            wells: wells[i],
        });
    }
    let weights: Vec<f64> = wells.iter().map(|&count| count.max(1) as f64).collect();
    let fes = occupancy_fes(&xy, Some(&weights)).unwrap_or(OccupancyFes {
        minima: 1,
        delta: None,
    });
    OccupancyBookMap {
        points,
        floor,
        left,
        right,
        communities: n_communities,
        holes: occupancy_book_holes(n_communities, &community_wells, fes.minima, cells),
        community_wells,
        cells,
        fes_minima: fes.minima,
        fes_delta: fes.delta,
    }
}

/// Landfold-sparsify the occupied packing book.
pub fn occupancy_sparsify_packing(book: &super::packing::PackingBook) -> OccupancyBookMap {
    let occupied = book.occupied_histograms();
    let histograms: Vec<Vec<f64>> = occupied
        .iter()
        .map(|(_, histogram)| histogram.clone())
        .collect();
    let family: Vec<usize> = occupied.iter().map(|(index, _)| *index).collect();
    let wells: Vec<u64> = family
        .iter()
        .map(|&index| book.well_visits_of(index))
        .collect();
    occupancy_sparsify_book(&histograms, &family, &wells)
}

/// Occupancy free energy. \(F/kT = -\ln(\rho/\rho_{\max})\).
///
/// Discrete packing FES uses leftover-well counts per DECAF family.
/// The landfold-map FES uses a Gaussian KDE on the Torgerson plane.
/// Equal occupancy is \(\Delta F = 0\), not a DECAF L1 split. This
/// does not retire.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OccupancyFes {
    /// Local minima of \(F/kT\).
    pub minima: usize,
    /// \(\Delta F/kT\) between the two deepest minima.
    pub delta: Option<f64>,
}

/// Invalid input to the continuous occupancy free-energy estimator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OccupancyFesError {
    /// A landfold coordinate is NaN or infinite.
    #[error("landfold point {index} is not finite")]
    NonFinitePoint {
        /// Zero-based point index.
        index: usize,
    },
    /// A weight vector must have one entry per landfold point.
    #[error("weight count {weights} does not match point count {points}")]
    WeightCountMismatch {
        /// Number of landfold points.
        points: usize,
        /// Number of supplied weights.
        weights: usize,
    },
    /// Weights are finite and non-negative.
    #[error("landfold weight {index} is not finite and non-negative")]
    InvalidWeight {
        /// Zero-based weight index.
        index: usize,
    },
    /// A non-empty sample needs positive density mass.
    #[error("landfold sample has no positive weight")]
    NoPositiveWeight,
    /// The histograms do not define a finite Torgerson map.
    #[error("histograms do not define a finite landfold map")]
    MapUnavailable,
}

const FES_MEAN_SHIFT_ITERATIONS: usize = 512;
const FES_MEAN_SHIFT_TOLERANCE: f64 = 1e-10;
const FES_MODE_MERGE_TOLERANCE: f64 = 1e-6;
const FES_DIAMETER_BANDWIDTH: f64 = 1.0 / 3.0;

/// Discrete packing \(\Delta F/kT = \ln(n_{\max}/n_2)\) from leftover-well
/// counts. None when fewer than two occupied families have wells.
pub fn occupancy_fes_delta(counts: &[u64]) -> Option<f64> {
    let mut occupied: Vec<u64> = counts.iter().copied().filter(|&n| n > 0).collect();
    if occupied.len() < 2 {
        return None;
    }
    occupied.sort_unstable();
    let n_max = occupied[occupied.len() - 1] as f64;
    let n_second = occupied[occupied.len() - 2] as f64;
    if n_max <= 0.0 || n_second <= 0.0 {
        return None;
    }
    Some((n_max / n_second).ln())
}

/// Map free energy from landfold points, optional per-point weights.
///
/// Density is a Gaussian KDE. \(F_i/kT = -\ln(\rho_i/\rho_{\max})\).
/// Minima of \(F\) are continuous maxima of \(\rho\), located by mean shift
/// from every observed point and merged at a scale-relative tolerance.
/// Bandwidth is \(\max(\mathrm{median\ NN}, \mathrm{diameter}/3)\), broad
/// enough that a connected interpolant chain remains a single basin.
///
/// Returns an error for non-finite coordinates, malformed weights, or a
/// non-empty sample with no positive density mass.
pub fn occupancy_fes(
    xy: &[[f64; 2]],
    weights: Option<&[f64]>,
) -> Result<OccupancyFes, OccupancyFesError> {
    let n = xy.len();
    if let Some(index) = xy
        .iter()
        .position(|point| point.iter().any(|value| !value.is_finite()))
    {
        return Err(OccupancyFesError::NonFinitePoint { index });
    }
    if let Some(values) = weights {
        if values.len() != n {
            return Err(OccupancyFesError::WeightCountMismatch {
                points: n,
                weights: values.len(),
            });
        }
        if let Some(index) = values
            .iter()
            .position(|&weight| !weight.is_finite() || weight < 0.0)
        {
            return Err(OccupancyFesError::InvalidWeight { index });
        }
        if n > 0 && values.iter().all(|&weight| weight == 0.0) {
            return Err(OccupancyFesError::NoPositiveWeight);
        }
    }
    if n < 2 {
        return Ok(OccupancyFes {
            minima: 1,
            delta: None,
        });
    }
    let mut nearest = Vec::with_capacity(n);
    let mut diameter = 0.0_f64;
    for i in 0..n {
        let mut best = f64::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = map_dist(xy[i], xy[j]);
            best = best.min(d);
            if j > i {
                diameter = diameter.max(d);
            }
        }
        nearest.push(best);
    }
    nearest.sort_by(|a, b| a.total_cmp(b));
    let sigma = nearest[n / 2]
        .max(FES_DIAMETER_BANDWIDTH * diameter)
        .max(1e-12);
    let sample_weight = |i: usize| {
        weights
            .and_then(|values| values.get(i).copied())
            .unwrap_or(1.0)
    };
    let mut modes: Vec<[f64; 2]> = Vec::new();
    for &seed in xy {
        let mut center = seed;
        for _ in 0..FES_MEAN_SHIFT_ITERATIONS {
            let mut total = 0.0;
            let mut next = [0.0, 0.0];
            for (i, &point) in xy.iter().enumerate() {
                let weight = sample_weight(i);
                if !weight.is_finite() || weight <= 0.0 {
                    continue;
                }
                let ratio = map_dist(center, point) / sigma;
                let kernel_weight = weight * (-0.5 * ratio * ratio).exp();
                total += kernel_weight;
                next[0] += kernel_weight * point[0];
                next[1] += kernel_weight * point[1];
            }
            if total <= 0.0 {
                break;
            }
            next[0] /= total;
            next[1] /= total;
            let shift = map_dist(center, next);
            center = next;
            if shift <= FES_MEAN_SHIFT_TOLERANCE * sigma {
                break;
            }
        }
        if modes
            .iter()
            .all(|&mode| map_dist(mode, center) > FES_MODE_MERGE_TOLERANCE * sigma)
        {
            modes.push(center);
        }
    }
    let density_at = |point: [f64; 2]| {
        xy.iter()
            .enumerate()
            .filter_map(|(i, &sample)| {
                let weight = sample_weight(i);
                if !weight.is_finite() || weight <= 0.0 {
                    return None;
                }
                let ratio = map_dist(point, sample) / sigma;
                Some(weight * (-0.5 * ratio * ratio).exp())
            })
            .sum::<f64>()
    };
    let density: Vec<f64> = modes.iter().copied().map(density_at).collect();
    let rho_max = density.iter().copied().fold(0.0_f64, f64::max);
    if rho_max <= 0.0 {
        return Err(OccupancyFesError::NoPositiveWeight);
    }
    let mut minima_f: Vec<f64> = density
        .into_iter()
        .filter(|&rho| rho > 0.0)
        .map(|rho| -(rho / rho_max).ln())
        .collect();
    minima_f.sort_by(|a, b| a.total_cmp(b));
    let minima = minima_f.len().max(1);
    let delta = if minima_f.len() >= 2 {
        Some(minima_f[1] - minima_f[0])
    } else {
        None
    };
    Ok(OccupancyFes { minima, delta })
}

/// Landfold map of DECAF histograms, then [`occupancy_fes`].
pub fn occupancy_fes_from_histograms(
    histograms: &[Vec<f64>],
) -> Result<OccupancyFes, OccupancyFesError> {
    if histograms.len() < 2 {
        if histograms.iter().flatten().any(|value| !value.is_finite()) {
            return Err(OccupancyFesError::MapUnavailable);
        }
        return Ok(OccupancyFes {
            minima: 1,
            delta: None,
        });
    }
    let xy = occupancy_map_from_histograms(histograms).ok_or(OccupancyFesError::MapUnavailable)?;
    occupancy_fes(&xy, None)
}

fn map_dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// Secondary family floor from Franzblau primitive-ring profiles.
///
/// One (triangles, squares, pentagons) per occupied DECAF family.
/// Two when two occupied families differ in those counts; one when
/// every occupied family shares a profile, so leftover ico wells of
/// one packing stay one community. This is not the hop-graph Fiedler
/// split and it does not retire.
pub fn occupancy_ring_floor(profiles: &[(usize, usize, usize)]) -> usize {
    occupancy_ring_split(profiles).0
}

/// Ring floor, distinct-profile count, and profile count.
pub fn occupancy_ring_split(profiles: &[(usize, usize, usize)]) -> (usize, usize, usize) {
    let n = profiles.len();
    let mut distinct = std::collections::BTreeSet::new();
    for &profile in profiles {
        distinct.insert(profile);
    }
    let n_distinct = distinct.len();
    let floor = if n_distinct >= 2 { 2 } else { 1 };
    (floor, n_distinct, n)
}

/// Franzblau (tri, sq, pent) at [`crate::structure::RING_CUTOFF_SCALE`]
/// times the structure's median nearest-neighbour distance.
pub fn occupancy_ring_profile(coordinates: &[f64]) -> Option<(usize, usize, usize)> {
    occupancy_ring_census(coordinates).map(|census| census.profile)
}

/// Contact-graph compactness of one quenched structure.
///
/// A cluster is one connected first-neighbour component with a cycle
/// (Franzblau ring, or at least `N` contacts). A path or a forest is
/// not a cluster: \(e\le N-c\) and no primitive ring. A straight
/// chain saturates \(R_g^2=a^2(N^2-1)/12\). GMIN's spherical
/// container is the same refusal of evaporated or unravelled atoms.
#[derive(Debug, Clone, PartialEq)]
pub struct OccupancyCompact {
    /// Atom count.
    pub n: usize,
    /// First-neighbour connected components.
    pub components: usize,
    /// Undirected contact edges.
    pub edges: usize,
    /// Franzblau 3-, 4- and 5-rings summed.
    pub rings: usize,
    /// Squared radius of gyration.
    pub rg2: f64,
    /// Straight-path \(R_g^2\) at the structure's median NN.
    pub path_rg2: f64,
    /// Largest atom–COM distance over \(N^{1/3}\). Published LJ minima
    /// sit in \(0.46\)–\(0.63\); the hop container is \(0.9\).
    pub rmax_over_cbrt: f64,
}

/// Hop `contain` sphere in units of \(\sigma N^{1/3}\). Published
/// compact minima are \(0.46\)–\(0.63\); a straight chain is \(\sim 2.5\).
pub const COMPACT_RMAX_OVER_CBRT: f64 = 0.9;

impl OccupancyCompact {
    /// Connected contact graph with a cycle, inside the hop container.
    /// Not a chain, not fragments, not an unravelled ring.
    pub fn is_cluster(&self) -> bool {
        self.n >= 3
            && self.components == 1
            && (self.rings > 0 || self.edges >= self.n)
            && self.rmax_over_cbrt < COMPACT_RMAX_OVER_CBRT
    }

    /// One component, a forest, no primitive ring: a path or a tree.
    pub fn is_pathlike(&self) -> bool {
        self.components == 1 && self.rings == 0 && self.edges + 1 <= self.n
    }
}

/// Compactness census at the Franzblau first-neighbour cutoff.
pub fn occupancy_compact(coordinates: &[f64]) -> Option<OccupancyCompact> {
    let n = coordinates.len() / 3;
    if n < 2 || coordinates.len() != 3 * n {
        return None;
    }
    if coordinates.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let x = ndarray::ArrayView1::from(coordinates);
    let (components, edges) = crate::structure::contact_census_nn(x, n);
    let rings = if n >= 3 {
        let profile = crate::structure::ring_census_nn(x, n).profile;
        profile.0 + profile.1 + profile.2
    } else {
        0
    };
    let rg2 = crate::structure::radius_of_gyration2(x, n);
    let path_rg2 = crate::structure::path_radius_of_gyration2(n, median_contact_spacing(x, n));
    let rmax_over_cbrt = max_com_radius(x, n) / (n as f64).cbrt();
    Some(OccupancyCompact {
        n,
        components,
        edges,
        rings,
        rg2,
        path_rg2,
        rmax_over_cbrt,
    })
}

fn max_com_radius(x: ndarray::ArrayView1<f64>, n: usize) -> f64 {
    let mut com = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            com[k] += x[3 * i + k];
        }
    }
    let inv = 1.0 / n as f64;
    for k in 0..3 {
        com[k] *= inv;
    }
    let mut rmax = 0.0_f64;
    for i in 0..n {
        let mut r2 = 0.0;
        for k in 0..3 {
            let d = x[3 * i + k] - com[k];
            r2 += d * d;
        }
        rmax = rmax.max(r2.sqrt());
    }
    rmax
}

fn median_contact_spacing(x: ndarray::ArrayView1<f64>, n: usize) -> f64 {
    let mut nn = Vec::with_capacity(n);
    for i in 0..n {
        let mut best = f64::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d2 = (0..3)
                .map(|k| {
                    let d = x[3 * i + k] - x[3 * j + k];
                    d * d
                })
                .sum::<f64>();
            if d2 < best {
                best = d2;
            }
        }
        if best.is_finite() {
            nn.push(best.sqrt());
        }
    }
    if nn.is_empty() {
        return 1.0;
    }
    nn.sort_by(|a, b| a.total_cmp(b));
    nn[nn.len() / 2]
}

/// Whether the coordinates are one compact cluster, not a chain or fragments.
pub fn occupancy_is_cluster(coordinates: &[f64]) -> bool {
    occupancy_compact(coordinates).is_some_and(|census| census.is_cluster())
}

/// Franzblau census of one structure, or `None` below three atoms.
pub fn occupancy_ring_census(coordinates: &[f64]) -> Option<crate::structure::RingCensus> {
    let n = coordinates.len() / 3;
    if n < 3 || coordinates.len() != 3 * n {
        return None;
    }
    let x = ndarray::ArrayView1::from(coordinates);
    Some(crate::structure::ring_census_nn(x, n))
}

/// L1 of two Franzblau profiles. Leave prefers a larger value.
pub fn ring_novelty(origin: (usize, usize, usize), trial: (usize, usize, usize)) -> usize {
    origin.0.abs_diff(trial.0) + origin.1.abs_diff(trial.1) + origin.2.abs_diff(trial.2)
}

/// Atom weight when leaving `origin`.
///
/// Five-rings, when the occupied packing has them, are the Franzblau
/// fivefold signature: extras move those atoms. Otherwise extras move
/// three-rings. Uniform incidence is a global scale and does not steer.
pub fn ring_leave_weight(origin: (usize, usize, usize), atom: [u32; 3]) -> f64 {
    if origin.2 > 0 {
        1.0 + f64::from(atom[2])
    } else {
        1.0 + f64::from(atom[0])
    }
}

/// Scale a Cartesian increment so leave-ring atoms move more.
///
/// Champion leftover SOAP does not call this. Occupancy archive
/// holes and packing kicks do. A uniform ring support is a global
/// scale; the hop's RMS cap removes it.
pub fn lens_ring_displacement(coordinates: &[f64], dr: &mut [f64]) {
    if coordinates.len() != dr.len() {
        return;
    }
    let Some(census) = occupancy_ring_census(coordinates) else {
        return;
    };
    let n = census.atom.len();
    if dr.len() != 3 * n {
        return;
    }
    for (i, &incidence) in census.atom.iter().enumerate() {
        let weight = ring_leave_weight(census.profile, incidence);
        dr[3 * i] *= weight;
        dr[3 * i + 1] *= weight;
        dr[3 * i + 2] *= weight;
    }
}

fn side_spread(histograms: &[Vec<f64>], members: &[usize]) -> f64 {
    let mut widest = 0.0;
    for (a, &i) in members.iter().enumerate() {
        for &j in members.iter().skip(a + 1) {
            let d = super::packing::packing_distance(&histograms[i], &histograms[j]);
            if d > widest {
                widest = d;
            }
        }
    }
    widest
}

/// How DECAF L1 becomes the Torgerson metric.
///
/// [`OccupancyFold::Switch`] is the production floor. Far L1 pairs
/// share one value (Lean `sat_far_cannot_tell`), so a packing fork
/// collapses to a line. [`OccupancyFold::Asinh`] keeps far L1 ordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OccupancyFold {
    /// \(F(d)=1-1/(1+(d/\sigma)^2)\). Intra-funnel short; far pairs saturate.
    Switch,
    /// \(F(d)=\mathrm{asinh}(d/\sigma)/(2\,\mathrm{asinh}\,1)\). Unbounded.
    Asinh,
    /// Raw L1. No transfer.
    Identity,
}

/// Torgerson (1952) classical MDS of DECAF L1 after a Ceriotti switch.
///
/// Raw L1 stretches a leftover-family chain so 2-means splits the
/// ends. The switch \(F(d)=1-1/(1+(d/\sigma)^2)\) with \(\sigma\) the
/// median pairwise L1 keeps intra-funnel distances short.
pub fn occupancy_map_from_histograms(histograms: &[Vec<f64>]) -> Option<Vec<[f64; 2]>> {
    occupancy_map_fold(histograms, OccupancyFold::Switch).map(|(xy, _)| xy)
}

/// Torgerson map and the two leading eigenvalues of the Gram matrix.
pub fn occupancy_map_fold(
    histograms: &[Vec<f64>],
    fold: OccupancyFold,
) -> Option<(Vec<[f64; 2]>, [f64; 2])> {
    let n = histograms.len();
    if n < 2 {
        return None;
    }
    let mut dist = vec![vec![0.0; n]; n];
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = super::packing::packing_distance(&histograms[i], &histograms[j]);
            if !d.is_finite() {
                return None;
            }
            dist[i][j] = d;
            dist[j][i] = d;
            if d > 0.0 {
                pairs.push(d);
            }
        }
    }
    let sigma = median(&mut pairs).unwrap_or(super::packing::PACKING_MERGE);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            dist[i][j] = match fold {
                OccupancyFold::Switch => ceriotti_switch(dist[i][j], sigma),
                OccupancyFold::Asinh => asinh_switch(dist[i][j], sigma),
                OccupancyFold::Identity => dist[i][j],
            };
        }
    }
    torgerson_2d(&dist)
}

fn ceriotti_switch(distance: f64, sigma: f64) -> f64 {
    if !(distance.is_finite() && sigma.is_finite()) || sigma <= 0.0 {
        return 0.0;
    }
    1.0 - 1.0 / (1.0 + (distance / sigma).powi(2))
}

fn asinh_switch(distance: f64, sigma: f64) -> f64 {
    if !(distance.is_finite() && sigma.is_finite()) || sigma <= 0.0 {
        return 0.0;
    }
    let u = distance / sigma;
    u.asinh() / (2.0 * 1.0_f64.asinh())
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    Some(values[values.len() / 2])
}

fn majority_family(family: &[usize], members: &[usize]) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for &i in members {
        let f = *family.get(i)?;
        let count = members
            .iter()
            .filter(|&&j| family.get(j) == Some(&f))
            .count();
        match best {
            Some((_, held)) if count <= held => {}
            _ => best = Some((f, count)),
        }
    }
    best.map(|(f, _)| f)
}

fn centroid_radius(xy: &[[f64; 2]], members: &[usize]) -> ([f64; 2], f64) {
    let n = members.len().max(1) as f64;
    let mut c = [0.0, 0.0];
    for &i in members {
        c[0] += xy[i][0];
        c[1] += xy[i][1];
    }
    c[0] /= n;
    c[1] /= n;
    let radius = members
        .iter()
        .map(|&i| {
            let dx = xy[i][0] - c[0];
            let dy = xy[i][1] - c[1];
            (dx * dx + dy * dy).sqrt()
        })
        .fold(0.0, f64::max);
    (c, radius)
}

fn two_means(xy: &[[f64; 2]]) -> Option<(Vec<usize>, Vec<usize>)> {
    let n = xy.len();
    let (a, b) = farthest_pair(xy)?;
    let mut c0 = xy[a];
    let mut c1 = xy[b];
    let mut left = Vec::new();
    let mut right = Vec::new();
    for _ in 0..16 {
        left.clear();
        right.clear();
        for i in 0..n {
            if dist2(xy[i], c0) <= dist2(xy[i], c1) {
                left.push(i);
            } else {
                right.push(i);
            }
        }
        if left.is_empty() || right.is_empty() {
            return None;
        }
        c0 = mean2(xy, &left);
        c1 = mean2(xy, &right);
    }
    Some((left, right))
}

fn farthest_pair(xy: &[[f64; 2]]) -> Option<(usize, usize)> {
    let n = xy.len();
    if n < 2 {
        return None;
    }
    let mut best = (0, 1);
    let mut best_d = -1.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist2(xy[i], xy[j]);
            if d > best_d {
                best_d = d;
                best = (i, j);
            }
        }
    }
    Some(best)
}

fn mean2(xy: &[[f64; 2]], members: &[usize]) -> [f64; 2] {
    let n = members.len().max(1) as f64;
    let mut c = [0.0, 0.0];
    for &i in members {
        c[0] += xy[i][0];
        c[1] += xy[i][1];
    }
    [c[0] / n, c[1] / n]
}

fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

fn torgerson_2d(dist: &[Vec<f64>]) -> Option<(Vec<[f64; 2]>, [f64; 2])> {
    let n = dist.len();
    if n < 2 {
        return None;
    }
    let mut d2 = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            d2[i][j] = dist[i][j] * dist[i][j];
        }
    }
    let mut row_mean = vec![0.0; n];
    let mut col_mean = vec![0.0; n];
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            row_mean[i] += d2[i][j];
            col_mean[j] += d2[i][j];
            total += d2[i][j];
        }
    }
    let nf = n as f64;
    for i in 0..n {
        row_mean[i] /= nf;
        col_mean[i] /= nf;
    }
    total /= nf * nf;
    let mut b = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            b[i][j] = -0.5 * (d2[i][j] - row_mean[i] - col_mean[j] + total);
        }
    }
    let mut y = vec![[0.0; 2]; n];
    let mut spectrum = [0.0; 2];
    for k in 0..2 {
        let mut v: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let norm0 = v.iter().map(|z| z * z).sum::<f64>().sqrt().max(1e-15);
        for z in &mut v {
            *z /= norm0;
        }
        if k == 1 {
            let prev: f64 = v.iter().zip(y.iter()).map(|(a, row)| a * row[0]).sum();
            for i in 0..n {
                v[i] -= prev * y[i][0];
            }
            let norm = v.iter().map(|z| z * z).sum::<f64>().sqrt().max(1e-15);
            for z in &mut v {
                *z /= norm;
            }
        }
        let mut lambda = 0.0;
        for _ in 0..80 {
            let mut w = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += b[i][j] * v[j];
                }
            }
            if k == 1 {
                let prev: f64 = w.iter().zip(y.iter()).map(|(a, row)| a * row[0]).sum();
                for i in 0..n {
                    w[i] -= prev * y[i][0];
                }
            }
            lambda = w.iter().zip(v.iter()).map(|(a, c)| a * c).sum();
            let norm = w.iter().map(|z| z * z).sum::<f64>().sqrt().max(1e-15);
            for i in 0..n {
                v[i] = w[i] / norm;
            }
        }
        let scale = lambda.max(0.0).sqrt();
        spectrum[k] = lambda.max(0.0);
        for i in 0..n {
            y[i][k] = scale * v[i];
        }
    }
    Some((y, spectrum))
}

/// Occupied-family floor. Spectral split unless `CATALOG_MIN_FAMILIES`
/// is set.
pub fn occupancy_min_families() -> usize {
    std::env::var("CATALOG_MIN_FAMILIES")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&count| count >= 1)
        .unwrap_or(DEFAULT_MIN_OCCUPIED_FAMILIES)
}

/// Generic occupancy completeness. `n_occupied_families` is the DECAF
/// packing count, not leftover-SOAP basin count.
pub fn occupancy_complete(
    mixing_certified: bool,
    catalog_saturated: bool,
    n_occupied_families: usize,
) -> Option<OccupancyCertificate> {
    occupancy_complete_at(
        mixing_certified,
        catalog_saturated,
        n_occupied_families,
        occupancy_min_families(),
    )
}

/// Completeness at an explicit family floor.
pub fn occupancy_complete_at(
    mixing_certified: bool,
    catalog_saturated: bool,
    n_occupied_families: usize,
    min_occupied_families: usize,
) -> Option<OccupancyCertificate> {
    if mixing_certified {
        Some(OccupancyCertificate::MixingCertified)
    } else if catalog_saturated && n_occupied_families >= min_occupied_families {
        Some(OccupancyCertificate::CatalogSaturated)
    } else {
        None
    }
}

/// Role of one walk against DECAF packing families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingRole {
    /// Lowest energy occupant of this family. Isomer walk.
    FamilyChampion,
    /// Same family, not the champion. Leave or keep-fraction extra.
    FamilyExtra,
    /// No catalog or live walk shares this packing.
    NovelFamily,
}

/// Per-family energy class, not the catalog-wide incumbent.
/// Equal energy to the family best is Champion here; the coordinator
/// keeps one replica via family_champion_replicas so extras Leave.
pub fn packing_role(same_family: bool, energy: f64, best_of_family: Option<f64>) -> PackingRole {
    if !same_family {
        return PackingRole::NovelFamily;
    }
    match best_of_family {
        None => PackingRole::FamilyChampion,
        Some(best) if energy <= best + 1e-8 => PackingRole::FamilyChampion,
        Some(_) => PackingRole::FamilyExtra,
    }
}

/// Champion of the occupied packing has no TIS interface.
pub const CHAMPION_RANK: u32 = u32::MAX;
/// Fixed leftover-SOAP horizon for the TIS ladder. Unit leftover
/// descriptors sit in a ball of diameter 2. Live `max(lambda)` is not
/// a horizon: it collapses every interface onto the well.
pub const INTERFACE_HORIZON: f64 = 2.0;

/// One leftover-SOAP interface seat owned by the catalog RPC.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterfaceSeat {
    /// Replica that holds this seat.
    pub replica: u32,
    /// Interface rank. [`CHAMPION_RANK`] is the A-ensemble isomer walk.
    pub rank: u32,
    /// Threshold \(\lambda_i\) this extra must reach.
    pub threshold: f64,
    /// Highest leftover-SOAP \(\lambda\) this replica has posted.
    pub lambda: f64,
}

impl InterfaceSeat {
    /// Occupied-packing champion: no interface, isomer walk only.
    pub fn champion(replica: u32) -> Self {
        Self {
            replica,
            rank: CHAMPION_RANK,
            threshold: 0.0,
            lambda: 0.0,
        }
    }

    /// Extra on interface `rank` with threshold \(\lambda_i\).
    pub fn extra(replica: u32, rank: u32, threshold: f64) -> Self {
        Self {
            replica,
            rank,
            threshold,
            lambda: 0.0,
        }
    }
}

/// Leftover-SOAP order parameter: distance from the occupied-well centroid.
pub fn leftover_lambda(current: &[f64], centroid: &[f64]) -> f64 {
    if current.is_empty() || current.len() != centroid.len() {
        return 0.0;
    }
    current
        .iter()
        .zip(centroid)
        .map(|(value, mean)| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

/// TIS ladder: \(n\) interfaces equally spaced out to `horizon`.
pub fn interface_ladder(n_extras: usize, horizon: f64) -> Vec<f64> {
    if n_extras == 0 || !horizon.is_finite() || horizon <= 0.0 {
        return Vec::new();
    }
    (1..=n_extras)
        .map(|index| horizon * index as f64 / n_extras as f64)
        .collect()
}

/// Whether this sample belongs in interface ensemble \(i\).
pub fn in_interface_ensemble(max_lambda: f64, threshold: f64) -> bool {
    max_lambda.is_finite() && threshold.is_finite() && max_lambda >= threshold
}

/// RETIS swap: each sample already satisfies the other's interface.
pub fn retis_should_swap(lambda_a: f64, thresh_a: f64, lambda_b: f64, thresh_b: f64) -> bool {
    thresh_a != thresh_b
        && in_interface_ensemble(lambda_a, thresh_b)
        && in_interface_ensemble(lambda_b, thresh_a)
}

/// Leave accepts a DECAF family change or an interface crossing.
pub fn leave_shot_accepted(family_changed: bool, trial_lambda: f64, threshold: f64) -> bool {
    family_changed || in_interface_ensemble(trial_lambda, threshold)
}

/// One frame on the occupancy Leave path.
#[derive(Debug, Clone, PartialEq)]
pub struct LeaveFrame {
    /// Cartesian coordinates at this frame.
    pub coordinates: Vec<f64>,
    /// Leftover-SOAP descriptor at this frame.
    pub leftover: Vec<f64>,
    /// Leftover-SOAP \(\lambda\) from the occupied centroid.
    pub lambda: f64,
}

/// Path the extra shoots from. OPS keeps the furthest frame; a failed
/// quench does not throw away the climb toward the interface.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LeavePath {
    frames: Vec<LeaveFrame>,
}

impl LeavePath {
    /// Append one checkpoint. The path is a short climbing window.
    pub fn push(&mut self, coordinates: Vec<f64>, leftover: Vec<f64>, lambda: f64) {
        if !lambda.is_finite() {
            return;
        }
        self.frames.push(LeaveFrame {
            coordinates,
            leftover,
            lambda,
        });
        while self.frames.len() > 32 {
            self.frames.remove(0);
        }
    }

    /// Drop the climbing window. Once the Leave is issued the frames
    /// that led to it describe a funnel the chain has left, and a later
    /// shoot taken from them re-enters it.
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Highest leftover-SOAP \(\lambda\) on the path.
    pub fn max_lambda(&self) -> f64 {
        self.frames
            .iter()
            .map(|frame| frame.lambda)
            .fold(0.0, f64::max)
    }

    /// Interior frame with the highest \(\lambda\), else the last frame.
    pub fn shoot_index(&self) -> Option<usize> {
        match self.frames.len() {
            0 => None,
            1 | 2 => Some(self.frames.len() - 1),
            n => self.frames[1..n - 1]
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.lambda.total_cmp(&right.1.lambda))
                .map(|(index, _)| index + 1),
        }
    }

    /// Coordinates to shoot from.
    pub fn shoot_coordinates(&self) -> Option<&[f64]> {
        self.shoot_index()
            .map(|index| self.frames[index].coordinates.as_slice())
    }

    /// Leftover-SOAP descriptor at the shoot frame.
    pub fn shoot_leftover(&self) -> Option<&[f64]> {
        self.shoot_index()
            .map(|index| self.frames[index].leftover.as_slice())
    }
}

/// Assign extras increasing leftover-SOAP interfaces. Champion is omitted.
/// Seats are ordered by leftover-SOAP \(\lambda\), not replica id.
pub fn assign_interfaces(extras: &[(u32, f64)], horizon: f64) -> Vec<InterfaceSeat> {
    seat_extras(&[], extras, horizon)
}

/// Seat extras on the TIS ladder, keeping the rank each replica already
/// holds in `previous`. An interface is an ensemble a replica owns until
/// an exchange move hands it on: ranking every request by live
/// \(\lambda\) leaves [`retis_exchange_adjacent`] and
/// [`promote_one_sided`] with an ordering they can never repair, and
/// discards their result on the next request. Unseated extras take the
/// free ranks in \(\lambda\) order, so an empty `previous` is the
/// fresh ladder.
pub fn seat_extras(
    previous: &[InterfaceSeat],
    extras: &[(u32, f64)],
    horizon: f64,
) -> Vec<InterfaceSeat> {
    let ladder = interface_ladder(extras.len(), horizon);
    let mut order = extras.to_vec();
    order.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
    let mut holder: Vec<Option<(u32, f64)>> = vec![None; extras.len()];
    let mut seated: Vec<u32> = Vec::new();
    for seat in previous {
        let rank = seat.rank as usize;
        if seat.rank == CHAMPION_RANK || rank >= holder.len() || holder[rank].is_some() {
            continue;
        }
        if seated.contains(&seat.replica) {
            continue;
        }
        let Some(extra) = order.iter().find(|(id, _)| *id == seat.replica) else {
            continue;
        };
        holder[rank] = Some(*extra);
        seated.push(seat.replica);
    }
    let mut free = holder
        .iter()
        .enumerate()
        .filter_map(|(rank, slot)| slot.is_none().then_some(rank))
        .collect::<Vec<_>>()
        .into_iter();
    for extra in &order {
        if seated.contains(&extra.0) {
            continue;
        }
        let Some(rank) = free.next() else {
            break;
        };
        holder[rank] = Some(*extra);
    }
    holder
        .into_iter()
        .enumerate()
        .filter_map(|(rank, slot)| {
            let (replica, lambda) = slot?;
            Some(InterfaceSeat {
                replica,
                rank: rank as u32,
                threshold: ladder.get(rank).copied().unwrap_or(horizon),
                lambda,
            })
        })
        .collect()
}

/// Swap adjacent interface seats when both extras have crossed and the
/// higher \(\lambda\) sits on the lower interface. The ordering guard is
/// what makes the move settle: two samples that each satisfy the other's
/// interface would otherwise trade seats on every request.
pub fn retis_exchange_adjacent(seats: &mut [InterfaceSeat]) -> bool {
    let mut swapped = false;
    let mut index = 0;
    while index + 1 < seats.len() {
        let left = seats[index];
        let right = seats[index + 1];
        if left.rank == CHAMPION_RANK || right.rank == CHAMPION_RANK {
            index += 1;
            continue;
        }
        if retis_should_swap(left.lambda, left.threshold, right.lambda, right.threshold)
            && left.lambda > right.lambda
        {
            seats[index].replica = right.replica;
            seats[index].lambda = right.lambda;
            seats[index + 1].replica = left.replica;
            seats[index + 1].lambda = left.lambda;
            swapped = true;
            index += 2;
        } else {
            index += 1;
        }
    }
    swapped
}

/// Promote an extra that already cleared the next threshold, without
/// waiting for the neighbor to cross this one.
pub fn promote_one_sided(seats: &mut [InterfaceSeat]) -> bool {
    let mut promoted = false;
    let mut index = 0;
    while index + 1 < seats.len() {
        let left = seats[index];
        let right = seats[index + 1];
        if left.rank == CHAMPION_RANK || right.rank == CHAMPION_RANK {
            index += 1;
            continue;
        }
        if in_interface_ensemble(left.lambda, right.threshold)
            && !in_interface_ensemble(right.lambda, left.threshold)
        {
            seats[index].replica = right.replica;
            seats[index].lambda = right.lambda;
            seats[index + 1].replica = left.replica;
            seats[index + 1].lambda = left.lambda;
            promoted = true;
            index += 2;
        } else {
            index += 1;
        }
    }
    promoted
}

/// Replica retire. Mixing certified, packing Good--Turing, FunnelModel
/// EI exhausted on seen packings, and the rematched family floor.
/// CatalogSaturated names census completeness and does not retire.
/// `catalog_saturated` is packing-family saturation, not leftover-SOAP.
/// Leftover-SOAP hatches are the champion isomer walk of a seen
/// packing. They are reported; they do not block retire after packing
/// Chao1 and MixingCertified. Requiring leftover-SOAP dwell walks the
/// force budget after the putative is already certified.
/// `leftover_dwell` stays on the signature as the census bit.
/// `ei_exhausted` is Jones remaining improvement on observed
/// FunnelModel morphologies, not a far-field GP probe.
/// `n_occupied_families` is the packing-book occupied-family count
/// (visits > 0), not leftover-SOAP basin count.
pub fn occupancy_retire(
    certificate: OccupancyCertificate,
    catalog_saturated: bool,
    leftover_dwell: bool,
    ei_exhausted: bool,
    n_occupied_families: usize,
) -> bool {
    occupancy_retire_at(
        certificate,
        catalog_saturated,
        leftover_dwell,
        ei_exhausted,
        n_occupied_families,
        occupancy_min_families(),
    )
}

/// Retire at an explicit family floor. Good--Turing alone is floor 1.
pub fn occupancy_retire_at(
    certificate: OccupancyCertificate,
    catalog_saturated: bool,
    leftover_dwell: bool,
    ei_exhausted: bool,
    n_occupied_families: usize,
    min_occupied_families: usize,
) -> bool {
    let _ = leftover_dwell;
    n_occupied_families >= min_occupied_families
        && catalog_saturated
        && ei_exhausted
        && matches!(certificate, OccupancyCertificate::MixingCertified)
}

#[cfg(test)]
mod tests {
    use super::{
        CHAMPION_RANK, InterfaceSeat, LeavePath, OCCUPANCY_SEAM_CONDUCTANCE, OccupancyCertificate,
        OccupancyFesError, OccupancyLeaveAdopt, OccupancyLeaveTarget, PackingRole,
        assign_interfaces, in_interface_ensemble, interface_ladder, is_occupancy_leave_action,
        leave_shot_accepted, leftover_esty_stable, leftover_esty_var, leftover_hatch_stable,
        leftover_lambda, leftover_sat_dwell, occupancy_book_holes, occupancy_compact,
        occupancy_complete, occupancy_complete_at, occupancy_ei_exhausted, occupancy_family_floor,
        occupancy_fes, occupancy_fes_delta, occupancy_fes_from_histograms, occupancy_is_cluster,
        occupancy_landfold_floor, occupancy_leave_adopt, occupancy_leave_new_class,
        occupancy_leave_target, occupancy_map_floor, occupancy_retire, occupancy_retire_at,
        occupancy_ring_class_changed, occupancy_ring_floor, occupancy_sparsify_book, packing_role,
        promote_one_sided, published_energy_score, retis_exchange_adjacent, retis_should_swap,
        ring_leave_weight, ring_novelty, seat_extras,
    };

    #[test]
    fn feynman_kac_extras_use_the_same_leave_as_occupancy() {
        assert!(is_occupancy_leave_action("population_reseed"));
        assert!(is_occupancy_leave_action("hyperband_reseed"));
        assert!(is_occupancy_leave_action("catalog_leave"));
        assert!(!is_occupancy_leave_action("catalog_incumbent"));
        assert!(!is_occupancy_leave_action("bridge"));
    }

    #[test]
    fn occupancy_leave_is_another_family_or_an_archive_hole() {
        assert_eq!(
            occupancy_leave_target(true, false, 2),
            OccupancyLeaveTarget::OtherFamily
        );
        assert_eq!(
            occupancy_leave_target(false, false, 2),
            OccupancyLeaveTarget::ArchiveHole
        );
        assert_ne!(
            occupancy_leave_target(false, false, 2),
            OccupancyLeaveTarget::OtherFamily
        );
    }

    #[test]
    fn one_packing_on_the_book_is_nothing_to_divide() {
        // A Leave trades a replica's walk for coverage. With one community
        // there is no second packing to cover, and the hole that would be
        // drawn has no measured yield on LJ75, so the trade is a loss.
        assert_eq!(
            occupancy_leave_target(true, false, 1),
            OccupancyLeaveTarget::Walk
        );
        assert_eq!(
            occupancy_leave_target(false, false, 1),
            OccupancyLeaveTarget::Walk
        );
        assert_eq!(
            occupancy_leave_target(true, false, 0),
            OccupancyLeaveTarget::Walk
        );
        assert_eq!(
            occupancy_leave_target(true, false, 2),
            OccupancyLeaveTarget::OtherFamily
        );
    }

    #[test]
    fn packing_sat_leave_still_draws_other_family() {
        assert_eq!(
            occupancy_leave_target(true, true, 2),
            OccupancyLeaveTarget::OtherFamily
        );
        assert_ne!(
            occupancy_leave_target(true, true, 2),
            OccupancyLeaveTarget::ArchiveHole
        );
        assert_eq!(
            occupancy_leave_target(false, true, 1),
            OccupancyLeaveTarget::Walk
        );
        assert_eq!(
            occupancy_leave_target(true, false, 2),
            OccupancyLeaveTarget::OtherFamily
        );
        assert_eq!(
            occupancy_leave_target(false, false, 2),
            OccupancyLeaveTarget::ArchiveHole
        );
    }

    #[test]
    fn catalog_leave_refuses_a_same_family_hole() {
        assert_eq!(
            occupancy_leave_adopt("catalog_leave", false),
            Some(OccupancyLeaveAdopt::Refuse)
        );
        assert_eq!(
            occupancy_leave_adopt("catalog_leave", true),
            Some(OccupancyLeaveAdopt::Quench)
        );
        assert_eq!(
            occupancy_leave_adopt("hyperband_reseed", false),
            Some(OccupancyLeaveAdopt::Quench)
        );
        assert_eq!(
            occupancy_leave_adopt("population_reseed", false),
            Some(OccupancyLeaveAdopt::Quench)
        );
        assert_eq!(occupancy_leave_adopt("catalog_incumbent", false), None);
        assert_eq!(occupancy_leave_adopt("histo", true), None);
    }

    #[test]
    fn a_published_energy_is_a_score_not_a_certificate() {
        assert!(published_energy_score(-173.928427, Some(-173.928427)));
        assert!(published_energy_score(-397.492331, Some(-397.492331)));
        assert!(!published_energy_score(-396.282249, Some(-397.492331)));
        assert!(!published_energy_score(-173.928427, None));
        assert_eq!(occupancy_complete(false, false, 0), None);
    }

    #[test]
    fn inverted_gr_mixing_certifies_the_search() {
        assert_eq!(
            occupancy_complete(true, false, 1),
            Some(OccupancyCertificate::MixingCertified)
        );
    }

    #[test]
    fn packing_saturation_below_the_family_floor_is_not_done() {
        assert_eq!(occupancy_complete_at(false, true, 1, 2), None);
        assert_eq!(occupancy_complete_at(false, true, 0, 2), None);
    }

    #[test]
    fn packing_good_turing_with_no_new_families_is_saturated() {
        assert_eq!(
            occupancy_complete_at(false, true, 1, 1),
            Some(OccupancyCertificate::CatalogSaturated)
        );
        assert_eq!(occupancy_complete_at(false, false, 5, 1), None);
    }

    #[test]
    fn packing_gt_with_two_families_meets_the_paper_floor() {
        assert_eq!(
            occupancy_complete(false, true, 2),
            Some(OccupancyCertificate::CatalogSaturated)
        );
        assert!(!occupancy_retire(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            1,
            2
        ));
    }

    #[test]
    fn catalog_saturation_does_not_retire_without_mixing() {
        assert!(!occupancy_retire(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            2
        ));
    }

    #[test]
    fn packing_role_is_per_family_not_catalog_wide() {
        assert_eq!(
            packing_role(true, -173.252378, Some(-173.252378)),
            PackingRole::FamilyChampion
        );
        assert_eq!(
            packing_role(true, -173.134317, Some(-173.252378)),
            PackingRole::FamilyExtra
        );
        assert_eq!(
            packing_role(false, -173.928427, Some(-173.252378)),
            PackingRole::NovelFamily
        );
        assert_eq!(
            packing_role(true, -173.928427, Some(-173.252378)),
            PackingRole::FamilyChampion
        );
    }

    #[test]
    fn good_turing_with_a_competing_family_is_a_certificate() {
        assert_eq!(
            occupancy_complete(false, true, 2),
            Some(OccupancyCertificate::CatalogSaturated)
        );
    }

    #[test]
    fn mixing_outranks_catalog_saturation() {
        assert_eq!(
            occupancy_complete(true, true, 2),
            Some(OccupancyCertificate::MixingCertified)
        );
    }

    #[test]
    fn mixing_alone_does_not_retire_extras_before_good_turing() {
        assert!(!occupancy_retire(
            OccupancyCertificate::MixingCertified,
            false,
            true,
            true,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            true,
            true,
            1,
            2
        ));
        assert!(occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            true,
            true,
            2,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            2,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            1,
            2
        ));
        assert_eq!(occupancy_complete(false, false, 8), None);
    }

    #[test]
    fn leftover_soap_does_not_block_mixing_packing_and_ei() {
        assert!(!leftover_sat_dwell(&[false]));
        assert!(!leftover_sat_dwell(&[true]));
        assert!(occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            false,
            true,
            2,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::CatalogSaturated,
            true,
            false,
            true,
            2,
            2
        ));
    }

    #[test]
    fn visit_floor_is_hatch_stable_min_for_three_singletons() {
        assert_eq!(
            crate::catalog::gt_min_visits(
                crate::catalog::PRODUCTION_UNSEEN_MASS_NUM,
                crate::catalog::PRODUCTION_UNSEEN_MASS_DEN,
                crate::catalog::PRODUCTION_SINGLETON_BUDGET
            ),
            20
        );
        assert_eq!(crate::catalog::PRODUCTION_MINIMUM_VISITS, 20);
        assert_eq!(
            crate::catalog::CERTIFY_MIN_SAMPLES,
            crate::catalog::CERTIFY_CHAINS
                * crate::catalog::CERTIFY_SPLIT_HALVES
                * crate::catalog::CERTIFY_DRAWS_PER_HALF
        );
        assert_eq!(crate::catalog::CERTIFY_MIN_SAMPLES, 16);
    }

    #[test]
    fn ico13_is_a_cluster_and_a_straight_chain_is_not() {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let mut verts = vec![[0.0, 0.0, 0.0]];
        for s in [1.0_f64, -1.0] {
            for t in [phi, -phi] {
                verts.push([0.0, s, t]);
                verts.push([s, t, 0.0]);
                verts.push([t, 0.0, s]);
            }
        }
        let mut ico = vec![0.0; 39];
        for (i, p) in verts.iter().enumerate() {
            for k in 0..3 {
                ico[3 * i + k] = p[k] * 0.55;
            }
        }
        let ico_c = occupancy_compact(&ico).unwrap();
        assert!(ico_c.is_cluster());
        assert!(!ico_c.is_pathlike());
        assert!(ico_c.rg2 < ico_c.path_rg2);
        assert!(ico_c.rmax_over_cbrt < crate::catalog::COMPACT_RMAX_OVER_CBRT);
        assert!(occupancy_is_cluster(&ico));
        assert!(!occupancy_ring_class_changed(&ico, &ico));

        let mut chain = vec![0.0; 39];
        for i in 0..13 {
            chain[3 * i] = i as f64;
        }
        assert!(occupancy_ring_class_changed(&ico, &chain));
        assert!(occupancy_leave_new_class(&ico, &chain));
        let chain_c = occupancy_compact(&chain).unwrap();
        assert_eq!(chain_c.components, 1);
        assert!(chain_c.is_pathlike());
        assert!(!chain_c.is_cluster());
        assert!(chain_c.rmax_over_cbrt > crate::catalog::COMPACT_RMAX_OVER_CBRT);
        assert!(!occupancy_is_cluster(&chain));
        let expected = crate::structure::path_radius_of_gyration2(13, 1.0);
        assert!((chain_c.rg2 - expected).abs() < 1e-12);
        assert!((chain_c.path_rg2 - expected).abs() < 1e-12);

        let mut split = ico.clone();
        for i in 0..6 {
            split[3 * i] += 20.0;
        }
        let split_c = occupancy_compact(&split).unwrap();
        assert!(split_c.components >= 2);
        assert!(!split_c.is_cluster());
        assert!(!occupancy_is_cluster(&split));
    }

    #[test]
    fn leftover_hatch_stable_is_the_good_turing_increment() {
        assert!(!leftover_hatch_stable(0, 0, 0.2));
        assert!(!leftover_hatch_stable(20, 4, 0.2));
        assert!(leftover_hatch_stable(20, 0, 0.2));
        assert!(leftover_hatch_stable(20, 3, 0.2));
        assert!(!leftover_hatch_stable(19, 3, 0.2));
        assert!(!leftover_hatch_stable(4, 0, 0.2));
        assert!(!leftover_hatch_stable(20, 21, 0.2));
    }

    #[test]
    fn leftover_esty_blocks_a_singleton_heavy_hatch() {
        assert!(leftover_hatch_stable(20, 3, 0.2));
        assert!(!leftover_esty_stable(20, 3, 0, 0.2));
        assert!(leftover_esty_stable(20, 0, 0, 0.2));
        assert!(leftover_esty_var(20, 0, 0).unwrap() < 1e-18);
    }

    #[test]
    fn leftover_nick_is_not_a_retire_dwell() {
        assert!(!leftover_sat_dwell(&[true, false]));
        assert!(!leftover_sat_dwell(&[false, true]));
        assert!(!leftover_sat_dwell(&[true; 4]));
        assert!(leftover_sat_dwell(&[true; 5]));
        let mut hatch_then_dwell = vec![false];
        hatch_then_dwell.extend(std::iter::repeat_n(true, 5));
        assert!(leftover_sat_dwell(&hatch_then_dwell));
        let mut dwell_then_hatch = vec![true; 5];
        dwell_then_hatch.push(false);
        assert!(!leftover_sat_dwell(&dwell_then_hatch));
        assert!(occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            leftover_sat_dwell(&[true]),
            true,
            2,
            2
        ));
        assert!(occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            false,
            true,
            1,
            1
        ));
        assert!(occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            false,
            true,
            34,
            1
        ));
    }

    #[test]
    fn leftover_dwell_plus_mixing_plus_packing_sat_plus_floor_retires() {
        assert!(leftover_sat_dwell(&[true; 5]));
        assert!(occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            leftover_sat_dwell(&[true; 5]),
            true,
            2,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            false,
            true,
            true,
            2,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            true,
            true,
            1,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            2,
            2
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            true,
            false,
            2,
            2
        ));
    }

    #[test]
    fn funnel_ei_below_noise_with_three_observations_is_exhausted() {
        assert!(!occupancy_ei_exhausted(1.0, 3, 1e-2));
        assert!(!occupancy_ei_exhausted(0.0, 2, 1e-2));
        assert!(occupancy_ei_exhausted(0.0, 3, 1e-2));
        assert!(occupancy_ei_exhausted(1e-2, 3, 1e-2));
        assert!(!occupancy_ei_exhausted(f64::INFINITY, 3, 1e-2));
    }

    #[test]
    fn map_floor_keeps_one_packing_together() {
        let xy = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [8.0, 8.0]];
        assert_eq!(occupancy_map_floor(&xy, &[0, 0, 0, 0]), 1);
    }

    #[test]
    fn map_floor_splits_two_separated_packings() {
        let xy = [[0.0, 0.0], [0.1, 0.0], [8.0, 8.0], [8.1, 8.0]];
        assert_eq!(occupancy_map_floor(&xy, &[0, 0, 1, 1]), 2);
    }

    #[test]
    fn landfold_floor_on_identical_histograms_is_one() {
        let ico = vec![1.0, 0.0];
        assert_eq!(
            occupancy_landfold_floor(&[ico.clone(), ico.clone()], &[0, 0]),
            1
        );
    }

    #[test]
    fn landfold_floor_on_separated_histograms_is_two() {
        let ico = vec![1.0, 0.0];
        let oh = vec![0.0, 1.0];
        assert_eq!(occupancy_landfold_floor(&[ico, oh], &[0, 1]), 2);
    }

    #[test]
    fn leftover_wells_of_one_packing_sparsify_to_one_community() {
        let ico = vec![1.0, 0.0];
        let map = occupancy_sparsify_book(&[ico.clone(), ico], &[0, 0], &[12, 8]);
        assert_eq!(map.communities, 1);
        assert_eq!(map.community_wells, vec![20]);
        assert!(!map.holes);
        assert!(map.saturated());
    }

    #[test]
    fn a_second_book_packing_with_no_wells_is_a_hole() {
        let ico = vec![1.0, 0.0];
        let oh = vec![0.0, 1.0];
        let map = occupancy_sparsify_book(&[ico, oh], &[0, 1], &[20, 0]);
        assert_eq!(map.communities, 2);
        assert_eq!(map.floor, 2);
        assert!(map.holes);
        assert!(!map.saturated());
        assert_eq!(
            occupancy_leave_target(true, map.saturated(), map.communities),
            OccupancyLeaveTarget::OtherFamily
        );
    }

    #[test]
    fn sparsified_chao1_closes_holes_on_both_book_packings() {
        let ico = vec![1.0, 0.0];
        let oh = vec![0.0, 1.0];
        let map = occupancy_sparsify_book(&[ico, oh], &[0, 1], &[20, 20]);
        assert_eq!(map.communities, 2);
        assert!(!map.holes);
        assert_eq!(
            occupancy_leave_target(true, map.saturated(), map.communities),
            OccupancyLeaveTarget::OtherFamily,
            "saturation does not disable the draw once a second packing is on the book"
        );
    }

    #[test]
    fn sparsified_singleton_packing_keeps_holes() {
        let ico = vec![1.0, 0.0];
        let oh = vec![0.0, 1.0];
        let map = occupancy_sparsify_book(&[ico, oh], &[0, 1], &[20, 1]);
        assert!(map.holes);
        assert_eq!(map.sample().n1, 1);
    }

    #[test]
    fn empty_book_has_holes_so_leave_continues() {
        let map = occupancy_sparsify_book(&[], &[], &[]);
        assert_eq!(map.communities, 0);
        assert!(map.holes);
        let closed = super::super::packing::GoodTuringSample::from_counts([20u64, 20]);
        let open = super::super::packing::GoodTuringSample::from_counts([20u64, 1]);
        assert!(occupancy_book_holes(0, &[], 0, open));
        assert!(occupancy_book_holes(1, &[5], 1, open));
        assert!(!occupancy_book_holes(1, &[20], 1, closed));
        assert!(
            occupancy_book_holes(1, &[20], 1, open),
            "one community is certified by the cells it folds, not by its own bin"
        );
        assert!(occupancy_book_holes(2, &[20, 0], 2, closed));
        assert!(!occupancy_book_holes(2, &[20, 20], 2, closed));
        assert!(
            !occupancy_book_holes(1, &[20], 2, closed),
            "leftover FES modes of one packing are not a packing hole"
        );
    }

    #[test]
    fn leftover_fes_modes_do_not_reopen_a_floor_one_packing() {
        // Cells of one packing: consecutive gaps at or under the link
        // radius, so the whole run chains, with the sample piled at the two
        // ends so the plane still carries two density modes.
        let cells = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.1, 0.0],
            vec![1.0, 0.35, 0.0],
            vec![1.0, 0.6, 0.0],
            vec![1.0, 0.7, 0.0],
        ];
        let map = occupancy_sparsify_book(&cells, &[0, 1, 2, 3, 4], &[20, 20, 20, 20, 20]);
        assert_eq!(
            map.communities, 1,
            "cells that chain at the link radius are one packing"
        );
        assert!(
            map.fes_minima >= 2,
            "separated leftover cells can still be two FES basins: {}",
            map.fes_minima
        );
        assert!(
            !map.holes,
            "Chao1 on the merged leftover community closes the packing census"
        );
    }

    #[test]
    fn an_ico_majority_does_not_absorb_oh_on_the_book() {
        let ico = vec![1.0, 0.0];
        let oh = vec![0.0, 1.0];
        let mut histograms = vec![ico.clone(); 16];
        histograms.push(oh);
        let family: Vec<usize> = (0..17).collect();
        let mut wells = vec![20u64; 16];
        wells.push(0);
        let map = occupancy_sparsify_book(&histograms, &family, &wells);
        assert!(
            map.communities >= 2,
            "Oh must peel off a leftover-ico majority: communities={}",
            map.communities
        );
        assert!(
            map.holes,
            "Oh on the book with no wells is a hole extras Leave into"
        );
        assert_eq!(
            occupancy_leave_target(true, map.saturated(), map.communities),
            OccupancyLeaveTarget::OtherFamily
        );
    }

    #[test]
    fn ring_floor_on_one_profile_is_one() {
        assert_eq!(occupancy_ring_floor(&[]), 1);
        assert_eq!(occupancy_ring_floor(&[(165, 17, 6)]), 1);
        assert_eq!(occupancy_ring_floor(&[(165, 17, 6), (165, 17, 6)]), 1);
    }

    #[test]
    fn ring_floor_on_distinct_profiles_is_two() {
        assert_eq!(occupancy_ring_floor(&[(165, 17, 6), (152, 22, 0)]), 2);
        assert_eq!(occupancy_ring_floor(&[(400, 42, 21), (360, 55, 2)]), 2);
    }

    #[test]
    fn packing_fes_delta_is_log_occupancy_ratio() {
        assert_eq!(occupancy_fes_delta(&[]), None);
        assert_eq!(occupancy_fes_delta(&[20]), None);
        assert_eq!(occupancy_fes_delta(&[0, 0]), None);
        let delta = occupancy_fes_delta(&[20, 5]).unwrap();
        assert!((delta - 4.0_f64.ln()).abs() < 1e-12);
        let even = occupancy_fes_delta(&[10, 10]).unwrap();
        assert!(even.abs() < 1e-12);
        assert_eq!(
            occupancy_fes_delta(&[0, 5, 20]),
            occupancy_fes_delta(&[100, 0, 25])
        );
    }

    #[test]
    fn fes_map_finds_the_continuous_mode_between_samples() {
        let shoulder = [[0.0, 0.0], [0.2, 0.0], [0.5, 0.0]];
        let fes = occupancy_fes(&shoulder, None).unwrap();
        assert_eq!(
            fes.minima, 1,
            "the Gaussian KDE has one mode between the observed points"
        );
        assert!(fes.delta.is_none());
    }

    #[test]
    fn fes_map_rejects_malformed_samples() {
        assert_eq!(
            occupancy_fes(&[[f64::NAN, 0.0]], None),
            Err(OccupancyFesError::NonFinitePoint { index: 0 })
        );
        assert_eq!(
            occupancy_fes(&[[0.0, 0.0], [1.0, 0.0]], Some(&[1.0])),
            Err(OccupancyFesError::WeightCountMismatch {
                points: 2,
                weights: 1,
            })
        );
        assert_eq!(
            occupancy_fes(&[[0.0, 0.0], [1.0, 0.0]], Some(&[1.0, -1.0])),
            Err(OccupancyFesError::InvalidWeight { index: 1 })
        );
        assert_eq!(
            occupancy_fes(&[[0.0, 0.0], [1.0, 0.0]], Some(&[0.0, 0.0])),
            Err(OccupancyFesError::NoPositiveWeight)
        );
        assert_eq!(
            occupancy_fes_from_histograms(&[vec![f64::INFINITY]]),
            Err(OccupancyFesError::MapUnavailable)
        );
    }

    #[test]
    fn fes_map_is_invariant_under_rigid_similarity_and_sample_order() {
        let points = [[0.0, 0.0], [0.1, 0.0], [8.0, 8.0], [8.1, 8.0]];
        let reference = occupancy_fes(&points, None).unwrap();
        let transformed = points.map(|[x, y]| [3.0 - 2.5 * y, -4.0 + 2.5 * x]);
        let permuted = [
            transformed[2],
            transformed[0],
            transformed[3],
            transformed[1],
        ];
        let observed = occupancy_fes(&permuted, None).unwrap();
        assert_eq!(observed.minima, reference.minima);
        assert!((observed.delta.unwrap() - reference.delta.unwrap()).abs() < 1e-12);
        assert!(reference.delta.unwrap().abs() < 1e-12);
    }

    #[test]
    fn fes_map_weights_set_the_relative_mode_depth() {
        let clumps = [[0.0, 0.0], [0.1, 0.0], [8.0, 8.0], [8.1, 8.0]];
        let fes = occupancy_fes(&clumps, Some(&[2.0, 2.0, 1.0, 1.0])).unwrap();
        assert_eq!(fes.minima, 2);
        let delta = fes.delta.unwrap();
        assert!(
            (delta - 0.672_940_481_892_900_6).abs() < 1e-12,
            "finite Gaussian tails contribute density across both modes: {delta}"
        );
    }

    #[test]
    fn fes_map_is_free_energy_not_a_decaf_split() {
        let chain: Vec<[f64; 2]> = (0..8).map(|i| [i as f64, 0.0]).collect();
        let chain_fes = occupancy_fes(&chain, None).unwrap();
        assert_eq!(chain_fes.minima, 1);
        assert!(chain_fes.delta.is_none());
        let clumps = [[0.0, 0.0], [0.1, 0.0], [8.0, 8.0], [8.1, 8.0]];
        let clump_fes = occupancy_fes(&clumps, None).unwrap();
        assert_eq!(clump_fes.minima, 2);
        assert!(clump_fes.delta.is_some());
        let one = occupancy_fes_from_histograms(&[vec![1.0, 0.0]]).unwrap();
        assert_eq!(one.minima, 1);
        assert!(one.delta.is_none());
    }

    #[test]
    fn ring_leave_weight_breaks_pentagons_on_ico_and_triangles_on_oh() {
        let ico = (165, 17, 6);
        let oh = (152, 22, 0);
        assert!(ring_leave_weight(ico, [0, 0, 3]) > ring_leave_weight(ico, [10, 0, 0]));
        assert!(ring_leave_weight(oh, [4, 0, 0]) > ring_leave_weight(oh, [0, 2, 0]));
        assert_eq!(ring_novelty(ico, ico), 0);
        assert!(ring_novelty(ico, oh) > 0);
    }

    #[test]
    fn spectral_seam_is_two_communities_otherwise_one() {
        assert_eq!(occupancy_family_floor(None, None, 0, 0, false), 1);
        assert_eq!(occupancy_family_floor(Some(0.2), Some(0.05), 4, 5, true), 1);
        assert_eq!(occupancy_family_floor(Some(0.05), Some(0.1), 4, 0, true), 1);
        assert_eq!(
            occupancy_family_floor(Some(0.05), Some(0.1), 4, 5, false),
            1
        );
        assert_eq!(occupancy_family_floor(Some(0.05), Some(0.1), 4, 5, true), 2);
        assert_eq!(occupancy_family_floor(Some(0.0), Some(0.0), 4, 5, true), 2);
        assert_eq!(occupancy_family_floor(Some(0.0), None, 4, 5, true), 2);
        assert_eq!(
            occupancy_family_floor(Some(OCCUPANCY_SEAM_CONDUCTANCE), None, 4, 5, true),
            1
        );
        assert_eq!(occupancy_family_floor(Some(0.05), None, 4, 5, true), 1);
    }

    #[test]
    fn a_user_family_floor_of_one_is_good_turing_alone() {
        assert_eq!(
            occupancy_complete_at(false, true, 1, 1),
            Some(OccupancyCertificate::CatalogSaturated)
        );
        assert!(!occupancy_retire_at(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            1,
            1
        ));
        assert!(!occupancy_retire_at(
            OccupancyCertificate::CatalogSaturated,
            true,
            true,
            true,
            2,
            3
        ));
    }

    #[test]
    fn leftover_lambda_is_the_distance_from_the_occupied_centroid() {
        let centroid = [0.0, 0.0, 0.0];
        let on_well = [0.0, 0.0, 0.0];
        let off_well = [3.0, 4.0, 0.0];
        assert_eq!(leftover_lambda(&on_well, &centroid), 0.0);
        assert_eq!(leftover_lambda(&off_well, &centroid), 5.0);
        assert_eq!(leftover_lambda(&[1.0], &centroid), 0.0);
    }

    #[test]
    fn interface_ladder_stages_extras_away_from_the_occupied_well() {
        assert!(interface_ladder(0, 1.0).is_empty());
        assert_eq!(interface_ladder(2, 2.0), vec![1.0, 2.0]);
        assert_eq!(interface_ladder(4, 1.0), vec![0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn a_shot_is_in_the_interface_ensemble_once_it_reaches_the_threshold() {
        assert!(!in_interface_ensemble(0.4, 0.5));
        assert!(in_interface_ensemble(0.5, 0.5));
        assert!(in_interface_ensemble(0.9, 0.5));
    }

    #[test]
    fn retis_swaps_adjacent_ranks_only_when_each_sample_satisfies_the_other() {
        assert!(retis_should_swap(1.0, 0.25, 0.6, 0.5));
        assert!(!retis_should_swap(0.4, 0.25, 0.6, 0.5));
        assert!(!retis_should_swap(1.0, 0.5, 1.0, 0.5));
        assert!(!retis_should_swap(1.0, 0.25, 0.2, 0.5));
    }

    #[test]
    fn leave_accepts_a_family_change_or_an_interface_crossing() {
        assert!(leave_shot_accepted(true, 0.1, 0.5));
        assert!(leave_shot_accepted(false, 0.5, 0.5));
        assert!(!leave_shot_accepted(false, 0.4, 0.5));
    }

    #[test]
    fn a_leave_path_shoots_from_the_highest_interior_lambda() {
        let mut path = LeavePath::default();
        path.push(vec![0.0], vec![0.0], 0.1);
        path.push(vec![1.0], vec![1.0], 0.8);
        path.push(vec![2.0], vec![2.0], 0.3);
        assert_eq!(path.max_lambda(), 0.8);
        assert_eq!(path.shoot_index(), Some(1));
        assert_eq!(path.shoot_coordinates(), Some([1.0].as_slice()));
    }

    #[test]
    fn a_cleared_leave_path_has_no_shoot_frame() {
        let mut path = LeavePath::default();
        path.push(vec![0.0], vec![0.0], 0.1);
        path.push(vec![1.0], vec![1.0], 0.8);
        path.clear();
        assert_eq!(path.shoot_index(), None);
        assert_eq!(path.shoot_coordinates(), None);
        assert_eq!(path.max_lambda(), 0.0);
    }

    #[test]
    fn interface_ranks_follow_lambda_not_replica_id() {
        let extras = [(3, 0.1), (1, 0.9), (2, 0.4)];
        let assigned = assign_interfaces(&extras, 1.0);
        assert_eq!(assigned[0].replica, 3);
        assert_eq!(assigned[0].rank, 0);
        assert_eq!(assigned[1].replica, 2);
        assert_eq!(assigned[2].replica, 1);
        assert!((assigned[2].threshold - 1.0).abs() < 1e-12);
    }

    #[test]
    fn one_sided_promotion_does_not_wait_for_the_neighbor() {
        let mut seats = vec![
            InterfaceSeat {
                replica: 1,
                rank: 0,
                threshold: 0.5,
                lambda: 1.0,
            },
            InterfaceSeat {
                replica: 2,
                rank: 1,
                threshold: 1.0,
                lambda: 0.2,
            },
        ];
        let mut unexchanged = seats.clone();
        assert!(!retis_exchange_adjacent(&mut unexchanged));
        assert!(promote_one_sided(&mut seats));
        assert_eq!(seats[1].replica, 1);
        assert_eq!(seats[0].replica, 2);
        assert!(in_interface_ensemble(seats[1].lambda, seats[1].threshold));
        assert!(!promote_one_sided(&mut seats));
    }

    #[test]
    fn extras_receive_increasing_interface_ranks_and_the_champion_does_not() {
        let extras = [(1, 0.1), (2, 0.4), (3, 0.9)];
        let assigned = assign_interfaces(&extras, 1.0);
        assert_eq!(assigned.len(), 3);
        assert_eq!(assigned[0].replica, 1);
        assert_eq!(assigned[0].rank, 0);
        assert!((assigned[0].threshold - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(assigned[2].replica, 3);
        assert!((assigned[2].threshold - 1.0).abs() < 1e-12);
        assert_eq!(InterfaceSeat::champion(0).rank, CHAMPION_RANK);
    }

    #[test]
    fn retis_exchanges_adjacent_seats_when_both_have_crossed() {
        let mut seats = vec![
            InterfaceSeat {
                replica: 1,
                rank: 0,
                threshold: 0.5,
                lambda: 1.0,
            },
            InterfaceSeat {
                replica: 2,
                rank: 1,
                threshold: 1.0,
                lambda: 0.8,
            },
        ];
        assert!(retis_exchange_adjacent(&mut seats));
        assert_eq!(seats[0].replica, 2);
        assert_eq!(seats[1].replica, 1);
        assert!(!retis_exchange_adjacent(&mut seats));
    }

    #[test]
    fn a_seated_extra_keeps_its_interface_when_lambda_moves() {
        let first = assign_interfaces(&[(1, 0.1), (2, 0.4), (3, 0.9)], 1.0);
        assert_eq!(first[0].replica, 1);
        assert_eq!(first[2].replica, 3);
        let climbed = seat_extras(&first, &[(1, 0.95), (2, 0.4), (3, 0.9)], 1.0);
        assert_eq!(climbed[0].replica, 1);
        assert_eq!(climbed[0].rank, 0);
        assert!((climbed[0].lambda - 0.95).abs() < 1e-12);
        assert_eq!(climbed[2].replica, 3);
    }

    #[test]
    fn a_new_extra_takes_the_rank_a_departed_one_freed() {
        let first = assign_interfaces(&[(1, 0.1), (2, 0.4)], 1.0);
        assert_eq!(first[1].replica, 2);
        let swapped = seat_extras(&first, &[(1, 0.1), (7, 0.4)], 1.0);
        assert_eq!(swapped[0].replica, 1);
        assert_eq!(swapped[1].replica, 7);
        assert_eq!(swapped[1].rank, 1);
    }

    #[test]
    fn a_shrunk_ladder_reseats_every_extra() {
        let first = assign_interfaces(&[(1, 0.1), (2, 0.4), (3, 0.9)], 1.0);
        let shrunk = seat_extras(&first, &[(3, 0.9)], 1.0);
        assert_eq!(shrunk.len(), 1);
        assert_eq!(shrunk[0].replica, 3);
        assert_eq!(shrunk[0].rank, 0);
        assert!((shrunk[0].threshold - 1.0).abs() < 1e-12);
    }
}
