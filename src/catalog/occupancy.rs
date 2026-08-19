//! Occupancy leave. Not Hyperband successive-halving.
//!
//! ## Packing identity (DECAF)
//!
//! Per-center SOAP class histograms. Same family iff
//! \(\|h-h'\|_1 \le\) [`crate::catalog::PACKING_MERGE`] \(= 0.20\).
//! Sealed LJ75 icosahedral-versus-Marks L1 is \(0.69\).
//!
//! ## Inverted Gelman--Rubin
//!
//! On the family-label series of assigned walks,
//! \(\hat R <\) [`crate::catalog::MIXED_RHAT`] \(= 1.2\) with
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
//! Another DECAF family already on file and packing not saturated:
//! take a catalog representative of the least-occupied one (funnel
//! exchange). After packing saturation, or if no other family is on
//! file: record the occupied packing in the shared SOAP archive and
//! step into a hole of that archive, or amplify the fivefold residual
//! when the archive is empty. Occupancy extras do not draw a random
//! cluster.
//! A single hole-and-quench returns to the same family; the hop
//! then requenches and widens until DECAF says the family changed.
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
//! mixed, a mixed competitor, strictly more occupied. On LJ75 that
//! putative is the icosahedral shelf until a second funnel is deeper.
//! Packing Good--Turing names completeness of the seen codebook
//! (`packing_saturated`). Leftover-SOAP arrivals stay the hole
//! generator; leftover Good--Turing is not the stop. Hop re-observes
//! of the same well are not draws. A saturated packing census of
//! shallow families is not retire: extras keep Leaving so an unseen
//! funnel can still appear. Replicas retire when a mixing putative is
//! certified and that packing census is saturated and leftover SOAP
//! has dwelt under the unseen-mass ceiling and FunnelModel EI on
//! the seen packings is exhausted and the rematched
//! family floor is met. [`occupancy_min_families`]
//! (`CATALOG_MIN_FAMILIES`) is the floor so a known two-funnel hurdle
//! does not stop on one packing. Default 1 is Good--Turing alone.
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
    /// The quench sat in the occupied family. Keep the unquenched hole step.
    HoleStep,
    /// Same-family leftover hole. Do not adopt; the extra stays put
    /// and the hop loop draws another Leave.
    Refuse,
}

/// Where an occupancy extra goes when it Leaves.
///
/// Funnel exchange first, only while packing is unsaturated: a
/// catalog representative of a different, under-occupied packing.
/// That is how Leave includes Oh once Oh is on file. After packing
/// saturation, or if no other family is on file, a hole of the
/// shared occupied-packing archive, not a random cluster. Serial
/// recommended is a different mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyLeaveTarget {
    /// Coordinator has a representative of a different DECAF family.
    OtherFamily,
    /// Shared SOAP archive hole, or fivefold residual if the archive is empty.
    ArchiveHole,
}

/// Consecutive leftover-SOAP occupancy_gt records under the unseen-mass
/// ceiling that count as a dwell. One leftover-sat nick is not enough:
/// a hatch raises \(\hat p_0\) by \((n-n_1)/(n(n+1))\).
pub const LEFTOVER_SAT_DWELL: usize = 5;

/// Bank and CSA turn FunnelModel EI on at three observed morphologies.
pub const OCCUPANCY_EI_MIN_OBS: usize = 3;

/// Jones remaining improvement on seen packing morphologies.
///
/// The FunnelModel is the bank/CSA GP. Exhausted when that model has
/// the bank observation floor and the largest EI at observed packings
/// is at most the model's noise. Unseen families are leftover-dwell,
/// not a far-field GP probe.
pub fn occupancy_ei_exhausted(max_ei: f64, n_obs: usize, noise: f64) -> bool {
    n_obs >= OCCUPANCY_EI_MIN_OBS && max_ei.is_finite() && max_ei <= noise
}

/// Leftover dwell from consecutive leftover-sat bits, newest last.
///
/// The last [`LEFTOVER_SAT_DWELL`] records must all be saturated.
/// A one-shot nick, or a nick then a hatch, is not a dwell.
pub fn leftover_sat_dwell(consecutive: &[bool]) -> bool {
    consecutive.len() >= LEFTOVER_SAT_DWELL
        && consecutive
            .iter()
            .rev()
            .take(LEFTOVER_SAT_DWELL)
            .all(|&sat| sat)
}

/// Leave destination. After packing saturation OtherFamily only
/// rematches families on file, so Leave is the archive hole.
pub fn occupancy_leave_target(
    other_family_in_catalog: bool,
    packing_saturated: bool,
) -> OccupancyLeaveTarget {
    if packing_saturated || !other_family_in_catalog {
        OccupancyLeaveTarget::ArchiveHole
    } else {
        OccupancyLeaveTarget::OtherFamily
    }
}

/// Occupancy Leave that quenches onto a new DECAF family is taken.
/// A leftover-SOAP hole that stays in the occupied family is refused:
/// off-well ico is still ico. Reseeds take the new start even when
/// DECAF still names the old family.
pub fn occupancy_leave_adopt(action: &str, family_changed: bool) -> Option<OccupancyLeaveAdopt> {
    if !is_occupancy_leave_action(action) {
        return None;
    }
    if action == "catalog_leave" && !family_changed {
        Some(OccupancyLeaveAdopt::Refuse)
    } else {
        Some(OccupancyLeaveAdopt::Quench)
    }
}

/// Why occupancy may retire a replica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyCertificate {
    /// Occupant chains mixed onto a uniquely deepest attractor that
    /// is strictly more occupied than every mixed competitor.
    MixingCertified,
    /// Packing Good--Turing unseen mass is small, and the rematched
    /// family count meets [`occupancy_min_families`]. Default floor 1
    /// is Good--Turing alone; paper ensembles set 2.
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
/// alone: stop when no new packings appear. Set
/// `CATALOG_MIN_FAMILIES=2` for two-funnel paper hurdles.
pub const DEFAULT_MIN_OCCUPIED_FAMILIES: usize = 1;

/// User floor on occupied DECAF families before saturation retires.
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

/// Replica retire. Mixing certified, packing Good--Turing, leftover
/// dwell, FunnelModel EI exhausted on seen packings, and the rematched
/// family floor. CatalogSaturated names census completeness and does
/// not retire: extras keep Leaving.
/// `catalog_saturated` is packing-family saturation, not leftover-SOAP.
/// `leftover_dwell` is consecutive leftover-sat occupancy_gt records,
/// not a one-shot leftover nick.
/// `ei_exhausted` is Jones remaining improvement on observed
/// FunnelModel morphologies, not a far-field GP probe.
/// `n_occupied_families` is the rematched packing count, not a
/// leftover-SOAP basin count and not `2 * certified`.
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
    n_occupied_families >= min_occupied_families
        && catalog_saturated
        && leftover_dwell
        && ei_exhausted
        && matches!(certificate, OccupancyCertificate::MixingCertified)
}

#[cfg(test)]
mod tests {
    use super::{
        CHAMPION_RANK, InterfaceSeat, LeavePath, OccupancyCertificate, OccupancyLeaveAdopt,
        OccupancyLeaveTarget, PackingRole, assign_interfaces, in_interface_ensemble,
        interface_ladder, is_occupancy_leave_action, leave_shot_accepted, leftover_lambda,
        leftover_sat_dwell, occupancy_complete, occupancy_complete_at, occupancy_ei_exhausted,
        occupancy_leave_adopt, occupancy_leave_target, occupancy_retire, occupancy_retire_at,
        packing_role,
        promote_one_sided,
        published_energy_score, retis_exchange_adjacent, retis_should_swap, seat_extras,
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
            occupancy_leave_target(true, false),
            OccupancyLeaveTarget::OtherFamily
        );
        assert_eq!(
            occupancy_leave_target(false, false),
            OccupancyLeaveTarget::ArchiveHole
        );
        assert_ne!(
            occupancy_leave_target(false, false),
            OccupancyLeaveTarget::OtherFamily
        );
    }

    #[test]
    fn packing_sat_leave_is_archive_hole_not_other_family() {
        assert_eq!(
            occupancy_leave_target(true, true),
            OccupancyLeaveTarget::ArchiveHole
        );
        assert_ne!(
            occupancy_leave_target(true, true),
            OccupancyLeaveTarget::OtherFamily
        );
        assert_eq!(
            occupancy_leave_target(false, true),
            OccupancyLeaveTarget::ArchiveHole
        );
        assert_eq!(
            occupancy_leave_target(true, false),
            OccupancyLeaveTarget::OtherFamily
        );
        assert_eq!(
            occupancy_leave_target(false, false),
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
    fn leftover_unsaturated_does_not_retire_with_mixing_and_packing_sat() {
        assert!(!leftover_sat_dwell(&[false]));
        assert!(!leftover_sat_dwell(&[true]));
        assert!(!occupancy_retire_at(
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
        assert!(!occupancy_retire_at(
            OccupancyCertificate::MixingCertified,
            true,
            leftover_sat_dwell(&[true]),
            true,
            2,
            2
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
