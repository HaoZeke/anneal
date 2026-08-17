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
//! Hole on the leftover-SOAP sphere: 48 unit samples in the open
//! hemisphere away from the occupied-well centroid, scored by
//! nearest-well distance. Walk the packing mean toward that hole.
//! Quench; if the quench sits in a stored well, retry. Feynman--Kac
//! extras use this start. They do not draw a random cluster.
//!
//! ## Stop
//!
//! A mixing certificate names a putative: uniquely deepest, occupant
//! mixed, a mixed competitor, strictly more occupied. It does not
//! retire extras. On LJ75 that putative is the icosahedral shelf
//! until a second funnel is deeper, so retiring on mixing alone
//! stops the search before Marks. Occupancy retires when Good--Turing
//! saturates two occupied families, or when mixing and that
//! saturation hold together. Occupant \(\hat R\) uses
//! [`crate::catalog::CERTIFY_MIN_SAMPLES`] traces; two-point quenches
//! on two random-start families are not a certificate.
//! A published energy (Cambridge or otherwise) is a score, not a
//! stop. Leftover-SOAP saturation on one family is collapse, not
//! completeness: revisiting an icosahedral well drives the unseen
//! mass down without opening a second funnel.

/// Actions that must land off the occupied leftover-SOAP well.
pub fn is_occupancy_leave_action(action: &str) -> bool {
    matches!(
        action,
        "hyperband_reseed" | "catalog_leave" | "population_reseed"
    )
}

/// Why occupancy may retire a replica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyCertificate {
    /// Occupant chains mixed onto a uniquely deepest attractor that
    /// is strictly more occupied than every mixed competitor.
    MixingCertified,
    /// Good--Turing unseen mass of the observed census is small, and
    /// at least two occupied packing families are on file.
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

/// Generic occupancy completeness. `n_occupied_families` is the DECAF
/// packing count, not leftover-SOAP basin count.
pub fn occupancy_complete(
    mixing_certified: bool,
    catalog_saturated: bool,
    n_occupied_families: usize,
) -> Option<OccupancyCertificate> {
    if mixing_certified {
        Some(OccupancyCertificate::MixingCertified)
    } else if catalog_saturated && n_occupied_families >= 2 {
        Some(OccupancyCertificate::CatalogSaturated)
    } else {
        None
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
    let mut extras = extras.to_vec();
    extras.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
    let ladder = interface_ladder(extras.len(), horizon);
    extras
        .iter()
        .enumerate()
        .map(|(index, (replica, lambda))| InterfaceSeat {
            replica: *replica,
            rank: index as u32,
            threshold: ladder.get(index).copied().unwrap_or(horizon),
            lambda: *lambda,
        })
        .collect()
}

/// Swap adjacent interface seats when both extras have crossed.
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
        if retis_should_swap(left.lambda, left.threshold, right.lambda, right.threshold) {
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
            && left.lambda > right.lambda
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

/// Ensemble stop. Leftover-SOAP Good--Turing names a putative; extras
/// still search. Twenty leftover-SOAP visits on 24 talking chains is
/// not two occupied funnels. Retire only when inverted-GR mixing is
/// certified, two rematched DECAF families are on file, and that
/// leftover-SOAP census is saturated. A fabricated family count of
/// `2 * certified` is not that count.
pub fn occupancy_retire(
    certificate: OccupancyCertificate,
    catalog_saturated: bool,
    n_occupied_families: usize,
) -> bool {
    n_occupied_families >= 2
        && catalog_saturated
        && matches!(certificate, OccupancyCertificate::MixingCertified)
}

#[cfg(test)]
mod tests {
    use super::{
        CHAMPION_RANK, InterfaceSeat, LeavePath, OccupancyCertificate, assign_interfaces,
        in_interface_ensemble, interface_ladder, is_occupancy_leave_action, leave_shot_accepted,
        leftover_lambda, occupancy_complete, occupancy_retire, promote_one_sided,
        published_energy_score, retis_exchange_adjacent, retis_should_swap,
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
    fn leftover_soap_saturation_on_one_family_is_not_done() {
        assert_eq!(occupancy_complete(false, true, 1), None);
        assert_eq!(occupancy_complete(false, true, 0), None);
    }

    #[test]
    fn leftover_soap_gt_with_two_packings_is_putative() {
        assert_eq!(
            occupancy_complete(false, true, 2),
            Some(OccupancyCertificate::CatalogSaturated)
        );
        assert!(!occupancy_retire(
            OccupancyCertificate::CatalogSaturated,
            true,
            2
        ));
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
            2
        ));
        assert!(!occupancy_retire(
            OccupancyCertificate::MixingCertified,
            true,
            1
        ));
        assert!(occupancy_retire(
            OccupancyCertificate::MixingCertified,
            true,
            2
        ));
        assert!(!occupancy_retire(
            OccupancyCertificate::CatalogSaturated,
            true,
            2
        ));
        assert!(!occupancy_retire(
            OccupancyCertificate::CatalogSaturated,
            true,
            1
        ));
        assert_eq!(occupancy_complete(false, false, 8), None);
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
        let extras = [(1, 1.0), (2, 0.1)];
        let mut seats = assign_interfaces(&extras, 1.0);
        assert_eq!(seats[0].replica, 2);
        assert_eq!(seats[1].replica, 1);
        assert!(promote_one_sided(&mut seats) || seats[1].replica == 1);
        assert_eq!(seats[1].replica, 1);
        assert!(in_interface_ensemble(seats[1].lambda, seats[1].threshold));
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
        let extras = [(1, 1.0), (2, 0.8)];
        let mut seats = assign_interfaces(&extras, 1.0);
        assert!(retis_exchange_adjacent(&mut seats));
        assert_eq!(seats[0].replica, 2);
        assert_eq!(seats[1].replica, 1);
        assert!(!retis_exchange_adjacent(&mut seats));
    }
}
