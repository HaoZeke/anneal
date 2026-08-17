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

/// Ensemble stop. Mixing names a putative; extras still Leave.
/// Saturation of two occupied families, alone or with mixing, retires.
/// A published energy is not an argument.
pub fn occupancy_retire(certificate: OccupancyCertificate, catalog_saturated: bool) -> bool {
    match certificate {
        OccupancyCertificate::CatalogSaturated => true,
        OccupancyCertificate::MixingCertified => catalog_saturated,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        OccupancyCertificate, is_occupancy_leave_action, occupancy_complete, occupancy_retire,
        published_energy_score,
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
    fn good_turing_with_a_competing_family_is_done() {
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
            false
        ));
        assert!(occupancy_retire(
            OccupancyCertificate::MixingCertified,
            true
        ));
        assert!(occupancy_retire(
            OccupancyCertificate::CatalogSaturated,
            true
        ));
        assert_eq!(occupancy_complete(false, false, 8), None);
    }
}
