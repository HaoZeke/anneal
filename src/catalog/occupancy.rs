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

/// Actions that must land off the occupied leftover-SOAP well.
pub fn is_occupancy_leave_action(action: &str) -> bool {
    matches!(
        action,
        "hyperband_reseed" | "catalog_leave" | "population_reseed"
    )
}

#[cfg(test)]
mod tests {
    use super::is_occupancy_leave_action;

    #[test]
    fn feynman_kac_extras_use_the_same_leave_as_occupancy() {
        assert!(is_occupancy_leave_action("population_reseed"));
        assert!(is_occupancy_leave_action("hyperband_reseed"));
        assert!(is_occupancy_leave_action("catalog_leave"));
        assert!(!is_occupancy_leave_action("catalog_incumbent"));
        assert!(!is_occupancy_leave_action("bridge"));
    }
}
