//! Validated-minimum history for cooperating optimization chains.

use std::sync::Mutex;
use std::time::Instant;

use ndarray::ArrayView1;

use crate::descriptor_space::{DescriptorError, DescriptorSpace, DescriptorVector};
use crate::methods::cluster_hopping::QuenchBoundary;
use crate::pes_exploration::{
    ExactStructureRelation, ExactStructureWitness, MinimumAdmission, PesNetwork, StructureContext,
};

/// Exact identity and accumulated visits returned to an escape controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistoryObservation {
    /// Descriptor-ordered, exact-witness identity decision.
    pub minimum: MinimumAdmission,
    /// Visits to this minimum, including the submitted observation.
    pub visits: u64,
}

/// A history observation could not be admitted.
#[derive(Debug, thiserror::Error)]
pub enum MinimumHistoryError {
    /// The gradient certificate needs a finite positive tolerance.
    #[error("minimum history requires a finite positive gradient tolerance")]
    InvalidTolerance,
    /// The charged quench has no sufficient fresh minimum certificate.
    #[error("minimum history requires a validated converged quench")]
    RejectedMinimum,
    /// Incompatible descriptor metadata prevents identity retrieval.
    #[error(transparent)]
    Descriptor(#[from] DescriptorError),
    /// The observation count cannot be represented.
    #[error("minimum history visit counter overflow")]
    CounterOverflow,
    /// Accepted membership requires an admitted minimum identity.
    #[error("minimum history has no identity {0}")]
    UnknownMinimum(usize),
}

/// Shared search memory without state adoption or an acceptance policy.
///
/// A private instance gives the isolated-chain comparison; serializing access
/// to one instance shares the same identity and visit rules across replicas.
/// Descriptors order the exact witness checks in [`PesNetwork`]. Descriptor
/// distance and packing-family membership never decide minimum identity.
pub struct MinimumHistory {
    network: PesNetwork,
    visits: Vec<u64>,
    accepted_visits: Vec<u64>,
    total_visits: u64,
    gradient_tolerance: f64,
}

impl MinimumHistory {
    /// Empty history with the caller's minimum-certificate tolerance.
    pub fn new(gradient_tolerance: f64) -> Result<Self, MinimumHistoryError> {
        if !gradient_tolerance.is_finite() || gradient_tolerance <= 0.0 {
            return Err(MinimumHistoryError::InvalidTolerance);
        }
        Ok(Self {
            network: PesNetwork::new(),
            visits: Vec::new(),
            accepted_visits: Vec::new(),
            total_visits: 0,
            gradient_tolerance,
        })
    }

    /// Number of exact minimum identities in the shared history.
    pub fn minimum_count(&self) -> usize {
        self.network.minimum_count()
    }

    /// Number of validated observations, including replica initializations.
    pub fn total_visits(&self) -> u64 {
        self.total_visits
    }

    /// Number of exact identities adopted by at least one chain.
    pub fn accepted_count(&self) -> usize {
        self.accepted_visits
            .iter()
            .filter(|visits| **visits > 0)
            .count()
    }

    /// Observations since first adoption, including that accepted visit.
    ///
    /// Zero denotes an archived proposal that no chain has accepted. Such a
    /// proposal remains eligible for an adaptive energy-threshold trial.
    pub fn accepted_visits(&self, minimum: usize) -> Option<u64> {
        self.accepted_visits.get(minimum).copied()
    }

    /// Publishes adoption without treating rejected proposals as visited states.
    ///
    /// Registration is idempotent. A shared caller holds its history lock
    /// across observation, threshold decision, and this publication.
    pub fn mark_accepted(&mut self, minimum: usize) -> Result<(), MinimumHistoryError> {
        let visits = self
            .accepted_visits
            .get_mut(minimum)
            .ok_or(MinimumHistoryError::UnknownMinimum(minimum))?;
        *visits = (*visits).max(1);
        Ok(())
    }

    /// Admit one charged minimum certificate and return only search history.
    pub fn observe<W: ExactStructureWitness + ?Sized>(
        &mut self,
        quench: &QuenchBoundary,
        descriptor: DescriptorVector,
        context: StructureContext,
        witness: &W,
    ) -> Result<HistoryObservation, MinimumHistoryError> {
        let gradient = quench
            .gradient()
            .ok_or(MinimumHistoryError::RejectedMinimum)?;
        let max_gradient = gradient.iter().map(|value| value.abs()).fold(0.0, f64::max);
        if max_gradient >= self.gradient_tolerance {
            return Err(MinimumHistoryError::RejectedMinimum);
        }
        let total_visits = self
            .total_visits
            .checked_add(1)
            .ok_or(MinimumHistoryError::CounterOverflow)?;
        let minimum = self.network.admit_minimum_with_context(
            quench.energy(),
            quench.state().to_owned(),
            max_gradient,
            descriptor,
            context,
            witness,
        )?;
        if minimum.is_new {
            self.visits.push(0);
            self.accepted_visits.push(0);
        }
        // Every basin count is bounded by the checked total observation count.
        self.visits[minimum.id] += 1;
        if self.accepted_visits[minimum.id] > 0 {
            self.accepted_visits[minimum.id] += 1;
        }
        self.total_visits = total_visits;
        Ok(HistoryObservation {
            minimum,
            visits: self.visits[minimum.id],
        })
    }
}

/// Which visits count as history when a chain classifies a reached minimum.
///
/// Goedecker's 2004 algorithm inserts a minimum into the history only after
/// threshold acceptance. Counting every certified observation instead spreads
/// exclusion to structures no chain ever stood in, which is a different
/// control. Both are kept so a comparison names which one it ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryMembership {
    /// Minima some chain has adopted count; archived proposals stay eligible.
    Accepted,
    /// Every certified observation counts, adopted or not.
    Observed,
}

impl HistoryMembership {
    /// Label used in run records.
    pub fn name(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Observed => "observed-exclusion",
        }
    }

    /// Parses the record label; `None` is the accepted-history default.
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value {
            None | Some("accepted") => Ok(Self::Accepted),
            Some("observed-exclusion") => Ok(Self::Observed),
            Some(other) => Err(format!(
                "invalid history policy {other:?}; expected accepted or observed-exclusion"
            )),
        }
    }
}

/// Membership decision for escape feedback under one policy.
///
/// Returns whether the minimum counts as new and the visit count the
/// feedback should scale by, both including the observation just made.
pub fn history_feedback_membership(
    policy: HistoryMembership,
    first_observation: bool,
    observed_visits: u64,
    accepted_visits: u64,
) -> (bool, u64) {
    match policy {
        HistoryMembership::Accepted => (accepted_visits == 0, accepted_visits),
        HistoryMembership::Observed => (first_observation, observed_visits),
    }
}

/// What one validated quench told the hop loop about the population's history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryReport {
    /// Exact minimum identity in the history.
    pub minimum: usize,
    /// Whether the minimum is new under the configured membership policy.
    pub is_new: bool,
    /// Visits under the policy, including this observation.
    pub visits: u64,
    /// Certified observations of this minimum by every chain, including this one.
    pub observed_visits: u64,
    /// Whether this observation created the exact identity.
    pub first_observation: bool,
}

/// Hop-loop side of a minimum history, private or shared.
///
/// The loop reports every certified quench it lands on and every state it
/// adopts. It receives identity and counts only: no coordinates, no
/// instruction to move. A private implementation is the isolated control;
/// one behind a lock shared by several chains is the communicating arm.
pub trait HistoryHook {
    /// Admits a certified minimum and returns its history.
    ///
    /// `gradient` is the fresh validation gradient of `state`. `None` means
    /// the history refused the observation, which the loop treats as no
    /// information rather than as a rejected structure.
    fn observe(
        &mut self,
        energy: f64,
        state: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
    ) -> Option<HistoryReport>;

    /// Publishes that the chain now stands in `minimum`.
    fn mark_accepted(&mut self, minimum: usize);

    /// Observations, refusals and seconds spent in the history.
    fn cost(&self) -> (usize, usize, f64);
}

/// Serialises one exact witness for use from several chains at once.
///
/// Exact matching is the only foreign call in the history, and it is not
/// re-entrant. The potential and the trajectories stay parallel; only the
/// identity decision queues.
pub struct SerializedWitness<W>(pub Mutex<W>);

impl<W: ExactStructureWitness> ExactStructureWitness for SerializedWitness<W> {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        self.0
            .lock()
            .expect("exact witness lock poisoned")
            .equivalent(left, right)
    }

    fn relation(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> ExactStructureRelation {
        self.0
            .lock()
            .expect("exact witness lock poisoned")
            .relation(left, right)
    }
}

/// [`HistoryHook`] over a [`MinimumHistory`] behind a lock.
///
/// The lock is held across observation and the policy read, so the report a
/// chain acts on is one consistent snapshot. Acceptance is published on a
/// separate lock acquisition; two chains adopting the same new minimum in the
/// same instant both classify it as new, which is the same race a serial
/// implementation would resolve by order of arrival.
pub struct SharedMinimumHistory<'a, W: ExactStructureWitness + ?Sized> {
    history: &'a Mutex<MinimumHistory>,
    descriptor: &'a DescriptorSpace,
    context: StructureContext,
    witness: &'a W,
    policy: HistoryMembership,
    observations: usize,
    refusals: usize,
    seconds: f64,
}

impl<'a, W: ExactStructureWitness + ?Sized> SharedMinimumHistory<'a, W> {
    /// A hook over `history` with the given membership policy.
    pub fn new(
        history: &'a Mutex<MinimumHistory>,
        descriptor: &'a DescriptorSpace,
        context: StructureContext,
        witness: &'a W,
        policy: HistoryMembership,
    ) -> Self {
        Self {
            history,
            descriptor,
            context,
            witness,
            policy,
            observations: 0,
            refusals: 0,
            seconds: 0.0,
        }
    }

    /// Membership policy this hook reports under.
    pub fn policy(&self) -> HistoryMembership {
        self.policy
    }
}

impl<W: ExactStructureWitness + ?Sized> HistoryHook for SharedMinimumHistory<'_, W> {
    fn observe(
        &mut self,
        energy: f64,
        state: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
    ) -> Option<HistoryReport> {
        let started = Instant::now();
        let report = (|| {
            let quench = QuenchBoundary::validated(energy, state.to_owned(), gradient.to_owned())?;
            let description = self
                .descriptor
                .describe(state, self.context.species())
                .ok()?;
            let mut history = self.history.lock().ok()?;
            let observation = history
                .observe(&quench, description, self.context.clone(), self.witness)
                .ok()?;
            let accepted = history.accepted_visits(observation.minimum.id)?;
            let (is_new, visits) = history_feedback_membership(
                self.policy,
                observation.minimum.is_new,
                observation.visits,
                accepted,
            );
            Some(HistoryReport {
                minimum: observation.minimum.id,
                is_new,
                visits,
                observed_visits: observation.visits,
                first_observation: observation.minimum.is_new,
            })
        })();
        self.seconds += started.elapsed().as_secs_f64();
        if report.is_some() {
            self.observations += 1;
        } else {
            self.refusals += 1;
        }
        report
    }

    fn mark_accepted(&mut self, minimum: usize) {
        let started = Instant::now();
        if let Ok(mut history) = self.history.lock() {
            // An unknown identity is a caller error; the publication is
            // idempotent otherwise and has nothing to report.
            let _ = history.mark_accepted(minimum);
        }
        self.seconds += started.elapsed().as_secs_f64();
    }

    fn cost(&self) -> (usize, usize, f64) {
        (self.observations, self.refusals, self.seconds)
    }
}

#[cfg(test)]
mod hook_tests {
    use super::*;
    use crate::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
    use ndarray::{Array1, array};

    /// Structures are the same point when their coordinates coincide.
    struct SameCoordinates;

    impl ExactStructureWitness for SameCoordinates {
        fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-9)
        }
    }

    fn triangle(scale: f64) -> Array1<f64> {
        array![
            0.0,
            0.0,
            0.0,
            scale,
            0.0,
            0.0,
            0.5 * scale,
            0.8 * scale,
            0.0
        ]
    }

    #[test]
    fn membership_policies_disagree_on_an_unadopted_minimum() {
        assert_eq!(
            history_feedback_membership(HistoryMembership::Accepted, false, 3, 0),
            (true, 0)
        );
        assert_eq!(
            history_feedback_membership(HistoryMembership::Observed, false, 3, 0),
            (false, 3)
        );
        assert_eq!(
            history_feedback_membership(HistoryMembership::Accepted, true, 1, 1),
            (false, 1)
        );
        assert_eq!(
            HistoryMembership::parse(None),
            Ok(HistoryMembership::Accepted)
        );
        assert!(HistoryMembership::parse(Some("both")).is_err());
    }

    #[test]
    fn two_hooks_over_one_history_see_each_other() {
        let history = Mutex::new(MinimumHistory::new(1e-3).unwrap());
        let descriptor = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
        let context = StructureContext::new(Some(vec![18; 3]), None, Some("test".into()));
        let witness = SerializedWitness(Mutex::new(SameCoordinates));
        let mut first = SharedMinimumHistory::new(
            &history,
            &descriptor,
            context.clone(),
            &witness,
            HistoryMembership::Accepted,
        );
        let mut second = SharedMinimumHistory::new(
            &history,
            &descriptor,
            context,
            &witness,
            HistoryMembership::Observed,
        );
        let zero = Array1::zeros(9);
        let a = first
            .observe(-1.0, triangle(1.0).view(), zero.view())
            .unwrap();
        assert!(a.first_observation && a.is_new && a.visits == 0);
        // Nobody has adopted it: accepted membership still calls it new for
        // the second chain, observed membership does not.
        let b = second
            .observe(-1.0, triangle(1.0).view(), zero.view())
            .unwrap();
        assert_eq!(b.minimum, a.minimum);
        assert!(!b.first_observation && !b.is_new);
        assert_eq!(b.observed_visits, 2);
        first.mark_accepted(a.minimum);
        let c = first
            .observe(-1.0, triangle(1.0).view(), zero.view())
            .unwrap();
        assert!(!c.is_new);
        assert_eq!(c.visits, 2, "accepted visits count from adoption onwards");
        // A different structure is new to both.
        let d = second
            .observe(-0.5, triangle(2.0).view(), zero.view())
            .unwrap();
        assert!(d.first_observation && d.is_new);
        assert_ne!(d.minimum, a.minimum);
        // An uncertified gradient is refused, not admitted.
        let bad = Array1::from_elem(9, 0.5);
        assert!(
            first
                .observe(-1.0, triangle(1.0).view(), bad.view())
                .is_none()
        );
        assert_eq!(first.cost().1, 1);
        assert_eq!(history.lock().unwrap().minimum_count(), 2);
    }
}
