use anneal_core::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
use anneal_core::methods::cluster_hopping::Ledger;
use anneal_core::methods::minima_hopping::{
    EscapeFeedback, HistoryHook, HistoryMembership, MinimumHistory, SerializedWitness,
    SharedMinimumHistory, Visit,
};
use anneal_core::pes_exploration::{
    ExactStructureRelation, ExactStructureWitness, StructureContext, StructureView,
};
use ndarray::{Array1, ArrayView1, array};
use std::cell::Cell;
use std::sync::Mutex;

struct CartesianWitness;

impl ExactStructureWitness for CartesianWitness {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        left == right
    }
}

fn certificate(state: &Array1<f64>, gradient: Option<Array1<f64>>) -> Ledger {
    let mut ledger = Ledger::new(1);
    assert!(ledger.charge());
    assert!(ledger.record_quench_boundary(0, -1.0, state.clone(), gradient));
    ledger
}

#[test]
fn a_peer_discovery_changes_escape_feedback_without_an_acceptance_trial() {
    let space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let first = array![0.0, 0.0, 0.0];
    let second = array![1.0, 0.0, 0.0];
    let first_certificate = certificate(&first, Some(Array1::zeros(3)));
    let second_certificate = certificate(&second, Some(Array1::zeros(3)));
    let mut history = MinimumHistory::new(1e-3).unwrap();
    let mut observe = |ledger: &Ledger| {
        let minimum = &ledger.quench_boundaries()[0];
        history
            .observe(
                minimum,
                space.describe(minimum.state(), Some(&[1])).unwrap(),
                StructureContext::default(),
                &CartesianWitness,
            )
            .unwrap()
    };
    let source = observe(&first_certificate);
    let peer = observe(&second_certificate);
    let revisited = observe(&second_certificate);
    let mut feedback = EscapeFeedback::new(1.0, 0.8);
    feedback.register_initial(source.minimum.id);
    let threshold = feedback.threshold();

    let visit = feedback.observe_shared(
        Some(source.minimum.id),
        revisited.minimum.id,
        revisited.minimum.is_new,
        revisited.visits,
    );

    assert!(peer.minimum.is_new);
    assert!(!revisited.minimum.is_new);
    assert_eq!(revisited.minimum.id, peer.minimum.id);
    assert_eq!(revisited.visits, 2);
    assert_eq!(visit, Visit::Known);
    assert!(feedback.escape() > 1.0);
    assert_eq!(feedback.threshold(), threshold);
    assert_eq!(history.minimum_count(), 2);
    assert_eq!(history.total_visits(), 3);
}

#[test]
fn descriptor_aliases_do_not_merge_exactly_distinct_minima() {
    let space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let first = array![0.0, 0.0, 0.0];
    let second = array![1.0, 0.0, 0.0];
    let descriptor = space.describe(first.view(), Some(&[1])).unwrap();
    let mut history = MinimumHistory::new(1e-3).unwrap();
    for state in [&first, &second] {
        let ledger = certificate(state, Some(Array1::zeros(3)));
        let observation = history
            .observe(
                &ledger.quench_boundaries()[0],
                descriptor.clone(),
                StructureContext::default(),
                &CartesianWitness,
            )
            .unwrap();
        assert!(observation.minimum.is_new);
        assert_eq!(observation.visits, 1);
    }
    assert_eq!(history.minimum_count(), 2);
    assert_eq!(history.total_visits(), 2);
}

#[test]
fn missing_or_insufficient_gradient_evidence_cannot_enter_shared_history() {
    let space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let state = array![0.0, 0.0, 0.0];
    let descriptor = space.describe(state.view(), Some(&[1])).unwrap();
    let mut history = MinimumHistory::new(1e-3).unwrap();
    for gradient in [None, Some(array![0.1, 0.0, 0.0])] {
        let ledger = certificate(&state, gradient);
        assert!(
            history
                .observe(
                    &ledger.quench_boundaries()[0],
                    descriptor.clone(),
                    StructureContext::default(),
                    &CartesianWitness,
                )
                .is_err()
        );
        assert_eq!(history.minimum_count(), 0);
        assert_eq!(history.total_visits(), 0);
    }
}

#[test]
fn invalid_gradient_tolerances_are_rejected() {
    for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(MinimumHistory::new(tolerance).is_err());
    }
}

#[test]
fn rejected_candidates_remain_eligible_until_the_energy_threshold_accepts() {
    let space = universal_descriptor_space(DescriptorGeometry::finite(1.0).unwrap());
    let source = array![0.0, 0.0, 0.0];
    let candidate = array![1.0, 0.0, 0.0];
    let source_certificate = certificate(&source, Some(Array1::zeros(3)));
    let candidate_certificate = certificate(&candidate, Some(Array1::zeros(3)));
    let mut history = MinimumHistory::new(1e-3).unwrap();
    let observe = |history: &mut MinimumHistory, ledger: &Ledger| {
        let boundary = &ledger.quench_boundaries()[0];
        history
            .observe(
                boundary,
                space.describe(boundary.state(), Some(&[1])).unwrap(),
                StructureContext::default(),
                &CartesianWitness,
            )
            .unwrap()
    };
    let source_id = observe(&mut history, &source_certificate).minimum.id;
    history.mark_accepted(source_id).unwrap();
    let mut feedback = EscapeFeedback::new(1.0, 0.8);
    feedback.register_initial(source_id);
    let mut rejections = 0;
    let mut accepted_id = None;
    for _ in 0..10 {
        let observed = observe(&mut history, &candidate_certificate);
        let id = observed.minimum.id;
        let visits = history.accepted_visits(id).unwrap();
        assert_eq!(visits, 0, "a rejected proposal is not an accepted visit");
        assert_eq!(
            feedback.observe_shared(Some(source_id), id, visits == 0, visits),
            Visit::New
        );
        if feedback.accept(1.0) {
            history.mark_accepted(id).unwrap();
            accepted_id = Some(id);
            break;
        }
        rejections += 1;
    }
    assert_eq!(rejections, 5);
    assert_eq!(history.minimum_count(), 2);
    assert_eq!(history.total_visits(), 7);
    assert_eq!(history.accepted_count(), 2);
    let accepted_id = accepted_id.expect("adaptive threshold must reconsider the uphill minimum");
    assert_eq!(history.accepted_visits(accepted_id), Some(1));
    let peer = observe(&mut history, &candidate_certificate);
    assert_eq!(
        peer.visits, 7,
        "the structural observation archive retains every proposal"
    );
    let visits = history.accepted_visits(peer.minimum.id).unwrap();
    assert_eq!(visits, 2);
    let threshold = feedback.threshold();
    assert_eq!(
        feedback.observe_shared(Some(source_id), peer.minimum.id, visits == 0, visits),
        Visit::Known
    );
    assert_eq!(feedback.threshold(), threshold);
}

#[test]
fn an_unknown_minimum_cannot_enter_accepted_history() {
    let mut history = MinimumHistory::new(1e-3).unwrap();
    assert!(history.mark_accepted(0).is_err());
    assert_eq!(history.accepted_visits(0), None);
    assert_eq!(history.accepted_count(), 0);
    assert_eq!(history.minimum_count(), 0);
    assert_eq!(history.total_visits(), 0);
}

struct ContextualWitness {
    context: StructureContext,
    left: Array1<f64>,
    right: Array1<f64>,
    raw_calls: Cell<usize>,
    contextual_calls: Cell<usize>,
}

impl ContextualWitness {
    fn slab_images() -> Self {
        let geometry = DescriptorGeometry::new(
            1.0,
            Some([6.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 12.0]),
            [true, true, false],
        )
        .unwrap();
        Self {
            context: StructureContext::new(
                Some(vec![29, 1]),
                Some(geometry),
                Some("fixed-substrate-history".into()),
            )
            .with_masses(Some(vec![63.546, 1.008])),
            left: array![0.0, 0.0, 0.0, 1.25, 0.5, 1.0],
            right: array![0.0, 0.0, 0.0, 7.25, 0.5, 1.0],
            raw_calls: Cell::new(0),
            contextual_calls: Cell::new(0),
        }
    }
}

impl ExactStructureWitness for ContextualWitness {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        self.relation(left, right).is_equivalent()
    }

    fn relation(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> ExactStructureRelation {
        assert_eq!(left, self.left.view());
        assert_eq!(right, self.right.view());
        self.raw_calls.set(self.raw_calls.get() + 1);
        ExactStructureRelation::Distinct
    }

    fn equivalent_structures(&self, left: StructureView<'_>, right: StructureView<'_>) -> bool {
        self.relation_structures(left, right).is_equivalent()
    }

    fn relation_structures(
        &self,
        left: StructureView<'_>,
        right: StructureView<'_>,
    ) -> ExactStructureRelation {
        assert_eq!(left.coordinates, self.left.view());
        assert_eq!(right.coordinates, self.right.view());
        assert_eq!(left.context, &self.context);
        assert_eq!(right.context, &self.context);
        self.contextual_calls.set(self.contextual_calls.get() + 1);
        ExactStructureRelation::Equivalent
    }
}

#[test]
fn serialized_witness_preserves_raw_coordinate_dispatch() {
    let inner = ContextualWitness::slab_images();
    let left = inner.left.clone();
    let right = inner.right.clone();
    let witness = SerializedWitness(Mutex::new(inner));

    assert_eq!(
        witness.relation(left.view(), right.view()),
        ExactStructureRelation::Distinct
    );
    assert!(!witness.equivalent(left.view(), right.view()));
    let inner = witness.0.lock().unwrap();
    assert_eq!(inner.raw_calls.get(), 2);
    assert_eq!(inner.contextual_calls.get(), 0);
}

#[test]
fn serialized_witness_preserves_contextual_relation_dispatch() {
    let inner = ContextualWitness::slab_images();
    let left = inner.left.clone();
    let right = inner.right.clone();
    let context = inner.context.clone();
    let witness = SerializedWitness(Mutex::new(inner));

    assert_eq!(
        witness.relation_structures(
            StructureView {
                coordinates: left.view(),
                context: &context,
            },
            StructureView {
                coordinates: right.view(),
                context: &context,
            },
        ),
        ExactStructureRelation::Equivalent
    );
    let inner = witness.0.lock().unwrap();
    assert_eq!(inner.raw_calls.get(), 0);
    assert_eq!(inner.contextual_calls.get(), 1);
}

#[test]
fn serialized_witness_preserves_contextual_equivalence_dispatch() {
    let inner = ContextualWitness::slab_images();
    let left = inner.left.clone();
    let right = inner.right.clone();
    let context = inner.context.clone();
    let witness = SerializedWitness(Mutex::new(inner));

    assert!(witness.equivalent_structures(
        StructureView {
            coordinates: left.view(),
            context: &context,
        },
        StructureView {
            coordinates: right.view(),
            context: &context,
        },
    ));
    let inner = witness.0.lock().unwrap();
    assert_eq!(inner.raw_calls.get(), 0);
    assert_eq!(inner.contextual_calls.get(), 1);
}

#[test]
fn serialized_witness_preserves_contextual_shared_history_admission() {
    let inner = ContextualWitness::slab_images();
    let left = inner.left.clone();
    let right = inner.right.clone();
    let context = inner.context.clone();
    let space = universal_descriptor_space(context.geometry().unwrap());
    let witness = SerializedWitness(Mutex::new(inner));
    let history = Mutex::new(MinimumHistory::new(1e-3).unwrap());
    let mut hook = SharedMinimumHistory::new(
        &history,
        &space,
        context,
        &witness,
        HistoryMembership::Accepted,
    );
    let gradient = Array1::zeros(left.len());

    let first = hook.observe(-1.0, left.view(), gradient.view()).unwrap();
    assert!(first.first_observation);
    hook.mark_accepted(first.minimum);
    let repeated = hook.observe(-1.0, right.view(), gradient.view()).unwrap();

    assert_eq!(repeated.minimum, first.minimum);
    assert!(!repeated.first_observation);
    assert!(!repeated.is_new);
    assert_eq!(repeated.observed_visits, 2);
    assert_eq!(repeated.visits, 2);
    let history = history.lock().unwrap();
    assert_eq!(history.minimum_count(), 1);
    assert_eq!(history.total_visits(), 2);
    assert_eq!(history.accepted_count(), 1);
    let inner = witness.0.lock().unwrap();
    assert_eq!(inner.raw_calls.get(), 0);
    assert_eq!(inner.contextual_calls.get(), 1);
}
