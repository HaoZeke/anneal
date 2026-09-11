use anneal_core::methods::cluster_hopping::{BiasUpdate, CheckpointAction};
use ndarray::Array1;

/// Flush enabled repulsive history when the decision continues local work.
/// Other actions retain their payloads and the queue's arrival multiplicity.
pub(crate) fn with_pending_deposits(
    pending: &mut Vec<Array1<f64>>,
    enabled: bool,
    decide: impl FnOnce(&mut Vec<Array1<f64>>) -> CheckpointAction,
) -> CheckpointAction {
    match decide(pending) {
        CheckpointAction::Continue if enabled && !pending.is_empty() => {
            CheckpointAction::DepositRemote {
                states: std::mem::take(pending),
            }
        }
        action => action,
    }
}

/// Select a pending bias update alongside the checkpoint's state decision.
pub(crate) fn with_pending_bias_update(
    action: CheckpointAction,
    pending: &mut Option<BiasUpdate>,
) -> CheckpointAction {
    match (action, pending.take()) {
        (
            CheckpointAction::Continue,
            Some(BiasUpdate::MergeWells {
                wells,
                weight,
                complete,
            }),
        ) => CheckpointAction::MergeBias {
            wells,
            weight,
            complete,
        },
        (
            CheckpointAction::Continue,
            Some(BiasUpdate::DepositDescriptors { deposits, weight }),
        ) => CheckpointAction::DepositDescriptors { deposits, weight },
        (action, _) => action,
    }
}

#[cfg(test)]
mod tests {
    use super::with_pending_deposits;
    use anneal_core::methods::cluster_hopping::CheckpointAction;
    use ndarray::{Array1, array};
    use std::cell::Cell;

    fn well(distance: f64) -> Array1<f64> {
        array![0.0, 0.0, 0.0, distance, 0.0, 0.0]
    }

    #[test]
    fn early_continue_drains_existing_and_decision_enqueued_deposits_once() {
        let first = well(1.2);
        let second = well(1.3);
        let mut pending = vec![first.clone()];
        let additions = vec![second.clone()];
        let decisions = Cell::new(0);
        let action = with_pending_deposits(&mut pending, true, |queue| {
            decisions.set(decisions.get() + 1);
            queue.extend(additions);
            CheckpointAction::Continue
        });

        assert_eq!(decisions.get(), 1);
        assert_eq!(
            action,
            CheckpointAction::DepositRemote {
                states: vec![first, second],
            }
        );
        assert!(pending.is_empty());
        assert_eq!(
            with_pending_deposits(&mut pending, true, |_| CheckpointAction::Continue),
            CheckpointAction::Continue
        );
    }

    #[test]
    fn disabled_sharing_runs_the_decision_without_draining_its_queue() {
        let first = well(1.2);
        let second = well(1.3);
        let mut pending = vec![first.clone()];
        let decisions = Cell::new(0);
        let action = with_pending_deposits(&mut pending, false, |queue| {
            decisions.set(decisions.get() + 1);
            queue.push(second.clone());
            CheckpointAction::Continue
        });

        assert_eq!(decisions.get(), 1);
        assert_eq!(action, CheckpointAction::Continue);
        assert_eq!(pending, vec![first, second]);
    }

    #[test]
    fn enabled_empty_queue_preserves_continue_and_calls_the_decision() {
        let mut pending = Vec::new();
        let decisions = Cell::new(0);
        let action = with_pending_deposits(&mut pending, true, |_| {
            decisions.set(decisions.get() + 1);
            CheckpointAction::Continue
        });

        assert_eq!(decisions.get(), 1);
        assert_eq!(action, CheckpointAction::Continue);
        assert!(pending.is_empty());
    }

    #[test]
    fn boundary_proposal_preserves_history_for_the_following_continue() {
        let first = well(1.2);
        let second = well(1.3);
        let mut pending = vec![first.clone()];
        let proposal = CheckpointAction::BoundaryProposal {
            state: well(1.4),
            action: "boundary-proposal".to_owned(),
        };
        let action = with_pending_deposits(&mut pending, true, |queue| {
            queue.push(second.clone());
            proposal.clone()
        });

        assert_eq!(action, proposal);
        assert_eq!(pending, vec![first.clone(), second.clone()]);
        let following = with_pending_deposits(&mut pending, true, |_| CheckpointAction::Continue);
        assert_eq!(
            following,
            CheckpointAction::DepositRemote {
                states: vec![first, second],
            }
        );
        assert!(pending.is_empty());
    }

    #[test]
    fn duplicate_occupancy_deposits_retain_their_multiplicity() {
        let occupied = well(1.2);
        let mut pending = vec![occupied.clone(), occupied.clone()];
        let action = with_pending_deposits(&mut pending, true, |queue| {
            queue.push(occupied.clone());
            CheckpointAction::Continue
        });

        assert_eq!(
            action,
            CheckpointAction::DepositRemote {
                states: vec![occupied.clone(), occupied.clone(), occupied],
            }
        );
        assert!(pending.is_empty());
    }

    #[test]
    fn other_actions_preserve_their_payloads_and_pending_history() {
        let actions = [
            CheckpointAction::ProbeProposal {
                state: well(1.4),
                action: "probe".to_owned(),
            },
            CheckpointAction::ExternalWork { external_calls: 37 },
            CheckpointAction::ExternalProposal {
                state: well(1.5),
                action: "external-proposal".to_owned(),
                external_calls: 41,
            },
            CheckpointAction::ExternalAdopt {
                state: well(1.6),
                action: "external-adopt".to_owned(),
                external_calls: 43,
            },
            CheckpointAction::DepositRemote {
                states: vec![well(1.7)],
            },
            CheckpointAction::Retire {
                reason: "occupancy".to_owned(),
            },
        ];
        for expected in actions {
            for enabled in [false, true] {
                let first = well(1.2);
                let second = well(1.3);
                let mut pending = vec![first.clone()];
                let decisions = Cell::new(0);
                let action = with_pending_deposits(&mut pending, enabled, |queue| {
                    decisions.set(decisions.get() + 1);
                    queue.push(second.clone());
                    expected.clone()
                });

                assert_eq!(decisions.get(), 1);
                assert_eq!(action, expected);
                assert_eq!(pending, vec![first, second]);
            }
        }
    }
}

#[cfg(test)]
mod pending_bias_tests {
    use super::with_pending_bias_update;
    use anneal_core::methods::cluster_hopping::{BiasUpdate, CheckpointAction};
    use ndarray::array;

    fn merge() -> BiasUpdate {
        BiasUpdate::MergeWells {
            wells: vec![(array![1.0, 2.0], 0.75)],
            weight: 0.5,
            complete: false,
        }
    }

    #[test]
    fn a_received_table_survives_every_state_action() {
        let actions = [
            CheckpointAction::Continue,
            CheckpointAction::DepositRemote {
                states: vec![array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            },
            CheckpointAction::BoundaryProposal {
                state: array![0.0, 0.0, 0.0, 1.2, 0.0, 0.0],
                action: "boundary".into(),
            },
            CheckpointAction::ExternalWork { external_calls: 37 },
            CheckpointAction::Retire {
                reason: "contract-boundary".into(),
            },
        ];
        for action in actions {
            let mut pending = Some(merge());
            assert_eq!(
                with_pending_bias_update(action.clone(), &mut pending),
                CheckpointAction::WithBiasUpdates {
                    updates: vec![merge()],
                    action: Box::new(action),
                }
            );
            assert!(pending.is_none());
            assert_eq!(
                with_pending_bias_update(CheckpointAction::Continue, &mut pending),
                CheckpointAction::Continue
            );
        }
    }

    #[test]
    fn no_received_table_preserves_the_state_action() {
        let action = CheckpointAction::ExternalAdopt {
            state: array![0.0, 0.0, 0.0, 1.2, 0.0, 0.0],
            action: "adoption".into(),
            external_calls: 43,
        };
        assert_eq!(with_pending_bias_update(action.clone(), &mut None), action);
    }

    #[test]
    fn an_existing_update_wrapper_keeps_its_payload_and_order() {
        let action = CheckpointAction::WithBiasUpdates {
            updates: vec![BiasUpdate::DepositDescriptors {
                deposits: vec![(array![1.0, 2.0], 3)],
                weight: 0.25,
            }],
            action: Box::new(CheckpointAction::Continue),
        };
        let mut pending = Some(merge());
        assert_eq!(
            with_pending_bias_update(action.clone(), &mut pending),
            CheckpointAction::WithBiasUpdates {
                updates: vec![merge()],
                action: Box::new(action),
            }
        );
        assert!(pending.is_none());
    }
}
