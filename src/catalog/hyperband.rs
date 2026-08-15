//! Asynchronous successive halving on cooperative walks.
//!
//! Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar, *Hyperband*, JMLR 18
//! (2018). Optuna and Ray Tune (Riedel / CoE RAISE) run the same
//! schedule: resource multiplies by \(\eta\) at each rung, and only the
//! top \(1/\eta\) of a cohort continue.
//!
//! The ranking here is packing-first. The sole occupant of a family is
//! a new basin and is never pruned. Extra occupants of a crowded
//! packing compete on energy. A walk with no packing yet is a possible
//! new family and is kept. Pruned hops reseed a new start so the
//! ensemble builds more basins instead of walking the same well.

use std::collections::BTreeMap;

/// Reduction factor \(\eta\). Same default as Optuna `HyperbandPruner`.
pub const REDUCTION_FACTOR: u32 = 3;
/// First rung, in hops. Shorter walks have no packing identity yet.
pub const MIN_RESOURCE: u64 = 64;
/// Hop cap used when the coordinator has no explicit rung ceiling.
pub const DEFAULT_MAX_RESOURCE: u64 = 1000;

/// One live walk at the moment a rung is scored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WalkRecord {
    /// Replica identity.
    pub id: u32,
    /// Hops consumed on the current start.
    pub resource: u64,
    /// Best energy on this start.
    pub energy: f64,
    /// Packing-family index, or none before DECAF assigns one.
    pub family: Option<usize>,
}

/// Successive-halving rungs \(r_{\min}, \eta r_{\min}, \ldots \le R\).
pub fn rungs(max_resource: u64, min_resource: u64, eta: u32) -> Vec<u64> {
    if eta < 2 || min_resource == 0 || max_resource < min_resource {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut rung = min_resource;
    while rung <= max_resource {
        out.push(rung);
        let next = rung.saturating_mul(u64::from(eta));
        if next <= rung {
            break;
        }
        rung = next;
    }
    out
}

/// Highest rung this walk has reached, if any.
pub fn current_rung(resource: u64, max_resource: u64) -> Option<u64> {
    rungs(max_resource, MIN_RESOURCE, REDUCTION_FACTOR)
        .into_iter()
        .rev()
        .find(|&rung| resource >= rung)
}

/// Whether `id` is discarded at its current rung.
///
/// Unique-family occupants and unassigned packings stay. Extra
/// occupants of a crowded family keep the best \(\lfloor n/\eta\rfloor\)
/// by energy.
pub fn prune(walks: &[WalkRecord], id: u32, max_resource: u64) -> bool {
    let Some(self_walk) = walks.iter().find(|walk| walk.id == id) else {
        return false;
    };
    let Some(rung) = current_rung(self_walk.resource, max_resource) else {
        return false;
    };
    if self_walk.family.is_none() {
        return false;
    }
    let cohort: Vec<&WalkRecord> = walks
        .iter()
        .filter(|walk| walk.resource >= rung)
        .collect();
    if cohort.len() < 2 {
        return false;
    }
    let mut best_of_family: BTreeMap<usize, (u32, f64)> = BTreeMap::new();
    for walk in &cohort {
        let Some(family) = walk.family else {
            continue;
        };
        match best_of_family.get(&family) {
            None => {
                best_of_family.insert(family, (walk.id, walk.energy));
            }
            Some((_, energy)) if walk.energy < *energy - 1e-12 => {
                best_of_family.insert(family, (walk.id, walk.energy));
            }
            _ => {}
        }
    }
    if self_walk.family.is_some_and(|family| {
        best_of_family
            .get(&family)
            .is_some_and(|(best_id, _)| *best_id == id)
    }) {
        return false;
    }
    let mut extras: Vec<&WalkRecord> = cohort
        .iter()
        .copied()
        .filter(|walk| {
            walk.family.is_some_and(|family| {
                best_of_family
                    .get(&family)
                    .is_some_and(|(best_id, _)| *best_id != walk.id)
            })
        })
        .collect();
    extras.sort_by(|left, right| {
        left.energy
            .partial_cmp(&right.energy)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.id.cmp(&right.id))
    });
    let keep = extras.len() / usize::try_from(REDUCTION_FACTOR).unwrap_or(3);
    extras.iter().skip(keep).any(|walk| walk.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walk(id: u32, resource: u64, energy: f64, family: Option<usize>) -> WalkRecord {
        WalkRecord {
            id,
            resource,
            energy,
            family,
        }
    }

    #[test]
    fn rungs_are_min_times_eta_up_to_the_hop_cap() {
        assert_eq!(rungs(1000, 64, 3), vec![64, 192, 576]);
        assert_eq!(current_rung(63, 1000), None);
        assert_eq!(current_rung(64, 1000), Some(64));
        assert_eq!(current_rung(200, 1000), Some(192));
        assert_eq!(current_rung(1000, 1000), Some(576));
    }

    #[test]
    fn a_sole_occupant_of_a_family_is_never_pruned() {
        let walks = [
            walk(0, 64, -173.25, Some(0)),
            walk(1, 64, -173.92, Some(1)),
        ];
        assert!(!prune(&walks, 0, 1000));
        assert!(!prune(&walks, 1, 1000));
    }

    #[test]
    fn an_unassigned_packing_is_never_pruned() {
        let walks = [
            walk(0, 64, -170.0, None),
            walk(1, 64, -173.25, Some(0)),
            walk(2, 64, -173.24, Some(0)),
        ];
        assert!(!prune(&walks, 0, 1000));
    }

    #[test]
    fn extra_ico_occupants_are_pruned_and_the_best_ico_and_oh_stay() {
        let mut walks = vec![
            walk(0, 64, -173.928427, Some(1)),
            walk(1, 64, -173.252378, Some(0)),
        ];
        for index in 2..12 {
            walks.push(walk(index, 64, -173.25 + f64::from(index) * 1e-4, Some(0)));
        }
        assert!(!prune(&walks, 0, 1000));
        assert!(!prune(&walks, 1, 1000));
        let pruned: Vec<u32> = walks
            .iter()
            .map(|walk| walk.id)
            .filter(|&id| prune(&walks, id, 1000))
            .collect();
        assert!(!pruned.is_empty());
        assert!(!pruned.contains(&0));
        assert!(!pruned.contains(&1));
        assert!(pruned.contains(&11));
    }

    #[test]
    fn walks_below_the_first_rung_are_not_ranked() {
        let walks = [
            walk(0, 10, -170.0, Some(0)),
            walk(1, 10, -169.0, Some(0)),
        ];
        assert!(!prune(&walks, 0, 1000));
        assert!(!prune(&walks, 1, 1000));
    }
}
