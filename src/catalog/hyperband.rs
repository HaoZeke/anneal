//! Shared packing-first ranking for Hyperband rungs and Feynman--Kac epochs.
//!
//! Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar, *Hyperband*, JMLR 18
//! (2018). Optuna and Ray Tune (Riedel / CoE RAISE) run the same
//! schedule: resource multiplies by \(\eta\) at each rung, and only the
//! top \(1/\eta\) of a cohort continue.
//!
//! Hyperband and Feynman--Kac are not two allocators. Both call
//! [`verdict`] and [`keep_ids`]. Ranking applies as soon as DECAF
//! assigns a family. Waiting for a rung lets extras adopt a deeper
//! isomer of a crowded packing instead of reseeding. Feynman--Kac
//! applies the same ranking at a closed epoch. A sole family occupant
//! is never reseeded. An unknown packing is a possible new family and
//! is kept. Extra occupants of a crowded family compete on energy;
//! beyond \(\lfloor n/\eta\rfloor\) extras they reseed a new random
//! start, not a hole and not a parent clone.

use std::collections::{BTreeMap, BTreeSet};

/// Reduction factor \(\eta\). Same default as Optuna `HyperbandPruner`.
pub const REDUCTION_FACTOR: u32 = 3;
/// First rung, in hops. Shorter walks have no packing identity yet.
/// Extras of a crowded family reseed here so they leave toward an
/// unexplored leftover-SOAP hole; they do not wait for a serial
/// recommended first-hit hop.
pub const MIN_RESOURCE: u64 = 64;
/// Hop cap used when the coordinator has no explicit rung ceiling.
///
/// Serial recommended Marks first hits sit at hops 33847--58779. A
/// 1000-hop ceiling reseeds extras before any of those walks can
/// leave the icosahedral shelf.
pub const DEFAULT_MAX_RESOURCE: u64 = 60000;

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

/// Keep-or-reseed decision for one walk in a Hyperband or Feynman--Kac cohort.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnsembleVerdict {
    /// Continue the current start.
    Keep,
    /// Draw a new random start. Not a hole and not a parent clone.
    Reseed,
}

/// Ranking of one walk among a cohort.
///
/// Unique-family occupants and unassigned packings stay. Extra
/// occupants of a crowded family keep the best \(\lfloor n/\eta\rfloor\)
/// by energy.
pub fn verdict(walks: &[WalkRecord], id: u32, max_resource: u64) -> EnsembleVerdict {
    let Some(self_walk) = walks.iter().find(|walk| walk.id == id) else {
        return EnsembleVerdict::Keep;
    };
    if self_walk.family.is_none() {
        return EnsembleVerdict::Keep;
    }
    let Some(rung) = current_rung(self_walk.resource, max_resource) else {
        return EnsembleVerdict::Keep;
    };
    occupancy_of_family(walks, id, rung)
}

fn occupancy_of_family(walks: &[WalkRecord], id: u32, rung: u64) -> EnsembleVerdict {
    let cohort: Vec<&WalkRecord> = walks
        .iter()
        .filter(|walk| walk.resource >= rung)
        .collect();
    if cohort.len() < 2 {
        return EnsembleVerdict::Keep;
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
    let self_family = walks.iter().find(|walk| walk.id == id).and_then(|walk| walk.family);
    if self_family.is_some_and(|family| {
        best_of_family
            .get(&family)
            .is_some_and(|(best_id, _)| *best_id == id)
    }) {
        return EnsembleVerdict::Keep;
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
    if extras.iter().skip(keep).any(|walk| walk.id == id) {
        EnsembleVerdict::Reseed
    } else {
        EnsembleVerdict::Keep
    }
}

/// Whether `id` is discarded at its current rung.
pub fn prune(walks: &[WalkRecord], id: u32, max_resource: u64) -> bool {
    verdict(walks, id, max_resource) == EnsembleVerdict::Reseed
}

/// Identities that [`verdict`] keeps, for a Feynman--Kac parent filter.
pub fn keep_ids(walks: &[WalkRecord], max_resource: u64) -> BTreeSet<u32> {
    walks
        .iter()
        .filter(|walk| verdict(walks, walk.id, max_resource) == EnsembleVerdict::Keep)
        .map(|walk| walk.id)
        .collect()
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
    fn unassigned_walks_are_not_ranked() {
        let walks = [
            walk(0, 10, -170.0, None),
            walk(1, 10, -169.0, None),
        ];
        assert!(!prune(&walks, 0, 1000));
        assert!(!prune(&walks, 1, 1000));
    }

    #[test]
    fn extras_below_the_first_rung_are_not_pruned() {
        let walks = [
            walk(0, 10, -396.282249, Some(0)),
            walk(1, 10, -395.0, Some(0)),
            walk(2, 10, -394.0, Some(0)),
            walk(3, 10, -393.0, Some(0)),
        ];
        assert!(!prune(&walks, 0, 1000));
        assert!(!prune(&walks, 3, 1000));
    }

    #[test]
    fn extras_of_a_crowded_family_reseed_at_the_first_rung() {
        // Talking extras leave the occupied packing at the first
        // identified rung. They do not wait for a serial first-hit hop.
        let mut walks = Vec::new();
        for index in 0..24 {
            walks.push(walk(
                index,
                MIN_RESOURCE,
                -396.282249 + f64::from(index) * 1e-3,
                Some(0),
            ));
        }
        assert!(!prune(&walks, 0, DEFAULT_MAX_RESOURCE), "champion stays");
        assert!(
            prune(&walks, 23, DEFAULT_MAX_RESOURCE),
            "surplus extra leaves at the first rung"
        );
    }

    #[test]
    fn a_kept_ico_extra_may_still_isomer_adopt() {
        let mut walks = vec![walk(0, 64, -396.282249, Some(0))];
        for index in 1..10 {
            walks.push(walk(
                index,
                64,
                -396.28 + f64::from(index) * 1e-3,
                Some(0),
            ));
        }
        assert!(!prune(&walks, 0, 1000), "ico champion stays");
        assert!(
            !prune(&walks, 1, 1000),
            "best extra stays and may adopt a deeper ico isomer"
        );
        assert!(prune(&walks, 9, 1000), "surplus extra reseeds");
    }

    #[test]
    fn keep_ids_retains_oh_and_best_ico_not_the_worst_ico() {
        let mut walks = vec![
            walk(0, 64, -173.928427, Some(1)),
            walk(1, 64, -173.252378, Some(0)),
        ];
        for index in 2..13 {
            walks.push(walk(index, 64, -173.25 + f64::from(index) * 1e-4, Some(0)));
        }
        let kept = keep_ids(&walks, 1000);
        assert!(kept.contains(&0));
        assert!(kept.contains(&1));
        assert!(!kept.contains(&12));
    }
}
