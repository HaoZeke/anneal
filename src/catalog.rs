//! Event catalogue keyed by local topology, after k-ART.
//!
//! An event is a landing produced from a local key. The first time a key is
//! seen it is unsaturated. Each later visit is allowed
//! `1 + floor(log2 n)` searches in total, so extras arrive when `n` crosses
//! the next power of two. That is the k-ART `log n` schedule without a
//! search-count knob.
//!
//! Recycle: a `(from, to)` pair already in the catalogue is not a new event.

use std::collections::HashMap;

/// One recorded exit from a local topology.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    /// Local key the search started from.
    pub from: u64,
    /// Local key of the same atom (or the bag) after the landing.
    pub to: u64,
    /// Quenched energy of the landing, when known.
    pub dest_energy: f64,
}

/// What the catalogue knows about one local key.
#[derive(Debug, Clone, Default)]
pub struct TopologyRecord {
    /// Times this key has been seen on a structure the search stood on.
    pub seen: u64,
    /// Fresh searches paid for this key.
    pub searches: u64,
    /// Distinct landings.
    pub events: Vec<Event>,
}

impl TopologyRecord {
    /// Searches this key is entitled to after `seen` visits.
    pub fn search_entitlement(seen: u64) -> u64 {
        if seen == 0 {
            return 0;
        }
        1 + log2_floor(seen)
    }

    /// Searches still owed.
    pub fn due(&self) -> u64 {
        Self::search_entitlement(self.seen).saturating_sub(self.searches)
    }

    /// Whether a residual search should still be paid.
    pub fn unsaturated(&self) -> bool {
        self.due() > 0
    }
}

fn log2_floor(n: u64) -> u64 {
    if n <= 1 {
        0
    } else {
        63 - n.leading_zeros() as u64
    }
}

/// Catalogue of local-topology events.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    rec: HashMap<u64, TopologyRecord>,
}

impl Catalog {
    /// Empty catalogue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `key` appears on the current structure.
    pub fn observe_visit(&mut self, key: u64) {
        self.rec.entry(key).or_default().seen += 1;
    }

    /// Record a visit for every key in a bag.
    pub fn observe_bag(&mut self, keys: &[u64]) {
        let mut seen = std::collections::HashSet::new();
        for &k in keys {
            if seen.insert(k) {
                self.observe_visit(k);
            }
        }
    }

    /// Borrow the record for `key`.
    pub fn get(&self, key: u64) -> Option<&TopologyRecord> {
        self.rec.get(&key)
    }

    /// Whether `key` still owes a search.
    pub fn unsaturated(&self, key: u64) -> bool {
        match self.rec.get(&key) {
            Some(r) => r.unsaturated(),
            None => true,
        }
    }

    /// A local key in `keys` that still owes a search, if any.
    pub fn unsaturated_in(&self, keys: &[u64]) -> Option<u64> {
        keys.iter().copied().find(|&k| self.unsaturated(k))
    }

    /// Pay one search from `from`. `landing` is `None` when the search failed
    /// to leave. Returns whether a *new* event was stored.
    pub fn record_search(&mut self, from: u64, landing: Option<Event>) -> bool {
        let rec = self.rec.entry(from).or_default();
        rec.searches += 1;
        let Some(ev) = landing else {
            return false;
        };
        if rec
            .events
            .iter()
            .any(|e| e.from == ev.from && e.to == ev.to)
        {
            return false;
        }
        rec.events.push(ev);
        true
    }

    /// Whether `(from, to)` is already a known event.
    pub fn known(&self, from: u64, to: u64) -> bool {
        self.rec.get(&from).is_some_and(|r| {
            r.events.iter().any(|e| e.from == from && e.to == to)
        })
    }

    /// Distinct events stored.
    pub fn event_count(&self) -> usize {
        self.rec.values().map(|r| r.events.len()).sum()
    }

    /// Distinct local keys visited.
    pub fn key_count(&self) -> usize {
        self.rec.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entitlement_is_one_plus_log2() {
        assert_eq!(TopologyRecord::search_entitlement(0), 0);
        assert_eq!(TopologyRecord::search_entitlement(1), 1);
        assert_eq!(TopologyRecord::search_entitlement(2), 2);
        assert_eq!(TopologyRecord::search_entitlement(3), 2);
        assert_eq!(TopologyRecord::search_entitlement(4), 3);
        assert_eq!(TopologyRecord::search_entitlement(8), 4);
    }

    #[test]
    fn first_visit_is_unsaturated_until_searched() {
        let mut c = Catalog::new();
        assert!(c.unsaturated(7));
        c.observe_visit(7);
        assert!(c.unsaturated(7));
        c.record_search(7, None);
        assert!(!c.unsaturated(7));
        c.observe_visit(7);
        assert!(c.unsaturated(7), "crossing n=2 owes one more search");
    }

    #[test]
    fn recycle_does_not_count_as_new() {
        let mut c = Catalog::new();
        c.observe_visit(1);
        let ev = Event {
            from: 1,
            to: 2,
            dest_energy: -1.0,
        };
        assert!(c.record_search(1, Some(ev.clone())));
        assert!(!c.record_search(1, Some(ev)));
        assert_eq!(c.event_count(), 1);
        assert!(c.known(1, 2));
    }
}
