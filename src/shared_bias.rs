//! Visit deltas exchanged between walkers that share one per-basin bias.
//!
//! Multiple-walker metadynamics (Raiteri, Gervasio, Micheletti, Parrinello,
//! J. Phys. Chem. B 110, 3533, 2006) has every walker deposit into one bias.
//! Here each chain keeps its own bias and, at every checkpoint, publishes the
//! visits it made since its last publication as descriptor centres with
//! counts, and takes every other chain's since its last read. The bias each
//! chain sees is the population's up to one checkpoint of lag. Nothing else
//! crosses: no coordinates, no energies, no instruction to move.

use ndarray::Array1;

/// One published batch of visits.
struct Batch {
    walker: usize,
    deposits: Vec<(Array1<f64>, u64)>,
}

/// Append-only exchange of visit deltas, pruned once every walker has read.
pub struct SharedDeposits {
    walkers: usize,
    /// Batches not yet read by every walker; index 0 is the oldest.
    batches: Vec<Batch>,
    /// Sequence number of the oldest retained batch.
    base: u64,
    /// Next sequence number each walker has yet to read.
    cursors: Vec<u64>,
    published: u64,
    delivered: u64,
}

impl SharedDeposits {
    /// Empty exchange for `walkers` chains.
    pub fn new(walkers: usize) -> Self {
        assert!(walkers > 0, "a shared bias needs at least one walker");
        Self {
            walkers,
            batches: Vec::new(),
            base: 0,
            cursors: vec![0; walkers],
            published: 0,
            delivered: 0,
        }
    }

    /// Walkers in the exchange.
    pub fn walkers(&self) -> usize {
        self.walkers
    }

    /// Visits published and visits delivered to other walkers, in deposits.
    pub fn counts(&self) -> (u64, u64) {
        (self.published, self.delivered)
    }

    /// Batches retained for readers that have not caught up.
    pub fn retained(&self) -> usize {
        self.batches.len()
    }

    /// Publish `walker`'s visits since its last publication.
    ///
    /// An empty batch is dropped. The publishing walker never reads its own
    /// batch back; its cursor skips it.
    pub fn publish(&mut self, walker: usize, deposits: Vec<(Array1<f64>, u64)>) {
        assert!(walker < self.walkers, "walker index out of range");
        if deposits.is_empty() {
            return;
        }
        self.published += deposits.iter().map(|(_, n)| *n).sum::<u64>();
        self.batches.push(Batch { walker, deposits });
        self.prune();
    }

    /// Every other walker's visits published since `walker` last read.
    pub fn drain(&mut self, walker: usize) -> Vec<(Array1<f64>, u64)> {
        assert!(walker < self.walkers, "walker index out of range");
        let next = self.base + self.batches.len() as u64;
        let first = usize::try_from(self.cursors[walker].saturating_sub(self.base))
            .expect("cursor offset fits");
        let mut out = Vec::new();
        for batch in &self.batches[first..] {
            if batch.walker != walker {
                for (centre, count) in &batch.deposits {
                    self.delivered += *count;
                    out.push((centre.clone(), *count));
                }
            }
        }
        self.cursors[walker] = next;
        self.prune();
        out
    }

    /// Drop the batches every walker has read.
    fn prune(&mut self) {
        let oldest = self.cursors.iter().copied().min().unwrap_or(self.base);
        let drop = usize::try_from(oldest.saturating_sub(self.base)).expect("prune offset fits");
        let drop = drop.min(self.batches.len());
        if drop > 0 {
            self.batches.drain(..drop);
            self.base += drop as u64;
        }
    }
}

/// Visits each basin received since the previous snapshot, as
/// `(centre, delta)` pairs, with the snapshot advanced.
///
/// `visits` is the live per-basin count, `seen` the count at the last
/// publication; basins beyond `seen.len()` are new since then.
pub fn visit_deltas<'a>(
    centres: impl Fn(usize) -> ndarray::ArrayView1<'a, f64>,
    visits: impl Fn(usize) -> u64,
    n_basins: usize,
    seen: &mut Vec<u64>,
) -> Vec<(Array1<f64>, u64)> {
    let mut out = Vec::new();
    for i in 0..n_basins {
        let now = visits(i);
        let before = seen.get(i).copied().unwrap_or(0);
        if now > before {
            out.push((centres(i).to_owned(), now - before));
        }
    }
    seen.resize(n_basins, 0);
    for (i, slot) in seen.iter_mut().enumerate() {
        *slot = visits(i);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn each_walker_reads_the_others_once_and_batches_are_pruned() {
        let mut x = SharedDeposits::new(3);
        x.publish(0, vec![(array![1.0], 5)]);
        x.publish(1, vec![(array![2.0], 2)]);
        assert_eq!(x.retained(), 2);
        let got = x.drain(0);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].1, 2, "walker 0 must not read its own batch");
        assert!(x.drain(0).is_empty(), "a second read returns nothing new");
        let got = x.drain(2);
        assert_eq!(got.iter().map(|(_, n)| *n).sum::<u64>(), 7);
        assert_eq!(x.retained(), 2, "walker 1 has not read yet");
        let got = x.drain(1);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, array![1.0]);
        assert_eq!(x.retained(), 0, "every walker has read; nothing retained");
        // Delivered: walker 0 read 2, walker 2 read 7, walker 1 read 5.
        assert_eq!(x.counts(), (7, 14));
        x.publish(2, Vec::new());
        assert_eq!(x.retained(), 0, "an empty batch is dropped");
    }

    #[test]
    fn deltas_report_only_new_visits_and_advance_the_snapshot() {
        let centres = [array![0.0, 1.0], array![1.0, 0.0], array![1.0, 1.0]];
        let visits = [4_u64, 1, 3];
        let mut seen = vec![4, 0];
        let out = visit_deltas(|i| centres[i].view(), |i| visits[i], 3, &mut seen);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], (array![1.0, 0.0], 1));
        assert_eq!(out[1], (array![1.0, 1.0], 3));
        assert_eq!(seen, vec![4, 1, 3]);
        let again = visit_deltas(|i| centres[i].view(), |i| visits[i], 3, &mut seen);
        assert!(again.is_empty());
    }
}
