//! Learning the funnel partition from the search's own transitions.
//!
//! Every descriptor tried in this crate was *chosen*: a bond-order parameter, a
//! sorted distance spectrum, per-point pair energies, template-matching
//! fractions, common-neighbour fractions. Each is a guess about what separates
//! the regions the search must tell apart, and the guesses have been poor. The
//! fourth-order bond-order parameter separates the two funnels at 75 points by
//! 0.023, under any usable deposition width; sorted distances need a threshold
//! that is a knife edge, 9 seeds in 16 at 0.7 and 0 in 18 at anything coarser.
//!
//! The search generates a better object for free. Its accepted hops are edges
//! between basins, and the graph they form has the funnel structure in it
//! whether or not any descriptor does. Cameron treats the 38-point landscape
//! this way, as a network of some seventy thousand minima, and reads the
//! metastability off the spectrum rather than off a coordinate.
//!
//! What that gives is a partition rather than a number: the sign of the second
//! eigenvector of the normalised Laplacian, the Fiedler vector, splits the
//! basins into the two sets between which transitions are rarest. The
//! corresponding eigenvalue says how well separated they are, so the method
//! reports its own confidence.
//!
//! # Why this is not another descriptor
//!
//! A descriptor is a function of one structure. This is a function of the
//! *search*: two structures land in the same part because the chain moves
//! between them, not because they look alike. A funnel is defined by what is
//! reachable, which is what makes it the right object, and it is exactly what a
//! chosen coordinate cannot see.
//!
//! # What it costs
//!
//! An eigendecomposition of a matrix the size of the basin count, so it is
//! refitted on a schedule rather than per hop. The basin count runs to a few
//! thousand in a long run, which is past what a dense solver should be asked
//! for every step and comfortable every few thousand.

use crate::spectral::{SpectralError, TransitionGraph, laplacian_embedding};
use ndarray::Array1;

/// A two-way split of the visited basins, with the confidence in it.
#[derive(Debug, Clone)]
pub struct Partition {
    /// Basin identifiers, in the order the parts refer to.
    pub basins: Vec<usize>,
    /// Which side of the split each basin fell on.
    pub side: Vec<bool>,
    /// Algebraic connectivity: the second eigenvalue of the normalised
    /// Laplacian.
    ///
    /// Near zero means the two parts are nearly disconnected, which is what a
    /// funnel boundary looks like. Large means the graph is well mixed and the
    /// split is arbitrary, and a caller should ignore it rather than act on it.
    pub connectivity: f64,
}

impl Partition {
    /// Which side a basin is on, if it was visited.
    pub fn side_of(&self, basin: usize) -> Option<bool> {
        self.basins
            .iter()
            .position(|b| *b == basin)
            .map(|i| self.side[i])
    }

    /// Basins on each side.
    pub fn sizes(&self) -> (usize, usize) {
        let a = self.side.iter().filter(|s| **s).count();
        (a, self.side.len() - a)
    }

    /// Whether the split is worth acting on.
    ///
    /// A partition of a well-mixed graph is a cut through the middle of one
    /// region, and steering by it would push the chain away from where it
    /// already is for no reason. The threshold is on the connectivity, not on
    /// the sizes: a small part is fine when it is genuinely separated.
    pub fn separated(&self, threshold: f64) -> bool {
        self.connectivity < threshold && self.sizes().0 > 0 && self.sizes().1 > 0
    }
}

/// Accumulates transitions and splits them when asked.
#[derive(Debug, Default)]
pub struct FunnelSpectrum {
    graph: TransitionGraph,
    /// Hops recorded since the last split.
    since_fit: usize,
    /// Splits computed.
    pub fits: usize,
    /// Splits refused because the graph was too small or disconnected.
    pub refusals: usize,
}

impl FunnelSpectrum {
    /// A fresh accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records an accepted hop between basins.
    ///
    /// Only accepted hops: a rejected proposal says the chain declined to move,
    /// which is a statement about the acceptance rule rather than about
    /// reachability, and including them makes every basin look adjacent to
    /// whatever the chain happened to try.
    pub fn record(&mut self, from: usize, to: usize) {
        self.graph.record(from, to, 1.0);
        self.since_fit += 1;
    }

    /// Basins seen.
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    /// Whether nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    /// Hops since the last split was computed.
    pub fn pending(&self) -> usize {
        self.since_fit
    }

    /// Splits the visited basins in two, or reports why it could not.
    pub fn split(&mut self) -> Result<Partition, SpectralError> {
        self.since_fit = 0;
        let (basins, weights) = self.graph.adjacency();
        let (values, vectors) = match laplacian_embedding(weights.view(), 2) {
            Ok(v) => v,
            Err(e) => {
                self.refusals += 1;
                return Err(e);
            }
        };
        self.fits += 1;
        // `laplacian_embedding` has already dropped the trivial eigenpair: it
        // returns `vals[k + 1]` and `vecs[:, k + 1]`, so its first column is
        // the Fiedler vector and its first value the algebraic connectivity.
        //
        // Taking the second value and the last column instead, as though the
        // trivial pair were still present, gets both wrong in ways that partly
        // cancel: the ordering came out inverted, with one crossing between two
        // cliques reporting 1.226 and forty crossings reporting 0.750.
        let fiedler: Array1<f64> = vectors.column(0).to_owned();
        let side: Vec<bool> = fiedler.iter().map(|v| *v >= 0.0).collect();
        let connectivity = values.first().copied().unwrap_or(f64::INFINITY);
        Ok(Partition {
            basins,
            side,
            connectivity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two cliques joined by one edge: the split has to find that edge, which
    /// is the whole claim. A chosen coordinate would have to be lucky; this
    /// reads it off the transitions.
    #[test]
    fn it_splits_two_weakly_joined_groups() {
        let mut f = FunnelSpectrum::new();
        // Group A: basins 0..4, densely connected.
        for i in 0..5 {
            for j in (i + 1)..5 {
                for _ in 0..10 {
                    f.record(i, j);
                }
            }
        }
        // Group B: basins 5..9, densely connected.
        for i in 5..10 {
            for j in (i + 1)..10 {
                for _ in 0..10 {
                    f.record(i, j);
                }
            }
        }
        // One rare crossing.
        f.record(4, 5);

        let p = f.split().expect("split refused");
        let a_side = p.side_of(0).unwrap();
        for b in 1..5 {
            assert_eq!(p.side_of(b), Some(a_side), "basin {b} left group A");
        }
        for b in 5..10 {
            assert_eq!(p.side_of(b), Some(!a_side), "basin {b} joined group A");
        }
        assert!(
            p.separated(0.5),
            "connectivity {} should read as separated",
            p.connectivity
        );
    }

    /// A well-mixed graph must not be reported as separated, or the caller
    /// steers by a cut through the middle of one region.
    #[test]
    fn a_well_mixed_graph_is_not_called_separated() {
        let mut f = FunnelSpectrum::new();
        for i in 0..8 {
            for j in (i + 1)..8 {
                for _ in 0..5 {
                    f.record(i, j);
                }
            }
        }
        let p = f.split().expect("split refused");
        assert!(
            !p.separated(0.2),
            "a complete graph reported connectivity {}",
            p.connectivity
        );
    }

    /// Connectivity has to order the two cases, which is what makes it usable
    /// as a confidence rather than a label.
    #[test]
    fn a_weaker_join_gives_a_smaller_connectivity() {
        let build = |crossings: usize| {
            let mut f = FunnelSpectrum::new();
            for i in 0..5 {
                for j in (i + 1)..5 {
                    for _ in 0..10 {
                        f.record(i, j);
                    }
                }
            }
            for i in 5..10 {
                for j in (i + 1)..10 {
                    for _ in 0..10 {
                        f.record(i, j);
                    }
                }
            }
            for _ in 0..crossings {
                f.record(4, 5);
            }
            f.split().expect("split refused").connectivity
        };
        let weak = build(1);
        let strong = build(40);
        assert!(
            weak < strong,
            "one crossing gave {weak}, forty gave {strong}"
        );
    }

    /// Self-transitions carry no connectivity information and must not create
    /// a graph out of a chain that never left.
    #[test]
    fn a_chain_that_never_left_yields_no_split() {
        let mut f = FunnelSpectrum::new();
        for _ in 0..100 {
            f.record(3, 3);
        }
        assert!(f.split().is_err(), "a single basin was split");
        assert_eq!(f.refusals, 1);
    }
}
