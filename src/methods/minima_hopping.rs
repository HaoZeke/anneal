//! History-conditioned escape feedback, after Goedecker's minima hopping.
//!
//! Goedecker, J. Chem. Phys. 120, 9911 (2004).
//!
//! The campaign's own measurements say what is missing. From the structure a
//! 75-point search settles into, none of 1800 single moves across the whole
//! kernel set reaches anything lower, so crossing needs a sequence of accepted
//! uphill quenches and a reason to accept them that does not destroy descent.
//! A continuous collective variable cannot supply that reason: the Q4 gap
//! between the two funnels is 0.023, under any usable deposition width.
//!
//! This crate already keeps a history of visited basins and uses it to deposit
//! bias. Minima hopping uses a history for something else: to scale the *next
//! escape*. Revisiting a known minimum makes the next attempt more violent
//! rather than the current one less attractive.
//!
//! That difference is the whole mechanism and it is worth stating precisely,
//! because the two are complementary and must stay separable:
//!
//! | | escape feedback | basin bias |
//! |---|---|---|
//! | what a revisit raises | the escape scale | the potential on that basin |
//! | acceptance | adaptive threshold on the energy rise | Metropolis on `F + V` |
//! | transition regions | left crossable | filled if revisited |
//!
//! Goedecker argues against flooding transition regions for exactly this
//! reason: the region between funnels has to stay crossable, so the response to
//! a revisit should be a harder push rather than a higher potential where the
//! chain needs to pass.
//!
//! The escape scale grows geometrically while a chain revisits, which is the
//! guarantee that no funnel is permanent: a chain that keeps returning keeps
//! escalating until it leaves. Schoenborn, Goedecker, Roy and Oganov,
//! J. Chem. Phys. 130, 144108 (2009), add feedback proportional to the visit
//! count and report that this finds the LJ75 Marks decahedron where a
//! cut-and-splice evolutionary algorithm does not.

use std::collections::HashMap;

/// What a quench was, relative to the history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visit {
    /// The basin the chain was already in.
    Same,
    /// A basin the history has seen before, but not the current one.
    Known,
    /// A basin not in the history.
    New,
}

/// Escape scale and acceptance threshold, both driven by the history.
///
/// Defaults are Goedecker's: the escape scale grows by 1.05 on a revisit and
/// shrinks by the same factor on a discovery, and the acceptance threshold
/// moves the other way, so roughly half of proposals are accepted without a
/// temperature being chosen.
#[derive(Debug, Clone)]
pub struct EscapeFeedback {
    /// Current escape scale, in whatever units the caller's move takes.
    escape: f64,
    /// Current acceptance threshold on the energy rise.
    threshold: f64,
    /// Growth on returning to the current basin.
    pub beta_same: f64,
    /// Growth on reaching a basin already in the history.
    pub beta_known: f64,
    /// Shrink on reaching a new basin.
    pub beta_new: f64,
    /// Enhanced feedback: the known-basin growth is multiplied by
    /// `1 + visits_coeff * visits`, so a repeatedly seen basin is escaped
    /// harder than one seen once (Schoenborn et al.).
    pub visits_coeff: f64,
    /// Threshold multiplier on acceptance; below one.
    pub alpha_accept: f64,
    /// Threshold multiplier on rejection; above one.
    pub alpha_reject: f64,
    /// Ceiling on the escape scale.
    ///
    /// A streak of revisits grows the scale geometrically, and an unbounded
    /// scale eventually proposes a structure so scattered that its relaxation
    /// costs far more than an ordinary one. The ceiling keeps the charged cost
    /// of a proposal bounded without removing the escalation.
    pub escape_ceiling: f64,
    /// Floor on the escape scale, so a run of discoveries cannot drive it to
    /// zero and freeze the search.
    pub escape_floor: f64,
    visits: HashMap<usize, u32>,
    /// Counts of each outcome, for reporting.
    pub n_same: usize,
    /// Quenches that landed in a known other basin.
    pub n_known: usize,
    /// Quenches that opened a new basin.
    pub n_new: usize,
}

impl EscapeFeedback {
    /// Controller starting at `escape` and `threshold`, with Goedecker's rates.
    pub fn new(escape: f64, threshold: f64) -> Self {
        assert!(escape > 0.0, "the escape scale must be positive");
        assert!(threshold > 0.0, "the acceptance threshold must be positive");
        Self {
            escape,
            threshold,
            beta_same: 1.05,
            beta_known: 1.05,
            beta_new: 1.0 / 1.05,
            visits_coeff: 0.1,
            alpha_accept: 1.0 / 1.05,
            alpha_reject: 1.05,
            escape_ceiling: escape * 64.0,
            escape_floor: escape / 64.0,
            visits: HashMap::new(),
            n_same: 0,
            n_known: 0,
            n_new: 0,
        }
    }

    /// Current escape scale.
    pub fn escape(&self) -> f64 {
        self.escape
    }

    /// Current acceptance threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Basins in the history.
    pub fn known_basins(&self) -> usize {
        self.visits.len()
    }

    /// Times `basin` has been recorded.
    pub fn visits(&self, basin: usize) -> u32 {
        self.visits.get(&basin).copied().unwrap_or(0)
    }

    /// Classifies a quench without recording it.
    pub fn classify(&self, current: Option<usize>, reached: usize) -> Visit {
        if current == Some(reached) {
            Visit::Same
        } else if self.visits.contains_key(&reached) {
            Visit::Known
        } else {
            Visit::New
        }
    }

    /// Records a quench and updates the escape scale.
    ///
    /// Returns what the quench was. The scale rises on a revisit and falls on a
    /// discovery, which is the feedback: a chain that keeps returning escalates
    /// until it leaves, and one that keeps finding new structures settles down
    /// to explore them.
    pub fn observe(&mut self, current: Option<usize>, reached: usize) -> Visit {
        let visit = self.classify(current, reached);
        match visit {
            Visit::Same => {
                self.escape *= self.beta_same;
                self.n_same += 1;
            }
            Visit::Known => {
                let v = self.visits(reached) as f64;
                self.escape *= self.beta_known * (1.0 + self.visits_coeff * v);
                self.n_known += 1;
            }
            Visit::New => {
                self.escape *= self.beta_new;
                self.n_new += 1;
            }
        }
        self.escape = self.escape.clamp(self.escape_floor, self.escape_ceiling);
        *self.visits.entry(reached).or_insert(0) += 1;
        visit
    }

    /// Whether to accept a move of energy rise `delta`, updating the threshold.
    ///
    /// The rule is Goedecker's: accept when the rise is under the threshold,
    /// then move the threshold so the acceptance rate sits near a half. No
    /// temperature appears, which is the point: a Metropolis temperature cold
    /// enough to polish cannot cross and one hot enough to cross cannot polish,
    /// while this threshold adapts to whichever the chain is currently failing
    /// at.
    pub fn accept(&mut self, delta: f64) -> bool {
        let ok = delta < self.threshold;
        if ok {
            self.threshold *= self.alpha_accept;
        } else {
            self.threshold *= self.alpha_reject;
        }
        // Kept positive and finite; a threshold at zero rejects everything and
        // one that has run away accepts everything.
        self.threshold = self.threshold.clamp(1e-9, 1e6);
        ok
    }

    /// Fraction of quenches that were discoveries.
    ///
    /// A run whose fraction is near zero is revisiting and should be
    /// escalating; one near one is wandering and should be settling. Reported
    /// because it says whether the feedback is doing anything.
    pub fn discovery_rate(&self) -> f64 {
        let total = self.n_same + self.n_known + self.n_new;
        if total == 0 {
            return f64::NAN;
        }
        self.n_new as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revisiting_the_same_basin_escalates_the_escape() {
        let mut f = EscapeFeedback::new(1.0, 1.0);
        let before = f.escape();
        for _ in 0..10 {
            assert_eq!(f.observe(Some(3), 3), Visit::Same);
        }
        assert!(
            f.escape() > before * 1.5,
            "ten returns should escalate: {} from {before}",
            f.escape()
        );
    }

    #[test]
    fn discovery_settles_the_escape_down() {
        let mut f = EscapeFeedback::new(1.0, 1.0);
        for k in 0..10 {
            assert_eq!(f.observe(Some(999), k), Visit::New);
        }
        assert!(
            f.escape() < 1.0,
            "a run of discoveries should settle: {}",
            f.escape()
        );
    }

    /// The property the mechanism exists for: a chain that cannot leave keeps
    /// escalating, so no funnel is permanent.
    #[test]
    fn a_trapped_chain_escalates_without_bound_up_to_the_ceiling() {
        let mut f = EscapeFeedback::new(1.0, 1.0);
        for _ in 0..500 {
            f.observe(Some(0), 0);
        }
        assert!(
            (f.escape() - f.escape_ceiling).abs() < 1e-9,
            "a permanently trapped chain should reach the ceiling, got {}",
            f.escape()
        );
    }

    #[test]
    fn a_repeatedly_seen_basin_is_escaped_harder_than_a_new_one() {
        let mut a = EscapeFeedback::new(1.0, 1.0);
        // Basin 1 seen many times, then returned to from elsewhere.
        for _ in 0..20 {
            a.observe(Some(1), 1);
        }
        let before = a.escape();
        a.observe(Some(2), 1);
        let jump_known = a.escape() / before;

        let mut b = EscapeFeedback::new(1.0, 1.0);
        b.observe(Some(2), 1);
        let jump_first = b.escape() / 1.0;
        assert!(
            jump_known > jump_first,
            "enhanced feedback should push harder on a familiar basin: \
             {jump_known} against {jump_first}"
        );
    }

    #[test]
    fn the_threshold_settles_near_half_acceptance() {
        let mut f = EscapeFeedback::new(1.0, 1.0);
        // A stream of rises drawn from a fixed distribution; the threshold
        // should find the level that accepts about half of them.
        let rises: Vec<f64> = (0..4000)
            .map(|i| ((i * 7919 + 13) % 1000) as f64 / 500.0)
            .collect();
        let mut accepted = 0usize;
        for (k, d) in rises.iter().enumerate() {
            let ok = f.accept(*d);
            if k >= 2000 && ok {
                accepted += 1;
            }
        }
        let rate = accepted as f64 / 2000.0;
        assert!(
            (0.3..0.7).contains(&rate),
            "acceptance settled at {rate}, which is not near a half"
        );
    }

    #[test]
    fn the_escape_scale_stays_within_its_bounds() {
        let mut f = EscapeFeedback::new(1.0, 1.0);
        for _ in 0..1000 {
            f.observe(Some(0), 0);
        }
        assert!(f.escape() <= f.escape_ceiling + 1e-12);
        let mut g = EscapeFeedback::new(1.0, 1.0);
        for k in 0..1000 {
            g.observe(Some(usize::MAX), k);
        }
        assert!(g.escape() >= g.escape_floor - 1e-12);
    }

    #[test]
    fn classification_distinguishes_the_three_cases() {
        let mut f = EscapeFeedback::new(1.0, 1.0);
        assert_eq!(f.classify(Some(1), 1), Visit::Same);
        assert_eq!(f.classify(Some(1), 2), Visit::New);
        f.observe(Some(1), 2);
        assert_eq!(f.classify(Some(1), 2), Visit::Known);
        assert_eq!(f.classify(Some(2), 2), Visit::Same);
    }
}
