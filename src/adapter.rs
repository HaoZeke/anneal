//! `trait Adapter<S>`: Stan-style A2 windowed adaptation surface.
//!
//! Stan's `base_adapter.hpp` factors adaptation state (step size, mass
//! matrix, dual-averaging counters) out of the sampler. The adaptation
//! is run in a windowed phase: `adapt_diag_e_nuts.hpp` advances the
//! adapter on every transition during the warmup phase, then freezes
//! the adapted state for the sampling phase.
//!
//! We model the same separation: a `Sampler<T>` runs unmodified; an
//! `Adapter<S>` observes its (state, accepted) pairs each step and
//! mutates *itself* (not the sampler) accordingly. Concrete impls
//! (e.g. dual-averaging step-size adaptation, Tsallis q_v adaptation)
//! land in subsequent commits; this commit ships the surface plus a
//! no-op `IdentityAdapter` proving the trait composes with `run_rs`.
//!
//! See `~/Git/Gitlab/obsidian-notes/Software/anneal/design_pass_04_lit_survey.org`
//! task A2 for the design rationale.

use num_traits::Float;

use crate::history::State;
use crate::sampler::Sampler;

/// Windowed adaptation interface.
///
/// `observe` is called by an adaptive `run_rs` driver after every
/// `Sampler::step` during the warmup phase. The adapter inspects
/// `(state, accepted, epoch)` and updates its own internal counters.
/// `freeze()` is called once at the end of warmup; subsequent samples
/// use whatever the adapter has settled to without further mutation.
pub trait Adapter<T: Float, S: Sampler<T>>: Send + Sync {
    /// Observes one transition. Called during warmup only.
    fn observe(&mut self, sampler: &S, state: &State, accepted: bool, epoch: usize);

    /// Marks adaptation complete. Called at the warmup -> sampling boundary.
    fn freeze(&mut self) {}

    /// Returns true iff the adapter has settled. Optional -- default
    /// `false` (always observe until external freeze).
    fn is_frozen(&self) -> bool {
        false
    }
}

/// A no-op adapter: never updates anything. Used as the default in
/// `run_rs` (no warmup phase) and as the type-system witness that a
/// non-adaptive sampler is identical to one wrapped in `IdentityAdapter`.
pub struct IdentityAdapter;

impl<T: Float, S: Sampler<T>> Adapter<T, S> for IdentityAdapter {
    fn observe(&mut self, _sampler: &S, _state: &State, _accepted: bool, _epoch: usize) {}
}
