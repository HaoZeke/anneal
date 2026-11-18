// MIT License
// Copyright 2023--present rgpot developers

//! One cache for every fused potential evaluation.
//!
//! A fused evaluation returns the energy and the gradient of one geometry
//! from a single engine call. Callers ask for the two halves separately: an
//! energy-only request, a gradient request, a value-and-gradient pair, the
//! centre of a Hessian-vector product. [`FusedCache`] makes those requests
//! cost one engine call per distinct [`FusedKey`], whether the engine
//! succeeded or failed.
//!
//! - The key is the position bits (`f64::to_bits`, never a tolerance) and a
//!   [`FusedFingerprint`] of everything else that changes the answer: the
//!   atomic numbers, the cell and periodicity, the charge, the spin, the
//!   accuracy, the electronic temperature, the SCC iteration budget and
//!   mixer, and the engine build.
//! - The value is the outcome of the call: the evaluation, or a typed
//!   [`FusedError`] that tells an SCC that did not converge from any other
//!   failure.
//! - A request with the stored key returns the stored outcome with no engine
//!   call. A request with another key runs the engine and replaces the entry.
//! - A retry reruns the engine only through a changed setting, which is a
//!   changed key. A plain retry of a failed geometry returns the stored
//!   failure, so a warm-start retry of a failed energy needs a setting that
//!   differs from the one that failed.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::status::rgpot_status_t;

/// Energy, variance and forces (`3 * n_atoms`, row-major) of one fused
/// evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedValue {
    pub energy: f64,
    pub variance: f64,
    pub forces: Vec<f64>,
}

/// Which kind of failure a fused evaluation ended in.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum rgpot_fused_error_kind_t {
    /// The stored outcome is a success, or nothing is stored.
    RGPOT_FUSED_OK = 0,
    /// The engine's self-consistent field did not converge.
    RGPOT_FUSED_SCC_NOT_CONVERGED = 1,
    /// The engine reported another failure.
    RGPOT_FUSED_ENGINE_FAILED = 2,
    /// The engine reported success with a malformed or non-finite result.
    RGPOT_FUSED_INVALID_RESULT = 3,
}

/// Why a fused evaluation failed. Cloneable, so the same value is returned
/// to every request that finds it stored.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FusedError {
    /// The engine's self-consistent field did not converge.
    SccNotConverged { message: String },
    /// The engine returned this status.
    Engine {
        status: rgpot_status_t,
        message: String,
    },
    /// The engine reported success with a malformed or non-finite result.
    Invalid { message: String },
}

impl FusedError {
    /// Classify a status the engine returned on failure.
    pub fn from_status(status: rgpot_status_t, message: impl Into<String>) -> Self {
        let message = message.into();
        match status {
            rgpot_status_t::RGPOT_SCC_NOT_CONVERGED => Self::SccNotConverged { message },
            status => Self::Engine { status, message },
        }
    }

    /// The failure kind.
    pub fn kind(&self) -> rgpot_fused_error_kind_t {
        match self {
            Self::SccNotConverged { .. } => rgpot_fused_error_kind_t::RGPOT_FUSED_SCC_NOT_CONVERGED,
            Self::Engine { .. } => rgpot_fused_error_kind_t::RGPOT_FUSED_ENGINE_FAILED,
            Self::Invalid { .. } => rgpot_fused_error_kind_t::RGPOT_FUSED_INVALID_RESULT,
        }
    }

    /// The status a C caller of the engine-level API receives.
    pub fn status(&self) -> rgpot_status_t {
        match self {
            Self::SccNotConverged { .. } => rgpot_status_t::RGPOT_SCC_NOT_CONVERGED,
            Self::Engine { status, .. } => *status,
            Self::Invalid { .. } => rgpot_status_t::RGPOT_INVALID_PARAMETER,
        }
    }

    /// The message the first failure carried.
    pub fn message(&self) -> &str {
        match self {
            Self::SccNotConverged { message }
            | Self::Engine { message, .. }
            | Self::Invalid { message } => message,
        }
    }
}

impl std::fmt::Display for FusedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for FusedError {}

/// Settings of the engine that change its answer at a fixed geometry.
///
/// Two requests at the same positions share an outcome only when every field
/// here, and the species and cell, are equal. Floating-point fields are
/// compared by bit pattern.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct FusedSettings {
    /// Periodicity along each cell vector.
    pub periodic: [bool; 3],
    /// Total charge, in elementary charges.
    pub charge: f64,
    /// Number of unpaired electrons.
    pub spin: i32,
    /// Numerical accuracy level of the engine.
    pub accuracy: f64,
    /// Electronic temperature, in kelvin.
    pub electronic_temperature: f64,
    /// SCC iteration budget.
    pub scc_max_iterations: u32,
    /// Name of the SCC mixer.
    pub scc_mixer: String,
    /// Build or version string of the engine.
    pub engine_build: String,
}

impl FusedSettings {
    #[must_use]
    pub fn with_periodic(mut self, periodic: [bool; 3]) -> Self {
        self.periodic = periodic;
        self
    }

    #[must_use]
    pub fn with_charge(mut self, charge: f64) -> Self {
        self.charge = charge;
        self
    }

    #[must_use]
    pub fn with_spin(mut self, spin: i32) -> Self {
        self.spin = spin;
        self
    }

    #[must_use]
    pub fn with_accuracy(mut self, accuracy: f64) -> Self {
        self.accuracy = accuracy;
        self
    }

    #[must_use]
    pub fn with_electronic_temperature(mut self, kelvin: f64) -> Self {
        self.electronic_temperature = kelvin;
        self
    }

    #[must_use]
    pub fn with_scc_max_iterations(mut self, iterations: u32) -> Self {
        self.scc_max_iterations = iterations;
        self
    }

    #[must_use]
    pub fn with_scc_mixer(mut self, mixer: impl Into<String>) -> Self {
        self.scc_mixer = mixer.into();
        self
    }

    #[must_use]
    pub fn with_engine_build(mut self, build: impl Into<String>) -> Self {
        self.engine_build = build.into();
        self
    }

    /// The fingerprint of these settings for a geometry with the given
    /// species and cell. `epoch` is the cache's invalidation counter.
    pub fn fingerprint(
        &self,
        atomic_numbers: Option<&[i32]>,
        cell: Option<&[f64]>,
        epoch: u64,
    ) -> FusedFingerprint {
        FusedFingerprint {
            atomic_numbers: atomic_numbers.map(<[i32]>::to_vec),
            cell: cell.map(bit_pattern),
            periodic: self.periodic,
            charge: self.charge.to_bits(),
            spin: self.spin,
            accuracy: self.accuracy.to_bits(),
            electronic_temperature: self.electronic_temperature.to_bits(),
            scc_max_iterations: self.scc_max_iterations,
            scc_mixer: self.scc_mixer.clone(),
            engine_build: self.engine_build.clone(),
            epoch,
        }
    }
}

fn bit_pattern(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Everything but the positions that decides the answer of an engine call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusedFingerprint {
    atomic_numbers: Option<Vec<i32>>,
    cell: Option<Vec<u64>>,
    periodic: [bool; 3],
    charge: u64,
    spin: i32,
    accuracy: u64,
    electronic_temperature: u64,
    scc_max_iterations: u32,
    scc_mixer: String,
    engine_build: String,
    epoch: u64,
}

/// Positions by bit pattern, plus the fingerprint. Equality is exact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusedKey {
    positions: Vec<u64>,
    fingerprint: FusedFingerprint,
}

impl FusedKey {
    pub fn new(positions: &[f64], fingerprint: FusedFingerprint) -> Self {
        Self {
            positions: bit_pattern(positions),
            fingerprint,
        }
    }
}

/// Whether an answer came from the engine or from the stored entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedSource {
    Computed,
    Stored,
}

struct Entry<V> {
    key: FusedKey,
    outcome: Result<V, FusedError>,
}

/// Single-entry cache of the outcome of a fused evaluation.
pub struct FusedCache<V> {
    entry: Mutex<Option<Entry<V>>>,
    settings: Mutex<FusedSettings>,
    epoch: AtomicU64,
    computed: AtomicU64,
    served: AtomicU64,
}

impl<V> Default for FusedCache<V> {
    fn default() -> Self {
        Self {
            entry: Mutex::new(None),
            settings: Mutex::new(FusedSettings::default()),
            epoch: AtomicU64::new(0),
            computed: AtomicU64::new(0),
            served: AtomicU64::new(0),
        }
    }
}

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl<V: Clone> FusedCache<V> {
    pub fn new() -> Self {
        Self::default()
    }

    /// The engine settings requests are keyed on.
    pub fn settings(&self) -> FusedSettings {
        lock(&self.settings).clone()
    }

    /// Replace the engine settings. The stored entry stays; a request made
    /// under the new settings has another key and runs the engine.
    pub fn set_settings(&self, settings: FusedSettings) {
        *lock(&self.settings) = settings;
    }

    /// The key of a request at `positions` under the current settings.
    pub fn key(
        &self,
        positions: &[f64],
        atomic_numbers: Option<&[i32]>,
        cell: Option<&[f64]>,
    ) -> FusedKey {
        let epoch = self.epoch.load(Ordering::Acquire);
        let fingerprint = lock(&self.settings).fingerprint(atomic_numbers, cell, epoch);
        FusedKey::new(positions, fingerprint)
    }

    /// Answer a request: the stored outcome when `key` matches the entry,
    /// otherwise the outcome of `engine`, which replaces the entry.
    ///
    /// The entry stays locked while `engine` runs, so concurrent requests
    /// for one key run the engine once.
    pub fn evaluate<F>(&self, key: FusedKey, engine: F) -> (Result<V, FusedError>, FusedSource)
    where
        F: FnOnce() -> Result<V, FusedError>,
    {
        let mut entry = lock(&self.entry);
        if let Some(stored) = entry.as_ref() {
            if stored.key == key {
                self.served.fetch_add(1, Ordering::Relaxed);
                return (stored.outcome.clone(), FusedSource::Stored);
            }
        }
        self.computed.fetch_add(1, Ordering::Relaxed);
        let outcome = engine();
        *entry = Some(Entry {
            key,
            outcome: outcome.clone(),
        });
        (outcome, FusedSource::Computed)
    }

    /// Declare that the potential changed in place. The next request has
    /// another key, so it runs the engine even at the stored positions.
    pub fn invalidate(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        *lock(&self.entry) = None;
    }

    /// The failure stored for the last evaluated key, if it failed.
    pub fn stored_error(&self) -> Option<FusedError> {
        lock(&self.entry)
            .as_ref()
            .and_then(|stored| stored.outcome.as_ref().err().cloned())
    }

    /// Engine calls made through the cache.
    pub fn computed(&self) -> u64 {
        self.computed.load(Ordering::Relaxed)
    }

    /// Requests answered from the stored entry.
    pub fn served(&self) -> u64 {
        self.served.load(Ordering::Relaxed)
    }

    pub fn reset_counts(&self) {
        self.computed.store(0, Ordering::Relaxed);
        self.served.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn positions() -> [f64; 6] {
        [0.3, -1.7, 2.9, 4.1, -0.05, 7.3]
    }

    fn key(cache: &FusedCache<f64>, x: &[f64]) -> FusedKey {
        cache.key(
            x,
            Some(&[1, 1]),
            Some(&[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]),
        )
    }

    #[test]
    fn the_same_key_is_served_for_success_and_failure() {
        let cache = FusedCache::<f64>::new();
        let calls = Cell::new(0u32);
        let x = positions();
        let ok = || {
            calls.set(calls.get() + 1);
            Ok(1.5)
        };
        assert_eq!(cache.evaluate(key(&cache, &x), ok).1, FusedSource::Computed);
        assert_eq!(cache.evaluate(key(&cache, &x), ok).1, FusedSource::Stored);
        assert_eq!(calls.get(), 1);

        let fail = || {
            calls.set(calls.get() + 1);
            Err(FusedError::from_status(
                rgpot_status_t::RGPOT_SCC_NOT_CONVERGED,
                "no SCC",
            ))
        };
        let moved = [0.3, -1.7, 2.9, 4.1, -0.05, 7.4];
        let (first, _) = cache.evaluate(key(&cache, &moved), fail);
        let (second, source) = cache.evaluate(key(&cache, &moved), fail);
        assert_eq!(source, FusedSource::Stored);
        assert_eq!(first, second);
        assert_eq!(calls.get(), 2);
        assert_eq!(cache.computed(), 2);
        assert_eq!(cache.served(), 2);
    }

    #[test]
    fn an_invalidated_entry_is_not_served() {
        let cache = FusedCache::<f64>::new();
        let x = positions();
        let _ = cache.evaluate(key(&cache, &x), || Ok(1.0));
        cache.invalidate();
        let (_, source) = cache.evaluate(key(&cache, &x), || Ok(1.0));
        assert_eq!(source, FusedSource::Computed);
    }

    #[test]
    fn a_changed_setting_is_a_new_key() {
        let cache = FusedCache::<f64>::new();
        let x = positions();
        let _ = cache.evaluate(key(&cache, &x), || Ok(1.0));
        cache.set_settings(cache.settings().with_scc_max_iterations(7));
        let (_, source) = cache.evaluate(key(&cache, &x), || Ok(1.0));
        assert_eq!(source, FusedSource::Computed);
    }

    #[test]
    fn every_part_of_the_fingerprint_is_a_new_key() {
        const CELL: [f64; 9] = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let x = positions();
        let base = FusedSettings::default();
        let reference = base.fingerprint(Some(&[1, 1]), Some(&CELL), 0);
        let mut other_cell = CELL;
        other_cell[4] = f64::from_bits(other_cell[4].to_bits() + 1);
        let variants = [
            (
                "atomic numbers",
                base.fingerprint(Some(&[1, 2]), Some(&CELL), 0),
            ),
            ("no atomic numbers", base.fingerprint(None, Some(&CELL), 0)),
            (
                "cell by one ulp",
                base.fingerprint(Some(&[1, 1]), Some(&other_cell), 0),
            ),
            ("no cell", base.fingerprint(Some(&[1, 1]), None, 0)),
            (
                "periodic x",
                base.clone()
                    .with_periodic([true, false, false])
                    .fingerprint(Some(&[1, 1]), Some(&CELL), 0),
            ),
            (
                "periodic y",
                base.clone()
                    .with_periodic([false, true, false])
                    .fingerprint(Some(&[1, 1]), Some(&CELL), 0),
            ),
            (
                "periodic z",
                base.clone()
                    .with_periodic([false, false, true])
                    .fingerprint(Some(&[1, 1]), Some(&CELL), 0),
            ),
            (
                "charge",
                base.clone()
                    .with_charge(1.0)
                    .fingerprint(Some(&[1, 1]), Some(&CELL), 0),
            ),
            (
                "negative zero charge",
                base.clone()
                    .with_charge(-0.0)
                    .fingerprint(Some(&[1, 1]), Some(&CELL), 0),
            ),
            (
                "spin",
                base.clone()
                    .with_spin(1)
                    .fingerprint(Some(&[1, 1]), Some(&CELL), 0),
            ),
            (
                "accuracy",
                base.clone()
                    .with_accuracy(2.0)
                    .fingerprint(Some(&[1, 1]), Some(&CELL), 0),
            ),
            (
                "temperature",
                base.clone().with_electronic_temperature(300.0).fingerprint(
                    Some(&[1, 1]),
                    Some(&CELL),
                    0,
                ),
            ),
            (
                "iterations",
                base.clone().with_scc_max_iterations(250).fingerprint(
                    Some(&[1, 1]),
                    Some(&CELL),
                    0,
                ),
            ),
            (
                "mixer",
                base.clone()
                    .with_scc_mixer("simple")
                    .fingerprint(Some(&[1, 1]), Some(&CELL), 0),
            ),
            (
                "engine build",
                base.clone()
                    .with_engine_build("6.7.1")
                    .fingerprint(Some(&[1, 1]), Some(&CELL), 0),
            ),
            ("epoch", base.fingerprint(Some(&[1, 1]), Some(&CELL), 1)),
        ];
        let cache = FusedCache::<f64>::new();
        let calls = Cell::new(0u32);
        let engine = || {
            calls.set(calls.get() + 1);
            Ok(0.0)
        };
        let _ = cache.evaluate(FusedKey::new(&x, reference.clone()), engine);
        for (name, fingerprint) in variants {
            let before = calls.get();
            let _ = cache.evaluate(FusedKey::new(&x, fingerprint), engine);
            assert_eq!(calls.get(), before + 1, "{name}");
            let _ = cache.evaluate(FusedKey::new(&x, reference.clone()), engine);
            assert_eq!(calls.get(), before + 2, "{name}: back to the reference");
        }
    }

    #[test]
    fn kinds_and_statuses_are_distinct() {
        let scc = FusedError::from_status(rgpot_status_t::RGPOT_SCC_NOT_CONVERGED, "a");
        let other = FusedError::from_status(rgpot_status_t::RGPOT_INTERNAL_ERROR, "a");
        assert_ne!(scc, other);
        assert_ne!(scc.kind(), other.kind());
        assert_eq!(scc.status(), rgpot_status_t::RGPOT_SCC_NOT_CONVERGED);
        assert_eq!(other.status(), rgpot_status_t::RGPOT_INTERNAL_ERROR);
    }
}
