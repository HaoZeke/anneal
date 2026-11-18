// MIT License
// Copyright 2023--present rgpot developers

//! Native loader for the minimum potential ABI profile.
//!
//! A backend prefix such as `nwchemc` or `cpmdc` selects one conforming
//! shared library. The library receives `PotentialConfig` bytes once, keeps a
//! persistent session, and receives `ForceInput` bytes for every geometry.
//! Results return as `PotentialResult` bytes in the units requested by the
//! input message.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::fmt;
use std::mem::ManuallyDrop;
use std::path::Path;

use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use libloading::Library;

use crate::compat::Expectation;
use crate::Potentials_capnp::{force_input, potential_result};

/// One atomistic evaluation sent through a profile session.
#[derive(Debug, Clone, Copy)]
pub struct ProfileRequest<'a> {
    /// Flat Cartesian coordinates, three values per atom.
    pub positions: &'a [f64],
    /// Atomic numbers in the same atom order as `positions`.
    pub atomic_numbers: &'a [i32],
    /// Optional row-major 3x3 simulation cell; molecular inputs use `None`.
    pub box_matrix: Option<&'a [f64; 9]>,
    /// Unit expression for coordinates and cell vectors.
    pub length_unit: &'a str,
    /// Unit expression requested for energy and forces.
    pub energy_unit: &'a str,
}

/// Energy and forces returned by one fused backend calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileEvaluation {
    /// Energy in the `ProfileRequest::energy_unit` unit.
    pub energy: f64,
    /// Forces in energy-unit per length-unit.
    pub forces: Vec<f64>,
}

/// Error raised while loading or driving a potential profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileError(String);

impl ProfileError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ProfileError {}

impl From<capnp::Error> for ProfileError {
    fn from(error: capnp::Error) -> Self {
        Self::new(error.to_string())
    }
}

/// Result type for profile loading and evaluation.
pub type ProfileResult<T> = Result<T, ProfileError>;

/// Ordered shared-library candidates for a backend prefix.
pub fn library_candidates(prefix: &str, explicit_path: Option<&str>) -> Vec<String> {
    let mut candidates = Vec::new();
    if let Some(path) = explicit_path.filter(|path| !path.is_empty()) {
        candidates.push(path.to_owned());
    }

    let upper = prefix.to_ascii_uppercase();
    for variable in [format!("{upper}_LIBRARY"), format!("RGPOT_{upper}_ENGINE")] {
        if let Some(path) = std::env::var_os(variable).and_then(|value| value.into_string().ok()) {
            if !path.is_empty() && !candidates.contains(&path) {
                candidates.push(path);
            }
        }
    }

    for path in [
        format!("lib{prefix}.so"),
        format!("./lib{prefix}.so"),
        format!("lib{prefix}.dylib"),
        format!("./lib{prefix}.dylib"),
        format!("{prefix}.dll"),
    ] {
        if !candidates.contains(&path) {
            candidates.push(path);
        }
    }
    candidates
}

/// Serialize a request as the shared `ForceInput` carrier.
pub fn encode_force_input(request: &ProfileRequest<'_>) -> ProfileResult<Vec<u8>> {
    let expected_positions = request.atomic_numbers.len() * 3;
    if request.positions.len() != expected_positions {
        return Err(ProfileError::new(format!(
            "ForceInput has {} coordinates for {} atoms; expected {expected_positions}",
            request.positions.len(),
            request.atomic_numbers.len()
        )));
    }
    if request.length_unit.is_empty() || request.energy_unit.is_empty() {
        return Err(ProfileError::new("ForceInput units must be non-empty"));
    }

    let mut message = Builder::new_default();
    {
        let mut input = message.init_root::<force_input::Builder>();
        let mut positions = input
            .reborrow()
            .init_pos(request.positions.len().try_into().map_err(|_| {
                ProfileError::new("ForceInput coordinate count exceeds the schema limit")
            })?);
        for (index, value) in request.positions.iter().copied().enumerate() {
            positions.set(index as u32, value);
        }

        let mut atomic_numbers =
            input
                .reborrow()
                .init_atmnrs(request.atomic_numbers.len().try_into().map_err(|_| {
                    ProfileError::new("ForceInput atom count exceeds the schema limit")
                })?);
        for (index, value) in request.atomic_numbers.iter().copied().enumerate() {
            atomic_numbers.set(index as u32, value);
        }

        if let Some(box_matrix) = request.box_matrix {
            let mut encoded_box = input.reborrow().init_box(9);
            for (index, value) in box_matrix.iter().copied().enumerate() {
                encoded_box.set(index as u32, value);
            }
        }
        input.set_length_unit(request.length_unit);
        input.set_energy_unit(request.energy_unit);
    }
    Ok(serialize::write_message_to_words(&message))
}

/// Decode the energy and force fields from a shared `PotentialResult` carrier.
pub fn decode_potential_result(
    encoded: &[u8],
    expected_force_count: usize,
) -> ProfileResult<ProfileEvaluation> {
    let mut bytes = encoded;
    let message = serialize::read_message_from_flat_slice(&mut bytes, ReaderOptions::new())?;
    let result = message.get_root::<potential_result::Reader>()?;
    let forces = result.get_forces()?;
    if forces.len() as usize != expected_force_count {
        return Err(ProfileError::new(format!(
            "PotentialResult has {} forces; expected {expected_force_count}",
            forces.len()
        )));
    }
    Ok(ProfileEvaluation {
        energy: result.get_energy(),
        forces: (0..forces.len()).map(|index| forces.get(index)).collect(),
    })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ProfileAbiResult {
    ok: c_int,
    energy_h: f64,
    message: [c_char; 512],
}

type AbiVersionFn = unsafe extern "C" fn() -> c_int;
type VersionFn = unsafe extern "C" fn() -> *const c_char;
type LastErrorFn = unsafe extern "C" fn() -> *const c_char;
type AvailableFn = unsafe extern "C" fn() -> c_int;
type FinalizeFn = unsafe extern "C" fn();
type ConfigureFn = unsafe extern "C" fn(*const c_void, usize) -> c_int;
type SessionCreateFromConfigFn = unsafe extern "C" fn(*const c_void, usize) -> *mut c_void;
type SessionConfigureFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> c_int;
type SessionDestroyFn = unsafe extern "C" fn(*mut c_void);
type SessionCalculateResultFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    usize,
    *mut c_void,
    usize,
    *mut usize,
) -> ProfileAbiResult;
type CalculateResultFromConfigFn = unsafe extern "C" fn(
    *const c_void,
    usize,
    *const c_void,
    usize,
    *mut c_void,
    usize,
    *mut usize,
) -> ProfileAbiResult;
type PotentialResultSizeFn = unsafe extern "C" fn(*const c_void, usize) -> usize;
type CapabilitiesResultFn = unsafe extern "C" fn(*mut c_void, usize, *mut usize) -> c_int;

struct ProcessLifetime<T>(ManuallyDrop<T>);

impl<T> ProcessLifetime<T> {
    fn new(value: T) -> Self {
        Self(ManuallyDrop::new(value))
    }
}

struct RuntimeLifecycle {
    session: *mut c_void,
    destroy: SessionDestroyFn,
    finalize: FinalizeFn,
}

impl Drop for RuntimeLifecycle {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.session);
            self.session = std::ptr::null_mut();
            (self.finalize)();
        }
    }
}

/// One persistent session loaded through the prefix-generic potential profile.
pub struct ProfileSession {
    prefix: String,
    library_path: String,
    version: String,
    abi_version: i32,
    _library: ProcessLifetime<Library>,
    lifecycle: RuntimeLifecycle,
    last_error: LastErrorFn,
    calculate: SessionCalculateResultFn,
    result_size: PotentialResultSizeFn,
}

impl ProfileSession {
    /// Load a conforming backend and create one session from `PotentialConfig` bytes.
    ///
    /// # Safety
    ///
    /// The selected shared library must implement the minimum potential ABI
    /// profile for `prefix` with the C layouts defined by the profile contract.
    pub unsafe fn load(
        prefix: &str,
        explicit_path: Option<&Path>,
        config: &[u8],
    ) -> ProfileResult<Self> {
        unsafe { Self::load_with(prefix, explicit_path, config, &Expectation::default()) }
    }

    /// [`ProfileSession::load`] with an explicit host [`Expectation`]. The
    /// backend's `Capabilities` are checked before the session is created, so
    /// an incompatible backend is refused before any work is dispatched.
    ///
    /// # Safety
    ///
    /// Same contract as [`ProfileSession::load`].
    pub unsafe fn load_with(
        prefix: &str,
        explicit_path: Option<&Path>,
        config: &[u8],
        expectation: &Expectation,
    ) -> ProfileResult<Self> {
        if prefix.is_empty()
            || !prefix
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(ProfileError::new(format!(
                "invalid potential profile prefix: {prefix:?}"
            )));
        }
        if config.is_empty() {
            return Err(ProfileError::new("PotentialConfig message is empty"));
        }

        let explicit = explicit_path.and_then(Path::to_str);
        let (library, library_path) = open_first(prefix, explicit)?;
        let abi_version = unsafe { resolve::<AbiVersionFn>(&library, prefix, "abi_version")? };
        let version_fn = unsafe { resolve::<VersionFn>(&library, prefix, "version")? };
        let last_error = unsafe { resolve::<LastErrorFn>(&library, prefix, "last_error")? };
        let available = unsafe { resolve::<AvailableFn>(&library, prefix, "available")? };
        let finalize = unsafe { resolve::<FinalizeFn>(&library, prefix, "finalize")? };
        let _configure = unsafe { resolve::<ConfigureFn>(&library, prefix, "configure")? };
        let session_create = unsafe {
            resolve::<SessionCreateFromConfigFn>(&library, prefix, "session_create_from_config")?
        };
        let _session_configure =
            unsafe { resolve::<SessionConfigureFn>(&library, prefix, "session_configure")? };
        let session_destroy =
            unsafe { resolve::<SessionDestroyFn>(&library, prefix, "session_destroy")? };
        let calculate = unsafe {
            resolve::<SessionCalculateResultFn>(&library, prefix, "session_calculate_result")?
        };
        let _one_shot = unsafe {
            resolve::<CalculateResultFromConfigFn>(
                &library,
                prefix,
                "calculate_result_from_config",
            )?
        };
        let result_size = unsafe {
            resolve::<PotentialResultSizeFn>(
                &library,
                prefix,
                "potential_result_size_for_force_input",
            )?
        };
        let capabilities =
            unsafe { resolve::<CapabilitiesResultFn>(&library, prefix, "capabilities_result")? };
        let words = read_capabilities(prefix, capabilities)?;
        check_capabilities_bytes(prefix, &words, expectation)?;

        if unsafe { available() } == 0 {
            return Err(ProfileError::new(format!(
                "potential profile {prefix} is not available"
            )));
        }
        let version = c_string(unsafe { version_fn() });
        let session = unsafe { session_create(config.as_ptr().cast(), config.len()) };
        if session.is_null() {
            let message = c_string(unsafe { last_error() });
            unsafe { finalize() };
            return Err(ProfileError::new(format!(
                "{prefix}_session_create_from_config failed: {message}"
            )));
        }

        Ok(Self {
            prefix: prefix.to_owned(),
            library_path,
            version,
            abi_version: unsafe { abi_version() },
            _library: ProcessLifetime::new(library),
            lifecycle: RuntimeLifecycle {
                session,
                destroy: session_destroy,
                finalize,
            },
            last_error,
            calculate,
            result_size,
        })
    }

    /// Backend prefix used to resolve the profile symbols.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Path of the shared library selected by the candidate search.
    pub fn library_path(&self) -> &str {
        &self.library_path
    }

    /// Human-readable backend version string.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Numeric ABI generation reported by the backend.
    pub fn abi_version(&self) -> i32 {
        self.abi_version
    }

    /// Evaluate one geometry through the persistent session.
    pub fn evaluate(&mut self, request: &ProfileRequest<'_>) -> ProfileResult<ProfileEvaluation> {
        let force_input = encode_force_input(request)?;
        let required =
            unsafe { (self.result_size)(force_input.as_ptr().cast(), force_input.len()) };
        if required == 0 {
            return Err(ProfileError::new(format!(
                "{}_potential_result_size_for_force_input failed: {}",
                self.prefix,
                self.last_error_message()
            )));
        }

        let word_count = required.div_ceil(std::mem::size_of::<u64>());
        let mut output = vec![0u64; word_count];
        let capacity = output.len() * std::mem::size_of::<u64>();
        let mut written = 0usize;
        let status = unsafe {
            (self.calculate)(
                self.lifecycle.session,
                force_input.as_ptr().cast(),
                force_input.len(),
                output.as_mut_ptr().cast(),
                capacity,
                &mut written,
            )
        };
        if status.ok == 0 {
            let message = abi_message(&status).unwrap_or_else(|| self.last_error_message());
            return Err(ProfileError::new(format!(
                "{}_session_calculate_result failed: {message}",
                self.prefix
            )));
        }
        if written == 0 || written > capacity || !written.is_multiple_of(std::mem::size_of::<u64>())
        {
            return Err(ProfileError::new(format!(
                "{}_session_calculate_result wrote invalid size {written} for capacity {capacity}",
                self.prefix
            )));
        }
        let encoded = unsafe { std::slice::from_raw_parts(output.as_ptr().cast::<u8>(), written) };
        decode_potential_result(encoded, request.positions.len())
    }

    fn last_error_message(&self) -> String {
        c_string(unsafe { (self.last_error)() })
    }
}

/// Capabilities message bytes as aligned words (size query, then fill).
fn read_capabilities(prefix: &str, f: CapabilitiesResultFn) -> ProfileResult<Vec<u64>> {
    let mut required = 0usize;
    unsafe { f(std::ptr::null_mut(), 0, &mut required) };
    if required == 0 || !required.is_multiple_of(std::mem::size_of::<u64>()) {
        return Err(ProfileError::new(format!(
            "{prefix}_capabilities_result reported invalid size {required}"
        )));
    }
    let mut words = vec![0u64; required / std::mem::size_of::<u64>()];
    let mut written = 0usize;
    let status = unsafe { f(words.as_mut_ptr().cast(), required, &mut written) };
    if status != 0 || written != required {
        return Err(ProfileError::new(format!(
            "{prefix}_capabilities_result failed (status {status}, wrote {written} of {required})"
        )));
    }
    Ok(words)
}

/// Decode a `Capabilities` message and apply the handshake to it.
pub fn check_capabilities_bytes(
    prefix: &str,
    words: &[u64],
    expectation: &Expectation,
) -> ProfileResult<()> {
    crate::compat::check_capabilities_words(words, expectation)
        .map_err(|why| ProfileError::new(format!("{prefix} backend refused: {why}")))
}

fn open_first(prefix: &str, explicit_path: Option<&str>) -> ProfileResult<(Library, String)> {
    let candidates = library_candidates(prefix, explicit_path);
    let mut failures = Vec::new();
    for candidate in candidates {
        match unsafe { Library::new(&candidate) } {
            Ok(library) => return Ok((library, candidate)),
            Err(error) => failures.push(format!("{candidate}: {error}")),
        }
    }
    Err(ProfileError::new(format!(
        "lib{prefix} not loaded; {}",
        failures.join("; ")
    )))
}

unsafe fn resolve<T: Copy>(library: &Library, prefix: &str, suffix: &str) -> ProfileResult<T> {
    let name = format!("{prefix}_{suffix}\0");
    unsafe { library.get::<T>(name.as_bytes()) }
        .map(|symbol| *symbol)
        .map_err(|error| ProfileError::new(format!("missing {prefix}_{suffix}: {error}")))
}

fn c_string(pointer: *const c_char) -> String {
    if pointer.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

fn abi_message(result: &ProfileAbiResult) -> Option<String> {
    let bytes = result
        .message
        .iter()
        .map(|value| *value as u8)
        .take_while(|value| *value != 0)
        .collect::<Vec<_>>();
    if bytes.is_empty() {
        None
    } else {
        Some(String::from_utf8_lossy(&bytes).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    struct DropProbe(Arc<AtomicBool>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn process_lifetime_handle_keeps_library_mapping() {
        let dropped = Arc::new(AtomicBool::new(false));
        {
            let _handle = ProcessLifetime::new(DropProbe(Arc::clone(&dropped)));
        }

        assert!(!dropped.load(Ordering::SeqCst));
    }

    static LIFECYCLE_STEP: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn record_destroy(_session: *mut c_void) {
        assert_eq!(LIFECYCLE_STEP.swap(1, Ordering::SeqCst), 0);
    }

    unsafe extern "C" fn record_finalize() {
        assert_eq!(LIFECYCLE_STEP.swap(2, Ordering::SeqCst), 1);
    }

    #[test]
    fn session_is_destroyed_before_runtime_finalize() {
        LIFECYCLE_STEP.store(0, Ordering::SeqCst);
        let mut token = 0u8;
        let lifecycle = RuntimeLifecycle {
            session: (&mut token as *mut u8).cast(),
            destroy: record_destroy,
            finalize: record_finalize,
        };

        drop(lifecycle);

        assert_eq!(LIFECYCLE_STEP.load(Ordering::SeqCst), 2);
    }

    fn flat_words(
        edit: impl FnOnce(&mut crate::Potentials_capnp::capabilities::Builder<'_>),
    ) -> Vec<u64> {
        use crate::Potentials_capnp::capabilities::{Builder as Caps, Operation};
        let mut msg = Builder::new_default();
        {
            let mut caps = msg.init_root::<Caps<'_>>();
            crate::compat::fill_compatibility(caps.reborrow());
            let mut ops = caps.reborrow().init_operations(2);
            ops.set(0, Operation::Energy);
            ops.set(1, Operation::Forces);
            edit(&mut caps);
        }
        let bytes = serialize::write_message_to_words(&msg);
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| u64::from_le_bytes(*c))
            .collect()
    }

    #[test]
    fn dispatch_check_accepts_compatible_and_names_the_refused_field() {
        let want = Expectation::default();
        assert!(check_capabilities_bytes("fake", &flat_words(|_| {}), &want).is_ok());

        let refused =
            check_capabilities_bytes("fake", &flat_words(|c| c.set_dlpack_major(2)), &want)
                .unwrap_err()
                .to_string();
        assert!(refused.contains("fake backend refused"), "{refused}");
        assert!(refused.contains("DLPack major"), "{refused}");
    }

    #[test]
    fn dispatch_check_rejects_a_truncated_message() {
        let mut words = flat_words(|_| {});
        words.truncate(1);
        assert!(check_capabilities_bytes("fake", &words, &Expectation::default()).is_err());
    }
}
