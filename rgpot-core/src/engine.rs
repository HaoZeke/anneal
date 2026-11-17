// MIT License
// Copyright 2023--present rgpot developers

//! Native loader for the generic engine C ABI of `rgpot/engine_c_abi.h`.
//!
//! An engine shared library (`libcpmd_engine.so`, `libuma_engine.so`, ...)
//! exports one symbol set. Which backend runs is which library is loaded, and
//! the backend's configuration is the opaque message the library's create
//! call reads. [`EnginePotential::load`] resolves that symbol set, refuses a
//! library whose ABI major differs from [`ENGINE_ABI_VERSION`], and keeps one
//! engine instance. [`EnginePotential::force`] is one geometry in Angstrom
//! and one answer in eV and eV/Angstrom, the units the header fixes.
//!
//! Return codes follow the header: zero is success, a positive code is an
//! evaluation that failed at this input ([`EngineError::Recoverable`]), and a
//! negative code leaves the instance unusable ([`EngineError::Fatal`]). The
//! message of a failed call is the engine's `rgpot_engine_last_error` when it
//! exports one.
//!
//! [`EnginePotential::into_potential`] wraps the instance as a
//! [`PotentialImpl`], the handle the rest of the crate evaluates through, so
//! an engine library takes part in everything that takes a potential.
//!
//! The library stays mapped for the life of the process, as for
//! [`crate::profile`]: a backend may own threads or MPI state that does not
//! survive `dlclose`.

use std::ffi::{c_char, c_int, c_long, c_void};
use std::fmt;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Mutex;

use libloading::Library;

use crate::potential::PotentialImpl;
use crate::status::{catch_unwind, rgpot_status_t, set_last_error};
use crate::tensor::{copy_f64_tensor, copy_i32_tensor, rgpot_tensor_owned_cpu_f64_2d};
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

/// The ABI major this loader speaks, `RGPOT_ENGINE_ABI_VERSION` of the header.
pub const ENGINE_ABI_VERSION: c_int = 1;

const ERROR_CAPACITY: usize = 1024;

/// Opaque engine instance, `RgpotEnginePot` of the header.
#[repr(C)]
pub struct RgpotEnginePot {
    _private: [u8; 0],
}

/// Coordinate transform of the header. This loader never passes one.
pub type CoordTransformFn = unsafe extern "C" fn(
    user: *mut c_void,
    n_atoms: c_long,
    positions: *mut f64,
    box_: *mut f64,
) -> c_int;

/// `rgpot_engine_abi_version`.
pub type AbiVersionFn = unsafe extern "C" fn() -> c_int;
/// `rgpot_engine_create`.
pub type CreateFn =
    unsafe extern "C" fn(*const c_void, usize, *mut c_char, usize) -> *mut RgpotEnginePot;
/// `rgpot_engine_destroy`.
pub type DestroyFn = unsafe extern "C" fn(*mut RgpotEnginePot);
/// `rgpot_engine_force`.
pub type ForceFn = unsafe extern "C" fn(
    *mut RgpotEnginePot,
    c_long,
    *const f64,
    *const c_int,
    *mut f64,
    *mut f64,
    *mut f64,
    *const f64,
    Option<CoordTransformFn>,
    *mut c_void,
) -> c_int;
/// `rgpot_engine_force_batch`.
pub type ForceBatchFn = unsafe extern "C" fn(
    *mut RgpotEnginePot,
    c_long,
    c_long,
    *const f64,
    *const c_int,
    *mut f64,
    *mut f64,
    *mut f64,
    *const f64,
    Option<CoordTransformFn>,
    *mut c_void,
) -> c_int;
/// `rgpot_engine_last_error`.
pub type LastErrorFn = unsafe extern "C" fn(*const RgpotEnginePot, *mut c_char, usize) -> usize;

/// The symbols of one engine library. The first four are required by the
/// header; the last two are probed and may be absent.
#[derive(Clone, Copy)]
pub struct EngineSymbols {
    /// Required symbol `rgpot_engine_abi_version`.
    pub abi_version: AbiVersionFn,
    /// Required symbol `rgpot_engine_create`.
    pub create: CreateFn,
    /// Required symbol `rgpot_engine_destroy`.
    pub destroy: DestroyFn,
    /// Required symbol `rgpot_engine_force`.
    pub force: ForceFn,
    /// Optional symbol `rgpot_engine_force_batch`.
    pub force_batch: Option<ForceBatchFn>,
    /// Optional symbol `rgpot_engine_last_error`.
    pub last_error: Option<LastErrorFn>,
}

/// Why an engine could not be loaded or an evaluation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// The shared library did not open.
    Load(String),
    /// A symbol the header requires is not exported.
    Missing(String),
    /// The library speaks another ABI major.
    Abi {
        /// The ABI major expected by this loader.
        expected: i32,
        /// The ABI major returned by `rgpot_engine_abi_version`.
        found: i32,
    },
    /// `rgpot_engine_create` returned no instance; the text is its error buffer.
    Create(String),
    /// The request does not fit the ABI: a length or a non-finite answer.
    Invalid(String),
    /// The library does not export the optional symbol.
    Unsupported(&'static str),
    /// A positive code: the evaluation failed at this input.
    Recoverable {
        /// Positive error code returned by the engine.
        code: i32,
        /// Error message from `rgpot_engine_last_error`.
        message: String,
    },
    /// A negative code: the instance is unusable.
    Fatal {
        /// Negative error code returned by the engine.
        code: i32,
        /// Error message from `rgpot_engine_last_error`.
        message: String,
    },
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(why) => write!(f, "engine library did not load: {why}"),
            Self::Missing(name) => write!(f, "engine library does not export {name}"),
            Self::Abi { expected, found } => {
                write!(f, "engine ABI major {found}, this loader speaks {expected}")
            }
            Self::Create(why) => write!(f, "rgpot_engine_create failed: {why}"),
            Self::Invalid(why) => write!(f, "invalid engine request: {why}"),
            Self::Unsupported(name) => write!(f, "engine library does not export {name}"),
            Self::Recoverable { code, message } => {
                write!(f, "engine evaluation failed with code {code}: {message}")
            }
            Self::Fatal { code, message } => {
                write!(f, "engine instance is unusable, code {code}: {message}")
            }
        }
    }
}

impl std::error::Error for EngineError {}

/// One geometry. Positions in Angstrom, row-major `[n_atoms, 3]`; the cell is
/// the row-major 3 by 3 matrix in Angstrom.
#[derive(Debug, Clone, Copy)]
pub struct EngineRequest<'a> {
    /// Atomic coordinates in Angstrom, row-major `[n_atoms, 3]`.
    pub positions: &'a [f64],
    /// Atomic numbers, one per atom.
    pub atomic_numbers: &'a [i32],
    /// Row-major 3 by 3 lattice vectors in Angstrom.
    pub box_matrix: &'a [f64; 9],
}

/// Energy in eV and forces in eV/Angstrom.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineEvaluation {
    /// Total potential energy in eV.
    pub energy: f64,
    /// Cartesian atomic forces in eV/Angstrom, row-major `[n_atoms, 3]`.
    pub forces: Vec<f64>,
}

/// One engine instance behind the generic engine C ABI.
pub struct EnginePotential {
    symbols: EngineSymbols,
    handle: NonNull<RgpotEnginePot>,
    _library: Option<ManuallyDrop<Library>>,
}

// The instance is used by one caller at a time through `&mut self`. The
// header's reentrancy classes govern concurrent handles, which this type
// does not offer.
unsafe impl Send for EnginePotential {}

impl EnginePotential {
    /// Open the engine library at `path` and create an instance from the
    /// backend's configuration message.
    ///
    /// # Safety
    ///
    /// The library must implement the engine C ABI with the signatures of
    /// this module, and `config` must be the message its create call reads.
    pub unsafe fn load(path: &Path, config: &[u8]) -> Result<Self, EngineError> {
        let library = unsafe { Library::new(path) }
            .map_err(|why| EngineError::Load(format!("{}: {why}", path.display())))?;
        let symbols = unsafe { resolve(&library)? };
        let mut engine = unsafe { Self::from_symbols(symbols, config)? };
        engine._library = Some(ManuallyDrop::new(library));
        Ok(engine)
    }

    /// Create an instance from symbols already in hand. [`Self::load`] is
    /// this after resolving them from a library.
    ///
    /// # Safety
    ///
    /// The symbols must honour the engine C ABI for as long as the instance
    /// lives, and `config` must be the message `symbols.create` reads.
    pub unsafe fn from_symbols(symbols: EngineSymbols, config: &[u8]) -> Result<Self, EngineError> {
        let found = unsafe { (symbols.abi_version)() };
        if found != ENGINE_ABI_VERSION {
            return Err(EngineError::Abi {
                expected: ENGINE_ABI_VERSION,
                found,
            });
        }
        // In-tree engines cast `config` to `capnp::word`, which requires
        // 8-byte alignment even when the caller passes an unaligned `&[u8]`.
        let mut aligned = vec![0_u64; config.len().div_ceil(8)];
        for (word, chunk) in aligned.iter_mut().zip(config.chunks(8)) {
            let mut bytes = [0_u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            *word = u64::from_ne_bytes(bytes);
        }
        let mut errbuf = vec![0 as c_char; ERROR_CAPACITY];
        let raw = unsafe {
            (symbols.create)(
                aligned.as_ptr().cast(),
                config.len(),
                errbuf.as_mut_ptr(),
                errbuf.len(),
            )
        };
        match NonNull::new(raw) {
            Some(handle) => Ok(Self {
                symbols,
                handle,
                _library: None,
            }),
            None => Err(EngineError::Create(text_of(&errbuf))),
        }
    }

    /// Energy and forces of one geometry.
    pub fn force(&mut self, request: &EngineRequest<'_>) -> Result<EngineEvaluation, EngineError> {
        let n_atoms = request.atomic_numbers.len();
        check_lengths(n_atoms, request.positions.len())?;
        let mut forces = vec![0.0; 3 * n_atoms];
        let mut energy = 0.0;
        let code = unsafe {
            (self.symbols.force)(
                self.handle.as_ptr(),
                n_atoms as c_long,
                request.positions.as_ptr(),
                request.atomic_numbers.as_ptr(),
                forces.as_mut_ptr(),
                &mut energy,
                std::ptr::null_mut(),
                request.box_matrix.as_ptr(),
                None,
                std::ptr::null_mut(),
            )
        };
        self.answer(code)?;
        finite(energy, &forces)?;
        Ok(EngineEvaluation { energy, forces })
    }

    /// `n_systems` geometries that share one composition, in one engine call.
    /// `positions` holds `3 * n_atoms` values per system and `boxes` nine.
    /// A library without `rgpot_engine_force_batch` is
    /// [`EngineError::Unsupported`]; the caller decides whether to loop.
    pub fn force_batch(
        &mut self,
        atomic_numbers: &[i32],
        positions: &[f64],
        boxes: &[f64],
    ) -> Result<Vec<EngineEvaluation>, EngineError> {
        let batch = self
            .symbols
            .force_batch
            .ok_or(EngineError::Unsupported("rgpot_engine_force_batch"))?;
        let n_atoms = atomic_numbers.len();
        if n_atoms == 0 || positions.is_empty() || !positions.len().is_multiple_of(3 * n_atoms) {
            return Err(EngineError::Invalid(format!(
                "{} coordinates are not a whole number of {n_atoms}-atom systems",
                positions.len()
            )));
        }
        let n_systems = positions.len() / (3 * n_atoms);
        if boxes.len() != 9 * n_systems {
            return Err(EngineError::Invalid(format!(
                "{} cell values for {n_systems} systems; expected {}",
                boxes.len(),
                9 * n_systems
            )));
        }
        let mut forces = vec![0.0; positions.len()];
        let mut energies = vec![0.0; n_systems];
        let code = unsafe {
            batch(
                self.handle.as_ptr(),
                n_systems as c_long,
                n_atoms as c_long,
                positions.as_ptr(),
                atomic_numbers.as_ptr(),
                forces.as_mut_ptr(),
                energies.as_mut_ptr(),
                std::ptr::null_mut(),
                boxes.as_ptr(),
                None,
                std::ptr::null_mut(),
            )
        };
        self.answer(code)?;
        let mut out = Vec::with_capacity(n_systems);
        for (energy, block) in energies.iter().zip(forces.chunks_exact(3 * n_atoms)) {
            finite(*energy, block)?;
            out.push(EngineEvaluation {
                energy: *energy,
                forces: block.to_vec(),
            });
        }
        Ok(out)
    }

    /// The instance as a [`PotentialImpl`]. The handle owns the engine and
    /// destroys it on drop. Its callback evaluates through [`Self::force`];
    /// the cell tensor is the 3 by 3 matrix in Angstrom.
    pub fn into_potential(self) -> PotentialImpl {
        let owned = Box::into_raw(Box::new(Mutex::new(self))).cast::<c_void>();
        PotentialImpl::new(engine_callback, owned, Some(drop_engine))
    }

    fn answer(&self, code: c_int) -> Result<(), EngineError> {
        if code == 0 {
            return Ok(());
        }
        let message = self.last_error_text(code);
        if code > 0 {
            Err(EngineError::Recoverable { code, message })
        } else {
            Err(EngineError::Fatal { code, message })
        }
    }

    fn last_error_text(&self, code: c_int) -> String {
        let Some(last_error) = self.symbols.last_error else {
            return format!("no rgpot_engine_last_error; code {code}");
        };
        let mut buffer = vec![0 as c_char; ERROR_CAPACITY];
        let length = unsafe { last_error(self.handle.as_ptr(), buffer.as_mut_ptr(), buffer.len()) };
        if length == 0 {
            return format!("the engine left no message; code {code}");
        }
        text_of(&buffer)
    }
}

impl Drop for EnginePotential {
    fn drop(&mut self) {
        unsafe { (self.symbols.destroy)(self.handle.as_ptr()) };
    }
}

fn check_lengths(n_atoms: usize, n_positions: usize) -> Result<(), EngineError> {
    if n_atoms == 0 {
        return Err(EngineError::Invalid(
            "a geometry needs at least one atom".into(),
        ));
    }
    if n_positions != 3 * n_atoms {
        return Err(EngineError::Invalid(format!(
            "{n_positions} coordinates for {n_atoms} atoms; expected {}",
            3 * n_atoms
        )));
    }
    Ok(())
}

fn finite(energy: f64, forces: &[f64]) -> Result<(), EngineError> {
    if !energy.is_finite() {
        return Err(EngineError::Invalid(
            "the engine energy is not finite".into(),
        ));
    }
    match forces.iter().position(|value| !value.is_finite()) {
        Some(index) => Err(EngineError::Invalid(format!(
            "the engine force at element {index} is not finite"
        ))),
        None => Ok(()),
    }
}

fn text_of(buffer: &[c_char]) -> String {
    let bytes: Vec<u8> = buffer
        .iter()
        .map(|value| *value as u8)
        .take_while(|value| *value != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

unsafe fn resolve(library: &Library) -> Result<EngineSymbols, EngineError> {
    unsafe fn required<T: Copy>(library: &Library, name: &str) -> Result<T, EngineError> {
        let symbol = unsafe { library.get::<T>(format!("{name}\0").as_bytes()) }
            .map_err(|_| EngineError::Missing(name.to_owned()))?;
        Ok(*symbol)
    }
    unsafe fn optional<T: Copy>(library: &Library, name: &str) -> Option<T> {
        unsafe { library.get::<T>(format!("{name}\0").as_bytes()) }
            .ok()
            .map(|symbol| *symbol)
    }
    Ok(EngineSymbols {
        abi_version: unsafe { required(library, "rgpot_engine_abi_version")? },
        create: unsafe { required(library, "rgpot_engine_create")? },
        destroy: unsafe { required(library, "rgpot_engine_destroy")? },
        force: unsafe { required(library, "rgpot_engine_force")? },
        force_batch: unsafe { optional(library, "rgpot_engine_force_batch") },
        last_error: unsafe { optional(library, "rgpot_engine_last_error") },
    })
}

unsafe extern "C" fn drop_engine(user_data: *mut c_void) {
    drop(unsafe { Box::from_raw(user_data.cast::<Mutex<EnginePotential>>()) });
}

unsafe extern "C" fn engine_callback(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user_data.is_null() || input.is_null() || output.is_null() {
            set_last_error("engine potential: null user data, input or output");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        let mut engine = unsafe { &*user_data.cast::<Mutex<EnginePotential>>() }
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let input = unsafe { &*input };
        let Some(n_atoms) = (unsafe { input.n_atoms() }) else {
            set_last_error("engine potential: positions tensor has no shape");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        };
        let positions = copy_f64_tensor(input.positions, "positions", 3 * n_atoms, true);
        let numbers = copy_i32_tensor(input.atomic_numbers, "atomic_numbers", n_atoms);
        let cell = copy_f64_tensor(input.box_matrix, "box_matrix", 9, false);
        let (positions, numbers, cell) = match (positions, numbers, cell) {
            (Ok(positions), Ok(numbers), Ok(cell)) => (positions, numbers, cell),
            (Err(why), _, _) | (_, Err(why), _) | (_, _, Err(why)) => {
                set_last_error(&why);
                return rgpot_status_t::RGPOT_INVALID_PARAMETER;
            }
        };
        let mut matrix = [0.0; 9];
        matrix.copy_from_slice(&cell);
        let request = EngineRequest {
            positions: &positions,
            atomic_numbers: &numbers,
            box_matrix: &matrix,
        };
        match engine.force(&request) {
            Ok(answer) => {
                let forces = unsafe {
                    rgpot_tensor_owned_cpu_f64_2d(answer.forces.as_ptr(), n_atoms as i64, 3)
                };
                if forces.is_null() {
                    return rgpot_status_t::RGPOT_INTERNAL_ERROR;
                }
                unsafe {
                    (*output).forces = forces;
                    (*output).energy = answer.energy;
                    (*output).variance = 0.0;
                }
                rgpot_status_t::RGPOT_SUCCESS
            }
            Err(EngineError::Invalid(why)) => {
                set_last_error(&why);
                rgpot_status_t::RGPOT_INVALID_PARAMETER
            }
            Err(why) => {
                set_last_error(&why.to_string());
                rgpot_status_t::RGPOT_INTERNAL_ERROR
            }
        }
    }))
}
