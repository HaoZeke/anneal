#![cfg(unix)]

use std::ffi::c_int;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rgpot_core::engine::{
    AcquireReason, CoordinateTransform, EngineCallbacks, EngineError, EngineEvaluation,
    EngineFailure, EnginePotential, EngineRequest, EngineSession, EngineStatus,
    RgpotEngineAbiStamp, RgpotEngineCaps, RgpotEngineComponent, ENGINE_ABI_LAYOUT_REVISION,
    ENGINE_ABI_MINOR, ENGINE_ABI_VERSION, ENGINE_ERR_UNSUPPORTED,
};
use rgpot_core::hvp::rgpot_hvp_scheme_t;
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::{
    rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3, rgpot_tensor_cpu_i32_1d,
    rgpot_tensor_data, rgpot_tensor_free,
};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

const SPRING: f64 = 2.5;
const CELL: [f64; 9] = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];

#[cfg(feature = "cpmd-fixture")]
#[derive(Debug)]
enum CpmdFixtureError {
    MissingEnvironment(&'static str),
    Read(std::io::Error),
    Engine(EngineError),
}

#[cfg(feature = "cpmd-fixture")]
impl std::fmt::Display for CpmdFixtureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingEnvironment(name) => {
                write!(f, "required CPMD fixture variable {name} is not set")
            }
            Self::Read(error) => write!(f, "read CPMD fixture parameters: {error}"),
            Self::Engine(error) => error.fmt(f),
        }
    }
}

#[cfg(feature = "cpmd-fixture")]
impl std::error::Error for CpmdFixtureError {}

#[cfg(feature = "cpmd-fixture")]
impl From<std::io::Error> for CpmdFixtureError {
    fn from(error: std::io::Error) -> Self {
        Self::Read(error)
    }
}

#[cfg(feature = "cpmd-fixture")]
impl From<EngineError> for CpmdFixtureError {
    fn from(error: EngineError) -> Self {
        Self::Engine(error)
    }
}

fn build(name: &str, defines: &[&str]) -> PathBuf {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/harmonic_engine.c");
    let include = Path::new(env!("CARGO_MANIFEST_DIR")).join("../CppCore");
    let suffix = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("lib{name}.{suffix}"));
    let target = env!("RGPOT_BUILD_TARGET");
    let compiler = match cc::Build::new()
        .target(target)
        .host(target)
        .opt_level(0)
        .try_get_compiler()
    {
        Ok(compiler) => compiler,
        Err(why) => panic!(
            "cannot compile C fixture '{name}': no C compiler available for target {target} ({why})"
        ),
    };
    let mut command = compiler.to_command();
    if cfg!(target_os = "macos") {
        command.arg("-dynamiclib");
    } else {
        command.args(["-shared", "-fPIC"]);
    }
    command
        .arg("-I")
        .arg(include)
        .arg("-o")
        .arg(&out)
        .arg(&source);
    for define in defines {
        command.arg(format!("-D{define}"));
    }
    let status = match command.status() {
        Ok(status) => status,
        Err(why) if why.kind() == std::io::ErrorKind::NotFound => panic!(
            "cannot compile C fixture '{name}': compiler {:?} not found ({why})",
            compiler.path()
        ),
        Err(why) => panic!("spawn C compiler {:?}: {why}", compiler.path()),
    };
    assert!(status.success(), "the fixture did not compile: {status}");
    out
}

/// Rounding bar of a sum of `terms` products: each costs one rounding of its own size.
fn within_roundoff(got: f64, want: f64, terms: usize) -> bool {
    (got - want).abs() <= terms as f64 * f64::EPSILON * got.abs().max(want.abs())
}

fn all_within_roundoff(got: &[f64], want: &[f64]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| within_roundoff(*g, *w, 1))
}

fn config() -> [u8; 8] {
    SPRING.to_ne_bytes()
}

fn load(path: &Path) -> EnginePotential {
    unsafe { EnginePotential::load(path, &config()) }.expect("load the harmonic engine")
}

/// Opens the engine library named by `RGPOT_TEST_ENGINE` with an empty
/// configuration and checks that the host accepts its ABI stamp.
///
/// Whether `rgpot_engine_create` then accepts the empty message is the
/// engine's own choice: the Lennard-Jones plugin builds its documented default
/// potential, while a backend that needs a params message refuses with
/// `EngineError::Create`. Any other error means the loader rejected the library
/// before create, which is the failure this test exists to catch.
///
/// The meson test `engine_abi_stamp_rust_loader` runs it with
/// `--ignored` against the Lennard-Jones plugin built in the same tree.
#[test]
#[ignore = "needs RGPOT_TEST_ENGINE to name a generic engine library"]
fn configured_external_engine_passes_the_abi_check() {
    let path = std::env::var_os("RGPOT_TEST_ENGINE")
        .expect("RGPOT_TEST_ENGINE names the engine library to open");
    let path = Path::new(&path);
    assert_eq!(
        RgpotEngineAbiStamp::HOST,
        RgpotEngineAbiStamp {
            abi_major: 1,
            abi_minor: 5,
            layout_revision: 1,
        }
    );
    match unsafe { EnginePotential::load(path, &[]) } {
        Ok(engine) => assert_eq!(
            engine.abi_stamp(),
            RgpotEngineAbiStamp::HOST,
            "{} reports another ABI stamp",
            path.display()
        ),
        Err(EngineError::Create(_)) => {}
        Err(error) => panic!(
            "rgpot host rejected {} before create: {error}",
            path.display()
        ),
    }
}

fn load_session(path: &Path) -> EngineSession {
    // SAFETY: `path` names the fixture compiled from the ABI header above.
    unsafe { EngineSession::load(path, &config()) }.expect("load the harmonic engine session")
}

struct FixtureObserver(libloading::Library);

impl FixtureObserver {
    fn open(path: &Path) -> Self {
        Self(unsafe { libloading::Library::new(path) }.expect("open the fixture observer"))
    }

    fn destroyed(&self) -> c_int {
        let count: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
            unsafe { self.0.get(b"fixture_destroy_count\0") }.expect("count symbol");
        unsafe { count() }
    }

    fn native_calls(&self) -> c_int {
        let count: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
            unsafe { self.0.get(b"fixture_native_call_count\0") }
                .expect("native call count symbol");
        unsafe { count() }
    }
}

#[cfg(target_os = "linux")]
fn library_is_mapped(path: &Path) -> bool {
    let canonical = path.canonicalize().expect("canonical fixture path");
    let maps = std::fs::read_to_string("/proc/self/maps").expect("read /proc/self/maps");
    maps.lines()
        .filter_map(|line| line.split_whitespace().last())
        .any(|mapped| Path::new(mapped) == canonical)
}

fn batch_calls(path: &Path) -> c_int {
    let library = unsafe { libloading::Library::new(path) }.expect("reopen the fixture");
    let count: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"fixture_batch_call_count\0") }.expect("batch call count symbol");
    unsafe { count() }
}

fn null_callback_clears(path: &Path) -> c_int {
    let library = unsafe { libloading::Library::new(path) }.expect("reopen the fixture");
    let count: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"fixture_null_callback_clear_count\0") }
            .expect("null callback clear count symbol");
    unsafe { count() }
}

fn unsupported_code(path: &Path) -> c_int {
    let library = unsafe { libloading::Library::new(path) }.expect("reopen the fixture");
    let value: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"fixture_unsupported_code\0") }.expect("code symbol");
    unsafe { value() }
}

#[test]
fn force_is_the_harmonic_energy_and_minus_the_gradient() {
    let path = build("harmonic", &[]);
    let mut engine = load(&path);
    let positions = [0.5, -1.0, 2.0, 0.0, 3.0, -0.25];
    let answer = engine
        .force(&EngineRequest {
            positions: &positions,
            atomic_numbers: &[1, 8],
            box_matrix: &CELL,
        })
        .expect("force");
    let sum: f64 = positions.iter().map(|x| x * x).sum();
    assert!(within_roundoff(
        answer.energy,
        0.5 * SPRING * sum,
        3 * positions.len()
    ));
    let expected: Vec<f64> = positions.iter().map(|x| -SPRING * x).collect();
    assert!(all_within_roundoff(&answer.forces, &expected));
}

struct Shift;

impl CoordinateTransform for Shift {
    fn transform(
        &mut self,
        positions: &mut [f64],
        _box_matrix: &mut [f64; 9],
    ) -> Result<(), EngineFailure> {
        for value in positions {
            *value += 1.0;
        }
        Ok(())
    }
}

#[test]
fn session_transform_changes_only_engine_scratch() {
    let path = build("harmonic_session_transform", &[]);
    let mut session = load_session(&path);
    let positions = [1.0, 2.0, 3.0];
    let mut shift = Shift;
    let answer = session
        .force(
            &EngineRequest {
                positions: &positions,
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            Some(&mut shift),
        )
        .expect("transformed force");
    assert_eq!(positions, [1.0, 2.0, 3.0]);
    assert_eq!(answer.energy.to_bits(), 36.25f64.to_bits());
    assert_eq!(answer.forces, [-5.0, -7.5, -10.0]);
}

#[derive(Default)]
struct CallbackRecord {
    acquisitions: Vec<AcquireReason>,
    states: Vec<u64>,
}

struct Oracle {
    record: Arc<Mutex<CallbackRecord>>,
}

impl EngineCallbacks for Oracle {
    fn oracle(&mut self, request: &EngineRequest<'_>) -> Result<EngineEvaluation, EngineFailure> {
        Ok(EngineEvaluation {
            energy: request.positions.iter().sum(),
            forces: request.positions.iter().map(|value| -*value).collect(),
        })
    }

    fn on_acquire(&mut self, reason: AcquireReason) {
        self.record
            .lock()
            .expect("callback record")
            .acquisitions
            .push(reason);
    }

    fn on_model_update(&mut self, model_state: u64) {
        self.record
            .lock()
            .expect("callback record")
            .states
            .push(model_state);
    }
}

#[test]
fn session_owns_callbacks_and_reports_model_state() {
    let path = build("harmonic_session_callbacks", &[]);
    let record = Arc::new(Mutex::new(CallbackRecord::default()));
    let mut session = load_session(&path);
    session
        .set_callbacks(Some(Box::new(Oracle {
            record: Arc::clone(&record),
        })))
        .expect("set callbacks");
    let answer = session
        .force(
            &EngineRequest {
                positions: &[1.0, 2.0, 3.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            None,
        )
        .expect("oracle force");
    assert_eq!(answer.energy, 6.0);
    assert_eq!(answer.forces, [-1.0, -2.0, -3.0]);
    assert_eq!(session.model_state().expect("model state"), 1);
    let record = record.lock().expect("callback record");
    assert_eq!(record.acquisitions, [AcquireReason::Variance]);
    assert_eq!(record.states, [1]);
}

#[test]
fn failed_callback_replacement_keeps_old_registration_and_none_passes_null() {
    let path = build("harmonic_callback_transaction", &[]);
    let old = Arc::new(Mutex::new(CallbackRecord::default()));
    let replacement = Arc::new(Mutex::new(CallbackRecord::default()));
    let mut session = load_session(&path);
    session
        .set_callbacks(Some(Box::new(Oracle {
            record: Arc::clone(&old),
        })))
        .expect("set initial callbacks");

    let error = session
        .set_callbacks(Some(Box::new(Oracle {
            record: Arc::clone(&replacement),
        })))
        .expect_err("fixture refuses callback replacement");
    assert_eq!(
        error,
        EngineError::Recoverable {
            code: 2,
            message: "callback replacement refused".into(),
        }
    );
    session
        .force(
            &EngineRequest {
                positions: &[1.0, 2.0, 3.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            None,
        )
        .expect("old oracle remains registered");
    assert_eq!(
        old.lock().expect("old callback record").acquisitions,
        [AcquireReason::Variance]
    );
    assert!(replacement
        .lock()
        .expect("replacement callback record")
        .acquisitions
        .is_empty());

    let before = null_callback_clears(&path);
    session.set_callbacks(None).expect("clear callbacks");
    assert_eq!(null_callback_clears(&path), before + 1);
    let answer = session
        .force(
            &EngineRequest {
                positions: &[1.0, 0.0, 0.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            None,
        )
        .expect("harmonic force after clearing callbacks");
    assert_eq!(answer.energy.to_bits(), 1.25f64.to_bits());
}

struct PanicOracle;

impl EngineCallbacks for PanicOracle {
    fn oracle(&mut self, _request: &EngineRequest<'_>) -> Result<EngineEvaluation, EngineFailure> {
        panic!("oracle panic for fixture")
    }
}

struct PanicModelUpdate;

impl EngineCallbacks for PanicModelUpdate {
    fn oracle(&mut self, request: &EngineRequest<'_>) -> Result<EngineEvaluation, EngineFailure> {
        Ok(EngineEvaluation {
            energy: 0.0,
            forces: vec![0.0; request.positions.len()],
        })
    }

    fn on_model_update(&mut self, _model_state: u64) {
        panic!("model update panic for fixture")
    }
}

struct PanicTransform;

impl CoordinateTransform for PanicTransform {
    fn transform(
        &mut self,
        _positions: &mut [f64],
        _box_matrix: &mut [f64; 9],
    ) -> Result<(), EngineFailure> {
        panic!("transform panic for fixture")
    }
}

fn assert_poisoned_without_an_extra_force(session: &mut EngineSession, path: &Path) {
    let observer = FixtureObserver::open(path);
    let before = observer.native_calls();
    let error = session
        .force(
            &EngineRequest {
                positions: &[1.0, 0.0, 0.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            None,
        )
        .expect_err("fatal session remains poisoned");
    assert_eq!(
        error.engine_status(),
        Some(EngineStatus::FatalInstanceFailure(-1))
    );
    assert_eq!(observer.native_calls(), before);
}

#[test]
fn oracle_panic_is_fatal_and_poisons_the_session() {
    let path = build("harmonic_oracle_panic", &[]);
    let mut session = load_session(&path);
    session
        .set_callbacks(Some(Box::new(PanicOracle)))
        .expect("set panic callback");
    let error = session
        .force(
            &EngineRequest {
                positions: &[1.0, 0.0, 0.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            None,
        )
        .expect_err("oracle panic is fatal");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -1,
            message: "oracle callback panicked".into(),
        }
    );
    assert_poisoned_without_an_extra_force(&mut session, &path);
}

#[test]
fn transform_panic_is_fatal_and_poisons_the_session() {
    let path = build("harmonic_transform_panic", &[]);
    let mut session = load_session(&path);
    let error = session
        .force(
            &EngineRequest {
                positions: &[1.0, 0.0, 0.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            Some(&mut PanicTransform),
        )
        .expect_err("transform panic is fatal");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -1,
            message: "coordinate transform panicked".into(),
        }
    );
    assert_poisoned_without_an_extra_force(&mut session, &path);
}

#[test]
fn hint_reports_status_updates_state_and_surfaces_callback_panics() {
    let path = build("harmonic_hint", &[]);
    let mut session = load_session(&path);
    session.hint(&[1]).expect("valid hint");
    assert_eq!(session.model_state().expect("model state"), 1);
    let error = session.hint(&[0xff]).expect_err("fixture refuses hint");
    assert_eq!(
        error,
        EngineError::Recoverable {
            code: 2,
            message: "hint refused".into(),
        }
    );
    assert_eq!(session.model_state().expect("model state"), 1);

    let panic_path = build("harmonic_hint_callback_panic", &[]);
    let mut panic_session = load_session(&panic_path);
    panic_session
        .set_callbacks(Some(Box::new(PanicModelUpdate)))
        .expect("set model-update callback");
    let error = panic_session
        .hint(&[1])
        .expect_err("model-update callback panic is fatal");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -1,
            message: "model-update callback panicked".into(),
        }
    );
    assert_poisoned_without_an_extra_force(&mut panic_session, &panic_path);
}

#[test]
fn native_fatal_status_poisons_an_engine_session() {
    let path = build("harmonic_session_fatal_once", &["FATAL_ONCE"]);
    let observer = FixtureObserver::open(&path);
    let mut session = load_session(&path);
    let error = session
        .force(
            &EngineRequest {
                positions: &[1.0, 0.0, 0.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            None,
        )
        .expect_err("fixture fires one fatal status");
    assert_eq!(
        error,
        EngineError::Fatal {
            code: -7,
            message: "fatal once".into(),
        }
    );
    let before = observer.native_calls();
    let later = session
        .force(
            &EngineRequest {
                positions: &[1.0, 0.0, 0.0],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            None,
        )
        .expect_err("session remains poisoned");
    assert_eq!(later, error);
    assert_eq!(observer.native_calls(), before);
}

#[test]
fn typed_status_does_not_retry_from_error_text() {
    assert_eq!(EngineStatus::from_code(0), EngineStatus::Success);
    assert_eq!(
        EngineStatus::from_code(2),
        EngineStatus::RecoverableInputFailure(2)
    );
    assert_eq!(
        EngineStatus::from_code(-2),
        EngineStatus::FatalInstanceFailure(-2)
    );

    let path = build("harmonic_session_no_retry", &[]);
    let observer = FixtureObserver::open(&path);
    let mut session = load_session(&path);
    let before = observer.native_calls();
    let error = session
        .force(
            &EngineRequest {
                positions: &[0.0, 0.0, 0.0],
                atomic_numbers: &[0],
                box_matrix: &CELL,
            },
            None,
        )
        .expect_err("recoverable input");
    assert_eq!(
        error.engine_status(),
        Some(EngineStatus::RecoverableInputFailure(1))
    );
    assert_eq!(observer.native_calls() - before, 2);
}

#[test]
fn batch_is_the_same_answer_as_one_call_per_system() {
    let path = build("harmonic_batch", &[]);
    let mut engine = load(&path);
    let numbers = [1, 8];
    let first = [0.5, -1.0, 2.0, 0.0, 3.0, -0.25];
    let second = [1.5, 1.0, -2.0, 0.5, -3.0, 0.75];
    let mut positions = first.to_vec();
    positions.extend_from_slice(&second);
    let mut boxes = CELL.to_vec();
    boxes.extend_from_slice(&CELL);
    let before = batch_calls(&path);
    let batch = engine
        .force_batch(&numbers, &positions, &boxes)
        .expect("batch");
    assert_eq!(batch_calls(&path), before + 1);
    assert_eq!(batch.len(), 2);
    for (system, one) in [first, second].iter().enumerate() {
        let single = engine
            .force(&EngineRequest {
                positions: one,
                atomic_numbers: &numbers,
                box_matrix: &CELL,
            })
            .expect("single");
        assert_eq!(batch[system].energy.to_bits(), single.energy.to_bits());
        assert_eq!(
            batch[system]
                .forces
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            single
                .forces
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[cfg(feature = "cpmd-fixture")]
#[test]
#[ignore = "needs RGPOT_ENGINE_CPMD_FIXTURE and RGPOT_ENGINE_CPMD_PARAMS; run with --ignored"]
fn cpmd_fixture_batch_preserves_single_call_bits_and_order() -> Result<(), CpmdFixtureError> {
    let engine_path = std::env::var_os("RGPOT_ENGINE_CPMD_FIXTURE").ok_or(
        CpmdFixtureError::MissingEnvironment("RGPOT_ENGINE_CPMD_FIXTURE"),
    )?;
    let params_path = std::env::var_os("RGPOT_ENGINE_CPMD_PARAMS").ok_or(
        CpmdFixtureError::MissingEnvironment("RGPOT_ENGINE_CPMD_PARAMS"),
    )?;
    let config = std::fs::read(params_path)?;
    let positions = [0.0, 0.0, 0.0, 0.8, 0.1, 0.0, 1.5, -0.2, 0.3, 0.1, 0.4, -0.7];
    let numbers = [7, 1];
    let mut boxes = CELL.to_vec();
    boxes.extend_from_slice(&CELL);

    // SAFETY: this feature deliberately trusts the externally supplied CPMD
    // fixture to implement the versioned engine ABI.
    let mut singles = unsafe { EngineSession::load(Path::new(&engine_path), &config) }?;
    let first = singles.force(
        &EngineRequest {
            positions: &positions[..6],
            atomic_numbers: &numbers,
            box_matrix: &CELL,
        },
        None,
    )?;
    let second = singles.force(
        &EngineRequest {
            positions: &positions[6..],
            atomic_numbers: &numbers,
            box_matrix: &CELL,
        },
        None,
    )?;

    // SAFETY: the same trusted CPMD fixture is loaded into a fresh instance.
    let mut batch = unsafe { EngineSession::load(Path::new(&engine_path), &config) }?;
    let answers = batch.force_batch(&numbers, &positions, &boxes, None)?;
    assert_eq!(answers[0].energy.to_bits(), first.energy.to_bits());
    assert_eq!(answers[1].energy.to_bits(), second.energy.to_bits());
    assert_eq!(answers[0].forces, first.forces);
    assert_eq!(answers[1].forces, second.forces);
    Ok(())
}

#[test]
fn batch_without_the_symbol_is_unsupported() {
    let path = build("harmonic_nobatch", &["NO_BATCH"]);
    let mut engine = load(&path);
    let error = engine
        .force_batch(&[1], &[0.0, 0.0, 0.0], &CELL)
        .unwrap_err();
    assert!(matches!(
        error,
        EngineError::Unsupported {
            operation: "rgpot_engine_force_batch",
            ..
        }
    ));
}

#[test]
fn create_failure_carries_the_engine_message() {
    let path = build("harmonic_create", &[]);
    let error = unsafe { EnginePotential::load(&path, &[1, 2, 3]) }
        .err()
        .expect("create must fail");
    match error {
        EngineError::Create(message) => assert!(message.contains("config is 3 bytes"), "{message}"),
        other => panic!("expected a create error, got {other}"),
    }
}

#[test]
fn positive_code_is_recoverable_and_negative_is_fatal() {
    let path = build("harmonic_codes", &[]);
    let mut engine = load(&path);
    let positions = [0.0; 6];
    let recoverable = engine
        .force(&EngineRequest {
            positions: &positions,
            atomic_numbers: &[1, 0],
            box_matrix: &CELL,
        })
        .unwrap_err();
    assert_eq!(
        recoverable,
        EngineError::Recoverable {
            code: 1,
            message: "atom 1 has atomic number 0".into()
        }
    );
    let fatal = engine
        .force(&EngineRequest {
            positions: &positions,
            atomic_numbers: &[-1, 1],
            box_matrix: &CELL,
        })
        .unwrap_err();
    assert_eq!(
        fatal,
        EngineError::Fatal {
            code: -1,
            message: "atom 0 has a negative atomic number".into()
        }
    );
}

#[test]
fn fatal_status_poisons_the_handle_and_makes_zero_later_native_calls() {
    let path = build("harmonic_fatal_once", &["FATAL_ONCE"]);
    let observer = FixtureObserver::open(&path);
    let before_destroy = observer.destroyed();
    let mut engine = load(&path);
    let expected = EngineError::Fatal {
        code: -7,
        message: "fatal once".into(),
    };
    let valid = EngineRequest {
        positions: &[1.0, 0.0, 0.0],
        atomic_numbers: &[1],
        box_matrix: &CELL,
    };

    assert_eq!(engine.force(&valid).unwrap_err(), expected);
    let after_fatal = observer.native_calls();
    assert!(after_fatal > 0);

    // The fixture returns -7 only once and 0 thereafter, so an unpoisoned
    // handle would both increment native_calls and return Ok here.
    assert_eq!(engine.force(&valid).unwrap_err(), expected);
    assert_eq!(
        engine
            .force(&EngineRequest {
                positions: &[0.0; 4],
                atomic_numbers: &[1, 1],
                box_matrix: &CELL,
            })
            .unwrap_err(),
        expected
    );
    assert_eq!(
        engine
            .force_batch(&[1], &[1.0, 0.0, 0.0], &CELL)
            .unwrap_err(),
        expected
    );
    assert_eq!(engine.hvp(&valid, &[1.0, 0.0, 0.0]).unwrap_err(), expected);
    assert_eq!(engine.caps().unwrap_err(), expected);
    assert_eq!(observer.native_calls(), after_fatal);

    let potential = engine.into_potential();
    assert!(!potential.as_handle().has_hessian_vector_callback());
    assert_eq!(observer.native_calls(), after_fatal);
    let mut positions = vec![1.0, 0.0, 0.0];
    let mut numbers = vec![1];
    let mut cell = CELL;
    let input = rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 1, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), 1) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
    };
    let mut output = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    assert_eq!(
        unsafe { potential.calculate(&input, &mut output) },
        rgpot_status_t::RGPOT_INTERNAL_ERROR
    );
    assert_eq!(observer.native_calls(), after_fatal);
    unsafe {
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
    drop(potential);
    assert_eq!(observer.destroyed(), before_destroy + 1);
}

#[test]
fn fatal_hvp_poisons_a_live_potential_handle_without_reentering_native_code() {
    let path = build("harmonic_fatal_hvp_handle", &["FATAL_ONCE"]);
    let observer = FixtureObserver::open(&path);
    let before_destroy = observer.destroyed();
    let potential = load(&path).into_potential();
    assert!(potential.as_handle().has_hessian_vector_callback());

    let mut positions = vec![0.5, -1.0, 2.0];
    let mut numbers = vec![1];
    let mut cell = CELL;
    let input = rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 1, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), 1) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
    };
    let mut product = [0.0; 3];
    let err = unsafe {
        potential.as_handle().hessian_vector(
            &input,
            &[1.0, 0.0, 0.0],
            0.0,
            rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC,
            &mut product,
        )
    }
    .unwrap_err();
    assert_eq!(err.status(), rgpot_status_t::RGPOT_INTERNAL_ERROR);
    let after_fatal = observer.native_calls();

    let err_again = unsafe {
        potential.as_handle().hessian_vector(
            &input,
            &[1.0, 0.0, 0.0],
            0.0,
            rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC,
            &mut product,
        )
    }
    .unwrap_err();
    assert_eq!(err_again.status(), rgpot_status_t::RGPOT_INTERNAL_ERROR);
    let mut output = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    assert_eq!(
        unsafe { potential.calculate(&input, &mut output) },
        rgpot_status_t::RGPOT_INTERNAL_ERROR
    );
    assert_eq!(observer.native_calls(), after_fatal);

    unsafe {
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
    drop(potential);
    assert_eq!(observer.destroyed(), before_destroy + 1);
}

#[test]
fn a_failure_without_last_error_names_the_code() {
    let path = build("harmonic_nolast", &["NO_LAST_ERROR"]);
    let mut engine = load(&path);
    let error = engine
        .force(&EngineRequest {
            positions: &[0.0; 3],
            atomic_numbers: &[0],
            box_matrix: &CELL,
        })
        .unwrap_err();
    assert_eq!(
        error,
        EngineError::Recoverable {
            code: 1,
            message: "no rgpot_engine_last_error; code 1".into()
        }
    );
}

#[test]
fn an_abi_major_other_than_the_loader_is_refused() {
    let path = build("harmonic_abi2", &["FIXTURE_ABI=2"]);
    let error = unsafe { EnginePotential::load(&path, &config()) }
        .err()
        .expect("ABI must be refused");
    assert_eq!(
        error,
        EngineError::Abi {
            expected: ENGINE_ABI_VERSION,
            found: 2
        }
    );
}

#[test]
fn abi_stamp_serves_equal_and_older_plugin_minors() {
    let host = RgpotEngineAbiStamp::HOST;
    assert!(host.serves(host));

    let older = RgpotEngineAbiStamp {
        abi_minor: host.abi_minor - 1,
        ..host
    };
    assert!(older.serves(host));

    let newer = RgpotEngineAbiStamp {
        abi_minor: host.abi_minor + 1,
        ..host
    };
    assert!(!newer.serves(host));
}

#[test]
fn an_incompatible_abi_stamp_is_refused_and_names_both_stamps() {
    let newer_minor = format!("FIXTURE_ABI_MINOR={}", ENGINE_ABI_MINOR + 1);
    for (name, define, found) in [
        (
            "harmonic_layout2",
            "FIXTURE_ABI_LAYOUT=2",
            RgpotEngineAbiStamp {
                abi_major: ENGINE_ABI_VERSION as u16,
                abi_minor: ENGINE_ABI_MINOR,
                layout_revision: 2,
            },
        ),
        (
            "harmonic_layout0",
            "FIXTURE_ABI_LAYOUT=0",
            RgpotEngineAbiStamp {
                abi_major: ENGINE_ABI_VERSION as u16,
                abi_minor: ENGINE_ABI_MINOR,
                layout_revision: 0,
            },
        ),
        (
            "harmonic_stamp_newer",
            newer_minor.as_str(),
            RgpotEngineAbiStamp {
                abi_major: ENGINE_ABI_VERSION as u16,
                abi_minor: ENGINE_ABI_MINOR + 1,
                layout_revision: ENGINE_ABI_LAYOUT_REVISION,
            },
        ),
        (
            "harmonic_stamp_major2",
            "FIXTURE_STAMP_MAJOR=2",
            RgpotEngineAbiStamp {
                abi_major: 2,
                abi_minor: ENGINE_ABI_MINOR,
                layout_revision: ENGINE_ABI_LAYOUT_REVISION,
            },
        ),
    ] {
        let path = build(name, &[define]);
        let error = unsafe { EnginePotential::load(&path, &config()) }
            .err()
            .expect("incompatible ABI stamp must be refused");
        assert_eq!(
            error,
            EngineError::AbiStamp {
                expected: RgpotEngineAbiStamp::HOST,
                found,
            }
        );
        let text = error.to_string();
        assert!(
            text.contains(&RgpotEngineAbiStamp::HOST.to_string()),
            "{text}"
        );
        assert!(text.contains(&found.to_string()), "{text}");
    }
}

#[test]
fn a_missing_abi_stamp_symbol_is_served_as_minor_zero() {
    let path = build("harmonic_nostamp", &["NO_STAMP"]);
    let mut engine = load(&path);
    assert_eq!(
        engine.abi_stamp(),
        RgpotEngineAbiStamp {
            abi_major: ENGINE_ABI_VERSION as u16,
            abi_minor: 0,
            layout_revision: 1,
        }
    );
    let answer = engine
        .force(&EngineRequest {
            positions: &[1.0, 0.0, 0.0],
            atomic_numbers: &[1],
            box_matrix: &CELL,
        })
        .expect("force on minor-0 engine");
    assert!(within_roundoff(answer.energy, 0.5 * SPRING, 3));
}

#[test]
fn minor_4_plugin_is_served_without_component_exports_minor_5_gets_them_and_newer_is_rejected() {
    assert_eq!(ENGINE_ABI_MINOR, 5);

    // A minor-4 plugin omits rgpot_engine_last_components and
    // rgpot_engine_component_name; the host serves it, resolves those symbols
    // as absent, and evaluates force and HVP normally.
    let minor4_path = build("harmonic_stamp_minor4", &["FIXTURE_ABI_MINOR=4"]);
    let mut minor4 = load(&minor4_path);
    assert_eq!(
        minor4.abi_stamp(),
        RgpotEngineAbiStamp {
            abi_major: ENGINE_ABI_VERSION as u16,
            abi_minor: 4,
            layout_revision: ENGINE_ABI_LAYOUT_REVISION,
        }
    );
    assert!(minor4.symbols().last_components.is_none());
    assert!(minor4.symbols().component_name.is_none());
    assert_eq!(
        minor4.last_components().unwrap_err(),
        EngineError::Unsupported {
            operation: "rgpot_engine_last_components",
            message: "symbol is not exported".into(),
        }
    );
    assert_eq!(
        minor4.component_name(0).unwrap_err(),
        EngineError::Unsupported {
            operation: "rgpot_engine_component_name",
            message: "symbol is not exported".into(),
        }
    );
    let req = EngineRequest {
        positions: &[1.0, -2.0, 0.5],
        atomic_numbers: &[1],
        box_matrix: &CELL,
    };
    let answer4 = minor4.force(&req).expect("force on minor-4 plugin");
    let sum: f64 = req.positions.iter().map(|x| x * x).sum();
    assert!(within_roundoff(answer4.energy, 0.5 * SPRING * sum, 9));
    let hv4 = minor4
        .hvp(&req, &[1.0, 0.0, -1.0])
        .expect("hvp on minor-4 plugin");
    assert!(all_within_roundoff(&hv4, &[SPRING, 0.0, -SPRING]));

    // A minor-5 plugin exports both component symbols and the host resolves
    // and calls them.
    let minor5_path = build("harmonic_stamp_minor5", &["FIXTURE_ABI_MINOR=5"]);
    let mut minor5 = load(&minor5_path);
    assert_eq!(minor5.abi_stamp(), RgpotEngineAbiStamp::HOST);
    assert!(minor5.symbols().last_components.is_some());
    assert!(minor5.symbols().component_name.is_some());
    assert_eq!(minor5.component_name(0).expect("component 0"), "harmonic");
    assert_eq!(
        minor5.component_name(1).expect("out-of-range component"),
        ""
    );
    assert_eq!(
        minor5.last_components().expect("components before force"),
        vec![RgpotEngineComponent {
            weight: 1.0,
            energy: 0.0,
            stress: [0.0; 9],
            enabled: 1,
            evaluated: 0,
            has_stress: 0,
        }]
    );
    let answer5 = minor5.force(&req).expect("force on minor-5 plugin");
    assert_eq!(
        minor5.last_components().expect("components after force"),
        vec![RgpotEngineComponent {
            weight: 1.0,
            energy: answer5.energy,
            stress: [0.0; 9],
            enabled: 1,
            evaluated: 1,
            has_stress: 0,
        }]
    );

    // A plugin whose minor is newer than the host's is rejected.
    let newer_define = format!("FIXTURE_ABI_MINOR={}", ENGINE_ABI_MINOR + 1);
    let newer_path = build("harmonic_stamp_minor_too_new", &[&newer_define]);
    let error = unsafe { EnginePotential::load(&newer_path, &config()) }
        .err()
        .expect("plugin newer than the host minor must be refused");
    assert_eq!(
        error,
        EngineError::AbiStamp {
            expected: RgpotEngineAbiStamp::HOST,
            found: RgpotEngineAbiStamp {
                abi_major: ENGINE_ABI_VERSION as u16,
                abi_minor: ENGINE_ABI_MINOR + 1,
                layout_revision: ENGINE_ABI_LAYOUT_REVISION,
            },
        }
    );
}

#[test]
fn a_missing_required_symbol_is_named() {
    let path = build("harmonic_noforce", &["NO_FORCE", "NO_BATCH"]);
    let error = unsafe { EnginePotential::load(&path, &config()) }
        .err()
        .expect("force is required");
    assert_eq!(error, EngineError::Missing("rgpot_engine_force".into()));
}

#[test]
fn a_request_that_does_not_fit_is_refused_before_the_engine() {
    let path = build("harmonic_shape", &[]);
    let mut engine = load(&path);
    let error = engine
        .force(&EngineRequest {
            positions: &[0.0; 4],
            atomic_numbers: &[1, 1],
            box_matrix: &CELL,
        })
        .unwrap_err();
    assert!(matches!(error, EngineError::Invalid(_)), "{error}");
    let empty = engine
        .force(&EngineRequest {
            positions: &[],
            atomic_numbers: &[],
            box_matrix: &CELL,
        })
        .unwrap_err();
    assert!(matches!(empty, EngineError::Invalid(_)), "{empty}");
}

#[test]
fn dropping_the_instance_destroys_the_engine() {
    let path = build("harmonic_drop", &[]);
    let observer = FixtureObserver::open(&path);
    let before = observer.destroyed();
    drop(load(&path));
    assert_eq!(observer.destroyed(), before + 1);
}

#[cfg(target_os = "linux")]
#[test]
fn dropping_the_instance_unloads_the_library() {
    let path = build("harmonic_unload", &[]);
    assert!(!library_is_mapped(&path));

    for iteration in 0..200 {
        let engine = load(&path);
        assert!(
            library_is_mapped(&path),
            "fixture is not mapped while iteration {iteration} is alive"
        );
        drop(engine);
        assert!(
            !library_is_mapped(&path),
            "fixture remains mapped after iteration {iteration}"
        );
    }
}

#[cfg(target_os = "linux")]
#[test]
fn load_failures_unload_the_library() {
    let failures = [
        (
            build("harmonic_unload_missing", &["NO_FORCE", "NO_BATCH"]),
            config().to_vec(),
        ),
        (
            build("harmonic_unload_abi", &["FIXTURE_ABI=2"]),
            config().to_vec(),
        ),
        (
            build("harmonic_unload_stamp", &["FIXTURE_ABI_LAYOUT=2"]),
            config().to_vec(),
        ),
        (build("harmonic_unload_create", &[]), vec![1, 2, 3]),
    ];

    for (path, config) in failures {
        assert!(!library_is_mapped(&path));
        unsafe { EnginePotential::load(&path, &config) }
            .err()
            .expect("fixture must fail to load");
        assert!(
            !library_is_mapped(&path),
            "{} remains mapped after load failed",
            path.display()
        );
    }
}

#[test]
fn the_potential_handle_evaluates_and_destroys_the_engine() {
    let path = build("harmonic_handle", &[]);
    let observer = FixtureObserver::open(&path);
    let before = observer.destroyed();
    let potential = load(&path).into_potential();
    assert!(potential.as_handle().has_hessian_vector_callback());
    let mut positions = vec![0.5, -1.0, 2.0, 0.0, 3.0, -0.25];
    let mut numbers = vec![1, 8];
    let mut cell = CELL;
    let input = rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 2, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), 2) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
    };
    let mut output = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    let status = unsafe { potential.calculate(&input, &mut output) };
    assert_eq!(status, rgpot_status_t::RGPOT_SUCCESS);
    let sum: f64 = positions.iter().map(|x| x * x).sum();
    assert!(within_roundoff(
        output.energy,
        0.5 * SPRING * sum,
        3 * positions.len()
    ));
    let forces =
        unsafe { std::slice::from_raw_parts(rgpot_tensor_data(output.forces).cast::<f64>(), 6) };
    let expected: Vec<f64> = positions.iter().map(|x| -SPRING * x).collect();
    assert!(all_within_roundoff(forces, &expected));
    let direction = [1.0, -0.5, 0.25, -2.0, 0.0, 4.0];
    let mut product = [0.0; 6];
    unsafe {
        potential
            .as_handle()
            .hessian_vector(
                &input,
                &direction,
                0.0,
                rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC,
                &mut product,
            )
            .expect("analytic HVP through the potential handle");
    }
    let expected_product: Vec<f64> = direction.iter().map(|value| SPRING * value).collect();
    assert!(all_within_roundoff(&product, &expected_product));
    let counts = potential.as_handle().hvp_counts();
    assert_eq!(counts.force_calls, 0);
    assert_eq!(counts.analytic, 1);
    unsafe {
        rgpot_tensor_free(output.forces);
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
    drop(potential);
    assert_eq!(observer.destroyed(), before + 1);
}

#[test]
fn the_potential_handle_reports_an_engine_failure_as_a_status() {
    let path = build("harmonic_status", &[]);
    let potential = load(&path).into_potential();
    let mut positions = vec![0.0; 3];
    let mut numbers = vec![0];
    let mut cell = CELL;
    let input = rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 1, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), 1) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
    };
    let mut output = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    let status = unsafe { potential.calculate(&input, &mut output) };
    assert_eq!(status, rgpot_status_t::RGPOT_INTERNAL_ERROR);
    assert!(output.forces.is_null());
    unsafe {
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
}

#[test]
fn unaligned_config_slice_is_word_aligned_for_create() {
    let path = build("harmonic_align", &[]);
    #[repr(align(8))]
    struct Padded([u8; 16]);
    let mut buf = Padded([0; 16]);
    buf.0[1..9].copy_from_slice(&config());
    let unaligned = &buf.0[1..9];
    assert_ne!(unaligned.as_ptr() as usize % 8, 0);
    let mut engine = unsafe { EnginePotential::load(&path, unaligned) }.expect("aligned copy");
    let answer = engine
        .force(&EngineRequest {
            positions: &[1.0, 0.0, 0.0],
            atomic_numbers: &[1],
            box_matrix: &CELL,
        })
        .expect("force");
    assert!(within_roundoff(answer.energy, 0.5 * SPRING, 3));
}

#[test]
fn hvp_returns_the_harmonic_hessian_product() {
    let path = build("harmonic_hvp", &[]);
    let mut engine = load(&path);
    let caps = engine.caps().expect("caps");
    assert!(caps.analytic_hvp);
    let positions = [0.5, -1.0, 2.0, 0.0, 3.0, -0.25];
    let direction = [1.0, -0.5, 0.25, -2.0, 0.0, 4.0];
    let hv = engine
        .hvp(
            &EngineRequest {
                positions: &positions,
                atomic_numbers: &[1, 8],
                box_matrix: &CELL,
            },
            &direction,
        )
        .expect("hvp");
    let expected: Vec<f64> = direction.iter().map(|d| SPRING * d).collect();
    assert!(all_within_roundoff(&hv, &expected));
}

#[test]
fn hvp_without_the_symbol_or_capability_is_unsupported() {
    let no_sym = build("harmonic_nohvp", &["NO_HVP"]);
    let mut engine = load(&no_sym);
    assert!(!engine.caps().expect("caps").analytic_hvp);
    let err = engine
        .hvp(
            &EngineRequest {
                positions: &[0.0; 3],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            &[1.0, 0.0, 0.0],
        )
        .unwrap_err();
    assert!(matches!(
        err,
        EngineError::Unsupported {
            operation: "rgpot_engine_hvp",
            ..
        }
    ));
    assert_eq!(err.status(), rgpot_status_t::RGPOT_INTERNAL_ERROR);
    let potential = load(&no_sym).into_potential();
    assert!(!potential.as_handle().has_hessian_vector_callback());

    let no_cap = build("harmonic_nohvpcap", &["NO_HVP_CAP"]);
    let mut engine = load(&no_cap);
    assert!(!engine.caps().expect("caps").analytic_hvp);
    let err = engine
        .hvp(
            &EngineRequest {
                positions: &[0.0; 3],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            &[1.0, 0.0, 0.0],
        )
        .unwrap_err();
    assert!(matches!(
        err,
        EngineError::Unsupported {
            operation: "rgpot_engine_hvp",
            ..
        }
    ));
    let potential = load(&no_cap).into_potential();
    assert!(!potential.as_handle().has_hessian_vector_callback());

    let no_caps = build("harmonic_nocaps", &["NO_CAPS"]);
    let mut engine = load(&no_caps);
    assert_eq!(
        engine.caps().unwrap_err(),
        EngineError::Unsupported {
            operation: "rgpot_engine_caps",
            message: "symbol is not exported".into(),
        }
    );
    let err = engine
        .hvp(
            &EngineRequest {
                positions: &[0.0; 3],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            &[1.0, 0.0, 0.0],
        )
        .unwrap_err();
    assert!(matches!(
        err,
        EngineError::Unsupported {
            operation: "rgpot_engine_hvp",
            ..
        }
    ));
    let potential = load(&no_caps).into_potential();
    assert!(!potential.as_handle().has_hessian_vector_callback());
}

#[test]
fn unsupported_code_keeps_the_engine_message_and_matches_the_header() {
    let path = build("harmonic_hvpunsupported", &["HVP_UNSUPPORTED"]);
    assert_eq!(unsupported_code(&path), ENGINE_ERR_UNSUPPORTED);
    let mut engine = load(&path);
    let err = engine
        .hvp(
            &EngineRequest {
                positions: &[0.0; 3],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            &[1.0, 0.0, 0.0],
        )
        .unwrap_err();
    assert_eq!(
        err,
        EngineError::Unsupported {
            operation: "rgpot_engine_hvp",
            message: "harmonic engine built without analytic HVP".into(),
        }
    );
}

#[test]
fn hvp_maps_recoverable_and_fatal_codes() {
    let path = build("harmonic_hvpcodes", &[]);
    let mut engine = load(&path);
    let positions = [0.0; 3];
    let recoverable = EngineRequest {
        positions: &positions,
        atomic_numbers: &[0],
        box_matrix: &CELL,
    };
    assert_eq!(
        engine.hvp(&recoverable, &[1.0, 0.0, 0.0]).unwrap_err(),
        EngineError::Recoverable {
            code: 1,
            message: "atom 0 has atomic number 0".into(),
        }
    );
    let fatal = EngineRequest {
        positions: &positions,
        atomic_numbers: &[-1],
        box_matrix: &CELL,
    };
    assert_eq!(
        engine.hvp(&fatal, &[1.0, 0.0, 0.0]).unwrap_err(),
        EngineError::Fatal {
            code: -1,
            message: "atom 0 has a negative atomic number".into(),
        }
    );
}

#[test]
fn hvp_rejects_a_non_finite_product() {
    let path = build("harmonic_hvpnan", &["HVP_NONFINITE"]);
    let mut engine = load(&path);
    let err = engine
        .hvp(
            &EngineRequest {
                positions: &[0.0; 3],
                atomic_numbers: &[1],
                box_matrix: &CELL,
            },
            &[1.0, 0.0, 0.0],
        )
        .unwrap_err();
    assert_eq!(
        err,
        EngineError::Invalid("the engine Hessian-vector product at element 1 is not finite".into())
    );
}

#[test]
fn hvp_rejects_mismatched_direction_length() {
    let path = build("harmonic_hvplen", &[]);
    let mut engine = load(&path);
    let err = engine
        .hvp(
            &EngineRequest {
                positions: &[0.0; 6],
                atomic_numbers: &[1, 8],
                box_matrix: &CELL,
            },
            &[1.0, 2.0, 3.0],
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Invalid(_)), "{err}");
    assert_eq!(err.status(), rgpot_status_t::RGPOT_INVALID_PARAMETER);
}

#[test]
fn hvp_c_api_rejects_overlapping_direction_and_output() {
    let path = build("harmonic_hvpalias", &[]);
    let potential = load(&path).into_potential();
    let mut positions = vec![0.5, -1.0, 2.0];
    let mut numbers = vec![1];
    let mut cell = CELL;
    let input = rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 1, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(numbers.as_mut_ptr(), 1) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
    };
    let initial = [1.0, -0.5, 0.25];
    let mut aliased = initial;
    let status = unsafe {
        rgpot_core::c_api::hvp::rgpot_potential_hvp(
            potential.as_handle() as *const _,
            &input,
            aliased.as_ptr(),
            0.0,
            rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC as c_int,
            aliased.as_mut_ptr(),
        )
    };
    assert_eq!(status, rgpot_status_t::RGPOT_INVALID_PARAMETER);
    assert_eq!(aliased, initial);
    unsafe {
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
}

/// `RgpotEngineCaps` holds the fields of `engine_c_abi.h` in the same order,
/// each four bytes wide, so every offset an engine writes is the one the
/// loader reads.
#[test]
fn engine_caps_layout_is_the_c_header_layout() {
    use std::mem::{offset_of, size_of};

    let header = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../CppCore/rgpot/engine_c_abi.h"),
    )
    .expect("read engine_c_abi.h");
    let end = header
        .find("} RgpotEngineCaps;")
        .expect("RgpotEngineCaps in engine_c_abi.h");
    let open = "typedef struct {";
    let start = header[..end].rfind(open).expect("start of RgpotEngineCaps") + open.len();
    let declared: Vec<&str> = header[start..end]
        .lines()
        .filter_map(|line| {
            let mut words = line.split_once(';')?.0.split_whitespace();
            let (kind, name) = (words.next()?, words.next()?);
            assert!(matches!(kind, "int" | "unsigned"), "{name} is a {kind}");
            Some(name)
        })
        .collect();

    let loader = [
        ("size", offset_of!(RgpotEngineCaps, size)),
        ("reentrancy", offset_of!(RgpotEngineCaps, reentrancy)),
        (
            "per_image_instances",
            offset_of!(RgpotEngineCaps, per_image_instances),
        ),
        ("batched", offset_of!(RgpotEngineCaps, batched)),
        ("periodic", offset_of!(RgpotEngineCaps, periodic)),
        ("stress", offset_of!(RgpotEngineCaps, stress)),
        (
            "group_collective",
            offset_of!(RgpotEngineCaps, group_collective),
        ),
        (
            "orbital_selection",
            offset_of!(RgpotEngineCaps, orbital_selection),
        ),
        ("orbital_copy", offset_of!(RgpotEngineCaps, orbital_copy)),
        ("analytic_hvp", offset_of!(RgpotEngineCaps, analytic_hvp)),
    ];
    let names: Vec<&str> = loader.iter().map(|(name, _)| *name).collect();
    assert_eq!(names, declared, "field order differs from engine_c_abi.h");
    for (index, (name, offset)) in loader.iter().enumerate() {
        assert_eq!(*offset, 4 * index, "offset of {name}");
    }
    assert_eq!(size_of::<RgpotEngineCaps>(), 4 * declared.len());
}

#[test]
fn coincident_pair_reports_recoverable_eval_error_and_internal_status() {
    let path = build("harmonic_coincident", &["REJECT_COINCIDENT"]);
    let mut engine = load(&path);
    let coincident = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    let numbers = [1, 1];
    let direction = [0.5, -0.25, 0.75, -0.5, 0.25, -0.75];
    let expected = EngineError::Recoverable {
        code: 2,
        message: "coincident atoms".into(),
    };

    assert_eq!(
        engine
            .force(&EngineRequest {
                positions: &coincident,
                atomic_numbers: &numbers,
                box_matrix: &CELL,
            })
            .unwrap_err(),
        expected
    );
    assert_eq!(
        engine
            .force_batch(&numbers, &coincident, &CELL)
            .unwrap_err(),
        expected
    );
    assert_eq!(
        engine
            .hvp(
                &EngineRequest {
                    positions: &coincident,
                    atomic_numbers: &numbers,
                    box_matrix: &CELL,
                },
                &direction,
            )
            .unwrap_err(),
        expected
    );

    let potential = engine.into_potential();
    let mut positions = coincident.to_vec();
    let mut atomic_numbers = numbers.to_vec();
    let mut cell = CELL;
    let input = rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), 2, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(atomic_numbers.as_mut_ptr(), 2) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
    };
    let mut output = rgpot_force_out_t {
        forces: std::ptr::null_mut(),
        energy: 0.0,
        variance: 0.0,
    };
    assert_eq!(
        unsafe { potential.calculate(&input, &mut output) },
        rgpot_status_t::RGPOT_INTERNAL_ERROR
    );
    assert!(output.forces.is_null());

    assert_eq!(
        unsafe {
            rgpot_core::c_api::hvp::rgpot_potential_hvp_centre(
                potential.as_handle() as *const _,
                &input,
                &mut output,
            )
        },
        rgpot_status_t::RGPOT_INTERNAL_ERROR
    );
    let mut hv_out = [0.0; 6];
    for scheme in [
        rgpot_hvp_scheme_t::RGPOT_HVP_AUTO,
        rgpot_hvp_scheme_t::RGPOT_HVP_ANALYTIC,
        rgpot_hvp_scheme_t::RGPOT_HVP_FORWARD,
    ] {
        assert_eq!(
            unsafe {
                rgpot_core::c_api::hvp::rgpot_potential_hvp(
                    potential.as_handle() as *const _,
                    &input,
                    direction.as_ptr(),
                    0.0,
                    scheme as c_int,
                    hv_out.as_mut_ptr(),
                )
            },
            rgpot_status_t::RGPOT_INTERNAL_ERROR
        );
    }

    unsafe {
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }

    let mut separated = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0];
    let sep_input = rgpot_force_input_t {
        positions: unsafe { rgpot_tensor_cpu_f64_2d(separated.as_mut_ptr(), 2, 3) },
        atomic_numbers: unsafe { rgpot_tensor_cpu_i32_1d(atomic_numbers.as_mut_ptr(), 2) },
        box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
    };
    assert_eq!(
        unsafe { potential.calculate(&sep_input, &mut output) },
        rgpot_status_t::RGPOT_SUCCESS
    );
    unsafe {
        rgpot_tensor_free(output.forces);
        rgpot_tensor_free(sep_input.positions);
        rgpot_tensor_free(sep_input.atomic_numbers);
        rgpot_tensor_free(sep_input.box_matrix);
    }
}

struct SuccessStatusFailure;

impl CoordinateTransform for SuccessStatusFailure {
    fn transform(
        &mut self,
        _positions: &mut [f64],
        _box_matrix: &mut [f64; 9],
    ) -> Result<(), EngineFailure> {
        Err(EngineFailure {
            status: EngineStatus::Success,
            message: "failure with success status".into(),
        })
    }
}

#[test]
fn transform_failure_with_success_status_is_an_error() {
    let path = build("harmonic_transform_success_status", &[]);
    let mut session = load_session(&path);
    let answer = session.force(
        &EngineRequest {
            positions: &[1.0, 2.0, 3.0],
            atomic_numbers: &[1],
            box_matrix: &CELL,
        },
        Some(&mut SuccessStatusFailure),
    );
    assert!(matches!(answer, Err(EngineError::Invalid(_))), "{answer:?}");
}
