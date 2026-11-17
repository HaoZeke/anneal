#![cfg(unix)]

use std::ffi::c_int;
use std::path::{Path, PathBuf};

use rgpot_core::engine::{EngineError, EnginePotential, EngineRequest, ENGINE_ABI_VERSION};
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::{
    rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3, rgpot_tensor_cpu_i32_1d,
    rgpot_tensor_data, rgpot_tensor_free,
};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

const SPRING: f64 = 2.5;
const CELL: [f64; 9] = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];

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

fn destroyed(path: &Path) -> c_int {
    let library = unsafe { libloading::Library::new(path) }.expect("reopen the fixture");
    let count: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"fixture_destroy_count\0") }.expect("count symbol");
    unsafe { count() }
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
    let batch = engine
        .force_batch(&numbers, &positions, &boxes)
        .expect("batch");
    assert_eq!(batch.len(), 2);
    for (system, one) in [first, second].iter().enumerate() {
        let single = engine
            .force(&EngineRequest {
                positions: one,
                atomic_numbers: &numbers,
                box_matrix: &CELL,
            })
            .expect("single");
        assert!(within_roundoff(
            batch[system].energy,
            single.energy,
            3 * one.len()
        ));
        assert!(all_within_roundoff(&batch[system].forces, &single.forces));
    }
}

#[test]
fn batch_without_the_symbol_is_unsupported() {
    let path = build("harmonic_nobatch", &["NO_BATCH"]);
    let mut engine = load(&path);
    let error = engine
        .force_batch(&[1], &[0.0, 0.0, 0.0], &CELL)
        .unwrap_err();
    assert_eq!(error, EngineError::Unsupported("rgpot_engine_force_batch"));
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
    let before = destroyed(&path);
    drop(load(&path));
    assert_eq!(destroyed(&path), before + 1);
}

#[test]
fn the_potential_handle_evaluates_and_destroys_the_engine() {
    let path = build("harmonic_handle", &[]);
    let before = destroyed(&path);
    let potential = load(&path).into_potential();
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
    unsafe {
        rgpot_tensor_free(output.forces);
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
    drop(potential);
    assert_eq!(destroyed(&path), before + 1);
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
