#![cfg(unix)]

use std::ffi::c_int;
use std::path::{Path, PathBuf};

use rgpot_core::engine::{
    EngineError, EnginePotential, EngineRequest, RgpotEngineAbiStamp, ENGINE_ABI_LAYOUT_REVISION,
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

fn native_calls(path: &Path) -> c_int {
    let library = unsafe { libloading::Library::new(path) }.expect("reopen the fixture");
    let count: libloading::Symbol<unsafe extern "C" fn() -> c_int> =
        unsafe { library.get(b"fixture_native_call_count\0") }.expect("native call count symbol");
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
    let before_destroy = destroyed(&path);
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
    let after_fatal = native_calls(&path);
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
    assert_eq!(native_calls(&path), after_fatal);

    let potential = engine.into_potential();
    assert!(!potential.as_handle().has_hessian_vector_callback());
    assert_eq!(native_calls(&path), after_fatal);
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
    assert_eq!(native_calls(&path), after_fatal);
    unsafe {
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
    drop(potential);
    assert_eq!(destroyed(&path), before_destroy + 1);
}

#[test]
fn fatal_hvp_poisons_a_live_potential_handle_without_reentering_native_code() {
    let path = build("harmonic_fatal_hvp_handle", &["FATAL_ONCE"]);
    let before_destroy = destroyed(&path);
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
    let after_fatal = native_calls(&path);

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
    assert_eq!(native_calls(&path), after_fatal);

    unsafe {
        rgpot_tensor_free(input.positions);
        rgpot_tensor_free(input.atomic_numbers);
        rgpot_tensor_free(input.box_matrix);
    }
    drop(potential);
    assert_eq!(destroyed(&path), before_destroy + 1);
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
fn an_incompatible_abi_stamp_is_refused_and_names_both_stamps() {
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
            "harmonic_minor2",
            "FIXTURE_ABI_MINOR=2",
            RgpotEngineAbiStamp {
                abi_major: ENGINE_ABI_VERSION as u16,
                abi_minor: 2,
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
fn a_missing_abi_stamp_symbol_is_refused_as_minor_zero() {
    let path = build("harmonic_nostamp", &["NO_STAMP"]);
    let error = unsafe { EnginePotential::load(&path, &config()) }
        .err()
        .expect("missing rgpot_engine_abi_stamp must be refused for minor 3 host");
    let found = RgpotEngineAbiStamp {
        abi_major: ENGINE_ABI_VERSION as u16,
        abi_minor: 0,
        layout_revision: 1,
    };
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

#[test]
fn a_compatible_abi_stamp_loads_at_equal_or_higher_minor() {
    let equal = build("harmonic_stamp_equal", &[]);
    let engine = load(&equal);
    assert_eq!(engine.abi_stamp(), RgpotEngineAbiStamp::HOST);

    let newer = build("harmonic_stamp_minor4", &["FIXTURE_ABI_MINOR=4"]);
    let mut engine = load(&newer);
    assert_eq!(
        engine.abi_stamp(),
        RgpotEngineAbiStamp {
            abi_major: ENGINE_ABI_VERSION as u16,
            abi_minor: 4,
            layout_revision: ENGINE_ABI_LAYOUT_REVISION,
        }
    );
    let answer = engine
        .force(&EngineRequest {
            positions: &[1.0, 0.0, 0.0],
            atomic_numbers: &[1],
            box_matrix: &CELL,
        })
        .expect("force on compatible minor-4 engine");
    assert!(within_roundoff(answer.energy, 0.5 * SPRING, 3));
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
