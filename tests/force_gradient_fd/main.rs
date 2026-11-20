//! Finite-difference checks for force and gradient conversions other than
//! the rgpot eindir callbacks. Those callbacks are already compared in
//! `rgpot_force_fd`. Each check below calls the function that publishes
//! the slope, at the same scalene geometries, with the same bound.
//!
//! A missing engine library is named and skipped. A variable that names a
//! missing file is a failure. A present library that resolves nothing fails.

#![allow(clippy::duplicate_mod)]

#[allow(unused_imports)]
#[path = "../../examples/common/force_sign.rs"]
mod force_sign;

#[allow(unused_imports)]
#[path = "../../examples/common/pipe_engine.rs"]
mod pipe_engine;

#[allow(unused_imports)]
#[path = "../../examples/common/rgpot_eindir.rs"]
mod rgpot_eindir;

#[allow(dead_code, unused_imports)]
#[path = "../../examples/h2h_sc.rs"]
mod h2h_sc;

#[allow(dead_code, unused_imports)]
#[path = "../../examples/h2h_covalent.rs"]
mod h2h_covalent;

#[allow(dead_code, unused_imports)]
#[path = "../../examples/globalopt_profile.rs"]
mod globalopt_profile;

#[allow(dead_code, unused_imports)]
#[path = "../../examples/water_hexamer.rs"]
mod water_hexamer;

#[cfg(all(feature = "bank-rpc", feature = "ira"))]
#[allow(dead_code, unused_imports)]
#[path = "../../examples/water_ride_search.rs"]
mod water_ride;

#[cfg(feature = "capi")]
#[allow(dead_code, unused_imports)]
#[path = "../../examples/rgpot_minimize.rs"]
mod rgpot_minimize;

use anneal_core::catalog::lj;
use anneal_core::catalog::molecular::{GFN2_ACCURACY, HARTREE_EV, VACUUM_BOX};
use force_sign::{XTB_CLI_ACCURACY, library_file, uniform_lengths};
use rgpot_eindir::OVERLAP_CALL_LIMIT_A;
use std::path::PathBuf;
use std::process::ExitCode;

const WATER_SPECIES: [i32; 3] = [8, 1, 1];
const WATER: [&[f64]; 3] = [
    &[0.20, 0.15, 0.05, 1.12, 0.28, -0.10, -0.18, 1.02, 0.22],
    &[1.40, -0.30, 0.80, 1.55, 0.62, 0.55, 0.70, -0.55, 1.45],
    &[-1.10, 2.40, -0.50, -0.20, 2.15, -0.30, -1.55, 1.70, 0.10],
];

// The CLI accuracy is 1, a hundred times the engine's 0.01, so its model
// step is wider. The first water's short bond is opened until that step
// fits inside the xtb overlap wall. The other two already do.
const PIPE_WATER: [&[f64]; 3] = [
    &[0.20, 0.15, 0.05, 1.18, 0.28, -0.10, -0.18, 1.02, 0.22],
    &[1.40, -0.30, 0.80, 1.55, 0.62, 0.55, 0.70, -0.55, 1.45],
    &[-1.10, 2.40, -0.50, -0.20, 2.15, -0.30, -1.55, 1.70, 0.10],
];

const CUH2_SPECIES: [i32; 3] = [29, 29, 1];
const CUH2: [&[f64]; 3] = [
    &[0.40, 0.20, 0.10, 2.85, 0.55, 0.30, 1.55, 1.85, 0.95],
    &[5.10, 4.20, 3.40, 7.70, 4.05, 3.90, 6.05, 5.85, 4.55],
    &[8.20, 1.10, 6.40, 10.15, 2.40, 6.10, 9.55, 1.35, 7.90],
];

const LJ: [&[f64]; 3] = [
    &[0.00, 0.00, 0.00, 1.55, 0.20, 0.05, 0.25, 1.70, 0.30],
    &[0.40, -0.20, 0.30, 2.10, 0.15, 0.10, 0.90, 1.85, 0.55],
    &[-0.30, 0.40, 1.20, 1.40, 0.10, 1.00, 0.20, 1.90, 1.70],
];

const TIP4P: [f64; 12] = [
    0.00, 0.00, 0.00, 4.20, 0.60, -0.30, 0.40, -0.25, 0.20, -0.15, 0.35, 0.50,
];

fn main() -> ExitCode {
    let mut failures = Vec::new();
    if let Err(error) = force_sign::the_step_minimizes_the_model_bound() {
        failures.push(error);
    }
    failures.extend(check_lj_catalog());
    failures.extend(check_tip4p_profile());
    failures.extend(check_rgpot_fresh());
    failures.extend(check_gfn2_surface());
    failures.extend(check_petmad());
    failures.extend(check_sutton_chen());
    failures.extend(check_covalent());
    failures.extend(check_quadratic());
    failures.extend(check_pipe());
    failures.extend(check_profile_engine());

    if failures.is_empty() {
        ExitCode::SUCCESS
    } else {
        for failure in &failures {
            eprintln!("FAIL {failure}");
        }
        ExitCode::from(1)
    }
}

fn check_points(
    backend: &str,
    geometries: &[&[f64]],
    precision_of: impl Fn(&[f64], f64) -> f64,
    overlap: Option<f64>,
    mut publish: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>), String>,
) -> Vec<String> {
    let mut failures = Vec::new();
    for (index, geometry) in geometries.iter().enumerate() {
        let published = match publish(geometry) {
            Ok(published) => published,
            Err(error) => {
                failures.push(format!("{backend} geometry {index} {error}"));
                continue;
            }
        };
        let (energy, slope) = published;
        let precision = precision_of(geometry, energy);
        println!(
            "{backend} geometry {index} energy precision {precision:.6e} (reference {energy:.12})"
        );
        let (lengths, step_limits) = match uniform_lengths(geometry, overlap) {
            Ok(limits) => limits,
            Err(error) => {
                failures.push(format!("{backend} geometry {index} {error}"));
                continue;
            }
        };
        match force_sign::compare_slope(
            backend,
            index,
            geometry,
            &slope,
            energy,
            precision,
            &lengths,
            &step_limits,
            |point| publish(point).map(|(energy, _)| energy),
        ) {
            Ok(row) => println!("{row}"),
            Err(error) => failures.push(error),
        }
    }
    if failures.is_empty() && geometries.is_empty() {
        failures.push(format!("{backend}: no geometry was compared"));
    }
    failures
}

fn check_lj_catalog() -> Vec<String> {
    check_points("lj-catalog", &LJ, lj_precision, None, |coordinates| {
        let fresh = lj::fresh_evaluation(coordinates.len() / 3, coordinates)?;
        let slope: Vec<f64> = fresh.forces.into_iter().map(|force| -force).collect();
        Ok((fresh.energy, slope))
    })
}

fn lj_precision(coordinates: &[f64], energy: f64) -> f64 {
    let n = coordinates.len() / 3;
    let mut terms = 0usize;
    let mut absolute = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut r2 = 0.0;
            for axis in 0..3 {
                let delta = coordinates[3 * i + axis] - coordinates[3 * j + axis];
                r2 += delta * delta;
            }
            if r2 > 0.0 {
                let inv2 = 1.0 / r2;
                let inv6 = inv2 * inv2 * inv2;
                let inv12 = inv6 * inv6;
                // The two addends inside `PairKind::LennardJones`.
                terms += 2;
                absolute += (4.0 * inv12).abs() + (4.0 * inv6).abs();
            }
        }
    }
    force_sign::summed_precision(terms, absolute, energy)
}

fn check_tip4p_profile() -> Vec<String> {
    let geometry = TIP4P.as_slice();
    let globalopt_profile::Tip4pPublished {
        energy,
        gradient: slope,
        lengths,
        step_limits,
        terms,
        absolute,
    } = match globalopt_profile::tip4p_published(geometry) {
        Ok(published) => published,
        Err(error) => return vec![error],
    };
    let precision = force_sign::summed_precision(terms, absolute, energy);
    println!(
        "tip4p-profile geometry 0 energy precision {precision:.6e} (forward error of {terms} terms, absolute sum {absolute:.6e}, energy {energy:.12})"
    );
    match force_sign::compare_slope(
        "tip4p-profile",
        0,
        geometry,
        &slope,
        energy,
        precision,
        &lengths,
        &step_limits,
        |point| Ok(globalopt_profile::tip4p_energy_at(point)),
    ) {
        Ok(row) => {
            println!("{row}");
            Vec::new()
        }
        Err(error) => vec![error],
    }
}

fn vacuum_box() -> [f64; 9] {
    [
        VACUUM_BOX, 0.0, 0.0, 0.0, VACUUM_BOX, 0.0, 0.0, 0.0, VACUUM_BOX,
    ]
}

fn check_rgpot_fresh() -> Vec<String> {
    let mut failures = Vec::new();
    let box_ = vacuum_box();
    let precision = force_sign::scc_precision(GFN2_ACCURACY, HARTREE_EV);
    println!(
        "xtb fresh energy precision {precision:.6e} eV from scfconv {:.1e} * accuracy {GFN2_ACCURACY} * hartree {HARTREE_EV}",
        force_sign::XTB_SCF_ENERGY_CHANGE_PER_ACCURACY
    );
    match library_file("RGPOT_XTB_ENGINE", "libxtb_engine.so") {
        Ok(None) => println!(
            "xtb-fresh: skipped, libxtb_engine.so is absent (RGPOT_XTB_ENGINE or ./libxtb_engine.so)"
        ),
        Ok(Some(path)) => {
            println!("xtb-fresh library {}", path.display());
            let mut pot = rgpot_eindir::RgpotObjective::xtb(&WATER_SPECIES, box_);
            failures.extend(check_points(
                "xtb-fresh",
                &WATER,
                |_, _| precision,
                Some(OVERLAP_CALL_LIMIT_A),
                |coordinates| {
                    pot.restart();
                    let fresh = pot.fresh_evaluation(coordinates)?;
                    let slope: Vec<f64> = fresh.forces.into_iter().map(|force| -force).collect();
                    Ok((fresh.energy, slope))
                },
            ));
        }
        Err(error) => failures.push(error),
    }
    match library_file("RGPOT_CUH2_LIBRARY", "librgpot_cuh2.so") {
        Ok(None) => println!(
            "cuh2-fresh: skipped, librgpot_cuh2.so is absent (RGPOT_CUH2_LIBRARY or ./librgpot_cuh2.so)"
        ),
        Ok(Some(path)) => {
            println!("cuh2-fresh library {}", path.display());
            let mut pot = rgpot_eindir::RgpotObjective::cuh2(&CUH2_SPECIES, box_);
            failures.extend(check_points(
                "cuh2-fresh",
                &CUH2,
                |_, energy| force_sign::unit_in_the_last_place(energy),
                None,
                |coordinates| {
                    pot.restart();
                    let fresh = pot.fresh_evaluation(coordinates)?;
                    let slope: Vec<f64> = fresh.forces.into_iter().map(|force| -force).collect();
                    Ok((fresh.energy, slope))
                },
            ));
        }
        Err(error) => failures.push(error),
    }
    failures
}

fn check_gfn2_surface() -> Vec<String> {
    #[cfg(all(feature = "bank-rpc", feature = "ira"))]
    {
        let precision = force_sign::scc_precision(GFN2_ACCURACY, HARTREE_EV);
        match library_file("RGPOT_XTB_ENGINE", "libxtb_engine.so") {
            Ok(None) => {
                println!(
                    "gfn2-surface: skipped, libxtb_engine.so is absent (RGPOT_XTB_ENGINE or ./libxtb_engine.so)"
                );
                Vec::new()
            }
            Ok(Some(path)) => {
                println!("gfn2-surface library {}", path.display());
                check_points(
                    "gfn2-surface",
                    &WATER,
                    |_, _| precision,
                    Some(OVERLAP_CALL_LIMIT_A),
                    |coordinates| {
                        let (energy, slope) =
                            water_ride::gfn2_surface_published(&WATER_SPECIES, coordinates)?;
                        Ok((energy, slope))
                    },
                )
            }
            Err(error) => vec![error],
        }
    }
    #[cfg(not(all(feature = "bank-rpc", feature = "ira")))]
    {
        match library_file("RGPOT_XTB_ENGINE", "libxtb_engine.so") {
            Ok(None) => {
                println!(
                    "gfn2-surface: skipped, WaterSurface needs bank-rpc and ira, and libxtb_engine.so is absent"
                );
                Vec::new()
            }
            Ok(Some(path)) => {
                println!(
                    "gfn2-surface: skipped, {} is present but this build has no bank-rpc and ira",
                    path.display()
                );
                Vec::new()
            }
            Err(error) => vec![error],
        }
    }
}

fn check_petmad() -> Vec<String> {
    match library_file("RGPOT_METATOMIC_ENGINE", "libmetatomic_engine.so") {
        Ok(None) => {
            println!(
                "petmad: skipped, libmetatomic_engine.so is absent (RGPOT_METATOMIC_ENGINE or ./libmetatomic_engine.so)"
            );
            Vec::new()
        }
        Ok(Some(path)) => {
            println!("petmad library {}", path.display());
            check_points(
                "petmad",
                &WATER,
                |_, energy| force_sign::unit_in_the_last_place(energy),
                None,
                |coordinates| water_hexamer::petmad_published(&WATER_SPECIES, coordinates),
            )
        }
        Err(error) => vec![error],
    }
}

fn named_library_missing(variables: &[&str]) -> Result<bool, Vec<String>> {
    let mut missing = Vec::new();
    let mut present = false;
    for variable in variables {
        if let Some(raw) = std::env::var_os(variable)
            && !raw.is_empty()
        {
            let path = PathBuf::from(&raw);
            if path.is_file() {
                present = true;
            } else {
                missing.push(format!("{variable}={}", path.display()));
            }
        }
    }
    if present {
        return Ok(false);
    }
    if missing.is_empty() {
        Ok(true)
    } else {
        Err(missing.into_iter().collect())
    }
}

fn check_sutton_chen() -> Vec<String> {
    let local = [
        "librgpot.so",
        "./librgpot.so",
        "librgpot.dylib",
        "./librgpot.dylib",
    ]
    .iter()
    .any(|name| PathBuf::from(name).is_file());
    match h2h_sc::sutton_chen_published(WATER[0]) {
        Ok(_) => check_points(
            "sutton-chen",
            &WATER,
            |_, energy| force_sign::unit_in_the_last_place(energy),
            None,
            h2h_sc::sutton_chen_published,
        ),
        Err(error) => {
            match named_library_missing(&["RGPOT_SC_LIB", "RGPOT_LIBRARY", "RGPOT_RGPOT_ENGINE"]) {
                Ok(true) if !local => {
                    println!(
                        "sutton-chen: skipped, librgpot.so is absent (RGPOT_SC_LIB, RGPOT_LIBRARY, RGPOT_RGPOT_ENGINE, or ./librgpot.so). {error}"
                    );
                    Vec::new()
                }
                Ok(_) => vec![format!("sutton-chen library is present but {error}")],
                Err(missing) => vec![format!(
                    "sutton-chen: {} and no loadable librgpot.so ({error})",
                    missing.join(", ")
                )],
            }
        }
    }
}

fn check_covalent() -> Vec<String> {
    let path = match library_file("RGPOT_COVALENT_LIB", "librgpot_covalent.so") {
        Ok(Some(path)) => path,
        Ok(None) => {
            let fallback = PathBuf::from("/work/covalent-h2h/lib/librgpot_covalent.so");
            if fallback.is_file() {
                fallback
            } else {
                println!(
                    "covalent: skipped, librgpot_covalent.so is absent (RGPOT_COVALENT_LIB, ./librgpot_covalent.so, or /work/covalent-h2h/lib/librgpot_covalent.so)"
                );
                return Vec::new();
            }
        }
        Err(error) => return vec![error],
    };
    println!("covalent library {}", path.display());
    let path = path.to_string_lossy().into_owned();
    check_points(
        "covalent",
        &WATER,
        |_, energy| force_sign::unit_in_the_last_place(energy),
        None,
        |coordinates| h2h_covalent::covalent_published(&path, coordinates),
    )
}

fn check_quadratic() -> Vec<String> {
    #[cfg(feature = "capi")]
    {
        check_points(
            "quadratic-rgpot",
            &WATER,
            |_, energy| force_sign::unit_in_the_last_place(energy),
            None,
            |coordinates| Ok(rgpot_minimize::quadratic_published(coordinates)),
        )
    }
    #[cfg(not(feature = "capi"))]
    {
        println!("quadratic-rgpot: skipped, this build has no capi feature");
        Vec::new()
    }
}

fn xtb_executable() -> Result<Option<PathBuf>, String> {
    if let Some(raw) = std::env::var_os("XTB_EXE")
        && !raw.is_empty()
    {
        let path = PathBuf::from(&raw);
        if path.is_file() {
            return Ok(Some(path));
        }
        return Err(format!(
            "XTB_EXE names {} which is not a file",
            path.display()
        ));
    }
    if let Some(path) = std::env::var_os("PATH") {
        for directory in std::env::split_paths(&path) {
            let candidate = directory.join("xtb");
            if candidate.is_file() {
                return Ok(Some(candidate));
            }
        }
    }
    Ok(None)
}

fn check_pipe() -> Vec<String> {
    let engine = std::env::var("ASE_ENGINE").ok();
    if let Some(engine) = engine.as_deref()
        && engine != "xtb-cli"
    {
        return vec![format!(
            "pipe: ASE_ENGINE={engine} is set, and this check has no cited energy precision for that engine"
        )];
    }
    let executable = match xtb_executable() {
        Ok(Some(path)) => path,
        Ok(None) => {
            if engine.as_deref() == Some("xtb-cli") {
                return vec!["pipe: ASE_ENGINE=xtb-cli but the xtb executable is absent".into()];
            }
            println!(
                "pipe: skipped, the xtb executable is absent (XTB_EXE or PATH) and ASE_ENGINE is unset"
            );
            return Vec::new();
        }
        Err(error) => return vec![error],
    };
    println!("pipe xtb {}", executable.display());
    let precision = force_sign::scc_precision(XTB_CLI_ACCURACY, HARTREE_EV);
    println!(
        "pipe energy precision {precision:.6e} eV from scfconv {:.1e} * xtb --acc default {XTB_CLI_ACCURACY} * hartree {HARTREE_EV}",
        force_sign::XTB_SCF_ENERGY_CHANGE_PER_ACCURACY
    );
    let symbols = vec!["O".to_string(), "H".to_string(), "H".to_string()];
    let mut pipe = pipe_engine::PipeEngine::start("xtb-cli", symbols, None);
    check_points(
        "pipe-xtb",
        &PIPE_WATER,
        |_, _| precision,
        Some(OVERLAP_CALL_LIMIT_A),
        |coordinates| {
            let point = ndarray::ArrayView1::from(coordinates);
            let (energy, gradient) = pipe
                .eval(point)
                .ok_or_else(|| "the ASE helper returned no energy".to_string())?;
            Ok((energy, gradient.to_vec()))
        },
    )
}

fn profile_library() -> Result<Option<(String, PathBuf)>, String> {
    let mut missing = Vec::new();
    for (variable, prefix) in [
        ("POTENTIAL_LIBRARY", ""),
        ("RGPOT_NWCHEMC_ENGINE", "nwchemc"),
        ("NWCHEMC_LIBRARY", "nwchemc"),
        ("RGPOT_CPMDC_ENGINE", "cpmdc"),
        ("CPMDC_LIBRARY", "cpmdc"),
    ] {
        if let Some(raw) = std::env::var_os(variable)
            && !raw.is_empty()
        {
            let path = PathBuf::from(&raw);
            if path.is_file() {
                let prefix = if prefix.is_empty() {
                    profile_prefix(&path)
                } else {
                    prefix.to_string()
                };
                return Ok(Some((prefix, path)));
            }
            missing.push(format!("{variable}={}", path.display()));
        }
    }
    for (file_name, prefix) in [("libnwchemc.so", "nwchemc"), ("libcpmdc.so", "cpmdc")] {
        let path = PathBuf::from(file_name);
        if path.is_file() {
            return Ok(Some((prefix.to_string(), path)));
        }
    }
    if missing.is_empty() {
        Ok(None)
    } else {
        Err(format!(
            "{} and libnwchemc.so is absent",
            missing.join(", ")
        ))
    }
}

fn profile_prefix(path: &std::path::Path) -> String {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if name.contains("cpmd") {
        "cpmdc".to_string()
    } else {
        "nwchemc".to_string()
    }
}

fn check_profile_engine() -> Vec<String> {
    // `examples/common/profile_engine.rs` calls `rgpot_core::profile`, which
    // is behind rgpot-core's `profile` feature. That feature runs capnpc.
    // This crate does not enable it. The conversion itself is the negation
    // in `optimizer_value_gradient`. A library that is present still has to
    // be named rather than ignored.
    match profile_library() {
        Ok(None) => {
            println!(
                "profile-engine: skipped, libnwchemc.so and libcpmdc.so are absent (POTENTIAL_LIBRARY, RGPOT_NWCHEMC_ENGINE, RGPOT_CPMDC_ENGINE, or ./libnwchemc.so), and rgpot-core profile needs capnpc"
            );
            Vec::new()
        }
        Ok(Some((prefix, path))) => vec![format!(
            "profile-engine: {prefix} library {} is present, but rgpot-core's profile feature is not enabled (it needs capnpc)",
            path.display()
        )],
        Err(error) => vec![error],
    }
}
