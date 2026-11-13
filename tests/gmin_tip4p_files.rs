//! Rigid TIP4P angle-axis files against GMIN's own energy.
//!
//! The recorded energies are GMIN `CHECKD 0` on `examples/h2h_gmin_tip4p.data`
//! (`NTIP 4`). GMIN's Coulomb prefactor is 1389.354848 and [`GminTip4p`]'s is
//! 1389.352660667, which moves these energies by under 1e-4 kJ/mol.
//!
//! `tip4p15-optbench-0.aa` is `500_tip4p15/0.aa` from optbench's
//! `website/globalopt/tip4p15/500_tip4p15.tar.gz` (sha256 2fc2f69894e58908...),
//! the GMIN `coords` of start 0. `tip4p15-seed2000.legacy-interleaved` is
//! `h2h_seed tip4p 15 0 1 2000`'s start as anneal 0.12.0's
//! `interleaved_gmin_from_block` wrote it.
//!
//! Set `ANNEAL_GMIN` to a GMIN binary to rerun `CHECKD 0` on every file here.

use anneal_core::potentials::{
    GminAngleAxisFile, GminTip4p, block_from_gmin_file, block_from_interleaved_gmin,
    gmin_coords_text, interleaved_gmin_from_block, random_tip4p_cluster,
};
use eindir_core::Objective;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;

const OPTBENCH_START: &str = include_str!("fixtures/tip4p15-optbench-0.aa");
const OPTBENCH_CHECKD: f64 = 155.557_211_980_144_59;
const LEGACY_START: &str = include_str!("fixtures/tip4p15-seed2000.legacy-interleaved");
const SEED_2000_CHECKD: f64 = -13.192_905_420_423_074;
const GMIN_TOLERANCE: f64 = 1.0e-3;

fn energy(block: &[f64]) -> f64 {
    GminTip4p::new(block.len() / 6).eval(Array1::from_vec(block.to_vec()).view())
}

fn h2h_seed_start(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(3));
    random_tip4p_cluster(n, &mut rng).to_vec()
}

fn rows(values: &[f64], width: usize) -> String {
    values
        .chunks(width)
        .map(|row| {
            row.iter()
                .map(|value| format!("{value:.10}"))
                .collect::<Vec<_>>()
                .join(" ")
                + "\n"
        })
        .collect()
}

#[test]
fn an_optbench_start_loads_at_gmins_energy() {
    let block = block_from_gmin_file(OPTBENCH_START, GminAngleAxisFile::Coords);
    let loaded = energy(&block);
    assert!(
        (loaded - OPTBENCH_CHECKD).abs() < GMIN_TOLERANCE,
        "coords {loaded} against GMIN {OPTBENCH_CHECKD}"
    );
    let misread = energy(&block_from_gmin_file(
        OPTBENCH_START,
        GminAngleAxisFile::LegacyInterleaved,
    ));
    assert!(
        (misread - OPTBENCH_CHECKD).abs() > 1.0,
        "a coords file read as legacy rows is another start, {misread}"
    );
}

#[test]
fn a_legacy_interleaved_file_loads_behind_its_flag() {
    let legacy = block_from_gmin_file(LEGACY_START, GminAngleAxisFile::LegacyInterleaved);
    let loaded = energy(&legacy);
    assert!(
        (loaded - SEED_2000_CHECKD).abs() < GMIN_TOLERANCE,
        "legacy {loaded} against GMIN {SEED_2000_CHECKD}"
    );
    let start = h2h_seed_start(15, 2000);
    for (left, right) in legacy.iter().zip(&start) {
        assert!((left - right).abs() < 1.0e-9, "{left} against {right}");
    }
    let unflagged = energy(&block_from_gmin_file(
        LEGACY_START,
        GminAngleAxisFile::Interleaved,
    ));
    assert!(
        (unflagged - SEED_2000_CHECKD).abs() > 1.0,
        "legacy rows read without the flag are another start, {unflagged}"
    );
}

#[test]
fn the_interleaved_converter_opens_h2h_seeds_start() {
    let start = h2h_seed_start(15, 2000);
    let seeded = energy(&start);
    assert!((seeded - SEED_2000_CHECKD).abs() < GMIN_TOLERANCE);

    let interleaved = interleaved_gmin_from_block(&start);
    let converted = block_from_interleaved_gmin(&interleaved);
    let through_file = block_from_gmin_file(&rows(&interleaved, 6), GminAngleAxisFile::Interleaved);
    let coords = block_from_gmin_file(&gmin_coords_text(&start), GminAngleAxisFile::Coords);
    for (label, block) in [
        ("converter", &converted),
        ("interleaved file", &through_file),
        ("coords file", &coords),
    ] {
        let value = energy(block);
        assert!(
            (value - seeded).abs() < GMIN_TOLERANCE,
            "{label} {value} against h2h_seed {seeded}"
        );
        assert!(
            (value - SEED_2000_CHECKD).abs() < GMIN_TOLERANCE,
            "{label} {value} against GMIN {SEED_2000_CHECKD}"
        );
    }
    for molecule in 0..15 {
        for k in 0..3 {
            assert_eq!(
                interleaved[6 * molecule + 3 + k],
                start[45 + 3 * molecule + k]
            );
        }
    }
}

fn gmin_checkd(binary: &str, block: &[f64], label: &str) -> f64 {
    let dir =
        std::env::temp_dir().join(format!("anneal-gmin-checkd-{}-{label}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch directory");
    std::fs::write(dir.join("coords"), gmin_coords_text(block)).expect("coords");
    let data = format!(
        "{}CHECKD 0\nRANSEED 1\n",
        include_str!("../examples/h2h_gmin_tip4p.data")
    );
    std::fs::write(dir.join("data"), data).expect("data");
    let output = std::process::Command::new(binary)
        .current_dir(&dir)
        .output()
        .expect("GMIN runs");
    let text = String::from_utf8_lossy(&output.stdout).into_owned()
        + &std::fs::read_to_string(dir.join("output")).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);
    text.lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("Energy  =")
                .and_then(|rest| rest.trim().parse::<f64>().ok())
        })
        .unwrap_or_else(|| panic!("GMIN printed no CHECKD energy for {label}"))
}

#[test]
fn every_file_form_round_trips_through_gmins_checkd() {
    let Ok(binary) = std::env::var("ANNEAL_GMIN") else {
        eprintln!("ANNEAL_GMIN is unset; the recorded CHECKD energies stand in");
        return;
    };
    let start = h2h_seed_start(15, 2000);
    let cases = [
        (
            "optbench-coords",
            block_from_gmin_file(OPTBENCH_START, GminAngleAxisFile::Coords),
        ),
        (
            "legacy-interleaved",
            block_from_gmin_file(LEGACY_START, GminAngleAxisFile::LegacyInterleaved),
        ),
        (
            "interleaved",
            block_from_gmin_file(
                &rows(&interleaved_gmin_from_block(&start), 6),
                GminAngleAxisFile::Interleaved,
            ),
        ),
        ("h2h-seed", start.clone()),
    ];
    for (label, block) in cases {
        let ours = energy(&block);
        let gmin = gmin_checkd(&binary, &block, label);
        assert!(
            (ours - gmin).abs() < GMIN_TOLERANCE,
            "{label}: GminTip4p {ours} against GMIN CHECKD {gmin}"
        );
    }
}
