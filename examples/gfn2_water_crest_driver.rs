// Copyright (c) anneal developers
//! The (H2O)m GFN2 objective of `gfn2_water_h2h`, served to xtb's driver.
//!
//! ```text
//! gfn2_water_crest_driver serve <socket> <rundir> <m_molecules> <budget> <target_eV>
//! gfn2_water_crest_driver call <socket>
//! gfn2_water_crest_driver stop <socket>
//! ```
//!
//! `serve` holds one rgpot GFN2 handle (`RGPOT_XTB_ENGINE`) for the seed,
//! started cold once, and adds the one-sided O-H wall of
//! `common/water_oh_wall.rs`: the objective anneal searches. Every request
//! is one energy and gradient call. The server counts calls, times the
//! handle, keeps the best objective and its geometry, and records the
//! first call at or below `target_eV`. After the call that reaches the
//! target or spends the budget it writes `<rundir>/stop` and refuses every
//! later request without evaluating it. `stop` ends the server, which
//! writes `<rundir>/driver.txt` and `<rundir>/best.xyz`.
//!
//! `call` is the program xtb runs for `--driver`: it reads `xtbdriver.xyz`
//! in angstrom, or `xtbdriver.coord` in Bohr when xtb was given a
//! Turbomole `coord`, from the working directory and writes the Turbomole
//! `gradient` file xtb reads back, energy in Hartree and gradient in
//! Hartree per Bohr. A refused call exits 1 without a `gradient` file.

#[path = "common/rgpot_eindir.rs"]
mod rgpot_eindir;
#[path = "common/water_oh_wall.rs"]
mod water_oh_wall;

use std::fs;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Instant;

use eindir_core::gradient::DifferentiableObjective;
use ndarray::ArrayView1;
use rgpot_eindir::RgpotObjective;
use water_oh_wall::{OH_WALL_EV_PER_A2, OH_WALL_ONSET_A, oh_wall};

/// xtb 6.7.1 conversion factors, so the driver round-trips xtb's own units.
const AUTOEV: f64 = 27.21138505;
const AUTOAA: f64 = 0.52917726;

const STATUS_OK: u8 = 0;
const STATUS_REFUSED: u8 = 1;
const STOP_REQUEST: u64 = u64::MAX;

fn read_xyz(path: &Path) -> (Vec<String>, Vec<f64>) {
    let text =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let mut lines = text.lines();
    let count: usize = lines
        .next()
        .and_then(|line| line.trim().parse().ok())
        .unwrap_or_else(|| panic!("{}: no atom count", path.display()));
    lines.next();
    let mut symbols = Vec::with_capacity(count);
    let mut coords = Vec::with_capacity(3 * count);
    for line in lines.take(count) {
        let mut parts = line.split_whitespace();
        symbols.push(parts.next().expect("symbol").to_string());
        for _ in 0..3 {
            coords.push(
                parts
                    .next()
                    .and_then(|value| value.parse::<f64>().ok())
                    .unwrap_or_else(|| panic!("{}: bad coordinate", path.display())),
            );
        }
    }
    assert_eq!(symbols.len(), count, "{}: short xyz", path.display());
    (symbols, coords)
}

fn read_coord(path: &Path) -> (Vec<String>, Vec<f64>) {
    let text =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let mut symbols = Vec::new();
    let mut coords = Vec::new();
    let mut inside = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('$') {
            inside = line.starts_with("$coord");
            continue;
        }
        if !inside || line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        assert!(parts.len() >= 4, "{}: short coord line", path.display());
        for value in &parts[..3] {
            let bohr: f64 = value
                .parse()
                .unwrap_or_else(|_| panic!("{}: bad coordinate", path.display()));
            coords.push(bohr * AUTOAA);
        }
        symbols.push(parts[3].to_string());
    }
    (symbols, coords)
}

fn write_u64(stream: &mut UnixStream, value: u64) {
    stream
        .write_all(&value.to_le_bytes())
        .expect("socket write");
}

fn read_u64(stream: &mut UnixStream) -> u64 {
    let mut bytes = [0u8; 8];
    stream.read_exact(&mut bytes).expect("socket read");
    u64::from_le_bytes(bytes)
}

fn write_f64s(stream: &mut UnixStream, values: &[f64]) {
    let mut bytes = Vec::with_capacity(8 * values.len());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    stream.write_all(&bytes).expect("socket write");
}

fn read_f64s(stream: &mut UnixStream, count: usize) -> Vec<f64> {
    let mut bytes = vec![0u8; 8 * count];
    stream.read_exact(&mut bytes).expect("socket read");
    bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("eight bytes")))
        .collect()
}

struct Served {
    calls: u64,
    inside_ns: u128,
    best: f64,
    best_coords: Vec<f64>,
    first_hit_call: Option<u64>,
    stop_reason: Option<&'static str>,
    refused: u64,
    failed: u64,
}

fn serve(socket: &Path, rundir: &Path, molecules: usize, budget: u64, target: f64) {
    fs::create_dir_all(rundir).expect("rundir");
    let _ = fs::remove_file(socket);
    let _ = fs::remove_file(rundir.join("stop"));
    let atmnrs: Vec<i32> = (0..molecules).flat_map(|_| [8, 1, 1]).collect();
    let box_ = [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0];
    let mut pot = RgpotObjective::xtb(&atmnrs, box_);
    pot.restart();
    let dim = 3 * atmnrs.len();
    let listener = UnixListener::bind(socket).expect("bind socket");
    fs::write(rundir.join("ready"), "ready\n").expect("ready");
    let mut served = Served {
        calls: 0,
        inside_ns: 0,
        best: f64::INFINITY,
        best_coords: Vec::new(),
        first_hit_call: None,
        stop_reason: None,
        refused: 0,
        failed: 0,
    };
    for stream in listener.incoming() {
        let mut stream = stream.expect("accept");
        let length = read_u64(&mut stream);
        if length == STOP_REQUEST {
            break;
        }
        assert_eq!(length as usize, dim, "request dimension");
        let coords = read_f64s(&mut stream, dim);
        if served.stop_reason.is_some() {
            served.refused += 1;
            stream.write_all(&[STATUS_REFUSED]).expect("socket write");
            continue;
        }
        let at = Instant::now();
        let (energy, gradient) = pot.wrapper().value_and_gradient(ArrayView1::from(&coords));
        served.inside_ns += at.elapsed().as_nanos();
        served.calls += 1;
        let mut gradient: Vec<f64> = gradient.iter().copied().collect();
        let finite = energy.is_finite()
            && gradient.len() == dim
            && gradient.iter().all(|value| value.is_finite());
        let total = if finite {
            let (wall, wall_gradient) = oh_wall(&coords, molecules);
            for (component, extra) in gradient.iter_mut().zip(wall_gradient) {
                *component += extra;
            }
            energy + wall
        } else {
            served.failed += 1;
            f64::NAN
        };
        if total < served.best {
            served.best = total;
            served.best_coords.clone_from(&coords);
        }
        if served.first_hit_call.is_none() && total <= target {
            served.first_hit_call = Some(served.calls);
            served.stop_reason = Some("target");
        } else if served.calls >= budget {
            served.stop_reason = Some("budget");
        }
        if let Some(reason) = served.stop_reason {
            fs::write(rundir.join("stop"), format!("{reason} {}\n", served.calls)).expect("stop");
        }
        if finite {
            stream.write_all(&[STATUS_OK]).expect("socket write");
            write_f64s(&mut stream, &[total]);
            write_f64s(&mut stream, &gradient);
        } else {
            stream.write_all(&[STATUS_REFUSED]).expect("socket write");
        }
    }
    let _ = fs::remove_file(socket);
    let inside_s = served.inside_ns as f64 / 1e9;
    let mut meta = fs::File::create(rundir.join("driver.txt")).expect("driver.txt");
    writeln!(meta, "objective: rgpot GFN2 plus the one-sided O-H wall").ok();
    writeln!(meta, "oh_wall_onset_A: {OH_WALL_ONSET_A}").ok();
    writeln!(meta, "oh_wall_eV_per_A2: {OH_WALL_EV_PER_A2}").ok();
    writeln!(meta, "molecules: {molecules}").ok();
    writeln!(meta, "budget: {budget}").ok();
    writeln!(meta, "target_eV: {target:.10}").ok();
    writeln!(meta, "calls: {}", served.calls).ok();
    writeln!(meta, "refused: {}", served.refused).ok();
    writeln!(meta, "failed: {}", served.failed).ok();
    writeln!(meta, "inside_s: {inside_s:.6}").ok();
    writeln!(meta, "best_eV: {:.10}", served.best).ok();
    match served.first_hit_call {
        Some(call) => writeln!(meta, "first_hit_call: {call}").ok(),
        None => writeln!(meta, "first_hit_call: none").ok(),
    };
    writeln!(meta, "stop: {}", served.stop_reason.unwrap_or("none")).ok();
    if !served.best_coords.is_empty() {
        let mut xyz = fs::File::create(rundir.join("best.xyz")).expect("best.xyz");
        writeln!(xyz, "{}", atmnrs.len()).ok();
        writeln!(xyz, "objective {:.10} eV", served.best).ok();
        for (index, atom) in served.best_coords.chunks_exact(3).enumerate() {
            let symbol = if index % 3 == 0 { "O" } else { "H" };
            writeln!(
                xyz,
                "{symbol} {:.10} {:.10} {:.10}",
                atom[0], atom[1], atom[2]
            )
            .ok();
        }
    }
}

fn call(socket: &Path) -> i32 {
    let (symbols, coords) = if Path::new("xtbdriver.xyz").is_file() {
        read_xyz(Path::new("xtbdriver.xyz"))
    } else {
        read_coord(Path::new("xtbdriver.coord"))
    };
    let _ = fs::remove_file("gradient");
    let mut stream = UnixStream::connect(socket).expect("connect");
    write_u64(&mut stream, coords.len() as u64);
    write_f64s(&mut stream, &coords);
    let mut status = [0u8; 1];
    stream.read_exact(&mut status).expect("socket read");
    if status[0] != STATUS_OK {
        return 1;
    }
    let energy = read_f64s(&mut stream, 1)[0] / AUTOEV;
    let gradient = read_f64s(&mut stream, coords.len());
    let norm = gradient
        .iter()
        .map(|value| (value * AUTOAA / AUTOEV).powi(2))
        .sum::<f64>()
        .sqrt();
    let mut text = String::new();
    text.push_str("$grad\n");
    text.push_str(&format!(
        "  cycle =      1    SCF energy = {energy:20.11}   |dE/dxyz| = {norm:10.6}\n"
    ));
    for (symbol, atom) in symbols.iter().zip(coords.chunks_exact(3)) {
        text.push_str(&format!(
            "{:22.14} {:22.14} {:22.14}      {}\n",
            atom[0] / AUTOAA,
            atom[1] / AUTOAA,
            atom[2] / AUTOAA,
            symbol.to_lowercase()
        ));
    }
    for component in gradient.chunks_exact(3) {
        text.push_str(&format!(
            "{:22.13E} {:22.13E} {:22.13E}\n",
            component[0] * AUTOAA / AUTOEV,
            component[1] * AUTOAA / AUTOEV,
            component[2] * AUTOAA / AUTOEV
        ));
    }
    text.push_str("$end\n");
    fs::write("gradient", text).expect("gradient");
    0
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("serve") => {
            let socket = PathBuf::from(args.get(1).expect("socket"));
            let rundir = PathBuf::from(args.get(2).expect("rundir"));
            let molecules: usize = args.get(3).and_then(|v| v.parse().ok()).expect("molecules");
            let budget: u64 = args.get(4).and_then(|v| v.parse().ok()).expect("budget");
            let target: f64 = args.get(5).and_then(|v| v.parse().ok()).expect("target_eV");
            assert!(
                molecules > 0 && budget > 0,
                "molecules and budget must be positive"
            );
            serve(&socket, &rundir, molecules, budget, target);
        }
        Some("call") => {
            let socket = PathBuf::from(args.get(1).expect("socket"));
            std::process::exit(call(&socket));
        }
        Some("stop") => {
            let socket = PathBuf::from(args.get(1).expect("socket"));
            let mut stream = UnixStream::connect(socket).expect("connect");
            write_u64(&mut stream, STOP_REQUEST);
        }
        _ => {
            eprintln!(
                "usage: gfn2_water_crest_driver serve <socket> <rundir> <m> <budget> <target_eV> | call <socket> | stop <socket>"
            );
            std::process::exit(2);
        }
    }
}
