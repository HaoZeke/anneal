//! Stock-preset searches for binary Lennard-Jones and rigid TIP4P.
//!
//! The presets are [`Config::recommended`] and [`Config::communicating`].
//! Species diameters and the rigid chart are applied after that construction.
//!
//! ```text
//! binary_water_campaign binary <coords-dir> <preset> <budget> <target> <out.jsonl> [limit]
//! binary_water_campaign tip4p <aa-dir> <molecules> <preset> <budget> <target> <out.jsonl> [limit]
//! ```
//!
//! `preset` is `recommended` or `communicating`. A positive `limit` keeps
//! the first starts in index order. Each line of the output is one start.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::Mutex;
use std::thread;

use anneal_core::binary_water::{BinaryLennardJonesProblem, search_binary, search_gmin_tip4p};
use anneal_core::potentials::block_from_interleaved_gmin;

fn main() {
    let mut args = std::env::args().skip(1);
    let kind = args.next().expect("binary or tip4p");
    match kind.as_str() {
        "binary" => run_binary(&mut args),
        "tip4p" => run_tip4p(&mut args),
        other => panic!("unknown campaign {other}"),
    }
}

fn run_binary(args: &mut impl Iterator<Item = String>) {
    let dir = args.next().expect("coords directory");
    let preset = args.next().expect("preset");
    let budget: usize = args.next().expect("budget").parse().expect("budget");
    let target: f64 = args.next().expect("target").parse().expect("target");
    let out = args.next().expect("output");
    let limit: usize = args
        .next()
        .map(|text| text.parse().expect("limit"))
        .unwrap_or(0);
    let communicating = preset_flag(&preset);
    let mut indexed = list_indexed(Path::new(&dir), "coords");
    if limit > 0 {
        indexed.truncate(limit);
    }
    let problem = BinaryLennardJonesProblem {
        n_a: 42,
        n_b: 58,
        epsilon_aa: 1.0,
        epsilon_ab: 1.0,
        epsilon_bb: 1.0,
        sigma_aa: 1.0,
        sigma_ab: 1.15,
        sigma_bb: 1.3,
    };
    let jobs: Vec<(usize, String, Vec<f64>)> = indexed
        .iter()
        .map(|(index, path)| {
            let name = path
                .file_name()
                .and_then(|text| text.to_str())
                .unwrap_or("start")
                .to_owned();
            (*index, name, read_xyz_rows(path))
        })
        .collect();
    stream_binary(
        &out,
        &preset,
        &problem,
        &jobs,
        budget,
        target,
        communicating,
    );
}

fn run_tip4p(args: &mut impl Iterator<Item = String>) {
    let dir = args.next().expect("angle-axis directory");
    let molecules: usize = args.next().expect("molecules").parse().expect("molecules");
    let preset = args.next().expect("preset");
    let budget: usize = args.next().expect("budget").parse().expect("budget");
    let target: f64 = args.next().expect("target").parse().expect("target");
    let out = args.next().expect("output");
    let limit: usize = args
        .next()
        .map(|text| text.parse().expect("limit"))
        .unwrap_or(0);
    let communicating = preset_flag(&preset);
    let mut indexed = list_indexed(Path::new(&dir), "aa");
    if limit > 0 {
        indexed.truncate(limit);
    }
    let jobs: Vec<(usize, String, Vec<f64>)> = indexed
        .iter()
        .map(|(index, path)| {
            let name = path
                .file_name()
                .and_then(|text| text.to_str())
                .unwrap_or("start")
                .to_owned();
            (*index, name, block_from_interleaved_gmin(&read_flat(path)))
        })
        .collect();
    stream_tip4p(
        &out,
        &preset,
        molecules,
        &jobs,
        budget,
        target,
        communicating,
    );
}

fn preset_flag(preset: &str) -> bool {
    match preset {
        "recommended" => false,
        "communicating" => true,
        other => panic!("preset is recommended or communicating, got {other}"),
    }
}

fn start_index(name: &str) -> Option<usize> {
    if let Some(rest) = name.strip_prefix("coords.") {
        return rest.parse().ok();
    }
    name.strip_suffix(".aa")?.parse().ok()
}

fn list_indexed(dir: &Path, kind: &str) -> Vec<(usize, std::path::PathBuf)> {
    let mut indexed = Vec::new();
    for entry in std::fs::read_dir(dir).expect("start directory") {
        let path = entry.expect("directory entry").path();
        let Some(name) = path.file_name().and_then(|text| text.to_str()) else {
            continue;
        };
        let Some(index) = start_index(name) else {
            continue;
        };
        if kind == "coords" && !name.starts_with("coords.") {
            continue;
        }
        if kind == "aa" && !name.ends_with(".aa") {
            continue;
        }
        indexed.push((index, path));
    }
    indexed.sort_by_key(|(index, _)| *index);
    indexed
}

fn read_xyz_rows(path: &Path) -> Vec<f64> {
    let file = File::open(path).expect("coords");
    let mut coords = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.expect("coords line");
        if line.split_whitespace().count() < 3 {
            continue;
        }
        for token in line.split_whitespace().take(3) {
            coords.push(token.parse::<f64>().expect("coordinate"));
        }
    }
    coords
}

fn read_flat(path: &Path) -> Vec<f64> {
    let file = File::open(path).expect("angle-axis");
    let mut coords = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.expect("angle-axis line");
        for token in line.split_whitespace() {
            coords.push(token.parse::<f64>().expect("angle-axis value"));
        }
    }
    coords
}

fn worker_count() -> usize {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|text| text.parse().ok())
        .filter(|value: &usize| *value > 0)
        .unwrap_or_else(|| {
            thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
}

fn stream_binary(
    out: &str,
    preset: &str,
    problem: &BinaryLennardJonesProblem,
    jobs: &[(usize, String, Vec<f64>)],
    budget: usize,
    target: f64,
    communicating: bool,
) {
    let file = Mutex::new(File::create(out).expect("output"));
    let cursor = Mutex::new(0usize);
    let calls = Mutex::new(Vec::<usize>::new());
    let successes = Mutex::new(0usize);
    let problem_name = "globalopt/lj100-binary";
    thread::scope(|scope| {
        for _ in 0..worker_count().min(jobs.len().max(1)) {
            scope.spawn(|| loop {
                let index = {
                    let mut cursor = cursor.lock().expect("cursor");
                    let index = *cursor;
                    *cursor += 1;
                    index
                };
                if index >= jobs.len() {
                    break;
                }
                let (start, name, coords) = &jobs[index];
                let found = search_binary(
                    problem,
                    ndarray::Array1::from_vec(coords.clone()).view(),
                    budget,
                    target,
                    *start as u64,
                    communicating,
                );
                let success = found.success(target);
                {
                    let mut calls = calls.lock().expect("calls");
                    calls.push(found.charged);
                    if success {
                        *successes.lock().expect("successes") += 1;
                    }
                }
                writeln!(
                    file.lock().expect("output"),
                    "{{\"start\":{start},\"file\":\"{name}\",\"problem\":\"{problem_name}\",\"preset\":\"{preset}\",\"energy\":{:.10},\"charged\":{},\"success\":{}}}",
                    found.energy, found.charged, success
                )
                .expect("write");
                eprintln!(
                    "{problem_name} {preset} start {start} energy {:.6} charged {} success {success}",
                    found.energy, found.charged
                );
            });
        }
    });
    summarize(
        problem_name,
        preset,
        &calls.into_inner().expect("calls"),
        *successes.lock().expect("successes"),
    );
}

fn stream_tip4p(
    out: &str,
    preset: &str,
    molecules: usize,
    jobs: &[(usize, String, Vec<f64>)],
    budget: usize,
    target: f64,
    communicating: bool,
) {
    let file = Mutex::new(File::create(out).expect("output"));
    let cursor = Mutex::new(0usize);
    let calls = Mutex::new(Vec::<usize>::new());
    let successes = Mutex::new(0usize);
    let problem_name = format!("globalopt/tip4p{molecules}");
    thread::scope(|scope| {
        for _ in 0..worker_count().min(jobs.len().max(1)) {
            scope.spawn(|| loop {
                let index = {
                    let mut cursor = cursor.lock().expect("cursor");
                    let index = *cursor;
                    *cursor += 1;
                    index
                };
                if index >= jobs.len() {
                    break;
                }
                let (start, name, coords) = &jobs[index];
                let found = search_gmin_tip4p(
                    molecules,
                    ndarray::Array1::from_vec(coords.clone()).view(),
                    budget,
                    target,
                    *start as u64,
                    communicating,
                );
                let success = found.success(target);
                {
                    let mut calls = calls.lock().expect("calls");
                    calls.push(found.charged);
                    if success {
                        *successes.lock().expect("successes") += 1;
                    }
                }
                writeln!(
                    file.lock().expect("output"),
                    "{{\"start\":{start},\"file\":\"{name}\",\"problem\":\"{problem_name}\",\"preset\":\"{preset}\",\"energy\":{:.10},\"charged\":{},\"success\":{}}}",
                    found.energy, found.charged, success
                )
                .expect("write");
                eprintln!(
                    "{problem_name} {preset} start {start} energy {:.6} charged {} success {success}",
                    found.energy, found.charged
                );
            });
        }
    });
    summarize(
        &problem_name,
        preset,
        &calls.into_inner().expect("calls"),
        *successes.lock().expect("successes"),
    );
}

fn summarize(problem: &str, preset: &str, calls: &[usize], successes: usize) {
    let mut ordered = calls.to_vec();
    ordered.sort_unstable();
    let median = ordered.get(ordered.len() / 2).copied().unwrap_or(0);
    let mean = if ordered.is_empty() {
        0.0
    } else {
        ordered.iter().sum::<usize>() as f64 / ordered.len() as f64
    };
    eprintln!(
        "{problem} {preset} n={} successes={successes} median={median} mean={mean:.2} min={} max={}",
        ordered.len(),
        ordered.first().copied().unwrap_or(0),
        ordered.last().copied().unwrap_or(0)
    );
}
