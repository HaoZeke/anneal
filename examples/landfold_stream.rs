// Copyright (c) anneal developers

//! Fit a landfold map on one minima book and stream larger books through it.
//!
//! ```text
//! landfold_stream FIT.min OUT.csv BOOK.min...
//! ```
//!
//! `FIT.min` must be the first book in `BOOK.min...`. Its rows fit the map and
//! retain their fitted coordinates; every later row is placed by the fitted
//! out-of-sample map. This bounds memory independently of the aggregate size.

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anneal_core::catalog::packing::{PackingBook, prepare_rows};
use anneal_core::landfold::{LandfoldConfig, landfold};
use rayon::prelude::*;

const CHUNK: usize = 128;

fn parse_row(line: &str, path: &Path, line_no: usize) -> Option<(f64, Vec<f64>)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let values: Vec<f64> = line
        .split_whitespace()
        .map(|value| {
            value.parse().unwrap_or_else(|err| {
                panic!("{}:{line_no}: {err}", path.display());
            })
        })
        .collect();
    assert!(
        values.len() >= 4 && (values.len() - 1).is_multiple_of(3),
        "{}:{line_no}: expected energy and 3N coordinates",
        path.display()
    );
    Some((values[0], values[1..].to_vec()))
}

fn read_chunk(
    rows: &mut impl Iterator<Item = std::io::Result<String>>,
    path: &Path,
    line_no: &mut usize,
) -> Vec<(f64, Vec<f64>)> {
    let mut chunk = Vec::with_capacity(CHUNK);
    while chunk.len() < CHUNK {
        let Some(line) = rows.next() else {
            break;
        };
        *line_no += 1;
        if let Some(row) = parse_row(
            &line.unwrap_or_else(|err| panic!("{}: {err}", path.display())),
            path,
            *line_no,
        ) {
            chunk.push(row);
        }
    }
    chunk
}

fn visit(path: &Path, mut consume: impl FnMut(&[(f64, Vec<f64>)])) {
    let file = File::open(path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let mut rows = BufReader::new(file).lines();
    let mut line_no = 0;
    loop {
        let chunk = read_chunk(&mut rows, path, &mut line_no);
        if chunk.is_empty() {
            break;
        }
        consume(&chunk);
    }
}

fn describe(book: &PackingBook, coordinates: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if coordinates.is_empty() {
        return Vec::new();
    }
    let groups = rayon::current_num_threads().min(coordinates.len());
    let width = coordinates.len().div_ceil(groups);
    let locals: Vec<PackingBook> = (0..groups).map(|_| book.clone()).collect();
    coordinates
        .par_chunks(width)
        .zip(locals.into_par_iter())
        .flat_map_iter(|(chunk, local)| {
            chunk.iter().map(move |coordinates| {
                local
                    .codebook_histogram(coordinates)
                    .expect("row has a descriptor")
            })
        })
        .collect()
}

fn main() {
    let args: Vec<PathBuf> = env::args_os().skip(1).map(PathBuf::from).collect();
    assert!(
        args.len() >= 3,
        "usage: landfold_stream FIT.min OUT.csv BOOK.min..."
    );
    let fit_path = &args[0];
    let output = &args[1];
    let books = &args[2..];
    assert_eq!(
        books.first(),
        Some(fit_path),
        "FIT.min must be the first aggregate book"
    );

    let mut book = PackingBook::default();
    let mut observed = 0usize;
    for path in books {
        visit(path, |chunk| {
            chunk
                .par_iter()
                .for_each(|(_, coordinates)| prepare_rows(coordinates));
            for (_, coordinates) in chunk {
                book.observe(coordinates)
                    .unwrap_or_else(|| panic!("{}: no finite descriptor", path.display()));
                observed += 1;
            }
        });
    }

    let mut fit_energy = Vec::new();
    let mut fit_coordinates = Vec::new();
    visit(fit_path, |chunk| {
        for (energy, coordinates) in chunk {
            fit_energy.push(*energy);
            fit_coordinates.push(coordinates.clone());
        }
    });
    let fit_rows = describe(&book, &fit_coordinates);
    let fold = landfold(&fit_rows, &LandfoldConfig::default()).expect("finite fit rows");
    let mut landmark = vec![false; fit_rows.len()];
    for &index in &fold.landmarks {
        landmark[index] = true;
    }

    let file = File::create(output).unwrap_or_else(|err| panic!("{}: {err}", output.display()));
    let mut out = BufWriter::new(file);
    writeln!(out, "index,energy,x,y,landmark,family,community").expect("write header");
    let mut index = 0usize;
    for (path_index, path) in books.iter().enumerate() {
        if path_index == 0 {
            for row in 0..fit_rows.len() {
                writeln!(
                    out,
                    "{index},{:.12},{:.12e},{:.12e},{},,",
                    fit_energy[row], fold.xy[row][0], fold.xy[row][1], landmark[row],
                )
                .expect("write fit point");
                index += 1;
            }
            continue;
        }
        visit(path, |chunk| {
            let coordinates: Vec<Vec<f64>> = chunk
                .iter()
                .map(|(_, coordinates)| coordinates.clone())
                .collect();
            let rows = describe(&book, &coordinates);
            let points: Vec<(f64, [f64; 2])> = chunk
                .par_iter()
                .zip(rows.par_iter())
                .map(|((energy, _), row)| (*energy, fold.map.project_histogram(row)))
                .collect();
            for (energy, xy) in points {
                writeln!(
                    out,
                    "{index},{energy:.12},{:.12e},{:.12e},false,,",
                    xy[0], xy[1],
                )
                .expect("write projected point");
                index += 1;
            }
        });
    }
    out.flush().expect("flush aggregate fold");
    assert_eq!(index, observed, "streamed every observed minimum");
    eprintln!(
        "landfold_stream: fit={} aggregate={} landmarks={}",
        fit_rows.len(),
        observed,
        fold.landmarks.len(),
    );
}
