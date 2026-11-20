// Copyright (c) anneal developers
//! The last frame of an extended XYZ file as a non-periodic `.con` file.
//!
//! ```text
//! xyz_to_con <in.xyz> <out.con> [symbol]
//! ```
//!
//! Coordinates are copied unchanged into an open 500-unit box with every
//! direction non-periodic. The values stay in the input's units; the
//! readcon-core v3 header names its default units because it has no name
//! for reduced ones. An `energy=` field on the frame's comment line becomes
//! the frame energy. Every atom gets `symbol` (default `Ar`).

use readcon_core::types::{ConFrameBuilder, meta};
use readcon_core::writer::ConFrameWriter;
use std::collections::BTreeMap;

const OPEN_BOX: f64 = 500.0;
const PRECISION: usize = 17;

fn last_frame(text: &str) -> Result<(Option<f64>, Vec<[f64; 3]>), String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut at = 0;
    let mut last = None;
    while at < lines.len() && !lines[at].trim().is_empty() {
        let n: usize = lines[at]
            .trim()
            .parse()
            .map_err(|_| format!("line {}: atom count", at + 1))?;
        if at + 1 + n >= lines.len() + 1 {
            return Err(format!("line {}: frame is short", at + 1));
        }
        let energy = lines[at + 1]
            .split_whitespace()
            .find_map(|field| field.strip_prefix("energy="))
            .and_then(|v| v.parse::<f64>().ok());
        let mut atoms = Vec::with_capacity(n);
        for line in &lines[at + 2..at + 2 + n] {
            let v: Vec<f64> = line
                .split_whitespace()
                .skip(1)
                .take(3)
                .map(|x| {
                    x.parse::<f64>()
                        .map_err(|_| format!("bad coordinate in {line}"))
                })
                .collect::<Result<_, _>>()?;
            if v.len() != 3 {
                return Err(format!("short atom line {line}"));
            }
            atoms.push([v[0], v[1], v[2]]);
        }
        last = Some((energy, atoms));
        at += 2 + n;
    }
    last.ok_or_else(|| "no frame".to_owned())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: xyz_to_con <in.xyz> <out.con> [symbol]");
        std::process::exit(2);
    }
    let symbol = args.get(2).map(String::as_str).unwrap_or("Ar");
    let text = match std::fs::read_to_string(&args[0]) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("{}: {err}", args[0]);
            std::process::exit(1);
        }
    };
    let (energy, atoms) = match last_frame(&text) {
        Ok(frame) => frame,
        Err(err) => {
            eprintln!("{}: {err}", args[0]);
            std::process::exit(1);
        }
    };
    let mut builder = ConFrameBuilder::new([OPEN_BOX; 3], [90.0; 3]);
    let mut metadata = BTreeMap::new();
    metadata.insert(meta::PBC.into(), serde_json::json!([false, false, false]));
    builder.metadata(metadata);
    if let Some(energy) = energy {
        builder.set_energy(energy);
    }
    for (id, atom) in atoms.iter().enumerate() {
        builder.add_atom(
            symbol, atom[0], atom[1], atom[2], [false; 3], id as u64, 1.0,
        );
    }
    let frame = match builder.build() {
        Ok(frame) => frame,
        Err(err) => {
            eprintln!("con frame: {err}");
            std::process::exit(1);
        }
    };
    let written = ConFrameWriter::from_path_with_precision(&args[1], PRECISION)
        .and_then(|mut writer| writer.write_frame(&frame));
    if let Err(err) = written {
        eprintln!("{}: {err}", args[1]);
        std::process::exit(1);
    }
}
