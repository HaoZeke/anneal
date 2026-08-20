//! Landfold the occupancy packing book.
//!
//! Each input is an xyz file or an energy-leading `.min` dump (one
//! quenched structure per line). Structures are observed into a
//! [`PackingBook`], leftover-well arrivals are credited once per
//! structure, and the landfold-sparsified map is printed as
//! `occupancy_landfold` JSON.
//!
//!     occupancy_landfold_book [--label NAME] INPUT [INPUT ...]

use anneal_core::catalog::{PackingBook, occupancy_sparsify_packing};
use std::env;
use std::fs;
use std::path::Path;

fn load_xyz(path: &Path) -> Vec<(f64, Vec<f64>)> {
    let text = fs::read_to_string(path).unwrap_or_else(|err| panic!("{path:?}: {err}"));
    let values: Vec<f64> = text
        .lines()
        .skip(2)
        .flat_map(|line| {
            line.split_whitespace()
                .skip(1)
                .filter_map(|token| token.parse::<f64>().ok())
        })
        .collect();
    assert!(
        !values.is_empty() && values.len().is_multiple_of(3),
        "{path:?}: xyz body is not a coordinate list"
    );
    vec![(f64::NAN, values)]
}

fn load_min(path: &Path) -> Vec<(f64, Vec<f64>)> {
    let text = fs::read_to_string(path).unwrap_or_else(|err| panic!("{path:?}: {err}"));
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut nums = line
                .split_whitespace()
                .map(|token| token.parse::<f64>().expect("min dump holds numbers"));
            let energy = nums.next().expect("min dump starts with energy");
            let coords: Vec<f64> = nums.collect();
            assert!(
                !coords.is_empty() && coords.len().is_multiple_of(3),
                "{path:?}: min row is not energy plus coordinates"
            );
            (energy, coords)
        })
        .collect()
}

fn load(path: &Path) -> Vec<(f64, Vec<f64>)> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("xyz") => load_xyz(path),
        _ => load_min(path),
    }
}

fn main() {
    let mut label = String::from("book");
    let mut inputs = Vec::new();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--label" {
            label = args.next().expect("--label needs a name");
        } else {
            inputs.push(arg);
        }
    }
    assert!(
        !inputs.is_empty(),
        "usage: occupancy_landfold_book [--label NAME] INPUT..."
    );
    let mut book = PackingBook::default();
    let mut n_in = 0u64;
    let mut n_obs = 0u64;
    for input in &inputs {
        for (energy, coords) in load(Path::new(input)) {
            n_in += 1;
            let Some(family) = book.observe(&coords) else {
                continue;
            };
            book.credit_well(family);
            n_obs += 1;
            let _ = energy;
        }
    }
    let map = occupancy_sparsify_packing(&book);
    let sample = map.sample();
    let mut points = String::new();
    for (index, point) in map.points.iter().enumerate() {
        if index > 0 {
            points.push(',');
        }
        points.push_str(&format!(
            "{{\"family\":{},\"community\":{},\"x\":{:.8},\"y\":{:.8},\"wells\":{}}}",
            point.family, point.community, point.xy[0], point.xy[1], point.wells
        ));
    }
    println!(
        "{{\"kind\":\"occupancy_landfold\",\"label\":\"{}\",\"n_in\":{},\"n_obs\":{},\"occupied\":{},\"floor\":{},\"left\":{},\"right\":{},\"communities\":{},\"holes\":{},\"sparsified_n\":{},\"sparsified_n1\":{},\"points\":[{}]}}",
        label,
        n_in,
        n_obs,
        map.points.len(),
        map.floor,
        map.left,
        map.right,
        map.communities,
        map.holes,
        sample.n,
        sample.n1,
        points
    );
}
