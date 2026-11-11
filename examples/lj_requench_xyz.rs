// Copyright (c) anneal developers
//! Re-quench one xyz file with [`anneal_core::methods::lattice_search::lj_requench_calls`].
//!
//! Usage: `lj_requench_xyz <file>`
//!
//! Prints the minimised energy and the number of value-and-gradient calls.
//! Those calls are the re-quench charge. They are not search calls.

fn coords_of(text: &str) -> Vec<f64> {
    let mut coords = Vec::new();
    for line in text.lines().skip(2) {
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let _symbol = parts.next().expect("element");
        for _ in 0..3 {
            coords.push(parts.next().expect("coordinate").parse::<f64>().unwrap());
        }
    }
    coords
}

#[cfg(test)]
mod tests {
    use super::coords_of;

    #[test]
    fn coords_skip_the_header_and_a_blank_line() {
        let text = "2\ncomment\nX 0.0 0.0 0.0\n\nX 1.0 0.0 0.0\n";
        assert_eq!(
            coords_of(text),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("xyz path");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{path}: {err}"));
    let coords = coords_of(&text);
    let (energy, calls) = anneal_core::methods::lattice_search::lj_requench_calls(&coords);
    println!("energy {energy:.16}");
    println!("calls {calls}");
}
