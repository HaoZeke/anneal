//! Single-point TIP4P energy of a GMIN `coords` block.
//!
//! Usage: `h2h_tip4p_point <n> <coords-file>`
//!
//! The file is GMIN's `NTIP` `coords`: every molecular centre, then every
//! angle-axis vector. The energy is the one `GminTip4p` scores, so the value
//! can be compared against GMIN's own single point at the same geometry to
//! confirm both codes share the TIP4P parameters and energy units.

use anneal_core::potentials::{GminAngleAxisFile, GminTip4p, block_from_gmin_file};
use eindir_core::Objective;
use ndarray::Array1;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(args.len(), 2, "see the usage line in the source");
    let n: usize = args[0].parse().expect("n");
    let text = std::fs::read_to_string(&args[1]).expect("coords file");
    let block = block_from_gmin_file(&text, GminAngleAxisFile::Coords);
    assert_eq!(block.len(), 6 * n, "coords hold 6 numbers per molecule");
    let state = Array1::from(block);
    let energy = GminTip4p::new(n).eval(state.view());
    println!("energy: {energy:.10}");
}
