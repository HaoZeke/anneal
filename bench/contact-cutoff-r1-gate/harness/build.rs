// Copyright (c) anneal developers

//! Each binary is one campaign driver from `../anneal/examples`, the tree
//! under test, compiled unchanged except for one wrapper. The wrapper sits
//! around the driver's single `search(...)` call and prints the best
//! energy's bits and a SHA-256 of the outcome after the search returns,
//! so the driver's own timing and row are untouched.

use std::path::PathBuf;

const DRIVERS: [&str; 4] = ["h2h_gmin_proof", "h2h_sc", "h2h_metal", "h2h_morse"];
const CALL: &str = "= search(";

const WRAPPER: &str = r#"
fn gate_record(
    run: (anneal_core::methods::cluster_hopping::Outcome, f64, f64),
) -> (anneal_core::methods::cluster_hopping::Outcome, f64, f64) {
    use sha2::Digest;
    let mut trace = run.0.clone();
    // Wall-clock seconds; every other field is a function of the seed.
    trace.history_seconds = 0.0;
    let digest = sha2::Sha256::digest(format!("{trace:?}").as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    println!(
        "gate_outcome,{:016x},{:?},{},{}",
        run.0.best.to_bits(),
        run.0.best,
        run.0.charged,
        hex
    );
    run
}
"#;

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let tree = root.join("../anneal/examples");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for name in DRIVERS {
        let path = tree.join(format!("{name}.rs"));
        println!("cargo:rerun-if-changed={}", path.display());
        let text = std::fs::read_to_string(&path).expect("driver source");
        assert_eq!(text.matches(CALL).count(), 1, "{name}: one search call");
        let at = text.find(CALL).unwrap();
        let end = at + text[at..].find(";\n").expect("call ends the line");
        assert!(!text[at..end].contains('\n'), "{name}: call on one line");
        let wrapped = format!(
            "{}= gate_record({}){}",
            &text[..at],
            &text[at + 2..end],
            &text[end..]
        );
        // Inner doc comments are not items, and `include!` takes items.
        let body: String = wrapped
            .lines()
            .filter(|line| !line.starts_with("//!"))
            .map(|line| format!("{line}\n"))
            .collect();
        std::fs::write(out.join(format!("{name}.rs")), body + WRAPPER).unwrap();
    }
}
