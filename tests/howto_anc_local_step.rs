//! The Rust fragments of `docs/orgmode/howto/anc-local-step.org`.
//!
//! Each fragment on the page must appear in this file verbatim, where it is
//! compiled, and the function the page shows is run on two Lennard-Jones
//! "waters".

use anneal_core::methods::anc_opt::{AncUnits, LocalStep, ModelHessian};
use anneal_core::methods::cluster_hopping::Config;
use anneal_core::potentials::PairPotential;

const PAGE: &str = include_str!("../docs/orgmode/howto/anc-local-step.org");
const THIS: &str = include_str!("howto_anc_local_step.rs");

mod page {
    use std::error::Error;

    use anneal_core::DifferentiableObjective;
    use anneal_core::methods::anc_opt::LocalStep;
    use anneal_core::methods::cluster_hopping::{Config, Ledger};
    use anneal_core::methods::cluster_search::search;

    pub fn relax_with_anc<O: DifferentiableObjective<f64>>(
        objective: &O,
        species: Vec<u32>,
        groups: Vec<Vec<usize>>,
        budget: usize,
    ) -> Result<(), Box<dyn Error>> {
        let mut cfg = Config::recommended_molecular(species, groups, 1.0);
        cfg.local_step = LocalStep::ANC_EV_ANGSTROM;
        let json = cfg.resolved_json()?;
        assert!(json.contains(r#""local_step":{"kind":"anc""#));
        let mut ledger = Ledger::new(budget);
        let (outcome, _) = search(objective, &cfg, &mut ledger, 7)?;
        println!("best {:.6} after {} hops", outcome.best, outcome.hops);
        Ok(())
    }
}

fn two_waters() -> (Vec<u32>, Vec<Vec<usize>>) {
    (vec![8, 1, 1, 8, 1, 1], vec![vec![0, 1, 2], vec![3, 4, 5]])
}

fn rust_blocks(org: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut lines = org.lines();
    while let Some(line) = lines.next() {
        if line.trim() == "#+begin_src rust" {
            let body: Vec<&str> = lines
                .by_ref()
                .take_while(|l| l.trim() != "#+end_src")
                .collect();
            blocks.push(body.join("\n"));
        }
    }
    blocks
}

fn without_indent(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn every_rust_fragment_on_the_page_is_compiled_here() {
    let blocks = rust_blocks(PAGE);
    assert!(blocks.len() >= 3, "found {} Rust blocks", blocks.len());
    let this = without_indent(THIS);
    for block in blocks {
        assert!(
            this.contains(&without_indent(&block)),
            "not compiled by this test:\n{block}"
        );
    }
}

#[test]
fn the_function_on_the_page_runs_a_search() {
    let (species, groups) = two_waters();
    let pot = PairPotential::lennard_jones(species.len());
    page::relax_with_anc(&pot, species, groups, 400).expect("finite start");
}

#[test]
fn the_one_line_alternatives_compile_against_the_same_config() {
    let (species, groups) = two_waters();
    let mut cfg = Config::recommended_molecular(species, groups, 1.0);
    cfg.local_step = LocalStep::Anc {
        model: ModelHessian::Swart,
        units: AncUnits::ATOMIC,
    };
    assert!(cfg.resolved_json().unwrap().contains(r#""model":"swart""#));
    cfg.local_step = LocalStep::Lbfgs;
    assert!(!cfg.resolved_json().unwrap().contains("local_step"));
}

// featomic switches the molecular keying, which changes both hashes.
#[cfg(not(feature = "featomic"))]
#[test]
fn the_hashes_the_page_quotes_are_those_of_two_waters() {
    let (species, groups) = two_waters();
    let mut cfg = Config::recommended_molecular(species, groups, 1.0);
    let lbfgs = cfg.resolved_sha256().unwrap();
    cfg.local_step = LocalStep::ANC_EV_ANGSTROM;
    let anc = cfg.resolved_sha256().unwrap();
    assert_eq!(
        lbfgs,
        "ade417d71a6712a43cca9456accff1a1b5d1a865ee773dc6813795fe78790671"
    );
    assert_eq!(
        anc,
        "5153a56bdb5079dc789e9d65da591bd186df7c1a93f0797e19814707dfc24c9d"
    );
    assert!(PAGE.contains(&lbfgs[..8]) && PAGE.contains(&anc[..8]));
}
