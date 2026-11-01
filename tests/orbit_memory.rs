//! Orbit completion under `Config::orbit_memory`, as `Config::recommended`
//! runs it.
//!
//! Off is the default and leaves every preset's resolved configuration as it
//! was. Under either memory a chain adopts a completion only downhill from
//! the minimum it started from. Under `Energy` it never adopts a completion
//! onto an energy it has kept, and a chain stuck repeating a completion
//! stops proposing it.

use anneal_core::methods::cluster_hopping::{Config, Ledger, OrbitMemory};
use anneal_core::methods::cluster_search;
use anneal_core::potentials::PairPotential;

const SAME_ENERGY: f64 = 1e-6;

struct Run {
    quenched: usize,
    skipped: usize,
    adopted: Vec<(f64, f64)>,
}

fn run(memory: OrbitMemory, seed: u64) -> Run {
    let mut cfg = Config::recommended(31);
    assert!(cfg.orbit_complete_on_new);
    cfg.orbit_memory = memory;
    let potential = PairPotential::lennard_jones(31);
    let mut ledger = Ledger::new(60_000);
    let (out, _) = cluster_search::search(&potential, &cfg, &mut ledger, seed).expect("search");
    Run {
        quenched: out.orbits.0,
        skipped: out.orbits_skipped,
        adopted: out
            .accepted_transitions
            .iter()
            .filter(|t| t.action == "orbit-completion")
            .map(|t| (t.from_energy, t.to_energy))
            .collect(),
    }
}

#[test]
fn presets_leave_the_memory_off_and_their_digests_unchanged() {
    for cfg in [
        Config::recommended(38),
        Config::communicating(38),
        Config::for_cluster(38),
    ] {
        assert_eq!(cfg.orbit_memory, OrbitMemory::Off);
        assert!(!cfg.resolved_json().expect("json").contains("orbit_memory"));
        let mut on = cfg.clone();
        on.orbit_memory = OrbitMemory::Structure;
        assert!(
            on.resolved_json()
                .expect("json")
                .contains("\"orbit_memory\":\"structure\"")
        );
        assert_ne!(on.resolved_sha256().ok(), cfg.resolved_sha256().ok());
    }
}

#[test]
fn under_the_energy_memory_completions_go_downhill_onto_energies_not_kept() {
    let mut quenched = 0;
    let mut skipped = 0;
    let mut adopted_total = 0;
    for seed in 0..4 {
        let r = run(OrbitMemory::Energy, seed);
        quenched += r.quenched;
        skipped += r.skipped;
        adopted_total += r.adopted.len();
        for (i, &(from, to)) in r.adopted.iter().enumerate() {
            assert!(
                to < from - SAME_ENERGY,
                "seed {seed}: completion {i} went from {from} to {to}"
            );
            for &(earlier_from, earlier_to) in &r.adopted[..i] {
                assert!(
                    (to - earlier_to).abs() > SAME_ENERGY,
                    "seed {seed}: completion {i} reached {to} again"
                );
                assert!(
                    (from - earlier_from).abs() > SAME_ENERGY,
                    "seed {seed}: completion {i} started from {from} again"
                );
                assert!(
                    (from - earlier_to).abs() > SAME_ENERGY,
                    "seed {seed}: completion {i} completed its own result {from}"
                );
            }
        }
    }
    assert!(quenched > 0, "no completion was quenched");
    assert!(adopted_total > 0, "no completion was adopted");
    assert!(skipped > 0, "no repeated completion was refused");
}

#[test]
fn under_the_structure_memory_completions_go_downhill_and_none_is_skipped() {
    let mut quenched = 0;
    let mut adopted_total = 0;
    for seed in 0..4 {
        let r = run(OrbitMemory::Structure, seed);
        quenched += r.quenched;
        adopted_total += r.adopted.len();
        assert_eq!(r.skipped, 0, "seed {seed}");
        for (i, &(from, to)) in r.adopted.iter().enumerate() {
            assert!(
                to < from - SAME_ENERGY,
                "seed {seed}: completion {i} went from {from} to {to}"
            );
        }
    }
    assert!(quenched > 0, "no completion was quenched");
    assert!(adopted_total > 0, "no completion was adopted");
}
