// Copyright (c) anneal developers

//! The minima file must contain every minimum the ledger records.

use anneal_core::methods::cluster_hopping::{Config, Ledger, run};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicU32, Ordering};

#[test]
fn a_recorded_quench_the_chain_rejects_reaches_the_dump() {
    let path = "dump-on-record.txt";
    let _ = std::fs::remove_file(path);
    // The driver reads the path from the environment. This process is the
    // test binary, and the variable is cleared before the test returns.
    // SAFETY: this test is the only thread, and it removes the variable
    // before returning.
    unsafe {
        std::env::set_var("ANNEAL_MIN_DUMP", "dump-on-record-missing/nope.txt");
    }
    let mut refused = Ledger::new(1);
    refused.record(-1.0, Array1::zeros(12).view());
    assert!(
        !std::path::Path::new("dump-on-record-missing/nope.txt").exists(),
        "a refused open must not create a minima file"
    );
    unsafe {
        std::env::set_var("ANNEAL_MIN_DUMP", path);
    }
    let mut nonfinite = Ledger::new(1);
    nonfinite.record(f64::NAN, Array1::zeros(12).view());
    let nonfinite_text = std::fs::read_to_string(path).unwrap_or_default();
    assert!(
        nonfinite_text.lines().all(|line| line.is_empty()),
        "a non-finite record must not be written, saw {nonfinite_text}"
    );
    let _ = std::fs::remove_file(path);

    let mut cfg = Config::for_cluster(4);
    cfg.bayes_screen = false;
    cfg.return_screen = false;
    cfg.escape_on_stall = false;
    cfg.symmetrise_on_stall = false;
    cfg.trail_on_stall = false;
    cfg.restart_on_stall = false;
    cfg.orbit_complete_on_new = false;
    cfg.tabu_on_stall = false;
    cfg.allocate_moves = false;
    cfg.minima_hopping = false;
    cfg.max_hops = Some(4);
    cfg.replicas = 1;

    let mut ledger = Ledger::new(64);
    // Below the first quench and above the second, so the hop that sets
    // the best stops at the record, before the acceptance test.
    ledger.stop_below(-1.5);
    let calls = AtomicU32::new(0);
    let mut relax = |ledger: &mut Ledger, state: ArrayView1<f64>, _steps: usize| {
        let index = calls.fetch_add(1, Ordering::Relaxed);
        if !ledger.charge() {
            return (f64::NAN, state.to_owned());
        }
        (-1.0 - f64::from(index), state.to_owned())
    };
    let start = Array1::from(vec![
        0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.5, 0.0, 1.5, 1.5, 0.0,
    ]);
    let mut rng = StdRng::seed_from_u64(4);
    let outcome = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng).expect("finite start");

    unsafe {
        std::env::remove_var("ANNEAL_MIN_DUMP");
    }

    assert!(
        outcome.best < -1.5,
        "the search recorded a quench below the stop"
    );
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut rows = Vec::new();
    for line in text.lines().filter(|line| !line.is_empty()) {
        let mut fields = line.split_whitespace();
        let energy = fields
            .next()
            .and_then(|field| field.parse::<f64>().ok())
            .expect("energy column");
        let accepted = fields.next().expect("acceptance column");
        assert!(
            accepted == "0" || accepted == "1",
            "acceptance is 0 or 1, saw {accepted}"
        );
        rows.push((energy, accepted.to_owned()));
    }
    assert!(
        !rows.is_empty(),
        "a recorded minimum is missing from the dump"
    );
    let (dump_best, flag) = rows
        .iter()
        .min_by(|left, right| left.0.partial_cmp(&right.0).expect("finite"))
        .expect("row");
    assert_eq!(
        format!("{:.8}", outcome.best),
        format!("{:.8}", dump_best),
        "dump minimum matches the printed best"
    );
    assert_eq!(flag, "0", "the quench that set the best was not accepted");
    assert!(
        rows.iter().any(|(_, accepted)| accepted == "1"),
        "the starting minimum is marked accepted"
    );
}
