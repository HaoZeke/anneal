//! The superbasin tracker records the transitions a chain proposes.

use anneal_core::methods::cluster_hopping::{Config, Ledger, run};
use ndarray::{Array1, ArrayView1, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

const SCREEN: usize = 2;
const FULL: usize = 3;

fn bare(n: usize) -> Config {
    let mut cfg = Config::for_cluster(n);
    cfg.angular_moves = false;
    cfg.two_phase = None;
    cfg.surfaces.clear();
    cfg.return_screen = false;
    cfg.bayes_screen = false;
    cfg.screen_margin = f64::INFINITY;
    cfg.screen_steps = SCREEN;
    cfg.relax_steps = FULL;
    cfg.polish_records = 0;
    cfg.escape_on_stall = false;
    cfg.symmetrise_on_stall = false;
    cfg.superbasin_report = true;
    cfg
}

#[test]
fn a_proposal_into_a_known_basin_is_a_transition() {
    let triangle: Array1<f64> = array![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let line: Array1<f64> = array![-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    // Every quench reaches one of two structures, alternately, so from the
    // second hop on each proposal reaches the basin the chain just left.
    let mut full_quenches = 0usize;
    let mut relax = |ledger: &mut Ledger, _: ArrayView1<f64>, steps: usize| {
        let _ = ledger.charge();
        let target = if full_quenches.is_multiple_of(2) {
            &triangle
        } else {
            &line
        };
        let out = target.clone();
        if steps == FULL {
            full_quenches += 1;
        }
        (0.0, out)
    };
    let mut ledger = Ledger::new(200);
    let mut rng = StdRng::seed_from_u64(5);
    let out = run(&bare(3), triangle.view(), &mut ledger, &mut relax, &mut rng).expect("finite");
    assert!(out.hops >= 4, "the chain hopped {} times", out.hops);
    let sb = out.superbasin.expect("the report is on");
    assert_eq!(sb.nodes, 2);
    assert_eq!(sb.edges, 2, "both directions between the two basins");
}
