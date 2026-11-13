use super::*;
use crate::methods::cluster_search::search;
use crate::methods::warm_lbfgs::WarmLbfgs;
use crate::potentials::{PairKind, PairPotential};
use rand::SeedableRng;
use rand::rngs::StdRng;

const CAP_ENERGY: f64 = -10.0;

/// A separated simple-cubic start of `n` points.
fn lattice(n: usize, spacing: f64) -> Array1<f64> {
    let side = (n as f64).cbrt().ceil() as usize;
    let mut out = Vec::with_capacity(3 * n);
    'fill: for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                if out.len() == 3 * n {
                    break 'fill;
                }
                out.extend([i as f64 * spacing, j as f64 * spacing, k as f64 * spacing]);
            }
        }
    }
    Array1::from(out)
}

fn walkless(n: usize, long_ranged: bool) -> Config {
    let mut cfg = Config::for_cluster(n);
    cfg.long_ranged = long_ranged;
    cfg.max_hops = Some(0);
    cfg
}

struct Run {
    out: Outcome,
    spent: usize,
    quenched: Vec<Array1<f64>>,
}

/// Runs with a mock potential. The start quenches to energy 0 with a zero
/// gradient. Every other geometry quenches in place to `cap_energy`, and its
/// gradient is zero only when `cap_converged`.
fn mocked(
    cfg: &Config,
    budget: usize,
    cap_energy: f64,
    cap_converged: bool,
    target: Option<f64>,
) -> Run {
    let n = cfg.n_points;
    let start = lattice(n, 1.1 * cfg.length_scale);
    let held = start.clone();
    let mut quenched = Vec::new();
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _steps: usize| {
        if !ledger.charge() {
            return (f64::INFINITY, x.to_owned());
        }
        if x == held.view() {
            return (0.0, x.to_owned());
        }
        quenched.push(x.to_owned());
        (cap_energy, x.to_owned())
    };
    let at_start = start.clone();
    let mut grad = move |ledger: &mut Ledger, x: ArrayView1<f64>| {
        if !ledger.charge() {
            return None;
        }
        let converged = x == at_start.view() || cap_converged;
        Some(Array1::from_elem(
            x.len(),
            if converged { 0.0 } else { 1.0 },
        ))
    };
    let mut ledger = Ledger::new(budget);
    if let Some(limit) = target {
        ledger.stop_below(limit);
    }
    let mut rng = StdRng::seed_from_u64(11);
    let out = run_with_gradient(
        cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    Run {
        out,
        spent: ledger.spent(),
        quenched,
    }
}

fn no_row_below_the_best(out: &Outcome) {
    assert!(
        out.improvements
            .iter()
            .all(|(_, _, _, energy)| *energy >= out.best),
        "best {} but the trace holds {:?}",
        out.best,
        out.improvements
    );
}

#[test]
fn an_unvalidated_cap_is_not_a_reported_improvement() {
    let run = mocked(&walkless(13, true), 1_000, CAP_ENERGY, false, None);
    assert_eq!(run.quenched.len(), 1, "one cap quench");
    assert_eq!(run.out.best, 0.0);
    no_row_below_the_best(&run.out);
}

#[test]
fn a_budget_that_ends_before_the_cap_gradient_records_nothing() {
    let off = mocked(&walkless(13, false), 1_000, CAP_ENERGY, true, None);
    // The cap quench itself is paid for. The validation gradient is not.
    let run = mocked(&walkless(13, true), off.spent + 1, CAP_ENERGY, true, None);
    assert_eq!(run.quenched.len(), 1);
    assert_eq!(run.spent, off.spent + 1);
    assert_eq!(run.out.best, 0.0);
    no_row_below_the_best(&run.out);
    // With nothing left after the start, the cap is not quenched at all.
    let dry = mocked(&walkless(13, true), off.spent, CAP_ENERGY, true, None);
    assert!(dry.quenched.is_empty());
    assert_eq!(dry.spent, off.spent);
}

#[test]
fn a_validated_cap_is_the_best_and_its_last_improvement() {
    let n = 13;
    let cfg = walkless(n, true);
    let run = mocked(&cfg, 1_000, CAP_ENERGY, true, None);
    assert_eq!(run.out.best, CAP_ENERGY);
    let (_, _, _, last) = *run.out.improvements.last().expect("a trace row");
    assert_eq!(last, CAP_ENERGY);
    no_row_below_the_best(&run.out);
    let cap = binary_geometry::mackay_cap(n, cfg.length_scale);
    assert_eq!(run.quenched[0], cap);
    assert_eq!(run.out.best_state.as_ref(), Some(&cap));
}

#[test]
fn a_cap_at_the_target_ends_the_run_before_a_hop() {
    let mut cfg = walkless(13, true);
    cfg.max_hops = None;
    let run = mocked(&cfg, 10_000, CAP_ENERGY, true, Some(CAP_ENERGY + 1.0));
    assert_eq!(run.out.best, CAP_ENERGY);
    assert_eq!(run.out.hops, 0);
    assert_eq!(run.quenched.len(), 1);
}

#[test]
fn a_non_finite_cap_energy_is_not_recorded() {
    for energy in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
        let run = mocked(&walkless(13, true), 1_000, energy, true, None);
        assert_eq!(run.quenched.len(), 1);
        assert_eq!(run.out.best, 0.0, "cap energy {energy}");
        no_row_below_the_best(&run.out);
    }
}

#[test]
fn the_flag_is_inert_past_the_two_shell_body() {
    let off = mocked(&walkless(56, false), 1_000, CAP_ENERGY, true, None);
    let on = mocked(&walkless(56, true), 1_000, CAP_ENERGY, true, None);
    assert!(on.quenched.is_empty());
    assert_eq!(on.spent, off.spent);
    assert_eq!(on.out.best.to_bits(), off.out.best.to_bits());
}

#[test]
fn the_flag_off_never_quenches_a_cap() {
    for n in [2, 13, 38, 55] {
        let run = mocked(&walkless(n, false), 1_000, CAP_ENERGY, true, None);
        assert!(run.quenched.is_empty(), "n {n}");
        assert_eq!(run.out.best, 0.0);
    }
}

#[test]
fn a_two_atom_cap_is_one_pair_at_the_length_scale() {
    let cfg = walkless(2, true);
    let run = mocked(&cfg, 1_000, CAP_ENERGY, true, None);
    let pair = &run.quenched[0];
    assert_eq!(pair.len(), 6);
    let distance = min_pair_distance(pair.view());
    assert!(
        (distance - cfg.length_scale).abs() < 1e-12,
        "pair at {distance}"
    );
}

fn quench(pot: &PairPotential, x: ArrayView1<f64>) -> f64 {
    let mut solver = WarmLbfgs::default();
    solver.gtol = 1e-8;
    let (energy, _, _) = solver.minimize(x, 10_000, |v| Some(pot.value_and_gradient(v)));
    energy
}

/// Cambridge Cluster Database, Morse rho = 6, as `examples/h2h_morse.rs`.
const M38_RHO6: f64 = -157.477108;
const M55_RHO6: f64 = -250.286609;

/// At rho = 6 the cap quenches to the minimum at 38 and 55.
#[test]
fn the_rho6_cap_quenches_to_the_minimum_at_38_and_55() {
    for (n, reference) in [(38, M38_RHO6), (55, M55_RHO6)] {
        let pot = PairPotential::morse(n, 6.0);
        let cap = binary_geometry::mackay_cap(n, 1.0);
        let energy = quench(&pot, cap.view());
        assert!(
            energy <= reference + 1e-3,
            "n {n}: cap at {energy}, table {reference}"
        );
    }
}

/// One quench before any hop is the M38 minimum, through the same search a
/// caller runs. Without the flag the walkless run keeps its random start.
#[test]
fn a_walkless_rho6_search_reaches_m38_through_the_cap() {
    let pot = PairPotential::morse(38, 6.0);
    assert!(PairKind::Morse { rho: 6.0 }.second_neighbour_energy() < -0.05);
    let mut cfg = Config::recommended(38);
    cfg.long_ranged = true;
    cfg.max_hops = Some(0);
    let mut ledger = Ledger::new(20_000);
    let (out, _) = search(&pot, &cfg, &mut ledger, 3).expect("finite start");
    assert!(out.best <= M38_RHO6 + 1e-3, "best {}", out.best);
    let (_, _, _, last) = *out.improvements.last().expect("a trace row");
    assert_eq!(last, out.best);
    cfg.long_ranged = false;
    let mut ledger = Ledger::new(20_000);
    let (plain, _) = search(&pot, &cfg, &mut ledger, 3).expect("finite start");
    assert!(
        plain.best > M38_RHO6 + 1e-3,
        "a random start is not the minimum"
    );
}

fn sites(flat: &Array1<f64>) -> Vec<[f64; 3]> {
    flat.as_slice()
        .expect("contiguous")
        .chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[test]
fn every_cap_up_to_the_complete_body_is_a_compact_packing() {
    let nearest = 1.0;
    for n in 1..=55 {
        let cap = binary_geometry::mackay_cap(n, nearest);
        assert_eq!(cap.len(), 3 * n, "n {n}");
        assert!(cap.iter().all(|v| v.is_finite()), "n {n}");
        if n < 2 {
            continue;
        }
        assert!(min_pair_distance(cap.view()) >= 0.9 * nearest, "n {n}");
        let points = sites(&cap);
        for (i, site) in points.iter().enumerate() {
            let closest = points
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, other)| distance(site, other))
                .fold(f64::INFINITY, f64::min);
            assert!(
                closest <= 1.25 * nearest,
                "n {n}: site {i} is {closest} from the rest"
            );
        }
    }
}

#[test]
fn a_cap_outside_the_body_or_without_a_length_is_empty() {
    for n in [0, 56, 147] {
        assert!(binary_geometry::mackay_cap(n, 1.0).is_empty(), "n {n}");
    }
    for nearest in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            binary_geometry::mackay_cap(13, nearest).is_empty(),
            "{nearest}"
        );
        assert!(
            binary_geometry::mackay_cut(13, nearest).is_empty(),
            "{nearest}"
        );
    }
}

#[test]
fn the_complete_cap_is_the_centred_mackay_body() {
    let mut cap = sites(&binary_geometry::mackay_cap(55, 1.0));
    let mut cut = sites(&binary_geometry::mackay_cut(55, 1.0));
    let order = |left: &[f64; 3], right: &[f64; 3]| {
        left.iter()
            .zip(right)
            .map(|(a, b)| a.total_cmp(b))
            .find(|o| o.is_ne())
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    cap.sort_by(order);
    cut.sort_by(order);
    for (a, b) in cap.iter().zip(&cut) {
        assert!(distance(a, b) < 1e-12, "{a:?} against {b:?}");
    }
}

/// The thirteen sites nearest the vertex are that vertex and its twelve
/// Mackay neighbours: seven radial at the shell radius and five tangential
/// at the edge, so the cap is centred on the vertex.
#[test]
fn the_thirteen_atom_cap_surrounds_the_vertex() {
    let points = sites(&binary_geometry::mackay_cap(13, 1.0));
    let centred = points.iter().any(|centre| {
        let mut shell: Vec<f64> = points
            .iter()
            .map(|other| distance(centre, other))
            .filter(|d| *d > 1e-9)
            .collect();
        shell.sort_by(f64::total_cmp);
        shell.len() == 12 && shell[11] < 1.06 * shell[0]
    });
    assert!(centred, "no site has twelve neighbours in one shell");
}

proptest::proptest! {
    #[test]
    fn a_cap_scales_with_its_length(n in 1usize..=55, scale in 1.0e-3f64..1.0e3) {
        let unit = binary_geometry::mackay_cap(n, 1.0);
        let scaled = binary_geometry::mackay_cap(n, scale);
        proptest::prop_assert_eq!(unit.len(), scaled.len());
        for (a, b) in unit.iter().zip(scaled.iter()) {
            proptest::prop_assert!((a * scale - b).abs() <= 1e-12 * scale.max(1.0) * (1.0 + a.abs()));
        }
    }
}
