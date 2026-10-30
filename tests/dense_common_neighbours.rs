//! Common-neighbour analysis returns on a compressed, unrelaxed lattice.
//!
//! A candidate squeezed well below the Lennard-Jones contact distance puts
//! the second and third shells inside the bond cutoff, and every bonded pair
//! then shares a dense set of common neighbours. The trail count over that
//! set is exponential in its bonds, so the call has to stop tracing rather
//! than run until the caller gives up.

use std::sync::mpsc;
use std::time::Duration;

use anneal_core::soap::{SoapSpec, ih_dominated};
use anneal_core::structure::{TRAIL_BOND_CAP, cna};
use ndarray::Array1;

/// The `n` sites of a face-centred cubic lattice nearest the origin, at
/// nearest-neighbour distance `nn`.
fn fcc_ball(nn: f64, n: usize) -> Array1<f64> {
    let a = nn * 2.0_f64.sqrt();
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut sites = Vec::new();
    for i in -3..=3 {
        for j in -3..=3 {
            for k in -3..=3 {
                for b in &basis {
                    let cell = [i as f64, j as f64, k as f64];
                    sites.push([0, 1, 2].map(|q| a * (cell[q] + b[q])));
                }
            }
        }
    }
    let r2 = |p: &[f64; 3]| p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    sites.sort_by(|p, q| r2(p).total_cmp(&r2(q)));
    Array1::from_iter(sites.iter().take(n).flatten().copied())
}

fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(secs)).ok()
}

#[test]
fn a_compressed_lattice_is_not_icosahedral_and_returns() {
    let x = fcc_ball(0.75, 38);
    let answer = within(30, move || ih_dominated(x.view(), SoapSpec::default()));
    assert_eq!(
        answer,
        Some(false),
        "ih_dominated did not return a defined answer"
    );
}

#[test]
fn untraced_pairs_carry_a_zero_trail() {
    let x = fcc_ball(0.75, 38);
    let counts = within(30, move || cna(x.view(), 38, 1.35)).expect("cna did not return");
    let untraced: usize = counts
        .counts
        .iter()
        .filter(|((_, s, _), _)| *s > TRAIL_BOND_CAP)
        .map(|((_, _, t), c)| {
            assert_eq!(
                *t, 0,
                "a pair past the cap was traced, counts {:?}",
                counts.counts
            );
            *c
        })
        .sum();
    assert!(
        untraced > 0,
        "the lattice was not dense enough, counts {:?}",
        counts.counts
    );
}

#[test]
fn an_uncompressed_lattice_keeps_its_signatures() {
    let x = fcc_ball(2.0_f64.powf(1.0 / 6.0), 38);
    let counts = cna(x.view(), 38, 1.35);
    assert!(counts.get((4, 2, 1)) > 0, "counts {:?}", counts.counts);
    assert!(
        counts
            .counts
            .iter()
            .all(|((_, s, _), _)| *s <= TRAIL_BOND_CAP),
        "counts {:?}",
        counts.counts
    );
}
