//! The icosahedral contact test bonds the first shell at every length.
//!
//! The Cambridge Sutton-Chen tables are in units of the face-centred cubic
//! (fcc) lattice constant, so nearest neighbours sit near `1/sqrt(2)`. A
//! fixed contact cutoff of 1.35 is the Lennard-Jones first-shell gap. On
//! those clusters it also bonds the second and third shells. Each bonded
//! pair then shares a dense set of common neighbours. The 555 test reads
//! zero on an icosahedron, and before the trail cap the trail search over
//! those sets did not return.

use std::sync::mpsc;
use std::time::Duration;

use anneal_core::soap::{CONTACT_CUTOFF, SoapSpec, contact_cutoff, ih_dominated};
use anneal_core::structure::{TRAIL_BOND_CAP, cna};
use ndarray::Array1;

/// Nearest-neighbour distance of fcc at lattice constant one.
const SC_NEIGHBOUR: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// The Lennard-Jones diameter whose pair minimum is that neighbour distance.
fn sc_length() -> f64 {
    SC_NEIGHBOUR / 2.0_f64.powf(1.0 / 6.0)
}

/// A centre and its twelve icosahedral neighbours at distance `nn`.
fn icosahedron13(nn: f64) -> Array1<f64> {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let s = nn / (1.0 + phi * phi).sqrt();
    let mut x = vec![0.0; 3];
    for a in [-1.0, 1.0] {
        for b in [-1.0, 1.0] {
            x.extend([0.0, a * s, b * phi * s]);
            x.extend([a * s, b * phi * s, 0.0]);
            x.extend([b * phi * s, 0.0, a * s]);
        }
    }
    Array1::from(x)
}

/// The `n` sites of an fcc lattice nearest a site, at neighbour distance `nn`.
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
fn the_unit_length_keeps_the_lennard_jones_cutoff() {
    assert_eq!(SoapSpec::at_length(1.0), SoapSpec::default());
    assert_eq!(contact_cutoff(SoapSpec::default()), CONTACT_CUTOFF);
    let half = contact_cutoff(SoapSpec::at_length(0.5));
    assert!((half - 0.5 * CONTACT_CUTOFF).abs() < 1e-15, "{half}");
}

/// The mis-bonding. At the fixed cutoff each vertex of a Sutton-Chen
/// icosahedron bonds ten of the other eleven, and nothing reads 555.
#[test]
fn a_sutton_chen_icosahedron_is_icosahedral_at_its_own_length() {
    let x = icosahedron13(SC_NEIGHBOUR);
    let fixed = cna(x.view(), 13, CONTACT_CUTOFF);
    assert_eq!(fixed.get((5, 5, 5)), 0, "counts {:?}", fixed.counts);
    assert!(!ih_dominated(x.view(), SoapSpec::default()));

    let spec = SoapSpec::at_length(sc_length());
    let scaled = cna(x.view(), 13, contact_cutoff(spec));
    assert_eq!(scaled.get((5, 5, 5)), 12, "counts {:?}", scaled.counts);
    assert!(ih_dominated(x.view(), spec));
}

/// The hang. A Sutton-Chen fcc 38 at the fixed cutoff puts pairs past the
/// trail cap, the regime whose trail search did not return before the cap.
/// At its own length no pair comes near the cap, and the 421 signature of
/// the lattice is read.
#[test]
fn a_sutton_chen_lattice_stays_in_the_first_shell_at_its_own_length() {
    let x = fcc_ball(SC_NEIGHBOUR, 38);
    let y = x.clone();
    let fixed = within(30, move || cna(y.view(), 38, CONTACT_CUTOFF)).expect("cna did not return");
    assert!(
        fixed
            .counts
            .iter()
            .any(|((_, s, _), _)| *s > TRAIL_BOND_CAP),
        "counts {:?}",
        fixed.counts
    );
    assert_eq!(fixed.get((4, 2, 1)), 0, "counts {:?}", fixed.counts);

    let spec = SoapSpec::at_length(sc_length());
    let scaled = cna(x.view(), 38, contact_cutoff(spec));
    assert!(
        scaled.counts.iter().all(|((_, s, _), _)| *s <= 6),
        "counts {:?}",
        scaled.counts
    );
    assert!(scaled.get((4, 2, 1)) > 0, "counts {:?}", scaled.counts);
    assert!(!ih_dominated(x.view(), spec));
}
