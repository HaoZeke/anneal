//! A reused SCC wavefunction has to land on the same GFN2 energy as a fresh guess.

use anneal_core::xtb_session::{WARM_MOVE_A, XtbSession};

fn dimer() -> (Vec<f64>, Vec<i32>) {
    (
        vec![
            0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0, 2.9, 0.1, 0.2, 3.5, 0.7, 0.1, 3.2,
            -0.4, 0.8,
        ],
        vec![8, 1, 1, 8, 1, 1],
    )
}

fn open_at(accuracy: f64) -> Option<XtbSession> {
    match XtbSession::open_with_accuracy(accuracy) {
        Ok(session) => Some(session),
        Err(err) if err.to_string().contains("dlopen") => None,
        Err(err) => panic!("{err}"),
    }
}

fn energy(session: &mut XtbSession, positions: &[f64], numbers: &[i32]) -> f64 {
    let mut forces = vec![0.0; positions.len()];
    let cell = [0.0; 9];
    session
        .evaluate(positions, numbers, &cell, &mut forces)
        .expect("finite GFN2 point")
        .energy_hartree
}

#[test]
fn a_reused_wavefunction_matches_a_cold_gfn2_energy() {
    let Some(mut warm) = open_at(1.0) else {
        return;
    };
    let (positions, numbers) = dimer();
    let mut nudged = positions.clone();
    nudged[0] += WARM_MOVE_A * 0.05;
    let _ = energy(&mut warm, &positions, &numbers);
    let revisited = energy(&mut warm, &nudged, &numbers);
    let back = energy(&mut warm, &positions, &numbers);

    let mut cold = open_at(1.0).expect("libxtb opened once");
    let fresh = energy(&mut cold, &positions, &numbers);
    let fresh_nudged = energy(&mut cold, &nudged, &numbers);
    assert!(
        (back - fresh).abs() < 1e-8,
        "warm {back} Eh and cold {fresh} Eh differ"
    );
    assert!(
        (revisited - fresh_nudged).abs() < 1e-8,
        "warm step {revisited} Eh and cold {fresh_nudged} Eh differ"
    );

    let mut tight = open_at(0.01).expect("tight session");
    let tight_energy = energy(&mut tight, &positions, &numbers);
    assert!(
        (fresh - tight_energy).abs() < 1e-8,
        "accuracy 1 {fresh} Eh and accuracy 0.01 {tight_energy} Eh differ"
    );

    let mut far = positions.clone();
    far[0] += WARM_MOVE_A + 1.0;
    let _ = energy(&mut warm, &far, &numbers);
    let after_reset = energy(&mut warm, &positions, &numbers);
    assert!(
        (after_reset - fresh).abs() < 1e-8,
        "reset {after_reset} Eh and cold {fresh} Eh differ"
    );
}
