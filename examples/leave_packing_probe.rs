//! Does an occupancy Leave actually install a packing from the LJ75 ico well?
//!
//!     leave_packing_probe [LEAVES] [RELAX_STEPS]
//!
//! Three generators are run from the same icosahedral minimum, each `LEAVES`
//! times, and each result is classified against the two sealed fixtures by
//! single linkage at [`anneal_core::catalog::PACKING_LINK`]:
//!
//! * `cartesian`, the old Leave: a SoftSaddle covering direction placed at
//!   Cartesian RMSD 0.35, quenched raw.
//! * `ridge`, that start climbed with ART / SoftSaddle MMF
//!   ([`anneal_core::methods::activation::activate_from_origin`]) until
//!   the force along the mode flips, then quenched raw from the overshoot.
//! * `armed`, that same start under the packing invert.
//! * `ladder`, the packing-map ladder under the invert.
//!
//! Reported per generator: how many left the icosahedral packing, how many
//! landed under the icosahedral floor, and how many landed in the Marks
//! community. Wales and Doye put the ico-Marks barriers at 8.69 and 7.48
//! epsilon, so a generator that never leaves is measuring the cap, not the
//! landscape.

use anneal_core::catalog::{PACKING_LINK, PackingBook, leaves_packing, packing_link_labels};
use anneal_core::known_basin;
use anneal_core::methods::activation::{Activation, activate_from_origin};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{PairKind, PairPotential};
use ndarray::{Array1, ArrayView1};

fn load_xyz(text: &str) -> Array1<f64> {
    let coordinates = text
        .lines()
        .skip(2)
        .filter(|line| !line.trim().is_empty())
        .flat_map(|line| {
            line.split_whitespace()
                .skip(1)
                .take(3)
                .map(|value| value.parse::<f64>().expect("fixture coordinate parses"))
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<f64>>();
    Array1::from(coordinates)
}

fn quench(potential: &PairPotential, x: ArrayView1<f64>, steps: usize) -> Array1<f64> {
    let mut opt = WarmLbfgs::default();
    let fg = |v: ArrayView1<f64>| {
        let (energy, gradient) = potential.value_and_gradient(v);
        Some(known_basin::effective(v, energy, gradient))
    };
    if known_basin::is_armed() {
        known_basin::step_xtsci(&mut opt, x, steps, fg).1
    } else {
        opt.minimize(x, steps, fg).1
    }
}

/// Which sealed community a structure lands in: 0 ico, 1 Marks, 2 neither.
fn community_of(ico: &[f64], marks: &[f64], trial: &[f64]) -> usize {
    let mut book = PackingBook::default();
    for state in [ico, marks, trial] {
        book.observe(state);
    }
    let histograms: Vec<Vec<f64>> = [ico, marks, trial]
        .into_iter()
        .filter_map(|state| book.histogram(state))
        .collect();
    if histograms.len() != 3 {
        return 2;
    }
    let labels = packing_link_labels(&histograms, PACKING_LINK);
    if labels[2] == labels[0] {
        0
    } else if labels[2] == labels[1] {
        1
    } else {
        2
    }
}

#[derive(Default)]
struct Tally {
    left: usize,
    below_ico: usize,
    marks: usize,
    novel: usize,
    best: f64,
    rungs: Vec<usize>,
}

fn report(label: &str, tally: &Tally, leaves: usize) {
    let mean_rung = if tally.rungs.is_empty() {
        f64::NAN
    } else {
        tally.rungs.iter().sum::<usize>() as f64 / tally.rungs.len() as f64
    };
    println!(
        "{{\"kind\":\"leave_probe\",\"generator\":\"{label}\",\"leaves\":{leaves},\"left_packing\":{},\"below_ico\":{},\"marks\":{},\"novel\":{},\"best\":{:.6},\"mean_rung\":{mean_rung:.2}}}",
        tally.left, tally.below_ico, tally.marks, tally.novel, tally.best
    );
}

fn main() {
    let leaves: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(48);
    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(600);
    let only_ridge = std::env::args().nth(3).as_deref() == Some("ridge");
    let ico_fixture = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
    let marks_fixture = load_xyz(include_str!("../tests/fixtures/lj75_marks.xyz"));
    let potential = PairPotential::new(75, PairKind::LennardJones, 40.0);
    let ico = quench(&potential, ico_fixture.view(), 600);
    let marks = quench(&potential, marks_fixture.view(), 600);
    let ico_energy = potential.value_and_gradient(ico.view()).0;
    let marks_energy = potential.value_and_gradient(marks.view()).0;
    let ico_slice = ico.as_slice().expect("state is contiguous").to_vec();
    let marks_slice = marks.as_slice().expect("state is contiguous").to_vec();
    println!(
        "{{\"kind\":\"leave_probe_floors\",\"ico\":{ico_energy:.6},\"marks\":{marks_energy:.6},\"link\":{PACKING_LINK}}}"
    );

    // The ladder is walked in barrier, so it needs the curvature of the
    // well it leaves and the depth that scales the barrier. Both measured.
    let curvature = anneal_core::curvature::curvature_features(
        ico.view(),
        |v: ArrayView1<f64>| Some(potential.value_and_gradient(v).1),
        64,
        1e-4,
    )
    .map_or(0.0, |features| features.lambda_min);
    let depth = ico_energy.abs() / (ico.len() / 3).max(1) as f64;
    println!(
        "{{\"kind\":\"leave_probe_rung\",\"lambda_min\":{curvature:.4},\"depth_per_atom\":{depth:.4},\"rung0\":{:.4},\"rung_top\":{:.4}}}",
        known_basin::rung_rmsd(
            curvature,
            ico.len() / 3,
            known_basin::rung_barrier(depth, 0)
        )
        .unwrap_or(f64::NAN),
        known_basin::rung_rmsd(
            curvature,
            ico.len() / 3,
            known_basin::rung_barrier(depth, known_basin::LEAVE_RUNGS - 1)
        )
        .unwrap_or(f64::NAN)
    );
    let references = vec![ico_slice.clone()];
    let mut cartesian = Tally {
        best: ico_energy,
        ..Tally::default()
    };
    let mut ridge = Tally {
        best: ico_energy,
        ..Tally::default()
    };
    let mut armed = Tally {
        best: ico_energy,
        ..Tally::default()
    };
    let mut ladder = Tally {
        best: ico_energy,
        ..Tally::default()
    };

    let mut classify = |label: &str,
                        index: usize,
                        tally: &mut Tally,
                        trial: &Array1<f64>,
                        rung: Option<usize>| {
        let Some(slice) = trial.as_slice() else {
            return;
        };
        let energy = potential.value_and_gradient(trial.view()).0;
        if !energy.is_finite() {
            return;
        }
        let left = leaves_packing(&ico_slice, slice, &[]);
        if left {
            tally.left += 1;
            if let Some(rung) = rung {
                tally.rungs.push(rung);
            }
        }
        if energy < ico_energy - 1e-6 {
            tally.below_ico += 1;
        }
        let community = community_of(&ico_slice, &marks_slice, slice);
        match community {
            1 => tally.marks += 1,
            2 => tally.novel += 1,
            _ => {}
        }
        if energy < tally.best {
            tally.best = energy;
        }
        println!(
            "{{\"kind\":\"leave\",\"generator\":\"{label}\",\"index\":{index},\"rung\":{},\"energy\":{energy:.6},\"left\":{left},\"community\":{community}}}",
            rung.map_or_else(|| "null".to_owned(), |value| value.to_string())
        );
    };

    for index in 0..leaves {
        // Old Leave: Cartesian covering point at 0.35, raw quench.
        let direction = anneal_core::hypersphere::cover_direction(
            anneal_core::hypersphere::default_cover_size(),
            ico.len(),
            index,
        );
        let start = Array1::from(anneal_core::hypersphere::place_around(
            &ico_slice,
            &direction,
            known_basin::LEAVE_RUNG_RMSD,
            None,
        ));
        let trial = quench(&potential, start.view(), steps);
        classify("cartesian", index, &mut cartesian, &trial, None);

        // Same start, climb the local ridge, quench from the overshoot.
        let climbed = activate_from_origin(
            start.view(),
            ico.view(),
            |v| Some(potential.value_and_gradient(v).1),
            &Activation::default(),
        );
        match &climbed {
            Some(outcome) => println!(
                "{{\"kind\":\"ridge_climb\",\"index\":{index},\"crossed\":{},\"lambda\":{:.6},\"steps\":{},\"evaluations\":{}}}",
                outcome.crossed, outcome.lambda, outcome.steps, outcome.evaluations
            ),
            None => println!(
                "{{\"kind\":\"ridge_climb\",\"index\":{index},\"crossed\":false,\"failed\":true}}"
            ),
        }
        let ridge_start = climbed
            .and_then(|outcome| outcome.crossed.then_some(outcome.state))
            .unwrap_or_else(|| start.clone());
        let trial = quench(&potential, ridge_start.view(), steps);
        classify("ridge", index, &mut ridge, &trial, None);
        if only_ridge {
            let _ = std::io::Write::flush(&mut std::io::stdout());
            continue;
        }

        // Same start, packing invert armed.
        known_basin::arm_leave(ico.view(), known_basin::LEAVE_RUNG_RMSD, &references);
        let trial = quench(&potential, start.view(), steps);
        known_basin::disarm();
        classify("armed", index, &mut armed, &trial, None);

        // Packing-map ladder under the invert, hill width set by the
        // curvature rather than by the old constant.
        let sigma = known_basin::rung_rmsd(
            curvature,
            ico.len() / 3,
            known_basin::rung_barrier(depth, 0),
        )
        .unwrap_or(known_basin::LEAVE_RUNG_RMSD);
        known_basin::arm_leave(ico.view(), sigma, &references);
        let walked = known_basin::leave_packing_ladder(
            ico.view(),
            index,
            &references,
            None,
            None,
            depth,
            known_basin::LEAVE_RUNGS,
            |trial| {
                let relaxed = quench(&potential, trial, steps);
                (potential.value_and_gradient(relaxed.view()).0, relaxed)
            },
            |v: ArrayView1<f64>| Some(potential.value_and_gradient(v).0),
        );
        known_basin::disarm();
        if let Some((_, trial, rung)) = walked {
            // Raw polish: the ladder walked on E+V.
            let polished = quench(&potential, trial.view(), steps);
            classify("ladder", index, &mut ladder, &polished, Some(rung));
        } else {
            println!(
                "{{\"kind\":\"leave\",\"generator\":\"ladder\",\"index\":{index},\"rung\":null,\"refused\":true}}"
            );
        }
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    // The driver's own path: a barrier-targeted rung, the min-mode climb
    // that follows it, the energy gate on the crossing, then the quench.
    // Reported per leave with the curvature the climb ended on, because a
    // crossing at 1e13 is a crushed cluster and not a saddle.
    let mut gated = Tally {
        best: ico_energy,
        ..Tally::default()
    };
    let mut melts = 0usize;
    for index in 0..leaves {
        let start = known_basin::leave_packing_rung_to(
            ico.view(),
            index,
            known_basin::rung_barrier(depth, 0),
            &references,
            None,
            None,
            |v: ArrayView1<f64>| Some(potential.value_and_gradient(v).0),
        );
        let act = anneal_core::methods::activation::Activation::default();
        let outcome = anneal_core::methods::activation::activate_from_origin(
            start.view(),
            ico.view(),
            |v: ArrayView1<f64>| Some(potential.value_and_gradient(v).1),
            &act,
        );
        let ceiling = ico_energy + known_basin::LEAVE_WALK_CLIMB * depth;
        let (climbed, lambda, crossed) = match outcome {
            Some(o) => {
                let energy = potential.value_and_gradient(o.state.view()).0;
                (
                    if o.crossed && energy.is_finite() && energy <= ceiling {
                        o.state
                    } else {
                        if o.crossed {
                            melts += 1;
                        }
                        start.clone()
                    },
                    o.lambda,
                    o.crossed,
                )
            }
            None => (start.clone(), f64::NAN, false),
        };
        let trial = quench(&potential, climbed.view(), steps);
        println!(
            "{{\"kind\":\"leave\",\"generator\":\"gated\",\"index\":{index},\"lambda\":{lambda:.3e},\"crossed\":{crossed},\"melts\":{}}}",
            melts
        );
        classify("gated", index, &mut gated, &trial, None);
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    println!(
        "{{\"kind\":\"leave_probe_melts\",\"refused_crossings\":{melts},\"leaves\":{leaves}}}"
    );
    report("gated", &gated, leaves);

    report("cartesian", &cartesian, leaves);
    report("ridge", &ridge, leaves);
    if only_ridge {
        return;
    }
    report("armed", &armed, leaves);
    report("ladder", &ladder, leaves);

    // A run does not Leave the same minimum every time. It adopts what the
    // Leave installed and leaves that, with the cells it has walked on file.
    let mut chain = Tally {
        best: ico_energy,
        ..Tally::default()
    };
    let mut origin = ico.clone();
    anneal_core::catalog::set_packing_references(vec![ico_slice.clone()]);
    for index in 0..leaves {
        let references = anneal_core::catalog::packing_references();
        let sigma = known_basin::rung_rmsd(
            curvature,
            origin.len() / 3,
            known_basin::rung_barrier(depth, 0),
        )
        .unwrap_or(known_basin::LEAVE_RUNG_RMSD);
        known_basin::arm_leave(origin.view(), sigma, &references);
        let walked = known_basin::leave_packing_ladder(
            origin.view(),
            index,
            &references,
            None,
            None,
            depth,
            known_basin::LEAVE_RUNGS,
            |trial| {
                let relaxed = quench(&potential, trial, steps);
                (potential.value_and_gradient(relaxed.view()).0, relaxed)
            },
            |v: ArrayView1<f64>| Some(potential.value_and_gradient(v).0),
        );
        known_basin::disarm();
        let Some((_, trial, rung)) = walked else {
            println!(
                "{{\"kind\":\"leave\",\"generator\":\"chain\",\"index\":{index},\"rung\":null,\"refused\":true}}"
            );
            continue;
        };
        classify("chain", index, &mut chain, &trial, Some(rung));
        if let Some(installed) = trial.as_slice() {
            anneal_core::catalog::remember_packing_reference(installed);
        }
        origin = trial;
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    report("chain", &chain, leaves);
}
