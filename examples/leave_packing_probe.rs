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

/// DECAF distance between two structures, in the same L1 the packing
/// grain is quoted in. `PACKING_LINK` is 0.35 and icosahedral-to-Marks
/// is 0.69, so this says how far along that road a walk actually got.
fn packing_gap(origin: &[f64], trial: &[f64]) -> f64 {
    let mut book = PackingBook::default();
    for state in [origin, trial] {
        book.observe(state);
    }
    match (book.histogram(origin), book.histogram(trial)) {
        (Some(a), Some(b)) => anneal_core::catalog::packing_distance(&a, &b),
        _ => f64::NAN,
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
    // The road the walk has to travel, measured rather than quoted. Every
    // judgement about whether a hill is wide enough or a rung long enough
    // is a comparison against this number, so it is reported beside the
    // grain it is compared with instead of being carried in from notes.
    let marks_gap = packing_gap(&ico_slice, &marks_slice);
    println!(
        "{{\"kind\":\"leave_probe_floors\",\"ico\":{ico_energy:.6},\"marks\":{marks_energy:.6},\"link\":{PACKING_LINK},\"marks_gap\":{marks_gap:.4}}}"
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
    // `scale` as the third argument runs only the chain-scaling block.
    let scale_only = std::env::args().nth(3).is_some_and(|mode| mode == "scale");
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

    if only_ridge {
        // Directed packing-map walk: 96 covering points on the packing
        // sphere miss Marks. This arm aims at Marks' mu, then at the
        // fivefold residual that opened LJ38, then at a covering ridge.
        let mut toward = Tally {
            best: ico_energy,
            ..Tally::default()
        };
        let mut fivefold = Tally {
            best: ico_energy,
            ..Tally::default()
        };
        let mut cover = Tally {
            best: ico_energy,
            ..Tally::default()
        };
        if let Some(direction) = known_basin::packing_direction_between(ico.view(), marks.view()) {
            println!("{{\"kind\":\"marks_dir\",\"dim\":{}}}", direction.len());
            for rung in 0..known_basin::LEAVE_RUNGS {
                let barrier = known_basin::rung_barrier(depth, rung);
                let start = known_basin::leave_packing_rung_to_dir(
                    ico.view(),
                    &direction,
                    barrier,
                    None,
                    None,
                    |v: ArrayView1<f64>| Some(potential.value_and_gradient(v).0),
                );
                let rise = potential.value_and_gradient(start.view()).0 - ico_energy;
                let trial = quench(&potential, start.view(), steps);
                classify("toward_rung", rung, &mut toward, &trial, Some(rung));
                println!(
                    "{{\"kind\":\"toward_rung\",\"rung\":{rung},\"barrier\":{barrier:.3},\"rise\":{rise:.3}}}"
                );
            }
        } else {
            println!("{{\"kind\":\"toward_rung\",\"failed\":\"no_direction\"}}");
        }
        let walked = known_basin::leave_packing_toward(
            ico.view(),
            marks.view(),
            &references,
            None,
            None,
            depth,
            steps,
            |trial, n| {
                let relaxed = quench(&potential, trial, n);
                (potential.value_and_gradient(relaxed.view()).0, relaxed)
            },
        );
        match walked {
            Some((_, trial, rung)) => classify("toward", 0, &mut toward, &trial, Some(rung)),
            None => println!(
                "{{\"kind\":\"leave\",\"generator\":\"toward\",\"index\":0,\"rung\":null,\"refused\":true}}"
            ),
        }
        for (index, rmsd) in [0.12_f64, 0.35, 0.50].into_iter().enumerate() {
            let start = anneal_core::soap::step_away_fivefold_measured(ico.view(), rmsd);
            let trial = quench(&potential, start.view(), steps);
            classify("fivefold", index, &mut fivefold, &trial, None);
        }
        for index in 0..leaves {
            let walked = known_basin::leave_packing_ridge(
                ico.view(),
                index,
                &references,
                None,
                None,
                depth,
                steps,
                |trial, n| {
                    let relaxed = quench(&potential, trial, n);
                    (potential.value_and_gradient(relaxed.view()).0, relaxed)
                },
            );
            match walked {
                Some((_, trial, rung)) => {
                    classify("ridge", index, &mut cover, &trial, Some(rung));
                }
                None => println!(
                    "{{\"kind\":\"leave\",\"generator\":\"ridge\",\"index\":{index},\"rung\":null,\"refused\":true}}"
                ),
            }
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
        report("toward", &toward, known_basin::LEAVE_RUNGS + 1);
        report("fivefold", &fivefold, 3);
        report("ridge", &cover, leaves);
        return;
    }

    for index in 0..leaves {
        if scale_only {
            break;
        }
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
    // Accumulated packing-map walk: each step is a barrier-sized
    // pullback from the *current* point, then a quench. Independent
    // rungs from the well all fall back into ico; the Hessian min mode
    // does the same.
    let mut ridge = Tally {
        best: ico_energy,
        ..Tally::default()
    };
    for index in 0..leaves {
        if scale_only {
            break;
        }
        known_basin::arm_leave(ico.view(), known_basin::LEAVE_RUNG_RMSD, &references);
        let walked = known_basin::leave_packing_ridge(
            ico.view(),
            index,
            &references,
            None,
            None,
            depth,
            steps,
            |trial, n| {
                let relaxed = quench(&potential, trial, n);
                (potential.value_and_gradient(relaxed.view()).0, relaxed)
            },
        );
        known_basin::disarm();
        match walked {
            Some((_, trial, rung)) => {
                classify("ridge", index, &mut ridge, &trial, Some(rung));
            }
            None => println!(
                "{{\"kind\":\"leave\",\"generator\":\"ridge\",\"index\":{index},\"rung\":null,\"refused\":true}}"
            ),
        }
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    report("ridge", &ridge, leaves);
    if only_ridge {
        return;
    }

    // Rung by rung, with the cost the step actually paid and where the
    // quench below it landed. A refusal can mean the step never left or the
    // accept refused a genuine packing, and only the energies tell them
    // apart.
    for index in 0..leaves {
        if scale_only {
            break;
        }
        for rung in 0..known_basin::LEAVE_RUNGS {
            let barrier = known_basin::rung_barrier(depth, rung);
            let start = known_basin::leave_packing_rung_to(
                ico.view(),
                index,
                barrier,
                &references,
                None,
                None,
                |v: ArrayView1<f64>| Some(potential.value_and_gradient(v).0),
            );
            let rise = potential.value_and_gradient(start.view()).0 - ico_energy;
            let moved = start
                .iter()
                .zip(ico.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>();
            let rmsd = (moved / (ico.len() / 3) as f64).sqrt();
            let far = start
                .iter()
                .zip(ico.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let trial = quench(&potential, start.view(), steps);
            let energy = potential.value_and_gradient(trial.view()).0;
            let left = trial
                .as_slice()
                .is_some_and(|slice| leaves_packing(&ico_slice, slice, &references));
            let community = trial
                .as_slice()
                .map_or(2, |slice| community_of(&ico_slice, &marks_slice, slice));
            println!(
                "{{\"kind\":\"rung\",\"index\":{index},\"rung\":{rung},\"barrier\":{barrier:.3},\"rmsd\":{rmsd:.4},\"max_atom\":{far:.4},\"rise\":{rise:.3},\"quench\":{energy:.6},\"left\":{left},\"community\":{community}}}"
            );
        }
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }

    // Does repulsion from more chains help the quench leave?
    //
    // The ensemble claim is that chains interact at the minimisation level:
    // each chain's quench is pushed away from the wells the others occupy,
    // so more chains means a larger repulsion set and a better chance of
    // leaving the packing they share. That is testable here without an
    // ensemble: seed the cloud with K distinct quenched icosahedral wells
    // and walk the same ladder against it.
    for k in [1usize, 4, 12, 24, 48] {
        // The book, not just the cloud. A quench that lands on a well
        // already held is the entropy measurement rather than a duplicate,
        // so the repeats are counted instead of thrown away: the log of
        // the count is the configurational entropy of that packing, and
        // the free-energy deposit needs it.
        let mut book: Vec<anneal_core::catalog::PackingReference> =
            vec![anneal_core::catalog::PackingReference {
                coordinates: ico_slice.clone(),
                visits: 1,
                deposit: 0.0,
            }];
        let mut seeded = 0usize;
        let mut step = 0usize;
        // The seeding budget is what the top of this curve measures, and
        // 400 steps of a sigma that cycles through fifteen values found
        // fifteen distinct wells and then repeated itself: cloud sizes 24
        // and 48 both came back as 15 wells and 375 arrivals, the same
        // experiment run twice. The budget is raised and the cycle
        // lengthened to a coprime stride so the two points differ.
        while seeded + 1 < k && step < 6000 {
            step += 1;
            let sigma = 0.08 + 0.02 * ((step % 37) as f64);
            let mut start = ico.clone();
            for (index, value) in start.iter_mut().enumerate() {
                let mix = (index * 31 + step * 17 + (step / 37) * 11) % 7;
                *value += sigma * (mix as f64 - 3.0) / 3.0;
            }
            let relaxed = quench(&potential, start.view(), steps);
            let energy = potential.value_and_gradient(relaxed.view()).0;
            if !energy.is_finite() || energy > ico_energy + 8.0 {
                continue;
            }
            let Some(slice) = relaxed.as_slice() else {
                continue;
            };
            if let Some(held) = book.iter_mut().find(|held| {
                anneal_core::catalog::packing_distance(&held.coordinates, slice) <= 1e-9
            }) {
                held.visits = held.visits.saturating_add(1);
                continue;
            }
            book.push(anneal_core::catalog::PackingReference {
                coordinates: slice.to_vec(),
                visits: 1,
                deposit: 0.0,
            });
            seeded += 1;
        }
        let cloud: Vec<Vec<f64>> = book
            .iter()
            .map(|held| held.coordinates.clone())
            .collect();
        // Total arrivals and the Shannon entropy of the packing histogram,
        // which is what an ensemble buys that a single chain does not.
        let arrivals: u32 = book.iter().map(|held| held.visits).sum();
        let shannon = if arrivals > 0 {
            -book
                .iter()
                .map(|held| {
                    let p = f64::from(held.visits) / f64::from(arrivals);
                    if p > 0.0 { p * p.ln() } else { 0.0 }
                })
                .sum::<f64>()
        } else {
            0.0
        };
        let mut left = 0usize;
        let mut best = ico_energy;
        // The hill the cloud deposits, so the run reads against the
        // barrier it has to clear rather than against whether it escaped.
        // K wells at amplitude A give at most K A; the ico-Marks saddles
        // are 8.69 and 7.48 eps above the icosahedral shelf.
        let mut lift = 0.0_f64;
        let mut sigma = 0.0_f64;
        // How far the walk got, and how much of it the raw polish gave
        // back. A deposit tall enough to clear the barrier and a walk
        // that still lands on the same minimum are two different
        // failures, and the escape count alone cannot tell them apart:
        // either the transformed surface never moved the quench, or it
        // moved it and the minimum below the ridge was the one it
        // started from.
        let mut start_r = 0.0_f64;
        let mut walk_r = 0.0_f64;
        let mut land_r = 0.0_f64;
        let mut land_e = f64::INFINITY;
        // The same walk stopped at the grain rather than run to the
        // minimum of E+V, which is the comparison that says whether the
        // projector is the barrier or the over-travel is.
        let mut stop_r = 0.0_f64;
        let mut stop_land_r = 0.0_f64;
        let mut stop_e = f64::INFINITY;
        let trials = leaves.min(16);
        for index in 0..trials {
            // T = 0.8 is the run temperature in the resolved config, and
            // the tempering scale is the same, so the converged pile is
            // half the free energy.
            known_basin::arm_leave_free(
                ico.view(),
                known_basin::LEAVE_RUNG_RMSD,
                &book,
                0.8,
                0.8,
            );
            let walked = known_basin::leave_packing_ladder(
                ico.view(),
                index,
                &cloud,
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
            if let Some((amplitude, width)) = known_basin::lift() {
                lift = lift.max(amplitude);
                sigma = width;
            }
            known_basin::disarm();
            // One walk instrumented outside the ladder's accept. The
            // ladder answers None when no rung leaves, which is the case
            // under study, so anything measured inside its Some arm is
            // silent exactly when it is needed.
            //
            // Rung 3 is the first that clears the icosahedral-to-Marks
            // saddles: the rungs aim at a quarter of the depth per atom
            // doubling each time, so on LJ75 they are 1.32, 2.64, 5.28,
            // 10.57, and the measured saddles are 8.69 and 7.48. Rung 2
            // stops short of both, which would measure a walk that was
            // never given enough to cross with.
            known_basin::arm_leave_free(
                ico.view(),
                known_basin::LEAVE_RUNG_RMSD,
                &book,
                0.8,
                0.8,
            );
            let start = known_basin::leave_packing_rung_to(
                ico.view(),
                index,
                known_basin::rung_barrier(depth, 3),
                &cloud,
                None,
                None,
                |v: ArrayView1<f64>| Some(potential.value_and_gradient(v).0),
            );
            if let Some(slice) = start.as_slice() {
                start_r = start_r.max(packing_gap(&ico_slice, slice));
            }
            let rode = quench(&potential, start.view(), steps);
            if let Some(slice) = rode.as_slice() {
                walk_r = walk_r.max(packing_gap(&ico_slice, slice));
            }
            // The same walk, stopped as soon as it clears the grain.
            //
            // Run to convergence the transformed quench minimises E+V,
            // which under a hill this tall is a distorted structure far
            // from any minimum of E: measured, it reaches 1.12 in DECAF
            // distance, past the 0.69 that separates the two packings,
            // and the raw quench below it returns to the icosahedral
            // floor exactly. Distance in the map is not a crossing of the
            // dividing surface. Stopping just past the grain is the
            // nearest thing to stopping on the ridge that costs no extra
            // gradient.
            let chunk = (steps / 12).max(4);
            let mut ridge = start.clone();
            let mut ridge_r = 0.0_f64;
            for _ in 0..12 {
                ridge = quench(&potential, ridge.view(), chunk);
                let Some(slice) = ridge.as_slice() else { break };
                ridge_r = packing_gap(&ico_slice, slice);
                if ridge_r > 1.2 * PACKING_LINK {
                    break;
                }
            }
            stop_r = stop_r.max(ridge_r);
            known_basin::disarm();
            let held = quench(&potential, ridge.view(), steps);
            if let Some(slice) = held.as_slice() {
                stop_land_r = stop_land_r.max(packing_gap(&ico_slice, slice));
            }
            let held_e = potential.value_and_gradient(held.view()).0;
            if held_e.is_finite() && held_e < stop_e {
                stop_e = held_e;
            }
            let fell = quench(&potential, rode.view(), steps);
            if let Some(slice) = fell.as_slice() {
                land_r = land_r.max(packing_gap(&ico_slice, slice));
            }
            let fell_e = potential.value_and_gradient(fell.view()).0;
            if fell_e.is_finite() && fell_e < land_e {
                land_e = fell_e;
            }
            if let Some((energy, trial, _)) = walked {
                let polished = quench(&potential, trial.view(), steps);
                let energy = potential.value_and_gradient(polished.view()).0.min(energy);
                if polished
                    .as_slice()
                    .is_some_and(|t| leaves_packing(&ico_slice, t, &[]))
                {
                    left += 1;
                }
                if energy < best {
                    best = energy;
                }
            }
        }
        println!(
            "{{\"kind\":\"chain_scaling\",\"chains\":{k},\"cloud\":{},\"trials\":{trials},\"left\":{left},\"best\":{best:.6},\"lift\":{lift:.4},\"sigma_phi\":{sigma:.4},\"arrivals\":{arrivals},\"shannon\":{shannon:.4},\"start_r\":{start_r:.4},\"walk_r\":{walk_r:.4},\"land_r\":{land_r:.4},\"land_e\":{land_e:.6},\"stop_r\":{stop_r:.4},\"stop_land_r\":{stop_land_r:.4},\"stop_e\":{stop_e:.6},\"grain\":0.35,\"marks_r\":0.69,\"barrier\":8.69}}",
            cloud.len()
        );
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }

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
        if scale_only {
            break;
        }
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
