//! Matched-call LJ global-minimum benchmark for joint-optimum exploration.
//!
//! The adaptive arm allocates between finite invariant basin proposals and
//! projected rgsaddle ridge/IRC actions by information about the identity and
//! energy of the lowest reachable minimum per expected PES call. Fixed-family
//! ablations, plain Wales--Doye basin hopping, rgsaddle NVE minima hopping, and
//! a history-conditioned displacement-feedback ablation use the same initial
//! coordinates, seeds, target, and charged-call ceiling.
//!
//! Usage:
//! `lj_joint_optimum <N> <budget> <seeds> [all|adaptive|ridge|basin|bh|mh|mh-soft|mh-bounded|mh-bounded-soft|feedback] [gs2|morokuma|both] [seed0]`

use std::error::Error;
use std::path::{Path, PathBuf};

use anneal_core::atomistic_hybrid::{
    AtomisticHybridConfig, AtomisticHybridPolicy, AtomisticSystem, explore_atomistic_with_policy,
};
use anneal_core::catalog::lj;
use anneal_core::methods::cluster_hopping::{
    Config as HoppingConfig, Ledger, MoveLibrary, Outcome, random_cluster, run, run_with_gradient,
};
use anneal_core::methods::cluster_search::{Encounter, first_encounter, median_encounter};
use anneal_core::methods::minima_hopping::{
    EscapeFeedback, MdEscapeConfig, MdEscapeGeometry, Visit, nve_escape,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::pes_exploration::{
    ExactStructureWitness, IrcKind, PesExplorationConfig, RideMethod,
};
use anneal_core::potentials::{PairKind, PairPotential};
use anneal_core::shape::IraStructureWitness;
use ndarray::ArrayView1;
use rand::{SeedableRng, rngs::StdRng};
use rgsaddle::VelocitySofteningConfig;
use serde_json::json;

const TARGET_TOLERANCE: f64 = 1e-3;
const MH_SOFTENING_STEPS: usize = 20;
const MH_SOFTENING_DISPLACEMENT: f64 = 0.1;
const MH_SOFTENING_MIXING: f64 = 0.15;

#[derive(Clone, Copy, Debug)]
enum Arm {
    Adaptive(IrcKind),
    Ridge(IrcKind),
    Basin,
    BasinHopping,
    MinimaHopping,
    MinimaHoppingSoftened,
    MinimaHoppingBounded,
    MinimaHoppingBoundedSoftened,
    MinimaFeedback,
}

impl Arm {
    fn label(self) -> String {
        match self {
            Self::Adaptive(kind) => format!("adaptive-{}", irc_name(kind)),
            Self::Ridge(kind) => format!("ridge-{}", irc_name(kind)),
            Self::Basin => "basin-ablation".into(),
            Self::BasinHopping => "basin-hopping".into(),
            Self::MinimaHopping => "minima-hopping".into(),
            Self::MinimaHoppingSoftened => "minima-hopping-softened".into(),
            Self::MinimaHoppingBounded => "minima-hopping-bounded".into(),
            Self::MinimaHoppingBoundedSoftened => "minima-hopping-bounded-softened".into(),
            Self::MinimaFeedback => "minima-feedback".into(),
        }
    }
}

#[derive(Default)]
struct Summary {
    encounters: Vec<Encounter>,
    deepest: f64,
    charged: u128,
    minima: u128,
    saddles: u128,
    failures: u128,
}

impl Summary {
    fn observe(
        &mut self,
        encounter: Encounter,
        best: f64,
        charged: u64,
        minima: usize,
        saddles: usize,
        failures: usize,
    ) {
        if self.encounters.is_empty() {
            self.deepest = best;
        } else {
            self.deepest = self.deepest.min(best);
        }
        self.encounters.push(encounter);
        self.charged += u128::from(charged);
        self.minima += minima as u128;
        self.saddles += saddles as u128;
        self.failures += failures as u128;
    }
}

fn reference(n: usize) -> Option<f64> {
    Some(match n {
        13 => -44.326_801,
        38 => -173.928_427,
        55 => -279.248_470,
        75 => -397.492_331,
        98 => -543.665_361,
        102 => -569.363_652,
        104 => -582.086_642,
        _ => return None,
    })
}

fn irc_name(kind: IrcKind) -> &'static str {
    match kind {
        IrcKind::Gs2 => "gs2",
        IrcKind::Morokuma => "morokuma",
    }
}

fn irc_kinds(selector: &str) -> Result<Vec<IrcKind>, String> {
    match selector {
        "gs2" => Ok(vec![IrcKind::Gs2]),
        "morokuma" => Ok(vec![IrcKind::Morokuma]),
        "both" => Ok(vec![IrcKind::Gs2, IrcKind::Morokuma]),
        _ => Err("IRC selector must be gs2, morokuma, or both".into()),
    }
}

fn selected_arms(selector: &str, irc: &[IrcKind]) -> Result<Vec<Arm>, String> {
    let mut arms = Vec::new();
    match selector {
        "all" => {
            arms.extend(irc.iter().copied().map(Arm::Adaptive));
            arms.extend(irc.iter().copied().map(Arm::Ridge));
            arms.extend([
                Arm::Basin,
                Arm::BasinHopping,
                Arm::MinimaHopping,
                Arm::MinimaHoppingSoftened,
                Arm::MinimaHoppingBounded,
                Arm::MinimaHoppingBoundedSoftened,
                Arm::MinimaFeedback,
            ]);
        }
        "adaptive" => arms.extend(irc.iter().copied().map(Arm::Adaptive)),
        "ridge" => arms.extend(irc.iter().copied().map(Arm::Ridge)),
        "basin" => arms.push(Arm::Basin),
        "bh" => arms.push(Arm::BasinHopping),
        "mh" => arms.push(Arm::MinimaHopping),
        "mh-soft" => arms.push(Arm::MinimaHoppingSoftened),
        "mh-bounded" => arms.push(Arm::MinimaHoppingBounded),
        "mh-bounded-soft" => arms.push(Arm::MinimaHoppingBoundedSoftened),
        "feedback" => arms.push(Arm::MinimaFeedback),
        _ => {
            return Err(
                "arm must be all, adaptive, ridge, basin, bh, mh, mh-soft, mh-bounded, mh-bounded-soft, or feedback"
                    .into(),
            );
        }
    }
    Ok(arms)
}

fn hybrid_config(n: usize, budget: u64, irc_kind: IrcKind) -> AtomisticHybridConfig {
    let dimension = u64::try_from(3 * n).unwrap_or(u64::MAX);
    let escape_cap = budget.min((20 * dimension).max(800));
    let ride_cap = budget.min((100 * dimension).max(4_000));
    let length = PairKind::LennardJones.r_min();
    let component_tolerance = 1e-5 / (dimension as f64).sqrt();
    AtomisticHybridConfig {
        evaluation_budget: budget,
        ride_evaluation_cap: ride_cap,
        escape_evaluation_cap: escape_cap,
        ride_modes_per_atom: 1,
        localization_radius: 1.5 * length,
        escape_scales: vec![0.08 * length, 0.16 * length, 0.32 * length, 0.64 * length],
        minimum_information_samples: 256,
        information_length_scale: 1.0,
        information_amplitude: 1.0,
        information_noise: 1e-6,
        expected_ride_cost: ride_cap as f64,
        expected_escape_cost: escape_cap as f64,
        cost_prior_strength: 1.0,
        exploration: PesExplorationConfig {
            ride_method: RideMethod::Lanczos,
            quench_steps: 1_000,
            saddle_steps: 1_000,
            minimum_mode_force_tolerance: 1e-2,
            irc_steps: 200,
            prfo_steps: 300,
            activation_attempts: 6,
            activation_growth: 1.6,
            activation_relaxation_steps: 3,
            quench_gradient_tolerance: component_tolerance,
            quench_gradient_norm_tolerance: Some(1e-5),
            saddle_force_tolerance: 1e-3,
            saddle_displacement: 0.1 * length,
            negative_curvature_tolerance: 1e-6,
            hessian_step: 1e-4 * length,
            maximum_move: 0.2 * length,
            irc_step: 0.1 * length,
            irc_kind,
            branch_attempts: 4,
            branch_growth: 2.0,
            irc_force_tolerance: 0.05,
            refine_with_prfo: true,
        },
    }
}

fn run_hopping(
    potential: &PairPotential,
    initial: ArrayView1<'_, f64>,
    n: usize,
    budget: usize,
    seed: u64,
    minima_hopping: bool,
) -> Outcome {
    let mut config = HoppingConfig::for_cluster(n);
    config.bias_height = 0.0;
    config.move_library = MoveLibrary::WalesDoye;
    config.minima_hopping = minima_hopping;
    let mut ledger = Ledger::new(budget);
    let mut optimizer = WarmLbfgs::default();
    let mut relax = |ledger: &mut Ledger, start: ArrayView1<'_, f64>, steps: usize| {
        let before = ledger.spent();
        optimizer.forget();
        let (energy, coordinates, _) = optimizer.minimize(start, steps, |point| {
            ledger.charge().then(|| potential.value_and_gradient(point))
        });
        ledger.record_quench_boundary(before, energy, coordinates.clone(), None);
        (energy, coordinates)
    };
    let mut gradient = |ledger: &mut Ledger, point: ArrayView1<'_, f64>| {
        ledger
            .charge()
            .then(|| potential.value_and_gradient(point).1)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    if minima_hopping {
        run_with_gradient(
            &config,
            initial,
            &mut ledger,
            &mut relax,
            Some(&mut gradient),
            &mut rng,
        )
    } else {
        run(&config, initial, &mut ledger, &mut relax, &mut rng)
    }
}

fn quench_minimum(
    potential: &PairPotential,
    optimizer: &mut WarmLbfgs,
    ledger: &mut Ledger,
    start: ArrayView1<'_, f64>,
    steps: usize,
    gradient_tolerance: f64,
) -> Option<(f64, ndarray::Array1<f64>, bool)> {
    if ledger.remaining() == 0 {
        return None;
    }
    let before = ledger.spent();
    optimizer.forget();
    let (energy, state, _) = optimizer.minimize(start, steps, |point| {
        ledger.charge().then(|| potential.value_and_gradient(point))
    });
    let gradient = ledger.charge().then(|| {
        let (_, gradient) = potential.value_and_gradient(state.view());
        gradient
    });
    let validated = energy.is_finite()
        && gradient.as_ref().is_some_and(|values| {
            values
                .iter()
                .all(|value| value.is_finite() && value.abs() < gradient_tolerance)
        });
    ledger.record_quench_boundary(
        before,
        energy,
        state.clone(),
        validated.then_some(gradient).flatten(),
    );
    Some((energy, state, validated))
}

fn exact_basin(
    witness: &impl ExactStructureWitness,
    minima: &[ndarray::Array1<f64>],
    candidate: ArrayView1<'_, f64>,
) -> Option<usize> {
    minima
        .iter()
        .position(|minimum| witness.equivalent(minimum.view(), candidate))
}

fn run_minima_hopping(
    potential: &PairPotential,
    initial: ArrayView1<'_, f64>,
    n: usize,
    budget: usize,
    seed: u64,
    witness: &impl ExactStructureWitness,
    soften: bool,
    bound_escape: bool,
) -> Outcome {
    let hopping = HoppingConfig::for_cluster(n);
    let escape_config = MdEscapeConfig {
        dt: 0.005,
        potential_minima: 2,
        maximum_steps: 2_000,
        geometry: MdEscapeGeometry::RigidQuotient,
        softening: soften.then_some(VelocitySofteningConfig {
            steps: MH_SOFTENING_STEPS,
            displacement: MH_SOFTENING_DISPLACEMENT,
            mixing: MH_SOFTENING_MIXING,
        }),
    };
    let mut ledger = Ledger::new(budget);
    let mut optimizer = WarmLbfgs::default();
    let Some((mut energy, mut state, initial_valid)) = quench_minimum(
        potential,
        &mut optimizer,
        &mut ledger,
        initial,
        hopping.relax_steps,
        hopping.record_gradient,
    ) else {
        return Outcome {
            best: ledger.best,
            best_state: ledger.best_state.clone(),
            final_energy: f64::INFINITY,
            charged: ledger.spent(),
            ..Outcome::default()
        };
    };
    if !initial_valid {
        return Outcome {
            best: ledger.best,
            best_state: ledger.best_state.clone(),
            final_state: Some(state),
            final_energy: energy,
            charged: ledger.spent(),
            unconverged_records: 1,
            ..Outcome::default()
        };
    }

    ledger.record(energy, state.view());
    let mut minima = vec![state.clone()];
    let mut current_basin = 0usize;
    let mut feedback = EscapeFeedback::new(hopping.energy_scale, 0.5 * hopping.energy_scale);
    if !bound_escape {
        feedback.escape_floor = f64::MIN_POSITIVE;
        feedback.escape_ceiling = f64::MAX;
    }
    feedback.register_initial(current_basin);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut hops = 0usize;
    let mut accepted = 0usize;
    let mut unconverged = 0usize;
    let mut improvements = vec![(0, ledger.spent(), minima.len(), energy)];

    while ledger.remaining() > 0 {
        let mut evaluate =
            |point: ArrayView1<f64>| ledger.charge().then(|| potential.value_and_gradient(point));
        let escape = nve_escape(
            state.view(),
            feedback.escape(),
            &escape_config,
            &mut evaluate,
            &mut rng,
        );
        drop(evaluate);
        hops += 1;
        let Ok(escape) = escape else {
            unconverged += 1;
            if ledger.remaining() == 0 {
                break;
            }
            feedback.observe(Some(current_basin), current_basin);
            continue;
        };
        if escape.potential_minima < escape_config.potential_minima {
            feedback.observe(Some(current_basin), current_basin);
            continue;
        }
        let Some((candidate_energy, candidate, validated)) = quench_minimum(
            potential,
            &mut optimizer,
            &mut ledger,
            escape.position.view(),
            hopping.relax_steps,
            hopping.record_gradient,
        ) else {
            break;
        };
        if !validated {
            unconverged += 1;
            feedback.observe(Some(current_basin), current_basin);
            continue;
        }

        if let Some(reached) = exact_basin(witness, &minima, candidate.view()) {
            feedback.observe(Some(current_basin), reached);
            continue;
        }

        let reached = minima.len();
        minima.push(candidate.clone());
        let visit = feedback.observe(Some(current_basin), reached);
        debug_assert_eq!(visit, Visit::New);
        let improved = candidate_energy < ledger.best;
        ledger.record(candidate_energy, candidate.view());
        if improved {
            improvements.push((hops, ledger.spent(), minima.len(), candidate_energy));
        }
        if feedback.accept(candidate_energy - energy) {
            current_basin = reached;
            energy = candidate_energy;
            state = candidate;
            accepted += 1;
        }
    }

    Outcome {
        best: ledger.best,
        best_state: ledger.best_state.clone(),
        final_state: Some(state),
        final_energy: energy,
        hops,
        basins: minima.len(),
        charged: ledger.spent(),
        escape_scale: feedback.escape(),
        escape_threshold: feedback.threshold(),
        visit_counts: (feedback.n_same, feedback.n_known, feedback.n_new),
        improvements,
        accepted,
        unconverged_records: unconverged,
        ..Outcome::default()
    }
}

fn wilson_interval(hits: usize, total: usize) -> (f64, f64) {
    if total == 0 {
        return (0.0, 0.0);
    }
    let n = total as f64;
    let probability = hits as f64 / n;
    let z = 1.959_963_984_540_054_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let centre = (probability + z2 / (2.0 * n)) / denominator;
    let half =
        z * ((probability * (1.0 - probability) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

fn encounter_fields(encounter: Encounter) -> (bool, usize) {
    (encounter.found(), encounter.charged())
}

fn optbench_start_path(root: &Path, n: usize, index: usize) -> Result<PathBuf, String> {
    let name = match n {
        38 if index < 100 => format!("{index}.con"),
        75 if index < 200 => format!("coords.{}", index + 1),
        98 if index <= 100 => {
            if index == 0 {
                "coords".into()
            } else {
                format!("coords.{index}")
            }
        }
        38 => return Err("OptBench LJ38 contains 100 starts indexed 0 through 99".into()),
        75 => return Err("OptBench LJ75 contains 200 starts indexed 0 through 199".into()),
        98 => return Err("OptBench LJ98 contains 101 starts indexed 0 through 100".into()),
        _ => return Err("OptBench starts are registered for LJ38, LJ75, and LJ98".into()),
    };
    Ok(root.join(name))
}

fn parse_plain_coordinates(text: &str, n: usize) -> Result<ndarray::Array1<f64>, String> {
    let coordinates = text
        .split_whitespace()
        .map(|token| {
            token
                .parse::<f64>()
                .map_err(|error| format!("invalid coordinate {token:?}: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if coordinates.len() != 3 * n || coordinates.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "OptBench start requires {} finite coordinates, found {}",
            3 * n,
            coordinates.len()
        ));
    }
    Ok(ndarray::Array1::from(coordinates))
}

fn read_optbench_start(
    root: &Path,
    n: usize,
    index: usize,
) -> Result<ndarray::Array1<f64>, String> {
    let path = optbench_start_path(root, n, index)?;
    if n == 38 {
        let frame = readcon_core::iterators::read_first_frame(&path)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if frame.atom_data.len() != n {
            return Err(format!(
                "{} contains {} atoms, expected {n}",
                path.display(),
                frame.atom_data.len()
            ));
        }
        let coordinates = frame
            .atom_data
            .iter()
            .flat_map(|atom| [atom.x, atom.y, atom.z])
            .collect::<Vec<_>>();
        if coordinates.iter().any(|value| !value.is_finite()) {
            return Err(format!("{} contains nonfinite coordinates", path.display()));
        }
        Ok(ndarray::Array1::from(coordinates))
    } else {
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        parse_plain_coordinates(&text, n)
    }
}

fn optbench_archive_digest(n: usize) -> Option<&'static str> {
    match n {
        38 => Some("9e5bb818adfad67a5afd53d083a6c8afd3addc0fe81a0f57413a73c80eb74e8f"),
        75 => Some("8590a6fddf96a8673d0e4b53aae2385b5d222a9200217d722bb017d23fc5fdb3"),
        98 => Some("e3eef6bb42a8d0b9d3b760862318aabaf932133e1dd3d63b663a556cfbef6149"),
        _ => None,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    let n = arguments
        .get(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(38);
    let budget = arguments
        .get(2)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(100_000);
    let seeds = arguments
        .get(3)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(8);
    let selector = arguments.get(4).map(String::as_str).unwrap_or("all");
    let irc_selector = arguments.get(5).map(String::as_str).unwrap_or("gs2");
    let seed0: u64 = arguments
        .get(6)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let optbench_root = std::env::var_os("ANNEAL_OPTBENCH_STARTS").map(PathBuf::from);
    if n < 2 || budget == 0 || seeds == 0 {
        return Err("N must be at least two and budget/seeds must be positive".into());
    }
    let target = reference(n).ok_or("no published LJ target is registered for this size")?;
    let arms = selected_arms(selector, &irc_kinds(irc_selector)?)?;
    let descriptor_space = lj::descriptor_space();
    let potential = PairPotential::lennard_jones(n);
    let witness = IraStructureWitness {
        kmax_factor: if n >= 55 { 2.5 } else { 1.8 },
        radius: lj::CALIBRATION_IRA_TOLERANCE,
    };
    let system = AtomisticSystem {
        species: vec![18; n],
        masses: vec![1.0; n],
        frozen_atoms: vec![false; n],
        identity_domain: format!("lj-reduced-n{n}"),
    };
    let hopping = HoppingConfig::for_cluster(n);
    let mut summaries = (0..arms.len())
        .map(|_| Summary::default())
        .collect::<Vec<_>>();

    println!(
        "{}",
        json!({
            "kind": "lj_joint_optimum_config",
            "n": n,
            "budget": budget,
            "seeds": seeds,
            "seed0": seed0,
            "target": target,
            "target_tolerance": TARGET_TOLERANCE,
            "start_protocol": if optbench_root.is_some() { "optbench-fixed" } else { "random-cluster" },
            "start_archive_sha256": optbench_root.as_ref().and_then(|_| optbench_archive_digest(n)),
            "minima_hopping": {
                "integrator": "rgsaddle-samd-nve",
                "dt": 0.005,
                "path_minima": 2,
                "initial_kinetic": hopping.energy_scale,
                "initial_ediff": 0.5 * hopping.energy_scale,
                "beta": 1.05,
                "enhanced_visit_coefficient": 0.1,
                "softening": {
                    "arm": "minima-hopping-softened",
                    "steps": MH_SOFTENING_STEPS,
                    "displacement": MH_SOFTENING_DISPLACEMENT,
                    "mixing": MH_SOFTENING_MIXING,
                },
            },
            "arms": arms.iter().copied().map(Arm::label).collect::<Vec<_>>(),
        })
    );

    for seed in seed0..seed0.saturating_add(seeds) {
        let initial = if let Some(root) = optbench_root.as_deref() {
            read_optbench_start(
                root,
                n,
                usize::try_from(seed).map_err(|_| "seed index overflow")?,
            )?
        } else {
            let mut initial_rng =
                StdRng::seed_from_u64(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(7));
            random_cluster(n, 0.7, hopping.min_separation, &mut initial_rng)
        };
        for (arm_index, arm) in arms.iter().copied().enumerate() {
            let label = arm.label();
            match arm {
                Arm::Adaptive(irc_kind) | Arm::Ridge(irc_kind) => {
                    let policy = if matches!(arm, Arm::Adaptive(_)) {
                        AtomisticHybridPolicy::Adaptive
                    } else {
                        AtomisticHybridPolicy::RidgeOnly
                    };
                    let report = explore_atomistic_with_policy(
                        &potential,
                        &descriptor_space,
                        initial.view(),
                        &system,
                        &hybrid_config(n, budget, irc_kind),
                        &witness,
                        seed,
                        policy,
                    )?;
                    let best = report.best_energy().unwrap_or(f64::INFINITY);
                    let encounter = report.first_encounter(target, TARGET_TOLERANCE);
                    let failures = report
                        .events
                        .iter()
                        .filter(|event| !event.converged)
                        .count();
                    let (found, first_charged) = encounter_fields(encounter);
                    println!(
                        "{}",
                        json!({
                            "kind": "lj_joint_optimum_run",
                            "arm": label,
                            "seed": seed,
                            "best_energy": best,
                            "gap": best - target,
                            "target_found": found,
                            "first_charged": first_charged,
                            "charged": report.charged_evaluations,
                            "minima": report.network.minimum_count(),
                            "saddles": report.network.saddle_count(),
                            "unresolved_saddles": report.network.unresolved_saddles().len(),
                            "events": report.events.len(),
                            "ridge_pulls": report.mechanism_pulls[0],
                            "basin_pulls": report.mechanism_pulls[1],
                            "failed_actions": failures,
                            "termination": format!("{:?}", report.termination),
                        })
                    );
                    summaries[arm_index].observe(
                        encounter,
                        best,
                        report.charged_evaluations,
                        report.network.minimum_count(),
                        report.network.saddle_count(),
                        failures,
                    );
                }
                Arm::Basin => {
                    let report = explore_atomistic_with_policy(
                        &potential,
                        &descriptor_space,
                        initial.view(),
                        &system,
                        &hybrid_config(n, budget, IrcKind::Gs2),
                        &witness,
                        seed,
                        AtomisticHybridPolicy::BasinEscapeOnly,
                    )?;
                    let best = report.best_energy().unwrap_or(f64::INFINITY);
                    let encounter = report.first_encounter(target, TARGET_TOLERANCE);
                    let failures = report
                        .events
                        .iter()
                        .filter(|event| !event.converged)
                        .count();
                    let (found, first_charged) = encounter_fields(encounter);
                    println!(
                        "{}",
                        json!({
                            "kind": "lj_joint_optimum_run",
                            "arm": label,
                            "seed": seed,
                            "best_energy": best,
                            "gap": best - target,
                            "target_found": found,
                            "first_charged": first_charged,
                            "charged": report.charged_evaluations,
                            "minima": report.network.minimum_count(),
                            "saddles": 0,
                            "events": report.events.len(),
                            "ridge_pulls": 0,
                            "basin_pulls": report.mechanism_pulls[1],
                            "failed_actions": failures,
                            "termination": format!("{:?}", report.termination),
                        })
                    );
                    summaries[arm_index].observe(
                        encounter,
                        best,
                        report.charged_evaluations,
                        report.network.minimum_count(),
                        0,
                        failures,
                    );
                }
                Arm::BasinHopping
                | Arm::MinimaHopping
                | Arm::MinimaHoppingSoftened
                | Arm::MinimaHoppingBounded
                | Arm::MinimaHoppingBoundedSoftened
                | Arm::MinimaFeedback => {
                    let budget = usize::try_from(budget).unwrap_or(usize::MAX);
                    let minima_hopping = matches!(
                        arm,
                        Arm::MinimaHopping
                            | Arm::MinimaHoppingSoftened
                            | Arm::MinimaHoppingBounded
                            | Arm::MinimaHoppingBoundedSoftened
                    );
                    let softened = matches!(
                        arm,
                        Arm::MinimaHoppingSoftened | Arm::MinimaHoppingBoundedSoftened
                    );
                    let bound_escape = matches!(
                        arm,
                        Arm::MinimaHoppingBounded | Arm::MinimaHoppingBoundedSoftened
                    );
                    let outcome = if minima_hopping {
                        run_minima_hopping(
                            &potential,
                            initial.view(),
                            n,
                            budget,
                            seed,
                            &witness,
                            softened,
                            bound_escape,
                        )
                    } else {
                        run_hopping(
                            &potential,
                            initial.view(),
                            n,
                            budget,
                            seed,
                            matches!(arm, Arm::MinimaFeedback),
                        )
                    };
                    let encounter =
                        first_encounter(&outcome, target, TARGET_TOLERANCE, outcome.charged);
                    let (found, first_charged) = encounter_fields(encounter);
                    println!(
                        "{}",
                        json!({
                            "kind": "lj_joint_optimum_run",
                            "arm": label,
                            "seed": seed,
                            "best_energy": outcome.best,
                            "gap": outcome.best - target,
                            "target_found": found,
                            "first_charged": first_charged,
                            "charged": outcome.charged,
                            "minima": outcome.basins,
                            "saddles": 0,
                            "hops": outcome.hops,
                            "accepted": outcome.accepted,
                            "escape_kinetic_or_scale": outcome.escape_scale,
                            "escape_bounded": bound_escape,
                            "velocity_softened": softened,
                            "acceptance_threshold": outcome.escape_threshold,
                            "visit_counts": outcome.visit_counts,
                            "failed_actions": outcome.unconverged_records,
                        })
                    );
                    summaries[arm_index].observe(
                        encounter,
                        outcome.best,
                        outcome.charged as u64,
                        outcome.basins,
                        0,
                        outcome.unconverged_records,
                    );
                }
            }
        }
    }

    for (arm, summary) in arms.iter().copied().zip(&summaries) {
        let total = summary.encounters.len();
        let hits = summary
            .encounters
            .iter()
            .filter(|encounter| encounter.found())
            .count();
        let (hit_low, hit_high) = wilson_interval(hits, total);
        println!(
            "{}",
            json!({
                "kind": "lj_joint_optimum_summary",
                "arm": arm.label(),
                "runs": total,
                "hits": hits,
                "hit_probability": hits as f64 / total as f64,
                "hit_probability_wilson95": [hit_low, hit_high],
                "km_median_first_charged": median_encounter(&summary.encounters),
                "deepest_energy": summary.deepest,
                "deepest_gap": summary.deepest - target,
                "mean_charged": summary.charged as f64 / total as f64,
                "mean_minima": summary.minima as f64 / total as f64,
                "mean_saddles": summary.saddles as f64 / total as f64,
                "mean_failed_actions": summary.failures as f64 / total as f64,
            })
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Arm, ExactStructureWitness, PairPotential, optbench_start_path, parse_plain_coordinates,
        run_minima_hopping, selected_arms,
    };
    use ndarray::{Array1, ArrayView1};
    use std::path::Path;

    struct DistinctWitness;

    impl ExactStructureWitness for DistinctWitness {
        fn equivalent(&self, _left: ArrayView1<f64>, _right: ArrayView1<f64>) -> bool {
            false
        }
    }

    #[test]
    fn optbench_paths_follow_each_published_archive_layout() {
        let root = Path::new("/corpus");
        assert_eq!(
            optbench_start_path(root, 38, 7).unwrap(),
            root.join("7.con")
        );
        assert_eq!(
            optbench_start_path(root, 75, 0).unwrap(),
            root.join("coords.1")
        );
        assert_eq!(
            optbench_start_path(root, 98, 0).unwrap(),
            root.join("coords")
        );
        assert_eq!(
            optbench_start_path(root, 98, 100).unwrap(),
            root.join("coords.100")
        );
    }

    #[test]
    fn plain_optbench_coordinates_require_one_finite_triplet_per_atom() {
        let parsed = parse_plain_coordinates("0 1 2\n3 4 5\n", 2).unwrap();
        assert_eq!(parsed.as_slice().unwrap(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(parse_plain_coordinates("0 1 2", 2).is_err());
        assert!(parse_plain_coordinates("0 1 NaN\n3 4 5", 2).is_err());
    }

    #[test]
    fn minima_hopping_selector_names_the_md_escape_algorithm() {
        let arms = selected_arms("mh", &[]).unwrap();

        assert!(matches!(arms.as_slice(), [Arm::MinimaHopping]));
        assert_eq!(arms[0].label(), "minima-hopping");
    }

    #[test]
    fn softened_selector_is_a_distinct_minima_hopping_arm() {
        let arms = selected_arms("mh-soft", &[]).unwrap();

        assert!(matches!(arms.as_slice(), [Arm::MinimaHoppingSoftened]));
        assert_eq!(arms[0].label(), "minima-hopping-softened");
    }

    #[test]
    fn bounded_soft_selector_names_the_cost_controlled_nve_arm() {
        let arms = selected_arms("mh-bounded-soft", &[]).unwrap();

        assert!(matches!(
            arms.as_slice(),
            [Arm::MinimaHoppingBoundedSoftened]
        ));
        assert_eq!(arms[0].label(), "minima-hopping-bounded-softened");
    }

    #[test]
    fn exhausted_initial_quench_cannot_report_zero_as_a_minimum() {
        let potential = PairPotential::lennard_jones(2);
        let initial = Array1::from(vec![0.0, 0.0, 0.0, 1.2, 0.0, 0.0]);

        let outcome = run_minima_hopping(
            &potential,
            initial.view(),
            2,
            0,
            7,
            &DistinctWitness,
            false,
            false,
        );

        assert_eq!(outcome.best, f64::INFINITY);
        assert!(outcome.best_state.is_none());
        assert_eq!(outcome.charged, 0);
    }
}
