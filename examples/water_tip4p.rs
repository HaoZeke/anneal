//! Rigid TIP4P water clusters against the Wales--Hodges putative minima.
//!
//! Usage: `water_tip4p <n> <budget> <seeds> [seed0]`
//!
//! `PRESET` names the chain: `plain` ([`Config::for_tip4p`], the default),
//! `recommended` ([`Config::recommended_tip4p`]) or `communicating`
//! ([`Config::communicating_tip4p`]). `OPTS` is a comma-separated list of
//! mechanisms: `sbreport` turns on the superbasin report and `sbasin` the
//! superbasin escape with it. `goat` runs GOAT's ladder of four rungs that
//! never exchange states, at GOAT's kcal/mol steps converted to kJ/mol.
//! `selft` lets each chain regulate its own temperature, and `boost`
//! accepts a chain's next new minimum after a stall. An unknown name stops
//! the run, so a label cannot stand for a mechanism the binary does not
//! bind.
//!
//! `HISTORY_REPLICAS` runs that many thread replicas per seed, splitting the
//! budget, over the channels `HISTORY` and the other ensemble variables of
//! `examples/common/ensemble_report.rs` name.
//!
//! A non-numeric fourth token is an options label and is ignored, so
//! `ttt.sh n budget seeds tip4p` can drive this binary the same way it
//! drives `lj_cluster_search`.

#[path = "common/ensemble_report.rs"]
mod ensemble_report;

use anneal_core::bias::BasinBias;
use anneal_core::descriptor_space::{
    DescriptorGeometry, DescriptorOutputSpec, DescriptorProviderContract, DescriptorProviderError,
    DescriptorProviderInput, DescriptorSpace, InvariantDescriptorProvider,
};
use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, ClusterFingerprint, Config, Ledger,
    run_with_bias_at_checkpoints, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::pes_exploration::{ExactStructureWitness, StructureContext, StructureView};
use anneal_core::potentials::{Tip4pCluster, random_tip4p_cluster, wales_hodges_minimum};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::io::{self, Write};

/// Wales and Hodges tabulate their minima to 0.01 kJ/mol, so a best within
/// that of the table is the tabulated minimum.
const HIT_TOLERANCE: f64 = 0.01;

/// Checks the option names and binds the ones this binary knows.
fn apply_options(cfg: &mut Config, opts: &[&str]) {
    for option in opts {
        assert!(
            matches!(*option, "sbreport" | "sbasin" | "goat" | "selft" | "boost"),
            "unknown search mechanism: {option}"
        );
    }
    cfg.superbasin_escape = cfg.superbasin_escape || opts.contains(&"sbasin");
    cfg.superbasin_report =
        cfg.superbasin_report || cfg.superbasin_escape || opts.contains(&"sbreport");
    cfg.self_regulating_temperature = cfg.self_regulating_temperature || opts.contains(&"selft");
    cfg.boost_on_stall = cfg.boost_on_stall || opts.contains(&"boost");
    if opts.contains(&"goat") {
        *cfg = cfg.clone().with_goat_ladder();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(6);
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let seeds: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut seed0: u64 = std::env::var("SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if let Some(fourth) = args.get(4)
        && let Ok(parsed) = fourth.parse::<u64>()
    {
        seed0 = parsed;
    }

    let preset = std::env::var("PRESET").unwrap_or_else(|_| "plain".into());
    let mut cfg = match preset.as_str() {
        "plain" => Config::for_tip4p(n),
        "recommended" => Config::recommended_tip4p(n),
        "communicating" => Config::communicating_tip4p(n),
        other => panic!("PRESET={other:?}; expected plain, recommended or communicating"),
    };
    let opts_text = std::env::var("OPTS").unwrap_or_default();
    let opts: Vec<&str> = opts_text
        .split(',')
        .map(str::trim)
        .filter(|option| !option.is_empty())
        .collect();
    apply_options(&mut cfg, &opts);

    let pot = Tip4pCluster::new(n);
    let reference = wales_hodges_minimum(n);
    println!(
        "TIP4P (H2O){n}, budget {budget}, seeds {seed0}..{}",
        seed0 + seeds
    );
    if let Some(r) = reference {
        println!("  Wales--Hodges target {r:.2} kJ/mol");
    }
    println!("  preset {preset}");
    if preset == "communicating" {
        println!("  communicating configuration");
    }
    if !opts.is_empty() {
        println!("  mechanisms: {}", opts.join(", "));
    }
    if opts.contains(&"goat") {
        println!(
            "  goat ladder: {} rungs from T={:.4} to T={:.4} kJ/mol, self-regulating",
            cfg.replicas,
            cfg.temperature,
            cfg.temperature * cfg.ladder_top
        );
    }
    if cfg.superbasin_report {
        println!(
            "  superbasin report{}",
            if cfg.superbasin_escape {
                format!(", escape offered every {} hops", cfg.superbasin_period)
            } else {
                String::new()
            }
        );
    }
    if let Some(replicas) = anneal_core::env::parsed::<usize>("HISTORY_REPLICAS") {
        run_history_ensembles(&cfg, n, budget, seed0, seeds, reference, replicas, &opts);
        return;
    }

    let mut solved = 0usize;
    let mut deepest = f64::INFINITY;
    let mut total_hops = 0usize;
    let mut total_charged = 0usize;
    for seed in seed0..(seed0 + seeds) {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(3));
        let start = random_tip4p_cluster(n, &mut rng);
        let mut ledger = Ledger::new(budget);
        let mut opt = WarmLbfgs::default();
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
            opt.forget();
            let (f, mut xr, _) = opt.minimize(x, iters, |v| {
                if !led.charge() {
                    return None;
                }
                Some(pot.value_and_gradient(v))
            });
            if let Some(slice) = xr.as_slice_mut() {
                pot.fold_rotations(slice);
            }
            (f, xr)
        };
        let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
            if !led.charge() {
                return None;
            }
            Some(pot.value_and_gradient(x).1)
        };
        // A ladder gives every rung a bias of its own, so only a single
        // chain can take the caller's.
        let mut out = if cfg.replicas > 1 {
            run_with_gradient(
                &cfg,
                start.view(),
                &mut ledger,
                &mut relax,
                Some(&mut grad),
                &mut rng,
            )
        } else {
            let mut bias = BasinBias::new(
                ClusterFingerprint::of_config(&cfg, &start),
                cfg.merge_radius,
                cfg.bias_height,
                cfg.bias_gamma,
            );
            let mut checkpoint = |_: ChainCheckpoint<'_>| CheckpointAction::Continue;
            run_with_bias_at_checkpoints(
                &cfg,
                start.view(),
                &mut ledger,
                &mut relax,
                Some(&mut grad),
                &mut bias,
                &mut rng,
                1_000,
                &mut checkpoint,
            )
        }
        .expect("finite start");
        if let Some(state) = out.best_state.as_ref() {
            let e = pot.value_and_gradient(state.view()).0;
            out.best = e;
        }
        let hit = reference
            .map(|r| out.best < r + HIT_TOLERANCE)
            .unwrap_or(false);
        if hit {
            solved += 1;
        }
        let first_hit = reference.and_then(|r| {
            out.improvements
                .iter()
                .find(|(_, _, _, e)| *e < r + HIT_TOLERANCE)
                .map(|(_, spent, _, _)| *spent)
        });
        deepest = deepest.min(out.best);
        total_hops += out.hops;
        total_charged += ledger.spent();
        println!(
            "  seed {seed}: best {:.6} kJ/mol  wales-hodges {}  first_hit {}  hops {}  charged {}{}",
            out.best,
            if hit { "yes" } else { "no" },
            first_hit
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into()),
            out.hops,
            ledger.spent(),
            if hit { "  SOLVED" } else { "" }
        );
        if let Some(sb) = &out.superbasin {
            println!("  seed {seed} {}", ensemble_report::superbasin_line(sb));
        }
        if let Some(goat) = ensemble_report::goat_line(&cfg, &out) {
            println!("  seed {seed} {goat}");
        }
        println!(
            "{}/1 solved, deepest {:.6}   mean hops {:.0}, force per hop {:.0}",
            usize::from(hit),
            out.best,
            out.hops as f64,
            ledger.spent() as f64 / out.hops.max(1) as f64
        );
        let _ = io::stdout().flush();
    }
    println!(
        "{solved}/{seeds} solved, deepest {deepest:.6}   \
         mean hops {:.0}, force per hop {:.0}",
        total_hops as f64 / seeds.max(1) as f64,
        total_charged as f64 / total_hops.max(1) as f64
    );
    if let Some(r) = reference {
        println!("gap to reference {:+.6}", deepest - r);
    }
}

/// Lab-frame O, H, H sites of every molecule, nine numbers per water.
fn oxygen_hydrogen_sites(pot: &Tip4pCluster, x: ArrayView1<f64>) -> Array1<f64> {
    let mut sites = Vec::with_capacity(9 * pot.n_molecules);
    for i in 0..pot.n_molecules {
        for site in &pot.molecule_sites(x, i)[..3] {
            sites.extend_from_slice(site);
        }
    }
    Array1::from(sites)
}

/// Sorted O-O distances of a rigid state: invariant under a rigid motion
/// and a relabelling of the waters.
///
/// The history only orders its exact comparisons by this descriptor, so it
/// sets how soon a match is found and not whether it is.
#[derive(Debug)]
struct OxygenDistances {
    pot: Tip4pCluster,
    contract: DescriptorProviderContract,
}

impl OxygenDistances {
    fn space(n: usize, cfg: &Config) -> DescriptorSpace {
        use sha2::{Digest, Sha256};
        let name = "tip4p-rigid-oxygen-distances";
        let digest: [u8; 32] = Sha256::digest(format!("{name} v1 n{n}").as_bytes()).into();
        let contract = DescriptorProviderContract::new(
            name,
            1,
            digest,
            DescriptorOutputSpec::new("oxygen-distances", n * (n - 1) / 2)
                .expect("two or more waters have an O-O distance"),
            None,
            // Every O-O distance of a state the moves generate.
            2.0 * cfg.container,
            "angstrom-distance-v1",
        )
        .expect("the O-O distance contract is valid");
        DescriptorSpace::from_provider(
            DescriptorGeometry::finite(cfg.length_scale).expect("the TIP4P length is positive"),
            Self {
                pot: Tip4pCluster::new(n),
                contract,
            },
        )
        .expect("the O-O distance descriptor is valid")
    }
}

impl InvariantDescriptorProvider for OxygenDistances {
    fn contract(&self) -> &DescriptorProviderContract {
        &self.contract
    }

    fn describe_system(
        &self,
        input: DescriptorProviderInput<'_>,
    ) -> Result<Vec<f64>, DescriptorProviderError> {
        let x = input.coordinates();
        if x.len() != self.pot.dim_coords() {
            return Err(DescriptorProviderError::new(format!(
                "a rigid state of {} waters has {} coordinates, not {}",
                self.pot.n_molecules,
                self.pot.dim_coords(),
                x.len()
            )));
        }
        let oxygens: Vec<[f64; 3]> = (0..self.pot.n_molecules)
            .map(|i| self.pot.molecule_sites(x.view(), i)[0])
            .collect();
        let mut distances = Vec::with_capacity(oxygens.len() * (oxygens.len() - 1) / 2);
        for (i, a) in oxygens.iter().enumerate() {
            for b in &oxygens[i + 1..] {
                distances.push(
                    (0..3)
                        .map(|k| (a[k] - b[k]) * (a[k] - b[k]))
                        .sum::<f64>()
                        .sqrt(),
                );
            }
        }
        distances.sort_by(f64::total_cmp);
        Ok(distances)
    }
}

/// Identity of two rigid states, judged on their O and H sites.
///
/// A rigid motion of the whole cluster moves the chart's rotation vectors
/// by composition, not by the same rotation, so the chart itself cannot be
/// matched; the lab-frame sites can.
struct RigidSiteWitness<W> {
    pot: Tip4pCluster,
    sites: W,
    context: StructureContext,
}

impl<W: ExactStructureWitness> ExactStructureWitness for RigidSiteWitness<W> {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        let (a, b) = (
            oxygen_hydrogen_sites(&self.pot, left),
            oxygen_hydrogen_sites(&self.pot, right),
        );
        self.sites.equivalent_structures(
            StructureView {
                coordinates: a.view(),
                context: &self.context,
            },
            StructureView {
                coordinates: b.view(),
                context: &self.context,
            },
        )
    }
}

/// The site witness for `n` waters and the name the header gives it.
fn site_witness(
    n: usize,
) -> (
    RigidSiteWitness<impl ExactStructureWitness + Send>,
    &'static str,
) {
    let context = StructureContext::new(
        Some([8, 1, 1].repeat(n)),
        None,
        Some(format!("tip4p-sites-n{n}")),
    );
    // The radius and search breadth of the molecular water ensembles, in
    // angstrom like the TIP4P sites.
    let radius = 1e-4;
    #[cfg(feature = "ira")]
    let (sites, name) = (
        anneal_core::shape::IraStructureWitness {
            kmax_factor: 1.8,
            radius,
        },
        "ira-sites",
    );
    #[cfg(not(feature = "ira"))]
    let (sites, name) = (
        ensemble_report::sorted_pairs_witness(3 * n, radius),
        "sorted-pairs-sites-fallback",
    );
    (
        RigidSiteWitness {
            pot: Tip4pCluster::new(n),
            sites,
            context,
        },
        name,
    )
}

/// Thread replicas of the rigid hop loop over the channels the environment
/// names, under the comparison contract of `lj_cluster_search`: identical
/// replica seeds and starts in every arm and one aggregate budget per seed.
#[allow(clippy::too_many_arguments)]
fn run_history_ensembles(
    cfg: &Config,
    n: usize,
    budget: usize,
    seed0: u64,
    seeds: u64,
    reference: Option<f64>,
    replicas: usize,
    opts: &[&str],
) {
    use anneal_core::methods::ensemble::{EnsembleProblem, ObjectiveFactory, StartFactory};
    use anneal_core::methods::minima_hopping::SerializedWitness;
    use ensemble_report::{config_from_env, print_header, run_seeds};
    use std::sync::Mutex;

    let ens = config_from_env(replicas, budget, reference.map(|r| r + HIT_TOLERANCE));
    ens.validate(cfg).unwrap_or_else(|error| panic!("{error}"));
    let descriptor = OxygenDistances::space(n, cfg);
    let context = StructureContext::new(None, None, Some(format!("tip4p-rigid-n{n}")));
    let (witness, witness_name) = site_witness(n);
    let witness = SerializedWitness(Mutex::new(witness));
    let objective: ObjectiveFactory<'_> = &|_| {
        let pot = Tip4pCluster::new(n);
        Box::new(move |x: ArrayView1<f64>| pot.value_and_gradient(x))
    };
    let start: StartFactory<'_> = &|_, rng| random_tip4p_cluster(n, rng);
    // Packing families are a cluster notion; no two-choice crowding here.
    let same_family = |_: &[f64], _: &[f64]| false;
    let problem = EnsembleProblem {
        objective,
        start,
        descriptor: &descriptor,
        context: &context,
        witness: &witness,
        same_family: &same_family,
        // The chain's own record criterion is the certificate, with the
        // polish window one decade above it as in the molecular ensembles.
        certificate: cfg.record_gradient,
        polish_below: 10.0 * cfg.record_gradient,
        callbacks_per_objective: 1,
    };
    print_header(
        &ens,
        witness_name,
        &opts.join(","),
        &format!(
            "record gradient {:.2e}, shared deposits {},",
            cfg.record_gradient, cfg.shared_deposits
        ),
    );
    let audit = Tip4pCluster::new(n);
    let verify = |replica: usize, x: &Array1<f64>, reported: f64| {
        assert_eq!(
            x.len(),
            6 * n,
            "replica {replica} returned {} coordinates",
            x.len()
        );
        let (e, g) = audit.value_and_gradient(x.view());
        let gmax = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(
            e.is_finite() && (e - reported).abs() < 1e-6,
            "replica {replica} reports {reported} but its coordinates have energy {e}"
        );
        (e, gmax)
    };
    run_seeds(
        cfg, &ens, seed0, seeds, &problem, " kJ/mol", &verify, reference,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_mechanism_is_rejected_before_search() {
        let result = std::panic::catch_unwind(|| {
            let mut cfg = Config::communicating_tip4p(4);
            apply_options(&mut cfg, &["sbasin", "unknown_mechanism"]);
        });
        assert!(result.is_err(), "an unknown mechanism must not label a run");
    }

    #[test]
    fn sbasin_turns_on_the_superbasin_report_and_escape() {
        let mut cfg = Config::communicating_tip4p(4);
        apply_options(&mut cfg, &["sbasin"]);
        assert!(cfg.superbasin_report && cfg.superbasin_escape);
        let mut cfg = Config::communicating_tip4p(4);
        apply_options(&mut cfg, &["sbreport"]);
        assert!(cfg.superbasin_report && !cfg.superbasin_escape);
    }

    #[test]
    fn goat_and_selft_bind_their_fields() {
        let mut cfg = Config::communicating_tip4p(4);
        apply_options(&mut cfg, &["sbasin", "selft"]);
        assert!(cfg.self_regulating_temperature && cfg.replicas == 1);
        let mut cfg = Config::communicating_tip4p(4);
        apply_options(&mut cfg, &["sbasin", "goat"]);
        assert_eq!(cfg.replicas, 4);
        assert!(cfg.self_regulating_temperature && !cfg.minima_hopping);
        let mut cfg = Config::communicating_tip4p(4);
        apply_options(&mut cfg, &["sbasin", "boost"]);
        assert!(cfg.boost_on_stall && cfg.minima_hopping);
    }

    #[test]
    fn relabelled_and_translated_waters_are_one_minimum() {
        let n = 4;
        let (witness, _) = site_witness(n);
        let mut rng = StdRng::seed_from_u64(7);
        let x = random_tip4p_cluster(n, &mut rng);
        let mut y = x.clone();
        for block in [0, 3 * n] {
            for k in 0..3 {
                y.swap(block + k, block + 3 + k);
            }
        }
        for i in 0..n {
            for (k, shift) in [1.0, -2.0, 0.5].into_iter().enumerate() {
                y[3 * i + k] += shift;
            }
        }
        assert!(witness.equivalent(x.view(), y.view()));
        let other = random_tip4p_cluster(n, &mut rng);
        assert!(!witness.equivalent(x.view(), other.view()));
    }
}
