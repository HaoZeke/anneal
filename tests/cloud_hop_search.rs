//! Real `optimize` / `run_with_gradient` path for the observed-cloud SOAP hop.
//!
//! Cluster, multi-species molecule, and a slab-shaped frozen frame. The
//! in-crate Lennard-Jones potential stands in for a QC engine.

use anneal_core::methods::cluster_hopping::{
    ClusterMove, Config, Ledger, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn water_dimer() -> (Array1<f64>, Vec<u32>, Vec<Vec<usize>>) {
    let x = Array1::from_vec(vec![
        0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0, 3.10, 0.15, 0.08, 3.98, 0.40, -0.05,
        2.82, 1.05, 0.18,
    ]);
    let z = vec![8, 1, 1, 8, 1, 1];
    let groups = vec![vec![0, 1, 2], vec![3, 4, 5]];
    (x, z, groups)
}

fn charged_relax<'a>(
    pot: &'a PairPotential,
    opt: &'a mut WarmLbfgs,
    frozen: Option<&'a [bool]>,
) -> impl FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>) + 'a {
    move |led, x, iters| {
        opt.forget();
        let pin = x.to_owned();
        let (f, mut xr, _) = opt.minimize_watched(
            x,
            iters,
            |v| {
                if !led.charge() {
                    return None;
                }
                let (e, mut g) = pot.value_and_gradient(v);
                if let Some(mask) = frozen {
                    for (i, &is_frozen) in mask.iter().enumerate() {
                        if is_frozen {
                            for k in 0..3 {
                                g[3 * i + k] = 0.0;
                            }
                        }
                    }
                }
                Some((e, g))
            },
            |_, _| true,
        );
        if let Some(mask) = frozen {
            for (i, &is_frozen) in mask.iter().enumerate() {
                if is_frozen {
                    for k in 0..3 {
                        xr[3 * i + k] = pin[3 * i + k];
                    }
                }
            }
        }
        (f, xr)
    }
}

#[test]
fn hop_shape_from_shipped_constructors() {
    let rec = Config::recommended(13);
    let soap = rec
        .move_library
        .kernels(&rec)
        .into_iter()
        .find(|k| matches!(k, ClusterMove::Soap { .. }));
    match soap {
        Some(ClusterMove::Soap {
            class,
            species,
            mobile,
            ..
        }) => {
            assert!(!class, "recommended SOAP shipped the 555->421 oracle");
            assert!(species.is_none());
            assert!(mobile.is_none());
            println!("rec class={class} species={species:?} mobile={mobile:?}");
        }
        _ => panic!("recommended LeanBurst has no SOAP arm"),
    }

    let mol = Config::recommended_molecular(vec![8, 1, 1], vec![vec![0, 1, 2]], 1.0);
    assert!(!mol.packing_cna_applies());
    assert!(!mol.soap_class_residual);
    let soap = mol
        .move_library
        .kernels(&mol)
        .into_iter()
        .find(|k| matches!(k, ClusterMove::Soap { .. }));
    match soap {
        Some(ClusterMove::Soap {
            class,
            species,
            ..
        }) => {
            assert!(!class);
            assert_eq!(species.as_deref(), Some(&[8, 1, 1][..]));
            println!("mol class={class} species={species:?}");
        }
        _ => panic!("recommended_molecular has no SOAP arm"),
    }

    let mut slab = Config::recommended_molecular(
        vec![29, 29, 1, 1],
        vec![vec![2, 3]],
        1.0,
    );
    slab.active_region = Some((vec![2, 3], 0));
    slab.n_points = 4;
    let soap = slab
        .move_library
        .kernels(&slab)
        .into_iter()
        .find(|k| matches!(k, ClusterMove::Soap { .. }));
    match soap {
        Some(ClusterMove::Soap {
            class,
            species,
            mobile,
            ..
        }) => {
            assert!(!class);
            assert_eq!(species.as_deref(), Some(&[29, 29, 1, 1][..]));
            assert_eq!(mobile.as_deref(), Some(&[2, 3][..]));
            println!("slab class={class} species={species:?} mobile={mobile:?}");
        }
        _ => panic!("slab-shaped library has no SOAP arm"),
    }
}

#[test]
fn recommended_molecular_search_hops_on_water_dimer() {
    let (start, species, groups) = water_dimer();
    let n = species.len();
    let cfg = Config::recommended_molecular(species, groups, 1.0);
    assert!(!cfg.packing_cna_applies());
    assert!(
        cfg.move_library
            .kernels(&cfg)
            .iter()
            .any(|k| matches!(k, ClusterMove::Soap { class: false, .. }))
    );
    let pot = PairPotential::lennard_jones(n);
    let mut ledger = Ledger::new(2_500);
    let mut opt = WarmLbfgs::default();
    let mut relax = charged_relax(&pot, &mut opt, None);
    let mut rng = StdRng::seed_from_u64(3);
    let out = run_with_gradient(&cfg, start.view(), &mut ledger, &mut relax, None, &mut rng);
    assert!(out.best.is_finite(), "best {}", out.best);
    assert!(out.hops > 0, "no hops on recommended_molecular");
    let soap_draws = out
        .arms
        .iter()
        .find(|(name, _, _, _)| name == "soap")
        .map(|(_, d, _, _)| *d)
        .unwrap_or(0);
    println!(
        "mol-rec best={:.6} hops={} charged={} soap_draws={} arms={:?}",
        out.best, out.hops, out.charged, soap_draws, out.arms
    );
}

#[test]
fn recommended_molecular_slab_keeps_frozen_coords() {
    let (start, species, groups) = water_dimer();
    let n = species.len();
    let mut cfg = Config::recommended_molecular(species, groups, 1.0);
    cfg.active_region = Some((vec![0, 1, 2], 0));
    let frozen: Vec<bool> = (0..n).map(|i| i >= 3).collect();
    cfg.frozen = Some(frozen.clone());
    let pot = PairPotential::lennard_jones(n);
    let mut ledger = Ledger::new(2_500);
    let mut opt = WarmLbfgs::default();
    let mut relax = charged_relax(&pot, &mut opt, Some(frozen.as_slice()));
    let mut rng = StdRng::seed_from_u64(5);
    let out = run_with_gradient(&cfg, start.view(), &mut ledger, &mut relax, None, &mut rng);
    assert!(out.best.is_finite(), "best {}", out.best);
    assert!(out.hops > 0, "no hops on slab-shaped recommended_molecular");
    let end = out
        .final_state
        .as_ref()
        .or(out.best_state.as_ref())
        .expect("search returned no state");
    for i in 3..6 {
        for k in 0..3 {
            assert_eq!(
                end[3 * i + k],
                start[3 * i + k],
                "frozen atom {i} axis {k} moved"
            );
        }
    }
    let soap_draws = out
        .arms
        .iter()
        .find(|(name, _, _, _)| name == "soap")
        .map(|(_, d, _, _)| *d)
        .unwrap_or(0);
    println!(
        "slab-rec best={:.6} hops={} charged={} soap_draws={} frozen_ok=1 arms={:?}",
        out.best, out.hops, out.charged, soap_draws, out.arms
    );
}
