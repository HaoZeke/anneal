//! PyO3 wrapper around [`crate::methods::cluster_hopping::Config`].

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::hmc::hop::HopConfig;
use crate::hmc::metric::MetricKind;
use crate::methods::cluster_hopping::{
    ArmSelection, Config, ContinuousSymmetry, Keying, LadderMode, MoveLibrary, OrbitCore,
    PairRange, REMOVED_PRIOR_KEYS, SoapProposalMode, covalent_radius,
};
use crate::methods::lattice_search::LatticeSearchConfig;
use crate::methods::minima_hopping::SharedVisitPolicy;
use crate::methods::tric_opt::LocalStep;
use crate::methods::two_phase::{Cutoff, TwoPhase};
use crate::potentials::PairKind;
use crate::rigid_body::RigidBodyGeometry;
use crate::symmetrise::ShellCore;

fn check_n_at_least_two(n: usize) -> PyResult<()> {
    if n < 2 {
        return Err(PyValueError::new_err("n must be at least 2"));
    }
    Ok(())
}

fn check_pos_finite(name: &str, v: f64) -> PyResult<()> {
    if !v.is_finite() || v <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and positive"
        )));
    }
    Ok(())
}

fn check_nonneg_finite(name: &str, v: f64) -> PyResult<()> {
    if !v.is_finite() || v < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and non-negative"
        )));
    }
    Ok(())
}

fn check_unit_closed(name: &str, v: f64) -> PyResult<()> {
    if !v.is_finite() || !(0.0..=1.0).contains(&v) {
        return Err(PyValueError::new_err(format!("{name} must lie in [0, 1]")));
    }
    Ok(())
}

fn check_unit_half_open_pos(name: &str, v: f64) -> PyResult<()> {
    if !v.is_finite() || v <= 0.0 || v > 1.0 {
        return Err(PyValueError::new_err(format!("{name} must lie in (0, 1]")));
    }
    Ok(())
}

fn check_unit_open(name: &str, v: f64) -> PyResult<()> {
    if !v.is_finite() || v <= 0.0 || v >= 1.0 {
        return Err(PyValueError::new_err(format!("{name} must lie in (0, 1)")));
    }
    Ok(())
}

fn check_usize_ge_one(name: &str, v: usize) -> PyResult<()> {
    if v < 1 {
        return Err(PyValueError::new_err(format!("{name} must be at least 1")));
    }
    Ok(())
}

fn parse_molecular_inputs(
    species: &[u32],
    groups: &[Vec<usize>],
    energy_scale: f64,
) -> PyResult<()> {
    if species.is_empty() {
        return Err(PyValueError::new_err("species must not be empty"));
    }
    check_pos_finite("energy_scale", energy_scale)?;
    let max_r = species
        .iter()
        .copied()
        .map(covalent_radius)
        .fold(0.0_f64, f64::max);
    if max_r <= 0.0 {
        return Err(PyValueError::new_err(
            "species must have known covalent radii",
        ));
    }
    let n = species.len();
    if groups.iter().flatten().any(|&atom| atom >= n) {
        return Err(PyValueError::new_err(
            "a group names an atom past len(species)",
        ));
    }
    Ok(())
}

fn keying_to_str(k: Keying) -> &'static str {
    match k {
        Keying::Distances => "distances",
        Keying::Core => "core",
        Keying::Shape => "shape",
        Keying::Sites => "sites",
        Keying::Canonical => "canonical",
        Keying::SoapPacking => "soap_packing",
        Keying::Triplet => "triplet",
        Keying::Q4 => "q4",
        Keying::Q4Q6 => "q4q6",
        Keying::Soap => "soap",
        Keying::Coordination => "coordination",
    }
}

fn parse_keying(s: &str) -> PyResult<Keying> {
    match s {
        "distances" => Ok(Keying::Distances),
        "core" => Ok(Keying::Core),
        "shape" => Ok(Keying::Shape),
        "sites" => Ok(Keying::Sites),
        "canonical" => Ok(Keying::Canonical),
        "soap_packing" => Ok(Keying::SoapPacking),
        "triplet" => Ok(Keying::Triplet),
        "q4" => Ok(Keying::Q4),
        "q4q6" => Ok(Keying::Q4Q6),
        "soap" => Ok(Keying::Soap),
        "coordination" => Ok(Keying::Coordination),
        other => Err(PyValueError::new_err(format!(
            "unknown keying {other:?}; expected distances, core, shape, sites, canonical, \
             soap_packing, triplet, q4, q4q6, soap, or coordination"
        ))),
    }
}

fn ladder_mode_to_str(m: LadderMode) -> &'static str {
    match m {
        LadderMode::Shipped => "shipped",
        LadderMode::Cyclic => "cyclic",
        LadderMode::Independent => "independent",
        LadderMode::Reversible => "reversible",
        LadderMode::NonReversible => "non_reversible",
    }
}

fn parse_ladder_mode(s: &str) -> PyResult<LadderMode> {
    match s {
        "shipped" => Ok(LadderMode::Shipped),
        "cyclic" => Ok(LadderMode::Cyclic),
        "independent" => Ok(LadderMode::Independent),
        "reversible" => Ok(LadderMode::Reversible),
        "non_reversible" => Ok(LadderMode::NonReversible),
        other => Err(PyValueError::new_err(format!(
            "unknown ladder_mode {other:?}; expected shipped, cyclic, independent, \
             reversible, or non_reversible"
        ))),
    }
}

fn soap_mode_to_str(m: SoapProposalMode) -> &'static str {
    match m {
        SoapProposalMode::Flexible => "flexible",
        SoapProposalMode::Rigid => "rigid",
        SoapProposalMode::Off => "off",
    }
}

fn parse_soap_mode(s: &str) -> PyResult<SoapProposalMode> {
    match s {
        "flexible" => Ok(SoapProposalMode::Flexible),
        "rigid" => Ok(SoapProposalMode::Rigid),
        "off" => Ok(SoapProposalMode::Off),
        other => Err(PyValueError::new_err(format!(
            "unknown soap_mode {other:?}; expected flexible, rigid, or off"
        ))),
    }
}

fn arm_selection_to_str(a: ArmSelection) -> &'static str {
    match a {
        ArmSelection::Posterior => "posterior",
        ArmSelection::Uniform => "uniform",
    }
}

fn parse_arm_selection(s: &str) -> PyResult<ArmSelection> {
    match s {
        "posterior" => Ok(ArmSelection::Posterior),
        "uniform" => Ok(ArmSelection::Uniform),
        other => Err(PyValueError::new_err(format!(
            "unknown arm_selection {other:?}; expected posterior or uniform"
        ))),
    }
}

fn local_step_to_str(step: LocalStep) -> &'static str {
    match step {
        LocalStep::Lbfgs => "lbfgs",
        LocalStep::Tric => "tric",
        LocalStep::Auto => "auto",
    }
}

fn parse_local_step(s: &str) -> PyResult<LocalStep> {
    match s {
        "lbfgs" => Ok(LocalStep::Lbfgs),
        "tric" => Ok(LocalStep::Tric),
        "auto" => Ok(LocalStep::Auto),
        other => Err(PyValueError::new_err(format!(
            "local_step must be \"lbfgs\", \"tric\" or \"auto\", got {other:?}"
        ))),
    }
}

fn shared_visit_policy_to_str(p: SharedVisitPolicy) -> &'static str {
    match p {
        SharedVisitPolicy::Tabu => "tabu",
        SharedVisitPolicy::Recognition => "recognition",
    }
}

fn parse_shared_visit_policy(s: &str) -> PyResult<SharedVisitPolicy> {
    match s {
        "tabu" => Ok(SharedVisitPolicy::Tabu),
        "recognition" => Ok(SharedVisitPolicy::Recognition),
        other => Err(PyValueError::new_err(format!(
            "shared_visit_policy must be \"tabu\" or \"recognition\", got {other:?}"
        ))),
    }
}

fn orbit_core_to_str(c: &OrbitCore) -> &'static str {
    match c {
        OrbitCore::Fraction => "fraction",
        OrbitCore::Shells(_) => "shells",
        OrbitCore::Both(_) => "both",
    }
}

fn move_library_to_str(lib: &MoveLibrary) -> &'static str {
    match lib {
        MoveLibrary::WalesDoye => "wales_doye",
        MoveLibrary::Atomic => "atomic",
        MoveLibrary::Lean => "lean",
        MoveLibrary::LeanBurst => "lean_burst",
        MoveLibrary::LeanBurstHollow => "lean_burst_hollow",
        MoveLibrary::LeanBurstFill => "lean_burst_fill",
        MoveLibrary::Visit => "visit",
        MoveLibrary::Twin => "twin",
        MoveLibrary::SelfReseed => "self_reseed",
        MoveLibrary::LearnedReseed => "learned_reseed",
        MoveLibrary::Reseed => "reseed",
        MoveLibrary::GrowthAndTwin => "growth_and_twin",
        MoveLibrary::Molecular { .. } => "molecular",
        MoveLibrary::RigidBody { .. } => "rigid_body",
    }
}

fn parse_atomic_move_library(s: &str) -> PyResult<MoveLibrary> {
    match s {
        "wales_doye" => Ok(MoveLibrary::WalesDoye),
        "atomic" => Ok(MoveLibrary::Atomic),
        "lean" => Ok(MoveLibrary::Lean),
        "lean_burst" => Ok(MoveLibrary::LeanBurst),
        "lean_burst_hollow" => Ok(MoveLibrary::LeanBurstHollow),
        "lean_burst_fill" => Ok(MoveLibrary::LeanBurstFill),
        "visit" => Ok(MoveLibrary::Visit),
        "twin" => Ok(MoveLibrary::Twin),
        "self_reseed" => Ok(MoveLibrary::SelfReseed),
        "learned_reseed" => Ok(MoveLibrary::LearnedReseed),
        "reseed" => Ok(MoveLibrary::Reseed),
        "growth_and_twin" => Ok(MoveLibrary::GrowthAndTwin),
        "molecular" | "rigid_body" => Err(PyValueError::new_err(
            "molecular and rigid_body move libraries are built via Config.for_molecular, \
             Config.recommended_molecular, or Config.for_tip4p",
        )),
        other => Err(PyValueError::new_err(format!(
            "unknown move_library {other:?}"
        ))),
    }
}

fn parse_continuous_symmetry(value: &Bound<'_, PyAny>) -> PyResult<ContinuousSymmetry> {
    if let Ok(s) = value.extract::<&str>() {
        return match s {
            "off" => Ok(ContinuousSymmetry::Off),
            _ => Err(PyValueError::new_err(format!(
                "string continuous_symmetry must be \"off\", got {s:?}; \
                 pass (\"inversion\", interval), (\"measure\", interval), or a dict"
            ))),
        };
    }
    if let Ok(tup) = value.cast::<PyTuple>() {
        if tup.len() == 2 {
            let group: String = tup.get_item(0)?.extract()?;
            let interval: usize = tup.get_item(1)?.extract()?;
            check_usize_ge_one("continuous_symmetry interval", interval)?;
            return match group.as_str() {
                "inversion" => Ok(ContinuousSymmetry::Inversion { interval }),
                "measure" => Ok(ContinuousSymmetry::Measure { interval }),
                other => Err(PyValueError::new_err(format!(
                    "unknown continuous_symmetry group {other:?}; expected off, inversion, or measure"
                ))),
            };
        }
        return Err(PyValueError::new_err(
            "continuous_symmetry tuple must be (group, interval)",
        ));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        for (k, _) in dict.iter() {
            let key_str: String = k.extract()?;
            if REMOVED_PRIOR_KEYS.contains(&key_str.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "config key `{key_str}` was removed (inert since 1c1b1238); \
                     ContinuousSymmetry::Measure takes only `interval`"
                )));
            }
            if key_str != "group" && key_str != "interval" {
                return Err(PyValueError::new_err(format!(
                    "unknown continuous_symmetry key {key_str:?}"
                )));
            }
        }
        let Some(group_any) = dict.get_item("group")? else {
            return Err(PyValueError::new_err(
                "continuous_symmetry dict requires a \"group\" key",
            ));
        };
        let group: String = group_any.extract()?;
        return match group.as_str() {
            "off" => {
                if dict.contains("interval")? {
                    return Err(PyValueError::new_err(
                        "continuous_symmetry \"off\" does not take an interval",
                    ));
                }
                Ok(ContinuousSymmetry::Off)
            }
            "inversion" | "measure" => {
                let Some(iv_any) = dict.get_item("interval")? else {
                    return Err(PyValueError::new_err(format!(
                        "continuous_symmetry {group:?} requires \"interval\""
                    )));
                };
                let interval: usize = iv_any.extract()?;
                check_usize_ge_one("continuous_symmetry interval", interval)?;
                if group == "inversion" {
                    Ok(ContinuousSymmetry::Inversion { interval })
                } else {
                    Ok(ContinuousSymmetry::Measure { interval })
                }
            }
            other => Err(PyValueError::new_err(format!(
                "unknown continuous_symmetry group {other:?}; expected off, inversion, or measure"
            ))),
        };
    }
    Err(PyTypeError::new_err(
        "continuous_symmetry must be \"off\", (group, interval), or {\"group\": ..., \"interval\": ...}",
    ))
}

fn parse_sites_vec(sites_any: &Bound<'_, PyAny>) -> PyResult<Vec<[f64; 3]>> {
    if let Ok(rows) = sites_any.extract::<Vec<[f64; 3]>>() {
        if rows.is_empty() {
            return Err(PyValueError::new_err("sites must not be empty"));
        }
        return Ok(rows);
    }
    if let Ok(flat) = sites_any.extract::<Vec<f64>>() {
        if flat.is_empty() || !flat.len().is_multiple_of(3) {
            return Err(PyValueError::new_err(
                "flat sites must have non-zero length divisible by 3",
            ));
        }
        return Ok(flat.as_chunks::<3>().0.to_vec());
    }
    Err(PyTypeError::new_err(
        "sites must be a sequence of (x, y, z) triples or a flat 3*k float sequence",
    ))
}

fn rigid_body_geometry_to_py(py: Python<'_>, geom: &RigidBodyGeometry) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let sites: Vec<(f64, f64, f64)> = geom.sites().iter().map(|s| (s[0], s[1], s[2])).collect();
    let bonds: Vec<[usize; 2]> = geom.bonds().to_vec();
    dict.set_item("sites", sites)?;
    dict.set_item("species", geom.species().to_vec())?;
    dict.set_item("bonds", bonds)?;
    Ok(dict.into())
}

fn parse_cutoff_kind_val(kind: &str, val: f64) -> PyResult<Cutoff> {
    check_pos_finite("two_phase cutoff", val)?;
    match kind {
        "fixed" | "Fixed" => Ok(Cutoff::Fixed(val)),
        "relative" | "Relative" => Ok(Cutoff::Relative(val)),
        other => Err(PyValueError::new_err(format!(
            "unknown two_phase cutoff kind {other:?}; expected \"fixed\" or \"relative\""
        ))),
    }
}

fn two_phase_to_py(py: Python<'_>, tp: &TwoPhase) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let (kind, cutoff) = match tp.cutoff {
        Cutoff::Fixed(v) => ("fixed", v),
        Cutoff::Relative(v) => ("relative", v),
    };
    dict.set_item("kind", kind)?;
    dict.set_item("cutoff", cutoff)?;
    dict.set_item("beta", tp.beta)?;
    dict.set_item("mu", tp.mu)?;
    dict.set_item("anisotropic", tp.anisotropic)?;
    Ok(dict.into())
}

fn parse_two_phase(value: &Bound<'_, PyAny>) -> PyResult<TwoPhase> {
    if let Ok(tup) = value.cast::<PyTuple>() {
        return match tup.len() {
            3 => {
                let kind: String = tup.get_item(0)?.extract()?;
                let cutoff_val: f64 = tup.get_item(1)?.extract()?;
                let beta: f64 = tup.get_item(2)?.extract()?;
                check_nonneg_finite("two_phase beta", beta)?;
                Ok(TwoPhase {
                    cutoff: parse_cutoff_kind_val(&kind, cutoff_val)?,
                    beta,
                    mu: 0.0,
                    anisotropic: false,
                })
            }
            5 => {
                let kind: String = tup.get_item(0)?.extract()?;
                let cutoff_val: f64 = tup.get_item(1)?.extract()?;
                let beta: f64 = tup.get_item(2)?.extract()?;
                let mu: f64 = tup.get_item(3)?.extract()?;
                let anisotropic: bool = tup.get_item(4)?.extract()?;
                check_nonneg_finite("two_phase beta", beta)?;
                check_nonneg_finite("two_phase mu", mu)?;
                Ok(TwoPhase {
                    cutoff: parse_cutoff_kind_val(&kind, cutoff_val)?,
                    beta,
                    mu,
                    anisotropic,
                })
            }
            _ => Err(PyValueError::new_err(
                "two_phase tuple must be (kind, cutoff, beta) or (kind, cutoff, beta, mu, anisotropic)",
            )),
        };
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        for (k, _) in dict.iter() {
            let key: String = k.extract()?;
            if !matches!(
                key.as_str(),
                "kind" | "cutoff" | "beta" | "mu" | "anisotropic"
            ) {
                return Err(PyValueError::new_err(format!(
                    "unknown two_phase key {key:?}"
                )));
            }
        }
        let Some(cutoff_any) = dict.get_item("cutoff")? else {
            return Err(PyValueError::new_err("two_phase dict requires \"cutoff\""));
        };
        let cutoff = if let Some(kind_any) = dict.get_item("kind")? {
            let kind: String = kind_any.extract()?;
            let val: f64 = cutoff_any.extract()?;
            parse_cutoff_kind_val(&kind, val)?
        } else if let Ok(tup) = cutoff_any.cast::<PyTuple>() {
            if tup.len() != 2 {
                return Err(PyValueError::new_err(
                    "two_phase cutoff tuple must be (kind, value)",
                ));
            }
            let kind: String = tup.get_item(0)?.extract()?;
            let val: f64 = tup.get_item(1)?.extract()?;
            parse_cutoff_kind_val(&kind, val)?
        } else if let Ok(sub) = cutoff_any.cast::<PyDict>() {
            if sub.len() != 1 {
                return Err(PyValueError::new_err(
                    "two_phase cutoff dict must have a single \"fixed\" or \"relative\" entry",
                ));
            }
            let (k, v) = sub.iter().next().expect("checked len == 1");
            let kind: String = k.extract()?;
            let val: f64 = v.extract()?;
            parse_cutoff_kind_val(&kind, val)?
        } else {
            return Err(PyValueError::new_err(
                "two_phase dict without \"kind\" requires cutoff=(kind, value)",
            ));
        };
        let Some(beta_any) = dict.get_item("beta")? else {
            return Err(PyValueError::new_err("two_phase dict requires \"beta\""));
        };
        let beta: f64 = beta_any.extract()?;
        check_nonneg_finite("two_phase beta", beta)?;
        let mu: f64 = match dict.get_item("mu")? {
            Some(v) => v.extract()?,
            None => 0.0,
        };
        check_nonneg_finite("two_phase mu", mu)?;
        let anisotropic: bool = match dict.get_item("anisotropic")? {
            Some(v) => v.extract()?,
            None => false,
        };
        return Ok(TwoPhase {
            cutoff,
            beta,
            mu,
            anisotropic,
        });
    }
    Err(PyTypeError::new_err(
        "two_phase entry must be a dict or (kind, cutoff, beta[, mu, anisotropic]) tuple",
    ))
}

fn parse_pair_kind_spec(value: &Bound<'_, PyAny>) -> PyResult<PairKind> {
    if let Ok(s) = value.extract::<&str>() {
        return match s {
            "lennard_jones" | "lj" => Ok(PairKind::LennardJones),
            other => Err(PyValueError::new_err(format!(
                "unknown reoccupy pair kind {other:?}; expected \"lennard_jones\" or (\"morse\", rho)"
            ))),
        };
    }
    if let Ok(tup) = value.cast::<PyTuple>() {
        if tup.len() == 2 {
            let name: String = tup.get_item(0)?.extract()?;
            let rho: f64 = tup.get_item(1)?.extract()?;
            if name != "morse" {
                return Err(PyValueError::new_err(format!(
                    "reoccupy tuple kind must be (\"morse\", rho), got {name:?}"
                )));
            }
            check_pos_finite("morse rho", rho)?;
            return Ok(PairKind::Morse { rho });
        }
        return Err(PyValueError::new_err(
            "reoccupy kind tuple must be (\"morse\", rho)",
        ));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        if dict.len() == 1
            && let Some(rho_any) = dict.get_item("morse")?
        {
            let rho: f64 = rho_any.extract()?;
            check_pos_finite("morse rho", rho)?;
            return Ok(PairKind::Morse { rho });
        }
        return Err(PyValueError::new_err(
            "reoccupy kind dict must be {\"morse\": rho}",
        ));
    }
    Err(PyTypeError::new_err(
        "reoccupy kind must be \"lennard_jones\" or (\"morse\", rho)",
    ))
}

fn reoccupy_to_py(py: Python<'_>, cfg: &LatticeSearchConfig) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("n_points", cfg.n_points)?;
    match cfg.kind {
        PairKind::LennardJones => dict.set_item("kind", "lennard_jones")?,
        PairKind::Morse { rho } => dict.set_item("kind", ("morse", rho))?,
    }
    dict.set_item("neighbour_cutoff", cfg.neighbour_cutoff)?;
    dict.set_item("relax_steps", cfg.relax_steps)?;
    dict.set_item("patience", cfg.patience)?;
    dict.set_item("max_moves", cfg.max_moves)?;
    dict.set_item("min_separation", cfg.min_separation)?;
    Ok(dict.into())
}

fn parse_reoccupy(default_n: usize, value: &Bound<'_, PyAny>) -> PyResult<LatticeSearchConfig> {
    if value.extract::<&str>().is_ok() || value.cast::<PyTuple>().is_ok() {
        return match parse_pair_kind_spec(value)? {
            PairKind::LennardJones => Ok(LatticeSearchConfig::lennard_jones(default_n)),
            PairKind::Morse { rho } => Ok(LatticeSearchConfig::morse(default_n, rho)),
        };
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        for (k, _) in dict.iter() {
            let key: String = k.extract()?;
            if !matches!(
                key.as_str(),
                "n_points"
                    | "kind"
                    | "neighbour_cutoff"
                    | "relax_steps"
                    | "patience"
                    | "max_moves"
                    | "min_separation"
            ) {
                return Err(PyValueError::new_err(format!(
                    "unknown reoccupy key {key:?}"
                )));
            }
        }
        let n_points: usize = match dict.get_item("n_points")? {
            Some(v) => v.extract()?,
            None => default_n,
        };
        check_n_at_least_two(n_points)?;
        if n_points != default_n {
            return Err(PyValueError::new_err(format!(
                "reoccupy n_points {n_points} must match parent Config n_points {default_n}"
            )));
        }
        let kind = match dict.get_item("kind")? {
            Some(v) => parse_pair_kind_spec(&v)?,
            None => PairKind::LennardJones,
        };
        let mut cfg = match kind {
            PairKind::LennardJones => LatticeSearchConfig::lennard_jones(n_points),
            PairKind::Morse { rho } => LatticeSearchConfig::morse(n_points, rho),
        };
        if let Some(v) = dict.get_item("neighbour_cutoff")? {
            let val: f64 = v.extract()?;
            check_pos_finite("reoccupy neighbour_cutoff", val)?;
            cfg.neighbour_cutoff = val;
        }
        if let Some(v) = dict.get_item("relax_steps")? {
            cfg.relax_steps = v.extract()?;
        }
        if let Some(v) = dict.get_item("patience")? {
            let val: usize = v.extract()?;
            check_usize_ge_one("reoccupy patience", val)?;
            cfg.patience = val;
        }
        if let Some(v) = dict.get_item("max_moves")? {
            cfg.max_moves = v.extract()?;
        }
        if let Some(v) = dict.get_item("min_separation")? {
            let val: f64 = v.extract()?;
            check_nonneg_finite("reoccupy min_separation", val)?;
            cfg.min_separation = val;
        }
        return Ok(cfg);
    }
    Err(PyTypeError::new_err(
        "reoccupy must be None, \"lennard_jones\", (\"morse\", rho), or a config dict",
    ))
}

fn metric_kind_to_str(m: MetricKind) -> &'static str {
    match m {
        MetricKind::Identity => "identity",
        MetricKind::Diagonal => "diagonal",
        MetricKind::ModelHessian => "model_hessian",
    }
}

fn parse_metric_kind(s: &str) -> PyResult<MetricKind> {
    match s {
        "identity" => Ok(MetricKind::Identity),
        "diagonal" => Ok(MetricKind::Diagonal),
        "model_hessian" | "modelhess" => Ok(MetricKind::ModelHessian),
        other => Err(PyValueError::new_err(format!(
            "unknown hmc metric {other:?}; expected \"identity\", \"diagonal\", or \"model_hessian\""
        ))),
    }
}

fn hmc_to_py(py: Python<'_>, cfg: &HopConfig) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("n_points", cfg.n_points)?;
    dict.set_item("metric", metric_kind_to_str(cfg.metric))?;
    dict.set_item("warmup_hops", cfg.warmup_hops)?;
    dict.set_item("max_depth", cfg.max_depth)?;
    dict.set_item("target_accept", cfg.target_accept)?;
    dict.set_item("control_step", cfg.control_step)?;
    dict.set_item("target_reach", cfg.target_reach)?;
    dict.set_item("drift_period", cfg.drift_period)?;
    dict.set_item("trajectory_temperature", cfg.trajectory_temperature)?;
    Ok(dict.into())
}

fn parse_hmc(default_n: usize, value: &Bound<'_, PyAny>) -> PyResult<HopConfig> {
    if let Ok(s) = value.extract::<&str>() {
        return Ok(HopConfig::new(default_n, parse_metric_kind(s)?));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        for (k, _) in dict.iter() {
            let key: String = k.extract()?;
            if !matches!(
                key.as_str(),
                "n_points"
                    | "metric"
                    | "warmup_hops"
                    | "max_depth"
                    | "target_accept"
                    | "control_step"
                    | "target_reach"
                    | "drift_period"
                    | "trajectory_temperature"
            ) {
                return Err(PyValueError::new_err(format!("unknown hmc key {key:?}")));
            }
        }
        let n_points: usize = match dict.get_item("n_points")? {
            Some(v) => v.extract()?,
            None => default_n,
        };
        check_n_at_least_two(n_points)?;
        if n_points != default_n {
            return Err(PyValueError::new_err(format!(
                "hmc n_points {n_points} must match parent Config n_points {default_n}"
            )));
        }
        let metric = match dict.get_item("metric")? {
            Some(v) => {
                let s: String = v.extract()?;
                parse_metric_kind(&s)?
            }
            None => MetricKind::Identity,
        };
        let mut cfg = HopConfig::new(n_points, metric);
        if let Some(v) = dict.get_item("warmup_hops")? {
            cfg.warmup_hops = v.extract()?;
        }
        if let Some(v) = dict.get_item("max_depth")? {
            let depth: u32 = v.extract()?;
            if depth < 1 {
                return Err(PyValueError::new_err("hmc max_depth must be at least 1"));
            }
            cfg.max_depth = depth;
        }
        if let Some(v) = dict.get_item("target_accept")? {
            let val: f64 = v.extract()?;
            check_unit_open("hmc target_accept", val)?;
            cfg.target_accept = val;
        }
        if let Some(v) = dict.get_item("control_step")? {
            let val: f64 = v.extract()?;
            check_pos_finite("hmc control_step", val)?;
            cfg.control_step = val;
        }
        if let Some(v) = dict.get_item("target_reach")? {
            let reach: Option<f64> = v.extract()?;
            if let Some(r) = reach {
                check_pos_finite("hmc target_reach", r)?;
            }
            cfg.target_reach = reach;
        }
        if let Some(v) = dict.get_item("drift_period")? {
            let val: usize = v.extract()?;
            check_usize_ge_one("hmc drift_period", val)?;
            cfg.drift_period = val;
        }
        if let Some(v) = dict.get_item("trajectory_temperature")? {
            let temp: Option<f64> = v.extract()?;
            if let Some(t) = temp {
                check_pos_finite("hmc trajectory_temperature", t)?;
            }
            cfg.trajectory_temperature = temp;
        }
        return Ok(cfg);
    }
    Err(PyTypeError::new_err(
        "hmc must be None, \"identity\", \"diagonal\", \"model_hessian\", or a config dict",
    ))
}

/// Scale-aware cluster-search configuration and named presets.
///
/// `Config.recommended(n)` is the measured default: composed surface
/// relocations, depth-rewarded move allocation, and tabu on stall.
#[pyclass(name = "Config")]
#[derive(Clone)]
pub struct PyClusterConfig {
    pub(crate) inner: Config,
    pub(crate) recommended: bool,
}

#[pymethods]
impl PyClusterConfig {
    /// Measured configuration for `n` points.
    #[staticmethod]
    fn recommended(n: usize) -> PyResult<Self> {
        check_n_at_least_two(n)?;
        Ok(Self {
            inner: Config::recommended(n),
            recommended: true,
        })
    }

    /// Measured configuration expressed at `length_scale` and `energy_scale`.
    #[staticmethod]
    #[pyo3(signature = (n, length_scale=1.0, energy_scale=1.0))]
    fn recommended_with_scales(n: usize, length_scale: f64, energy_scale: f64) -> PyResult<Self> {
        check_n_at_least_two(n)?;
        check_pos_finite("length_scale", length_scale)?;
        check_pos_finite("energy_scale", energy_scale)?;
        Ok(Self {
            inner: Config::recommended_with_scales(n, length_scale, energy_scale),
            recommended: true,
        })
    }

    /// Measured configuration in the natural units of a unit-depth pair potential.
    #[staticmethod]
    #[pyo3(signature = (n, kind="lj", rho=None))]
    fn recommended_for_pair(n: usize, kind: &str, rho: Option<f64>) -> PyResult<Self> {
        check_n_at_least_two(n)?;
        let pair_kind = match (kind, rho) {
            ("lj" | "lennard_jones" | "lennard-jones", None) => PairKind::LennardJones,
            ("lj" | "lennard_jones" | "lennard-jones", Some(_)) => {
                return Err(PyValueError::new_err(
                    "rho must be None when kind is \"lj\"",
                ));
            }
            ("morse", Some(r)) => {
                check_pos_finite("rho", r)?;
                PairKind::Morse { rho: r }
            }
            ("morse", None) => {
                return Err(PyValueError::new_err("kind=\"morse\" requires rho"));
            }
            (other, _) => {
                return Err(PyValueError::new_err(format!(
                    "kind must be \"lj\" or \"morse\", got {other:?}"
                )));
            }
        };
        Ok(Self {
            inner: Config::recommended_for_pair(n, pair_kind),
            recommended: true,
        })
    }

    /// Communicating paper arm: orbit kernel plus recognition, not recommended.
    #[staticmethod]
    fn communicating(n: usize) -> PyResult<Self> {
        check_n_at_least_two(n)?;
        Ok(Self {
            inner: Config::communicating(n),
            recommended: false,
        })
    }

    /// Recommended flags with the cost-asymmetric screen and budget-window
    /// temperature.
    #[staticmethod]
    fn derived(n: usize) -> PyResult<Self> {
        check_n_at_least_two(n)?;
        Ok(Self {
            inner: Config::derived(n),
            recommended: true,
        })
    }

    /// Plain Wales-Doye protocol for `n` points (comparison baseline).
    #[staticmethod]
    fn for_cluster(n: usize) -> PyResult<Self> {
        check_n_at_least_two(n)?;
        Ok(Self {
            inner: Config::for_cluster(n),
            recommended: false,
        })
    }

    /// `Config.for_cluster` expressed at `length_scale` and `energy_scale`.
    #[staticmethod]
    #[pyo3(signature = (n, length_scale=1.0, energy_scale=1.0))]
    fn for_cluster_with_scales(n: usize, length_scale: f64, energy_scale: f64) -> PyResult<Self> {
        check_n_at_least_two(n)?;
        check_pos_finite("length_scale", length_scale)?;
        check_pos_finite("energy_scale", energy_scale)?;
        Ok(Self {
            inner: Config::for_cluster_with_scales(n, length_scale, energy_scale),
            recommended: false,
        })
    }

    /// `Config.for_cluster` with dual-core orbit completion and stall restart.
    #[staticmethod]
    fn for_cluster_dual_core(n: usize) -> PyResult<Self> {
        check_n_at_least_two(n)?;
        Ok(Self {
            inner: Config::for_cluster_dual_core(n),
            recommended: false,
        })
    }

    /// Lennard-Jones preset expressed against `length_scale` and `energy_scale`.
    #[staticmethod]
    #[pyo3(signature = (n, length_scale=1.0, energy_scale=1.0))]
    fn with_scales(n: usize, length_scale: f64, energy_scale: f64) -> PyResult<Self> {
        check_n_at_least_two(n)?;
        check_pos_finite("length_scale", length_scale)?;
        check_pos_finite("energy_scale", energy_scale)?;
        Ok(Self {
            inner: Config::with_scales(n, length_scale, energy_scale),
            recommended: false,
        })
    }

    /// Unmeasured SOAP-packing superbasin stack for `n` points.
    #[staticmethod]
    fn packing_superbasin(n: usize) -> PyResult<Self> {
        check_n_at_least_two(n)?;
        Ok(Self {
            inner: Config::packing_superbasin(n),
            recommended: false,
        })
    }

    /// Basin-hopping settings for `n` rigid TIP4P waters.
    #[staticmethod]
    fn for_tip4p(n: usize) -> PyResult<Self> {
        check_n_at_least_two(n)?;
        Ok(Self {
            inner: Config::for_tip4p(n),
            recommended: false,
        })
    }

    /// The communicating arm on `n` rigid TIP4P waters.
    #[staticmethod]
    fn communicating_tip4p(n: usize) -> PyResult<Self> {
        check_n_at_least_two(n)?;
        Ok(Self {
            inner: Config::communicating_tip4p(n),
            recommended: false,
        })
    }

    /// The recommended arm on `n` rigid TIP4P waters.
    #[staticmethod]
    fn recommended_tip4p(n: usize) -> PyResult<Self> {
        check_n_at_least_two(n)?;
        Ok(Self {
            inner: Config::recommended_tip4p(n),
            recommended: true,
        })
    }

    /// Species-aware molecular preset with rigid groups.
    #[staticmethod]
    #[pyo3(signature = (species, groups, energy_scale=1.0))]
    fn for_molecular(
        species: Vec<u32>,
        groups: Vec<Vec<usize>>,
        energy_scale: f64,
    ) -> PyResult<Self> {
        parse_molecular_inputs(&species, &groups, energy_scale)?;
        Ok(Self {
            inner: Config::for_molecular(species, groups, energy_scale),
            recommended: false,
        })
    }

    /// Measured allocation and stall controls over a molecular move library.
    #[staticmethod]
    #[pyo3(signature = (species, groups, energy_scale=1.0))]
    fn recommended_molecular(
        species: Vec<u32>,
        groups: Vec<Vec<usize>>,
        energy_scale: f64,
    ) -> PyResult<Self> {
        parse_molecular_inputs(&species, &groups, energy_scale)?;
        Ok(Self {
            inner: Config::recommended_molecular(species, groups, energy_scale),
            recommended: true,
        })
    }

    /// Return a copy with species labels and optional per-atom diameters.
    #[pyo3(signature = (species, sigmas=None))]
    fn with_species(&self, species: Vec<u32>, sigmas: Option<Vec<f64>>) -> PyResult<Self> {
        if species.len() != self.inner.n_points {
            return Err(PyValueError::new_err(format!(
                "species length {} must match n_points = {}",
                species.len(),
                self.inner.n_points
            )));
        }
        if let Some(ref s) = sigmas {
            if s.len() != self.inner.n_points {
                return Err(PyValueError::new_err(format!(
                    "sigmas length {} must match n_points = {}",
                    s.len(),
                    self.inner.n_points
                )));
            }
            if !s.iter().all(|v| v.is_finite() && *v > 0.0) {
                return Err(PyValueError::new_err("sigmas must be finite and positive"));
            }
        }
        Ok(Self {
            inner: self.inner.clone().with_species(species, sigmas),
            recommended: self.recommended,
        })
    }

    /// Return a copy with dual-core orbit completion and stall restart.
    fn with_dual_orbit_core(&self) -> Self {
        Self {
            inner: self.inner.clone().with_dual_orbit_core(),
            recommended: self.recommended,
        }
    }

    /// Return a copy configured on a rigid centre-and-rotation chart with optional intrabody bonds.
    #[pyo3(signature = (sites, species, bonds = None))]
    fn with_rigid_bodies(
        &self,
        sites: &Bound<'_, PyAny>,
        species: Vec<u32>,
        bonds: Option<Vec<Vec<usize>>>,
    ) -> PyResult<Self> {
        check_n_at_least_two(self.inner.n_points)?;
        let parsed_sites = parse_sites_vec(sites)?;
        let declared = match bonds {
            None => Vec::new(),
            Some(pairs) => pairs
                .into_iter()
                .map(|pair| {
                    if pair.len() != 2 {
                        return Err(PyValueError::new_err("a bond is two site indices"));
                    }
                    Ok([pair[0], pair[1]])
                })
                .collect::<PyResult<Vec<[usize; 2]>>>()?,
        };
        let geometry = RigidBodyGeometry::new_bonded(parsed_sites, species, declared)
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: self.inner.clone().with_rigid_bodies(geometry),
            recommended: self.recommended,
        })
    }

    /// Merge covalently bonded molecular groups on `start`, or `None` if unchanged.
    fn with_flexible_molecules(&self, start: PyReadonlyArray1<'_, f64>) -> PyResult<Option<Self>> {
        let sl = start.as_slice()?;
        Ok(self
            .inner
            .with_flexible_molecules(ndarray::ArrayView1::from(sl))
            .map(|cfg| Self {
                inner: cfg,
                recommended: self.recommended,
            }))
    }

    /// Return an independent shallow clone of this configuration.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// True when every atom is the same element.
    fn monoatomic(&self) -> bool {
        self.inner.monoatomic()
    }

    /// True when CNA 555 / Ih packing diagnostics apply to this configuration.
    fn packing_cna_applies(&self) -> bool {
        self.inner.packing_cna_applies()
    }

    /// Radius of the preset's initial cluster sphere.
    fn start_radius(&self) -> f64 {
        self.inner.start_radius()
    }

    /// Liquid-density start radius for `n` points, `n_points` when omitted.
    #[pyo3(signature = (n=None))]
    fn liquid_start_radius(&self, n: Option<usize>) -> f64 {
        self.inner
            .liquid_start_radius(n.unwrap_or(self.inner.n_points))
    }

    /// The start a seeded search draws, `ras=True` included.
    fn search_start<'py>(&self, py: Python<'py>, seed: u64) -> Bound<'py, PyArray1<f64>> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        PyArray1::from_vec(
            py,
            crate::methods::cluster_hopping::search_start(&self.inner, &mut rng).to_vec(),
        )
    }

    /// Canonical, versioned JSON for the resolved configuration.
    fn resolved_json(&self) -> PyResult<String> {
        self.inner
            .resolved_json()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    /// Lowercase hexadecimal SHA-256 of `resolved_json()`.
    fn resolved_sha256(&self) -> PyResult<String> {
        self.inner
            .resolved_sha256()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    /// Reject JSON configurations that set removed structure-prior keys.
    #[staticmethod]
    fn validate_json(text: &str) -> PyResult<()> {
        Config::validate_json_keys(text).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    /// Reject TOML configurations that set removed structure-prior keys.
    #[staticmethod]
    fn validate_toml(text: &str) -> PyResult<()> {
        Config::validate_toml_keys(text).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    /// Reject removed environment-variable toggles in the current process environment.
    #[staticmethod]
    fn check_removed_env() -> PyResult<()> {
        Config::check_removed_env().map_err(|err| PyValueError::new_err(err.to_string()))
    }

    /// Parse a `H2H_PRIORS` specification, rejecting removed prior names.
    #[staticmethod]
    fn parse_prior_switches(py: Python<'_>, spec: &str) -> PyResult<Py<PyDict>> {
        let switches = Config::parse_prior_switches(spec)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let d = PyDict::new(py);
        d.set_item("soap", switches.soap)?;
        d.set_item("all_off", switches.all_off)?;
        Ok(d.into())
    }

    // --- Structural & derived properties ---

    #[getter]
    fn n_points(&self) -> usize {
        self.inner.n_points
    }

    #[getter]
    fn rigid_body(&self) -> bool {
        self.inner.rigid_body_geometry.is_some()
    }

    #[getter]
    fn burst_moves(&self) -> bool {
        matches!(self.inner.move_library, MoveLibrary::LeanBurst)
    }

    #[setter]
    fn set_burst_moves(&mut self, value: bool) -> PyResult<()> {
        if self.inner.move_library.is_molecular() || self.inner.move_library.is_rigid_body() {
            return Err(PyValueError::new_err(
                "burst_moves cannot be toggled on a molecular or rigid_body move library",
            ));
        }
        self.inner.move_library = if value {
            MoveLibrary::LeanBurst
        } else {
            MoveLibrary::Atomic
        };
        Ok(())
    }

    #[getter]
    fn move_library(&self) -> &'static str {
        move_library_to_str(&self.inner.move_library)
    }

    #[setter]
    fn set_move_library(&mut self, value: &str) -> PyResult<()> {
        if self.inner.move_library.is_molecular() || self.inner.move_library.is_rigid_body() {
            return Err(PyValueError::new_err(
                "cannot replace a molecular or rigid_body move library in place",
            ));
        }
        self.inner.move_library = parse_atomic_move_library(value)?;
        Ok(())
    }

    // --- Scales, cutoffs, container & geometry ---

    #[getter]
    fn length_scale(&self) -> f64 {
        self.inner.length_scale
    }

    #[setter]
    fn set_length_scale(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("length_scale", value)?;
        self.inner.length_scale = value;
        Ok(())
    }

    #[getter]
    fn energy_scale(&self) -> f64 {
        self.inner.energy_scale
    }

    #[setter]
    fn set_energy_scale(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("energy_scale", value)?;
        self.inner.energy_scale = value;
        Ok(())
    }

    #[getter]
    fn neighbour_cutoff(&self) -> f64 {
        self.inner.neighbour_cutoff
    }

    #[setter]
    fn set_neighbour_cutoff(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("neighbour_cutoff", value)?;
        self.inner.neighbour_cutoff = value;
        Ok(())
    }

    #[getter]
    fn symmetrise_cutoff(&self) -> f64 {
        self.inner.symmetrise_cutoff
    }

    #[setter]
    fn set_symmetrise_cutoff(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("symmetrise_cutoff", value)?;
        self.inner.symmetrise_cutoff = value;
        Ok(())
    }

    #[getter]
    fn container(&self) -> f64 {
        self.inner.container
    }

    #[setter]
    fn set_container(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("container", value)?;
        self.inner.container = value;
        Ok(())
    }

    #[getter]
    fn min_separation(&self) -> f64 {
        self.inner.min_separation
    }

    #[setter]
    fn set_min_separation(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("min_separation", value)?;
        self.inner.min_separation = value;
        Ok(())
    }

    #[getter]
    fn pair_range(&self) -> Option<(f64, f64)> {
        self.inner.pair_range.map(|pr| (pr.reach, pr.bond))
    }

    #[setter]
    fn set_pair_range(&mut self, value: Option<(f64, f64)>) -> PyResult<()> {
        self.inner.pair_range = match value {
            None => None,
            Some((reach, bond)) => {
                check_pos_finite("pair_range reach", reach)?;
                check_pos_finite("pair_range bond", bond)?;
                if reach < bond {
                    return Err(PyValueError::new_err(
                        "pair_range reach must be at least bond",
                    ));
                }
                Some(PairRange { reach, bond })
            }
        };
        Ok(())
    }

    #[getter]
    fn call_floor(&self) -> Option<f64> {
        self.inner.call_floor
    }

    #[setter]
    fn set_call_floor(&mut self, value: Option<f64>) -> PyResult<()> {
        if let Some(v) = value {
            check_pos_finite("call_floor", v)?;
        }
        self.inner.call_floor = value;
        Ok(())
    }

    #[getter]
    fn call_floor_elements(&self) -> Option<Vec<u32>> {
        self.inner.call_floor_elements.clone()
    }

    #[setter]
    fn set_call_floor_elements(&mut self, value: Option<Vec<u32>>) -> PyResult<()> {
        if let Some(ref elems) = value
            && elems.len() != self.inner.n_points
        {
            return Err(PyValueError::new_err(format!(
                "call_floor_elements length {} must match n_points = {}",
                elems.len(),
                self.inner.n_points
            )));
        }
        self.inner.call_floor_elements = value;
        Ok(())
    }

    #[getter]
    fn group_cutoff(&self) -> f64 {
        self.inner.group_cutoff
    }

    #[setter]
    fn set_group_cutoff(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("group_cutoff", value)?;
        self.inner.group_cutoff = value;
        Ok(())
    }

    #[getter]
    fn covalent_cutoff(&self) -> f64 {
        self.inner.covalent_cutoff
    }

    #[setter]
    fn set_covalent_cutoff(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("covalent_cutoff", value)?;
        self.inner.covalent_cutoff = value;
        Ok(())
    }

    #[getter]
    fn bond_tolerance(&self) -> f64 {
        self.inner.bond_tolerance
    }

    #[setter]
    fn set_bond_tolerance(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("bond_tolerance", value)?;
        self.inner.bond_tolerance = value;
        Ok(())
    }

    #[getter]
    fn species(&self) -> Option<Vec<u32>> {
        self.inner.species.clone()
    }

    #[setter]
    fn set_species(&mut self, value: Option<Vec<u32>>) -> PyResult<()> {
        if let Some(ref sp) = value
            && sp.len() != self.inner.n_points
        {
            return Err(PyValueError::new_err(format!(
                "species length {} must match n_points = {}",
                sp.len(),
                self.inner.n_points
            )));
        }
        self.inner.species = value;
        Ok(())
    }

    #[getter]
    fn species_sigma(&self) -> Option<Vec<f64>> {
        self.inner.species_sigma.clone()
    }

    #[setter]
    fn set_species_sigma(&mut self, value: Option<Vec<f64>>) -> PyResult<()> {
        if let Some(ref sig) = value {
            if sig.len() != self.inner.n_points {
                return Err(PyValueError::new_err(format!(
                    "species_sigma length {} must match n_points = {}",
                    sig.len(),
                    self.inner.n_points
                )));
            }
            if !sig.iter().all(|v| v.is_finite() && *v > 0.0) {
                return Err(PyValueError::new_err(
                    "species_sigma must be finite and positive",
                ));
            }
        }
        self.inner.species_sigma = value;
        Ok(())
    }

    #[getter]
    fn frozen(&self) -> Option<Vec<bool>> {
        self.inner.frozen.clone()
    }

    #[setter]
    fn set_frozen(&mut self, value: Option<Vec<bool>>) -> PyResult<()> {
        if let Some(ref fr) = value
            && fr.len() != self.inner.n_points
        {
            return Err(PyValueError::new_err(format!(
                "frozen length {} must match n_points = {}",
                fr.len(),
                self.inner.n_points
            )));
        }
        self.inner.frozen = value;
        Ok(())
    }

    #[getter]
    fn active_region(&self) -> Option<(Vec<usize>, usize)> {
        self.inner.active_region.clone()
    }

    #[setter]
    fn set_active_region(&mut self, value: Option<(Vec<usize>, usize)>) -> PyResult<()> {
        if let Some((ref seeds, _)) = value
            && seeds.iter().any(|&idx| idx >= self.inner.n_points)
        {
            return Err(PyValueError::new_err(format!(
                "active_region seed index out of range for n_points = {}",
                self.inner.n_points
            )));
        }
        self.inner.active_region = value;
        Ok(())
    }

    // --- Temperature, basin bias & keying ---

    #[getter]
    fn temperature(&self) -> f64 {
        self.inner.temperature
    }

    #[setter]
    fn set_temperature(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("temperature", value)?;
        self.inner.temperature = value;
        Ok(())
    }

    #[getter]
    fn bias_height(&self) -> f64 {
        self.inner.bias_height
    }

    #[setter]
    fn set_bias_height(&mut self, value: f64) -> PyResult<()> {
        check_nonneg_finite("bias_height", value)?;
        self.inner.bias_height = value;
        Ok(())
    }

    #[getter]
    fn bias_gamma(&self) -> f64 {
        self.inner.bias_gamma
    }

    #[setter]
    fn set_bias_gamma(&mut self, value: f64) -> PyResult<()> {
        if !value.is_finite() || value <= 1.0 {
            return Err(PyValueError::new_err(
                "bias_gamma must be finite and strictly greater than 1",
            ));
        }
        self.inner.bias_gamma = value;
        Ok(())
    }

    #[getter]
    fn basin_stride(&self) -> usize {
        self.inner.basin_stride
    }

    #[setter]
    fn set_basin_stride(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("basin_stride", value)?;
        self.inner.basin_stride = value;
        Ok(())
    }

    #[getter]
    fn merge_radius(&self) -> f64 {
        self.inner.merge_radius
    }

    #[setter]
    fn set_merge_radius(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("merge_radius", value)?;
        self.inner.merge_radius = value;
        Ok(())
    }

    #[getter]
    fn keying(&self) -> &'static str {
        keying_to_str(self.inner.keying)
    }

    #[setter]
    fn set_keying(&mut self, value: &str) -> PyResult<()> {
        self.inner.keying = parse_keying(value)?;
        Ok(())
    }

    #[getter]
    fn keying_sigma(&self) -> f64 {
        self.inner.keying_sigma
    }

    #[setter]
    fn set_keying_sigma(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("keying_sigma", value)?;
        self.inner.keying_sigma = value;
        Ok(())
    }

    #[getter]
    fn morphology_scale(&self) -> f64 {
        self.inner.morphology_scale
    }

    #[setter]
    fn set_morphology_scale(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("morphology_scale", value)?;
        self.inner.morphology_scale = value;
        Ok(())
    }

    #[getter]
    fn shape_keyed(&self) -> bool {
        self.inner.shape_keyed
    }

    #[setter]
    fn set_shape_keyed(&mut self, value: bool) {
        self.inner.shape_keyed = value;
    }

    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta
    }

    #[setter]
    fn set_theta(&mut self, value: f64) -> PyResult<()> {
        if !value.is_finite() || value <= 0.0 || value >= 2.0 {
            return Err(PyValueError::new_err("theta must lie in (0, 2)"));
        }
        self.inner.theta = value;
        Ok(())
    }

    #[getter]
    fn budget_window(&self) -> bool {
        self.inner.budget_window
    }

    #[setter]
    fn set_budget_window(&mut self, value: bool) {
        self.inner.budget_window = value;
    }

    #[getter]
    fn adaptive_height(&self) -> bool {
        self.inner.adaptive_height
    }

    #[setter]
    fn set_adaptive_height(&mut self, value: bool) {
        self.inner.adaptive_height = value;
    }

    #[getter]
    fn height_revisits(&self) -> f64 {
        self.inner.height_revisits
    }

    #[setter]
    fn set_height_revisits(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("height_revisits", value)?;
        self.inner.height_revisits = value;
        Ok(())
    }

    #[getter]
    fn calibrate_radius(&self) -> bool {
        self.inner.calibrate_radius
    }

    #[setter]
    fn set_calibrate_radius(&mut self, value: bool) {
        self.inner.calibrate_radius = value;
    }

    #[getter]
    fn calibrate_quantile(&self) -> f64 {
        self.inner.calibrate_quantile
    }

    #[setter]
    fn set_calibrate_quantile(&mut self, value: f64) -> PyResult<()> {
        check_unit_open("calibrate_quantile", value)?;
        self.inner.calibrate_quantile = value;
        Ok(())
    }

    #[getter]
    fn calibrate_warmup(&self) -> u64 {
        self.inner.calibrate_warmup
    }

    #[setter]
    fn set_calibrate_warmup(&mut self, value: u64) {
        self.inner.calibrate_warmup = value;
    }

    #[getter]
    fn anneal_diversity(&self) -> bool {
        self.inner.anneal_diversity
    }

    #[setter]
    fn set_anneal_diversity(&mut self, value: bool) {
        self.inner.anneal_diversity = value;
    }

    #[getter]
    fn diversity_floor(&self) -> f64 {
        self.inner.diversity_floor
    }

    #[setter]
    fn set_diversity_floor(&mut self, value: f64) -> PyResult<()> {
        check_unit_half_open_pos("diversity_floor", value)?;
        self.inner.diversity_floor = value;
        Ok(())
    }

    #[getter]
    fn flat_histogram(&self) -> bool {
        self.inner.flat_histogram
    }

    #[setter]
    fn set_flat_histogram(&mut self, value: bool) {
        self.inner.flat_histogram = value;
    }

    #[getter]
    fn flat_sweep(&self) -> usize {
        self.inner.flat_sweep
    }

    #[setter]
    fn set_flat_sweep(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("flat_sweep", value)?;
        self.inner.flat_sweep = value;
        Ok(())
    }

    #[getter]
    fn flat_quantile(&self) -> f64 {
        self.inner.flat_quantile
    }

    #[setter]
    fn set_flat_quantile(&mut self, value: f64) -> PyResult<()> {
        check_unit_half_open_pos("flat_quantile", value)?;
        self.inner.flat_quantile = value;
        Ok(())
    }

    #[getter]
    fn statistical_temperature(&self) -> bool {
        self.inner.statistical_temperature
    }

    #[setter]
    fn set_statistical_temperature(&mut self, value: bool) {
        self.inner.statistical_temperature = value;
    }

    #[getter]
    fn energy_bias(&self) -> bool {
        self.inner.energy_bias
    }

    #[setter]
    fn set_energy_bias(&mut self, value: bool) {
        self.inner.energy_bias = value;
    }

    #[getter]
    fn plateau_heat(&self) -> bool {
        self.inner.plateau_heat
    }

    #[setter]
    fn set_plateau_heat(&mut self, value: bool) {
        self.inner.plateau_heat = value;
    }

    #[getter]
    fn incumbent_gap(&self) -> f64 {
        self.inner.incumbent_gap
    }

    #[setter]
    fn set_incumbent_gap(&mut self, value: f64) -> PyResult<()> {
        check_nonneg_finite("incumbent_gap", value)?;
        self.inner.incumbent_gap = value;
        Ok(())
    }

    #[getter]
    fn measured_energy_scale(&self) -> bool {
        self.inner.measured_energy_scale
    }

    #[setter]
    fn set_measured_energy_scale(&mut self, value: bool) {
        self.inner.measured_energy_scale = value;
    }

    #[getter]
    fn long_ranged(&self) -> bool {
        self.inner.long_ranged
    }

    #[setter]
    fn set_long_ranged(&mut self, value: bool) {
        self.inner.long_ranged = value;
    }

    // --- Move allocation & proposals ---

    #[getter]
    fn allocate_moves(&self) -> bool {
        self.inner.allocate_moves
    }

    #[setter]
    fn set_allocate_moves(&mut self, value: bool) {
        self.inner.allocate_moves = value;
    }

    #[getter]
    fn depth_reward(&self) -> bool {
        self.inner.depth_reward
    }

    #[setter]
    fn set_depth_reward(&mut self, value: bool) {
        self.inner.depth_reward = value;
    }

    #[getter]
    fn arm_selection(&self) -> &'static str {
        arm_selection_to_str(self.inner.arm_selection)
    }

    #[setter]
    fn set_arm_selection(&mut self, value: &str) -> PyResult<()> {
        self.inner.arm_selection = parse_arm_selection(value)?;
        Ok(())
    }

    #[getter]
    fn novel_reward(&self) -> bool {
        self.inner.novel_reward
    }

    #[setter]
    fn set_novel_reward(&mut self, value: bool) {
        self.inner.novel_reward = value;
    }

    #[getter]
    fn improvement_rewards(&self) -> bool {
        self.inner.improvement_rewards
    }

    #[setter]
    fn set_improvement_rewards(&mut self, value: bool) {
        self.inner.improvement_rewards = value;
    }

    #[getter]
    fn contextual_moves(&self) -> bool {
        self.inner.contextual_moves
    }

    #[setter]
    fn set_contextual_moves(&mut self, value: bool) {
        self.inner.contextual_moves = value;
    }

    #[getter]
    fn contextual_floor(&self) -> f64 {
        self.inner.contextual_floor
    }

    #[setter]
    fn set_contextual_floor(&mut self, value: f64) -> PyResult<()> {
        check_unit_closed("contextual_floor", value)?;
        self.inner.contextual_floor = value;
        Ok(())
    }

    #[getter]
    fn displacement_only(&self) -> bool {
        self.inner.displacement_only
    }

    #[setter]
    fn set_displacement_only(&mut self, value: bool) {
        self.inner.displacement_only = value;
    }

    #[getter]
    fn angular_moves(&self) -> bool {
        self.inner.angular_moves
    }

    #[setter]
    fn set_angular_moves(&mut self, value: bool) {
        self.inner.angular_moves = value;
    }

    #[getter]
    fn angular_target(&self) -> f64 {
        self.inner.angular_target
    }

    #[setter]
    fn set_angular_target(&mut self, value: f64) -> PyResult<()> {
        check_unit_open("angular_target", value)?;
        self.inner.angular_target = value;
        Ok(())
    }

    #[getter]
    fn energy_lattice(&self) -> bool {
        self.inner.energy_lattice
    }

    #[setter]
    fn set_energy_lattice(&mut self, value: bool) {
        self.inner.energy_lattice = value;
    }

    #[getter]
    fn construct_width(&self) -> usize {
        self.inner.construct_width
    }

    #[setter]
    fn set_construct_width(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("construct_width", value)?;
        self.inner.construct_width = value;
        Ok(())
    }

    #[getter]
    fn reoccupy_interval(&self) -> usize {
        self.inner.reoccupy_interval
    }

    #[setter]
    fn set_reoccupy_interval(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("reoccupy_interval", value)?;
        self.inner.reoccupy_interval = value;
        Ok(())
    }

    #[getter]
    fn prior_soap(&self) -> bool {
        self.inner.prior_soap
    }

    #[setter]
    fn set_prior_soap(&mut self, value: bool) {
        self.inner.prior_soap = value;
    }

    #[getter]
    fn soap_mode(&self) -> &'static str {
        soap_mode_to_str(self.inner.soap_mode)
    }

    #[setter]
    fn set_soap_mode(&mut self, value: &str) -> PyResult<()> {
        self.inner.soap_mode = parse_soap_mode(value)?;
        Ok(())
    }

    #[getter]
    fn soap_class_residual(&self) -> bool {
        self.inner.soap_class_residual
    }

    #[setter]
    fn set_soap_class_residual(&mut self, value: bool) {
        self.inner.soap_class_residual = value;
    }

    #[getter]
    fn soap_repel(&self) -> bool {
        self.inner.soap_repel
    }

    #[setter]
    fn set_soap_repel(&mut self, value: bool) {
        self.inner.soap_repel = value;
    }

    #[getter]
    fn ring_moves(&self) -> bool {
        self.inner.ring_moves
    }

    #[setter]
    fn set_ring_moves(&mut self, value: bool) {
        self.inner.ring_moves = value;
    }

    #[getter]
    fn hold_stereocentres(&self) -> bool {
        self.inner.hold_stereocentres
    }

    #[setter]
    fn set_hold_stereocentres(&mut self, value: bool) {
        self.inner.hold_stereocentres = value;
    }

    #[getter]
    fn homotop_descent(&self) -> bool {
        self.inner.homotop_descent
    }

    #[setter]
    fn set_homotop_descent(&mut self, value: bool) {
        self.inner.homotop_descent = value;
    }

    #[getter]
    fn homotop_ranked(&self) -> bool {
        self.inner.homotop_ranked
    }

    #[setter]
    fn set_homotop_ranked(&mut self, value: bool) {
        self.inner.homotop_ranked = value;
    }

    #[getter]
    fn homotop_window(&self) -> bool {
        self.inner.homotop_window
    }

    #[setter]
    fn set_homotop_window(&mut self, value: bool) {
        self.inner.homotop_window = value;
    }

    #[getter]
    fn homotop_reorder(&self) -> bool {
        self.inner.homotop_reorder
    }

    #[setter]
    fn set_homotop_reorder(&mut self, value: bool) {
        self.inner.homotop_reorder = value;
    }

    #[getter]
    fn general_collective(&self) -> bool {
        self.inner.general_collective
    }

    #[setter]
    fn set_general_collective(&mut self, value: bool) {
        self.inner.general_collective = value;
    }

    #[getter]
    fn cap_twist(&self) -> bool {
        self.inner.cap_twist
    }

    #[setter]
    fn set_cap_twist(&mut self, value: bool) {
        self.inner.cap_twist = value;
    }

    #[getter]
    fn soft_perturb(&self) -> bool {
        self.inner.soft_perturb
    }

    #[setter]
    fn set_soft_perturb(&mut self, value: bool) {
        self.inner.soft_perturb = value;
    }

    #[getter]
    fn soft_modes(&self) -> usize {
        self.inner.soft_modes
    }

    #[setter]
    fn set_soft_modes(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("soft_modes", value)?;
        self.inner.soft_modes = value;
        Ok(())
    }

    #[getter]
    fn soft_steps(&self) -> usize {
        self.inner.soft_steps
    }

    #[setter]
    fn set_soft_steps(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("soft_steps", value)?;
        self.inner.soft_steps = value;
        Ok(())
    }

    #[getter]
    fn cov_perturb(&self) -> bool {
        self.inner.cov_perturb
    }

    #[setter]
    fn set_cov_perturb(&mut self, value: bool) {
        self.inner.cov_perturb = value;
    }

    // --- Screening & local relaxation ---

    #[getter]
    fn relax_steps(&self) -> usize {
        self.inner.relax_steps
    }

    #[setter]
    fn set_relax_steps(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("relax_steps", value)?;
        self.inner.relax_steps = value;
        Ok(())
    }

    #[getter]
    fn screen_steps(&self) -> usize {
        self.inner.screen_steps
    }

    #[setter]
    fn set_screen_steps(&mut self, value: usize) {
        self.inner.screen_steps = value;
    }

    #[getter]
    fn screen_margin(&self) -> f64 {
        self.inner.screen_margin
    }

    #[setter]
    fn set_screen_margin(&mut self, value: f64) -> PyResult<()> {
        check_nonneg_finite("screen_margin", value)?;
        self.inner.screen_margin = value;
        Ok(())
    }

    #[getter]
    fn calibrated_screen(&self) -> bool {
        self.inner.calibrated_screen
    }

    #[setter]
    fn set_calibrated_screen(&mut self, value: bool) {
        self.inner.calibrated_screen = value;
    }

    #[getter]
    fn bayes_screen(&self) -> bool {
        self.inner.bayes_screen
    }

    #[setter]
    fn set_bayes_screen(&mut self, value: bool) {
        self.inner.bayes_screen = value;
    }

    #[getter]
    fn bayes_threshold(&self) -> f64 {
        self.inner.bayes_threshold
    }

    #[setter]
    fn set_bayes_threshold(&mut self, value: f64) -> PyResult<()> {
        check_unit_closed("bayes_threshold", value)?;
        self.inner.bayes_threshold = value;
        Ok(())
    }

    #[getter]
    fn bayes_exploration(&self) -> f64 {
        self.inner.bayes_exploration
    }

    #[setter]
    fn set_bayes_exploration(&mut self, value: f64) -> PyResult<()> {
        check_unit_closed("bayes_exploration", value)?;
        self.inner.bayes_exploration = value;
        Ok(())
    }

    #[getter]
    fn bayes_warmup(&self) -> usize {
        self.inner.bayes_warmup
    }

    #[setter]
    fn set_bayes_warmup(&mut self, value: usize) {
        self.inner.bayes_warmup = value;
    }

    #[getter]
    fn adaptive_screen(&self) -> bool {
        self.inner.adaptive_screen
    }

    #[setter]
    fn set_adaptive_screen(&mut self, value: bool) {
        self.inner.adaptive_screen = value;
    }

    #[getter]
    fn probe_screen(&self) -> bool {
        self.inner.probe_screen
    }

    #[setter]
    fn set_probe_screen(&mut self, value: bool) {
        self.inner.probe_screen = value;
    }

    #[getter]
    fn quench_warmup(&self) -> usize {
        self.inner.quench_warmup
    }

    #[setter]
    fn set_quench_warmup(&mut self, value: usize) {
        self.inner.quench_warmup = value;
    }

    #[getter]
    fn quench_confidence(&self) -> f64 {
        self.inner.quench_confidence
    }

    #[setter]
    fn set_quench_confidence(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("quench_confidence", value)?;
        self.inner.quench_confidence = value;
        Ok(())
    }

    #[getter]
    fn return_screen(&self) -> bool {
        self.inner.return_screen
    }

    #[setter]
    fn set_return_screen(&mut self, value: bool) {
        self.inner.return_screen = value;
    }

    #[getter]
    fn return_polish(&self) -> usize {
        self.inner.return_polish
    }

    #[setter]
    fn set_return_polish(&mut self, value: usize) {
        self.inner.return_polish = value;
    }

    #[getter]
    fn return_polish_after(&self) -> usize {
        self.inner.return_polish_after
    }

    #[setter]
    fn set_return_polish_after(&mut self, value: usize) {
        self.inner.return_polish_after = value;
    }

    #[getter]
    fn finish_improving(&self) -> bool {
        self.inner.finish_improving
    }

    #[setter]
    fn set_finish_improving(&mut self, value: bool) {
        self.inner.finish_improving = value;
    }

    #[getter]
    fn quench_once(&self) -> bool {
        self.inner.quench_once
    }

    #[setter]
    fn set_quench_once(&mut self, value: bool) {
        self.inner.quench_once = value;
    }

    #[getter]
    fn record_gradient(&self) -> f64 {
        self.inner.record_gradient
    }

    #[setter]
    fn set_record_gradient(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("record_gradient", value)?;
        self.inner.record_gradient = value;
        Ok(())
    }

    #[getter]
    fn answer_gradient(&self) -> f64 {
        self.inner.answer_gradient
    }

    #[setter]
    fn set_answer_gradient(&mut self, value: f64) -> PyResult<()> {
        check_nonneg_finite("answer_gradient", value)?;
        self.inner.answer_gradient = value;
        Ok(())
    }

    #[getter]
    fn polish_records(&self) -> usize {
        self.inner.polish_records
    }

    #[setter]
    fn set_polish_records(&mut self, value: usize) {
        self.inner.polish_records = value;
    }

    #[getter]
    fn local_step(&self) -> &'static str {
        local_step_to_str(self.inner.local_step)
    }

    #[setter]
    fn set_local_step(&mut self, value: &str) -> PyResult<()> {
        self.inner.local_step = parse_local_step(value)?;
        Ok(())
    }

    #[getter]
    fn precondition_bonds(&self) -> bool {
        self.inner.precondition_bonds
    }

    #[setter]
    fn set_precondition_bonds(&mut self, value: bool) {
        self.inner.precondition_bonds = value;
    }

    #[getter]
    fn capped_lindh_quench(&self) -> bool {
        self.inner.capped_lindh_quench
    }

    #[setter]
    fn set_capped_lindh_quench(&mut self, value: bool) {
        self.inner.capped_lindh_quench = value;
    }

    #[getter]
    fn staged_quench(&self) -> bool {
        self.inner.staged_quench
    }

    #[setter]
    fn set_staged_quench(&mut self, value: bool) {
        self.inner.staged_quench = value;
    }

    #[getter]
    fn settle_iters(&self) -> usize {
        self.inner.settle_iters
    }

    #[setter]
    fn set_settle_iters(&mut self, value: usize) {
        self.inner.settle_iters = value;
    }

    #[getter]
    fn delayed_acceptance(&self) -> bool {
        self.inner.delayed_acceptance
    }

    #[setter]
    fn set_delayed_acceptance(&mut self, value: bool) {
        self.inner.delayed_acceptance = value;
    }

    #[getter]
    fn surrogate_tolerance(&self) -> f64 {
        self.inner.surrogate_tolerance
    }

    #[setter]
    fn set_surrogate_tolerance(&mut self, value: f64) -> PyResult<()> {
        check_nonneg_finite("surrogate_tolerance", value)?;
        self.inner.surrogate_tolerance = value;
        Ok(())
    }

    // --- Budget, stop & stall / escape mechanisms ---

    #[getter]
    fn max_hops(&self) -> Option<usize> {
        self.inner.max_hops
    }

    #[setter]
    fn set_max_hops(&mut self, value: Option<usize>) -> PyResult<()> {
        if let Some(v) = value {
            check_usize_ge_one("max_hops", v)?;
        }
        self.inner.max_hops = value;
        Ok(())
    }

    #[getter]
    fn target_energy(&self) -> Option<f64> {
        self.inner.target_energy
    }

    #[setter]
    fn set_target_energy(&mut self, value: Option<f64>) -> PyResult<()> {
        if let Some(v) = value
            && !v.is_finite()
        {
            return Err(PyValueError::new_err("target_energy must be finite"));
        }
        self.inner.target_energy = value;
        Ok(())
    }

    #[getter]
    fn witness_tripleton_ceiling(&self) -> bool {
        self.inner.witness_tripleton_ceiling
    }

    #[setter]
    fn set_witness_tripleton_ceiling(&mut self, value: bool) {
        self.inner.witness_tripleton_ceiling = value;
    }

    #[getter]
    fn minima_hopping(&self) -> bool {
        self.inner.minima_hopping
    }

    #[setter]
    fn set_minima_hopping(&mut self, value: bool) {
        self.inner.minima_hopping = value;
    }

    #[getter]
    fn md_escape(&self) -> bool {
        self.inner.md_escape
    }

    #[setter]
    fn set_md_escape(&mut self, value: bool) {
        self.inner.md_escape = value;
    }

    #[getter]
    fn md_escape_dt(&self) -> f64 {
        self.inner.md_escape_dt
    }

    #[setter]
    fn set_md_escape_dt(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("md_escape_dt", value)?;
        self.inner.md_escape_dt = value;
        Ok(())
    }

    #[getter]
    fn md_escape_kinetic(&self) -> f64 {
        self.inner.md_escape_kinetic
    }

    #[setter]
    fn set_md_escape_kinetic(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("md_escape_kinetic", value)?;
        self.inner.md_escape_kinetic = value;
        Ok(())
    }

    #[getter]
    fn md_escape_minima(&self) -> usize {
        self.inner.md_escape_minima
    }

    #[setter]
    fn set_md_escape_minima(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("md_escape_minima", value)?;
        self.inner.md_escape_minima = value;
        Ok(())
    }

    #[getter]
    fn md_escape_max_steps(&self) -> usize {
        self.inner.md_escape_max_steps
    }

    #[setter]
    fn set_md_escape_max_steps(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("md_escape_max_steps", value)?;
        self.inner.md_escape_max_steps = value;
        Ok(())
    }

    #[getter]
    fn md_escape_soften(&self) -> usize {
        self.inner.md_escape_soften
    }

    #[setter]
    fn set_md_escape_soften(&mut self, value: usize) {
        self.inner.md_escape_soften = value;
    }

    #[getter]
    fn escape_on_stall(&self) -> bool {
        self.inner.escape_on_stall
    }

    #[setter]
    fn set_escape_on_stall(&mut self, value: bool) {
        self.inner.escape_on_stall = value;
    }

    #[getter]
    fn retry_noop_stall_leave(&self) -> bool {
        self.inner.retry_noop_stall_leave
    }

    #[setter]
    fn set_retry_noop_stall_leave(&mut self, value: bool) {
        self.inner.retry_noop_stall_leave = value;
    }

    #[getter]
    fn align_directed_stall_source(&self) -> bool {
        self.inner.align_directed_stall_source
    }

    #[setter]
    fn set_align_directed_stall_source(&mut self, value: bool) {
        self.inner.align_directed_stall_source = value;
    }

    #[getter]
    fn escape_lanczos_steps(&self) -> usize {
        self.inner.escape_lanczos_steps
    }

    #[setter]
    fn set_escape_lanczos_steps(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("escape_lanczos_steps", value)?;
        self.inner.escape_lanczos_steps = value;
        Ok(())
    }

    #[getter]
    fn escape_epsilon(&self) -> f64 {
        self.inner.escape_epsilon
    }

    #[setter]
    fn set_escape_epsilon(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("escape_epsilon", value)?;
        self.inner.escape_epsilon = value;
        Ok(())
    }

    #[getter]
    fn escape_amplitude(&self) -> f64 {
        self.inner.escape_amplitude
    }

    #[setter]
    fn set_escape_amplitude(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("escape_amplitude", value)?;
        self.inner.escape_amplitude = value;
        Ok(())
    }

    #[getter]
    fn escape_overshoot(&self) -> f64 {
        self.inner.escape_overshoot
    }

    #[setter]
    fn set_escape_overshoot(&mut self, value: f64) -> PyResult<()> {
        check_nonneg_finite("escape_overshoot", value)?;
        self.inner.escape_overshoot = value;
        Ok(())
    }

    #[getter]
    fn escape_max_climb(&self) -> usize {
        self.inner.escape_max_climb
    }

    #[setter]
    fn set_escape_max_climb(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("escape_max_climb", value)?;
        self.inner.escape_max_climb = value;
        Ok(())
    }

    #[getter]
    fn escape_stall_patience(&self) -> usize {
        self.inner.escape_stall_patience
    }

    #[setter]
    fn set_escape_stall_patience(&mut self, value: usize) {
        self.inner.escape_stall_patience = value;
    }

    #[getter]
    fn escape_stall_factor(&self) -> f64 {
        self.inner.escape_stall_factor
    }

    #[setter]
    fn set_escape_stall_factor(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("escape_stall_factor", value)?;
        self.inner.escape_stall_factor = value;
        Ok(())
    }

    #[getter]
    fn trail_on_stall(&self) -> bool {
        self.inner.trail_on_stall
    }

    #[setter]
    fn set_trail_on_stall(&mut self, value: bool) {
        self.inner.trail_on_stall = value;
    }

    #[getter]
    fn tabu_on_stall(&self) -> bool {
        self.inner.tabu_on_stall
    }

    #[setter]
    fn set_tabu_on_stall(&mut self, value: bool) {
        self.inner.tabu_on_stall = value;
    }

    #[getter]
    fn tabu_capacity(&self) -> usize {
        self.inner.tabu_capacity
    }

    #[setter]
    fn set_tabu_capacity(&mut self, value: usize) {
        self.inner.tabu_capacity = value;
    }

    #[getter]
    fn restart_on_stall(&self) -> bool {
        self.inner.restart_on_stall
    }

    #[setter]
    fn set_restart_on_stall(&mut self, value: bool) {
        self.inner.restart_on_stall = value;
    }

    #[getter]
    fn restart_patience(&self) -> usize {
        self.inner.restart_patience
    }

    #[setter]
    fn set_restart_patience(&mut self, value: usize) {
        self.inner.restart_patience = value;
    }

    #[getter]
    fn branch_on_stall(&self) -> bool {
        self.inner.branch_on_stall
    }

    #[setter]
    fn set_branch_on_stall(&mut self, value: bool) {
        self.inner.branch_on_stall = value;
    }

    /// Polish the incumbent when a hop improves it.
    #[getter]
    fn polish_on_improve(&self) -> bool {
        self.inner.polish_on_improve
    }

    #[setter]
    fn set_polish_on_improve(&mut self, value: bool) {
        self.inner.polish_on_improve = value;
    }

    #[getter]
    fn jump_on_stall(&self) -> bool {
        self.inner.jump_on_stall
    }

    #[setter]
    fn set_jump_on_stall(&mut self, value: bool) {
        self.inner.jump_on_stall = value;
    }

    #[getter]
    fn jump_patience(&self) -> usize {
        self.inner.jump_patience
    }

    #[setter]
    fn set_jump_patience(&mut self, value: usize) {
        self.inner.jump_patience = value;
    }

    #[getter]
    fn jump_steps(&self) -> usize {
        self.inner.jump_steps
    }

    #[setter]
    fn set_jump_steps(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("jump_steps", value)?;
        self.inner.jump_steps = value;
        Ok(())
    }

    #[getter]
    fn jump_step(&self) -> f64 {
        self.inner.jump_step
    }

    #[setter]
    fn set_jump_step(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("jump_step", value)?;
        self.inner.jump_step = value;
        Ok(())
    }

    #[getter]
    fn path_on_stall(&self) -> bool {
        self.inner.path_on_stall
    }

    #[setter]
    fn set_path_on_stall(&mut self, value: bool) {
        self.inner.path_on_stall = value;
    }

    #[getter]
    fn stall_patience(&self) -> usize {
        self.inner.stall_patience
    }

    #[setter]
    fn set_stall_patience(&mut self, value: usize) {
        self.inner.stall_patience = value;
    }

    #[getter]
    fn path_images(&self) -> usize {
        self.inner.path_images
    }

    #[setter]
    fn set_path_images(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("path_images", value)?;
        self.inner.path_images = value;
        Ok(())
    }

    // --- Symmetry & surface hops ---

    #[getter]
    fn symmetrise_on_stall(&self) -> bool {
        self.inner.symmetrise_on_stall
    }

    #[setter]
    fn set_symmetrise_on_stall(&mut self, value: bool) {
        self.inner.symmetrise_on_stall = value;
    }

    #[getter]
    fn symmetrise_patience(&self) -> usize {
        self.inner.symmetrise_patience
    }

    #[setter]
    fn set_symmetrise_patience(&mut self, value: usize) {
        self.inner.symmetrise_patience = value;
    }

    #[getter]
    fn point_symmetrise(&self) -> bool {
        self.inner.point_symmetrise
    }

    #[setter]
    fn set_point_symmetrise(&mut self, value: bool) {
        self.inner.point_symmetrise = value;
    }

    #[getter]
    fn point_symmetrise_on_new(&self) -> bool {
        self.inner.point_symmetrise_on_new
    }

    #[setter]
    fn set_point_symmetrise_on_new(&mut self, value: bool) {
        self.inner.point_symmetrise_on_new = value;
    }

    #[getter]
    fn point_symmetrise_every_accept(&self) -> bool {
        self.inner.point_symmetrise_every_accept
    }

    #[setter]
    fn set_point_symmetrise_every_accept(&mut self, value: bool) {
        self.inner.point_symmetrise_every_accept = value;
    }

    #[getter]
    fn symmetrise_core_fraction(&self) -> f64 {
        self.inner.symmetrise_core_fraction
    }

    #[setter]
    fn set_symmetrise_core_fraction(&mut self, value: f64) -> PyResult<()> {
        check_unit_half_open_pos("symmetrise_core_fraction", value)?;
        self.inner.symmetrise_core_fraction = value;
        Ok(())
    }

    #[getter]
    fn symmetry_tolerance(&self) -> f64 {
        self.inner.symmetry_tolerance
    }

    #[setter]
    fn set_symmetry_tolerance(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("symmetry_tolerance", value)?;
        self.inner.symmetry_tolerance = value;
        Ok(())
    }

    #[getter]
    fn symmetry_merge_radius(&self) -> f64 {
        self.inner.symmetry_merge_radius
    }

    #[setter]
    fn set_symmetry_merge_radius(&mut self, value: f64) -> PyResult<()> {
        check_pos_finite("symmetry_merge_radius", value)?;
        self.inner.symmetry_merge_radius = value;
        Ok(())
    }

    #[getter]
    fn orbit_complete_on_new(&self) -> bool {
        self.inner.orbit_complete_on_new
    }

    #[setter]
    fn set_orbit_complete_on_new(&mut self, value: bool) {
        self.inner.orbit_complete_on_new = value;
    }

    #[getter]
    fn orbit_arm(&self) -> bool {
        self.inner.orbit_arm
    }

    #[setter]
    fn set_orbit_arm(&mut self, value: bool) {
        self.inner.orbit_arm = value;
    }

    #[getter]
    fn orbit_core(&self) -> &'static str {
        orbit_core_to_str(&self.inner.orbit_core)
    }

    #[setter]
    fn set_orbit_core(&mut self, value: &str) -> PyResult<()> {
        let scale = self.inner.length_scale;
        self.inner.orbit_core = match value {
            "fraction" => OrbitCore::Fraction,
            "shells" => OrbitCore::Shells(ShellCore::scaled(scale)),
            "both" => OrbitCore::Both(ShellCore::scaled(scale)),
            other => {
                return Err(PyValueError::new_err(format!(
                    "orbit_core must be \"fraction\", \"shells\", or \"both\", got {other:?}"
                )));
            }
        };
        Ok(())
    }

    #[getter]
    fn continuous_symmetry(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        match self.inner.continuous_symmetry {
            ContinuousSymmetry::Off => {
                dict.set_item("group", "off")?;
            }
            ContinuousSymmetry::Inversion { interval } => {
                dict.set_item("group", "inversion")?;
                dict.set_item("interval", interval)?;
            }
            ContinuousSymmetry::Measure { interval } => {
                dict.set_item("group", "measure")?;
                dict.set_item("interval", interval)?;
            }
        }
        Ok(dict.into())
    }

    #[setter]
    fn set_continuous_symmetry(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.continuous_symmetry = parse_continuous_symmetry(value)?;
        Ok(())
    }

    #[getter]
    fn detected_symmetry(&self) -> bool {
        self.inner.detected_symmetry
    }

    #[setter]
    fn set_detected_symmetry(&mut self, value: bool) {
        self.inner.detected_symmetry = value;
    }

    #[getter]
    fn surface_hop_on_best(&self) -> bool {
        self.inner.surface_hop_on_best
    }

    #[setter]
    fn set_surface_hop_on_best(&mut self, value: bool) {
        self.inner.surface_hop_on_best = value;
    }

    // --- Replicas, tempering & shared history ---

    #[getter]
    fn replicas(&self) -> usize {
        self.inner.replicas
    }

    #[setter]
    fn set_replicas(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("replicas", value)?;
        self.inner.replicas = value;
        Ok(())
    }

    #[getter]
    fn swap_period(&self) -> usize {
        self.inner.swap_period
    }

    #[setter]
    fn set_swap_period(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("swap_period", value)?;
        self.inner.swap_period = value;
        Ok(())
    }

    #[getter]
    fn bias_by_rung(&self) -> bool {
        self.inner.bias_by_rung
    }

    #[setter]
    fn set_bias_by_rung(&mut self, value: bool) {
        self.inner.bias_by_rung = value;
    }

    #[getter]
    fn ladder_top(&self) -> f64 {
        self.inner.ladder_top
    }

    #[setter]
    fn set_ladder_top(&mut self, value: f64) -> PyResult<()> {
        if !value.is_finite() || value < 1.0 {
            return Err(PyValueError::new_err(
                "ladder_top must be finite and at least 1.0",
            ));
        }
        self.inner.ladder_top = value;
        Ok(())
    }

    #[getter]
    fn ladder_mode(&self) -> &'static str {
        ladder_mode_to_str(self.inner.ladder_mode)
    }

    #[setter]
    fn set_ladder_mode(&mut self, value: &str) -> PyResult<()> {
        self.inner.ladder_mode = parse_ladder_mode(value)?;
        Ok(())
    }

    #[getter]
    fn ladder_window(&self) -> usize {
        self.inner.ladder_window
    }

    #[setter]
    fn set_ladder_window(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("ladder_window", value)?;
        self.inner.ladder_window = value;
        Ok(())
    }

    #[getter]
    fn ladder_pilot(&self) -> usize {
        self.inner.ladder_pilot
    }

    #[setter]
    fn set_ladder_pilot(&mut self, value: usize) {
        self.inner.ladder_pilot = value;
    }

    #[getter]
    fn ladder_target_accept(&self) -> f64 {
        self.inner.ladder_target_accept
    }

    #[setter]
    fn set_ladder_target_accept(&mut self, value: f64) -> PyResult<()> {
        check_unit_open("ladder_target_accept", value)?;
        self.inner.ladder_target_accept = value;
        Ok(())
    }

    #[getter]
    fn shared_deposits(&self) -> usize {
        self.inner.shared_deposits
    }

    #[setter]
    fn set_shared_deposits(&mut self, value: usize) {
        self.inner.shared_deposits = value;
    }

    #[getter]
    fn shared_visit_policy(&self) -> &'static str {
        shared_visit_policy_to_str(self.inner.shared_visit_policy)
    }

    #[setter]
    fn set_shared_visit_policy(&mut self, value: &str) -> PyResult<()> {
        self.inner.shared_visit_policy = parse_shared_visit_policy(value)?;
        Ok(())
    }

    #[getter]
    fn exchange_metropolis(&self) -> bool {
        self.inner.exchange_metropolis
    }

    #[setter]
    fn set_exchange_metropolis(&mut self, value: bool) {
        self.inner.exchange_metropolis = value;
    }

    // --- Diagnostics, funnels & superbasins ---

    #[getter]
    fn trace_quenched(&self) -> bool {
        self.inner.trace_quenched
    }

    #[setter]
    fn set_trace_quenched(&mut self, value: bool) {
        self.inner.trace_quenched = value;
    }

    #[getter]
    fn energy_trace(&self) -> bool {
        self.inner.energy_trace
    }

    #[setter]
    fn set_energy_trace(&mut self, value: bool) {
        self.inner.energy_trace = value;
    }

    #[getter]
    fn track_funnels(&self) -> bool {
        self.inner.track_funnels
    }

    #[setter]
    fn set_track_funnels(&mut self, value: bool) {
        self.inner.track_funnels = value;
    }

    #[getter]
    fn funnel_period(&self) -> usize {
        self.inner.funnel_period
    }

    #[setter]
    fn set_funnel_period(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("funnel_period", value)?;
        self.inner.funnel_period = value;
        Ok(())
    }

    #[getter]
    fn superbasin_report(&self) -> bool {
        self.inner.superbasin_report
    }

    #[setter]
    fn set_superbasin_report(&mut self, value: bool) {
        self.inner.superbasin_report = value;
    }

    #[getter]
    fn superbasin_escape(&self) -> bool {
        self.inner.superbasin_escape
    }

    #[setter]
    fn set_superbasin_escape(&mut self, value: bool) {
        self.inner.superbasin_escape = value;
    }

    #[getter]
    fn superbasin_period(&self) -> usize {
        self.inner.superbasin_period
    }

    #[setter]
    fn set_superbasin_period(&mut self, value: usize) -> PyResult<()> {
        check_usize_ge_one("superbasin_period", value)?;
        self.inner.superbasin_period = value;
        Ok(())
    }

    #[getter]
    fn superbasin_quotient(&self) -> bool {
        self.inner.superbasin_quotient
    }

    #[setter]
    fn set_superbasin_quotient(&mut self, value: bool) {
        self.inner.superbasin_quotient = value;
    }

    #[getter]
    fn superbasin_features(&self) -> bool {
        self.inner.superbasin_features
    }

    #[setter]
    fn set_superbasin_features(&mut self, value: bool) {
        self.inner.superbasin_features = value;
    }

    /// Rigid-body site geometry dict `{"sites": (M, 3), "species": list[int], "bonds": list[list[int]]}` or `None`.
    #[getter]
    fn rigid_body_geometry(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.inner
            .rigid_body_geometry
            .as_ref()
            .map(|geom| rigid_body_geometry_to_py(py, geom))
            .transpose()
    }

    /// Optional two-phase relaxation schedule dict or `None`.
    #[getter]
    fn two_phase(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.inner
            .two_phase
            .as_ref()
            .map(|tp| two_phase_to_py(py, tp))
            .transpose()
    }

    #[setter]
    fn set_two_phase(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if value.is_none() {
            self.inner.two_phase = None;
            return Ok(());
        }
        self.inner.two_phase = Some(parse_two_phase(value)?);
        Ok(())
    }

    /// Per-replica two-phase relaxation schedules for communicating searches.
    #[getter]
    fn surfaces(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        self.inner
            .surfaces
            .iter()
            .map(|tp| two_phase_to_py(py, tp))
            .collect()
    }

    #[setter]
    fn set_surfaces(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let items: Vec<Bound<'_, PyAny>> = value.extract()?;
        let parsed: PyResult<Vec<TwoPhase>> = items.iter().map(parse_two_phase).collect();
        self.inner.surfaces = parsed?;
        Ok(())
    }

    /// Optional lattice-reoccupation stage configuration dict or `None`.
    #[getter]
    fn reoccupy(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.inner
            .reoccupy
            .as_ref()
            .map(|cfg| reoccupy_to_py(py, cfg))
            .transpose()
    }

    #[setter]
    fn set_reoccupy(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if value.is_none() {
            self.inner.reoccupy = None;
            return Ok(());
        }
        self.inner.reoccupy = Some(parse_reoccupy(self.inner.n_points, value)?);
        Ok(())
    }

    /// Optional Hamiltonian Monte Carlo escape configuration dict or `None`.
    #[getter]
    fn hmc(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.inner
            .hmc
            .as_ref()
            .map(|cfg| hmc_to_py(py, cfg))
            .transpose()
    }

    #[setter]
    fn set_hmc(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if value.is_none() {
            self.inner.hmc = None;
            return Ok(());
        }
        self.inner.hmc = Some(parse_hmc(self.inner.n_points, value)?);
        Ok(())
    }

    fn __repr__(&self) -> String {
        if self.inner.bayes_screen && self.inner.budget_window {
            format!("Config.derived({})", self.inner.n_points)
        } else if !self.inner.surfaces.is_empty()
            && self.inner.orbit_complete_on_new
            && matches!(
                self.inner.shared_visit_policy,
                SharedVisitPolicy::Recognition
            )
        {
            format!("Config.communicating({})", self.inner.n_points)
        } else if self.recommended {
            format!("Config.recommended({})", self.inner.n_points)
        } else {
            format!("Config.for_cluster({})", self.inner.n_points)
        }
    }
}
