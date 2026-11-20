#![allow(dead_code)]

#[cfg(feature = "profile")]
use ndarray::{Array1, ArrayView1};
#[cfg(feature = "profile")]
use rgpot_core::profile::{ProfileEvaluation, ProfileRequest, ProfileSession};

pub(crate) fn profile_prefix(engine: &str) -> Option<&str> {
    match engine {
        "nwchemc" | "cpmdc" => Some(engine),
        _ => None,
    }
}

/// Resolve the `PotentialConfig` file path for `prefix`.
///
/// The engine-specific variable (`<PREFIX>_CONFIG`, e.g. `CPMDC_CONFIG`
/// or `NWCHEMC_CONFIG`) takes precedence over the generic
/// `POTENTIAL_CONFIG` fallback so generated engine decks are not
/// silently shadowed by an ambient `POTENTIAL_CONFIG`. Empty values are
/// treated as unset.
pub(crate) fn resolve_config_path(
    prefix: &str,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<String, String> {
    let config_variable = format!("{}_CONFIG", prefix.to_ascii_uppercase());
    if let Some(path) = lookup(&config_variable).filter(|value| !value.is_empty()) {
        return Ok(path);
    }
    if let Some(path) = lookup("POTENTIAL_CONFIG").filter(|value| !value.is_empty()) {
        return Ok(path);
    }
    Err(format!(
        "neither {config_variable} nor POTENTIAL_CONFIG is set"
    ))
}

#[cfg(feature = "profile")]
pub(crate) fn profile_request<'a>(
    positions: &'a [f64],
    atomic_numbers: &'a [i32],
    box_matrix: Option<&'a [f64; 9]>,
) -> ProfileRequest<'a> {
    ProfileRequest {
        positions,
        atomic_numbers,
        box_matrix,
        length_unit: "angstrom",
        energy_unit: "eV",
    }
}

#[cfg(feature = "profile")]
pub(crate) fn optimizer_value_gradient(evaluation: ProfileEvaluation) -> (f64, Array1<f64>) {
    let gradient = Array1::from(
        evaluation
            .forces
            .into_iter()
            .map(|force| -force)
            .collect::<Vec<_>>(),
    );
    (evaluation.energy, gradient)
}

/// One persistent in-process session for any backend implementing rgpot's
/// minimum profile.
#[cfg(feature = "profile")]
pub(crate) struct ProfileEngine {
    session: ProfileSession,
    atomic_numbers: Vec<i32>,
    box_matrix: Option<[f64; 9]>,
    failures: usize,
}

#[cfg(feature = "profile")]
impl ProfileEngine {
    pub(crate) fn load(
        prefix: &str,
        atomic_numbers: Vec<i32>,
        box_matrix: Option<[f64; 9]>,
    ) -> Self {
        let config_path = resolve_config_path(prefix, |name| std::env::var(name).ok())
            .unwrap_or_else(|error| panic!("{error}"));
        let config = std::fs::read(&config_path).expect("PotentialConfig message");
        let explicit_library = std::env::var("POTENTIAL_LIBRARY").ok();
        let mut expectation = rgpot_core::compat::Expectation::default();
        // cpmdc and nwchemc report the Potentials.capnp file id
        // without the 0x prefix rgpot-core stores. The strings name
        // that same id.
        if prefix == "cpmdc" || prefix == "nwchemc" {
            expectation.schema_id = "bd1f89fa17369103";
        }
        let session = unsafe {
            ProfileSession::load_with(
                prefix,
                explicit_library.as_deref().map(std::path::Path::new),
                &config,
                &expectation,
            )
        }
        .unwrap_or_else(|error| panic!("load {prefix} profile: {error}"));
        println!(
            "  profile {} {} ABI {} from {}",
            session.prefix(),
            session.version(),
            session.abi_version(),
            session.library_path()
        );
        Self {
            session,
            atomic_numbers,
            box_matrix,
            failures: 0,
        }
    }

    pub(crate) fn eval(&mut self, x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
        let positions = x.iter().copied().collect::<Vec<_>>();
        let request = profile_request(&positions, &self.atomic_numbers, self.box_matrix.as_ref());
        match self.session.evaluate(&request) {
            Ok(evaluation) => Some(optimizer_value_gradient(evaluation)),
            Err(error) => {
                self.failures += 1;
                if self.failures == 1 || self.failures.is_multiple_of(500) {
                    eprintln!("  profile failure {}: {error}", self.failures);
                }
                None
            }
        }
    }

    pub(crate) fn failures(&self) -> usize {
        self.failures
    }
}
