use ndarray::{Array1, ArrayView1};
use rgpot_core::profile::{ProfileEvaluation, ProfileRequest, ProfileSession};

pub(crate) fn profile_prefix(engine: &str) -> Option<&str> {
    match engine {
        "nwchemc" | "cpmdc" => Some(engine),
        _ => None,
    }
}

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

pub(crate) fn optimizer_value_gradient(
    evaluation: ProfileEvaluation,
) -> (f64, Array1<f64>) {
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
pub(crate) struct ProfileEngine {
    session: ProfileSession,
    atomic_numbers: Vec<i32>,
    box_matrix: Option<[f64; 9]>,
    failures: usize,
}

impl ProfileEngine {
    pub(crate) fn load(
        prefix: &str,
        atomic_numbers: Vec<i32>,
        box_matrix: Option<[f64; 9]>,
    ) -> Self {
        let config_variable = format!("{}_CONFIG", prefix.to_ascii_uppercase());
        let config_path = std::env::var("POTENTIAL_CONFIG")
            .or_else(|_| std::env::var(&config_variable))
            .unwrap_or_else(|_| panic!("POTENTIAL_CONFIG or {config_variable}"));
        let config = std::fs::read(&config_path).expect("PotentialConfig message");
        let explicit_library = std::env::var("POTENTIAL_LIBRARY").ok();
        let session = unsafe {
            ProfileSession::load(
                prefix,
                explicit_library.as_deref().map(std::path::Path::new),
                &config,
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
        let request = profile_request(
            &positions,
            &self.atomic_numbers,
            self.box_matrix.as_ref(),
        );
        match self.session.evaluate(&request) {
            Ok(evaluation) => Some(optimizer_value_gradient(evaluation)),
            Err(error) => {
                self.failures += 1;
                if self.failures == 1 || self.failures % 500 == 0 {
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
