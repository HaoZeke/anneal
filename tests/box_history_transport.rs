use anneal_core::methods::box_hopping::{
    BoxEnsembleConfig, box_ensemble_optimize, box_values_ensemble_optimize,
};
use anneal_core::methods::ensemble::HistoryMode;
use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

const CHILD_CASE: &str = "ANNEAL_BOX_HISTORY_TRANSPORT_CHILD";
const TEST_NAME: &str = "requested_shared_transport_cannot_silently_disable_history";

struct CountedQuadratic {
    bounds: Bounds<f64>,
    evaluations: AtomicUsize,
    gradients: AtomicUsize,
}

impl Objective<f64> for CountedQuadratic {
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        x.dot(&x)
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn dim(&self) -> usize {
        1
    }
}

impl Gradient<f64> for CountedQuadratic {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.gradients.fetch_add(1, Ordering::Relaxed);
        2.0 * &x
    }

    fn dim(&self) -> usize {
        1
    }
}

#[test]
fn requested_shared_transport_cannot_silently_disable_history() {
    if let Ok(case) = std::env::var(CHILD_CASE) {
        let (driver, mode) = case.split_once(':').expect("driver and history mode");
        let history = match mode {
            "shared" => HistoryMode::Shared,
            "private" => HistoryMode::Private,
            "none" => HistoryMode::None,
            _ => panic!("unknown history mode"),
        };
        let objective = CountedQuadratic {
            bounds: Bounds::new(array![-1.0], array![1.0], 0.0),
            evaluations: AtomicUsize::new(0),
            gradients: AtomicUsize::new(0),
        };
        let config = BoxEnsembleConfig {
            replicas: 2,
            budget: 16,
            history,
            ..BoxEnsembleConfig::default()
        };
        let result = catch_unwind(AssertUnwindSafe(|| match driver {
            "gradient" => box_ensemble_optimize(&objective, &objective, 7, None, &config),
            "values" => box_values_ensemble_optimize(&objective, 7, None, &config),
            _ => panic!("unknown driver"),
        }));
        let evaluations = objective.evaluations.load(Ordering::Relaxed);
        let gradients = objective.gradients.load(Ordering::Relaxed);
        if matches!(history, HistoryMode::Shared) {
            assert!(
                result.is_err(),
                "an unsupported requested transport must fail, not return an unshared search"
            );
            assert_eq!((evaluations, gradients), (0, 0));
        } else {
            let result = result.expect("private and disabled history do not request a transport");
            assert!(evaluations > 0);
            assert_eq!((result.n_evals, result.n_grads), (evaluations, gradients));
            assert!(evaluations + gradients <= config.budget);
        }
        return;
    }

    for driver in ["gradient", "values"] {
        for mode in ["private", "none", "shared"] {
            let output = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", TEST_NAME, "--nocapture"])
                .env(CHILD_CASE, format!("{driver}:{mode}"))
                .env("HISTORY_NNG", "udp://127.0.0.1:1")
                .env_remove("HISTORY_NNG_SERVE")
                .output()
                .expect("launch isolated history configuration");
            assert!(
                output.status.success(),
                "{driver}:{mode} failed:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}

#[cfg(feature = "history-nng")]
#[test]
fn box_driver_admissions_reach_an_independent_nng_client() {
    use anneal_core::history_nng::{HistoryNngClient, HistoryNngServer};
    use anneal_core::methods::minima_hopping::{HistoryHook, HistoryMembership};

    const DRIVER_CASE: &str = "ANNEAL_BOX_HISTORY_POSITIVE_CHILD";
    const NAME: &str = "box_driver_admissions_reach_an_independent_nng_client";
    if let Ok(driver) = std::env::var(DRIVER_CASE) {
        let url = std::env::var("HISTORY_NNG").unwrap();
        let _server = HistoryNngServer::bind(&url, 1e-3, 1e-3).unwrap();
        let objective = CountedQuadratic {
            bounds: Bounds::new(array![-1.0], array![1.0], 0.0),
            evaluations: AtomicUsize::new(0),
            gradients: AtomicUsize::new(0),
        };
        let config = BoxEnsembleConfig {
            replicas: 1,
            budget: if driver == "gradient" { 2 } else { 3 },
            history: HistoryMode::Shared,
            membership: HistoryMembership::Accepted,
            ..BoxEnsembleConfig::default()
        };
        let anchor = array![0.0];
        let result = match driver.as_str() {
            "gradient" => {
                box_ensemble_optimize(&objective, &objective, 7, Some(anchor.view()), &config)
            }
            "values" => box_values_ensemble_optimize(&objective, 7, Some(anchor.view()), &config),
            _ => panic!("unknown driver"),
        };
        let expected_calls = if driver == "gradient" { (1, 1) } else { (3, 0) };
        assert_eq!((result.n_evals, result.n_grads), expected_calls);
        assert_eq!(
            (
                objective.evaluations.load(Ordering::Relaxed),
                objective.gradients.load(Ordering::Relaxed)
            ),
            expected_calls,
        );
        assert_eq!((result.history_minima, result.history_observations), (1, 1));
        assert_eq!(result.best_pos, anchor);
        let mut observer =
            HistoryNngClient::dial(&url, array![2.0], HistoryMembership::Accepted).unwrap();
        assert_eq!(observer.minimum_count(), Some(1));
        let shared = observer
            .observe(0.0, anchor.view(), array![0.0].view())
            .unwrap();
        assert!(!shared.is_new && !shared.first_observation);
        assert_eq!((shared.visits, shared.observed_visits), (2, 2));
        return;
    }

    for driver in ["gradient", "values"] {
        let url = format!(
            "ipc:///tmp/anneal-box-history-{}-{driver}",
            std::process::id()
        );
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", NAME, "--nocapture"])
            .env(DRIVER_CASE, driver)
            .env("HISTORY_NNG", url)
            .env_remove("HISTORY_NNG_SERVE")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{driver} failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}
