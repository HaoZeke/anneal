use anneal_core::run_rs_qmc_variant;
use anneal_core::variant::gsa;
use eindir_core::{Bounds, Objective};
use ndarray::{ArrayView1, array};

#[derive(Clone)]
struct DeceptiveBasin {
    bounds: Bounds<f64>,
}

impl DeceptiveBasin {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(array![-1.0, -1.0], array![1.0, 1.0], 0.0),
        }
    }
}

impl Objective<f64> for DeceptiveBasin {
    fn dim(&self) -> usize {
        2
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let shallow = (x[0] - 0.35).powi(2) + (x[1] - 0.35).powi(2);
        let deep = 0.03 * ((x[0] + 0.5).powi(2) + (x[1] - 1.0 / 3.0).powi(2)) - 0.75;
        shallow.min(deep)
    }
}

#[test]
fn qmc_multistart_sees_deceptive_basin() {
    let variant = gsa(DeceptiveBasin::new(), 1.0, 2.2, 1.5).expect("GSA construction");
    let history = run_rs_qmc_variant(variant, 8, 2, 2, 7);

    assert!(history.best.val < -0.7, "best value: {}", history.best.val);
}
