use anneal_core::accept::Metropolis;
use anneal_core::cool::LogCool;
use anneal_core::methods::cluster_hopping::Config;
use anneal_core::run_rs;
use anneal_core::sampler::HoppingSampler;
use eindir_core::FPair;
use ndarray::{Array1, ArrayView1, array};

fn hs1_value_gradient(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let a = x[1] - x[0] * x[0];
    (
        100.0 * a * a + (1.0 - x[0]).powi(2),
        array![-400.0 * x[0] * a + 2.0 * (x[0] - 1.0), 200.0 * a],
    )
}

fn lj_value_gradient(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut value = 0.0;
    let mut gradient = Array1::zeros(x.len());
    for i in 0..n {
        for j in i + 1..n {
            let d = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            let r2 = d.iter().map(|v| v * v).sum::<f64>();
            let inv2 = 1.0 / r2;
            let inv6 = inv2.powi(3);
            let inv12 = inv6 * inv6;
            value += 4.0 * (inv12 - inv6);
            let coefficient = 24.0 * inv2 * (2.0 * inv12 - inv6);
            for k in 0..3 {
                gradient[3 * i + k] -= coefficient * d[k];
                gradient[3 * j + k] += coefficient * d[k];
            }
        }
    }
    (value, gradient)
}

fn gradient_quench(
    mut position: Array1<f64>,
    steps: usize,
    rate: f64,
    value_gradient: fn(ArrayView1<f64>) -> (f64, Array1<f64>),
) -> FPair<f64> {
    let mut value = f64::INFINITY;
    for _ in 0..steps {
        let (next_value, gradient) = value_gradient(position.view());
        value = next_value;
        position.scaled_add(-rate, &gradient);
    }
    value = value_gradient(position.view()).0;
    FPair {
        pos: position,
        val: value,
    }
}

#[test]
fn cutest_hs1_and_lj_hexamer_share_the_hop_loop() {
    let cooling = LogCool::new(1.0, 2.0);
    let hs1 = HoppingSampler::new(
        array![-1.2, 1.0],
        anneal_core::movekernel::Gaussian::new(0.25),
        cooling.clone(),
        Metropolis,
        80,
        |x, steps| gradient_quench(x.to_owned(), steps, 1.0e-3, hs1_value_gradient),
    );
    let hs1_history = run_rs(hs1, &cooling, 4, 8, 17);

    let hexamer_start = array![
        1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, -1.0,
    ];
    let hexamer_config = Config::with_scales(6, 1.0, 1.0);
    let hexamer = HoppingSampler::new(
        hexamer_start,
        hexamer_config.proposal_kernel(),
        cooling.clone(),
        Metropolis,
        40,
        |x, steps| gradient_quench(x.to_owned(), steps, 1.0e-4, lj_value_gradient),
    );
    let hexamer_history = run_rs(hexamer, &cooling, 4, 8, 19);

    for history in [&hs1_history, &hexamer_history] {
        assert_eq!(history.total_accepted() + history.total_rejected(), 32);
        assert!(history.best.val.is_finite());
    }
}
