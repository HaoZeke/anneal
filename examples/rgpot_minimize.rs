//! Minimize an rgpot potential with anneal.
//!
//! End-to-end demonstration of the eindir lingua franca:
//!
//! 1. rgpot builds a `rgpot_potential_t` from a force/energy callback. That
//!    struct embeds eindir's `eindir_objective_t` as its first member, so a
//!    `rgpot_potential_t*` IS-A `eindir_objective_t*` at zero cost.
//! 2. eindir's `EindirObjectiveWrapper` borrows that `eindir_objective_t` and
//!    presents it as a Rust `Objective<f64>`.
//! 3. anneal's Boltzmann simulated-annealing driver minimizes any
//!    `Objective<f64>` -- including the wrapped rgpot potential.
//!
//! The potential here is a separable quadratic bowl with its minimum at
//! `TARGET`: energy = sum_i (x_i - TARGET_i)^2, force_i = -2 (x_i - TARGET_i).
//! It is a stand-in for any real rgpot potential (LJ, CuH2, NWChem, ...): all
//! of them plug into rgpot through the exact same force-callback ABI used here.
//!
//! Run: `cargo run --features capi --example rgpot_minimize`
//! It is also wired as an asserting integration test (see the bottom module).

use std::os::raw::c_void;

use anneal_core::run_rs_qmc_variant;
use anneal_core::variant::boltzmann;
use eindir_core::Objective;
use eindir_core::ffi::{EindirObjectiveWrapper, eindir_objective_t};

use rgpot_core::eindir::{rgpot_potential_free_eindir, rgpot_potential_new_eindir};
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::rgpot_tensor_owned_cpu_f64_2d;
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

/// Minimum of the quadratic potential, flat `[atom0_xyz, atom1_xyz, ...]`.
const TARGET: [f64; 9] = [0.5, -0.5, 1.0, -1.0, 0.0, 0.25, 0.75, -0.25, -0.75];

/// rgpot force callback: a separable quadratic bowl centred on `user_data`.
///
/// This is the standard rgpot potential ABI -- the same shape every rgpot
/// backend (LJ, CuH2, NWChem dlopen, ...) implements. It reads positions from
/// the DLPack input tensor, writes energy + an owned forces tensor.
unsafe extern "C" fn quadratic_pes(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    let target = unsafe { &*(user_data as *const [f64; 9]) };
    let inp = unsafe { &*input };
    let n = match unsafe { inp.n_atoms() } {
        Some(n) => n,
        None => return rgpot_status_t::RGPOT_INVALID_PARAMETER,
    };
    let dim = n * 3;
    let pt = unsafe { &(*inp.positions).dl_tensor };
    let x = unsafe { std::slice::from_raw_parts(pt.data as *const f64, dim) };

    let mut energy = 0.0;
    let mut forces = vec![0.0f64; dim];
    for i in 0..dim {
        let d = x[i] - target[i];
        energy += d * d;
        forces[i] = -2.0 * d; // F = -dE/dx
    }

    let out = unsafe { &mut *output };
    out.energy = energy;
    out.variance = 0.0;
    out.forces = unsafe { rgpot_tensor_owned_cpu_f64_2d(forces.as_ptr(), n as i64, 3) };
    rgpot_status_t::RGPOT_SUCCESS
}

/// Build the rgpot potential, minimize it with Boltzmann SA, return
/// `(best_value, best_position)`. `n_starts`/`n_epochs`/`steps` set the QMC
/// multistart budget (the example uses a thorough budget; the test a lighter
/// one that still reaches the basin quickly).
fn minimize_rgpot_with_anneal(
    seed: u64,
    n_starts: usize,
    n_epochs: usize,
    steps: usize,
) -> (f64, Vec<f64>) {
    let n_atoms = 3usize;
    let dim = n_atoms * 3;
    let atmnrs = [1i32; 3];
    let box_ = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
    let low = vec![-5.0f64; dim];
    let high = vec![5.0f64; dim];

    // 1. rgpot potential (IS-A eindir objective via first-member embedding).
    let pot = unsafe {
        rgpot_potential_new_eindir(
            quadratic_pes,
            &TARGET as *const [f64; 9] as *mut c_void,
            None,
            n_atoms,
            atmnrs.as_ptr(),
            box_.as_ptr(),
            low.as_ptr(),
            high.as_ptr(),
        )
    };
    assert!(!pot.is_null(), "rgpot potential construction failed");

    let result = {
        // 2. Zero-cost IS-A cast + eindir Rust view over the C objective.
        let obj_ptr = pot as *mut eindir_objective_t;
        let objective = unsafe { EindirObjectiveWrapper::new(&*obj_ptr) };

        // Sanity: the wrapper reports the rgpot potential's dimensionality.
        assert_eq!(objective.dim(), dim);

        // 3. anneal minimizes the wrapped rgpot objective. A small move scale
        //    (sigma) lets the cooled chain settle tightly into the basin, and
        //    QMC multistart from low-discrepancy points makes the result robust
        //    to the random `[-5, 5]^9` start.
        let variant = boltzmann(objective, 3.0, 0.15).expect("Boltzmann construction");
        let history = run_rs_qmc_variant(variant, n_starts, n_epochs, steps, seed);
        (history.best.val, history.best.pos.to_vec())
    };

    unsafe { rgpot_potential_free_eindir(pot) };
    result
}

fn main() {
    // Moderate budget for the standalone demo (runs in a few seconds).
    let (best_val, best_pos) = minimize_rgpot_with_anneal(42, 24, 300, 200);

    println!("anneal minimized an rgpot potential (Boltzmann SA, 24-start QMC, 300x200 steps)");
    println!("  best energy : {best_val:.6}  (target minimum: 0.0)");
    println!("  best coords : {best_pos:?}");
    println!("  target      : {:?}", TARGET.to_vec());

    let max_err = best_pos
        .iter()
        .zip(TARGET.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("  max per-coord error: {max_err:.4}");
    assert!(
        best_val < 0.1,
        "anneal did not reach the rgpot minimum: best energy {best_val} (want < 0.1)"
    );
    println!("OK: anneal drove the rgpot potential to its minimum.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anneal_minimizes_rgpot_potential() {
        // Lighter, CI-friendly budget; still drops energy ~500x from the random
        // [-5, 5]^9 start (mean ~75) into the basin around the minimum.
        let (best_val, best_pos) = minimize_rgpot_with_anneal(42, 16, 250, 150);
        assert!(
            best_val < 0.15,
            "best energy {best_val} not near the rgpot minimum (0.0)"
        );
        for (got, want) in best_pos.iter().zip(TARGET.iter()) {
            assert!(
                (got - want).abs() < 0.3,
                "coordinate {got} far from target {want}"
            );
        }
    }
}
