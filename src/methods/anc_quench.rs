//! Lindh-preconditioned quench for a molecular Cartesian library.
//!
//! A cold L-BFGS step on this library is the raw steepest direction. With
//! an empty memory its length is one coordinate unit, which is enough to
//! climb a bond; the strong Wolfe search then spends up to two line-search
//! budgets walking back, and a failed search throws the curvature away.
//! The next step opens at unit length again. Energy and gradient already
//! arrive together, so splitting them is not where the evaluations go.
//!
//! The model Hessian is Lindh, Bernhardsson, Karlström, and Malmqvist,
//! Chem. Phys. Lett. 241, 423 (1995). It is the same initial curvature
//! xtb's approximate normal-coordinate step is built on. The two-loop
//! recursion keeps the pairs from this quench, so later steps reuse that
//! curvature instead of a collapsed scalar step. The trial is shortened
//! to the current trust radius before any evaluation, and Armijo
//! backtracks at most `MAX_BACKTRACK` times. A step that used the whole
//! radius and lowered the energy may try a longer one next time.

use ndarray::{Array1, ArrayView1};
use rgmin::{Ghost, Lindh, LindhParams};

/// Largest motion of one atom in one trial, in the coordinate unit.
pub const MAX_ATOM_STEP: f64 = 0.2;

/// Opening trust radius. A Lindh bond is softer than a GFN2 O–H, so the
/// first model step can be long; this bound is what the first trial uses.
const TRUST_START: f64 = 0.05;

/// Armijo constant.
const ARMIJO: f64 = 1.0e-4;

/// Halvings of a rejected trial.
const MAX_BACKTRACK: usize = 10;

/// Curvature pairs retained inside one quench.
const MEMORY: usize = 8;

struct Pair {
    s: Array1<f64>,
    y: Array1<f64>,
    rho: f64,
}

/// Relaxes `x0` until the infinity-norm of the gradient is below `gtol`.
///
/// `z` is one atomic number per atom. `fg` returns `None` when the caller
/// is out of budget. The third value is the number of evaluations,
/// including the one at `x0`.
pub fn anc_minimize<F>(
    x0: ArrayView1<f64>,
    z: &[i32],
    max_iter: usize,
    gtol: f64,
    fg: F,
) -> (f64, Array1<f64>, usize)
where
    F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
{
    anc_minimize_watched(x0, z, max_iter, gtol, fg, |_, _| true)
}

/// [`anc_minimize`] that stops when `watch` returns false.
///
/// `watch` sees the iteration index and the value at the last accepted
/// point, which is the same contract as the L-BFGS hook.
pub fn anc_minimize_watched<F, W>(
    x0: ArrayView1<f64>,
    z: &[i32],
    max_iter: usize,
    gtol: f64,
    mut fg: F,
    mut watch: W,
) -> (f64, Array1<f64>, usize)
where
    F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    W: FnMut(usize, f64) -> bool,
{
    let mut x = x0.to_owned();
    let mut evals = 0usize;
    let Some((mut energy, mut gradient)) = fg(x.view()) else {
        return (f64::INFINITY, x, evals);
    };
    evals += 1;
    let mut memory: Vec<Pair> = Vec::new();
    let mut trust = TRUST_START;
    for iteration in 0..max_iter {
        if !watch(iteration, energy) {
            break;
        }
        if infinity_norm(gradient.view()) < gtol {
            break;
        }
        let accepted = attempt(
            &mut x,
            &mut energy,
            &mut gradient,
            z,
            &mut memory,
            trust,
            &mut fg,
            &mut evals,
        );
        let Some(taken) = accepted else {
            memory.clear();
            trust *= 0.25;
            if trust < 1.0e-4 {
                break;
            }
            let retry = attempt(
                &mut x,
                &mut energy,
                &mut gradient,
                z,
                &mut memory,
                trust,
                &mut fg,
                &mut evals,
            );
            let Some(taken) = retry else {
                break;
            };
            trust = (taken * 2.0).clamp(1.0e-3, MAX_ATOM_STEP);
            continue;
        };
        if taken > 0.9 * trust {
            trust = (trust * 1.5).min(MAX_ATOM_STEP);
        } else {
            trust = (taken * 2.0).clamp(1.0e-3, MAX_ATOM_STEP);
        }
    }
    (energy, x, evals)
}

fn attempt<F>(
    x: &mut Array1<f64>,
    energy: &mut f64,
    gradient: &mut Array1<f64>,
    z: &[i32],
    memory: &mut Vec<Pair>,
    trust: f64,
    fg: &mut F,
    evals: &mut usize,
) -> Option<f64>
where
    F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
{
    let mut direction = preconditioned_direction(x.view(), gradient.view(), z, memory);
    if !descent(&direction, gradient.view()) {
        memory.clear();
        direction = preconditioned_direction(x.view(), gradient.view(), z, memory);
    }
    cap_atoms(&mut direction, trust);
    if !descent(&direction, gradient.view()) {
        return None;
    }
    let slope = direction.dot(gradient);
    backtrack(x, energy, gradient, &direction, slope, memory, fg, evals)
}

fn backtrack<F>(
    x: &mut Array1<f64>,
    energy: &mut f64,
    gradient: &mut Array1<f64>,
    direction: &Array1<f64>,
    slope: f64,
    memory: &mut Vec<Pair>,
    fg: &mut F,
    evals: &mut usize,
) -> Option<f64>
where
    F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
{
    let mut alpha = 1.0;
    for _ in 0..MAX_BACKTRACK {
        let mut trial = x.clone();
        trial.scaled_add(alpha, direction);
        let (value, next) = fg(trial.view())?;
        *evals += 1;
        let finite = value.is_finite() && next.iter().all(|component| component.is_finite());
        if finite && value <= *energy + ARMIJO * alpha * slope {
            let s = &trial - &*x;
            let taken = max_atom_len(&s);
            let y = &next - &*gradient;
            remember(memory, s, y);
            *x = trial;
            *energy = value;
            *gradient = next;
            return Some(taken);
        }
        alpha *= 0.5;
    }
    None
}

fn remember(memory: &mut Vec<Pair>, s: Array1<f64>, y: Array1<f64>) {
    let sy = s.dot(&y);
    let yy = y.dot(&y);
    let ss = s.dot(&s);
    if !(sy > 1.0e-8 * ss.sqrt() * yy.sqrt()) {
        return;
    }
    memory.push(Pair {
        s,
        y,
        rho: 1.0 / sy,
    });
    if memory.len() > MEMORY {
        memory.remove(0);
    }
}

fn descent(direction: &Array1<f64>, gradient: ArrayView1<f64>) -> bool {
    let slope = direction.dot(&gradient);
    slope.is_finite() && slope < 0.0 && direction.iter().any(|component| component.abs() > 0.0)
}

fn preconditioned_direction(
    x: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
    z: &[i32],
    memory: &[Pair],
) -> Array1<f64> {
    let mut q = gradient.to_owned();
    let mut alpha = Vec::with_capacity(memory.len());
    for pair in memory.iter().rev() {
        let a = pair.rho * pair.s.dot(&q);
        alpha.push(a);
        q.scaled_add(-a, &pair.y);
    }
    let mut r = model_inverse(x, z, q.view());
    for (pair, a) in memory.iter().zip(alpha.iter().rev()) {
        let b = pair.rho * pair.y.dot(&r);
        r.scaled_add(*a - b, &pair.s);
    }
    r.mapv_inplace(|component| -component);
    r
}

fn model_inverse(x: ArrayView1<f64>, z: &[i32], q: ArrayView1<f64>) -> Array1<f64> {
    let ghosts: &[Ghost] = &[];
    if z.len().saturating_mul(3) != x.len() || z.iter().any(|number| *number <= 0) {
        return q.to_owned();
    }
    let Ok(model) = Lindh::assemble(x, z, ghosts, &LindhParams::default()) else {
        return q.to_owned();
    };
    model.solve(q).unwrap_or_else(|| q.to_owned())
}

fn infinity_norm(gradient: ArrayView1<f64>) -> f64 {
    gradient
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()))
}

fn max_atom_len(step: &Array1<f64>) -> f64 {
    let atoms = step.len() / 3;
    let mut longest = 0.0_f64;
    for atom in 0..atoms {
        let mut square = 0.0;
        for axis in 0..3 {
            let component = step[3 * atom + axis];
            square += component * component;
        }
        longest = longest.max(square.sqrt());
    }
    longest
}

fn cap_atoms(step: &mut Array1<f64>, cap: f64) {
    let longest = max_atom_len(step);
    if longest > cap && longest.is_finite() {
        *step *= cap / longest;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use rgmin::Lbfgs;

    /// A stiff bond. Cold L-BFGS opens with a unit-length steepest step and
    /// spends the line search walking it back. The model step is capped
    /// first and converges in fewer evaluations.
    #[test]
    fn a_stiff_bond_converges_without_a_long_line_search() {
        let gtol = 1.0e-2;
        let start = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.6, 0.1, 0.0]);
        let z = [1, 1];
        let (energy, point, evals) = anc_minimize(start.view(), &z, 40, gtol, |x| Some(stiff(x)));
        assert!(
            infinity_norm(stiff(point.view()).1.view()) < gtol,
            "gradient stayed large, energy {energy} after {evals}"
        );
        assert!(evals < 40, "the model step spent {evals} evaluations");

        let mut solver = Lbfgs::default();
        solver.gtol = gtol;
        solver.max_line_evals = 64;
        let (_, _, lbfgs_evals) = solver.minimize(start.view(), 40, |x| Some(stiff(x)));
        assert!(
            evals < lbfgs_evals,
            "model step used {evals}, cold L-BFGS used {lbfgs_evals}"
        );
    }

    fn stiff(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let mut bond = [0.0; 3];
        for axis in 0..3 {
            bond[axis] = x[3 + axis] - x[axis];
        }
        let length = (bond[0] * bond[0] + bond[1] * bond[1] + bond[2] * bond[2]).sqrt();
        let stretch = length - 1.0;
        let energy = 200.0 * stretch * stretch + 0.05 * x[3] * x[3];
        let mut gradient = Array1::zeros(6);
        if length > 1e-8 {
            let slope = 400.0 * stretch / length;
            for axis in 0..3 {
                gradient[axis] = -slope * bond[axis];
                gradient[3 + axis] = slope * bond[axis];
            }
        }
        gradient[3] += 0.1 * x[3];
        (energy, gradient)
    }
}
