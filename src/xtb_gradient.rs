//! xtb reports ∂E/∂x. The rgpot callback buffer is a force.
//!
//! rgpot builds the objective gradient by negating that buffer. Publishing
//! the xtb gradient unchanged makes the objective gradient minus the true
//! slope, and a descent step walks uphill.

/// Replace an xtb gradient, in place, with the force rgpot expects.
pub fn store_as_force(gradient: &mut [f64]) {
    for component in gradient {
        *component = -*component;
    }
}

/// Objective component after rgpot negates a published force.
///
/// This is the composition the GFN2 callback relies on: the engine buffer
/// is ∂E/∂x, [`store_as_force`] flips it, and rgpot flips it back.
pub fn objective_component(xtb_gradient: f64) -> f64 {
    let mut force = [xtb_gradient];
    store_as_force(&mut force);
    -force[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_positive_energy_slope_stays_positive_after_rgpot_negates_the_force() {
        let slope = 4.895_135_30;
        let objective = objective_component(slope);
        assert!(
            (objective - slope).abs() < 1e-12,
            "objective {objective} does not match the energy slope {slope}"
        );
    }

    #[test]
    fn the_gfn2_callback_stores_the_xtb_gradient_as_a_force() {
        let source = include_str!("../examples/common/rgpot_eindir.rs");
        assert!(
            source.contains("xtb_gradient::store_as_force"),
            "the callback publishes the xtb gradient without converting it to a force"
        );
    }
}
