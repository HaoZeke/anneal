//! The pinned xtb engine returns a force, and rgpot negates it.
//!
//! `rgpot_xtb_force` writes xtb's ∂E/∂x multiplied by `NEG_GRAD_TO_FORCE`,
//! which is a force in eV/Å. rgpot builds the objective gradient by negating
//! that buffer. Negating it in the callback as well makes the objective
//! gradient minus the true slope.

/// Objective component after rgpot negates a published engine force.
pub fn objective_from_engine_force(force: f64) -> f64 {
    -force
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repulsive_force_is_a_positive_energy_slope() {
        // A positive force on the outgoing atom is a negative energy slope.
        let force = -4.895_135_30;
        let objective = objective_from_engine_force(force);
        assert!((objective - 4.895_135_30).abs() < 1e-12);
    }

    #[test]
    fn the_gfn2_callback_publishes_the_engine_force_unchanged() {
        let source = include_str!("../examples/common/rgpot_eindir.rs");
        assert!(
            !source.contains("store_as_force"),
            "the callback negates an engine buffer that is already a force"
        );
    }
}
