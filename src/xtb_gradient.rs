//! The xtb engine publishes a force, in eV/Å.
//!
//! xtb's own buffer is ∂E/∂x. `rgpot_xtb_force` turns that into a force
//! before it returns. rgpot then builds the objective gradient by negating
//! the published buffer. The callback therefore publishes the engine
//! buffer unchanged. A second negation makes the objective gradient the
//! uphill direction, and a descent step walks away from the minimum.

/// Objective component after rgpot negates a published engine force.
pub fn objective_component(engine_force: f64) -> f64 {
    -engine_force
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgpot_negates_the_engine_force_once() {
        let force = -4.895_135_30;
        let objective = objective_component(force);
        assert!(
            (objective + force).abs() < 1e-12,
            "objective {objective} is not the negation of the engine force {force}"
        );
    }

    #[test]
    fn the_gfn2_callback_publishes_the_engine_force_unchanged() {
        let source = include_str!("../examples/common/rgpot_eindir.rs");
        assert!(
            !source.contains("xtb_gradient::store_as_force"),
            "the callback negates an engine force that is already a force"
        );
        assert!(
            source.contains("rgpot_xtb_force"),
            "the callback no longer documents the engine force"
        );
    }
}
