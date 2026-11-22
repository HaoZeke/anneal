//! rgpot's xtb engine reports force and the objective boundary negates it.
//!
//! The C API entry point is named `rgpot_xtb_force`, and direct dimer
//! finite differences confirm that its buffer is minus the energy slope.

/// Objective component after rgpot negates an xtb force.
pub fn objective_component(xtb_force: f64) -> f64 {
    -xtb_force
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_negative_force_becomes_a_positive_energy_slope() {
        let force = -4.895_135_30;
        let objective = objective_component(force);
        assert!(
            (objective + force).abs() < 1e-12,
            "objective {objective} does not negate force {force}"
        );
    }

    #[test]
    fn the_gfn2_callback_publishes_the_engine_force_unchanged() {
        let source = include_str!("../examples/common/rgpot_eindir.rs");
        assert!(
            !source.contains("xtb_gradient::store_as_force"),
            "the callback negates the engine force before rgpot negates it"
        );
    }
}
