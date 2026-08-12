//! Swarm roles on a shared packing catalog.
//!
//! Whale optimization (Mirjalili & Lewis 2016) and swallow swarm
//! (Neshat et al. 2013) do not teleport every agent onto a random
//! neighbour every step. They keep a live catalog and split the
//! population:
//!
//! * **Leader** (swallow) / encircling prey (whale): the deepest
//!   class walks locally and publishes. It does not pull.
//! * **Explorer**: mid-rank. Early in the budget (`|A| ≥ 1`) it
//!   searches prey — another competitive packing, not the incumbent.
//!   Late (`|A| < 1`) it encircles: only a win.
//! * **Aimless**: the catalog of the occupied class is saturated.
//!   The hop is a SOAP hole, not a redraw from the bank.
//!
//! The leftover walk is still the hop. This module only decides
//! whether to talk to the catalog.

/// Swallow role against the shared catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Deepest class. Local leftover only. Publish wins.
    Leader,
    /// Mid pack. May pull a competitive other funnel.
    Explorer,
    /// Raised / saturated class. Leave the archive.
    Aimless,
}

/// Whale phase from remaining budget.
///
/// `a` falls from 2 to 0. `|A| ≥ 1` while `a` is large (first half
/// of the budget): search prey. Then encircle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Do not encircle the incumbent.
    SearchPrey,
    /// Close on a win.
    Encircle,
}

/// What one chain does with the catalog this slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decision {
    /// Swallow role.
    pub role: Role,
    /// Whale phase.
    pub phase: Phase,
    /// Take one sample from the bank.
    pub pull: bool,
    /// That sample must be a win, not another class.
    pub win_only: bool,
    /// Step into a SOAP hole of the occupied packing.
    pub leave: bool,
}

/// WOA `a(t)`: 2 → 0 as `progress` goes 0 → 1.
pub fn whale_a(progress: f64) -> f64 {
    2.0 * (1.0 - progress.clamp(0.0, 1.0))
}

/// Search-prey while `a ≥ 1` (first half of the budget).
pub fn whale_phase(progress: f64) -> Phase {
    if whale_a(progress) >= 1.0 {
        Phase::SearchPrey
    } else {
        Phase::Encircle
    }
}

/// Swallow role from catalog rank and saturation.
pub fn swallow_role(
    my_energy: f64,
    bank_best: f64,
    bank_size: usize,
    catalog_saturated: bool,
    on_raised_packing: bool,
) -> Role {
    if catalog_saturated || on_raised_packing {
        return Role::Aimless;
    }
    if bank_size > 0 && my_energy.is_finite() && my_energy <= bank_best + 0.05 {
        return Role::Leader;
    }
    Role::Explorer
}

/// Combine whale phase and swallow role into one catalog action.
pub fn decide(
    progress: f64,
    my_energy: f64,
    bank_best: f64,
    bank_size: usize,
    catalog_saturated: bool,
    on_raised_packing: bool,
) -> Decision {
    let phase = whale_phase(progress);
    let role = swallow_role(
        my_energy,
        bank_best,
        bank_size,
        catalog_saturated,
        on_raised_packing,
    );
    match role {
        Role::Leader => Decision {
            role,
            phase,
            pull: false,
            win_only: true,
            leave: false,
        },
        Role::Aimless => Decision {
            role,
            phase,
            pull: false,
            win_only: true,
            leave: true,
        },
        Role::Explorer => Decision {
            role,
            phase,
            pull: true,
            win_only: phase == Phase::Encircle,
            leave: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn early_budget_is_search_prey() {
        assert_eq!(whale_phase(0.0), Phase::SearchPrey);
        assert_eq!(whale_phase(0.4), Phase::SearchPrey);
    }

    #[test]
    fn late_budget_is_encircle() {
        assert_eq!(whale_phase(0.6), Phase::Encircle);
        assert_eq!(whale_phase(1.0), Phase::Encircle);
    }

    #[test]
    fn a_leader_does_not_pull() {
        let d = decide(0.2, -173.93, -173.93, 8, false, false);
        assert_eq!(d.role, Role::Leader);
        assert!(!d.pull);
        assert!(!d.leave);
    }

    #[test]
    fn a_saturated_catalog_is_aimless() {
        let d = decide(0.3, -396.28, -396.28, 18, true, false);
        assert_eq!(d.role, Role::Aimless);
        assert!(d.leave);
        assert!(!d.pull);
    }

    #[test]
    fn an_explorer_searches_prey_then_encircles() {
        let early = decide(0.1, -390.0, -396.28, 5, false, false);
        assert_eq!(early.role, Role::Explorer);
        assert!(early.pull);
        assert!(!early.win_only);
        let late = decide(0.8, -390.0, -396.28, 5, false, false);
        assert_eq!(late.role, Role::Explorer);
        assert!(late.pull);
        assert!(late.win_only);
    }
}
