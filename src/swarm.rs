//! How a leftover chain uses the shared packing catalog.
//!
//! The hop is leftover SOAP. The catalog is the set of packing wells
//! the chains have already published. This module only answers three
//! questions that catalog already implies:
//!
//! * This chain holds the deepest published packing: keep walking it.
//!   Do not redraw a member.
//! * This chain is on another packing and the ledger is still open:
//!   a competitive other class may be adopted. Late in the ledger,
//!   only a deeper member.
//! * The occupied packing is exhausted for this chain — Good-Turing
//!   missing mass is low, or this chain has not deepened for
//!   [`STALL_LEAVE`] slices: step into a SOAP hole. Do not redraw
//!   the catalog incumbent.
//!
//! A chain that just deepened its own quench does not adopt. That is
//! the same leftover walk, not a second method.

/// Where this chain sits relative to the published packings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Deepest published packing. Walk it. Publish if it deepens.
    Occupant,
    /// A different packing. May adopt a competitive class.
    Other,
    /// This packing is exhausted for the chain. Leave through a hole.
    Leave,
}

/// Open ledger: other packings. Closed ledger: only a deeper member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LedgerHalf {
    /// First half of the charged budget.
    Open,
    /// Second half.
    Closed,
}

/// Catalog action for one slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decision {
    /// Relation to the published packings.
    pub role: Role,
    /// Open or closed half of the ledger.
    pub half: LedgerHalf,
    /// Take one sample from the catalog.
    pub pull: bool,
    /// That sample must be deeper than this chain's quench.
    pub win_only: bool,
    /// Step into a SOAP hole of the occupied packing.
    pub leave: bool,
}

/// First half of the budget is open.
pub fn ledger_half(progress: f64) -> LedgerHalf {
    if progress.clamp(0.0, 1.0) < 0.5 {
        LedgerHalf::Open
    } else {
        LedgerHalf::Closed
    }
}

/// Role from catalog rank and whether the occupied packing is exhausted.
pub fn catalog_role(
    my_energy: f64,
    catalog_best: f64,
    catalog_size: usize,
    catalog_saturated: bool,
    on_raised_packing: bool,
) -> Role {
    if catalog_saturated || on_raised_packing {
        return Role::Leave;
    }
    if catalog_size > 0 && my_energy.is_finite() && my_energy <= catalog_best + 0.05 {
        return Role::Occupant;
    }
    Role::Other
}

/// Slices without a deeper own quench before the packing is treated
/// as exhausted for this chain.
pub const STALL_LEAVE: u32 = 8;

/// Catalog policy. `stall` is slices since this chain last deepened.
pub fn policy(
    progress: f64,
    my_energy: f64,
    own_best: f64,
    catalog_best: f64,
    catalog_size: usize,
    catalog_saturated: bool,
    on_raised_packing: bool,
    stall: u32,
) -> Decision {
    let half = ledger_half(progress);
    if stall >= STALL_LEAVE && catalog_size > 0 {
        return Decision {
            role: Role::Leave,
            half,
            pull: false,
            win_only: true,
            leave: true,
        };
    }
    let role = catalog_role(
        my_energy,
        catalog_best,
        catalog_size,
        catalog_saturated,
        on_raised_packing,
    );
    match role {
        Role::Occupant => Decision {
            role,
            half,
            pull: false,
            win_only: true,
            leave: false,
        },
        Role::Leave => Decision {
            role,
            half,
            pull: false,
            win_only: true,
            leave: true,
        },
        Role::Other => {
            let descending = stall == 0 && own_best.is_finite() && my_energy <= own_best + 1e-12;
            Decision {
                role,
                half,
                pull: !descending,
                win_only: half == LedgerHalf::Closed,
                leave: false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_half_of_the_ledger_is_open() {
        assert_eq!(ledger_half(0.0), LedgerHalf::Open);
        assert_eq!(ledger_half(0.4), LedgerHalf::Open);
    }

    #[test]
    fn the_second_half_of_the_ledger_is_closed() {
        assert_eq!(ledger_half(0.6), LedgerHalf::Closed);
        assert_eq!(ledger_half(1.0), LedgerHalf::Closed);
    }

    #[test]
    fn the_deepest_packing_is_walked_not_redrawn() {
        let d = policy(0.2, -173.93, -173.93, -173.93, 8, false, false, 1);
        assert_eq!(d.role, Role::Occupant);
        assert!(!d.pull);
        assert!(!d.leave);
    }

    #[test]
    fn a_saturated_packing_is_left() {
        let d = policy(0.3, -396.28, -396.28, -396.28, 18, true, false, 1);
        assert_eq!(d.role, Role::Leave);
        assert!(d.leave);
        assert!(!d.pull);
    }

    #[test]
    fn another_packing_adopts_a_class_then_only_a_win() {
        let early = policy(0.1, -390.0, -390.0, -396.28, 5, false, false, 1);
        assert_eq!(early.role, Role::Other);
        assert!(early.pull);
        assert!(!early.win_only);
        let late = policy(0.8, -390.0, -390.0, -396.28, 5, false, false, 1);
        assert_eq!(late.role, Role::Other);
        assert!(late.pull);
        assert!(late.win_only);
    }

    #[test]
    fn a_stalled_chain_leaves_without_redrawing() {
        let d = policy(0.2, -396.28, -396.28, -396.28, 10, false, false, 8);
        assert_eq!(d.role, Role::Leave);
        assert!(d.leave);
        assert!(!d.pull);
    }

    #[test]
    fn a_descending_chain_is_not_restamped() {
        let d = policy(0.2, -390.0, -390.0, -396.28, 5, false, false, 0);
        assert_eq!(d.role, Role::Other);
        assert!(!d.pull);
    }
}
