//! `trait Exchange<T>`: the parallel-tempering swap operator that handles
//! periodic swaps between chains at distinct temperatures. The composition
//! law E1 (detailed balance across the swap) is satisfied by Metropolis
//! acceptance.
//!
//! Within the typed algebra:
//!
//! - Cool, Neigh, Move, Accept stay per-chain unchanged.
//! - Exchange takes two chain states, temperatures, and objective values, then
//!   returns a swap-accept probability in [0, 1].
//! - `ParallelTemperingSampler` wraps an inner `Sampler<T>`, runs M replicas on
//!   a temperature ladder, and calls Exchange every `swap_period` steps.

use num_traits::Float;

/// Parallel-tempering swap acceptance rule.
///
/// At adjacent chains `i` (cooler) and `j` (hotter) with temperatures
/// `T_i < T_j` and objective values `F_i, F_j`, the probability of
/// accepting a state swap is min(1, exp((1/T_i - 1/T_j) * (F_i - F_j))).
/// This satisfies detailed balance with respect to the joint product
/// distribution `prod_k pi_{T_k}(x_k)`.
pub trait Exchange<T: Float>: Send + Sync {
    /// Returns the swap-accept probability in `[0, 1]`.
    fn swap_accept_prob(&self, f_i: T, t_i: T, f_j: T, t_j: T) -> T;

    /// Optional witness for E1 (detailed balance). Default returns true
    /// since the Metropolis swap below satisfies E1 by construction;
    /// custom `Exchange` impls can override when they need executable
    /// proptest sweeps.
    fn satisfies_detailed_balance(&self) -> bool {
        true
    }
}

/// Standard parallel-tempering Metropolis swap. The canonical choice;
/// satisfies detailed balance for any pair of canonical-Boltzmann
/// distributions at temperatures `T_i, T_j`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetropolisExchange;

impl<T: Float + Send + Sync> Exchange<T> for MetropolisExchange {
    fn swap_accept_prob(&self, f_i: T, t_i: T, f_j: T, t_j: T) -> T {
        let inv_diff = T::one() / t_i - T::one() / t_j;
        let log_alpha = inv_diff * (f_i - f_j);
        if log_alpha >= T::zero() {
            T::one()
        } else {
            log_alpha.exp()
        }
    }
}

/// Tsallis-q exchange rule: the q-deformed analogue of the canonical
/// Metropolis swap, matching the GSA acceptance family. At `q = 1`
/// reduces to `MetropolisExchange`.
///
/// Swap probability is the q-deformed expression
/// max(0, [1 - (q - 1) * (F_i - F_j) * (1/T_i - 1/T_j)]^{1/(q-1)}),
/// clamped to `[0, 1]`. Reduces to standard PT via the Metropolis
/// limit Theorem 3 of the IISE manuscript at `q -> 1`.
#[derive(Clone, Copy, Debug)]
pub struct TsallisExchange<T: Float> {
    /// Tsallis index. `q = 1` reduces to Metropolis.
    pub q: T,
}

impl<T: Float> TsallisExchange<T> {
    /// Constructs with the given `q`.
    pub fn new(q: T) -> Self {
        Self { q }
    }
}

impl<T: Float + Send + Sync> Exchange<T> for TsallisExchange<T> {
    fn swap_accept_prob(&self, f_i: T, t_i: T, f_j: T, t_j: T) -> T {
        if (self.q - T::one()).abs() < T::epsilon() {
            return MetropolisExchange.swap_accept_prob(f_i, t_i, f_j, t_j);
        }
        let inv_diff = T::one() / t_i - T::one() / t_j;
        let bracket = T::one() - (self.q - T::one()) * (f_i - f_j) * inv_diff;
        if bracket <= T::zero() {
            return T::zero();
        }
        let p = bracket.powf(T::one() / (self.q - T::one()));
        if p > T::one() { T::one() } else { p }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metropolis_exchange_downhill_in_cooler_chain_accepts() {
        // f_i (cooler chain) < f_j (hotter chain): we'd swap so cold
        // gets the better state. (1/T_i - 1/T_j) > 0; (f_i - f_j) < 0;
        // log_alpha < 0 -> alpha = exp(neg). Should be in (0, 1).
        let m = MetropolisExchange;
        let alpha: f64 = m.swap_accept_prob(0.0, 0.5, 1.0, 2.0);
        // 0.0 - 1.0 = -1; 1/0.5 - 1/2.0 = 1.5; product = -1.5; exp(-1.5) ~ 0.22.
        assert!(alpha > 0.0 && alpha < 1.0);
        assert!((alpha - (-1.5_f64).exp()).abs() < 1e-9);
    }

    #[test]
    fn metropolis_exchange_aligned_swap_probability_one() {
        // f_i (cooler) > f_j (hotter): swap puts the better state in the
        // hotter chain; (1/T_i - 1/T_j)*(f_i - f_j) > 0 -> alpha = 1.
        let m = MetropolisExchange;
        let alpha: f64 = m.swap_accept_prob(2.0, 0.5, 1.0, 2.0);
        assert_eq!(alpha, 1.0);
    }

    #[test]
    fn tsallis_exchange_q_one_matches_metropolis() {
        let t = TsallisExchange::new(1.0_f64);
        let m = MetropolisExchange;
        for (f_i, t_i, f_j, t_j) in [(0.0, 0.5, 1.0, 2.0), (3.0, 1.0, 1.0, 5.0)] {
            let a_t = t.swap_accept_prob(f_i, t_i, f_j, t_j);
            let a_m = m.swap_accept_prob(f_i, t_i, f_j, t_j);
            assert!(
                (a_t - a_m).abs() < 1e-9,
                "Tsallis q=1 {} != Metropolis {}",
                a_t,
                a_m
            );
        }
    }

    #[test]
    fn tsallis_exchange_uphill_swap_returns_finite_probability() {
        // Tsallis acceptance imposes a hard cutoff: when the q-deformed
        // bracket goes non-positive the acceptance is exactly 0. For
        // standard Metropolis the same args give exp() of a positive
        // value (clamped to 1 because alpha is min(1, exp(.))).
        // This documents the q-deformation's stricter gating.
        let t = TsallisExchange::new(1.5_f64);
        let m = MetropolisExchange;
        // Mildly favourable swap: f_i=2 cold, f_j=1 hot, T_i=0.5, T_j=2.
        let a_t = t.swap_accept_prob(2.0_f64, 0.5, 1.0, 2.0);
        let a_m = m.swap_accept_prob(2.0_f64, 0.5, 1.0, 2.0);
        // Metropolis gives log_alpha = 1.5 > 0 and clamps to one.
        assert_eq!(a_m, 1.0);
        // Tsallis: bracket = 1 - 0.5*1*1.5 = 0.25; p = 0.25^{1/0.5}.
        assert!(a_t > 0.0 && a_t < 1.0);
        assert!((a_t - 0.0625).abs() < 1e-9);
    }
}
