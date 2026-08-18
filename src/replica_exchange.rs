//! Temperature spread across a cooperative ensemble, and the exchange
//! that makes it worth having.
//!
//! An ensemble at one temperature is one search repeated. The funnel
//! problem it fails on is the classical one: a chain equilibrated in a
//! narrow funnel does not cross to a wide one on any reasonable
//! timescale, and LJ38 is the benchmark case where the global minimum
//! sits in the narrow funnel while the entropically favoured region is
//! elsewhere. Spreading replicas in temperature and exchanging between
//! them is the reference answer to exactly that, older than any of the
//! descriptor machinery above it.
//!
//! Nothing here owns a socket or a chain. The ladder is arithmetic and
//! the exchange is a predicate, so a coordinator that brokers swaps and
//! a test that replays them see the same rule.

/// Geometric temperature ladder for `replicas` chains.
///
/// Geometric rather than linear because the acceptance of an exchange
/// depends on the gap in inverse temperature, so equal ratios give
/// comparable acceptance along the ladder instead of crowding the cold
/// end. A single replica keeps `base`, and a non-finite or non-positive
/// bound yields an empty ladder rather than a silent default.
pub fn temperature_ladder(replicas: usize, base: f64, top: f64) -> Vec<f64> {
    if replicas == 0 || !base.is_finite() || !top.is_finite() || base <= 0.0 || top < base {
        return Vec::new();
    }
    if replicas == 1 {
        return vec![base];
    }
    let ratio = (top / base).powf(1.0 / (replicas - 1) as f64);
    (0..replicas)
        .map(|rung| base * ratio.powi(rung as i32))
        .collect()
}

/// Temperature this replica walks at, or `base` when it has no rung.
pub fn replica_temperature(replica: u32, replicas: usize, base: f64, top: f64) -> f64 {
    temperature_ladder(replicas, base, top)
        .get(replica as usize)
        .copied()
        .unwrap_or(base)
}

/// Metropolis acceptance for swapping two rungs of the ladder.
///
/// The exchange is accepted with probability
/// \(\min[1, e^{(\beta_a - \beta_b)(E_a - E_b)}]\). The sign is what
/// carries the meaning: when the hotter chain is holding the lower
/// energy the exponent is positive and the swap is certain, which is
/// how a discovery made at high temperature is handed down to a chain
/// cold enough to refine it. When the cold chain already holds the
/// better structure the swap is possible but unlikely, so the ladder
/// does not casually throw away what it has.
///
/// `draw` is the uniform variate, supplied so the decision is a pure
/// function of its inputs and a replay reproduces it.
pub fn replica_exchange_accepts(
    energy_a: f64,
    temperature_a: f64,
    energy_b: f64,
    temperature_b: f64,
    draw: f64,
) -> bool {
    if !energy_a.is_finite() || !energy_b.is_finite() {
        return false;
    }
    if !temperature_a.is_finite() || !temperature_b.is_finite() {
        return false;
    }
    if temperature_a <= 0.0 || temperature_b <= 0.0 {
        return false;
    }
    let exponent = (1.0 / temperature_a - 1.0 / temperature_b) * (energy_a - energy_b);
    if exponent >= 0.0 {
        return true;
    }
    draw < exponent.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ladder_spans_its_bounds_geometrically() {
        let ladder = temperature_ladder(5, 0.1, 1.6);
        assert_eq!(ladder.len(), 5);
        assert!((ladder[0] - 0.1).abs() < 1e-12);
        assert!((ladder[4] - 1.6).abs() < 1e-12);
        // Equal ratios, which is what keeps acceptance comparable along
        // the ladder rather than crowded at the cold end.
        for rung in 1..ladder.len() {
            let ratio = ladder[rung] / ladder[rung - 1];
            assert!((ratio - 2.0).abs() < 1e-12, "rung {rung} ratio {ratio}");
        }
    }

    #[test]
    fn a_degenerate_ladder_is_empty_rather_than_guessed() {
        assert!(temperature_ladder(0, 0.1, 1.0).is_empty());
        assert!(temperature_ladder(4, 0.0, 1.0).is_empty());
        assert!(temperature_ladder(4, 1.0, 0.5).is_empty());
        assert!(temperature_ladder(4, f64::NAN, 1.0).is_empty());
        assert_eq!(temperature_ladder(1, 0.3, 2.0), vec![0.3]);
    }

    #[test]
    fn a_replica_without_a_rung_walks_at_the_base() {
        assert!((replica_temperature(0, 3, 0.2, 0.8) - 0.2).abs() < 1e-12);
        assert!((replica_temperature(9, 3, 0.2, 0.8) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn a_discovery_made_hot_is_always_handed_down() {
        // Hot chain (b) holds the lower energy: the cold chain should
        // take it every time, whatever the draw.
        assert!(replica_exchange_accepts(-10.0, 0.1, -12.0, 1.0, 0.999));
        assert!(replica_exchange_accepts(-10.0, 0.1, -12.0, 1.0, 0.0));
    }

    #[test]
    fn the_cold_chain_does_not_casually_give_up_what_it_has() {
        // Cold chain (a) holds the lower energy: swapping is possible
        // but must not be certain, or the ladder discards its best.
        let exponent = (1.0 / 0.1 - 1.0 / 1.0) * (-12.0 - -10.0);
        let probability = exponent.exp();
        assert!(probability < 1.0);
        assert!(!replica_exchange_accepts(-12.0, 0.1, -10.0, 1.0, 0.5));
        assert!(replica_exchange_accepts(
            -12.0,
            0.1,
            -10.0,
            1.0,
            probability * 0.5
        ));
    }

    #[test]
    fn an_unusable_rung_refuses_the_swap() {
        assert!(!replica_exchange_accepts(f64::NAN, 0.1, -1.0, 1.0, 0.0));
        assert!(!replica_exchange_accepts(-1.0, 0.0, -2.0, 1.0, 0.0));
        assert!(!replica_exchange_accepts(
            -1.0,
            0.1,
            -2.0,
            f64::INFINITY,
            0.0
        ));
    }
}
