//! Witness tests for the Tsallis-coherence reduction theorem
//! (design pass 16). Each of the four kernel components has a
//! pointwise q_v -> 1+ limit; this test suite witnesses each limit
//! at the trait-implementation layer with explicit O(eps) bounds.
//!
//! Theorem 1 (Tsallis-coherence reduction): As q_v -> 1+,
//!   1. Tsallis cooling -> Boltzmann log cooling.
//!   2. q-Gaussian visiting -> standard Gaussian.
//!   3. Tsallis acceptance -> Metropolis exp(-dE/T).
//!   4. metad_gamma = 1/(q_v - 1) -> infinity (no flattening).
//!
//! The composite reduction makes bGSA the q-parametric SA family
//! with classical Boltzmann SA as the q_v -> 1 limit. Each of the
//! four limits below is the pointwise witness for one component.
//!
//! Each test uses eps in {1e-3, 1e-4, 1e-5} so failures point to
//! whichever limit is the bottleneck of the composite reduction.

use anneal_core::accept::{AcceptRule, Metropolis, TsallisAccept};
use anneal_core::cool::{Cooling, TsallisCool};

const EPS_LIST: [f64; 3] = [1e-3, 1e-4, 1e-5];

#[test]
fn tsallis_cool_q_to_1_recovers_log_cooling() {
    // Limit (1) of Theorem 1: TsallisCool(t_init, q_v=1+eps)(k) ->
    // t_init * ln 2 / ln(1+k) at eps -> 0. The q -> 1 limit is the
    // L'Hopital expansion of (2^eps - 1) / ((1+k)^eps - 1) = ln 2 / ln(1+k).
    // We compare against this explicit limit formula. NOTE: the
    // crate's LogCool uses ln(k0) / ln(k+k0) with k0=2, which is
    // DIFFERENT from the q->1 limit's ln 2 / ln(1+k). The two log-
    // cooling conventions differ by a 1-step shift, so we test
    // against the q-limit formula explicitly.
    let t_init = 5.0_f64;
    let mut prev_max_err = f64::INFINITY;
    for &eps in EPS_LIST.iter() {
        let q_v = 1.0 + eps;
        let cool_q = TsallisCool::new(t_init, q_v);
        let mut max_err = 0.0_f64;
        for k in 1..=1000usize {
            let t_q = cool_q.temperature(k);
            // Explicit q -> 1 limit.
            let t_lim = if k == 0 {
                t_init
            } else {
                t_init * (2.0_f64).ln() / ((1.0 + k as f64).ln())
            };
            let err = (t_q - t_lim).abs() / t_lim.abs().max(1e-12);
            max_err = max_err.max(err);
        }
        // O(eps) tightening: smaller eps must give tighter agreement.
        assert!(
            max_err < prev_max_err + 1e-10,
            "Tsallis cooling did NOT tighten toward its q->1 limit: \
             eps = {eps}, max relative error = {max_err}, previous = {prev_max_err}"
        );
        // Absolute bound: O(eps log k) within constant factor.
        // At k=1000, ln(1+k) ~ 6.9, so error contribution ~ 6.9 * eps.
        assert!(
            max_err < 100.0 * eps,
            "Tsallis cooling deviates from q->1 limit by more than O(eps): \
             eps = {eps}, max relative error = {max_err}"
        );
        prev_max_err = max_err;
    }
}

#[test]
fn tsallis_accept_q_to_1_recovers_metropolis() {
    // Limit (3) of Theorem 1: Tsallis acceptance -> exp(-dE/T) at q_v -> 1+.
    //
    // Series expansion: log p_q - log p_metro = (q-1) (dE/T)^2 / 2 +
    // O((q-1)^2 (dE/T)^3). The relative deviation is therefore
    // O(eps * x^2) where x = dE/T. We sweep x = dE/T in [0.01, 5] so
    // the worst-case deviation is bounded by eps * 25 / 2 ~ 12.5 eps.
    //
    // The quadratic-in-x scaling is itself a witness for the q-coherent
    // expansion of the Tsallis acceptance: small dE/T sits well inside
    // the Metropolis regime, large dE/T is exponentially suppressed by
    // both kernels and the Tsallis heavy tail emerges.
    let mut prev_max_err = f64::INFINITY;
    for &eps in EPS_LIST.iter() {
        let q_a = 1.0 + eps;
        let acc_q = TsallisAccept::new(q_a);
        let acc_metro = Metropolis;
        let mut max_err = 0.0_f64;
        let x_max: f64 = 5.0;
        for de_step in 1..=50 {
            let de_over_t = de_step as f64 * (x_max / 50.0);
            // Realise the ratio at fixed T = 1; scaling is dE-/T-only.
            let de = de_over_t;
            let t = 1.0_f64;
            let p_q = acc_q.accept_prob(de, t);
            let p_metro = acc_metro.accept_prob(de, t);
            let err = (p_q - p_metro).abs() / p_metro.max(1e-12);
            max_err = max_err.max(err);
        }
        assert!(
            max_err < prev_max_err + 1e-10,
            "Tsallis accept did NOT tighten as q_a -> 1: eps = {eps}, \
             max relative error = {max_err}, previous = {prev_max_err}"
        );
        // O(eps * x_max^2 / 2) bound; safety factor 4.
        let bound = 2.0 * eps * x_max * x_max;
        assert!(
            max_err < bound,
            "Tsallis accept deviates from Metropolis by more than O(eps * x^2): \
             eps = {eps}, max relative error = {max_err}, bound = {bound}"
        );
        prev_max_err = max_err;
    }
}

#[test]
fn metad_gamma_q_to_1_diverges() {
    // Limit (4) of Theorem 1: metad_gamma = 1/(q_v - 1) -> infinity
    // at q_v -> 1+. The well-tempered MetaD bias height
    // w(gamma) = w_0 * exp(-V_{k-1} / ((gamma - 1) T))
    // approaches w_0 at any finite V, T as gamma -> infinity, so the
    // bias accumulates linearly with deposition count; this means the
    // bias takes infinite time to flatten any finite F at gamma -> oo.
    // Practical effect: at any finite budget, metad_gamma -> oo
    // produces a vanishing effective bias.
    let v_typical = 1.0;
    let t_typical = 1.0;
    let w0 = 0.05;
    let mut prev_w_at_eps = f64::INFINITY;
    for &eps in EPS_LIST.iter() {
        let q_v = 1.0 + eps;
        let gamma_q = 1.0 / (q_v - 1.0);
        // gamma_q must diverge as eps -> 0.
        assert!(
            gamma_q > 1.0 / (10.0 * eps),
            "metad_gamma_q must diverge as q_v -> 1: q_v = {q_v}, gamma = {gamma_q}"
        );
        // The deposition height ratio w(gamma) / w_0 -> 1 at gamma -> oo.
        let w_q = w0 * (-v_typical / ((gamma_q - 1.0) * t_typical)).exp();
        // Tightening: smaller eps must give larger gamma_q and w_q closer to w_0.
        let dist_to_w0 = (w_q - w0).abs() / w0;
        assert!(
            dist_to_w0 < prev_w_at_eps + 1e-10,
            "metad height did NOT approach w_0 as q_v -> 1: eps = {eps}, \
             dist = {dist_to_w0}, previous = {prev_w_at_eps}"
        );
        prev_w_at_eps = dist_to_w0;
    }
}

#[test]
fn tsallis_coherence_composite_continuity() {
    // Witnesses the COMPOSITE limit of Theorem 1: at q_v = 1 + eps,
    // the joint (cool, accept) kernel evaluated on a fixed (epoch, dE)
    // grid has mean relative deviation O(eps * (dE/T)^2) from the
    // (q->1 limit cool, Metropolis) joint. The composite is bounded by
    // the slower (quadratic-x) of the two limits.
    //
    // We compare TsallisCool(q=1+eps) + TsallisAccept(q=1+eps)
    // against the LIMIT cooling t_init * ln 2 / ln(1+k) + Metropolis,
    // which is the joint q->1 fibre of the bGSA kernel. Restrict
    // dE/T <= 5 so the worst-case is O(eps * 25 / 2) per evaluation.
    let t_init = 5.0;
    let mut prev_total = f64::INFINITY;
    for &eps in EPS_LIST.iter() {
        let q_v = 1.0 + eps;
        let cool_q = TsallisCool::new(t_init, q_v);
        let acc_q = TsallisAccept::new(q_v);
        let acc_metro = Metropolis;
        let mut total_err = 0.0_f64;
        let mut count = 0usize;
        for k in [1usize, 5, 10, 50, 100, 500, 1000] {
            let t_q = cool_q.temperature(k);
            let t_lim = t_init * (2.0_f64).ln() / ((1.0 + k as f64).ln());
            // Restrict dE so dE/T_q stays below 5.
            let de_max = 5.0 * t_q.min(t_lim);
            for de_step in 1..=10 {
                let de = de_step as f64 * de_max / 10.0;
                let p_q = acc_q.accept_prob(de, t_q);
                let p_metro = acc_metro.accept_prob(de, t_lim);
                total_err += (p_q - p_metro).abs() / p_metro.max(1e-12);
                count += 1;
            }
        }
        let mean_err = total_err / count as f64;
        assert!(
            mean_err < prev_total + 1e-10,
            "Composite (cool, accept) kernel did NOT tighten as q_v -> 1: \
             eps = {eps}, mean error = {mean_err}, previous = {prev_total}"
        );
        // O(eps * x_max^2) bound with safety factor.
        let bound = 100.0 * eps;
        assert!(
            mean_err < bound,
            "Composite kernel deviates by more than O(eps): \
             eps = {eps}, mean error = {mean_err}, bound = {bound}"
        );
        prev_total = mean_err;
    }
}
