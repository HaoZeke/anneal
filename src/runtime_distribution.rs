//! Run-time distributions for a search that finds the answer eventually.
//!
//! A global optimiser that reaches the minimum given enough time is a
//! Las Vegas algorithm: always right, randomly slow. What characterises
//! one is the distribution of its time to a target, not the quality it
//! reports at a budget somebody picked. The two answer different
//! questions and disagree exactly where a funnel-exchange move earns
//! its place, because escaping a funnel raises the spread and a mean at
//! fixed budget rewards whichever arm is reliably mediocre.
//!
//! Three things follow, and each is a number this module computes
//! rather than an opinion:
//!
//! - How many runs a comparison needs before it can resolve anything.
//!   A success rate near a quarter needs runs in the dozens to place to
//!   a few points, and comparisons at eight runs resolve nothing.
//! - Whether running replicas in parallel buys time proportionally.
//!   For a time-to-target that is exponential the answer is yes; for a
//!   shifted exponential the offset is dead weight every replica pays
//!   and the speedup has a ceiling, which is [`parallel_speedup`] and
//!   its limit [`speedup_ceiling`].
//! - Whether restarting beats continuing, which is a question about the
//!   tail and not about the mean.
//!
//! Nothing here runs a search. It takes the times a search reported,
//! including the runs that never arrived, and says what they support.

/// One run's outcome: evaluations spent reaching the target, or `None`
/// when the run ended without reaching it.
///
/// The censored runs are the ones a mean would quietly drop, and they
/// are the informative half when the target is rare.
pub type RunTime = Option<u64>;

/// Fraction of runs that reached the target.
pub fn success_rate(runs: &[RunTime]) -> f64 {
    if runs.is_empty() {
        return 0.0;
    }
    runs.iter().filter(|run| run.is_some()).count() as f64 / runs.len() as f64
}

/// Runs needed to place a success rate within `half_width`, at about
/// ninety-five per cent confidence.
///
/// The usual normal interval: \(n = z^2 p(1-p)/\delta^2\). A rate near
/// a half is the expensive one to measure, so an unknown rate should be
/// budgeted at `0.5`. This is the function that says whether a
/// comparison was ever going to conclude anything.
pub fn runs_for_resolution(rate: f64, half_width: f64) -> usize {
    if !(0.0..=1.0).contains(&rate) || !half_width.is_finite() || half_width <= 0.0 {
        return 0;
    }
    let z = 1.96_f64;
    (z * z * rate * (1.0 - rate) / (half_width * half_width)).ceil() as usize
}

/// Maximum shortfall in successes that `runs` cannot resolve.
///
/// Two rates measured over the same number of runs differ by a standard
/// error of \(\sqrt{2\hat p(1-\hat p)/n}\) at the pooled rate. Two of
/// those, in runs, is the bound below which a difference is a sample.
pub fn resolvable_shortfall(hits_a: usize, hits_b: usize, runs: usize) -> f64 {
    if runs == 0 {
        return 0.0;
    }
    let n = runs as f64;
    let pooled = (hits_a + hits_b) as f64 / (2.0 * n);
    2.0 * (2.0 * pooled * (1.0 - pooled) / n).sqrt() * n
}

/// Shifted-exponential fit to the runs that reached the target.
///
/// Returns the offset and the rate beyond it.
///
/// The offset is the awkward one and a caller has to know why. Its
/// maximum-likelihood estimate is the smallest observed time, an
/// extreme-order statistic: it can only fall as runs are added, so the
/// estimate drifts down and every quantity drawn from it, the speedup
/// ceiling above all, drifts up. Measured on LJ38 the offset read 4629
/// over 96 arrivals and 2815 over 384, taking the ceiling from 2.84 to
/// 4.89 on the same search, and the two arms swapped which of them had
/// the lower one.
///
/// This returns the bias-corrected estimate,
/// \(t_{(1)} - (\bar t - t_{(1)})/(n-1)\), which removes the leading
/// bias but not the drift. An offset read off tens of arrivals is not a
/// converged number and must not be quoted as one;
/// [`offset_is_resolved`] is the guard.
///
/// Censored runs are excluded, which biases the fit optimistic when
/// most runs never arrive; [`success_rate`] says whether to trust it at
/// all.
pub fn shifted_exponential_fit(runs: &[RunTime]) -> Option<(f64, f64)> {
    let mut reached: Vec<f64> = runs
        .iter()
        .filter_map(|run| run.map(|t| t as f64))
        .collect();
    if reached.len() < 2 {
        return None;
    }
    reached.sort_by(f64::total_cmp);
    let n = reached.len() as f64;
    let smallest = reached[0];
    let mean = reached.iter().sum::<f64>() / n;
    let raw_excess = mean - smallest;
    if !(raw_excess > 0.0) {
        return None;
    }
    let offset = (smallest - raw_excess / (n - 1.0)).max(0.0);
    let excess = mean - offset;
    if !(excess > 0.0) {
        return None;
    }
    Some((offset, 1.0 / excess))
}

/// Whether an offset estimate is worth quoting.
///
/// The offset's standard error goes as \(1/(n\lambda)\) in the
/// arrivals, so it is placed to within a tenth of itself only once the
/// arrivals are numerous relative to the excess-to-offset ratio. Below
/// that the estimate is still moving and a ceiling drawn from it is a
/// property of the sample rather than of the search.
pub fn offset_is_resolved(runs: &[RunTime]) -> bool {
    let arrivals = runs.iter().filter(|run| run.is_some()).count();
    let Some((offset, rate)) = shifted_exponential_fit(runs) else {
        return false;
    };
    if offset <= 0.0 {
        return arrivals >= 30;
    }
    let standard_error = 1.0 / (arrivals as f64 * rate);
    standard_error < 0.1 * offset
}

/// Expected time to target when `processors` independent runs race.
///
/// For a shifted exponential the minimum of `n` draws is again shifted
/// exponential with the same offset and `n` times the rate, so the
/// expectation is \(\mu + 1/(n\lambda)\). Everything past the offset
/// divides; the offset does not.
pub fn parallel_speedup(offset: f64, rate: f64, processors: usize) -> f64 {
    if processors == 0 || !offset.is_finite() || !rate.is_finite() || rate <= 0.0 || offset < 0.0 {
        return 1.0;
    }
    let one = offset + 1.0 / rate;
    let many = offset + 1.0 / (processors as f64 * rate);
    if many <= 0.0 { 1.0 } else { one / many }
}

/// Speedup no number of processors can exceed.
///
/// \((\mu + 1/\lambda)/\mu\), the limit of [`parallel_speedup`]. An
/// offset of zero is the exponential case and returns infinity, which
/// is the honest answer: there is no ceiling, speedup is linear in the
/// processors. A large offset relative to the mean is the warning that
/// cores are being spent on a queue every one of them has to stand in.
pub fn speedup_ceiling(offset: f64, rate: f64) -> f64 {
    if !offset.is_finite() || !rate.is_finite() || rate <= 0.0 {
        return 1.0;
    }
    if offset <= 0.0 {
        return f64::INFINITY;
    }
    (offset + 1.0 / rate) / offset
}

/// Empirical quantile of the runs that reached the target.
pub fn reached_quantile(runs: &[RunTime], quantile: f64) -> Option<u64> {
    if !(0.0..=1.0).contains(&quantile) {
        return None;
    }
    let mut reached: Vec<u64> = runs.iter().filter_map(|run| *run).collect();
    if reached.is_empty() {
        return None;
    }
    reached.sort_unstable();
    let index = ((reached.len() - 1) as f64 * quantile).round() as usize;
    reached.get(index).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_censored_run_counts_against_the_rate() {
        let runs = vec![Some(10), None, Some(30), None];
        assert!((success_rate(&runs) - 0.5).abs() < 1e-12);
        assert_eq!(success_rate(&[]), 0.0);
    }

    #[test]
    fn eight_runs_cannot_place_a_rate_near_a_quarter() {
        // The comparison that started this: a quarter-ish success rate
        // placed to five points needs runs in the hundreds, and to ten
        // points still needs dozens. Eight was never going to do it.
        assert!(runs_for_resolution(0.25, 0.05) > 200);
        assert!(runs_for_resolution(0.25, 0.10) > 50);
        // An unknown rate is budgeted at the worst case.
        assert!(runs_for_resolution(0.5, 0.05) >= runs_for_resolution(0.25, 0.05));
        assert_eq!(runs_for_resolution(0.25, 0.0), 0);
    }

    #[test]
    fn a_three_run_gap_in_thirty_two_is_a_sample() {
        // Thirteen against sixteen, which read as a regression.
        let slack = resolvable_shortfall(13, 16, 32);
        assert!(slack > 3.0, "shortfall {slack} should cover three runs");
    }

    #[test]
    fn the_fit_corrects_the_offset_downward() {
        let runs: Vec<RunTime> = vec![Some(100), Some(120), Some(180)];
        let (offset, _) = shifted_exponential_fit(&runs).expect("three runs fit");
        // The smallest observation is 100 and it is biased high as an
        // offset, because no draw can fall below the true one. The
        // correction subtracts (mean - min)/(n - 1) = 33.33/2.
        assert!(offset < 100.0, "offset {offset} was not corrected");
        assert!(
            (offset - (100.0 - 100.0 / 6.0)).abs() < 1e-6,
            "offset {offset}"
        );
    }

    #[test]
    fn an_offset_read_off_few_arrivals_is_not_resolved() {
        // The LJ38 shape: an offset comparable to the excess, read off
        // tens of arrivals. This is the case that moved from 4629 to
        // 2815 between 96 and 384 runs, taking the ceiling from 2.84 to
        // 4.89, so the guard has to refuse it.
        let few: Vec<RunTime> = (0..10).map(|i| Some(3000 + i * 900)).collect();
        assert!(
            !offset_is_resolved(&few),
            "ten arrivals should not resolve an offset of this size"
        );
        // Same shape, far more arrivals.
        let many: Vec<RunTime> = (0..4000).map(|i| Some(3000 + (i % 90) * 100)).collect();
        assert!(offset_is_resolved(&many));
    }

    #[test]
    fn a_fit_needs_runs_that_arrived() {
        assert!(shifted_exponential_fit(&[None, None]).is_none());
        assert!(shifted_exponential_fit(&[Some(5)]).is_none());
        // Every run identical: no spread beyond the offset to fit.
        assert!(shifted_exponential_fit(&[Some(7), Some(7)]).is_none());
    }

    #[test]
    fn an_exponential_search_divides_by_its_processors() {
        // No offset: the whole distribution divides, so forty-eight
        // replicas are worth forty-eight times one.
        let speedup = parallel_speedup(0.0, 0.01, 48);
        assert!((speedup - 48.0).abs() < 1e-9, "speedup {speedup}");
        assert_eq!(speedup_ceiling(0.0, 0.01), f64::INFINITY);
    }

    #[test]
    fn an_offset_is_a_queue_every_replica_stands_in() {
        // Offset 900 with a mean excess of 100: one run averages 1000,
        // and no number of replicas beats 900. Forty-eight get nowhere
        // near forty-eight times.
        let (offset, rate) = (900.0, 1.0 / 100.0);
        let speedup = parallel_speedup(offset, rate, 48);
        assert!(speedup < 1.2, "speedup {speedup} should be nearly nothing");
        let ceiling = speedup_ceiling(offset, rate);
        assert!((ceiling - 1000.0 / 900.0).abs() < 1e-9, "ceiling {ceiling}");
        // And the ceiling binds: more processors do not approach it.
        assert!(parallel_speedup(offset, rate, 1_000_000) <= ceiling);
    }

    #[test]
    fn a_quantile_reads_the_runs_that_arrived() {
        let runs = vec![Some(30), None, Some(10), Some(20)];
        assert_eq!(reached_quantile(&runs, 0.0), Some(10));
        assert_eq!(reached_quantile(&runs, 1.0), Some(30));
        assert_eq!(reached_quantile(&[None], 0.5), None);
    }
}
