//! Fit a first-passage mixture to one campaign's per-seed first-target
//! forces and predict the hit rate of independent chains at any split.
//!
//!     first_passage_fit <budget> [components=2] [k:aggregate ...] < values
//!
//! `values` has one entry per seed: the forces at first target, or `-`
//! for a seed that did not reach it within `budget` (right-censored).
//! Each `k:aggregate` asks for the predicted hit probability of `k`
//! chains sharing `aggregate` forces; with none given, the table covers
//! k in {1, 2, 4, 8, 16, 48} at the budget, four times it, and 0.6 of it.
use anneal_core::first_passage::{ExponentialMixture, FirstPassage};
use std::io::Read;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let budget: f64 = args
        .first()
        .and_then(|v| v.parse().ok())
        .expect("usage: first_passage_fit <budget> [components] [k:aggregate ...]");
    let components: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(2);
    let mut text = String::new();
    std::io::stdin()
        .read_to_string(&mut text)
        .expect("values on stdin");
    let observations: Vec<FirstPassage> = text
        .split_whitespace()
        .map(|token| match token.parse::<f64>() {
            Ok(v) if v > 0.0 => FirstPassage::Hit(v),
            _ => FirstPassage::Censored(budget),
        })
        .collect();
    let hits = observations
        .iter()
        .filter(|o| matches!(o, FirstPassage::Hit(_)))
        .count();
    let fit = ExponentialMixture::fit(&observations, components, 2000)
        .unwrap_or_else(|error| panic!("fit: {error}"));
    println!(
        "seeds {}  hits {}  budget {budget:.3e}  components {components}  iterations {}  log-likelihood {:.3}",
        observations.len(),
        hits,
        fit.iterations,
        fit.log_likelihood
    );
    for (w, m) in fit.weights.iter().zip(&fit.means) {
        println!("  component weight {w:.3}  mean forces {m:.3e}");
    }
    println!(
        "  one chain at the budget: predicted {:.3}  measured {:.3}",
        fit.hit_probability(budget),
        hits as f64 / observations.len() as f64
    );
    let requests: Vec<(usize, f64)> = args
        .iter()
        .skip(2)
        .filter_map(|token| {
            let (k, b) = token.split_once(':')?;
            Some((k.parse().ok()?, b.parse().ok()?))
        })
        .collect();
    let requests = if requests.is_empty() {
        [budget, 4.0 * budget, 0.6 * budget]
            .iter()
            .flat_map(|&b| [1usize, 2, 4, 8, 16, 48].into_iter().map(move |k| (k, b)))
            .collect()
    } else {
        requests
    };
    println!("  chains  aggregate   per-chain   p(hit)  expected of 48");
    for (k, aggregate) in requests {
        let p = fit.ensemble_hit_probability(k, aggregate);
        println!(
            "  {k:>6}  {aggregate:>9.3e}  {:>9.3e}  {p:.3}  {:.1}",
            aggregate / k as f64,
            48.0 * p
        );
    }
}
