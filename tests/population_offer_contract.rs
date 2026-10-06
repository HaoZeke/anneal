use std::fs;
use std::path::PathBuf;

/// The ensemble driver admits a child through the bank. A child is not
/// written into its own slot before that call.
#[test]
fn population_offer_goes_through_the_bank() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("lj_ensemble_splice.rs");
    let source = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let start = source
        .find("fn offer(&mut self, p: usize, energy: f64, state: &[f64], dcut_scale: f64)")
        .expect("Population::offer");
    let rest = &source[start..];
    let end = rest.find("\n    fn ").expect("the next method");
    let body = &rest[..end];
    assert!(
        body.contains("self.bank.offer("),
        "Population::offer must admit the child through Bank::offer"
    );
    assert!(
        !body.contains("self.members[p]"),
        "Population::offer must not write slot p before the bank decides"
    );
}
