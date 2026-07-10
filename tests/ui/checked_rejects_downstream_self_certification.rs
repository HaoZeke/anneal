use anneal_core::accept::AcceptRule;
use anneal_core::variant::CertifiedAcceptance;

struct AlwaysReject;

impl AcceptRule<f64> for AlwaysReject {
    fn accept_prob(&self, _delta_e: f64, _temp: f64) -> f64 {
        0.0
    }
}

impl CertifiedAcceptance<f64> for AlwaysReject {}

fn main() {}
