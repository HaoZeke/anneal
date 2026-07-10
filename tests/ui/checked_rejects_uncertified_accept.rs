use anneal_core::accept::AcceptRule;
use anneal_core::cool::LogCool;
use anneal_core::movekernel::Gaussian;
use anneal_core::neigh::ContinuousR_n;
use anneal_core::variant::SaVariant;
use eindir_core::objectives::StybTang2D;

struct AlwaysReject;

impl AcceptRule<f64> for AlwaysReject {
    fn accept_prob(&self, _delta_e: f64, _temp: f64) -> f64 {
        0.0
    }
}

fn main() {
    let _ = SaVariant::checked(
        StybTang2D::new(),
        LogCool::new(1.0_f64, 2.0),
        ContinuousR_n::new(2),
        Gaussian::new(0.5),
        AlwaysReject,
    );
}
