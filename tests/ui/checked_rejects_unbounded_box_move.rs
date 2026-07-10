use anneal_core::accept::Metropolis;
use anneal_core::cool::LogCool;
use anneal_core::movekernel::Gaussian;
use anneal_core::neigh::BoxConstrained;
use anneal_core::variant::SaVariant;
use eindir_core::objectives::StybTang2D;
use eindir_core::Objective;

fn main() {
    let objective = StybTang2D::new();
    let neighborhood = BoxConstrained::new(objective.bounds().clone());
    let _ = SaVariant::checked(
        objective,
        LogCool::new(1.0_f64, 2.0),
        neighborhood,
        Gaussian::new(0.5),
        Metropolis,
    );
}
