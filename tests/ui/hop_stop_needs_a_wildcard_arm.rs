use anneal_core::methods::cluster_hopping::HopStop;

fn main() {
    let _ = code(HopStop::Budget);
}

// Naming all seven variants is not enough outside the crate. A later
// release may add a reason, so the match needs a wildcard arm.
fn code(stop: HopStop) -> &'static str {
    match stop {
        HopStop::Budget => "budget",
        HopStop::Target => "target",
        HopStop::CallerStop => "caller_stop",
        HopStop::Retired { .. } => "retired",
        HopStop::NveUnavailable => "nve_unavailable",
        HopStop::MaxHops => "max_hops",
        HopStop::NoTrajectory => "no_trajectory",
    }
}
