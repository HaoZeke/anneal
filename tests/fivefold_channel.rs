use anneal_core::catalog::{SeamBank, local_packing_cohesion, packing_doorway_channel};
use anneal_core::methods::cluster_hopping::Config;
use ndarray::Array1;

#[test]
fn the_measured_shares_fire_inside_two_epsilon_and_a_doorway_does_not() {
    let floor = -173.928427;
    assert!(
        packing_doorway_channel(floor + 2.0, floor, 0.120, 0.307),
        "a reduced packing cohesion on the floor is the channel"
    );
    assert!(
        !packing_doorway_channel(floor + 2.0 + 1e-9, floor, 0.120, 0.307),
        "two epsilon is the window"
    );
    assert!(
        !packing_doorway_channel(floor, floor, 0.307, 0.307),
        "an uncollapsed cohesion is not a fire"
    );
    assert!(
        !packing_doorway_channel(floor + 6.0, floor, 0.120, 0.307),
        "the twelve-epsilon doorway is not the channel"
    );
    assert!(!packing_doorway_channel(floor, f64::INFINITY, 0.120, 0.307));
    assert!(SeamBank::bin_of(0.45).is_some());
    assert!(!packing_doorway_channel(floor + 1.86, floor, 0.307, 0.307));
    let sample = Array1::from_vec(vec![
        0.0, 0.0, 0.0, 1.12, 0.0, 0.0, 0.56, 0.97, 0.0, 0.56, 0.32, 0.91,
    ]);
    let cohesion = local_packing_cohesion(sample.view());
    assert!((0.0..=1.0).contains(&cohesion));
}

#[test]
fn restart_is_not_a_recommended_stall_arm() {
    for n in [38usize, 75, 98] {
        assert!(!Config::recommended(n).restart_on_stall);
        assert!(!Config::derived(n).restart_on_stall);
        assert!(Config::recommended(n).escape_on_stall);
        assert!(!Config::communicating(n).restart_on_stall);
        assert!(!Config::communicating(n).escape_on_stall);
    }
}
