use anneal_core::catalog::{SeamBank, fivefold_channel};
use anneal_core::methods::cluster_hopping::Config;

#[test]
fn the_measured_shares_fire_inside_two_epsilon_and_a_doorway_does_not() {
    let floor = -173.928427;
    assert!(
        fivefold_channel(floor + 2.0, floor, 0.120, 0.307),
        "a collapsed icosahedral share on the floor is the channel"
    );
    assert!(
        !fivefold_channel(floor + 2.0 + 1e-9, floor, 0.120, 0.307),
        "two epsilon is the window"
    );
    assert!(
        !fivefold_channel(floor, floor, 0.307, 0.307),
        "an uncollapsed share is not a fire"
    );
    assert!(
        !fivefold_channel(floor + 6.0, floor, 0.120, 0.307),
        "the twelve-epsilon doorway is not the channel"
    );
    assert!(!fivefold_channel(floor, f64::INFINITY, 0.120, 0.307));
    assert!(SeamBank::bin_of(0.45).is_some());
    assert!(!fivefold_channel(floor + 1.86, floor, 0.307, 0.307));
}

#[test]
fn restart_is_a_recommended_stall_arm() {
    for n in [38usize, 75] {
        assert!(Config::recommended(n).restart_on_stall);
        assert!(Config::derived(n).restart_on_stall);
        assert!(Config::recommended(n).escape_on_stall);
        assert!(!Config::communicating(n).restart_on_stall);
        assert!(!Config::communicating(n).escape_on_stall);
    }
}
