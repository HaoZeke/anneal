// Copyright (c) anneal developers

//! Common-neighbour analysis remains reporting-only and does not inherit a
//! search-scale-dependent motif cutoff.

use anneal_core::soap::{CONTACT_CUTOFF, SoapSpec, contact_cutoff};
use anneal_core::structure::cna;
use ndarray::array;

#[test]
fn the_reporting_cutoff_is_independent_of_the_search_length_scale() {
    for length in [0.25, 0.5, 1.0, 2.0, 8.0] {
        assert_eq!(contact_cutoff(SoapSpec::at_length(length)), CONTACT_CUTOFF);
    }
}

#[test]
fn analysis_callers_can_still_choose_an_explicit_cutoff() {
    let points = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.1, 0.0, 0.0];
    let near = cna(points.view(), 3, 1.05);
    let far = cna(points.view(), 3, 2.2);
    assert_eq!(near.bonds, 1);
    assert_eq!(far.bonds, 3);
}
