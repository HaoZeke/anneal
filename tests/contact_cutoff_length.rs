// Copyright (c) anneal developers

//! The common-neighbour cutoff follows the length the spec is written at.

use anneal_core::soap::{CONTACT_CUTOFF, SoapSpec, contact_cutoff};
use anneal_core::structure::cna;
use ndarray::array;

#[test]
fn the_default_spec_gives_the_lennard_jones_cutoff() {
    assert_eq!(contact_cutoff(SoapSpec::default()), CONTACT_CUTOFF);
    assert_eq!(contact_cutoff(SoapSpec::at_length(1.0)), CONTACT_CUTOFF);
}

#[test]
fn half_the_length_gives_half_the_cutoff() {
    let half = SoapSpec::at_length(0.5);
    assert_eq!(half.rcut_nn, SoapSpec::default().rcut_nn * 0.5);
    assert_eq!(contact_cutoff(half), CONTACT_CUTOFF * 0.5);
}

#[test]
fn analysis_callers_can_still_choose_an_explicit_cutoff() {
    let points = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.1, 0.0, 0.0];
    let near = cna(points.view(), 3, 1.05);
    let far = cna(points.view(), 3, 2.2);
    assert_eq!(near.bonds, 1);
    assert_eq!(far.bonds, 3);
}
