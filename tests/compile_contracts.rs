//! Compile-time checks for the certified component constructor.

#[test]
fn checked_rejects_uncertified_component_combinations() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/checked_rejects_*.rs");
}
