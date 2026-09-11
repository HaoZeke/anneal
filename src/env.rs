//! Environment knobs, read the same way everywhere.
//!
//! The drivers and the coordinator read well over a hundred `KEY=value`
//! knobs, and each site spelled the same three lines: read the variable,
//! parse it, fall back. These three functions are those lines once; a
//! knob that fails to parse is treated as unset, as every site did.

use std::str::FromStr;

/// The parsed value of `name`, or `None` when unset, empty, or unparsable.
pub fn parsed<T: FromStr>(name: &str) -> Option<T> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
}

/// The parsed value of `name`, or `fallback` when unset or unparsable.
pub fn parsed_or<T: FromStr>(name: &str, fallback: T) -> T {
    parsed(name).unwrap_or(fallback)
}

/// Whether `name` is set to exactly `1`, the convention for on/off knobs.
pub fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| value == "1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_empty_and_unparsable_knobs_read_as_absent() {
        // Environment mutation is process-wide; each test uses its own name.
        unsafe { std::env::remove_var("ANNEAL_ENV_TEST_A") };
        assert_eq!(parsed::<u64>("ANNEAL_ENV_TEST_A"), None);
        unsafe { std::env::set_var("ANNEAL_ENV_TEST_A", "") };
        assert_eq!(parsed::<u64>("ANNEAL_ENV_TEST_A"), None);
        unsafe { std::env::set_var("ANNEAL_ENV_TEST_A", "twelve") };
        assert_eq!(parsed_or("ANNEAL_ENV_TEST_A", 7u64), 7);
        unsafe { std::env::set_var("ANNEAL_ENV_TEST_A", " 12 ") };
        assert_eq!(parsed::<u64>("ANNEAL_ENV_TEST_A"), Some(12));
        assert!(!flag("ANNEAL_ENV_TEST_A"));
        unsafe { std::env::set_var("ANNEAL_ENV_TEST_A", "1") };
        assert!(flag("ANNEAL_ENV_TEST_A"));
        unsafe { std::env::remove_var("ANNEAL_ENV_TEST_A") };
    }
}
