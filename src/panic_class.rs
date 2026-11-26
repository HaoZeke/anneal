//! Whether a native panic text is a caller mistake.
//!
//! `assert!` puts that text in the panic payload. `assert_eq!` wraps it as
//! `assertion \`left == right\` failed: <text>`. A marker counts only as that
//! text, not as a substring of some other message.

const MARKERS: &[&str] = &[
    "call floor must be a positive distance",
    "call-floor elements must name every point",
    "continuous-symmetry interval must be positive",
    "continuous-symmetry projection requires a free non-periodic point set",
    "continuous-symmetry classes must match n_points",
    "delayed acceptance needs a single chain",
    "shape keying needs the `ira` feature",
    "screen_steps must be strictly less than relax_steps",
    "length_scale must be finite and positive",
    "energy_scale must be finite and positive",
    "species must not be empty",
    "species must have known covalent radii",
    "a rigid cluster needs at least two bodies",
    "checkpoint interval must be positive",
    "a shared bias and a replica ladder are different things",
    "interleaved TIP4P state is 6 per molecule",
    "block TIP4P state is 6 per molecule",
];

const ASSERT_EQ_PREFIX: &str = "assertion `left == right` failed: ";

/// True when `msg` is an input-contract panic, not an internal failure.
///
/// The python bindings are the caller. Unit tests cover the predicate when
/// that feature is off, so a default build may not name it.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn panic_is_bad_input(msg: &str) -> bool {
    MARKERS.iter().any(|marker| marker_matches(msg, marker))
}

fn marker_matches(msg: &str, marker: &str) -> bool {
    bounded_prefix(msg, marker)
        || msg
            .strip_prefix(ASSERT_EQ_PREFIX)
            .is_some_and(|rest| bounded_prefix(rest, marker))
}

fn bounded_prefix(msg: &str, marker: &str) -> bool {
    let Some(rest) = msg.strip_prefix(marker) else {
        return false;
    };
    rest.chars()
        .next()
        .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_text_matches_a_contract_only_as_itself() {
        assert!(panic_is_bad_input(
            "length_scale must be finite and positive"
        ));
        assert!(panic_is_bad_input(
            "call floor must be a positive distance, not nan"
        ));
        assert!(panic_is_bad_input(
            "screen_steps must be strictly less than relax_steps (got 2, 4)"
        ));
        let wrapped = "assertion `left == right` failed: call-floor elements must name every point\n  left: 1\n right: 2";
        assert!(panic_is_bad_input(wrapped));
        assert!(
            !panic_is_bad_input(
                "quench invariant failed after length_scale must be finite and positive"
            ),
            "a marker buried in another message is not bad input"
        );
        assert!(
            !panic_is_bad_input("length_scale must be finite and positiveextra"),
            "a marker that runs into more identifier text is not that contract"
        );
        assert!(!panic_is_bad_input(
            "Gupta needs the rgpot kernels: call anneal_core::install_rgpot_kernels!()"
        ));
        assert!(!panic_is_bad_input(
            "PairPotential expects 6 coordinates, got 3"
        ));
        assert!(!panic_is_bad_input(
            "internal panic in native cluster kernel"
        ));
    }
}
