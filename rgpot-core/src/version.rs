// MIT License
// Copyright 2023--present rgpot developers

//! Version of the loaded library, for callers to check against the header
//! they compiled with.
//!
//! `rgpot.h` stamps `RGPOT_VERSION` and `RGPOT_VERSION_MAJOR` / `_MINOR` /
//! `_PATCH` at generation time; these functions report the same numbers
//! from the shared object actually loaded. The C ABI follows semver: a
//! caller built against major M runs on any library of major M whose minor
//! is at least the header's.
//!
//! The ABI stamp is separate from the crate version: see
//! `rgpot_abi_stamp()` and `rgpot_abi_compatible()`.
//!
//! ```c
//! if (rgpot_version_major() != RGPOT_VERSION_MAJOR ||
//!     rgpot_version_minor() < RGPOT_VERSION_MINOR) {
//!   fprintf(stderr, "rgpot %s loaded, built against %s\n",
//!           rgpot_version(), RGPOT_VERSION);
//!   abort();
//! }
//! ```

use std::os::raw::c_char;

/// Compatibility identity of the rgpot-core C ABI.
///
/// The numbers move with the layout of the exported structs and the
/// signatures of the exported functions, not with the crate version.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub struct rgpot_abi_stamp_t {
    /// Incompatible changes increment this value.
    pub abi_major: u16,
    /// Additive compatible changes increment this value.
    pub abi_minor: u16,
    /// Struct and function-layout revision for this ABI major.
    pub layout_revision: u16,
}

/// ABI major of this build.
pub const RGPOT_ABI_VERSION_MAJOR: u16 = 1;
/// ABI minor of this build.
pub const RGPOT_ABI_VERSION_MINOR: u16 = 4;
/// Layout revision of this build.
pub const RGPOT_ABI_LAYOUT_REVISION: u16 = 1;

static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

/// Full version string of the loaded library, as `MAJOR.MINOR.PATCH`.
///
/// The pointer refers to static storage: never free it.
#[no_mangle]
pub extern "C" fn rgpot_version() -> *const c_char {
    VERSION.as_ptr().cast()
}

static REVISION: &str = concat!(env!("RGPOT_SOURCE_REVISION"), "\0");

/// Source revision the loaded library was built from (a git short hash), or
/// an empty string when the build had no revision to record.
///
/// The pointer refers to static storage: never free it.
#[no_mangle]
pub extern "C" fn rgpot_source_revision() -> *const c_char {
    REVISION.as_ptr().cast()
}

/// Major version of the loaded library.
#[no_mangle]
pub extern "C" fn rgpot_version_major() -> u32 {
    parse(env!("CARGO_PKG_VERSION_MAJOR"))
}

/// Minor version of the loaded library.
#[no_mangle]
pub extern "C" fn rgpot_version_minor() -> u32 {
    parse(env!("CARGO_PKG_VERSION_MINOR"))
}

/// Patch version of the loaded library.
#[no_mangle]
pub extern "C" fn rgpot_version_patch() -> u32 {
    parse(env!("CARGO_PKG_VERSION_PATCH"))
}

/// Compatibility identity of the loaded library.
///
/// Compare it with the `RGPOT_ABI_*` macros of the header the caller
/// compiled with, or pass the caller's own stamp to
/// [`rgpot_abi_compatible`].
#[no_mangle]
pub extern "C" fn rgpot_abi_stamp() -> rgpot_abi_stamp_t {
    rgpot_abi_stamp_t {
        abi_major: RGPOT_ABI_VERSION_MAJOR,
        abi_minor: RGPOT_ABI_VERSION_MINOR,
        layout_revision: RGPOT_ABI_LAYOUT_REVISION,
    }
}

/// Return nonzero when the loaded library can serve a caller built with
/// `stamp`: equal major, equal layout revision, and a library minor at least
/// the caller's. A null `stamp` is not compatible.
///
/// # Safety
///
/// `stamp` must be null or point to a readable `rgpot_abi_stamp_t`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_abi_compatible(stamp: *const rgpot_abi_stamp_t) -> i32 {
    if stamp.is_null() {
        return 0;
    }
    i32::from(accepts(rgpot_abi_stamp(), unsafe { *stamp }))
}

/// Whether `library` can serve a caller built with `caller`.
const fn accepts(library: rgpot_abi_stamp_t, caller: rgpot_abi_stamp_t) -> bool {
    caller.abi_major == library.abi_major
        && caller.layout_revision == library.layout_revision
        && caller.abi_minor <= library.abi_minor
}

const fn parse(s: &str) -> u32 {
    let b = s.as_bytes();
    let mut v = 0u32;
    let mut i = 0;
    while i < b.len() {
        v = v * 10 + (b[i] - b'0') as u32;
        i += 1;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn version_string_matches_the_crate() {
        let s = unsafe { CStr::from_ptr(rgpot_version()) };
        assert_eq!(s.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
        let joined = format!(
            "{}.{}.{}",
            rgpot_version_major(),
            rgpot_version_minor(),
            rgpot_version_patch()
        );
        assert!(env!("CARGO_PKG_VERSION").starts_with(&joined));
    }

    #[test]
    fn stamp_equals_the_macros() {
        assert_eq!(
            rgpot_abi_stamp(),
            stamp(
                RGPOT_ABI_VERSION_MAJOR,
                RGPOT_ABI_VERSION_MINOR,
                RGPOT_ABI_LAYOUT_REVISION
            )
        );
        let stamp = rgpot_abi_stamp();
        assert_eq!(unsafe { rgpot_abi_compatible(&stamp) }, 1);
    }

    fn stamp(major: u16, minor: u16, layout: u16) -> rgpot_abi_stamp_t {
        rgpot_abi_stamp_t {
            abi_major: major,
            abi_minor: minor,
            layout_revision: layout,
        }
    }

    #[test]
    fn a_lower_caller_minor_is_accepted() {
        let library = stamp(2, 3, 5);
        assert!(accepts(library, stamp(2, 2, 5)));
        assert!(accepts(library, stamp(2, 0, 5)));
        assert!(accepts(library, stamp(2, 3, 5)));
    }

    #[test]
    fn a_higher_caller_minor_is_refused() {
        assert!(!accepts(stamp(2, 3, 5), stamp(2, 4, 5)));
    }

    #[test]
    fn a_different_major_is_refused() {
        assert!(!accepts(stamp(2, 3, 5), stamp(1, 3, 5)));
        assert!(!accepts(stamp(2, 3, 5), stamp(3, 3, 5)));
    }

    #[test]
    fn a_different_layout_revision_is_refused() {
        assert!(!accepts(stamp(2, 3, 5), stamp(2, 3, 4)));
        assert!(!accepts(stamp(2, 3, 5), stamp(2, 3, 6)));
    }

    #[test]
    fn the_exported_check_refuses_a_foreign_layout() {
        let mut foreign = rgpot_abi_stamp();
        foreign.layout_revision += 1;
        assert_eq!(unsafe { rgpot_abi_compatible(&foreign) }, 0);
        let mut ahead = rgpot_abi_stamp();
        ahead.abi_minor += 1;
        assert_eq!(unsafe { rgpot_abi_compatible(&ahead) }, 0);
    }

    #[test]
    fn a_null_stamp_is_refused() {
        assert_eq!(unsafe { rgpot_abi_compatible(std::ptr::null()) }, 0);
    }

    /// The committed header must carry this crate's version: cbindgen
    /// stamps it only on regeneration and a release bump stamps it with
    /// potctl, so drift means one of the two was skipped.
    #[test]
    fn source_revision_is_a_c_string_of_the_build_revision() {
        let got = unsafe { std::ffi::CStr::from_ptr(rgpot_source_revision()) }
            .to_str()
            .unwrap();
        assert_eq!(got, env!("RGPOT_SOURCE_REVISION"));
        assert!(got.bytes().all(|b| b.is_ascii_alphanumeric()), "{got:?}");
    }

    #[test]
    fn committed_header_matches_the_crate() {
        let header =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/include/rgpot.h"))
                .unwrap();
        let want = format!("#define RGPOT_VERSION \"{}\"", env!("CARGO_PKG_VERSION"));
        assert!(header.contains(&want), "rgpot.h lacks {want}");
        for (name, value) in [
            ("RGPOT_ABI_VERSION_MAJOR", RGPOT_ABI_VERSION_MAJOR),
            ("RGPOT_ABI_VERSION_MINOR", RGPOT_ABI_VERSION_MINOR),
            ("RGPOT_ABI_LAYOUT_REVISION", RGPOT_ABI_LAYOUT_REVISION),
        ] {
            let want = format!("#define {name} {value}");
            assert!(header.contains(&want), "rgpot.h lacks {want}");
        }
        for f in [
            "rgpot_version(void)",
            "rgpot_version_major(void)",
            "rgpot_version_minor(void)",
            "rgpot_version_patch(void)",
            "rgpot_abi_stamp(void)",
            "rgpot_abi_compatible(const struct rgpot_abi_stamp_t *stamp)",
            "rgpot_potential_hvp(const struct rgpot_potential_t *pot,",
            "rgpot_potential_hvp_centre(const struct rgpot_potential_t *pot,",
            "rgpot_potential_set_hvp_callback(const struct rgpot_potential_t *pot,",
            "rgpot_potential_hvp_invalidate(const struct rgpot_potential_t *pot)",
            "rgpot_potential_set_taylor_jet(const struct rgpot_potential_t *pot,",
            "rgpot_potential_taylor_jet(const struct rgpot_potential_t *pot,",
            "(*rgpot_taylor_jet_fn)(void *user_data,",
            "typedef struct rgpot_interval_t {",
            "RGPOT_NO_JET = 6,",
            "RGPOT_JET_BRANCH = 7,",
            "RGPOT_JET_DOMAIN = 8,",
            "rgpot_potential_hvp_counts(const struct rgpot_potential_t *pot,",
            "rgpot_potential_reset_hvp_counts(const struct rgpot_potential_t *pot)",
            "(*HessianVectorCallback)(void *user_data,",
            "typedef enum rgpot_hvp_scheme_t {",
            "typedef struct rgpot_hvp_counts_t {",
            "typedef enum rgpot_fused_error_kind_t {",
            "typedef struct rgpot_fused_settings_t {",
            "rgpot_fused_settings_default(struct rgpot_fused_settings_t *out)",
            "rgpot_potential_set_fused_settings(",
            "rgpot_potential_fused_error_kind(const struct rgpot_potential_t *pot)",
            "rgpot_set_last_error(const char *msg)",
            "RGPOT_SCC_NOT_CONVERGED = 5",
        ] {
            assert!(header.contains(f), "rgpot.h lacks {f}");
        }
        let default_step = format!(
            "#define RGPOT_HVP_DEFAULT_STEP {:.1e}",
            crate::hvp::RGPOT_HVP_DEFAULT_STEP
        );
        assert!(
            header.contains(&default_step),
            "rgpot.h lacks {default_step}"
        );
    }

    /// Every name in the rgpot-core ABI baseline is still declared by the
    /// committed header. The meson test `abi_core_symbols` checks the same
    /// list against the symbols the shared library exports.
    #[test]
    fn the_abi_baseline_is_declared_by_the_header() {
        let root = env!("CARGO_MANIFEST_DIR");
        let header = std::fs::read_to_string(format!("{root}/include/rgpot.h")).unwrap();
        let Ok(baseline) =
            std::fs::read_to_string(format!("{root}/../scripts/abi_core_baseline.txt"))
        else {
            // A packaged crate carries the header but not the repository scripts.
            return;
        };
        for name in baseline.lines().filter(|l| !l.is_empty()) {
            assert!(header.contains(&format!("{name}(")), "rgpot.h lacks {name}");
        }
        for name in [
            "rgpot_potential_hvp",
            "rgpot_potential_hvp_centre",
            "rgpot_potential_set_hvp_callback",
            "rgpot_potential_hvp_invalidate",
            "rgpot_potential_hvp_counts",
            "rgpot_potential_reset_hvp_counts",
            "rgpot_set_last_error",
            "rgpot_fused_settings_default",
            "rgpot_potential_set_fused_settings",
            "rgpot_potential_fused_error_kind",
            "rgpot_potential_set_taylor_jet",
            "rgpot_potential_taylor_jet",
        ] {
            assert!(baseline.lines().any(|l| l == name), "baseline lacks {name}");
        }
    }

    /// A library built without the schema feature lacks these names, so the
    /// header declares them only when RGPOT_HAS_SCHEMA is defined.
    #[test]
    fn schema_only_names_are_behind_rgpot_has_schema() {
        let root = env!("CARGO_MANIFEST_DIR");
        let header = std::fs::read_to_string(format!("{root}/include/rgpot.h")).unwrap();
        for needle in [
            "#define PROTOCOL_MAJOR ",
            "#define PROTOCOL_MINOR ",
            "rgpot_potential_set_peer_capabilities(",
        ] {
            let name = needle.trim_end_matches([' ', '(']);
            let at = header
                .find(needle)
                .unwrap_or_else(|| panic!("rgpot.h lacks {name}"));
            let open = header[..at].rfind("#if").unwrap_or(0);
            assert!(
                header[open..].starts_with("#if defined(RGPOT_HAS_SCHEMA)")
                    && !header[open..at].contains("#endif"),
                "{name} is not behind #if defined(RGPOT_HAS_SCHEMA)"
            );
        }
    }
}
