// Copyright (c) anneal developers
//! The crate-root sample in `docs/orgmode/reference/rust-api.org`.
//!
//! `build.rs` extracts that Org source block, and this test compiles it.

include!(concat!(
    env!("ANNEAL_RUST_API_SNIPPET_DIR"),
    "/rust_api_snippet.rs"
));
