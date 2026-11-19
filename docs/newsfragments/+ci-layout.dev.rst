The default-feature test suite runs once per operating system in its own job, beside the feature jobs.
Test binaries are built with line tables only and no incremental state, and ``Swatinem/rust-cache`` restores the third-party crates.
The macOS Python matrix runs only when a path it depends on changes.
The test job fetches full history, which the paper audit needs.
