The release workflow's token reads the repository by default. Only the GitHub
release job is granted ``contents: write``; the build, sdist, wheel and
crates.io jobs could write to the repository before.
