The package version is the ``version`` field in the workspace
``Cargo.toml``.
Cog (https://nedbatchelder.com/code/cog/) generates the copies from that
field.
Those copies are the Python, Pixi, Meson, citation, Zenodo, README and C
header files.
CMake, Sphinx and the compiled libraries read the same field when they
configure or compile.
Towncrier (https://towncrier.readthedocs.io/en/stable/) reads it when
assembling the changelog.
A release bump edits that one line, runs cog, then builds the news
fragments.
Meson and CMake had stayed at 0.5.0.
