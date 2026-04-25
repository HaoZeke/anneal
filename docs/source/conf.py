import os

project   = "anneal"
copyright = "2026--present, anneal developers"
author    = "anneal developers"
release   = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinxcontrib_rust",
    "sphinx_rustdoc_postprocess",
]

templates_path = ["_templates"]
exclude_patterns = []

rust_crates       = {"anneal_core": os.path.abspath("../../")}
rust_doc_dir      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crates")
rust_rustdoc_fmt  = "rst"
rust_generate_mode = "always"

html_theme        = "shibuya"
html_static_path  = ["_static"]
html_theme_options = {"github_url": "https://github.com/HaoZeke/anneal"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "eindir": ("https://haozeke.github.io/eindir", None),
}
