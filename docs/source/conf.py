import os

# -- Project information -----------------------------------------------------
project = "anneal"
copyright = "2026--present, anneal developers"
author = "anneal developers"
html_logo = "../../branding/logo/anneal_logo.png"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_sitemap",
    "sphinxcontrib_rust",
    "sphinx_rustdoc_postprocess",
]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "eindir": ("https://eindir.rgoswami.me", None),
}

# -- sphinxcontrib-rust configuration ----------------------------------------
rust_crates = {
    "anneal_core": os.path.abspath("../../"),
}
rust_doc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crates")
rust_rustdoc_fmt = "rst"
rust_generate_mode = "always"

# -- sphinx-rustdoc-postprocess configuration --------------------------------
rustdoc_postprocess_toctree_target = "reference/rust-api.rst"
rustdoc_postprocess_toctree_rst = """
Rust API (``anneal-core``)
--------------------------

.. toctree::
   :maxdepth: 2

   ../crates/anneal_core/lib
"""

# -- Options for HTML output -------------------------------------------------
html_theme = "shibuya"
html_static_path = ["_static"]
html_favicon = "_static/favicon.png"
html_js_files = [
    ("https://antics-api.turtletech.us/antics.js", {"defer": "defer"}),
]

html_theme_options = {
    "github_url": "https://github.com/HaoZeke/anneal",
    "light_logo": "_static/anneal_icon.png",
    "dark_logo": "_static/anneal_icon.png",
    "og_image_url": "https://anneal.rgoswami.me/_static/og-image.png",
    "accent_color": "teal",
    "dark_code": True,
    "globaltoc_expand_depth": 1,
    "nav_links": [
        {
            "title": "Ecosystem",
            "children": [
                {
                    "title": "eindir",
                    "url": "https://eindir.rgoswami.me",
                    "summary": "Typed primitives for ND objectives and sampling (used by anneal)",
                },
                {
                    "title": "rgpycrumbs",
                    "url": "https://rgpycrumbs.rgoswami.me",
                    "summary": "Chemical physics utilities and visualization",
                },
            ],
        },
    ],
}

html_context = {
    "source_type": "github",
    "source_user": "HaoZeke",
    "source_repo": "anneal",
    "source_version": "main",
    "source_docs_path": "/docs/source/",
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "sidebars/repo-stats.html",
        "sidebars/edit-this-page.html",
    ],
}

html_baseurl = "https://anneal.rgoswami.me/"
