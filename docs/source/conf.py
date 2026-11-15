#!/usr/bin/env python3

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../../subprojects/doxyrest/sphinx"))

# -- Project information -----------------------------------------------------
project = "rgpot"
copyright = "2025--present, rgpot developers"
author = "Rohit Goswami"
# [[[cog import rgpot_version as v; cog.outl(f'version = "{v.VERSION}"'); cog.outl(f'release = "{v.VERSION}"')]]]
version = "3.7.0"
release = "3.7.0"
# [[[end]]]
# Navbar falls back to html_logo when light_logo/dark_logo are unset.
html_logo = "_static/logo-light.svg"
html_favicon = "_static/favicons/favicon.svg"

# -- General configuration ---------------------------------------------------
extensions = [
    "doxyrest",
    "cpplexer",
    "myst_parser",
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
    "eon": ("https://eondocs.org", None),
}

# -- sphinxcontrib-rust configuration ----------------------------------------
rust_crates = {
    "rgpot_core": os.path.abspath("../../rgpot-core/"),
}
rust_doc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crates")
rust_rustdoc_fmt = "rst"
rust_generate_mode = "always"

# -- sphinx-rustdoc-postprocess configuration --------------------------------
rustdoc_postprocess_toctree_target = "api/index.rst"
rustdoc_postprocess_toctree_rst = """
Rust API (``rgpot-core``)
-------------------------

.. toctree::
   :maxdepth: 2

   ../crates/rgpot_core/lib
"""

# -- Options for HTML output -------------------------------------------------
html_theme = "shibuya"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/OmniPotentRPC/rgpot",
    "accent_color": "teal",
    "dark_code": True,
    "globaltoc_expand_depth": 1,
    # Shibuya toggles .light-logo / .dark-logo with color mode (paths via pathto).
    "light_logo": "_static/logo-light.svg",
    "dark_logo": "_static/logo-dark.svg",
    # Shared Ecosystem list. Same order, titles, URLs, and summaries on every
    # site. A site drops its own row. This copy keeps rgpot so the list is whole.
    "nav_links": [
        {
            "title": "Ecosystem",
            "children": [
                {
                    "title": "eOn",
                    "url": "https://eondocs.org",
                    "summary": "Saddle point finding on potential energy surfaces",
                },
                {
                    "title": "rgpot",
                    "url": "https://rgpot.rgoswami.me",
                    "summary": "Potential-energy library and Cap'n Proto RPC server",
                },
                {
                    "title": "rgmin",
                    "url": "https://omnipotentrpc.github.io/rgmin",
                    "summary": "Local minimization over eindir objectives",
                },
                {
                    "title": "rgsaddle",
                    "url": "https://omnipotentrpc.github.io/rgsaddle",
                    "summary": "Band and minimum-mode saddle mechanics over rgmin",
                },
                {
                    "title": "cpmdc",
                    "url": "https://cpmdc.rgoswami.me",
                    "summary": "C ABI layer for embedding OpenCPMD",
                },
                {
                    "title": "rgpycrumbs",
                    "url": "https://rgpycrumbs.rgoswami.me",
                    "summary": "Pure-Python computational library and CLI for chemical physics",
                },
                {
                    "title": "minimage",
                    "url": "https://lode-org.github.io/minimage",
                    "summary": "Minimum-image convention on a general parallelepiped",
                },
                {
                    "title": "linkcell",
                    "url": "https://docs.rs/linkcell",
                    "summary": "Periodic linked-cell k-nearest neighbour search",
                },
                {
                    "title": "readcon",
                    "url": "https://lode-org.github.io/readcon-core",
                    "summary": "Reference implementation of versioned CON checkpoints",
                },
                {
                    "title": "readcon-db",
                    "url": "https://lode-org.github.io/readcon-db",
                    "summary": "Mmap-backed CON campaign corpus store",
                },
                {
                    "title": "readcon-ops",
                    "url": "https://github.com/lode-org/readcon-ops",
                    "summary": "Comparisons and rigid alignment of coordinate frames",
                },
            ],
        },
    ],
}

html_context = {
    "source_type": "github",
    "source_user": "OmniPotentRPC",
    "source_repo": "rgpot",
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

html_baseurl = "https://rgpot.rgoswami.me/"
# Sphinx sets language to "en". The default sitemap scheme is
# {lang}{version}{link}, so every page would be published under /en/.
sitemap_url_scheme = "{link}"

html_css_files = [
    "custom.css",
]


def _doxyrest_unrendered(app, env, node, contnode):
    """Render a Doxygen link to a page doxyrest does not write as plain text.

    Doxyrest writes no page for source listings, Fortran derived types or
    undocumented private members, but still links to them.
    """
    if node.get("refdomain") == "std" and node.get("reftarget", "").startswith("doxid-"):
        return contnode
    return None


def setup(app):
    from pygments.lexers.special import TextLexer

    # Pygments has no LAMMPS input lexer.
    app.add_lexer("lammps", TextLexer)
    app.connect("missing-reference", _doxyrest_unrendered)
