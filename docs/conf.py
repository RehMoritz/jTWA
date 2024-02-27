import os
import sys
import pathlib

sys.path.append(str(pathlib.PosixPath(os.getcwd()) / "../"))

import jTWA

# -- Project information -----------------------------------------------------

project = "jTWA"
copyright = "2024, Moritz Reh"
author = "Moritz Reh"
release = "0.0.3"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "html_admonition"]
myst_update_mathjax = True
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]


html_theme_options = {
    "home_page_in_toc": False,
    "show_navbar_depth": 1,
    "show_toc_level": 3,
    "repository_url": "https://github.com/RehMoritz/jTWA",
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs",
    "navigation_with_keys": True,
}
