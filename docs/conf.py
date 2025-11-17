# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
from importlib.metadata import version as _version
from pathlib import Path
from tomllib import load as _load_toml


project = "denspp.offline"
copyright = "2025, ies-ude (Intelligent Embedded System - University of Duisburg-Essen)"
author = "es-ude"
release = _version("denspp.offline")
version = ".".join(_version("denspp.offline").split(".")[0:2])
html_title = "denspp.offline"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "autodoc2",
    "sphinxext.opengraph",
    "sphinxcontrib.mermaid"
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# Default theme configuration
html_show_sourcelink = False


# Configure theme based on build type
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "content_footer_items": ["last-updated"],
    "navigation_depth": 4,
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/es-ude/denspp.offline/",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
}


# only github flavored markdown
myst_gfm_only = False
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# allow mermaid usage like on github in markdown
myst_fence_as_directive = ["mermaid"]

running_in_autobuild = os.getenv("SPHINX_AUTOBUILD", "NO") == "YES"


autodoc2_packages = [
    {
        "path": "../denspp/offline",
        "module": "denspp.offline",
    },
]

autodoc2_skip_module_regexes = [
    ".*_test"
]

autodoc2_render_plugin = "myst"
autodoc2_hidden_objects = {"inherited", "private"}

myst_heading_anchors = 3
myst_heading_slug = True
