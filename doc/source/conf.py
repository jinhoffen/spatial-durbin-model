import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "sgm"
copyright = "2023, Justus Inhoffen"
author = "Justus Inhoffen"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # shorten external links
    "sphinx.ext.extlinks",
    # test code snippets in docstring
    "sphinx.ext.doctest",
    # todo items
    "sphinx.ext.todo",
    # add links to highlighted source code
    "sphinx.ext.viewcode",
    # import modules' docstrings
    "sphinx.ext.autodoc",
    # math
    "sphinx.ext.mathjax",
    # template
    # "sphinx_immaterial",
    # convert Google style docstring to restructured text
    "sphinx.ext.napoleon"
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster" # sphinx_immaterial
html_static_path = ["_static"]

# html_css_files = [
#     'css/custom.css',
# ]