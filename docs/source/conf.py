# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------

project = 'DiCE'
copyright = '2020, Ramaravind, Amit, Chenhao'
author = 'Ramaravind, Amit, Chenhao'

# The full version, including alpha/beta/rc tags
release = '0.5'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'nbsphinx',
    'sphinx_rtd_theme'
]

autodoc_mock_imports = ['numpy', 'pandas', 'matplotlib', 'os', 'tensorflow', 'random', 'collections', 'timeit', 'tensorflow.keras', 'sklearn', 'sklearn.model_selection.train_test_split', 'copy', 'IPython', 'IPython.display.display', 'collections', 'collections.OrderedDict', 'logging', 'torch', 'torchvision']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }
