import os
import sys

project = 'HistoKit'
author = 'Julia Merta, Michal Marczyk'
version = '1.0'
release = '0.0.1'

sys.path.insert(0, os.path.abspath('../../src/'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex'
]
html_static_path = ['images']

bibtex_bibfiles = ['biblio.bib']

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'inherited-members': True,
    'show-inheritance': True
}
html_theme = 'furo'
napoleon_google_docstring = False
napoleon_numpy_docstring = True
