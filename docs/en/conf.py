# flake8: noqa
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
import subprocess
import sys

import pytorch_sphinx_theme
from sphinx.builders.html import StandaloneHTMLBuilder

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'MMClassification'
copyright = '2020, OpenMMLab'
author = 'MMClassification Authors'

# The full version, including alpha/beta/rc tags
version_file = '../../mmcls/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'sphinx_copybutton',
]

autodoc_mock_imports = ['mmcv._ext', 'matplotlib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

language = 'en'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# yapf: disable
html_theme_options = {
    'logo_url': 'https://mmclassification.readthedocs.io/en/latest/',
    'menu': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/open-mmlab/mmclassification'
        },
        {
            'name': 'Colab Tutorials',
            'children': [
                {
                    'name': 'Train and inference with shell commands',
                    'url': 'https://colab.research.google.com/github/open-mmlab/mmclassification/blob/master/docs/en/tutorials/MMClassification_tools.ipynb',
                },
                {
                    'name': 'Train and inference with Python APIs',
                    'url': 'https://colab.research.google.com/github/open-mmlab/mmclassification/blob/master/docs/en/tutorials/MMClassification_python.ipynb',
                },
            ]
        },
        {
            'name': 'Version',
            'children': [
                {
                    'name': 'MMClassification 0.x',
                    'url': 'https://mmclassification.readthedocs.io/en/latest/',
                    'description': 'master branch'
                },
                {
                    'name': 'MMClassification 1.x',
                    'url': 'https://mmclassification.readthedocs.io/en/dev-1.x/',
                    'description': '1.x branch'
                },
            ],
        }
    ],
    # Specify the language of shared menu
    'menu_lang': 'en',
    'header_note': {
        'content':
        'You are reading the documentation for MMClassification 0.x, which '
        'will soon be deprecated at the end of 2022. We recommend you upgrade '
        'to MMClassification 1.0 to enjoy fruitful new features and better '
        'performance brought by OpenMMLab 2.0. Check the '
        '<a href="https://mmclassification.readthedocs.io/en/dev-1.x/get_started.html#installation">installation tutorial</a>, '
        '<a href="https://mmclassification.readthedocs.io/en/dev-1.x/migration.html">migration tutorial</a> '
        'and <a href="https://mmclassification.readthedocs.io/en/dev-1.x/notes/changelog.html">changelog</a> '
        'for more details.',
    }
}
# yapf: enable

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/readthedocs.css']
html_js_files = ['js/custom.js']

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'mmclsdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    'preamble':
    r'''
\hypersetup{unicode=true}
\usepackage{CJKutf8}
\DeclareUnicodeCharacter{00A0}{\nobreakspace}
\DeclareUnicodeCharacter{2203}{\ensuremath{\exists}}
\DeclareUnicodeCharacter{2200}{\ensuremath{\forall}}
\DeclareUnicodeCharacter{2286}{\ensuremath{\subseteq}}
\DeclareUnicodeCharacter{2713}{x}
\DeclareUnicodeCharacter{27FA}{\ensuremath{\Longleftrightarrow}}
\DeclareUnicodeCharacter{221A}{\ensuremath{\sqrt{}}}
\DeclareUnicodeCharacter{221B}{\ensuremath{\sqrt[3]{}}}
\DeclareUnicodeCharacter{2295}{\ensuremath{\oplus}}
\DeclareUnicodeCharacter{2297}{\ensuremath{\otimes}}
\begin{CJK}{UTF8}{gbsn}
\AtEndDocument{\end{CJK}}
''',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'mmcls.tex', 'MMClassification Documentation', author,
     'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'mmcls', 'MMClassification Documentation', [author],
              1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'mmcls', 'MMClassification Documentation', author, 'mmcls',
     'OpenMMLab image classification toolbox and benchmark.', 'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# set priority when building html
StandaloneHTMLBuilder.supported_image_types = [
    'image/svg+xml', 'image/gif', 'image/png', 'image/jpeg'
]

# -- Extension configuration -------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True
# Auto-generated header anchors
myst_heading_anchors = 3
# Configuration for intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'mmcv': ('https://mmcv.readthedocs.io/en/master/', None),
}


def builder_inited_handler(app):
    subprocess.run(['./stat.py'])


def setup(app):
    app.connect('builder-inited', builder_inited_handler)
