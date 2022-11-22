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

sys.path.insert(0, os.path.abspath('../..'))

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

language = 'zh_CN'

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
# yapf: disable
html_theme_options = {
    'logo_url': 'https://mmclassification.readthedocs.io/zh_CN/latest/',
    'menu': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/open-mmlab/mmclassification'
        },
        {
            'name': 'Colab 教程',
            'children': [
                {
                    'name': '用命令行工具训练和推理',
                    'url': 'https://colab.research.google.com/github/open-mmlab/mmclassification/blob/master/docs/zh_CN/tutorials/MMClassification_tools_cn.ipynb',
                },
                {
                    'name': '用 Python API 训练和推理',
                    'url': 'https://colab.research.google.com/github/open-mmlab/mmclassification/blob/master/docs/zh_CN/tutorials/MMClassification_python_cn.ipynb',
                },
            ]
        },
        {
            'name': '版本',
            'children': [
                {
                    'name': 'MMClassification 0.x',
                    'url': 'https://mmclassification.readthedocs.io/zh_CN/latest/',
                    'description': 'master 分支'
                },
                {
                    'name': 'MMClassification 1.x',
                    'url': 'https://mmclassification.readthedocs.io/zh_CN/dev-1.x/',
                    'description': '1.x 分支'
                },
            ],
        }
    ],
    # Specify the language of shared menu
    'menu_lang': 'cn',
    'header_note': {
        'content':
        '您正在阅读 MMClassification 0.x 版本的文档。MMClassification 0.x 会在 2022 年末'
        '被切换为次要分支。建议您升级到 MMClassification 1.0 版本，体验更多新特性和新功能。'
        '请查阅 MMClassification 1.0 的'
        '<a href="https://mmclassification.readthedocs.io/zh_CN/dev-1.x/get_started.html#installation">安装教程</a>、'
        '<a href="https://mmclassification.readthedocs.io/zh_CN/dev-1.x/migration.html">迁移教程</a>'
        '以及<a href="https://mmclassification.readthedocs.io/en/dev-1.x/notes/changelog.html">更新日志</a>。',
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

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
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
    'mmcv': ('https://mmcv.readthedocs.io/zh_CN/latest/', None),
}


def builder_inited_handler(app):
    subprocess.run(['./stat.py'])


def setup(app):
    app.add_config_value('no_underscore_emphasis', False, 'env')
    app.connect('builder-inited', builder_inited_handler)
