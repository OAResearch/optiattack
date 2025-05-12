""" Sphinx configuration. """
# -*- coding: utf-8 -*-

import os
import sys
import datetime

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx']
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = u'OptiAttack'
copyright = u'%s, OptiAttack Team' % datetime.datetime.now().year
version = '0.1.6'
exclude_patterns = ['_build']
autodoc_member_order = 'bysource'
html_use_modindex = False
html_show_sphinx = False
htmlhelp_basename = 'Optiattackdoc'
latex_documents = [
    ('index', 'OptiAttack.tex', u'OptiAttack Documentation',
        u'OptiAttack Team', 'manual'),
]
latex_use_modindex = False
latex_use_parts = True
man_pages = [
    ('index', 'optiattack', u'OptiAttack Documentation',
     [u'OptiAttack Team'], 1)
]
pygments_style = 'tango'
html_theme = 'default'
html_theme_options = {}
