# -*- coding: utf-8 -*-
"""
    pint.compat
    ~~~~~~~~~~~

    Compatibility layer.

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

try:
    from collections import Chainmap
except:
    from .chainmap import ChainMap

try:
    from functools import lru_cache
except:
    from .lrucache import lru_cache
