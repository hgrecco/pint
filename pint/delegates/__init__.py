"""
    pint.delegates
    ~~~~~~~~~~~~~~

    Defines methods and classes to handle autonomous tasks.

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from . import txt_defparser
from .base_defparser import ParserConfig, build_disk_cache_class

__all__ = [txt_defparser, ParserConfig, build_disk_cache_class]
