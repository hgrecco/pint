"""
pint.delegates.txt_defparser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parser for the original textual Pint Definition file.

This module has a relatively high import cost (imports ``flexcache`` and
``flexparser``), so import it lazily.

:copyright: 2022 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from .defparser import DefParser

__all__ = [
    "DefParser",
]
