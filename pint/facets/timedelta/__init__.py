"""
pint.facets.timedelta
~~~~~~~~~~~~~~~~~~~~~

Adds pint the capability to interoperate with `datetime.timedelta`,
`datetime.datetime` and `numpy.timedelta64`.

:copyright: 2022 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from .registry import GenericTimedeltaRegistry, TimedeltaRegistry

__all__ = ["TimedeltaRegistry", "GenericTimedeltaRegistry"]
