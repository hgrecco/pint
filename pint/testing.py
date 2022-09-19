from __future__ import annotations

import math
import warnings
from numbers import Number

from . import Quantity
from .compat import ndarray

try:
    import numpy as np
except ImportError:
    np = None


def _get_comparable_magnitudes(first, second, msg):
    if isinstance(first, Quantity) and isinstance(second, Quantity):
        ctx = first._REGISTRY._active_ctx.contexts
        if first.is_compatible_with(second, *ctx):
            second = second.to(first)
        assert first.units == second.units, msg + " Units are not equal."
        m1, m2 = first.magnitude, second.magnitude
    elif isinstance(first, Quantity):
        assert first.dimensionless, msg + " The first is not dimensionless."
        first = first.to("")
        m1, m2 = first.magnitude, second
    elif isinstance(second, Quantity):
        assert second.dimensionless, msg + " The second is not dimensionless."
        second = second.to("")
        m1, m2 = first, second.magnitude
    else:
        m1, m2 = first, second

    return m1, m2


def assert_equal(first, second, msg=None):
    if msg is None:
        msg = "Comparing %r and %r. " % (first, second)

    m1, m2 = _get_comparable_magnitudes(first, second, msg)
    msg += " (Converted to %r and %r): Magnitudes are not equal" % (m1, m2)

    if isinstance(m1, ndarray) or isinstance(m2, ndarray):
        np.testing.assert_array_equal(m1, m2, err_msg=msg)
    elif not isinstance(m1, Number):
        warnings.warn(RuntimeWarning)
        return
    elif not isinstance(m2, Number):
        warnings.warn(RuntimeWarning)
        return
    elif math.isnan(m1):
        assert math.isnan(m2), msg
    elif math.isnan(m2):
        assert math.isnan(m1), msg
    else:
        assert m1 == m2, msg


def assert_allclose(first, second, rtol=1e-07, atol=0, msg=None):
    if msg is None:
        try:
            msg = "Comparing %r and %r. " % (first, second)
        except TypeError:
            try:
                msg = "Comparing %s and %s. " % (first, second)
            except Exception:
                msg = "Comparing"

    m1, m2 = _get_comparable_magnitudes(first, second, msg)
    msg += " (Converted to %r and %r)" % (m1, m2)

    if isinstance(m1, ndarray) or isinstance(m2, ndarray):
        np.testing.assert_allclose(m1, m2, rtol=rtol, atol=atol, err_msg=msg)
    elif not isinstance(m1, Number):
        warnings.warn(RuntimeWarning)
        return
    elif not isinstance(m2, Number):
        warnings.warn(RuntimeWarning)
        return
    elif math.isnan(m1):
        assert math.isnan(m2), msg
    elif math.isnan(m2):
        assert math.isnan(m1), msg
    elif math.isinf(m1):
        assert math.isinf(m2), msg
    elif math.isinf(m2):
        assert math.isinf(m1), msg
    else:
        # Numpy version (don't like because is not symmetric)
        # assert abs(m1 - m2) <= atol + rtol * abs(m2), msg
        assert abs(m1 - m2) <= max(rtol * max(abs(m1), abs(m2)), atol), msg
