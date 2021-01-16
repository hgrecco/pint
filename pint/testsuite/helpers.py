import doctest
import math
import pickle
import re
import warnings
from distutils.version import LooseVersion
from numbers import Number

import pytest

from pint import Quantity
from pint.compat import ndarray, np

from ..compat import (
    HAS_BABEL,
    HAS_NUMPY,
    HAS_NUMPY_ARRAY_FUNCTION,
    HAS_UNCERTAINTIES,
    NUMPY_VER,
)

_number_re = r"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"
_q_re = re.compile(
    r"<Quantity\("
    + r"\s*"
    + r"(?P<magnitude>%s)" % _number_re
    + r"\s*,\s*"
    + r"'(?P<unit>.*)'"
    + r"\s*"
    + r"\)>"
)

_sq_re = re.compile(
    r"\s*" + r"(?P<magnitude>%s)" % _number_re + r"\s" + r"(?P<unit>.*)"
)

_unit_re = re.compile(r"<Unit\((.*)\)>")


class PintOutputChecker(doctest.OutputChecker):
    def check_output(self, want, got, optionflags):
        check = super().check_output(want, got, optionflags)
        if check:
            return check

        try:
            if eval(want) == eval(got):
                return True
        except Exception:
            pass

        for regex in (_q_re, _sq_re):
            try:
                parsed_got = regex.match(got.replace(r"\\", "")).groupdict()
                parsed_want = regex.match(want.replace(r"\\", "")).groupdict()

                v1 = float(parsed_got["magnitude"])
                v2 = float(parsed_want["magnitude"])

                if abs(v1 - v2) > abs(v1) / 1000:
                    return False

                if parsed_got["unit"] != parsed_want["unit"]:
                    return False

                return True
            except Exception:
                pass

        cnt = 0
        for regex in (_unit_re,):
            try:
                parsed_got, tmp = regex.subn("\1", got)
                cnt += tmp
                parsed_want, temp = regex.subn("\1", want)
                cnt += tmp

                if parsed_got == parsed_want:
                    return True

            except Exception:
                pass

        if cnt:
            # If there was any replacement, we try again the previous methods.
            return self.check_output(parsed_want, parsed_got, optionflags)

        return False


def _get_comparable_magnitudes(first, second, msg):
    if isinstance(first, Quantity) and isinstance(second, Quantity):
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


def assert_quantity_equal(first, second, msg=None):
    if msg is None:
        msg = "Comparing %r and %r. " % (first, second)

    m1, m2 = _get_comparable_magnitudes(first, second, msg)
    msg += " (Converted to %r and %r)" % (m1, m2)

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


def assert_quantity_almost_equal(first, second, rtol=1e-07, atol=0, msg=None):
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


requires_numpy = pytest.mark.skipif(not HAS_NUMPY, reason="Requires NumPy")
requires_not_numpy = pytest.mark.skipif(
    HAS_NUMPY, reason="Requires NumPy not to be installed."
)


def requires_array_function_protocol():
    if not HAS_NUMPY:
        return pytest.mark.skip("Requires NumPy")
    return pytest.mark.skipif(
        not HAS_NUMPY_ARRAY_FUNCTION,
        reason="Requires __array_function__ protocol to be enabled",
    )


def requires_not_array_function_protocol():
    if not HAS_NUMPY:
        return pytest.mark.skip("Requires NumPy")
    return pytest.mark.skipif(
        HAS_NUMPY_ARRAY_FUNCTION,
        reason="Requires __array_function__ protocol to be unavailable or disabled",
    )


def requires_numpy_previous_than(version):
    if not HAS_NUMPY:
        return pytest.mark.skip("Requires NumPy")
    return pytest.mark.skipif(
        not LooseVersion(NUMPY_VER) < LooseVersion(version),
        reason="Requires NumPy < %s" % version,
    )


def requires_numpy_at_least(version):
    if not HAS_NUMPY:
        return pytest.mark.skip("Requires NumPy")
    return pytest.mark.skipif(
        not LooseVersion(NUMPY_VER) >= LooseVersion(version),
        reason="Requires NumPy >= %s" % version,
    )


requires_babel = pytest.mark.skipif(
    not HAS_BABEL, reason="Requires Babel with units support"
)
requires_not_babel = pytest.mark.skipif(
    HAS_BABEL, reason="Requires Babel not to be installed"
)
requires_uncertainties = pytest.mark.skipif(
    not HAS_UNCERTAINTIES, reason="Requires Uncertainties"
)
requires_not_uncertainties = pytest.mark.skipif(
    HAS_UNCERTAINTIES, reason="Requires Uncertainties not to be installed."
)

# Parametrization

allprotos = pytest.mark.parametrize(
    ("protocol",), [(p,) for p in range(pickle.HIGHEST_PROTOCOL + 1)]
)

check_all_bool = pytest.mark.parametrize("check_all", [False, True])
