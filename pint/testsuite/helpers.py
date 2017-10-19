# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import


import doctest
from distutils.version import StrictVersion
import re
import unittest

from pint.compat import HAS_NUMPY, HAS_PROPER_BABEL, HAS_UNCERTAINTIES, NUMPY_VER, PYTHON3


def requires_numpy18():
    if not HAS_NUMPY:
        return unittest.skip('Requires NumPy')
    return unittest.skipUnless(StrictVersion(NUMPY_VER) >= StrictVersion('1.8'), 'Requires NumPy >= 1.8')


def requires_numpy_previous_than(version):
    if not HAS_NUMPY:
        return unittest.skip('Requires NumPy')
    return unittest.skipUnless(StrictVersion(NUMPY_VER) < StrictVersion(version), 'Requires NumPy < %s' % version)


def requires_numpy():
    return unittest.skipUnless(HAS_NUMPY, 'Requires NumPy')


def requires_not_numpy():
    return unittest.skipIf(HAS_NUMPY, 'Requires NumPy is not installed.')


def requires_proper_babel():
    return unittest.skipUnless(HAS_PROPER_BABEL, 'Requires Babel with units support')


def requires_uncertainties():
    return unittest.skipUnless(HAS_UNCERTAINTIES, 'Requires Uncertainties')


def requires_not_uncertainties():
    return unittest.skipIf(HAS_UNCERTAINTIES, 'Requires Uncertainties is not installed.')


def requires_python2():
    return unittest.skipIf(PYTHON3, 'Requires Python 2.X.')


def requires_python3():
    return unittest.skipUnless(PYTHON3, 'Requires Python 3.X.')


_number_re = '([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
_q_re = re.compile('<Quantity\(' + '\s*' + '(?P<magnitude>%s)' % _number_re +
                   '\s*,\s*' + "'(?P<unit>.*)'" + '\s*' + '\)>')

_sq_re = re.compile('\s*' + '(?P<magnitude>%s)' % _number_re +
                    '\s' + "(?P<unit>.*)")

_unit_re = re.compile('<Unit\((.*)\)>')


class PintOutputChecker(doctest.OutputChecker):

    def check_output(self, want, got, optionflags):
        check = super(PintOutputChecker, self).check_output(want, got, optionflags)
        if check:
            return check

        try:
            if eval(want) == eval(got):
                return True
        except:
            pass

        for regex in (_q_re, _sq_re):
            try:
                parsed_got = regex.match(got.replace(r'\\', '')).groupdict()
                parsed_want = regex.match(want.replace(r'\\', '')).groupdict()

                v1 = float(parsed_got['magnitude'])
                v2 = float(parsed_want['magnitude'])

                if abs(v1 - v2) > abs(v1) / 1000:
                    return False

                if parsed_got['unit'] != parsed_want['unit']:
                    return False

                return True
            except:
                pass

        cnt = 0
        for regex in (_unit_re, ):
            try:
                parsed_got, tmp = regex.subn('\1', got)
                cnt += tmp
                parsed_want, temp = regex.subn('\1', want)
                cnt += tmp

                if parsed_got == parsed_want:
                    return True

            except:
                pass

        if cnt:
            # If there was any replacement, we try again the previous methods.
            return self.check_output(parsed_want, parsed_got, optionflags)

        return False

