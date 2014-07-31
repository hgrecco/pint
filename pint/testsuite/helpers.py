# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

from pint.compat import unittest, HAS_NUMPY, HAS_UNCERTAINTIES, NUMPY_VER, PYTHON3


def requires_numpy18():
    return unittest.skipUnless(NUMPY_VER >= '1.8', 'Requires NumPy >= 1.8')


def requires_numpy():
    return unittest.skipUnless(HAS_NUMPY, 'Requires NumPy')


def requires_not_numpy():
    return unittest.skipIf(HAS_NUMPY, 'Requires NumPy is not installed.')


def requires_uncertainties():
    return unittest.skipUnless(HAS_UNCERTAINTIES, 'Requires Uncertainties')


def requires_not_uncertainties():
    return unittest.skipIf(HAS_UNCERTAINTIES, 'Requires Uncertainties is not installed.')


def requires_python2():
    return unittest.skipIf(PYTHON3, 'Requires Python 2.X.')


def requires_python3():
    return unittest.skipUnless(PYTHON3, 'Requires Python 3.X.')
