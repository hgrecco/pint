# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals, print_function, absolute_import


from pint.errors import DimensionalityError, UndefinedUnitError
from pint.testsuite import BaseTestCase

class TestErrors(BaseTestCase):

    def test_errors(self):
        x = ('meter', )
        msg = "'meter' is not defined in the unit registry"
        self.assertEqual(str(UndefinedUnitError(x)), msg)
        self.assertEqual(str(UndefinedUnitError(list(x))), msg)
        self.assertEqual(str(UndefinedUnitError(set(x))), msg)

        msg = "Cannot convert from 'a' (c) to 'b' (d)msg"
        ex = DimensionalityError('a', 'b', 'c', 'd', 'msg')
        self.assertEqual(str(ex), msg)
