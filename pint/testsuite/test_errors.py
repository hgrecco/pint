# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals, print_function, absolute_import
import pickle


from pint.errors import (
    DimensionalityError,
    UndefinedUnitError,
    OffsetUnitCalculusError,
    DefinitionSyntaxError,
    RedefinitionError
)
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

    def test_errors_can_be_pickled(self):

        error_list = [
            DimensionalityError('a', 'b', 'c', 'd', 'msg'),
            UndefinedUnitError(('a', )),
            OffsetUnitCalculusError('a', 'b', 'msg'),
            DefinitionSyntaxError('msg', 'filename', 'lineno'),
            RedefinitionError('name', 'definition_type')
        ]

        for error in error_list:
            try:
                pickle.loads(pickle.dumps(error))
            except TypeError:
                raise TypeError(f'{type(error)} cannot be pickled')
