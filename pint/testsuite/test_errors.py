# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals, print_function, absolute_import


from pint import (
    DefinitionSyntaxError,
    DimensionalityError,
    Quantity,
    OffsetUnitCalculusError,
    RedefinitionError,
    UndefinedUnitError,
)
from pint.testsuite import BaseTestCase


class TestErrors(BaseTestCase):
    def test_definition_syntax_error(self):
        ex = DefinitionSyntaxError("foo")
        self.assertEqual(str(ex), "foo")

        # filename and lineno can be attached after init
        ex.filename = "a.txt"
        ex.lineno = 123
        self.assertEqual(str(ex), "While opening a.txt, in line 123: foo")

        ex = DefinitionSyntaxError("foo", lineno=123)
        self.assertEqual(str(ex), "In line 123: foo")

        ex = DefinitionSyntaxError("foo", filename="a.txt")
        self.assertEqual(str(ex), "While opening a.txt: foo")

        ex = DefinitionSyntaxError("foo", filename="a.txt", lineno=123)
        self.assertEqual(str(ex), "While opening a.txt, in line 123: foo")

    def test_redefinition_error(self):
        ex = RedefinitionError("foo", "bar")
        self.assertEqual(str(ex), "Cannot redefine 'foo' (bar)")

        # filename and lineno can be attached after init
        ex.filename = "a.txt"
        ex.lineno = 123
        self.assertEqual(
            str(ex), "While opening a.txt, in line 123: Cannot redefine 'foo' (bar)"
        )

        ex = RedefinitionError("foo", "bar", lineno=123)
        self.assertEqual(str(ex), "In line 123: Cannot redefine 'foo' (bar)")

        ex = RedefinitionError("foo", "bar", filename="a.txt")
        self.assertEqual(str(ex), "While opening a.txt: Cannot redefine 'foo' (bar)")

        ex = RedefinitionError("foo", "bar", filename="a.txt", lineno=123)
        self.assertEqual(
            str(ex), "While opening a.txt, in line 123: Cannot redefine 'foo' (bar)"
        )

    def test_undefined_unit_error(self):
        x = ("meter",)
        msg = "'meter' is not defined in the unit registry"
        self.assertEqual(str(UndefinedUnitError(x)), msg)
        self.assertEqual(str(UndefinedUnitError(list(x))), msg)
        self.assertEqual(str(UndefinedUnitError(set(x))), msg)

    def test_undefined_unit_error_multi(self):
        x = ("meter", "kg")
        msg = "('meter', 'kg') are not defined in the unit registry"
        self.assertEqual(str(UndefinedUnitError(x)), msg)
        self.assertEqual(str(UndefinedUnitError(list(x))), msg)

    def test_dimensionality_error(self):
        ex = DimensionalityError("a", "b")
        self.assertEqual(str(ex), "Cannot convert from 'a' to 'b'")
        ex = DimensionalityError("a", "b", "c")
        self.assertEqual(str(ex), "Cannot convert from 'a' (c) to 'b' ()")
        ex = DimensionalityError("a", "b", "c", "d", extra_msg=": msg")
        self.assertEqual(str(ex), "Cannot convert from 'a' (c) to 'b' (d): msg")

    def test_offset_unit_calculus_error(self):
        ex = OffsetUnitCalculusError(Quantity("1 kg")._units)
        self.assertEqual(
            str(ex),
            "Ambiguous operation with offset unit (kilogram). See "
            "https://pint.readthedocs.io/en/latest/nonmult.html for guidance.",
        )
        ex = OffsetUnitCalculusError(Quantity("1 kg")._units, Quantity("1 s")._units)
        self.assertEqual(
            str(ex),
            "Ambiguous operation with offset unit (kilogram, second). See "
            "https://pint.readthedocs.io/en/latest/nonmult.html for guidance.",
        )
