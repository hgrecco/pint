import pickle

import pytest

from pint import (
    DefinitionSyntaxError,
    DimensionalityError,
    LogarithmicUnitCalculusError,
    OffsetUnitCalculusError,
    Quantity,
    RedefinitionError,
    UndefinedUnitError,
    UnitRegistry,
)
from pint.errors import LOG_ERROR_DOCS_HTML, OFFSET_ERROR_DOCS_HTML


class TestErrors:
    def test_definition_syntax_error(self):
        ex = DefinitionSyntaxError("foo")
        assert str(ex) == "foo"

        # filename and lineno can be attached after init
        ex.filename = "a.txt"
        ex.lineno = 123
        assert str(ex) == "While opening a.txt, in line 123: foo"

        ex = DefinitionSyntaxError("foo", lineno=123)
        assert str(ex) == "In line 123: foo"

        ex = DefinitionSyntaxError("foo", filename="a.txt")
        assert str(ex) == "While opening a.txt: foo"

        ex = DefinitionSyntaxError("foo", filename="a.txt", lineno=123)
        assert str(ex) == "While opening a.txt, in line 123: foo"

    def test_redefinition_error(self):
        ex = RedefinitionError("foo", "bar")
        assert str(ex) == "Cannot redefine 'foo' (bar)"

        # filename and lineno can be attached after init
        ex.filename = "a.txt"
        ex.lineno = 123
        assert (
            str(ex) == "While opening a.txt, in line 123: Cannot redefine 'foo' (bar)"
        )

        ex = RedefinitionError("foo", "bar", lineno=123)
        assert str(ex) == "In line 123: Cannot redefine 'foo' (bar)"

        ex = RedefinitionError("foo", "bar", filename="a.txt")
        assert str(ex) == "While opening a.txt: Cannot redefine 'foo' (bar)"

        ex = RedefinitionError("foo", "bar", filename="a.txt", lineno=123)
        assert (
            str(ex) == "While opening a.txt, in line 123: Cannot redefine 'foo' (bar)"
        )

    def test_undefined_unit_error(self):
        x = ("meter",)
        msg = "'meter' is not defined in the unit registry"
        assert str(UndefinedUnitError(x)) == msg
        assert str(UndefinedUnitError(list(x))) == msg
        assert str(UndefinedUnitError(set(x))) == msg

    def test_undefined_unit_error_multi(self):
        x = ("meter", "kg")
        msg = "('meter', 'kg') are not defined in the unit registry"
        assert str(UndefinedUnitError(x)) == msg
        assert str(UndefinedUnitError(list(x))) == msg

    def test_dimensionality_error(self):
        ex = DimensionalityError("a", "b")
        assert str(ex) == "Cannot convert from 'a' to 'b'"
        ex = DimensionalityError("a", "b", "c")
        assert str(ex) == "Cannot convert from 'a' (c) to 'b' ()"
        ex = DimensionalityError("a", "b", "c", "d", extra_msg=": msg")
        assert str(ex) == "Cannot convert from 'a' (c) to 'b' (d): msg"

    def test_offset_unit_calculus_error(self):
        ex = OffsetUnitCalculusError(Quantity("1 kg")._units)
        assert (
            str(ex)
            == "Ambiguous operation with offset unit (kilogram). See "
            + OFFSET_ERROR_DOCS_HTML
            + " for guidance."
        )
        ex = OffsetUnitCalculusError(Quantity("1 kg")._units, Quantity("1 s")._units)
        assert (
            str(ex)
            == "Ambiguous operation with offset unit (kilogram, second). See "
            + OFFSET_ERROR_DOCS_HTML
            + " for guidance."
        )

    def test_logarithmic_unit_calculus_error(self):
        Quantity = UnitRegistry(autoconvert_offset_to_baseunit=True).Quantity
        ex = LogarithmicUnitCalculusError(Quantity("1 dB")._units)
        assert (
            str(ex)
            == "Ambiguous operation with logarithmic unit (decibel). See "
            + LOG_ERROR_DOCS_HTML
            + " for guidance."
        )
        ex = LogarithmicUnitCalculusError(
            Quantity("1 dB")._units, Quantity("1 octave")._units
        )
        assert (
            str(ex)
            == "Ambiguous operation with logarithmic unit (decibel, octave). See "
            + LOG_ERROR_DOCS_HTML
            + " for guidance."
        )

    def test_pickle_definition_syntax_error(self, subtests):
        # OffsetUnitCalculusError raised from a custom ureg must be pickleable even if
        # the ureg is not registered as the application ureg
        ureg = UnitRegistry(filename=None)
        ureg.define("foo = [bar]")
        ureg.define("bar = 2 foo")
        q1 = ureg.Quantity("1 foo")
        q2 = ureg.Quantity("1 bar")

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            for ex in [
                DefinitionSyntaxError("foo", filename="a.txt", lineno=123),
                RedefinitionError("foo", "bar"),
                UndefinedUnitError("meter"),
                DimensionalityError("a", "b", "c", "d", extra_msg=": msg"),
                OffsetUnitCalculusError(
                    Quantity("1 kg")._units, Quantity("1 s")._units
                ),
                OffsetUnitCalculusError(q1._units, q2._units),
            ]:
                with subtests.test(protocol=protocol, etype=type(ex)):
                    pik = pickle.dumps(ureg.Quantity("1 foo"), protocol)
                    with pytest.raises(UndefinedUnitError):
                        pickle.loads(pik)

                    # assert False, ex.__reduce__()
                    ex2 = pickle.loads(pickle.dumps(ex, protocol))
                    assert type(ex) is type(ex2)
                    assert ex.args == ex2.args
                    assert ex.__dict__ == ex2.__dict__
                    assert str(ex) == str(ex2)
