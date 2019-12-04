# -*- coding: utf-8 -*-
"""
    pint.errors
    ~~~~~~~~~~~

    Functions and classes related to unit definitions and conversions.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


class DefinitionSyntaxError(SyntaxError):
    """Raised when a textual definition has a syntax error.
    """

    def __init__(self, msg, filename=None, lineno=None):
        super().__init__(msg)
        self.filename = None
        self.lineno = None

    def __str__(self):
        return f"While opening {self.filename}, in line {self.lineno}: {self.args[0]}"


class RedefinitionError(ValueError):
    """Raised when a unit or prefix is redefined.
    """

    def __init__(self, name, definition_type):
        super().__init__()
        self.name = name
        self.definition_type = definition_type
        self.filename = None
        self.lineno = None

    def __str__(self):
        msg = f"Cannot redefine '{self.name}' ({self.definition_type})"
        if self.filename:
            return f"While opening {self.filename}, in line {self.lineno}: {msg}"
        return msg


class UndefinedUnitError(AttributeError):
    """Raised when the units are not defined in the unit registry.
    """

    def __init__(self, unit_names):
        super().__init__()
        self.unit_names = unit_names

    def __str__(self):
        mess = "'{}' is not defined in the unit registry"
        mess_plural = "'{}' are not defined in the unit registry"
        if isinstance(self.unit_names, str):
            return mess.format(self.unit_names)
        elif isinstance(self.unit_names, (list, tuple)) and len(self.unit_names) == 1:
            return mess.format(self.unit_names[0])
        elif isinstance(self.unit_names, set) and len(self.unit_names) == 1:
            uname = list(self.unit_names)[0]
            return mess.format(uname)
        else:
            return mess_plural.format(self.unit_names)


class PintTypeError(TypeError):
    pass


class DimensionalityError(PintTypeError):
    """Raised when trying to convert between incompatible units.
    """

    def __init__(self, units1, units2, dim1=None, dim2=None, *, extra_msg=""):
        super().__init__()
        self.units1 = units1
        self.units2 = units2
        self.dim1 = dim1
        self.dim2 = dim2
        self.extra_msg = extra_msg

    def __str__(self):
        if self.dim1 or self.dim2:
            dim1 = f" ({self.dim1})"
            dim2 = f" ({self.dim2})"
        else:
            dim1 = ""
            dim2 = ""

        return (
            f"Cannot convert from '{self.units1}'{dim1} to "
            f"'{self.units2}'{dim2}{self.extra_msg}"
        )


class OffsetUnitCalculusError(PintTypeError):
    """Raised on ambiguous operations with offset units.
    """

    def __init__(self, units1, units2="", *, extra_msg=""):
        super().__init__()
        self.units1 = units1
        self.units2 = units2
        self.extra_msg = extra_msg

    def __str__(self):
        msg = (
            "Ambiguous operation with offset unit (%s)."
            % ", ".join(["%s" % u for u in [self.units1, self.units2] if u])
            + " See https://pint.readthedocs.io/en/latest/nonmult.html for guidance."
            + self.extra_msg
        )
        return msg.format(self.units1, self.units2)


class UnitStrippedWarning(UserWarning):
    pass
