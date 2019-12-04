# -*- coding: utf-8 -*-
"""
    pint.errors
    ~~~~~~~~~~~

    Functions and classes related to unit definitions and conversions.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


class FilenameMixin:
    def __init__(self, filename=None, lineno=None):
        self.filename = filename
        self.lineno = lineno

    def __str__(self):
        if self.filename and self.lineno is not None:
            return f"While opening {self.filename}, in line {self.lineno}: "
        elif self.filename:
            return f"While opening {self.filename}: "
        elif self.lineno is not None:
            return f"In line {self.lineno}: "
        else:
            return ""


class DefinitionSyntaxError(SyntaxError, FilenameMixin):
    """Raised when a textual definition has a syntax error.
    """

    def __init__(self, msg, *, filename=None, lineno=None):
        SyntaxError.__init__(self, msg)
        FilenameMixin.__init__(self, filename, lineno)

    def __str__(self):
        return f"{FilenameMixin.__str__(self)}{self.args[0]}"


class RedefinitionError(ValueError, FilenameMixin):
    """Raised when a unit or prefix is redefined.
    """

    def __init__(self, name, definition_type, filename=None, lineno=None):
        ValueError().__init__(self)
        FilenameMixin.__init__(self, filename, lineno)
        self.name = name
        self.definition_type = definition_type

    def __str__(self):
        msg = f"Cannot redefine '{self.name}' ({self.definition_type})"
        return FilenameMixin.__str__(self) + msg


class UndefinedUnitError(AttributeError):
    """Raised when the units are not defined in the unit registry.
    """

    def __init__(self, *unit_names):
        if len(unit_names) == 1 and not isinstance(unit_names[0], str):
            unit_names = unit_names[0]
        super().__init__(*unit_names)

    def __str__(self):
        if len(self.args) == 1:
            return f"'{self.args[0]}' is not defined in the unit registry"
        return f"{self.args} are not defined in the unit registry"


class PintTypeError(TypeError):
    pass


class DimensionalityError(PintTypeError):
    """Raised when trying to convert between incompatible units.
    """

    def __init__(self, units1, units2, dim1="", dim2="", *, extra_msg=""):
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

    def __str__(self):
        return (
            "Ambiguous operation with offset unit (%s)."
            % ", ".join(str(u) for u in self.args)
            + " See https://pint.readthedocs.io/en/latest/nonmult.html for guidance."
        )


class UnitStrippedWarning(UserWarning):
    pass
