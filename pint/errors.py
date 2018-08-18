# -*- coding: utf-8 -*-
"""
    pint.errors
    ~~~~~~~~~

    Functions and classes related to unit definitions and conversions.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)

from .compat import string_types


class DefinitionSyntaxError(ValueError):
    """Raised when a textual definition has a syntax error.
    """

    def __init__(self, msg, filename=None, lineno=None):
        super(DefinitionSyntaxError, self).__init__()
        self.msg = msg
        self.filename = None
        self.lineno = None

    def __str__(self):
        mess = "While opening {}, in line {}: "
        return mess.format(self.filename, self.lineno) + self.msg


class RedefinitionError(ValueError):
    """Raised when a unit or prefix is redefined.
    """

    def __init__(self, name, definition_type):
        super(RedefinitionError, self).__init__()
        self.name = name
        self.definition_type = definition_type
        self.filename = None
        self.lineno = None

    def __str__(self):
        msg = "cannot redefine '{}' ({})".format(self.name,
                                                   self.definition_type)
        if self.filename:
            mess = "While opening {}, in line {}: "
            return mess.format(self.filename, self.lineno) + msg
        return msg


class UndefinedUnitError(AttributeError):
    """Raised when the units are not defined in the unit registry.
    """

    def __init__(self, unit_names):
        super(UndefinedUnitError, self).__init__()
        self.unit_names = unit_names

    def __str__(self):
        mess = "'{}' is not defined in the unit registry"
        mess_plural = "'{}' are not defined in the unit registry"
        if isinstance(self.unit_names, string_types):
            return mess.format(self.unit_names)
        elif isinstance(self.unit_names, (list, tuple))\
                and len(self.unit_names) == 1:
            return mess.format(self.unit_names[0])
        elif isinstance(self.unit_names, set) and len(self.unit_names) == 1:
            uname = list(self.unit_names)[0]
            return mess.format(uname)
        else:
            return mess_plural.format(self.unit_names)


class DimensionalityError(ValueError):
    """Raised when trying to convert between incompatible units.
    """

    def __init__(self, units1, units2, dim1=None, dim2=None, extra_msg=''):
        super(DimensionalityError, self).__init__()
        self.units1 = units1
        self.units2 = units2
        self.dim1 = dim1
        self.dim2 = dim2
        self.extra_msg = extra_msg

    def __str__(self):
        if self.dim1 or self.dim2:
            dim1 = ' ({})'.format(self.dim1)
            dim2 = ' ({})'.format(self.dim2)
        else:
            dim1 = ''
            dim2 = ''

        msg = "Cannot convert from '{}'{} to '{}'{}" + self.extra_msg

        return msg.format(self.units1, dim1, self.units2, dim2)


class OffsetUnitCalculusError(ValueError):
    """Raised on ambiguous operations with offset units.
    """
    def __init__(self, units1, units2='', extra_msg=''):
        super(ValueError, self).__init__()
        self.units1 = units1
        self.units2 = units2
        self.extra_msg = extra_msg

    def __str__(self):
        msg = ("Ambiguous operation with offset unit (%s)." %
               ', '.join(['%s' % u for u in [self.units1, self.units2] if u])
               + self.extra_msg)
        return msg.format(self.units1, self.units2)


class UnitStrippedWarning(UserWarning):
    pass
