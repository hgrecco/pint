# -*- coding: utf-8 -*-
"""
    pint.registry_helpers
    ~~~~~~~~~~~~~~~~~~~~~

    Miscellaneous methods of the registry writen as separate functions.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details..
    :license: BSD, see LICENSE for more details.
"""

import functools
from itertools import izip_longest

from .compat import string_types, zip_longest
from .errors import DimensionalityError
from .util import to_units_container


def _replace_units(original_units, values_by_name):
    """Convert a unit compatible type to a UnitsContainer.

    :param original_units: a UnitsContainer instance.
    :param values_by_name: a map between original names and the new values.
    """
    q = 1
    for arg_name, exponent in original_units.items():
        q = q * values_by_name[arg_name] ** exponent

    return to_units_container(q)


def _to_units_container(a):
    """Convert a unit compatible type to a UnitsContainer,
    checking if it is string field prefixed with an equal
    (which is considered a reference)

    Return a tuple with the unit container and a boolean indicating if it was a reference.
    """
    if isinstance(a, string_types) and '=' in a:
        return to_units_container(a.split('=', 1)[1]), True
    return to_units_container(a), False


def _parse_wrap_args(args):

    # Arguments which contain definitions
    # (i.e. names that appear alone and for the first time)
    defs_args = set()
    defs_args_ndx = set()

    # Arguments which depend on others
    dependent_args_ndx = set()

    # Arguments which have units.
    unit_args_ndx = set()

    # _to_units_container
    args_as_uc = [_to_units_container(arg) for arg in args]

    # Check for references in args, remove None values
    for ndx, (arg, is_ref) in enumerate(args_as_uc):
        if arg is None:
            continue
        elif is_ref:
            if len(arg) == 1:
                [(key, value)] = arg.items()
                if value == 1 and key not in defs_args:
                    # This is the first time that
                    # a variable is used => it is a definition.
                    defs_args.add(key)
                    defs_args_ndx.add(ndx)
                    args_as_uc[ndx] = (key, True)
                else:
                    # The variable was already found elsewhere,
                    # we consider it a dependent variable.
                    dependent_args_ndx.add(ndx)
            else:
                dependent_args_ndx.add(ndx)
        else:
            unit_args_ndx.add(ndx)

    # Check that all valid dependent variables
    for ndx in dependent_args_ndx:
        arg, is_ref = args_as_uc[ndx]
        if not isinstance(arg, dict):
            continue
        if not set(arg.keys()) <= defs_args:
            raise ValueError('Found a missing token while wrapping a function: '
                             'Not all variable referenced in %s are defined using !' % args[ndx])

    def _converter(ureg, values, strict):
        new_values = list(value for value in values)

        values_by_name = {}

        # first pass: Grab named values
        for ndx in defs_args_ndx:
            values_by_name[args_as_uc[ndx][0]] = values[ndx]
            new_values[ndx] = values[ndx]._magnitude

        # second pass: calculate derived values based on named values
        for ndx in dependent_args_ndx:
            new_values[ndx] = ureg._convert(values[ndx]._magnitude,
                                            values[ndx]._units,
                                            _replace_units(args_as_uc[ndx][0], values_by_name))

        # third pass: convert other arguments
        for ndx in unit_args_ndx:

            if isinstance(values[ndx], ureg.Quantity):
                new_values[ndx] = ureg._convert(values[ndx]._magnitude,
                                                values[ndx]._units,
                                                args_as_uc[ndx][0])
            else:
                if strict:
                    raise ValueError('A wrapped function using strict=True requires '
                                     'quantity for all arguments with not None units. '
                                     '(error found for {0}, {1})'.format(args_as_uc[ndx][0], new_values[ndx]))

        return new_values, values_by_name

    return _converter


def wraps(ureg, ret, args, strict=True):
    """Wraps a function to become pint-aware.

    Use it when a function requires a numerical value but in some specific
    units. The wrapper function will take a pint quantity, convert to the units
    specified in `args` and then call the wrapped function with the resulting
    magnitude.

    The value returned by the wrapped function will be converted to the units
    specified in `ret`.

    Use None to skip argument conversion.
    Set strict to False, to accept also numerical values.

    :param ureg: a UnitRegistry instance.
    :param ret: output units.
    :param args: iterable of input units.
    :param strict: boolean to indicate that only quantities are accepted.
    :return: the wrapped function.
    :raises:
        :class:`ValueError` if strict and one of the arguments is not a Quantity.
    """

    if not isinstance(args, (list, tuple)):
        args = (args, )

    converter = _parse_wrap_args(args)

    if isinstance(ret, (list, tuple)):
        container, ret = True, ret.__class__([_to_units_container(arg) for arg in ret])
    else:
        container, ret = False, _to_units_container(ret)

    def decorator(func):
        assigned = tuple(attr for attr in functools.WRAPPER_ASSIGNMENTS if hasattr(func, attr))
        updated = tuple(attr for attr in functools.WRAPPER_UPDATES if hasattr(func, attr))

        @functools.wraps(func, assigned=assigned, updated=updated)
        def wrapper(*values, **kw):

            # In principle, the values are used as is
            # When then extract the magnitudes when needed.
            new_values, values_by_name = converter(ureg, values, strict)

            result = func(*new_values, **kw)

            if container:
                out_units = (_replace_units(r, values_by_name) if is_ref else r
                             for (r, is_ref) in ret)
                return ret.__class__(res if unit is None else ureg.Quantity(res, unit)
                                     for unit, res in izip_longest(out_units, result))

            if ret[0] is None:
                return result

            return ureg.Quantity(result,
                                 _replace_units(ret[0], values_by_name) if ret[1] else ret[0])

        return wrapper
    return decorator


def check(ureg, *args):
    """Decorator to for quantity type checking for function inputs.

    Use it to ensure that the decorated function input parameters match
    the expected type of pint quantity.

    Use None to skip argument checking.

    :param ureg: a UnitRegistry instance.
    :param args: iterable of input units.
    :return: the wrapped function.
    :raises:
        :class:`DimensionalityError` if the parameters don't match dimensions
    """
    dimensions = [ureg.get_dimensionality(dim) for dim in args]

    def decorator(func):
        assigned = tuple(attr for attr in functools.WRAPPER_ASSIGNMENTS if hasattr(func, attr))
        updated = tuple(attr for attr in functools.WRAPPER_UPDATES if hasattr(func, attr))

        @functools.wraps(func, assigned=assigned, updated=updated)
        def wrapper(*values, **kwargs):
            for dim, value in zip_longest(dimensions, values):
                if dim and value.dimensionality != dim:
                    raise DimensionalityError(value, 'a quantity of',
                                              value.dimensionality, dim)
            return func(*values, **kwargs)
        return wrapper
    return decorator
