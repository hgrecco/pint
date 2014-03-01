# -*- coding: utf-8 -*-
"""
    pint.formatter
    ~~~~~~~~~

    Format units for pint.

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import re

__JOIN_REG_EXP = re.compile("\{\d*\}")

def _join(fmt, iterable):
    if not iter:
        return ''
    if not __JOIN_REG_EXP.search(fmt):
        return fmt.join(iterable)
    miter = iter(iterable)
    first = next(miter)
    for val in miter:
        ret = fmt.format(first, val)
        first = ret
    return first

_PRETTY_EXPONENTS = '⁰¹²³⁴⁵⁶⁷⁸⁹'

def _pretty_fmt_exponent(num):
    ret = '{0:n}'.format(num).replace('-', '⁻')
    for n in range(10):
        ret = ret.replace(str(n), _PRETTY_EXPONENTS[n])
    return ret

# _FORMATS maps format specifications to the corresponding argument set to
# formatter().
_FORMATS = {
    'P': {   # Pretty format.
        'as_ratio': True,
        'single_denominator': False,
        'product_fmt': '·',
        'division_fmt': '/',
        'power_fmt': '{0}{1}',
        'parentheses_fmt': '({0})',
        'exp_call': _pretty_fmt_exponent,
        },

    'L': {   # Latex print format.
        'as_ratio': True,
        'single_denominator': True,
        'product_fmt': r' \cdot ',
        'division_fmt': r'\frac[{0}][{1}]',
        'power_fmt': '{0}^[{1}]',
        'parentheses_fmt': '{0}^[{1}]',
        },

    'H': {   # Latex format.
        'as_ratio': True,
        'single_denominator': True,
        'product_fmt': r' ',
        'division_fmt': r'{0}/{1}',
        'power_fmt': '{0}<sup>{1}</sup>',
        'parentheses_fmt': r'({0})',
        },

    '': {   # Default format.
        'as_ratio': True,
        'single_denominator': False,
        'product_fmt': ' * ',
        'division_fmt': ' / ',
        'power_fmt': '{0} ** {1}',
        'parentheses_fmt': r'({0})',
        },

     'C': {  # Compact format.
        'as_ratio': True,
        'single_denominator': False,
        'product_fmt': '*',  # TODO: Should this just be ''?
        'division_fmt': '/',
        'power_fmt': '{0}**{1}',
        'parentheses_fmt': r'({0})',
        },
    }

def formatter(items, as_ratio=True, single_denominator=False,
              product_fmt=' * ', division_fmt=' / ', power_fmt='{0} ** {1}',
              parentheses_fmt='({0})', exp_call=lambda x: '{0:n}'.format(x)):
    """Format a list of (name, exponent) pairs.

    :param items: a list of (name, exponent) pairs.
    :param as_ratio: True to display as ratio, False as negative powers.
    :param single_denominator: all with terms with negative exponents are
                               collected together.
    :param product_fmt: the format used for multiplication.
    :param division_fmt: the format used for division.
    :param power_fmt: the format used for exponentiation.
    :param parentheses_fmt: the format used for parenthesis.

    :return: the formula as a string.
    """
    if as_ratio:
        fun = lambda x: exp_call(abs(x))
    else:
        fun = exp_call

    pos_terms, neg_terms = [], []

    for key, value in sorted(items):
        if value == 1:
            pos_terms.append(key)
        elif value > 0:
            pos_terms.append(power_fmt.format(key, fun(value)))
        elif value == -1:
            neg_terms.append(key)
        else:
            neg_terms.append(power_fmt.format(key, fun(value)))

    if pos_terms:
        pos_ret = _join(product_fmt, pos_terms)
    elif as_ratio and neg_terms:
        pos_ret = '1'
    else:
        pos_ret = ''

    if not neg_terms:
        return pos_ret

    if as_ratio:
        if single_denominator:
            neg_ret = _join(product_fmt, neg_terms)
            if len(neg_terms) > 1:
                neg_ret = parentheses_fmt.format(neg_ret)
        else:
            neg_ret = _join(division_fmt, neg_terms)
    else:
        neg_ret = product_fmt.join(neg_terms)

    return _join(division_fmt, [pos_ret, neg_ret])

# Extract just the type from the specification mini-langage: see
# http://docs.python.org/2/library/string.html#format-specification-mini-language

_BASIC_TYPES = frozenset('bcdeEfFgGnosxX%')
_KNOWN_TYPES = frozenset(_FORMATS.keys())

def _parse_spec(spec):
    result = ''
    for ch in reversed(spec):
        if ch == '~' or ch in _BASIC_TYPES:
            continue
        elif ch in _KNOWN_TYPES:
            if result:
                raise ValueError("expected ':' after format specifier")
            else:
                result = ch
        elif ch.isalpha():
            raise ValueError("Unknown conversion specified " + ch)
        else:
             break
    return result

def format_unit(unit, spec):
    if not unit:
        return 'dimensionless'
    spec = _parse_spec(spec)
    fmt = _FORMATS.get(spec)
    if not fmt:
        raise ValueError('Unknown conversion specifier ' + spec)

    result = formatter(unit.items(), **fmt)
    if spec == 'L':
        result = result.replace('[', '{').replace(']', '}')
    return result

def remove_custom_flags(spec):
    for flag in _FORMATS.keys():
         if flag:
             spec = spec.replace(flag, '')
    return spec
