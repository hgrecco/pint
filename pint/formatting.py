# -*- coding: utf-8 -*-
"""
    pint.formatter
    ~~~~~~~~~~~~~~

    Format units for pint.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import re

from .babel_names import _babel_units, _babel_lengths
from pint.compat import babel_units, Loc

__JOIN_REG_EXP = re.compile("\{\d*\}")


def _join(fmt, iterable):
    """Join an iterable with the format specified in fmt.

    The format can be specified in two ways:
    - PEP3101 format with two replacement fields (eg. '{0} * {1}')
    - The concatenating string (eg. ' * ')
    """
    if not iterable:
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
    """Format an number into a pretty printed exponent.
    """
    # TODO: Will not work for decimals
    ret = '{0:n}'.format(num).replace('-', '⁻')
    for n in range(10):
        ret = ret.replace(str(n), _PRETTY_EXPONENTS[n])
    return ret


#: _FORMATS maps format specifications to the corresponding argument set to
#: formatter().
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

    'L': {   # Latex format.
        'as_ratio': True,
        'single_denominator': True,
        'product_fmt': r' \cdot ',
        'division_fmt': r'\frac[{0}][{1}]',
        'power_fmt': '{0}^[{1}]',
        'parentheses_fmt': r'\left({0}\right)',
        },

    'H': {   # HTML format.
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
              parentheses_fmt='({0})', exp_call=lambda x: '{0:n}'.format(x),
              locale=None, babel_length='long', babel_plural_form='one'):
    """Format a list of (name, exponent) pairs.

    :param items: a list of (name, exponent) pairs.
    :param as_ratio: True to display as ratio, False as negative powers.
    :param single_denominator: all with terms with negative exponents are
                               collected together.
    :param product_fmt: the format used for multiplication.
    :param division_fmt: the format used for division.
    :param power_fmt: the format used for exponentiation.
    :param parentheses_fmt: the format used for parenthesis.
    :param locale: the locale object as defined in babel.
    :param babel_length: the length of the translated unit, as defined in babel cldr.
    :param babel_plural_form: the plural form, calculated as defined in babel.

    :return: the formula as a string.
    """

    if not items:
        return ''

    if as_ratio:
        fun = lambda x: exp_call(abs(x))
    else:
        fun = exp_call

    pos_terms, neg_terms = [], []

    for key, value in sorted(items):
        if locale and babel_length and babel_plural_form and key in _babel_units:
            _key = _babel_units[key]
            locale = Loc.parse(locale)
            unit_patterns = locale._data['unit_patterns']
            compound_unit_patterns = locale._data["compound_unit_patterns"]
            plural = 'one' if abs(value) <= 0 else babel_plural_form
            if babel_length not in _babel_lengths:
                other_lengths = [
                    _babel_length for _babel_length in reversed(_babel_lengths) \
                    if babel_length != _babel_length
                ]
            else:
                other_lengths = []
            for _babel_length in [babel_length] + other_lengths:
                pat = unit_patterns.get(_key, {}).get(_babel_length, {}).get(plural)
                print(plural, _babel_length, pat)
                if pat is not None:
                    key = pat.replace('{0}', '').strip()
                    break
            division_fmt = compound_unit_patterns.get("per", {}).get(babel_length, division_fmt)
            power_fmt = '{0}{1}'
            exp_call = _pretty_fmt_exponent
        if value == 1:
            pos_terms.append(key)
        elif value > 0:
            pos_terms.append(power_fmt.format(key, fun(value)))
        elif value == -1 and as_ratio:
            neg_terms.append(key)
        else:
            neg_terms.append(power_fmt.format(key, fun(value)))

    if not as_ratio:
        # Show as Product: positive * negative terms ** -1
        return _join(product_fmt, pos_terms + neg_terms)

    # Show as Ratio: positive terms / negative terms
    pos_ret = _join(product_fmt, pos_terms) or '1'

    if not neg_terms:
        return pos_ret

    if single_denominator:
        neg_ret = _join(product_fmt, neg_terms)
        if len(neg_terms) > 1:
            neg_ret = parentheses_fmt.format(neg_ret)
    else:
        neg_ret = _join(division_fmt, neg_terms)

    return _join(division_fmt, [pos_ret, neg_ret])

# Extract just the type from the specification mini-langage: see
# http://docs.python.org/2/library/string.html#format-specification-mini-language
# We also add uS for uncertainties.
_BASIC_TYPES = frozenset('bcdeEfFgGnosxX%uS')
_KNOWN_TYPES = frozenset(list(_FORMATS.keys()) + ['~'])

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


def format_unit(unit, spec, **kwspec):
    if not unit:
        return 'dimensionless'

    spec = _parse_spec(spec)
    fmt = dict(_FORMATS[spec])
    fmt.update(kwspec)

    if spec == 'L':
        rm = [(r'\mathrm{{{0}}}'.format(u), p) for u, p in unit.items()]
        result = formatter(rm, **fmt)
    else:
        result = formatter(unit.items(), **fmt)
    if spec == 'L':
        result = result.replace('[', '{').replace(']', '}')
    return result


def siunitx_format_unit(units):
    '''Returns LaTeX code for the unit that can be put into an siunitx command.'''
    # NOTE: unit registry is required to identify unit prefixes.
    registry = units._REGISTRY

    def _tothe(power):
        if isinstance(power, int) or (isinstance(power, float) and power.is_integer()):
            if power == 1:
                return ''
            elif power == 2:
                return r'\squared'
            elif power == 3:
                return r'\cubed'
            else:
                return r'\tothe{{{:d}}}'.format(int(power))
        else:
            # limit float powers to 3 decimal places
            return r'\tothe{{{:.3f}}}'.format(power).rstrip('0')

    lpos = []
    lneg = []
    # loop through all units in the container
    for unit, power in sorted(units._units.items()):
        # remove unit prefix if it exists
        # siunitx supports \prefix commands

        l = lpos if power >= 0 else lneg
        prefix = None
        for p in registry._prefixes.values():
            p = str(p)
            if len(p) > 0 and unit.find(p) == 0:
                prefix = p
                unit = unit.replace(prefix, '', 1)

        if power < 0:
            l.append(r'\per')
        if prefix is not None:
            l.append(r'\{0}'.format(prefix))
        l.append(r'\{0}'.format(unit))
        l.append(r'{0}'.format(_tothe(abs(power))))

    return ''.join(lpos) + ''.join(lneg)


def remove_custom_flags(spec):
    for flag in _KNOWN_TYPES:
         if flag:
             spec = spec.replace(flag, '')
    return spec


def vector_to_latex(vec, fmtfun=lambda x: format(x, '.2f')):
    return matrix_to_latex([vec], fmtfun)


def matrix_to_latex(matrix, fmtfun=lambda x: format('.2f', x)):
    ret = []

    for row in matrix:
        ret += [' & '.join(fmtfun(f) for f in row)]

    return r'\begin{pmatrix}%s\end{pmatrix}' % '\\\\ \n'.join(ret)


def ndarray_to_latex_parts(ndarr, fmtfun=lambda x: format(x, '.2f'), dim=()):
    if isinstance(fmtfun, str):
        fmt = fmtfun
        fmtfun = lambda x: format(x, fmt)

    if ndarr.ndim == 1:
        return [vector_to_latex(ndarr, fmtfun)]
    if ndarr.ndim == 2:
        return [matrix_to_latex(ndarr, fmtfun)]
    else:
        ret = []
        if ndarr.ndim == 3:
            header = ('arr[%s,' % ','.join('%d' % d for d in dim)) + '%d,:,:]'
            for elno, el in enumerate(ndarr):
                ret += [header % elno + ' = ' + matrix_to_latex(el, fmtfun)]
        else:
            for elno, el in enumerate(ndarr):
                ret += ndarray_to_latex_parts(el, fmtfun, dim + (elno, ))

        return ret


def ndarray_to_latex(ndarr, fmtfun=lambda x: format(x, '.2f'), dim=()):
        return '\n'.join(ndarray_to_latex_parts(ndarr, fmtfun, dim))
