"""
    pint.formatter
    ~~~~~~~~~~~~~~

    Format units for pint.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations


# Backwards compatiblity stuff
from .delegates.formatter.latex import (
    vector_to_latex,  # noqa
    matrix_to_latex,  # noqa
    ndarray_to_latex_parts,  # noqa
    ndarray_to_latex,  # noqa
    latex_escape,  # noqa
    siunitx_format_unit,  # noqa
    _EXP_PATTERN,  # noqa
)  # noqa
from .delegates.formatter._spec_helpers import (
    FORMATTER,  # noqa
    _BASIC_TYPES,  # noqa
    parse_spec as _parse_spec,  # noqa
    _JOIN_REG_EXP as __JOIN_REG_EXP,  # noqa,
    _join,  # noqa
    _PRETTY_EXPONENTS,  # noqa
    pretty_fmt_exponent as _pretty_fmt_exponent,  # noqa
    extract_custom_flags,  # noqa
    remove_custom_flags,  # noqa
    split_format,  # noqa
    REGISTERED_FORMATTERS,
)  # noqa
from .delegates.formatter._to_register import register_unit_format  # noqa


def format_unit(unit, spec: str, registry=None, **options):
    # registry may be None to allow formatting `UnitsContainer` objects
    # in that case, the spec may not be "Lx"

    if not unit:
        if spec.endswith("%"):
            return ""
        else:
            return "dimensionless"

    if not spec:
        spec = "D"

    if registry is None:
        _formatter = REGISTERED_FORMATTERS.get(spec, None)
    else:
        try:
            _formatter = registry._formatters[spec]
        except Exception:
            _formatter = registry._formatters.get(spec, None)

    if _formatter is None:
        raise ValueError(f"Unknown conversion specified: {spec}")

    return _formatter.format_unit(unit)
